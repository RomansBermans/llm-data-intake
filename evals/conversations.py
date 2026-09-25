"""Paid live checks of extraction, verification, and complete intake flows."""

import argparse
import json
import time
from copy import deepcopy
from datetime import date
from pathlib import Path
from uuid import uuid4

from src.extractor import create_model_clients
from src.screening import JsonPersistence, ScreeningController, State, load_role
from evals.recording import CachedExtractor, RecordingVerifier


ANSWERS = {
    "role_confirmation": "Yes, I applied for the Software Engineer role.",
    "experience": "I have 5 total years of professional experience and use Python.",
    "availability": "I can start on 2026-10-01.",
    "availability_confirmation": "Yes, that date is correct.",
    "work_authorization": "Yes, I am authorized to work in Spain.",
    "salary": "My annual salary expectation is 70000 EUR.",
    "review_confirmation": "Yes, the entire summary is correct.",
}
EARLY = ("Yes, I applied for the Software Engineer role. I have 5 total years of professional "
         "experience and use Python. I can start on 2026-10-01. I am authorized to work in Spain. "
         "My annual salary expectation is 70000 EUR.")
CASES = [
    {"name": "standard", "early": False, "correction": False, "max_turns": 6},
    {"name": "volunteered_details", "early": True, "correction": False, "max_turns": 2},
    {"name": "review_correction", "early": False, "correction": True, "max_turns": 8},
]

VERIFIERS = ("provider", "jev")


def run_case(controller, case, turn_limit=16, comparison=None):
    verifier = controller.verifier
    controller.verifier = RecordingVerifier(verifier, comparison)
    try:
        return _run_case(controller, case, turn_limit)
    finally:
        controller.verifier = verifier


def _run_case(controller, case, turn_limit):
    controller.opening()
    visits = {}
    turns = 0
    repeated_answers = 0
    correction_requested = False
    trace = []
    failure_reason = None
    started = time.monotonic()
    while controller.state not in (State.COMPLETE, State.STOPPED) and turns < turn_limit:
        state = controller.state.value
        if state not in ANSWERS and state != "review_correction":
            failure_reason = f"Unexpected state: {state}"
            break
        visits[state] = visits.get(state, 0) + 1
        repeated = visits[state] > 1 and state != "review_confirmation" and not controller.active_correction
        message = ANSWERS.get(state, "Please correct my salary expectation.")
        if case["early"] and turns == 0:
            message = EARLY
        elif case["early"] and state != "review_confirmation":
            repeated = True
        repeated_answers += int(repeated)
        if case["correction"] and state == "review_confirmation" and not correction_requested:
            message = "Please correct my salary expectation."
            correction_requested = True
        elif case["correction"] and controller.active_correction == "salary":
            message = "My correct annual salary expectation is 75000 EUR."
        elif state == "review_confirmation":
            wrong = [field for field, correct in record_checks(controller.record, case).items()
                     if field != "confirmation" and not correct]
            if wrong:
                failure_reason = f"Refused to confirm incorrect summary: {', '.join(wrong)}"
                break
        controller.verifier.current = None
        response = controller.handle(message)
        turn = controller.verifier.current or {"message": message, "state": state,
                                               "error": "Extraction failed before verification"}
        turn.update(response=response, next_state=controller.state.value,
                    record=deepcopy(controller.record))
        trace.append(turn)
        turns += 1
    checks = record_checks(controller.record, case)
    completed = controller.state == State.COMPLETE
    if failure_reason is None and not completed:
        failure_reason = "Turn limit reached" if turns >= turn_limit else f"Session ended: {controller.state.value}"
    comparison_errors = sum("comparison_error" in turn for turn in trace)
    return {"case": case["name"], "completed": completed, "record_correct": all(checks.values()),
            "failed_fields": [key for key, passed in checks.items() if not passed],
            "failure_reason": failure_reason,
            "comparison_errors": comparison_errors,
            "turns": turns, "repeated_answers": repeated_answers,
            "seconds": round(time.monotonic() - started, 2), "trace": trace,
            "passed": completed and all(checks.values()) and turns <= case["max_turns"] and not comparison_errors}


def record_checks(record, case):
    expected_salary = 75000 if case["correction"] else 70000
    return {
        "role": record["role"]["confirmed"] is True,
        "experience": record["experience"]["years"] == 5,
        "skills": record["experience"]["skills"] == ["Python"],
        "availability": record["availability"]["date"] == "2026-10-01",
        "authorization": record["work_authorization"]["authorized"] is True,
        "salary": all(record["salary_expectation"][key] == value for key, value in
                      (("min", expected_salary), ("max", expected_salary), ("currency", "EUR"))),
        "confirmation": record["candidate_confirmation"]["confirmed"] is True,
    }


def parse_verifiers(value):
    names = tuple(name.strip() for name in value.split(","))
    if any(name not in VERIFIERS for name in names):
        raise argparse.ArgumentTypeError(
            f"Choose a comma-separated list from: {', '.join(VERIFIERS)}"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("List each verifier only once")
    return names


def format_report(reports):
    headers = ("Case", "Verifier", "Result", "Record", "Turns", "Repeated", "Time")
    rows = [(
        report["case"], f"{report['verifier']} ({report['verifier_model']})",
        "PASS" if report["passed"] else "FAIL",
        "Correct" if report["record_correct"] else "Incorrect",
        str(report["turns"]), str(report["repeated_answers"]), f"{report['seconds']:.1f} s",
    ) for report in reports]
    widths = [max(len(header), *(len(row[index]) for row in rows))
              for index, header in enumerate(headers)]

    def table_row(values):
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    lines = ["Complete conversation evaluations", ""]
    for provider, model in dict.fromkeys((r["provider"], r["extraction_model"]) for r in reports):
        lines.append(f"Extraction: {provider} ({model})")
    lines.extend(["", table_row(headers), "-+-".join("-" * width for width in widths),
                  *(table_row(row) for row in rows), "",
                  f"Result: {sum(r['passed'] for r in reports)}/{len(reports)} conversations passed."])
    for report in reports:
        if not report["passed"]:
            reasons = [report["failure_reason"]] if report["failure_reason"] else []
            if report["failed_fields"]:
                reasons.append(f"Incorrect fields: {', '.join(report['failed_fields'])}")
            if report["comparison_errors"]:
                reasons.append(f"Comparison errors: {report['comparison_errors']}")
            lines.append(f"{report['case']} ({report['verifier']}): "
                         + "; ".join(reasons or ["Exceeded expected turn count"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "azure"), required=True)
    parser.add_argument("--verifier", type=parse_verifiers, default="provider", metavar="NAMES",
                        help="comma-separated verifiers to run in order: provider,jev (default: provider)")
    parser.add_argument("--output", type=Path, default=Path("data/output/evaluations"))
    parser.add_argument("--json", action="store_true", help="print detailed results as JSON instead of a table")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    role = load_role(root / "data/input/roles.json", "software-engineer")
    schema = json.loads((root / "data/input/schema.json").read_text())
    names = args.verifier
    clients = {name: create_model_clients(args.provider, name) for name in names}
    extractor = CachedExtractor(clients[names[0]][0])
    directory = args.output / uuid4().hex
    directory.mkdir(parents=True)
    reports = []
    for name in names:
        for case in CASES:
            comparison_name = next((other for other in names if other != name), None)
            comparison = clients[comparison_name][1] if comparison_name else None
            controller = ScreeningController(role, extractor, clients[name][1],
                JsonPersistence(directory / name / case["name"], schema), date(2026, 9, 25))
            report = run_case(controller, case, comparison=comparison)
            report.update(provider=args.provider, verifier=name, comparison_verifier=comparison_name,
                          extraction_model=extractor.extractor.model, verifier_model=clients[name][1].model)
            reports.append(report)
            (directory / "report.json").write_text(json.dumps(reports, indent=2))
    if args.json:
        print(json.dumps([{key: value for key, value in report.items() if key != "trace"}
                          for report in reports], indent=2))
    else:
        print(format_report(reports))
    print(f"Saved turn evidence: {directory / 'report.json'}")
    raise SystemExit(0 if all(report["passed"] for report in reports) else 1)


if __name__ == "__main__":
    main()
