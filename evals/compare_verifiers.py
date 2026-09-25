from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.holdout_cases import CASES as HOLDOUT_CASES
from evals.correction_cases import CASES as CORRECTION_CASES
from evals.verifier_cases import CASES as BENCHMARK_CASES, QUESTIONS
from src.extractor import AzureOpenAIVerifier, JevVerifier, OpenAIVerifier, Verifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATE = "2026-09-20"
PRICES_PER_MILLION = {
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "jev-1.13.0": {"input": 0.042, "output": 0.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare provider and Jev verification")
    parser.add_argument("--provider", choices=("openai", "azure"), required=True)
    parser.add_argument("--suite", choices=("benchmark", "holdout", "corrections"), default="benchmark")
    parser.add_argument("--high-risk-repeats", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args()
    if args.high_risk_repeats < 0:
        parser.error("--high-risk-repeats cannot be negative")

    roles = {
        role["id"]: role
        for role in json.loads((PROJECT_ROOT / "data/input/roles.json").read_text())
    }
    provider: Verifier = (
        AzureOpenAIVerifier() if args.provider == "azure" else OpenAIVerifier()
    )
    cases = {"benchmark": BENCHMARK_CASES, "holdout": HOLDOUT_CASES, "corrections": CORRECTION_CASES}[args.suite]
    report = {
        "suite": args.suite,
        "case_count": len(cases),
        "high_risk_case_count": sum(case["high_risk"] for case in cases),
        "high_risk_repeats": args.high_risk_repeats,
        "provider": evaluate(provider, roles, cases, args.high_risk_repeats),
        "jev": evaluate(JevVerifier(), roles, cases, args.high_risk_repeats),
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(report))


def format_report(report: dict[str, Any]) -> str:
    provider = report["provider"]
    jev = report["jev"]

    def ratio(result: dict[str, Any], passed: str, total: str) -> str:
        return f"{result[passed]}/{result[total]}"

    def cost(result: dict[str, Any]) -> str:
        value = result["estimated_usd"]
        return "n/a" if value is None else f"${value:.4f}"

    rows = [
        ("Correct cases", ratio(provider, "base_passed", "base_total"), ratio(jev, "base_passed", "base_total")),
        ("Correct high-risk cases", ratio(provider, "high_risk_passed", "high_risk_total"), ratio(jev, "high_risk_passed", "high_risk_total")),
        ("False accepts", str(provider["false_accepts"]), str(jev["false_accepts"])),
        ("False rejects", str(provider["false_rejects"]), str(jev["false_rejects"])),
        (
            "Inconsistent high-risk cases",
            str(len(provider["inconsistent_high_risk_cases"])),
            str(len(jev["inconsistent_high_risk_cases"])),
        ),
        ("Elapsed time", f"{provider['elapsed_seconds']:.1f} s", f"{jev['elapsed_seconds']:.1f} s"),
        ("Estimated cost", cost(provider), cost(jev)),
    ]
    headers = ("Measure", provider["model"], jev["model"])
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(3)
    ]

    def table_row(values: tuple[str, str, str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    lines = [
        f"Verifier comparison: {report['suite']}",
        f"Requests per verifier: {provider['request_count']}",
        "",
        table_row(headers),
        "-+-".join("-" * width for width in widths),
        *(table_row(row) for row in rows),
        "",
    ]
    if jev["base_passed"] > provider["base_passed"]:
        lines.append(
            f"Result: Jev had higher overall accuracy "
            f"({jev['base_passed']}/{jev['base_total']} versus "
            f"{provider['base_passed']}/{provider['base_total']})."
        )
    elif jev["base_passed"] < provider["base_passed"]:
        lines.append(
            f"Result: {provider['model']} had higher overall accuracy "
            f"({provider['base_passed']}/{provider['base_total']} versus "
            f"{jev['base_passed']}/{jev['base_total']})."
        )
    else:
        lines.append(f"Result: overall accuracy was tied at {jev['base_passed']}/{jev['base_total']}.")
    return "\n".join(lines)


def evaluate(
    verifier: Verifier,
    roles: dict[str, dict[str, Any]],
    cases: list[dict[str, Any]],
    high_risk_repeats: int,
) -> dict[str, Any]:
    runs = [(case, 0) for case in cases]
    runs.extend(
        (case, repetition)
        for repetition in range(1, high_risk_repeats + 1)
        for case in cases
        if case["high_risk"]
    )
    outcomes: dict[str, list[str]] = defaultdict(list)
    rows = []
    input_tokens = 0
    output_tokens = 0
    started = time.perf_counter()
    for case, repetition in runs:
        row = evaluate_case(verifier, roles[case["role_id"]], case)
        row["repetition"] = repetition
        rows.append(row)
        outcomes[case["id"]].append(row["outcome"])
        usage = getattr(verifier, "last_usage", {})
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))

    base_rows = [row for row in rows if row["repetition"] == 0]
    fact_rows = [row for row in base_rows if row["expected_fact"] is not None]
    kind_rows = [row for row in base_rows if row["expected_kind"] is not None]
    high_risk_ids = {case["id"] for case in cases if case["high_risk"]}
    high_risk_rows = [row for row in base_rows if row["id"] in high_risk_ids]
    inconsistent = sorted(case_id for case_id, values in outcomes.items() if len(set(values)) > 1)
    model = getattr(verifier, "model", "unknown")
    price = PRICES_PER_MILLION.get(model)
    cost = None
    if price:
        cost = input_tokens / 1_000_000 * price["input"]
        cost += output_tokens / 1_000_000 * price["output"]
    return {
        "model": model,
        "request_count": len(rows),
        "base_passed": sum(row["passed"] for row in base_rows),
        "base_total": len(base_rows),
        "fact_passed": sum(row["passed"] for row in fact_rows),
        "fact_total": len(fact_rows),
        "kind_passed": sum(row["passed"] for row in kind_rows),
        "kind_total": len(kind_rows),
        "high_risk_passed": sum(row["passed"] for row in high_risk_rows),
        "high_risk_total": len(high_risk_rows),
        "false_accepts": sum(
            row["expected_fact"] == "rejected" and row["actual_fact"] == "supported"
            for row in fact_rows
        ),
        "false_rejects": sum(
            row["expected_fact"] == "supported" and row["actual_fact"] == "rejected"
            for row in fact_rows
        ),
        "failures_by_category": dict(
            Counter(row["category"] for row in base_rows if not row["passed"])
        ),
        "inconsistent_high_risk_cases": inconsistent,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_usd": round(cost, 6) if cost is not None else None,
        "failures": [
            {
                "id": row["id"],
                "expected": row["expected_fact"] or row["expected_kind"],
                "actual": row["actual_fact"] or row["actual_kind"],
            }
            for row in base_rows
            if not row["passed"]
        ],
    }


def evaluate_case(
    verifier: Verifier, role: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    fact = case["fact"]
    extraction = {
        "response_kind": case["response_kind"],
        "facts": [] if fact is None else [fact],
        "availability_date": case["availability_date"],
        "availability_is_relative": case["availability_is_relative"],
    }
    question = QUESTIONS[case["state"]].format(**role)
    result = verifier.verify(
        case["message"],
        case["state"],
        role,
        {"current_question": question, "reference_date": REFERENCE_DATE},
        extraction,
    )
    actual_fact = result["facts"][0]["verdict"] if fact is not None else None
    actual_kind = result["response_kind_verdict"]
    expected_fact = case["expected_fact"]
    expected_kind = case["expected_kind"]
    passed = True
    if expected_fact is not None:
        passed = passed and actual_fact == expected_fact
    if expected_kind is not None:
        passed = passed and actual_kind == expected_kind
    outcome = actual_fact or actual_kind or "missing"
    return {
        "id": case["id"],
        "category": case["category"],
        "expected_fact": expected_fact,
        "actual_fact": actual_fact,
        "expected_kind": expected_kind,
        "actual_kind": actual_kind,
        "outcome": outcome,
        "passed": passed,
    }


if __name__ == "__main__":
    main()
