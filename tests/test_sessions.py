import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from evals.conversations import CASES, run_case
from src.sessions import open_session, summary
from src.screening import State
from src.extractor import ModelProviderError
from src.web import make_handler, render_summary, serve
from tests.test_screening import ROLE, SCHEMA, FakeExtractor, FakeVerifier, fact, result
from tests.test_screening import verification


def volunteered(relative=False):
    availability = "next Monday" if relative else "2026-10-01"
    message = f"Yes. 5 years Python {availability} authorized in Spain 70000 EUR"
    extraction = result(
        fact("role_confirmation", True, "Yes"),
        fact("experience_years", 5, "5 years"), fact("skills", ["Python"], "Python"),
        fact("availability", availability, availability),
        fact("work_authorization", True, "authorized in Spain"),
        fact("salary", {"min": 70000, "max": 70000, "currency": "EUR"}, "70000 EUR"),
        availability_date="2026-10-01", availability_is_relative=relative,
    )
    return message, extraction


class SessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def create(self, *extractions, resume=None):
        return open_session(self.root, ROLE, SCHEMA, FakeExtractor(*extractions), FakeVerifier(), resume)

    def test_new_runs_do_not_overwrite_previous_session(self):
        first, controller = self.create()
        controller.opening()
        before = (self.root / first / "session.json").read_text()
        second, another = self.create()
        another.opening()
        self.assertNotEqual(first, second)
        self.assertEqual(before, (self.root / first / "session.json").read_text())

    def test_session_saves_only_snapshot_and_summary_after_resume(self):
        identifier, controller = self.create(result(fact("role_confirmation", True, "Yes")))
        controller.opening()
        controller.handle("Yes")
        _, resumed = self.create(resume=identifier)
        directory = self.root / identifier
        self.assertEqual({path.name for path in directory.iterdir()}, {"session.json", "summary.md"})
        snapshot = json.loads((directory / "session.json").read_text())
        self.assertEqual(snapshot["record"], resumed.record)
        self.assertEqual(snapshot["history"], resumed.history)
        self.assertEqual(resumed.state, State.EXPERIENCE)
        self.assertEqual((directory / "summary.md").read_text(), summary(resumed))

    def test_early_answers_skip_questions_and_require_final_confirmation(self):
        message, extraction = volunteered()
        _, controller = self.create(extraction, result(fact("record_confirmation", True, "Yes")))
        controller.opening()
        controller.handle(message)
        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertIsNone(controller.record["candidate_confirmation"]["confirmed"])
        controller.handle("Yes")
        self.assertEqual(controller.state, State.COMPLETE)
        self.assertIn('Candidate replied: "70000 EUR"', summary(controller))
        self.assertIn("**Candidate confirmed summary:** Yes", summary(controller))
        self.assertIn("# Recruitment Intake Assistant", summary(controller))
        self.assertIn("## Recruiter Summary", summary(controller))
        self.assertIn("## Annual salary expectation", summary(controller))
        self.assertIn('> Candidate replied: "70000 EUR"', summary(controller))

    def test_resume_pending_date_and_early_answers(self):
        message, extraction = volunteered(relative=True)
        identifier, controller = self.create(extraction)
        controller.opening()
        controller.handle(message)
        self.assertEqual(controller.state, State.AVAILABILITY_CONFIRMATION)
        _, resumed = self.create(result(fact("availability_confirmation", True, "Yes")), resume=identifier)
        self.assertEqual(resumed.opening(), controller.current_question)
        self.assertEqual(resumed.reference_date, controller.reference_date)
        resumed.handle("Yes")
        self.assertEqual(resumed.state, State.REVIEW_CONFIRMATION)
        self.assertEqual(resumed.record["salary_expectation"]["min"], 70000)

    def test_resume_pending_correction_and_completed_session(self):
        message, extraction = volunteered()
        identifier, controller = self.create(extraction, result(fact("record_correction", ["skills"], "skills")))
        controller.opening()
        controller.handle(message)
        controller.handle("Correct my skills")
        _, resumed = self.create(result(fact("skills", ["Java"], "Java")), result(fact("record_confirmation", True, "Yes")), resume=identifier)
        resumed.handle("Java")
        resumed.handle("Yes")
        _, finished = self.create(resume=identifier)
        self.assertEqual(finished.state, State.COMPLETE)
        self.assertEqual(finished.record["experience"]["skills"], ["Java"])
        self.assertIn("complete", finished.opening())

    def test_invalid_or_missing_session_does_not_create_output(self):
        for identifier in ("../escape", "missing"):
            with self.assertRaises(ValueError):
                self.create(resume=identifier)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejected_early_fact_is_not_saved(self):
        message, extraction = volunteered()
        _, controller = self.create(extraction)
        controller.verifier = FakeVerifier({"response_kind_verdict": "supported", "facts": [
            {"fact_index": i, "verdict": "rejected" if i == 5 else "supported"} for i in range(6)
        ]})
        controller.handle(message)
        self.assertEqual(controller.state, State.SALARY)
        self.assertIsNone(controller.record["salary_expectation"]["min"])

    def test_early_salary_still_requires_alignment(self):
        message, extraction = volunteered()
        message = message.replace("70000", "90000")
        extraction["facts"][-1] = fact("salary", {"min": 90000, "max": 90000, "currency": "EUR"}, "90000 EUR")
        _, controller = self.create(extraction)
        response = controller.handle(message)
        self.assertEqual(controller.state, State.SALARY_ALIGNMENT)
        self.assertIn("80000", response)
        self.assertIsNone(controller.record["candidate_confirmation"]["confirmed"])

    def test_evaluation_records_failure_and_repeated_answers(self):
        _, controller = self.create(result(), result(), result())
        report = run_case(controller, CASES[0], turn_limit=3)
        self.assertFalse(report["completed"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["repeated_answers"], 2)

    def test_evaluation_detects_correct_fast_completion_and_wrong_record(self):
        _, extraction = volunteered()
        # Use exact evidence from the evaluation script's candidate message.
        extraction["facts"][1]["evidence"] = "5 total years"
        extraction["facts"][4]["evidence"] = "authorized to work in Spain"
        _, controller = self.create(extraction, result(fact("record_confirmation", True, "Yes")))
        report = run_case(controller, CASES[1])
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["turns"], 2)
        controller.record["experience"]["years"] = 99
        report = run_case(controller, CASES[1])
        self.assertFalse(report["record_correct"])
        self.assertIn("experience", report["failed_fields"])

    def test_evaluation_never_confirms_summary_after_rejected_correction(self):
        _, extraction = volunteered()
        extraction["facts"][1]["evidence"] = "5 total years"
        extraction["facts"][4]["evidence"] = "authorized to work in Spain"
        correction = result(fact("record_correction", ["salary"], "Please correct my salary expectation."))
        _, controller = self.create(extraction, correction)
        controller.verifier = FakeVerifier(verification(extraction), verification(correction, "rejected"))
        report = run_case(controller, {**CASES[2], "early": True})
        self.assertFalse(report["completed"])
        self.assertIn("Refused to confirm incorrect summary: salary", report["failure_reason"])
        self.assertEqual(report["turns"], 2)
        self.assertIsNone(controller.record["candidate_confirmation"]["confirmed"])
        self.assertEqual(report["trace"][1]["extraction"], correction)
        self.assertEqual(report["trace"][1]["verification"]["facts"][0]["verdict"], "rejected")

    def test_recorded_comparison_uses_same_facts_without_driving_controller(self):
        extraction = result(fact("role_confirmation", True, "Yes"))
        _, controller = self.create(extraction)
        comparison = FakeVerifier(verification(extraction, "rejected"))
        report = run_case(controller, CASES[0], turn_limit=1, comparison=comparison)
        turn = report["trace"][0]
        self.assertEqual(comparison.calls[0]["extraction"], extraction)
        self.assertEqual(turn["verification"]["facts"][0]["verdict"], "supported")
        self.assertEqual(turn["comparison_verification"]["facts"][0]["verdict"], "rejected")
        self.assertEqual(turn["next_state"], "experience")

    def test_comparison_failure_is_recorded_and_cannot_pass(self):
        _, extraction = volunteered()
        extraction["facts"][1]["evidence"] = "5 total years"
        extraction["facts"][4]["evidence"] = "authorized to work in Spain"
        _, controller = self.create(extraction, result(fact("record_confirmation", True, "Yes")))
        comparison = FakeVerifier(ModelProviderError("Unavailable"))
        report = run_case(controller, CASES[1], comparison=comparison)
        self.assertTrue(report["completed"])
        self.assertTrue(report["record_correct"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["comparison_errors"], 1)
        self.assertEqual(report["trace"][0]["comparison_error"], "Unavailable")

    def test_summary_html_formats_generated_markdown_and_escapes_candidate_text(self):
        _, controller = self.create()
        controller.record["experience"]["skills"] = ['<img src=x onerror=alert(1)>', 'C++', '*Python*']
        controller.record["experience"]["evidence"]["skills"] = ['<script>alert(1)</script>']
        html = render_summary(summary(controller))
        self.assertIn("<h2>Recruitment Intake Assistant</h2>", html)
        self.assertIn("<h3>Recruiter Summary</h3>", html)
        self.assertIn("<strong>Role:</strong>", html)
        self.assertIn("<h3>Skills</h3>", html)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;, C++, *Python*", html)
        self.assertIn("<blockquote>Candidate replied:", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script", html)
        self.assertNotIn("<img", html)

    def test_startup_link_preserves_token_and_plain_redirected_output(self):
        _, controller = self.create()
        url = "http://127.0.0.1:8123/#test-token"
        for terminal in (True, False):
            with self.subTest(terminal=terminal), patch("src.web.HTTPServer") as server_type, \
                    patch("src.web.secrets.token_urlsafe", return_value="test-token"), \
                    patch("src.web.sys.stdout.isatty", return_value=terminal), \
                    patch("builtins.print") as output:
                server = server_type.return_value.__enter__.return_value
                server.server_port = 8123
                serve(controller, 8123)
                expected = f"\033]8;;{url}\033\\{url}\033]8;;\033\\" if terminal else url
                output.assert_called_once_with(f"Open {expected}")

    def test_browser_rejects_unauthorized_writes_and_serves_persisted_turn(self):
        _, controller = self.create(result(fact("role_confirmation", True, "Yes")))
        controller.opening()
        server = HTTPServer(("127.0.0.1", 0), make_handler(controller, "secret"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("POST", "/message", json.dumps({"message": "Yes"}))
        response = connection.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        self.assertEqual(controller.state, State.ROLE)
        connection.request("POST", "/message", json.dumps({"message": "Yes"}), {"X-Session-Token": "secret"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        data = json.loads(response.read())
        self.assertEqual(data["state"], "experience")
        self.assertEqual(data["summary_html"], render_summary(data["summary"]))
        self.assertEqual(json.loads((controller.persistence.directory / "session.json").read_text())["state"], "experience")
