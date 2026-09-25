import argparse
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evals.compare_verifiers import format_report
from evals.recording import CachedExtractor
from evals import conversations
from tests.test_screening import FakeExtractor, ROLE, fact, result


class EvaluationReportTests(unittest.TestCase):
    def test_verifier_list_preserves_order_and_accepts_single_selection(self):
        self.assertEqual(conversations.parse_verifiers("jev,provider"), ("jev", "provider"))
        self.assertEqual(conversations.parse_verifiers("provider, jev"), ("provider", "jev"))
        self.assertEqual(conversations.parse_verifiers("jev"), ("jev",))

    def test_verifier_list_rejects_invalid_or_duplicate_names(self):
        for value in ("", "all", "both", "unknown", "jev,", ",provider", "jev,jev"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                conversations.parse_verifiers(value)

    def test_cli_runs_only_selected_verifiers_in_requested_order(self):
        for names in (None, "jev", "jev,provider", "provider,jev"):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                args = ["conversations", "--provider", "azure", "--output", directory]
                if names is not None:
                    args.extend(["--verifier", names])
                expected = names.split(",") if names else ["provider"]
                with patch("sys.argv", args), patch.object(conversations, "create_model_clients",
                    side_effect=lambda *_: (SimpleNamespace(model="extractor"), SimpleNamespace(model="verifier"))) as clients, patch.object(
                        conversations, "run_case", side_effect=lambda _controller, case, **_kwargs: {
                            "case": case["name"], "passed": True, "record_correct": True,
                            "turns": 2, "repeated_answers": 0, "seconds": 1.25,
                        }
                    ) as run, patch("builtins.print"), self.assertRaises(SystemExit) as exited:
                    conversations.main()
                self.assertEqual(exited.exception.code, 0)
                self.assertEqual([call.args[1] for call in clients.call_args_list], expected)
                self.assertEqual(run.call_count, len(expected) * len(conversations.CASES))

    def test_extraction_reused_only_for_identical_input_and_cannot_be_mutated(self):
        source = FakeExtractor(result(fact("role_confirmation", True, "Yes")), result())
        cached = CachedExtractor(source)
        first = cached.extract("Yes", "role_confirmation", ROLE, {"current_question": "Applied?"})
        first["facts"].clear()
        second = cached.extract("Yes", "role_confirmation", ROLE, {"current_question": "Applied?"})
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(len(second["facts"]), 1)
        cached.extract("Yes", "role_confirmation", ROLE, {"current_question": "Different?"})
        self.assertEqual(len(source.calls), 2)

    def test_formats_terminal_summary(self):
        result = {
            "model": "model",
            "request_count": 3,
            "base_passed": 2,
            "base_total": 2,
            "high_risk_passed": 1,
            "high_risk_total": 1,
            "false_accepts": 0,
            "false_rejects": 0,
            "inconsistent_high_risk_cases": [],
            "elapsed_seconds": 1.25,
            "estimated_usd": 0.001,
        }
        report = {"suite": "holdout", "provider": {**result, "base_passed": 1}, "jev": result}

        output = format_report(report)

        self.assertIn("Verifier comparison: holdout", output)
        self.assertIn("Correct cases", output)
        self.assertIn("Result: Jev had higher overall accuracy", output)

    def test_conversation_table_shows_single_and_multiple_verifiers_and_failures(self):
        report = {"case": "standard", "provider": "azure", "extraction_model": "gpt-4.1",
                  "verifier": "jev", "verifier_model": "jev-1.13.0", "passed": True,
                  "record_correct": True, "turns": 6, "repeated_answers": 0, "seconds": 1.25}
        output = conversations.format_report([report])
        self.assertIn("jev (jev-1.13.0)", output)
        self.assertIn("PASS", output)
        self.assertIn("1/1 conversations passed", output)
        failed = {**report, "verifier": "provider", "verifier_model": "gpt-4.1",
                  "passed": False, "record_correct": False, "failed_fields": ["salary"],
                  "failure_reason": "Turn limit reached", "comparison_errors": 1}
        output = conversations.format_report([report, failed])
        self.assertIn("provider (gpt-4.1)", output)
        self.assertIn("FAIL", output)
        self.assertIn("1/2 conversations passed", output)
        self.assertIn("Turn limit reached; Incorrect fields: salary; Comparison errors: 1", output)
