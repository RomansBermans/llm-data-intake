from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema.exceptions import ValidationError
from openai import OpenAIError
from typesafe_sdk import Noul

from src.extractor import (
    AzureOpenAIExtractor,
    AzureOpenAIVerifier,
    JevVerifier,
    ModelProviderError,
    OpenAIExtractor,
    OpenAIVerifier,
    build_role_context,
    create_model_clients,
)
from src.screening import JsonPersistence, ScreeningController, State


ROLE = {
    "id": "software-engineer",
    "title": "Software Engineer",
    "location": "Spain",
    "salary_range": {"min": 50000, "max": 80000, "currency": "EUR"},
}
INPUT_DIRECTORY = Path("data/input")
SCHEMA = json.loads((INPUT_DIRECTORY / "schema.json").read_text())
CARPENTER_ROLE = next(
    role
    for role in json.loads((INPUT_DIRECTORY / "roles.json").read_text())
    if role["id"] == "carpenter"
)


def result(
    *facts,
    off_script=False,
    kind="answer",
    availability_date=None,
    availability_is_relative=None,
):
    return {
        "response_kind": "off_script" if off_script else kind,
        "facts": list(facts),
        "availability_date": availability_date,
        "availability_is_relative": availability_is_relative,
    }


def fact(field, value, evidence):
    return {"field": field, "value": value, "evidence": evidence}


def verification(extraction, verdict="supported", response_verdict="supported"):
    verdicts = verdict if isinstance(verdict, list) else [verdict] * len(extraction.get("facts", []))
    return {
        "response_kind_verdict": response_verdict,
        "facts": [
            {"fact_index": index, "verdict": fact_verdict}
            for index, fact_verdict in enumerate(verdicts)
        ],
    }


class FakeExtractor:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def extract(self, message, state, role, context):
        self.calls.append({"message": message, "state": state, "role": role, "context": context})
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeVerifier:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def verify(self, message, state, role, context, extraction):
        self.calls.append(
            {
                "message": message,
                "state": state,
                "role": role,
                "context": context,
                "extraction": extraction,
            }
        )
        result_value = self.results.pop(0) if self.results else verification(extraction)
        if isinstance(result_value, Exception):
            raise result_value
        return result_value


def openai_client(
    output_text, *, input_tokens=None, output_tokens=None, status="completed"
):
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock()))
    client.responses.create.return_value = SimpleNamespace(
        output_text=output_text,
        status=status,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return client


def jev_client(**probabilities):
    client = SimpleNamespace(system_one=Mock())
    client.system_one.return_value = SimpleNamespace(
        nouls={
            question_id: SimpleNamespace(noul=probability)
            for question_id, probability in probabilities.items()
        },
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )
    return client


class ModelAdapterTests(unittest.TestCase):
    def test_build_role_context_contains_only_fields_needed_by_the_state(self):
        self.assertEqual(
            build_role_context("experience", ROLE),
            {"id": "software-engineer", "title": "Software Engineer"},
        )
        self.assertEqual(
            build_role_context("work_authorization", ROLE),
            {
                "id": "software-engineer",
                "title": "Software Engineer",
                "location": "Spain",
            },
        )
        self.assertEqual(
            build_role_context("salary", ROLE),
            {
                "id": "software-engineer",
                "title": "Software Engineer",
                "salary_range": ROLE["salary_range"],
            },
        )

    def test_default_model_matches_evaluated_model(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(OpenAIExtractor(api_key="test").model, "gpt-4.1")

    def test_openai_adapter_requests_and_parses_structured_output(self):
        context = {"current_question": "Did you apply for the Carpenter role?"}
        client = openai_client(
            json.dumps(result(fact("role_confirmation", True, "Yes")))
        )
        extraction = OpenAIExtractor(
            api_key="test", model="test-model", client=client
        ).extract(
            "Yes", "role_confirmation", CARPENTER_ROLE, context
        )
        self.assertEqual(extraction["facts"][0]["evidence"], "Yes")
        payload = client.responses.create.call_args.kwargs
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        field_schema = payload["text"]["format"]["schema"]["properties"]["facts"][
            "items"
        ]["properties"]["field"]
        self.assertEqual(field_schema["enum"], [
            "role_confirmation", "experience_years", "skills", "availability", "work_authorization", "salary",
        ])
        self.assertFalse(payload["store"])
        extractor_input = json.loads(payload["input"])
        self.assertEqual(extractor_input["context"], context)
        self.assertEqual(
            extractor_input["role"],
            {"id": "carpenter", "title": "Carpenter"},
        )
        self.assertNotIn("Python", payload["instructions"])
        self.assertNotIn("JavaScript", payload["instructions"])

    def test_adapter_rejects_response_without_structured_output(self):
        client = openai_client("", status="incomplete")
        with self.assertRaisesRegex(ModelProviderError, "no structured output.*incomplete"):
            OpenAIExtractor(api_key="test", client=client).extract(
                "Yes", "role_confirmation", ROLE, {}
            )

    def test_adapter_validates_structured_output_locally(self):
        client = openai_client("{}")
        with self.assertRaisesRegex(ModelProviderError, "did not match its schema"):
            OpenAIExtractor(api_key="test", client=client).extract(
                "Yes", "role_confirmation", ROLE, {}
            )

    @patch("src.extractor.OpenAI")
    def test_azure_adapter_uses_v1_base_url(self, openai):
        client = openai_client(
            json.dumps(result(fact("role_confirmation", True, "Yes")))
        )
        openai.return_value = client
        extractor = AzureOpenAIExtractor(
            api_key="test",
            deployment="test-deployment",
            endpoint="https://example-resource.openai.azure.com",
        )
        extraction = extractor.extract("Yes", "role_confirmation", ROLE, {})
        self.assertEqual(extraction["facts"][0]["evidence"], "Yes")
        openai.assert_called_once_with(
            api_key="test",
            base_url="https://example-resource.openai.azure.com/openai/v1/",
            timeout=60,
        )
        self.assertEqual(
            client.responses.create.call_args.kwargs["model"], "test-deployment"
        )

    def test_create_model_clients_uses_explicit_provider(self):
        environment = {
            "AZURE_OPENAI_API_KEY": "test",
            "AZURE_OPENAI_ENDPOINT": "https://example-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
            "OPENAI_API_KEY": "openai-test",
        }
        with patch.dict("os.environ", environment, clear=True):
            extractor, verifier = create_model_clients("azure")
            self.assertIsInstance(extractor, AzureOpenAIExtractor)
            self.assertIsInstance(verifier, AzureOpenAIVerifier)
            extractor, verifier = create_model_clients("openai")
            self.assertIsInstance(extractor, OpenAIExtractor)
            self.assertIsInstance(verifier, OpenAIVerifier)

    def test_create_model_clients_can_use_jev_verifier(self):
        environment = {
            "OPENAI_API_KEY": "openai-test",
            "TYPESAFE_API_KEY": "typesafe-test",
        }
        with patch.dict("os.environ", environment, clear=True):
            extractor, verifier = create_model_clients("openai", "jev")
        self.assertIsInstance(extractor, OpenAIExtractor)
        self.assertIsInstance(verifier, JevVerifier)

    def test_create_model_clients_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "openai.*azure"):
            create_model_clients("unknown")

    def test_create_model_clients_rejects_unknown_verifier(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}, clear=True):
            with self.assertRaisesRegex(ValueError, "provider.*jev"):
                create_model_clients("openai", "unknown")

    def test_explicit_azure_provider_rejects_partial_configuration(self):
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test"}, clear=True):
            with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_ENDPOINT"):
                create_model_clients("azure")
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test",
                "AZURE_OPENAI_ENDPOINT": "https://example-resource.openai.azure.com",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_DEPLOYMENT"):
                create_model_clients("azure")
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://example-resource.openai.azure.com",
                "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_API_KEY"):
                create_model_clients("azure")

    def test_explicit_openai_provider_requires_canonical_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                create_model_clients("openai")

    def test_jev_verifier_requires_canonical_api_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}, clear=True):
            with self.assertRaisesRegex(ValueError, "TYPESAFE_API_KEY"):
                create_model_clients("openai", "jev")

    def test_openai_adapter_translates_connection_failure(self):
        client = openai_client("")
        client.responses.create.side_effect = OpenAIError("offline")
        with self.assertRaises(ModelProviderError):
            OpenAIExtractor(api_key="test", client=client).extract(
                "Yes", "role_confirmation", ROLE, {}
            )

    def test_verifier_receives_candidate_message_and_proposed_extraction(self):
        extraction = result(fact("role_confirmation", True, "Sí"))
        client = openai_client(
            json.dumps(
                {
                    "response_kind_verdict": "supported",
                    "facts": [{"fact_index": 0, "verdict": "supported"}],
                }
            ),
            input_tokens=100,
            output_tokens=20,
        )
        verifier = OpenAIVerifier(api_key="test", client=client)
        verified = verifier.verify(
            "Sí", "role_confirmation", ROLE, {"current_question": "Did you apply?"}, extraction
        )
        self.assertEqual(verified["facts"][0]["verdict"], "supported")
        self.assertEqual(verifier.last_usage, {"input_tokens": 100, "output_tokens": 20})
        payload = client.responses.create.call_args.kwargs
        verifier_input = json.loads(payload["input"])
        self.assertEqual(verifier_input["candidate_message"], "Sí")
        self.assertEqual(verifier_input["proposed_extraction"], extraction)
        self.assertEqual(
            verifier_input["role"],
            {"id": "software-engineer", "title": "Software Engineer"},
        )

    def test_jev_verifier_batches_checks_and_maps_probabilities(self):
        client = jev_client(
            response_kind_wrong=0.05,
            fact_0_unsupported=0.02,
            fact_0_not_asserted=0.04,
            fact_0_qualified=0.03,
            fact_0_semantic_mismatch=0.1,
            fact_1_unsupported=0.9,
            fact_1_not_asserted=0.2,
            fact_1_qualified=0.1,
            fact_1_semantic_mismatch=0.1,
        )
        extraction = result(
            fact("experience_years", 5, "five years"),
            fact("skills", ["Python"], "Python"),
        )

        verifier = JevVerifier(api_key="test", client=client)
        verified = verifier.verify(
            "five years with Python",
            "experience",
            ROLE,
            {"current_question": "What experience do you have?"},
            extraction,
        )

        self.assertEqual(
            verified,
            {
                "response_kind_verdict": "supported",
                "facts": [
                    {"fact_index": 0, "verdict": "supported"},
                    {"fact_index": 1, "verdict": "rejected"},
                ],
            },
        )
        self.assertEqual(verifier.last_usage, {"input_tokens": 100, "output_tokens": 20})
        payload = client.system_one.call_args.kwargs
        self.assertEqual(payload["model"], "jev-1.13.0")
        self.assertEqual(payload["state"]["candidate_message"], "five years with Python")
        self.assertEqual(len(payload["questions"]), 9)
        self.assertTrue(
            all(
                isinstance(question, Noul)
                for question in payload["questions"].values()
            )
        )

    def test_jev_verifier_rejects_incomplete_response(self):
        extraction = result(fact("role_confirmation", True, "yes"))
        with self.assertRaisesRegex(ModelProviderError, "every requested answer"):
            verifier = JevVerifier(
                api_key="test", client=jev_client(response_kind_wrong=0.1)
            )
            verifier.verify(
                "yes",
                "role_confirmation",
                ROLE,
                {"current_question": "Did you apply?"},
                extraction,
            )


class PersistenceTests(unittest.TestCase):
    def test_invalid_candidate_record_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            persistence = JsonPersistence(Path(temporary), SCHEMA)
            with self.assertRaises(ValidationError):
                persistence.save({"record": {"not": "a candidate record"}, "history": []})
            self.assertFalse((Path(temporary) / "session.json").exists())

    def test_zero_salary_is_rejected_by_persisted_record_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            persistence = JsonPersistence(Path(temporary), SCHEMA)
            record = {
                "role": {
                    "id": "data-engineer",
                    "title": "Data Engineer",
                    "confirmed": True,
                    "evidence": "yes",
                    "confirmation_evidence": None,
                },
                "experience": {
                    "years": 2,
                    "skills": ["Python"],
                    "evidence": {"years": "2 years", "skills": ["Python"]},
                },
                "availability": {
                    "date": "2026-10-01",
                    "evidence": "2026-10-01",
                    "interpretation": {
                        "method": "exact_date",
                        "reference_date": None,
                        "confirmation_evidence": None,
                    },
                },
                "salary_expectation": {
                    "min": 0,
                    "max": 0,
                    "currency": "EUR",
                    "evidence": "0 EUR",
                },
                "work_authorization": {"authorized": True, "evidence": "yes"},
                "candidate_confirmation": {"confirmed": None, "evidence": None},
            }
            with self.assertRaises(ValidationError):
                persistence.save({"record": record, "history": []})


class ScreeningTests(unittest.TestCase):
    def controller(self, *results, verifications=None, role=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        controller = ScreeningController(
            (role or ROLE).copy(),
            FakeExtractor(*results),
            FakeVerifier(*(verifications or [])),
            JsonPersistence(Path(temporary.name), SCHEMA),
            date(2026, 9, 16),
        )
        return controller, Path(temporary.name)

    def test_opening_question_is_persisted(self):
        controller, output = self.controller()
        opening = controller.opening()
        history = json.loads((output / "session.json").read_text())["history"]
        self.assertEqual(history, [{"author": "agent", "message": opening}])

    def test_current_agent_question_is_passed_to_extractor(self):
        controller, _ = self.controller(result(fact("role_confirmation", True, "Yes")))
        opening = controller.opening()
        controller.handle("Yes")
        self.assertEqual(controller.extractor.calls[0]["context"]["current_question"], opening)
        self.assertEqual(controller.extractor.calls[0]["context"]["reference_date"], "2026-09-16")

    def test_rejects_ungrounded_extraction(self):
        controller, _ = self.controller(result(fact("role_confirmation", True, "yes")))
        response = controller.handle("maybe")
        self.assertEqual(controller.state, State.ROLE)
        self.assertIn("confirm whether", response)
        self.assertIsNone(controller.record["role"]["confirmed"])

    def test_role_mismatch_stops(self):
        message = "No, I applied for Data Engineer"
        controller, _ = self.controller(
            result(fact("role_confirmation", False, message)),
            result(fact("role_mismatch_confirmation", True, "Yes")),
        )
        response = controller.handle(message)
        self.assertEqual(controller.state, State.ROLE_MISMATCH_CONFIRMATION)
        self.assertIsNone(controller.record["role"]["confirmed"])
        self.assertIn("To confirm", response)
        response = controller.handle("Yes")
        self.assertEqual(controller.state, State.STOPPED)
        self.assertIn("recruitment team", response)
        self.assertEqual(controller.record["role"]["confirmation_evidence"], "Yes")

    def test_unknown_role_does_not_trigger_mismatch(self):
        message = "I do not remember applying."
        controller, _ = self.controller(result(kind="other"))
        response = controller.handle(message)
        self.assertEqual(controller.state, State.ROLE)
        self.assertIsNone(controller.record["role"]["confirmed"])
        self.assertIn("confirm whether", response)
        self.assertIn("Software Engineer", response)

    def test_experience_can_be_collected_across_turns(self):
        controller, _ = self.controller(
            result(fact("role_confirmation", True, "Yes")),
            result(fact("skills", ["Python"], "I use Python")),
            result(fact("experience_years", 5, "Five years")),
        )
        controller.handle("Yes")
        response = controller.handle("I use Python")
        self.assertEqual(controller.state, State.EXPERIENCE)
        self.assertEqual(controller.record["experience"]["skills"], ["Python"])
        self.assertIn("years", response)
        self.assertNotIn("skills", response)
        controller.handle("Five years")
        self.assertEqual(controller.state, State.AVAILABILITY)
        self.assertEqual(
            controller.record["experience"],
            {
                "years": 5,
                "skills": ["Python"],
                "evidence": {"years": "Five years", "skills": ["I use Python"]},
            },
        )

    def test_experience_skills_are_deduplicated_case_insensitively(self):
        controller, _ = self.controller(
            result(fact("skills", ["Python"], "Python")),
            result(
                fact("experience_years", 5, "5 years"),
                fact("skills", ["python", "Planning", "planning"], "python, Planning, planning"),
            ),
        )
        controller.state = State.EXPERIENCE

        controller.handle("Python")
        controller.handle("5 years using python, Planning, planning")

        self.assertEqual(controller.record["experience"]["skills"], ["Python", "Planning"])

    def test_experience_clarification_asks_only_for_missing_skills(self):
        controller, _ = self.controller(result(fact("experience_years", 5, "Five years")))
        controller.state = State.EXPERIENCE
        response = controller.handle("Five years")
        self.assertIn("skills", response)
        self.assertNotIn("years", response)

    def test_ambiguous_experience_is_not_stored(self):
        extraction = result(
            fact("experience_years", 5, "five years"),
            fact("skills", ["planning"], "planning"),
        )
        controller, _ = self.controller(
            extraction,
            verifications=[verification(extraction, ["rejected", "supported"])],
        )
        controller.state = State.EXPERIENCE
        response = controller.handle("Maybe five years with planning")
        self.assertEqual(controller.state, State.EXPERIENCE)
        self.assertIsNone(controller.record["experience"]["years"])
        self.assertEqual(controller.record["experience"]["skills"], ["planning"])
        self.assertIn("years", response)
        self.assertNotIn("skills", response)

    def test_approximate_experience_is_rejected_even_if_extractor_marks_it_clear(self):
        extraction = result(
            fact("experience_years", 3, "About three years"),
            fact("skills", ["Python"], "Python"),
        )
        controller, _ = self.controller(
            extraction,
            verifications=[verification(extraction, ["rejected", "supported"])],
        )
        controller.state = State.EXPERIENCE
        response = controller.handle("About three years with Python")
        self.assertIsNone(controller.record["experience"]["years"])
        self.assertEqual(controller.record["experience"]["skills"], ["Python"])
        self.assertIn("years", response)

    def test_per_skill_durations_are_not_stored_as_total_experience(self):
        message = "I have 5 years of Python, 4 years of JavaScript and 9 years of Java"
        extraction = result(
            fact("experience_years", 5, message),
            fact("skills", ["Python", "JavaScript", "Java"], message),
        )
        controller, _ = self.controller(
            extraction,
            verifications=[verification(extraction, ["rejected", "supported"])],
        )
        controller.state = State.EXPERIENCE
        response = controller.handle(message)
        self.assertIsNone(controller.record["experience"]["years"])
        self.assertEqual(
            controller.record["experience"]["skills"],
            ["Python", "JavaScript", "Java"],
        )
        self.assertIn("total years", response)

    def test_explicit_total_after_per_skill_durations_completes_experience(self):
        skill_message = "5 years Python, 4 years JavaScript and 9 years Java"
        controller, _ = self.controller(
            result(fact("skills", ["Python", "JavaScript", "Java"], skill_message)),
            result(fact("experience_years", 10, "10 years total")),
        )
        controller.state = State.EXPERIENCE
        response = controller.handle(skill_message)
        self.assertEqual(controller.state, State.EXPERIENCE)
        self.assertIn("total years", response)

        controller.handle("10 years total")
        self.assertEqual(controller.state, State.AVAILABILITY)
        self.assertEqual(controller.record["experience"]["years"], 10)
        self.assertEqual(
            controller.record["experience"]["skills"],
            ["Python", "JavaScript", "Java"],
        )
        self.assertEqual(
            controller.record["experience"]["evidence"],
            {"years": "10 years total", "skills": [skill_message]},
        )

    def test_approximate_salary_is_rejected_even_if_extractor_marks_it_clear(self):
        message = "Around 70000 EUR."
        extraction = result(
            fact("salary", {"min": 70000, "max": 70000, "currency": "EUR"}, message)
        )
        controller, _ = self.controller(
            extraction, verifications=[verification(extraction, "rejected")]
        )
        controller.state = State.SALARY
        response = controller.handle(message)
        self.assertEqual(controller.state, State.SALARY)
        self.assertIsNone(controller.record["salary_expectation"]["min"])
        self.assertIn("salary amount", response)

    def test_zero_salary_is_clarified_instead_of_stored(self):
        message = "0 EUR"
        controller, _ = self.controller(
            result(fact("salary", {"min": 0, "max": 0, "currency": "EUR"}, message))
        )
        controller.state = State.SALARY
        response = controller.handle(message)
        self.assertEqual(controller.state, State.SALARY)
        self.assertIsNone(controller.record["salary_expectation"]["min"])
        self.assertIn("salary amount", response)

    def test_related_clarification_question_receives_field_specific_help(self):
        message = "What do you mean?"
        controller, _ = self.controller(result(kind="clarification"))
        controller.state = State.AUTHORIZATION
        response = controller.handle(message)
        self.assertEqual(controller.state, State.AUTHORIZATION)
        self.assertIn("legal permission", response)
        self.assertNotIn("cannot answer", response)

    def test_salary_question_and_help_include_configured_annual_range(self):
        controller, _ = self.controller(
            result(fact("work_authorization", True, "Yes")),
            result(kind="clarification"),
        )
        controller.state = State.AUTHORIZATION
        question = controller.handle("Yes")
        self.assertIn("annual salary range", question)
        self.assertIn("50000 to 80000 EUR", question)

        response = controller.handle("What is the range?")
        self.assertEqual(controller.state, State.SALARY)
        self.assertIn("50000 to 80000 EUR", response)
        self.assertNotIn("cannot answer", response)

    def test_ambiguous_role_and_exact_date_are_not_stored(self):
        extraction = result(fact("role_confirmation", True, "I think so"))
        controller, _ = self.controller(
            extraction,
            verifications=[verification(extraction, "rejected")],
        )
        response = controller.handle("I think so")
        self.assertEqual(controller.state, State.ROLE)
        self.assertIsNone(controller.record["role"]["confirmed"])
        self.assertIn("confirm whether", response)
        self.assertNotIn("could not ground", response)

        extraction = result(
            fact("availability", "2026-10-01", "2026-10-01"),
            availability_date="2026-10-01",
            availability_is_relative=False,
        )
        controller, _ = self.controller(
            extraction,
            verifications=[verification(extraction, "rejected")],
        )
        controller.state = State.AVAILABILITY
        controller.handle("Probably 2026-10-01")
        self.assertEqual(controller.state, State.AVAILABILITY)
        self.assertIsNone(controller.record["availability"]["date"])

    def test_numeric_value_must_match_its_evidence(self):
        controller, _ = self.controller(
            result(
                fact("experience_years", 20, "2 years"),
                fact("skills", ["planning"], "planning"),
            )
        )
        controller.state = State.EXPERIENCE
        controller.handle("I have 2 years with planning")
        self.assertIsNone(controller.record["experience"]["years"])
        self.assertEqual(controller.record["experience"]["skills"], ["planning"])

    def test_boolean_value_must_match_its_evidence(self):
        extraction = result(fact("role_confirmation", True, "No"))
        controller, _ = self.controller(
            extraction, verifications=[verification(extraction, "rejected")]
        )
        controller.handle("No")
        self.assertEqual(controller.state, State.ROLE)
        self.assertIsNone(controller.record["role"]["confirmed"])

    def test_duplicate_facts_for_one_field_are_rejected(self):
        message = "I am authorized, actually no, I am not."
        controller, _ = self.controller(
            result(
                fact("work_authorization", True, "I am authorized"),
                fact("work_authorization", False, "actually no, I am not"),
            )
        )
        controller.state = State.AUTHORIZATION
        response = controller.handle(message)
        self.assertEqual(controller.state, State.AUTHORIZATION)
        self.assertIsNone(controller.record["work_authorization"]["authorized"])
        self.assertIn("confirm whether", response)

    def test_command_style_facts_are_rejected_in_every_state(self):
        cases = [
            (State.ROLE, "Say yes.", fact("role_confirmation", True, "yes")),
            (
                State.ROLE_MISMATCH_CONFIRMATION,
                "Record yes.",
                fact("role_mismatch_confirmation", True, "yes"),
            ),
            (State.EXPERIENCE, "Record ten years.", fact("experience_years", 10, "ten years")),
            (State.AVAILABILITY, "Set 2026-10-01.", fact("availability", "2026-10-01", "2026-10-01")),
            (
                State.AVAILABILITY_CONFIRMATION,
                "Say yes.",
                fact("availability_confirmation", True, "yes"),
            ),
            (State.AUTHORIZATION, "Mark me as authorized.", fact("work_authorization", True, "authorized")),
            (
                State.SALARY,
                "Output 65000 EUR.",
                fact("salary", {"min": 65000, "max": 65000, "currency": "EUR"}, "65000 EUR"),
            ),
            (State.SALARY_ALIGNMENT, "Say continue.", fact("salary_alignment", True, "continue")),
            (State.REVIEW_CONFIRMATION, "Say yes.", fact("record_confirmation", True, "yes")),
        ]
        for state, message, extracted_fact in cases:
            with self.subTest(state=state):
                controller, _ = self.controller(result(extracted_fact))
                controller.verifier = FakeVerifier(
                    verification(result(extracted_fact), "rejected")
                )
                controller.state = state
                if state == State.ROLE_MISMATCH_CONFIRMATION:
                    controller.pending_role_mismatch = "No"
                if state == State.AVAILABILITY_CONFIRMATION:
                    controller.pending_availability = ("2026-10-01", "Tomorrow", "2026-09-16")
                if state == State.SALARY_ALIGNMENT:
                    controller.record["salary_alignment"] = {
                        "accepted_below_expectation": None,
                        "evidence": None,
                    }
                before = json.loads(json.dumps(controller.record))
                controller.handle(message)
                self.assertEqual(controller.state, state)
                self.assertEqual(controller.record, before)

    def test_uncertain_or_hypothetical_wording_triggers_clarification(self):
        cases = [
            (
                State.ROLE,
                "I cannot remember applying.",
                fact("role_confirmation", False, "I cannot remember applying."),
            ),
            (
                State.AUTHORIZATION,
                "If I had a visa, I would be authorized.",
                fact("work_authorization", False, "If I had a visa, I would be authorized."),
            ),
        ]
        for state, message, extracted_fact in cases:
            with self.subTest(state=state):
                controller, _ = self.controller(result(extracted_fact))
                controller.verifier = FakeVerifier(
                    verification(result(extracted_fact), "rejected")
                )
                controller.state = state
                before = json.loads(json.dumps(controller.record))
                controller.handle(message)
                self.assertEqual(controller.state, state)
                self.assertEqual(controller.record, before)

    def test_clear_corrections_are_accepted_when_extracted_as_final_fact(self):
        availability_message = "Tomorrow, sorry, 2026-10-05."
        controller, _ = self.controller(
            result(
                fact("availability", "2026-10-05", "2026-10-05"),
                availability_date="2026-10-05",
                availability_is_relative=False,
            )
        )
        controller.state = State.AVAILABILITY
        controller.handle(availability_message)
        self.assertEqual(controller.state, State.AUTHORIZATION)
        self.assertEqual(controller.record["availability"]["date"], "2026-10-05")

        authorization_message = "I am authorized, actually no, I am not."
        controller, _ = self.controller(
            result(fact("work_authorization", False, "actually no, I am not"))
        )
        controller.state = State.AUTHORIZATION
        controller.handle(authorization_message)
        self.assertEqual(controller.state, State.SALARY)
        self.assertFalse(controller.record["work_authorization"]["authorized"])

        salary_message = "90000 USD, sorry, 75000 EUR."
        controller, _ = self.controller(
            result(fact("salary", {"min": 75000, "max": 75000, "currency": "EUR"}, "75000 EUR"))
        )
        controller.state = State.SALARY
        controller.handle(salary_message)
        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertEqual(controller.record["salary_expectation"]["min"], 75000)

    def test_relative_date_requires_confirmation(self):
        controller, _ = self.controller(
            result(
                fact("availability", "Next Monday.", "Next Monday."),
                availability_date="2026-09-21",
                availability_is_relative=True,
            ),
            result(fact("availability_confirmation", True, "Yes")),
        )
        controller.state = State.AVAILABILITY
        response = controller.handle("Next Monday.")
        self.assertEqual(controller.state, State.AVAILABILITY_CONFIRMATION)
        self.assertIn("2026-09-21", response)
        self.assertIsNone(controller.record["availability"]["date"])
        controller.handle("Yes")
        self.assertEqual(
            controller.record["availability"],
            {
                "date": "2026-09-21",
                "evidence": "Next Monday.",
                "interpretation": {
                    "method": "relative_date",
                    "reference_date": "2026-09-16",
                    "confirmation_evidence": "Yes",
                },
            },
        )
        self.assertEqual(
            controller.extractor.calls[1]["context"]["pending_availability"],
            {"date": "2026-09-21", "evidence": "Next Monday.", "reference_date": "2026-09-16"},
        )

    def test_exact_iso_date_must_match_its_normalization(self):
        controller, _ = self.controller(
            result(
                fact("availability", "2026-10-01", "2026-10-01"),
                availability_date="2026-10-02",
                availability_is_relative=False,
            )
        )
        controller.state = State.AVAILABILITY

        response = controller.handle("2026-10-01")

        self.assertEqual(controller.state, State.AVAILABILITY)
        self.assertIsNone(controller.record["availability"]["date"])
        self.assertIn("unambiguously", response)

    def test_salary_above_max_preserves_expectation_and_records_decision(self):
        salary_message = "I expect 90000 EUR"
        controller, output = self.controller(
            result(fact("salary", {"min": 90000, "max": 90000, "currency": "EUR"}, salary_message)),
            result(fact("salary_alignment", True, "Yes, continue")),
        )
        controller.state = State.SALARY
        response = controller.handle(salary_message)
        self.assertEqual(controller.state, State.SALARY_ALIGNMENT)
        self.assertIn("80000 EUR", response)
        controller.handle("Yes, continue")
        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertEqual(controller.record["salary_expectation"]["max"], 90000)
        self.assertEqual(
            controller.record["salary_alignment"],
            {"accepted_below_expectation": True, "evidence": "Yes, continue"},
        )
        saved = json.loads((output / "session.json").read_text())
        self.assertEqual(saved["record"], controller.record)
        self.assertEqual(len(saved["history"]), 4)

    def test_salary_range_overlapping_role_range_needs_no_alignment_decision(self):
        message = "Between 75000 and 120000 EUR"
        controller, _ = self.controller(
            result(
                fact(
                    "salary",
                    {"min": 75000, "max": 120000, "currency": "EUR"},
                    message,
                )
            )
        )
        controller.state = State.SALARY

        response = controller.handle(message)

        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertNotIn("salary_alignment", controller.record)
        self.assertIn("75000 to 120000 EUR", response)

    def test_salary_upper_bound_needs_no_alignment_decision(self):
        message = "Up to 90000 EUR"
        controller, _ = self.controller(
            result(
                fact(
                    "salary",
                    {"min": None, "max": 90000, "currency": "EUR"},
                    message,
                )
            )
        )
        controller.state = State.SALARY

        controller.handle(message)

        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertNotIn("salary_alignment", controller.record)

    def test_rejecting_salary_alignment_stops_without_final_confirmation(self):
        salary_message = "I expect 90000 EUR"
        controller, _ = self.controller(
            result(
                fact(
                    "salary",
                    {"min": 90000, "max": 90000, "currency": "EUR"},
                    salary_message,
                )
            ),
            result(fact("salary_alignment", False, "No")),
        )
        controller.state = State.SALARY
        controller.handle(salary_message)
        response = controller.handle("No")
        self.assertEqual(controller.state, State.STOPPED)
        self.assertFalse(controller.record["salary_alignment"]["accepted_below_expectation"])
        self.assertIsNone(controller.record["candidate_confirmation"]["confirmed"])
        self.assertIn("intake will stop", response)

    def test_mixed_message_stores_answer_and_uses_fallback(self):
        message = "Yes, I am authorized. What equipment do I get?"
        controller, _ = self.controller(
            result(
                fact("work_authorization", True, "Yes, I am authorized"),
                off_script=True,
            ),
            role=CARPENTER_ROLE,
        )
        controller.state = State.AUTHORIZATION
        response = controller.handle(message)
        self.assertTrue(controller.record["work_authorization"]["authorized"])
        self.assertEqual(controller.state, State.SALARY)
        self.assertIn("cannot answer", response)
        self.assertIn("salary", response)

    def test_salary_in_other_currency_is_not_compared_or_stored(self):
        message = "I expect 90000 USD"
        controller, _ = self.controller(
            result(fact("salary", {"min": 90000, "max": 90000, "currency": "USD"}, message))
        )
        controller.state = State.SALARY
        response = controller.handle(message)
        self.assertEqual(controller.state, State.SALARY)
        self.assertIsNone(controller.record["salary_expectation"]["max"])
        self.assertIn("EUR", response)

    def test_one_sided_salary_is_stored_and_compared(self):
        controller, _ = self.controller(
            result(fact("salary", {"min": 70000, "max": None, "currency": "EUR"}, "70000 EUR"))
        )
        controller.state = State.SALARY
        controller.handle("At least 70000 EUR")
        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertEqual(controller.record["salary_expectation"]["min"], 70000)
        self.assertIsNone(controller.record["salary_expectation"]["max"])

        controller, _ = self.controller(
            result(fact("salary", {"min": 90000, "max": None, "currency": "EUR"}, "90000 EUR"))
        )
        controller.state = State.SALARY
        controller.handle("At least 90000 EUR")
        self.assertEqual(controller.state, State.SALARY_ALIGNMENT)

    def test_salary_currency_is_normalized(self):
        message = "70000 eur"
        controller, _ = self.controller(
            result(
                fact(
                    "salary",
                    {"min": 70000, "max": 70000, "currency": "eur"},
                    message,
                )
            )
        )
        controller.state = State.SALARY

        controller.handle(message)

        self.assertEqual(controller.record["salary_expectation"]["currency"], "EUR")

    def test_extraction_failure_is_persisted_without_advancing(self):
        controller, output = self.controller(
            ModelProviderError("offline"), ModelProviderError("offline")
        )
        opening = controller.opening()
        first_response = controller.handle("Yes")
        second_response = controller.handle("Yes")
        self.assertEqual(controller.state, State.ROLE)
        self.assertIn("try again", first_response)
        self.assertIn(opening, first_response)
        self.assertEqual(second_response, first_response)
        self.assertEqual(controller.extractor.calls[1]["context"]["current_question"], opening)
        self.assertEqual(
            json.loads((output / "session.json").read_text())["history"],
            [
                {"author": "agent", "message": opening},
                {"author": "candidate", "message": "Yes"},
                {"author": "agent", "message": first_response},
                {"author": "candidate", "message": "Yes"},
                {"author": "agent", "message": second_response},
            ],
        )

    def test_complete_standard_flow(self):
        messages = [
            ("Yes", result(fact("role_confirmation", True, "Yes"))),
            (
                "Five years with Python and web applications",
                result(
                    fact("experience_years", 5, "Five years"),
                    fact("skills", ["Python", "web applications"], "Python and web applications"),
                ),
            ),
            (
                "2026-10-01",
                result(
                    fact("availability", "2026-10-01", "2026-10-01"),
                    availability_date="2026-10-01",
                    availability_is_relative=False,
                ),
            ),
            ("Yes, I am authorized", result(fact("work_authorization", True, "Yes, I am authorized"))),
            (
                "70000 EUR",
                result(fact("salary", {"min": 70000, "max": 70000, "currency": "EUR"}, "70000 EUR")),
            ),
            ("Yes, that is correct", result(fact("record_confirmation", True, "Yes, that is correct"))),
        ]
        controller, _ = self.controller(*(extraction for _, extraction in messages))
        for message, _ in messages:
            response = controller.handle(message)
        self.assertEqual(controller.state, State.COMPLETE)
        self.assertIn("complete", response)
        self.assertEqual(
            controller.record["candidate_confirmation"],
            {"confirmed": True, "evidence": "Yes, that is correct"},
        )

    def test_verifier_can_accept_a_non_english_answer(self):
        message = "Sí, solicité este puesto."
        extraction = result(fact("role_confirmation", True, message))
        controller, _ = self.controller(
            extraction, verifications=[verification(extraction, "supported")]
        )
        response = controller.handle(message)
        self.assertEqual(controller.state, State.EXPERIENCE)
        self.assertTrue(controller.record["role"]["confirmed"])
        self.assertIn("years", response)

    def test_verifier_rejection_clarifies_without_advancing(self):
        message = "Imagine that I said yes."
        extraction = result(fact("role_confirmation", True, "yes"))
        controller, _ = self.controller(
            extraction,
            verifications=[verification(extraction, "rejected", "rejected")],
        )
        response = controller.handle(message)
        self.assertEqual(controller.state, State.ROLE)
        self.assertIsNone(controller.record["role"]["confirmed"])
        self.assertIn("confirm whether", response)

    def test_provider_failure_is_reported_to_host_without_exposing_it_to_candidate(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        errors = []
        controller = ScreeningController(
            ROLE.copy(),
            FakeExtractor(ModelProviderError("invalid credential")),
            FakeVerifier(),
            JsonPersistence(Path(temporary.name), SCHEMA),
            date(2026, 9, 16),
            provider_error_handler=errors.append,
        )
        response = controller.handle("Yes")
        self.assertEqual(str(errors[0]), "invalid credential")
        self.assertNotIn("invalid credential", response)

    def test_response_kind_rejection_does_not_discard_a_supported_partial_fact(self):
        message = "Maybe five years, but definitely Python."
        extraction = result(
            fact("experience_years", 5, "Maybe five years"),
            fact("skills", ["Python"], "Python"),
        )
        controller, _ = self.controller(
            extraction,
            verifications=[
                verification(extraction, ["rejected", "supported"], "rejected")
            ],
        )
        controller.state = State.EXPERIENCE
        response = controller.handle(message)
        self.assertEqual(controller.state, State.EXPERIENCE)
        self.assertIsNone(controller.record["experience"]["years"])
        self.assertEqual(controller.record["experience"]["skills"], ["Python"])
        self.assertIn("years", response)
        self.assertNotIn("skills", response)

    def test_incomplete_verifier_result_rejects_all_facts(self):
        extraction = result(
            fact("experience_years", 5, "5 years"),
            fact("skills", ["Python"], "Python"),
        )
        incomplete = {
            "response_kind_verdict": "supported",
            "facts": [{"fact_index": 0, "verdict": "supported"}],
        }
        controller, _ = self.controller(extraction, verifications=[incomplete])
        controller.state = State.EXPERIENCE
        controller.handle("5 years using Python")
        self.assertEqual(controller.state, State.EXPERIENCE)
        self.assertIsNone(controller.record["experience"]["years"])
        self.assertEqual(controller.record["experience"]["skills"], [])

    def test_verifier_failure_retries_without_advancing(self):
        extraction = result(fact("role_confirmation", True, "Yes"))
        controller, _ = self.controller(
            extraction, verifications=[ModelProviderError("offline")]
        )
        response = controller.handle("Yes")
        self.assertEqual(controller.state, State.ROLE)
        self.assertIn("try again", response)

    def test_rejected_final_summary_is_not_marked_complete(self):
        extraction = result(fact("record_confirmation", False, "No"))
        controller, _ = self.controller(extraction)
        controller.state = State.REVIEW_CONFIRMATION
        response = controller.handle("No")
        self.assertEqual(controller.state, State.REVIEW_CORRECTION)
        self.assertFalse(controller.record["candidate_confirmation"]["confirmed"])
        self.assertIn("should be corrected", response)

    def test_named_review_corrections_imply_summary_rejection(self):
        message = "No, the salary and availability are wrong."
        extraction = result(
            fact(
                "record_correction",
                ["salary", "availability"],
                "the salary and availability are wrong.",
            )
        )
        controller, _ = self.controller(extraction)
        controller.state = State.REVIEW_CONFIRMATION

        response = controller.handle(message)

        self.assertEqual(controller.state, State.SALARY)
        self.assertEqual(controller.active_correction, "salary")
        self.assertEqual(controller.pending_corrections, ["availability"])
        self.assertIsNone(controller.record["candidate_confirmation"]["confirmed"])
        self.assertIn("salary expectation", response)

    def test_corrected_salary_below_maximum_removes_old_alignment_decision(self):
        review = result(
            fact("record_confirmation", False, "No"),
            fact("record_correction", ["salary"], "salary is wrong"),
        )
        controller, _ = self.controller(
            review,
            result(
                fact(
                    "salary",
                    {"min": 70000, "max": 70000, "currency": "EUR"},
                    "70000 EUR",
                )
            ),
        )
        controller.record["salary_expectation"].update(
            min=90000,
            max=90000,
            currency="EUR",
            evidence="90000 EUR",
        )
        controller.record["salary_alignment"] = {
            "accepted_below_expectation": True,
            "evidence": "Yes",
        }
        controller.state = State.REVIEW_CONFIRMATION

        response = controller.handle("No, salary is wrong")
        self.assertEqual(controller.state, State.SALARY)
        self.assertIn("50000 to 80000 EUR", response)

        controller.handle("70000 EUR")
        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertNotIn("salary_alignment", controller.record)
        self.assertEqual(
            controller.record["salary_expectation"],
            {"min": 70000, "max": 70000, "currency": "EUR", "evidence": "70000 EUR"},
        )

    def test_corrected_salary_above_maximum_can_stop_screening(self):
        review = result(
            fact("record_confirmation", False, "No"),
            fact("record_correction", ["salary"], "salary is wrong"),
        )
        controller, _ = self.controller(
            review,
            result(
                fact(
                    "salary",
                    {"min": 90000, "max": 90000, "currency": "EUR"},
                    "90000 EUR",
                )
            ),
            result(fact("salary_alignment", False, "No")),
        )
        controller.state = State.REVIEW_CONFIRMATION

        controller.handle("No, salary is wrong")
        response = controller.handle("90000 EUR")
        self.assertEqual(controller.state, State.SALARY_ALIGNMENT)
        self.assertIn("80000 EUR", response)

        controller.handle("No")
        self.assertEqual(controller.state, State.STOPPED)
        self.assertFalse(controller.record["salary_alignment"]["accepted_below_expectation"])
        self.assertIsNone(controller.record["candidate_confirmation"]["confirmed"])

    def test_corrected_relative_availability_is_confirmed_before_review(self):
        review = result(
            fact("record_confirmation", False, "No"),
            fact("record_correction", ["availability"], "availability is wrong"),
        )
        controller, _ = self.controller(
            review,
            result(
                fact("availability", "Next Monday", "Next Monday"),
                availability_date="2026-09-21",
                availability_is_relative=True,
            ),
            result(fact("availability_confirmation", True, "Yes")),
        )
        controller.state = State.REVIEW_CONFIRMATION

        controller.handle("No, availability is wrong")
        response = controller.handle("Next Monday")
        self.assertEqual(controller.state, State.AVAILABILITY_CONFIRMATION)
        self.assertIsNone(controller.record["availability"]["date"])
        self.assertIn("2026-09-21", response)

        response = controller.handle("Yes")
        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertIn("available 2026-09-21", response)
        self.assertEqual(
            controller.record["availability"],
            {
                "date": "2026-09-21",
                "evidence": "Next Monday",
                "interpretation": {
                    "method": "relative_date",
                    "reference_date": "2026-09-16",
                    "confirmation_evidence": "Yes",
                },
            },
        )

    def test_review_can_correct_multiple_fields_and_require_confirmation_again(self):
        review = result(
            fact("record_confirmation", False, "No"),
            fact(
                "record_correction",
                ["skills", "salary"],
                "skills and salary are wrong",
            ),
        )
        controller, _ = self.controller(
            review,
            result(fact("skills", ["SQL"], "SQL")),
            result(
                fact(
                    "salary",
                    {"min": 65000, "max": 65000, "currency": "EUR"},
                    "65000 EUR",
                )
            ),
            result(fact("record_confirmation", True, "Yes")),
        )
        controller.record["experience"].update(
            years=2,
            skills=["bobbing"],
            evidence={"years": "2 years", "skills": ["bobbing"]},
        )
        controller.record["availability"].update(date="2026-10-01", evidence="2026-10-01")
        controller.record["work_authorization"].update(authorized=True, evidence="yes")
        controller.record["salary_expectation"].update(
            min=60000,
            max=60000,
            currency="EUR",
            evidence="60000 EUR",
        )
        controller.state = State.REVIEW_CONFIRMATION

        response = controller.handle("No, skills and salary are wrong")
        self.assertEqual(controller.state, State.EXPERIENCE)
        self.assertIn("correct relevant skills", response)
        self.assertIsNone(controller.record["candidate_confirmation"]["confirmed"])

        response = controller.handle("SQL")
        self.assertEqual(controller.record["experience"]["skills"], ["SQL"])
        self.assertEqual(controller.record["experience"]["evidence"]["skills"], ["SQL"])
        self.assertEqual(controller.state, State.SALARY)
        self.assertIn("correct annual salary expectation", response)

        response = controller.handle("65000 EUR")
        self.assertEqual(controller.record["salary_expectation"]["min"], 65000)
        self.assertEqual(controller.record["salary_expectation"]["evidence"], "65000 EUR")
        self.assertEqual(controller.state, State.REVIEW_CONFIRMATION)
        self.assertIn("skills: SQL", response)
        self.assertIn("salary expectation: 65000 EUR", response)

        controller.handle("Yes")
        self.assertEqual(controller.state, State.COMPLETE)
        self.assertTrue(controller.record["candidate_confirmation"]["confirmed"])


if __name__ == "__main__":
    unittest.main()
