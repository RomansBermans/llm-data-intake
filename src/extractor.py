from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from openai import OpenAI, OpenAIError
from typesafe_sdk import Noul, TypeSafeClient, TypeSafeError


class Extractor(Protocol):
    def extract(
        self, message: str, state: str, role: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]: ...


class Verifier(Protocol):
    def verify(
        self,
        message: str,
        state: str,
        role: dict[str, Any],
        context: dict[str, Any],
        extraction: dict[str, Any],
    ) -> dict[str, Any]: ...


class ModelProviderError(RuntimeError):
    """The model provider could not return usable structured output."""


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "response_kind": {
            "type": "string",
            "enum": ["answer", "clarification", "off_script", "other"],
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "value": {
                        "anyOf": [
                            {"type": "boolean"},
                            {"type": "integer"},
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                            {
                                "type": "object",
                                "properties": {
                                    "min": {"type": ["number", "null"]},
                                    "max": {"type": ["number", "null"]},
                                    "currency": {"type": ["string", "null"]},
                                },
                                "required": ["min", "max", "currency"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["field", "value", "evidence"],
                "additionalProperties": False,
            },
        },
        "availability_date": {"type": ["string", "null"]},
        "availability_is_relative": {"type": ["boolean", "null"]},
    },
    "required": [
        "response_kind",
        "facts",
        "availability_date",
        "availability_is_relative",
    ],
    "additionalProperties": False,
}


FIELDS_BY_STATE = {
    "role_confirmation": ["role_confirmation"],
    "role_mismatch_confirmation": ["role_mismatch_confirmation"],
    "experience": ["experience_years", "skills"],
    "availability": ["availability"],
    "availability_confirmation": ["availability_confirmation"],
    "work_authorization": ["work_authorization"],
    "salary": ["salary"],
    "salary_alignment": ["salary_alignment"],
    "review_confirmation": ["record_confirmation", "record_correction"],
    "review_correction": ["record_correction"],
}


VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "response_kind_verdict": {
            "type": "string",
            "enum": ["supported", "rejected"],
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact_index": {"type": "integer", "minimum": 0},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "rejected"],
                    },
                },
                "required": ["fact_index", "verdict"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["response_kind_verdict", "facts"],
    "additionalProperties": False,
}


def extraction_schema(state: str) -> dict[str, Any]:
    schema = deepcopy(EXTRACTION_SCHEMA)
    schema["properties"]["facts"]["description"] = (
        "All explicit supported details in the entire candidate message, including details "
        "volunteered before their question. Check every allowed field; do not stop after "
        "finding the answer to the current question. Omit fields without evidence."
    )
    field_schema = schema["properties"]["facts"]["items"]["properties"]["field"]
    field_schema["enum"] = list(FIELDS_BY_STATE[state])
    if state in {"role_confirmation", "experience", "availability", "work_authorization", "salary"}:
        field_schema["enum"] = list(dict.fromkeys(field_schema["enum"] + [
            "experience_years", "skills", "availability", "work_authorization", "salary",
        ]))
    return schema


def build_role_context(state: str, role: dict[str, Any]) -> dict[str, Any]:
    context = {"id": role["id"], "title": role["title"]}
    if state == "work_authorization":
        context["location"] = role["location"]
    elif state in {"salary", "salary_alignment"}:
        context["salary_range"] = role["salary_range"]
    return context


class _ResponsesAPIClient:
    """Strict structured output through the official OpenAI SDK."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.last_usage: dict[str, int] = {}

    def _request_structured_output(
        self, instructions: str, input_value: dict[str, Any], name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            response = self.client.responses.create(
                model=self.model,
                store=False,
                instructions=instructions,
                input=json.dumps(input_value),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except OpenAIError as error:
            raise ModelProviderError(f"Model request failed: {error}") from error
        self.last_usage = _usage_dict(response.usage)
        if not response.output_text:
            raise ModelProviderError(
                f"Model response contained no structured output (status: {response.status})"
            )
        try:
            result = json.loads(response.output_text)
        except (json.JSONDecodeError, TypeError) as error:
            raise ModelProviderError("Model structured output was not valid JSON") from error
        if not isinstance(result, dict):
            raise ModelProviderError("Model structured output was not an object")
        try:
            Draft202012Validator(schema).validate(result)
        except ValidationError as error:
            raise ModelProviderError(
                "Model structured output did not match its schema"
            ) from error
        return result


class _OpenAIResponsesClient(_ResponsesAPIClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: OpenAI | None = None,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when --provider openai is used")
        super().__init__(
            client=client or OpenAI(api_key=api_key, timeout=60),
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4.1"),
        )


class _AzureOpenAIResponsesClient(_ResponsesAPIClient):
    def __init__(
        self,
        api_key: str | None = None,
        deployment: str | None = None,
        endpoint: str | None = None,
        client: OpenAI | None = None,
    ):
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required when --provider azure is used")
        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required when --provider azure is used")
        deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is required when --provider azure is used")
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/openai/v1"):
            endpoint = f"{endpoint}/openai/v1"
        super().__init__(
            client=client
            or OpenAI(api_key=api_key, base_url=f"{endpoint}/", timeout=60),
            model=deployment,
        )


class _ExtractorMixin:
    def extract(
        self, message: str, state: str, role: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        instructions = (
            "Extract explicit candidate facts answering the current question or volunteering other screening details allowed by the schema. "
            "The conversation state tells you which question was asked, NOT which details to collect. "
            "Read the whole message and check EVERY extraction_fields entry independently. "
            "Return every supported field, even if the current question has already been answered. "
            "A bare yes or no answers only the current question; never reuse it for other fields. "
            "Volunteered work authorization must explicitly refer to permission to work in the role location. "
            "Never infer missing facts. Copy each evidence value character-for-character as one "
            "contiguous substring of candidate_message. Preserve the candidate's wording for strings "
            "and skills. Return at most one fact per field and include every safely extractable partial "
            "fact. Values found only in commands, requests, quotations, examples, hypotheticals, or "
            "third-party statements are not candidate facts. For a clear self-correction, extract only "
            "the final value. Exception: record_correction captures a request to change named details, "
            "not a factual value; extract the named fields from an explicit correction request. "
            "This exception only covers a direct, present request about the candidate's own record. "
            "Do not extract correction intents from hypotheticals, quotations, or third-party requests. "
            "Do not choose between unresolved alternatives. A qualified value such as "
            "'maybe five years' or 'about 70000' is not an exact fact; omit it while preserving any "
            "separate clear facts in the same response. Apply this rule in every language. "
            "Classify the response as answer, clarification, off_script, or other. Use "
            "clarification when the candidate asks what the current question means or requests available "
            "context such as the role title or salary range. Use other for anything that is neither an "
            "answer nor a question. Use off_script for an unrelated question, "
            "including when the same message also contains a valid fact. "
            "An explicit request to correct the summary is an answer to the review question. "
            "Valid fields are role_confirmation, role_mismatch_confirmation, experience_years, skills, "
            "availability, availability_confirmation, work_authorization, salary, "
            "salary_alignment, record_confirmation, and record_correction. The response schema limits "
            "which fields are valid in the current state. Boolean fields must be booleans. "
            "experience_years is an explicitly stated total, not a duration attached only to one skill. "
            "When separate skills have different durations without a stated total, extract the skills "
            "and omit experience_years. Skills must be presented as occupational or professional "
            "abilities, without judging suitability for the role. "
            "salary is an object with min, max, and currency. Use null for an unstated bound, and null "
            "for all three values only when the candidate is explicitly open or has no expectation. "
            "record_correction is an array of experience_years, skills, availability, "
            "work_authorization, or salary. A general experience correction maps to both experience "
            "fields. In role_mismatch_confirmation, true means the candidate confirms they did not "
            "apply for the stated role. "
            "When an availability fact is extracted, set availability_date to the interpreted ISO date and "
            "availability_is_relative to whether the wording was relative. Use context.reference_date. "
            "Return null for both availability fields when no availability fact is extracted."
        )
        return self._request_structured_output(
            instructions,
            {
                "state": state,
                "extraction_fields": extraction_schema(state)["properties"]["facts"]["items"]["properties"]["field"]["enum"],
                "role": build_role_context(state, role),
                "context": context,
                "candidate_message": message,
            },
            "candidate_facts",
            extraction_schema(state),
        )


class _VerifierMixin:
    def verify(
        self,
        message: str,
        state: str,
        role: dict[str, Any],
        context: dict[str, Any],
        extraction: dict[str, Any],
    ) -> dict[str, Any]:
        instructions = (
            "Independently verify the proposed extraction against candidate_message and the current "
            "question. Understand the candidate's language directly. Return one verdict for every "
            "proposed fact index and a separate verdict for response_kind. Do not repair or add facts. "
            "A supported fact must be a clear assertion about the candidate answering the current "
            "question or volunteering another screening detail. Commands, requests to record a value, quotations, examples, third-party claims, "
            "and bare yes/no answers reused for a different question must be rejected. "
            "Exception: record_correction is a request to correct explicitly named details, not a "
            "factual assertion. It is supported ONLY for a direct, present request about the candidate's "
            "own record; a replacement value is not required. Reject hypothetical corrections ('if ... "
            "I would ask'), quotations, and corrections to someone else's record ('my friend wants ...'). "
            "This exception never permits accepting a requested factual value as an assertion. "
            "Classify a correction request during summary review as an answer. "
            "Hypotheticals, approximations, and unresolved alternatives must be rejected. "
            "In particular, reject a proposed numeric fact qualified by uncertainty or approximation, "
            "such as 'maybe five years' or 'about 70000', in any language. Judge each fact separately "
            "so a different clear fact in the same response can remain supported. "
            "A clear final self-correction may be supported. Verify meaning, not candidate suitability. "
            "experience_years requires an explicit total; a duration attached only to one skill is not "
            "a total. A skill must be presented as an occupational ability but need not appear in the "
            "role requirements. An explicitly open or negotiable "
            "salary supports null bounds and currency. For availability, verify the ISO normalization "
            "and relative-date flag against context.reference_date."
        )
        return self._request_structured_output(
            instructions,
            {
                "state": state,
                "role": build_role_context(state, role),
                "context": context,
                "candidate_message": message,
                "proposed_extraction": extraction,
            },
            "candidate_fact_verification",
            VERIFICATION_SCHEMA,
        )


class OpenAIExtractor(_ExtractorMixin, _OpenAIResponsesClient):
    pass


class OpenAIVerifier(_VerifierMixin, _OpenAIResponsesClient):
    pass


class AzureOpenAIExtractor(_ExtractorMixin, _AzureOpenAIResponsesClient):
    pass


class AzureOpenAIVerifier(_VerifierMixin, _AzureOpenAIResponsesClient):
    pass


class JevVerifier:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-1.13.0",
        rejection_threshold: float = 0.7,
        client: TypeSafeClient | None = None,
    ):
        api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY is required when --verifier jev is used")
        if not 0 <= rejection_threshold <= 1:
            raise ValueError("Jev rejection threshold must be between 0 and 1")
        self.model = model
        self.rejection_threshold = rejection_threshold
        self.client = client or TypeSafeClient(api_key=api_key, timeout=60)
        self.last_usage: dict[str, int] = {}

    def verify(
        self,
        message: str,
        state: str,
        role: dict[str, Any],
        context: dict[str, Any],
        extraction: dict[str, Any],
    ) -> dict[str, Any]:
        del role
        proposed_facts = {
            f"fact_{index}": {
                **fact,
                "field_rule": _jev_field_rule(fact.get("field")),
            }
            for index, fact in enumerate(extraction.get("facts", []))
        }
        jev_state = {
            "screening_state": state,
            "current_question": context.get("current_question"),
            "role_location": context.get("role_location"),
            "candidate_message": message,
            "proposed_response_kind": extraction.get("response_kind"),
            "response_kind_definitions": {
                "answer": "answers the current question, including a request to correct named details during summary review",
                "clarification": "asks what the current question means or requests relevant context",
                "off_script": "asks an unrelated question, with or without an answer",
                "other": "is neither an answer nor a question",
            },
            "proposed_facts": proposed_facts,
            "availability_interpretation": {
                "date": extraction.get("availability_date"),
                "is_relative": extraction.get("availability_is_relative"),
                "reference_date": context.get("reference_date"),
            },
        }
        questions = {
            "response_kind_wrong": _jev_question(
                "Does proposed_response_kind misclassify candidate_message relative to "
                "current_question and response_kind_definitions?",
                "the proposed response kind is wrong",
                "the proposed response kind is correct",
            )
        }
        for fact_id in proposed_facts:
            questions[f"{fact_id}_unsupported"] = _jev_question(
                f"Is proposed_facts.{fact_id} unsupported by or absent from candidate_message?",
                "the proposed value is not directly supported by the candidate's words",
                "the proposed value is directly supported by the candidate's words",
            )
            questions[f"{fact_id}_not_asserted"] = _jev_question(
                f"Is proposed_facts.{fact_id} based on a command, request, quotation, example, "
                "hypothetical, or third-party claim rather than a direct assertion about the candidate?",
                "the proposed fact is not a clear assertion about the candidate",
                "the proposed fact is a clear assertion about the candidate",
            )
            if proposed_facts[fact_id]["field"] == "record_correction":
                questions[f"{fact_id}_not_asserted"] = _jev_question(
                    f"Is proposed_facts.{fact_id} NOT a genuine request by the candidate to correct "
                    "their own named details? A direct request such as 'please correct my salary' "
                    "is valid; it need not assert a replacement value. Quotations, hypothetical "
                    "requests, and requests on someone else's behalf are not valid.",
                    "the candidate does not directly request this correction to their own details",
                    "the candidate directly requests this correction to their own details",
                )
            questions[f"{fact_id}_qualified"] = _jev_question(
                f"Does candidate_message qualify proposed_facts.{fact_id} with uncertainty, "
                "approximation, or an unresolved alternative? An explicitly open or negotiable "
                "salary with null bounds is a clear answer, not uncertainty.",
                "the proposed value is uncertain, approximate, or unresolved",
                "the proposed value is stated clearly and exactly, including explicit salary openness",
            )
            questions[f"{fact_id}_semantic_mismatch"] = _jev_question(
                f"Does proposed_facts.{fact_id} violate its field_rule? A candidate may volunteer "
                "screening details before being asked; that alone is not a violation. "
                "A bare yes/no answers only current_question and cannot support other fields.",
                "the field meaning does not support this proposed fact",
                "the proposed fact follows its field meaning",
            )

        answers = self._request(jev_state, questions)
        kind_verdict = self._verdict(answers, "response_kind_wrong")
        fact_verdicts = []
        for index in range(len(proposed_facts)):
            probabilities = [
                self._probability(answers, f"fact_{index}_{check}")
                for check in (
                    "unsupported",
                    "not_asserted",
                    "qualified",
                    "semantic_mismatch",
                )
            ]
            fact_verdicts.append(
                {
                    "fact_index": index,
                    "verdict": (
                        "rejected"
                        if max(probabilities) >= self.rejection_threshold
                        else "supported"
                    ),
                }
            )
        return {"response_kind_verdict": kind_verdict, "facts": fact_verdicts}

    def _request(
        self, state: dict[str, Any], questions: dict[str, Noul]
    ) -> dict[str, Any]:
        try:
            response = self.client.system_one(
                state=state,
                questions=questions,
                model=self.model,
            )
        except TypeSafeError as error:
            raise ModelProviderError(f"Jev request failed: {error}") from error
        answers = response.nouls
        if set(answers) != set(questions):
            raise ModelProviderError("Jev response did not contain every requested answer")
        self.last_usage = _usage_dict(response.usage)
        return answers

    def _verdict(self, answers: dict[str, Any], question_id: str) -> str:
        return (
            "rejected"
            if self._probability(answers, question_id) >= self.rejection_threshold
            else "supported"
        )

    @staticmethod
    def _probability(answers: dict[str, Any], question_id: str) -> float:
        answer = answers.get(question_id)
        probability = getattr(answer, "noul", None)
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not 0 <= probability <= 1
        ):
            raise ModelProviderError(f"Jev answer was invalid: {question_id}")
        return float(probability)


def _jev_question(instructions: str, true: str, false: str) -> Noul:
    return Noul(instructions=instructions, criteria={"true": true, "false": false})


def _usage_dict(usage: Any) -> dict[str, int]:
    return {
        name: value
        for name in ("input_tokens", "output_tokens")
        if isinstance(value := getattr(usage, name, None), int)
    }


def _jev_field_rule(field: Any) -> str:
    return {
        "experience_years": "The value is an explicitly stated total, not a duration for one skill.",
        "skills": "Every value is presented as an occupational or professional ability.",
        "availability": "The evidence states when the candidate can start.",
        "work_authorization": "The candidate explicitly states permission to work in role_location, or directly answers the current authorization question.",
        "role_mismatch_confirmation": (
            "True means the candidate confirms they did not apply for the stated role."
        ),
        "salary": "Null bounds are valid only when the candidate is explicitly open or negotiable.",
        "record_correction": "Every value names a detail the candidate says should be corrected.",
    }.get(field, "The value has the ordinary meaning of the named field.")


def create_model_clients(
    provider: str, verifier_provider: str = "provider"
) -> tuple[Extractor, Verifier]:
    if verifier_provider not in {"provider", "jev"}:
        raise ValueError("verifier provider must be 'provider' or 'jev'")
    if provider == "azure":
        extractor: Extractor = AzureOpenAIExtractor()
        verifier: Verifier = (
            JevVerifier() if verifier_provider == "jev" else AzureOpenAIVerifier()
        )
    elif provider == "openai":
        extractor = OpenAIExtractor()
        verifier = JevVerifier() if verifier_provider == "jev" else OpenAIVerifier()
    else:
        raise ValueError("provider must be 'openai' or 'azure'")
    return extractor, verifier
