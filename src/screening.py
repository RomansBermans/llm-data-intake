from __future__ import annotations

import json
import re
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator, FormatChecker

from .extractor import Extractor, ModelProviderError, Verifier


class State(str, Enum):
    ROLE = "role_confirmation"
    ROLE_MISMATCH_CONFIRMATION = "role_mismatch_confirmation"
    EXPERIENCE = "experience"
    AVAILABILITY = "availability"
    AVAILABILITY_CONFIRMATION = "availability_confirmation"
    AUTHORIZATION = "work_authorization"
    SALARY = "salary"
    SALARY_ALIGNMENT = "salary_alignment"
    REVIEW_CONFIRMATION = "review_confirmation"
    REVIEW_CORRECTION = "review_correction"
    COMPLETE = "complete"
    STOPPED = "stopped"


QUESTIONS = {
    State.ROLE: "Can you confirm that you applied for the {title} role?",
    State.ROLE_MISMATCH_CONFIRMATION: "To confirm, did you mean that you did not apply for the {title} role?",
    State.EXPERIENCE: "How many total years of relevant professional experience do you have, and which skills have you used?",
    State.AVAILABILITY: "When would you be available to start?",
    State.AUTHORIZATION: "Are you authorized to work in {location}?",
}

CORRECTION_STATES = {
    "experience_years": State.EXPERIENCE,
    "skills": State.EXPERIENCE,
    "availability": State.AVAILABILITY,
    "work_authorization": State.AUTHORIZATION,
    "salary": State.SALARY,
}

CORRECTION_QUESTIONS = {
    "experience_years": "Please state the correct total years of relevant professional experience.",
    "skills": "Please state the correct relevant skills you have used.",
    "availability": "Please state the correct date when you would be available to start.",
    "work_authorization": "Please confirm whether you are authorized to work in {location}.",
}


class JsonPersistence:
    def __init__(self, directory: Path, schema: dict[str, Any]):
        self.directory = directory
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema, format_checker=FormatChecker())

    def save(self, snapshot: dict[str, Any]) -> None:
        self.validator.validate(snapshot["record"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write("session.json", snapshot)

    def _write(self, name: str, value: Any) -> None:
        target = self.directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(target)


class ScreeningController:
    def __init__(
        self,
        role: dict[str, Any],
        extractor: Extractor,
        verifier: Verifier,
        persistence: JsonPersistence,
        reference_date: date | None = None,
        provider_error_handler: Callable[[ModelProviderError], None] | None = None,
    ):
        self.role = role
        self.extractor = extractor
        self.verifier = verifier
        self.persistence = persistence
        self.reference_date = reference_date or date.today()
        self.provider_error_handler = provider_error_handler
        self.state = State.ROLE
        self.history: list[dict[str, str]] = []
        self.current_question: str | None = None
        self.pending_role_mismatch: str | None = None
        self.pending_availability: tuple[str, str, str] | None = None
        self.pending_corrections: list[str] = []
        self.active_correction: str | None = None
        self.early_facts: dict[str, dict[str, Any]] = {}
        self.record = {
            "role": {
                "id": role["id"],
                "title": role["title"],
                "confirmed": None,
                "evidence": None,
                "confirmation_evidence": None,
            },
            "experience": {"years": None, "skills": [], "evidence": {"years": None, "skills": []}},
            "availability": {
                "date": None,
                "evidence": None,
                "interpretation": {
                    "method": None,
                    "reference_date": None,
                    "confirmation_evidence": None,
                },
            },
            "salary_expectation": {"min": None, "max": None, "currency": None, "evidence": None},
            "work_authorization": {"authorized": None, "evidence": None},
            "candidate_confirmation": {"confirmed": None, "evidence": None},
        }

    def opening(self) -> str:
        if self.history:
            return self.current_question or self.history[-1]["message"]
        message = self._question_for_state()
        self.current_question = message
        if not self.history:
            self.history.append({"author": "agent", "message": message})
            self.save()
        return message

    def handle(self, message: str) -> str:
        if self.state in (State.COMPLETE, State.STOPPED):
            return "This intake has ended."
        self.history.append({"author": "candidate", "message": message})
        context = self._extraction_context()
        try:
            extraction = self.extractor.extract(message, self.state.value, self.role, context)
            verification = self.verifier.verify(
                message, self.state.value, self.role, context, extraction
            )
        except ModelProviderError as error:
            if self.provider_error_handler:
                self.provider_error_handler(error)
            response = "I could not process that response right now. Please try again."
            if context["current_question"]:
                response = f"{response} {context['current_question']}"
            self.history.append({"author": "agent", "message": response})
            self.save()
            return response
        verified_indices = self._supported_fact_indices(extraction, verification)
        facts = self._grounded_facts(extraction, message, verified_indices)
        if self.state in (State.ROLE, State.EXPERIENCE, State.AVAILABILITY, State.AUTHORIZATION, State.SALARY) and not self.active_correction:
            stages = {"experience_years": 1, "skills": 1, "availability": 2, "work_authorization": 3, "salary": 4}
            stage = {State.ROLE: 0, State.EXPERIENCE: 1, State.AVAILABILITY: 2, State.AUTHORIZATION: 3, State.SALARY: 4}[self.state]
            for field, order in stages.items():
                if field in facts and order > stage:
                    self.early_facts[field] = {"fact": facts[field], "extraction": extraction}
        kind_verified = verification.get("response_kind_verdict") == "supported"
        response_kind = extraction.get("response_kind")
        off_script = response_kind == "off_script" and kind_verified
        if response_kind == "clarification" and kind_verified and not facts:
            response = self._help_prompt()
        else:
            response = self._apply_facts(facts, off_script, extraction)
            response = self._use_early_facts(response, off_script)
        self.current_question = response
        self.history.append({"author": "agent", "message": response})
        self.save()
        return response

    def _use_early_facts(self, response: str, off_script: bool) -> str:
        fields = {State.EXPERIENCE: ("experience_years", "skills"), State.AVAILABILITY: ("availability",),
                  State.AUTHORIZATION: ("work_authorization",), State.SALARY: ("salary",)}
        while self.state in fields and not self.active_correction:
            pending = {key: self.early_facts.pop(key) for key in fields[self.state] if key in self.early_facts}
            if not pending:
                break
            previous = self.state
            response = self._apply_facts(
                {key: value["fact"] for key, value in pending.items()}, off_script,
                next(iter(pending.values()))["extraction"],
            )
            if self.state == previous:
                break
        return response

    def save(self) -> None:
        from .sessions import summary

        snapshot = {key: getattr(self, key) for key in (
            "role", "history", "record", "current_question", "pending_role_mismatch",
            "pending_availability", "pending_corrections", "active_correction", "early_facts",
        )}
        snapshot.update(version=1, state=self.state.value, reference_date=self.reference_date.isoformat())
        self.persistence.save(snapshot)
        target = self.persistence.directory / "summary.md"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(summary(self), encoding="utf-8")
        temporary.replace(target)

    @staticmethod
    def _supported_fact_indices(
        extraction: dict[str, Any], verification: dict[str, Any]
    ) -> set[int]:
        fact_count = len(extraction.get("facts", []))
        decisions: dict[int, str] = {}
        valid = True
        for item in verification.get("facts", []):
            index = item.get("fact_index")
            verdict = item.get("verdict")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= fact_count
                or index in decisions
                or verdict not in {"supported", "rejected"}
            ):
                valid = False
                continue
            decisions[index] = verdict
        complete = valid and set(decisions) == set(range(fact_count))
        if not complete:
            return set()
        return {index for index, verdict in decisions.items() if verdict == "supported"}

    def _grounded_facts(
        self, extraction: dict[str, Any], message: str, verified_indices: set[int]
    ) -> dict[str, dict[str, Any]]:
        grounded = {}
        duplicates = set()
        for index, fact in enumerate(extraction.get("facts", [])):
            if index not in verified_indices:
                continue
            evidence = fact.get("evidence")
            field = fact.get("field")
            if (
                isinstance(field, str)
                and isinstance(evidence, str)
                and evidence
                and evidence in message
                and self._value_matches_evidence(field, fact.get("value"), evidence)
            ):
                if field in grounded:
                    duplicates.add(field)
                else:
                    grounded[field] = fact
        for field in duplicates:
            grounded.pop(field)
        return grounded

    @staticmethod
    def _value_matches_evidence(field: str, value: Any, evidence: str) -> bool:
        if field in {
            "role_confirmation",
            "role_mismatch_confirmation",
            "availability_confirmation",
            "work_authorization",
            "salary_alignment",
            "record_confirmation",
        }:
            return isinstance(value, bool)
        if field == "experience_years":
            numbers = _numbers_in_evidence(evidence)
            return (
                isinstance(value, int)
                and not isinstance(value, bool)
                and (not numbers or value in numbers)
            )
        if field == "skills":
            return isinstance(value, list) and all(
                isinstance(skill, str) and skill.casefold() in evidence.casefold() for skill in value
            )
        if field == "record_correction":
            return isinstance(value, list) and bool(value) and all(
                isinstance(target, str) and target in CORRECTION_STATES for target in value
            )
        if field == "availability":
            return isinstance(value, str) and value.casefold() in evidence.casefold()
        if field == "salary":
            if not isinstance(value, dict):
                return False
            numbers = _numbers_in_evidence(evidence)
            for amount in (value.get("min"), value.get("max")):
                if amount is not None and amount not in numbers:
                    return False
            currency = value.get("currency")
            if currency is None:
                return True
            markers = {
                "EUR": ("eur", "€"),
                "USD": ("usd", "$"),
                "GBP": ("gbp", "£"),
            }.get(currency.upper(), (currency.casefold(),))
            return any(marker in evidence.casefold() for marker in markers)
        return True

    def _apply_facts(
        self,
        facts: dict[str, dict[str, Any]],
        off_script: bool,
        extraction: dict[str, Any],
    ) -> str:
        if self.state == State.ROLE:
            fact = facts.get("role_confirmation")
            if not fact or not isinstance(fact["value"], bool):
                return self._clarify(
                    f"Please confirm whether you applied for the {self.role['title']} role.",
                    off_script,
                )
            if not fact["value"]:
                self.early_facts.clear()
                self.pending_role_mismatch = fact["evidence"]
                self.state = State.ROLE_MISMATCH_CONFIRMATION
                return self._next_question(off_script)
            self.record["role"].update(
                confirmed=True,
                evidence=fact["evidence"],
                confirmation_evidence=None,
            )
            self.state = State.EXPERIENCE
            return self._next_question(off_script)

        if self.state == State.ROLE_MISMATCH_CONFIRMATION:
            fact = facts.get("role_mismatch_confirmation")
            if not fact or not isinstance(fact["value"], bool):
                return self._clarify("Please answer yes or no to confirm the role mismatch.", off_script)
            if not fact["value"]:
                self.pending_role_mismatch = None
                self.state = State.ROLE
                return self._with_fallback(
                    f"Understood. Please confirm whether you applied for the {self.role['title']} role.",
                    off_script,
                )
            assert self.pending_role_mismatch
            self.record["role"].update(
                confirmed=False,
                evidence=self.pending_role_mismatch,
                confirmation_evidence=fact["evidence"],
            )
            self.pending_role_mismatch = None
            self.state = State.STOPPED
            return "It looks like the role does not match. Please contact the recruitment team so they can correct it."

        if self.state == State.EXPERIENCE:
            years = facts.get("experience_years")
            skills = facts.get("skills")
            valid_years = (
                years
                and isinstance(years["value"], int)
                and not isinstance(years["value"], bool)
                and years["value"] >= 0
            )
            valid_skills = (
                skills
                and isinstance(skills["value"], list)
                and bool(skills["value"])
                and all(isinstance(skill, str) and skill.strip() for skill in skills["value"])
            )
            experience = self.record["experience"]
            if self.active_correction == "experience_years":
                if not valid_years:
                    return self._clarify(CORRECTION_QUESTIONS["experience_years"], off_script)
                experience["years"] = years["value"]
                experience["evidence"]["years"] = years["evidence"]
                return self._complete_correction(off_script)
            if self.active_correction == "skills":
                if not valid_skills:
                    return self._clarify(CORRECTION_QUESTIONS["skills"], off_script)
                experience["skills"] = _unique_case_insensitive(skills["value"])
                experience["evidence"]["skills"] = [skills["evidence"]]
                return self._complete_correction(off_script)
            if valid_years:
                experience["years"] = years["value"]
                experience["evidence"]["years"] = years["evidence"]
            if valid_skills:
                known_skills = {skill.casefold() for skill in experience["skills"]}
                new_skills = [
                    skill for skill in skills["value"] if skill.casefold() not in known_skills
                ]
                experience["skills"].extend(_unique_case_insensitive(new_skills))
                if new_skills and skills["evidence"] not in experience["evidence"]["skills"]:
                    experience["evidence"]["skills"].append(skills["evidence"])
            if experience["years"] is None and not experience["skills"]:
                return self._clarify(
                    "Please state both your total years of relevant professional experience and "
                    "the skills you have used.",
                    off_script,
                )
            if experience["years"] is None:
                return self._clarify(
                    "Please state your total years of relevant professional experience.",
                    off_script,
                )
            if not experience["skills"]:
                return self._clarify("Please state the relevant skills you have used.", off_script)
            self.state = State.AVAILABILITY
            return self._next_question(off_script)

        if self.state == State.AVAILABILITY:
            fact = facts.get("availability")
            if not fact or not isinstance(fact["value"], str):
                return self._clarify("Please give a clear date or relative date for when you can start.", off_script)
            normalized_value = extraction.get("availability_date")
            relative = extraction.get("availability_is_relative")
            try:
                normalized_date = date.fromisoformat(normalized_value)
            except (TypeError, ValueError):
                return self._clarify("Please give a date I can interpret unambiguously, such as 2026-10-01 or next Monday.", off_script)
            if not isinstance(relative, bool):
                return self._clarify("Please give a clear date or relative date for when you can start.", off_script)
            if not relative:
                try:
                    stated_date = date.fromisoformat(fact["value"])
                except ValueError:
                    stated_date = None
                if stated_date and stated_date != normalized_date:
                    return self._clarify(
                        "Please give a date I can interpret unambiguously, such as 2026-10-01.",
                        off_script,
                    )
            if relative:
                self.pending_availability = (
                    normalized_date.isoformat(),
                    fact["evidence"],
                    self.reference_date.isoformat(),
                )
                self.state = State.AVAILABILITY_CONFIRMATION
                return self._with_fallback(
                    f"I interpreted that as {normalized_date.isoformat()}. Is that correct?", off_script
                )
            self.record["availability"].update(
                date=normalized_date.isoformat(),
                evidence=fact["evidence"],
                interpretation={
                    "method": "exact_date",
                    "reference_date": None,
                    "confirmation_evidence": None,
                },
            )
            if self.active_correction == "availability":
                return self._complete_correction(off_script)
            self.state = State.AUTHORIZATION
            return self._next_question(off_script)

        if self.state == State.AVAILABILITY_CONFIRMATION:
            fact = facts.get("availability_confirmation")
            if not fact or not isinstance(fact["value"], bool):
                return self._clarify("Please answer yes or no to confirm the interpreted date.", off_script)
            if not fact["value"]:
                self.pending_availability = None
                self.state = State.AVAILABILITY
                return self._with_fallback("Understood. What exact date would you be available?", off_script)
            assert self.pending_availability
            self.record["availability"].update(
                date=self.pending_availability[0],
                evidence=self.pending_availability[1],
                interpretation={
                    "method": "relative_date",
                    "reference_date": self.pending_availability[2],
                    "confirmation_evidence": fact["evidence"],
                },
            )
            self.pending_availability = None
            if self.active_correction == "availability":
                return self._complete_correction(off_script)
            self.state = State.AUTHORIZATION
            return self._next_question(off_script)

        if self.state == State.AUTHORIZATION:
            fact = facts.get("work_authorization")
            if not fact or not isinstance(fact["value"], bool):
                return self._clarify(f"Please confirm whether you are authorized to work in {self.role['location']}.", off_script)
            self.record["work_authorization"].update(authorized=fact["value"], evidence=fact["evidence"])
            if self.active_correction == "work_authorization":
                return self._complete_correction(off_script)
            self.state = State.SALARY
            return self._next_question(off_script)

        if self.state == State.SALARY:
            fact = facts.get("salary")
            salary = fact and fact["value"]
            if not fact or not self._valid_salary(salary):
                return self._clarify("Please state a salary amount or range and its currency, or explicitly say you are open.", off_script)
            if salary["currency"] and salary["currency"].upper() != self.role["salary_range"]["currency"].upper():
                return self._clarify(
                    f"Please state your expectation in {self.role['salary_range']['currency']} so I can compare it with the role range.",
                    off_script,
                )
            normalized_salary = dict(salary)
            if normalized_salary["currency"]:
                normalized_salary["currency"] = normalized_salary["currency"].upper()
            self.record["salary_expectation"].update(
                **normalized_salary, evidence=fact["evidence"]
            )
            self.record.pop("salary_alignment", None)
            lowest_acceptable = salary["min"]
            if lowest_acceptable is not None and lowest_acceptable > self.role["salary_range"]["max"]:
                self.record["salary_alignment"] = {"accepted_below_expectation": None, "evidence": None}
                self.state = State.SALARY_ALIGNMENT
                maximum = self.role["salary_range"]
                return self._with_fallback(
                    f"The maximum for this role is {maximum['max']} {maximum['currency']}. "
                    "Would you still like to continue?",
                    off_script,
                )
            if self.active_correction == "salary":
                return self._complete_correction(off_script)
            self.state = State.REVIEW_CONFIRMATION
            return self._with_fallback(self._review_question(), off_script)

        if self.state == State.SALARY_ALIGNMENT:
            fact = facts.get("salary_alignment")
            if not fact or not isinstance(fact["value"], bool):
                return self._clarify("Please confirm whether you want to continue at the role's maximum salary.", off_script)
            self.record["salary_alignment"].update(
                accepted_below_expectation=fact["value"], evidence=fact["evidence"]
            )
            if not fact["value"]:
                self.state = State.STOPPED
                return self._with_fallback(
                    "Understood. Since the role's maximum does not meet your expectation, "
                    "the intake will stop here.",
                    off_script,
                )
            if self.active_correction == "salary":
                return self._complete_correction(off_script)
            self.state = State.REVIEW_CONFIRMATION
            return self._with_fallback(self._review_question(), off_script)

        if self.state == State.REVIEW_CORRECTION:
            targets = self._correction_targets(facts.get("record_correction"))
            if not targets:
                return self._clarify(self._correction_selection_question(), off_script)
            return self._start_corrections(targets, off_script)

        targets = self._correction_targets(facts.get("record_correction"))
        if targets:
            return self._start_corrections(targets, off_script)

        fact = facts.get("record_confirmation")
        if not fact or not isinstance(fact["value"], bool):
            return self._clarify("Please confirm whether the summary is correct.", off_script)
        self.record["candidate_confirmation"].update(
            confirmed=fact["value"], evidence=fact["evidence"]
        )
        if fact["value"]:
            self.state = State.COMPLETE
            return self._with_fallback("Thank you. The intake is complete.", off_script)
        self.state = State.REVIEW_CORRECTION
        return self._with_fallback(self._correction_selection_question(), off_script)

    def _valid_salary(self, salary: Any) -> bool:
        if not isinstance(salary, dict) or set(salary) != {"min", "max", "currency"}:
            return False
        low, high, currency = salary["min"], salary["max"], salary["currency"]
        if low is None and high is None:
            return currency is None
        if not isinstance(currency, str) or not currency.strip():
            return False
        for amount in (low, high):
            if amount is not None and (
                not isinstance(amount, (int, float)) or isinstance(amount, bool) or amount <= 0
            ):
                return False
        return low is None or high is None or low <= high

    @staticmethod
    def _correction_targets(fact: dict[str, Any] | None) -> list[str]:
        if not fact or not isinstance(fact.get("value"), list):
            return []
        return list(dict.fromkeys(
            target for target in fact["value"] if target in CORRECTION_STATES
        ))

    def _start_corrections(self, targets: list[str], off_script: bool) -> str:
        self.record["candidate_confirmation"].update(confirmed=None, evidence=None)
        self.active_correction = targets[0]
        self.pending_corrections = targets[1:]
        self.state = CORRECTION_STATES[self.active_correction]
        question = self._correction_question(self.active_correction)
        return self._with_fallback(question, off_script)

    def _complete_correction(self, off_script: bool) -> str:
        if self.pending_corrections:
            self.active_correction = self.pending_corrections.pop(0)
            self.state = CORRECTION_STATES[self.active_correction]
            question = self._correction_question(self.active_correction)
            return self._with_fallback(question, off_script)
        self.active_correction = None
        self.state = State.REVIEW_CONFIRMATION
        return self._with_fallback(self._review_question(), off_script)

    @staticmethod
    def _correction_selection_question() -> str:
        return (
            "Which details should be corrected: experience years, skills, availability, "
            "work authorization, or salary?"
        )

    def _next_question(self, off_script: bool) -> str:
        return self._with_fallback(self._question_for_state(), off_script)

    def _question_for_state(self) -> str:
        if self.state == State.SALARY:
            return f"{self._salary_range_text()} What is your annual salary expectation?"
        return QUESTIONS[self.state].format(**self.role)

    def _correction_question(self, target: str) -> str:
        if target == "salary":
            return (
                f"{self._salary_range_text()} Please state your correct annual salary expectation."
            )
        return CORRECTION_QUESTIONS[target].format(**self.role)

    def _salary_range_text(self) -> str:
        salary = self.role["salary_range"]
        return (
            f"The annual salary range for this role is {salary['min']} to {salary['max']} "
            f"{salary['currency']}."
        )

    def _review_question(self) -> str:
        experience = self.record["experience"]
        availability = self.record["availability"]["date"]
        authorization = self.record["work_authorization"]["authorized"]
        salary = self.record["salary_expectation"]
        if salary["min"] is None and salary["max"] is None:
            salary_text = "open"
        elif salary["max"] is None:
            salary_text = f"at least {salary['min']}"
        elif salary["min"] is None:
            salary_text = f"up to {salary['max']}"
        elif salary["min"] == salary["max"]:
            salary_text = str(salary["min"])
        else:
            salary_text = f"{salary['min']} to {salary['max']}"
        if salary["currency"]:
            salary_text = f"{salary_text} {salary['currency']}"
        return (
            f"Please review: {experience['years']} years of experience; skills: "
            f"{', '.join(experience['skills'])}; available {availability}; "
            f"authorized to work in {self.role['location']}: {'yes' if authorization else 'no'}; "
            f"salary expectation: {salary_text}. Is this correct?"
        )

    def _extraction_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "current_question": self.current_question,
            "reference_date": self.reference_date.isoformat(),
            "role_location": self.role["location"],
            "salary_currency": self.role["salary_range"]["currency"],
        }
        if self.pending_availability:
            context["pending_availability"] = {
                "date": self.pending_availability[0],
                "evidence": self.pending_availability[1],
                "reference_date": self.pending_availability[2],
            }
        if self.pending_role_mismatch:
            context["pending_role_mismatch"] = {"evidence": self.pending_role_mismatch}
        if self.active_correction:
            context["active_correction"] = self.active_correction
        return context

    def _help_prompt(self) -> str:
        prompts = {
            State.ROLE: f"Please answer whether you applied for the {self.role['title']} role.",
            State.EXPERIENCE: (
                "Please provide your total years of relevant professional experience and name "
                "the professional or technical skills you used."
            ),
            State.AVAILABILITY: (
                "Please provide the earliest date you could start, for example 2026-10-01 or next Monday."
            ),
            State.AUTHORIZATION: (
                f"Please answer whether you currently have legal permission to work in {self.role['location']}."
            ),
            State.SALARY: (
                f"{self._salary_range_text()} "
                "Please provide your annual expectation, or say explicitly that you are open."
            ),
            State.REVIEW_CORRECTION: self._correction_selection_question(),
        }
        return prompts.get(self.state, self.current_question or "Please answer the current question.")

    def _clarify(self, prompt: str, off_script: bool) -> str:
        return self._with_fallback(prompt, off_script)

    @staticmethod
    def _with_fallback(response: str, off_script: bool) -> str:
        if off_script:
            return f"I cannot answer that within this intake. {response}"
        return response


def _numbers_in_evidence(evidence: str) -> set[int | float]:
    numbers: set[int | float] = set()
    for match in re.finditer(r"(?<!\w)(\d[\d,]*(?:\.\d+)?)\s*([kK])?", evidence):
        number = float(match.group(1).replace(",", ""))
        if match.group(2):
            number *= 1000
        numbers.add(int(number) if number.is_integer() else number)
    return numbers


def _unique_case_insensitive(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def load_role(path: Path, role_id: str) -> dict[str, Any]:
    roles = json.loads(path.read_text())
    for role in roles:
        if role["id"] == role_id:
            return role
    raise ValueError(f"Unknown role: {role_id}")
