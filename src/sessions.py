"""Local session creation, restoration, and recruiter handoff."""

import json
import re
from datetime import date
from pathlib import Path
from uuid import uuid4

from .screening import JsonPersistence, ScreeningController, State


def open_session(root, role, schema, extractor, verifier, session_id=None):
    if session_id is not None and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", session_id):
        raise ValueError("Invalid session ID")
    identifier = session_id or uuid4().hex
    directory = Path(root) / identifier
    snapshot = None
    if session_id:
        try:
            snapshot = json.loads((directory / "session.json").read_text())
        except FileNotFoundError as error:
            raise ValueError("Session not found") from error
        if snapshot.get("version") != 1:
            raise ValueError("Unsupported session version")
        role = snapshot["role"]
    controller = ScreeningController(role, extractor, verifier, JsonPersistence(directory, schema))
    if snapshot:
        controller.persistence.validator.validate(snapshot["record"])
        for key in ("history", "record", "current_question", "pending_role_mismatch",
                    "pending_availability", "pending_corrections", "active_correction", "early_facts"):
            setattr(controller, key, snapshot[key])
        controller.state = State(snapshot["state"])
        controller.reference_date = date.fromisoformat(snapshot["reference_date"])
        controller.save()
    return identifier, controller


def summary(controller):
    record = controller.record
    lines = ["# Recruitment Intake Assistant", "", "## Recruiter Summary", "",
             f"**Role:** {_markdown(controller.role['title'])}", "",
             f"**Status:** {controller.state.value}", "",
             f"**Candidate confirmed summary:** {_display(record['candidate_confirmation']['confirmed'])}", ""]
    experience = record["experience"]
    salary = record["salary_expectation"]
    salary_text = ("Open to discussion" if salary["evidence"] else "Not provided")
    if salary["min"] is not None or salary["max"] is not None:
        salary_text = f"Minimum: {_display(salary['min'])}; maximum: {_display(salary['max'])}; currency: {salary['currency']}"
    items = [
        ("Applied for role", _display(record["role"]["confirmed"]), record["role"]["evidence"]),
        ("Experience in years", _display(experience["years"]), experience["evidence"]["years"]),
        ("Skills", ", ".join(experience["skills"]) or "Not provided", experience["evidence"]["skills"]),
        ("Available from", _display(record["availability"]["date"]), record["availability"]["evidence"]),
        (f"Authorized to work in {controller.role['location']}", _display(record["work_authorization"]["authorized"]), record["work_authorization"]["evidence"]),
        ("Annual salary expectation", salary_text, salary["evidence"]),
    ]
    if "salary_alignment" in record:
        alignment = record["salary_alignment"]
        items.append(("Willing to continue at role maximum", _display(alignment["accepted_below_expectation"]), alignment["evidence"]))
    items.append(("Final confirmation", _display(record["candidate_confirmation"]["confirmed"]), record["candidate_confirmation"]["evidence"]))
    for label, value, evidence in items:
        lines.extend([f"## {_markdown(label)}", "", _markdown(value), ""])
        for quote in (evidence if isinstance(evidence, list) else [evidence]):
            if quote:
                lines.append(f"> Candidate replied: {_markdown(json.dumps(quote, ensure_ascii=False))}")
        lines.append("")
    lines.append("This record captures stated information; it does not assess candidate suitability.")
    return "\n".join(lines) + "\n"


def _markdown(value):
    return re.sub(r"([\\`*_{}\[\]<>()#+.!|~-])", r"\\\1", value)


def _display(value):
    if value is None:
        return "Not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)
