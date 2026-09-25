from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conversation import ConversationError, create_conversation
from .extractor import create_model_clients
from .screening import load_role
from .sessions import open_session


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Recruitment Intake Assistant")
    parser.add_argument("--role", default="software-engineer", help="role ID from data/input/roles.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/output")
    parser.add_argument("--provider", choices=("openai", "azure"), required=True)
    parser.add_argument("--verifier", choices=("provider", "jev"), default="provider")
    parser.add_argument("--mode", choices=("text", "voice", "web"), default="text")
    parser.add_argument("--resume", help="resume a session ID from the output directory")
    parser.add_argument("--port", type=int, default=8000, help="local browser interface port")
    args = parser.parse_args()

    try:
        role = load_role(PROJECT_ROOT / "data/input/roles.json", args.role)
        schema = json.loads((PROJECT_ROOT / "data/input/schema.json").read_text())
        extractor, verifier = create_model_clients(args.provider, args.verifier)
        identifier, controller = open_session(args.output, role, schema, extractor, verifier, args.resume)
        conversation = create_conversation(args.mode) if args.mode != "web" else None
    except ValueError as error:
        parser.error(str(error))
    controller.provider_error_handler = lambda error: print(f"Provider error: {error}", file=sys.stderr)
    print(f"Session: {identifier}\nOutput: {controller.persistence.directory}")
    if args.mode == "web":
        from .web import serve
        serve(controller, args.port)
        return
    try:
        conversation.send_agent(controller.opening())
    except ConversationError as error:
        _report_conversation_error(error)
    while controller.state.value not in ("complete", "stopped"):
        try:
            message = conversation.read_candidate()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        except ConversationError as error:
            _report_conversation_error(error)
            continue
        response = controller.handle(message)
        try:
            conversation.send_agent(response)
        except ConversationError as error:
            _report_conversation_error(error)


def _report_conversation_error(error: ConversationError) -> None:
    print(f"Conversation error: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
