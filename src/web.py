"""Single-session browser interface, available only on the local computer."""

import json
import re
import secrets
import sys
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .sessions import summary


def render_summary(markdown):
    """Render the headings, labels, and quotes emitted by sessions.summary, not arbitrary HTML."""
    def text(value):
        return escape(re.sub(r"\\([\\`*_{}\[\]<>()#+.!|~-])", r"\1", value))

    blocks = []
    for line in markdown.splitlines():
        if not line:
            continue
        if line.startswith("# "):
            blocks.append(f"<h2>{text(line[2:])}</h2>")
        elif line.startswith("## "):
            blocks.append(f"<h3>{text(line[3:])}</h3>")
        elif line.startswith("> "):
            blocks.append(f"<blockquote>{text(line[2:])}</blockquote>")
        elif match := re.fullmatch(r"\*\*(.+?):\*\* (.*)", line):
            blocks.append(f"<p><strong>{text(match[1])}:</strong> {text(match[2])}</p>")
        else:
            blocks.append(f"<p>{text(line)}</p>")
    return "\n".join(blocks)


def make_handler(controller, token):
    class Handler(BaseHTTPRequestHandler):
        def session_data(self):
            markdown = summary(controller)
            return json.dumps({"history": controller.history, "state": controller.state.value,
                               "summary": markdown, "summary_html": render_summary(markdown)})

        def reply(self, status, content, content_type="application/json"):
            payload = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            return secrets.compare_digest(self.headers.get("X-Session-Token", ""), token)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, Path(__file__).with_name("web.html").read_text(), "text/html")
            elif self.path == "/session" and self.authorized():
                self.reply(200, self.session_data())
            else:
                self.reply(404, '{}')

        def do_POST(self):
            if self.path != "/message" or not self.authorized():
                self.reply(403, '{}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32000:
                    raise ValueError("Message too large or empty")
                message = json.loads(self.rfile.read(length))["message"]
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("Enter a message")
            except (ValueError, KeyError, TypeError):
                self.reply(400, json.dumps({"error": "Enter a message under 32 KB."}))
                return
            controller.handle(message)
            self.reply(200, self.session_data())

        def log_message(self, *_args):
            pass

    return Handler


def serve(controller, port):
    controller.opening()
    token = secrets.token_urlsafe(32)
    with HTTPServer(("127.0.0.1", port), make_handler(controller, token)) as server:
        url = f"http://127.0.0.1:{server.server_port}/#{token}"
        link = f"\033]8;;{url}\033\\{url}\033]8;;\033\\" if sys.stdout.isatty() else url
        print(f"Open {link}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
