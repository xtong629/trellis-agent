"""Shared fixtures. ``fake_openai`` is a real HTTP server on loopback speaking the chat-completions
wire format, so client tests exercise the true SDK code path without touching the internet."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest


class FakeOpenAI:
    def __init__(self) -> None:
        self.queue: deque[tuple[int, Any]] = deque()
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output quiet
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                fake.requests.append(json.loads(self.rfile.read(length) or b"{}"))
                fake.headers.append({k.lower(): v for k, v in self.headers.items()})
                status, payload = fake.queue.popleft() if fake.queue else (500, {"error": "empty"})
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    # -- helpers to script replies -------------------------------------------------
    def reply(self, message: dict[str, Any], **extra: Any) -> None:
        """Queue a well-formed chat completion."""
        body: dict[str, Any] = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", **message},
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        }
        body.update(extra)
        self.queue.append((200, body))

    def raw(self, status: int, payload: Any) -> None:
        self.queue.append((status, payload))


@pytest.fixture
def fake_openai() -> Iterator[FakeOpenAI]:
    server = FakeOpenAI()
    server.start()
    try:
        yield server
    finally:
        server.stop()
