"""Optional protocol-preserving streaming proxy, independent of the testbed.

The upstream must serve the same API as the client; this is not an API
translator. Native protocol token accounting needs an injected renderer.
"""
from __future__ import annotations

import argparse
import codecs
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit

APIS = {"/v1/chat/completions": "openai_chat", "/v1/messages": "anthropic", "/v1/responses": "responses"}
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailer", "upgrade"}


def text_delta(value: dict) -> str:
    for choice in value.get("choices", []):
        text = choice.get("delta", {}).get("content")
        if isinstance(text, str) and text:
            return text
    if value.get("type") == "content_block_delta":
        return value.get("delta", {}).get("text", "")
    if value.get("type") == "response.output_text.delta":
        return value.get("delta", "")
    return ""


def make_proxy_handler(target_base, encoder=None, *, record=lambda row: None, max_encoding_workers=2,
                       timeout_seconds=120, max_body_bytes=8_000_000):
    slots = threading.BoundedSemaphore(max_encoding_workers)
    # Conservative serialization for tokenizer implementations with mutable state.
    encoding_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.forward(None)

        def do_POST(self):
            if self.headers.get("Transfer-Encoding"):
                self.send_error(411, "Content-Length required")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400)
                return
            if not 0 <= length <= max_body_bytes:
                self.send_error(413)
                return
            self.forward(self.rfile.read(length))

        def forward(self, body):
            import httpx
            received = time.time_ns()
            row = {"event": "prompt_proxy.request", "attempt_id": uuid.uuid4().hex,
                   "request_id": self.headers.get("x-request-id", ""), "received_ns": received,
                   "encoding_status": "disabled", "first_content_ns": None, "first_client_content_ns": None}
            path = urlsplit(self.path).path
            sent_headers = False
            try:
                if body is not None and encoder is not None and path in APIS:
                    if slots.acquire(blocking=False):
                        try:
                            # No queueing behind another encoding job: bypass instead.
                            if encoding_lock.acquire(blocking=False):
                                try:
                                    payload = json.loads(body)
                                    result = encoder.encode(payload, APIS[path])
                                    row.update(result.evidence())
                                    if result.status == "applied":
                                        body = json.dumps(result.payload, ensure_ascii=False).encode()
                                finally:
                                    encoding_lock.release()
                            else:
                                row.update(encoding_status="skipped", encoding_reason="encoder_busy")
                        finally:
                            slots.release()
                    else:
                        row.update(encoding_status="skipped", encoding_reason="encoder_busy")
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
                headers["accept-encoding"] = "identity"
                row["forward_started_ns"] = time.time_ns()
                with httpx.Client(timeout=timeout_seconds) as client:
                    with client.stream(self.command, target_base.rstrip("/") + self.path, content=body, headers=headers) as response:
                        row["status"] = response.status_code
                        self.send_response(response.status_code)
                        for key, value in response.headers.items():
                            if key.lower() not in HOP_HEADERS:
                                self.send_header(key, value)
                        self.end_headers()
                        sent_headers = True
                        decoder = codecs.getincrementaldecoder("utf-8")("replace")
                        pending = ""
                        is_sse = "text/event-stream" in response.headers.get("content-type", "")
                        for chunk in response.iter_raw():
                            if is_sse and row["first_content_ns"] is None:
                                pending += decoder.decode(chunk)
                                while "\n" in pending:
                                    line, pending = pending.split("\n", 1)
                                    if line.startswith("data:"):
                                        try:
                                            if text_delta(json.loads(line[5:])):
                                                row["first_content_ns"] = time.time_ns()
                                        except (ValueError, TypeError, AttributeError):
                                            pass
                                if len(pending) > max_body_bytes:
                                    pending = ""  # bound evidence parser, never truncate forwarded bytes
                            self.wfile.write(chunk)
                            self.wfile.flush()
                            if row["first_content_ns"] is not None and row["first_client_content_ns"] is None:
                                row["first_client_content_ns"] = time.time_ns()
            except Exception as exc:
                row["error"] = type(exc).__name__
                if not sent_headers:
                    self.send_error(502, "upstream request failed")
            finally:
                row["finished_ns"] = time.time_ns()
                # First client write is not proof of receipt at a remote client.
                record(row)

    return Handler


def main():
    from .config import load_encoder
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-base", required=True)
    parser.add_argument("--port", type=int, default=32080)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    encoder = load_encoder(args.config) if args.config else None
    args.events.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def record(row):
        with lock, args.events.open("a") as file:
            file.write(json.dumps(row) + "\n")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_proxy_handler(args.target_base, encoder, record=record))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
