#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from run_multi_harness_replay_driver import (
    claude_command,
    codex_command,
    deepseek_harness_command,
    nemo_agent_toolkit_command,
    opencode_command,
    qwen_command,
)


class FakeSGLangHandler(BaseHTTPRequestHandler):
    server_version = "FakeSGLangSmoke/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.startswith("/model_info"):
            return self._send_json(200, {"model_path": "fake-smoke-model", "is_generation": True})
        return self._send_json(200, {"ok": True})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0") or 0)
        if length:
            self.rfile.read(length)
        data = (
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":[{"delta":{}}]}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def free_server() -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSGLangHandler)
    return server, int(server.server_address[1])


def run_fake_sglang() -> tuple[ThreadingHTTPServer, str]:
    server, port = free_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def wait_for_gateway(url: str) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"gateway did not become ready at {url}")


def assert_harness_trace(harness: str, trace: Path, label: str | None = None) -> None:
    rows = read_jsonl(trace)
    submitted = [
        row
        for row in rows
        if row.get("event") == "m27.request.submitted"
        and row.get("harness") == harness
        and row.get("phase") == "replay"
        and (label is None or row.get("label") == label)
    ]
    if not submitted:
        raise AssertionError(f"{harness}: missing replay submission")
    priorities = {str(row.get("sglang_priority") or "") for row in submitted}
    if "100" not in priorities:
        raise AssertionError(f"{harness}: replay priority was not propagated as 100; saw {sorted(priorities)}")


def has_marked_replay(trace: Path, harness: str, label: str) -> bool:
    return any(
        row.get("event") == "m27.request.submitted"
        and row.get("harness") == harness
        and row.get("phase") == "replay"
        and row.get("label") == label
        and str(row.get("sglang_priority") or "") == "100"
        for row in read_jsonl(trace)
    )


def command_for_harness(
    harness: str,
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str], str]:
    if harness == "codex":
        cmd, env = codex_command(gateway_base, model, prompt, meta)
        return cmd, env, "/tmp"
    if harness == "claude_code":
        cmd, env = claude_command(gateway_base, prompt, meta)
        return cmd, env, "/tmp"
    if harness == "opencode":
        cmd, env = opencode_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, "/tmp"
    if harness == "qwen_code":
        cmd, env = qwen_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, str(log_dir / "qwen_workspace" / str(meta["label"]))
    if harness == "nemo_agent_toolkit":
        cmd, env = nemo_agent_toolkit_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, "/tmp"
    if harness == "deepseek_harness":
        cmd, env = deepseek_harness_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, "/tmp"
    raise ValueError(f"unsupported CLI smoke harness: {harness}")


def run_single_replay_probe(
    harness: str,
    gateway_base: str,
    model: str,
    trace: Path,
    work_dir: Path,
    timeout_secs: float,
) -> None:
    label = f"{harness}_smoke_replay"
    prompt = (
        "Smoke replay request. Return one short sentence. "
        "This request carries a hidden harness replay marker for the gateway."
    )
    meta = {
        "harness": harness,
        "mode": "e2e_priority_hints",
        "pressure_level": "p0_control",
        "session_id": f"{harness}_smoke_session",
        "phase": "replay",
        "label": label,
        "task_index": "0",
        "prompt_hash": f"{harness}_smoke_hash",
        "priority_label": "high",
        "deadline_offset_ms": 0,
        "high_priority": 100,
        "low_priority": -100,
        "max_tokens": 2,
    }
    log_dir = work_dir / f"{harness}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd, extra_env, cwd = command_for_harness(harness, gateway_base, model, prompt, meta, log_dir)
    env = {**os.environ, **extra_env}
    log_path = log_dir / f"{label}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + timeout_secs
    try:
        while time.monotonic() < deadline:
            if has_marked_replay(trace, harness, label):
                return
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        if not has_marked_replay(trace, harness, label):
            raise AssertionError(f"{harness}: no marked replay reached the gateway before timeout; see {log_path}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test multi-harness CLI wiring through the SGLang gateway.")
    parser.add_argument("--harnesses", nargs="+", default=["nemo_agent_toolkit", "deepseek_harness"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--timeout-secs", type=float, default=120)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="multi_harness_smoke_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    fake_server, fake_base = run_fake_sglang()
    gateway_server, gateway_port = free_server()
    gateway_server.server_close()
    gateway_base = f"http://127.0.0.1:{gateway_port}"
    trace = work_dir / "smoke_trace.jsonl"
    gateway_log = work_dir / "gateway_events.jsonl"
    gateway_stdout = work_dir / "gateway.log"
    gateway_cmd = [
        args.python_bin,
        str(root / "scripts" / "harness_sglang_gateway.py"),
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        str(gateway_port),
        "--target-base",
        fake_base,
        "--trace",
        str(trace),
        "--log",
        str(gateway_log),
        "--model",
        args.model,
    ]

    with gateway_stdout.open("w", encoding="utf-8") as handle:
        gateway_proc = subprocess.Popen(gateway_cmd, cwd=str(root), stdout=handle, stderr=subprocess.STDOUT, text=True)
    try:
        wait_for_gateway(gateway_base)
        for harness in args.harnesses:
            print(f"smoke: running {harness}")
            run_single_replay_probe(harness, gateway_base, args.model, trace, work_dir, args.timeout_secs)
            assert_harness_trace(harness, trace, f"{harness}_smoke_replay")
            print(f"smoke: {harness} propagated replay priority through gateway")
    finally:
        gateway_proc.terminate()
        try:
            gateway_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gateway_proc.kill()
            gateway_proc.wait(timeout=5)
        fake_server.shutdown()
        fake_server.server_close()

    print(f"smoke: ok; artifacts at {work_dir}")


if __name__ == "__main__":
    main()
