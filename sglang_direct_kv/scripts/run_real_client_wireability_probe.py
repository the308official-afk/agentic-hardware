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
    hermes_agent_command,
    marker,
    nemo_agent_toolkit_command,
    opencode_command,
    openclaw_command,
    pi_agent_harness_command,
    qwen_command,
    qwen_workspace_path,
)


CLIENTS = (
    "codex",
    "claude_code",
    "opencode",
    "qwen_code",
    "pi_agent_harness",
    "openclaw",
    "nemo_agent_toolkit",
    "hermes_agent",
)


class FakeSGLangHandler(BaseHTTPRequestHandler):
    server_version = "FakeSGLangRealClientProbe/0.1"

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
            return self._send_json(200, {"model_path": "fake-real-client-probe-model", "is_generation": True})
        return self._send_json(200, {"ok": True})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0") or 0)
        if length:
            self.rfile.read(length)
        data = (
            'data: {"choices":[{"delta":{"content":"The failing test expects a guard for zero divisors."}}]}\n\n'
            'data: {"choices":[{"delta":{}}]}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def free_port() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSGLangHandler)
    port = int(server.server_address[1])
    server.server_close()
    return port


def run_fake_sglang() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSGLangHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{int(server.server_address[1])}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def wait_for_url(url: str, timeout_secs: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}")


def write_fixture(root: Path) -> Path:
    fixture = root / "buggy_math"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "math_tools.py").write_text(
        "\n".join(
            [
                "def safe_ratio(numerator: float, denominator: float) -> float:",
                "    return numerator / denominator",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (fixture / "test_math_tools.py").write_text(
        "\n".join(
            [
                "from math_tools import safe_ratio",
                "",
                "",
                "def test_safe_ratio_zero_denominator():",
                "    assert safe_ratio(10, 0) == 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (fixture / "README.md").write_text(
        "Tiny coding task fixture. The test documents the desired behavior for a zero denominator.\n",
        encoding="utf-8",
    )
    return fixture


def realistic_prompt(fixture: Path) -> str:
    return "\n".join(
        [
            "You are running a coding-agent wireability probe against a local model gateway.",
            "Do not edit files. Inspect the task below and answer with the smallest code change needed.",
            "",
            f"Repository path: {fixture}",
            "",
            "Files:",
            "math_tools.py:",
            "def safe_ratio(numerator: float, denominator: float) -> float:",
            "    return numerator / denominator",
            "",
            "test_math_tools.py:",
            "from math_tools import safe_ratio",
            "",
            "def test_safe_ratio_zero_denominator():",
            "    assert safe_ratio(10, 0) == 0",
            "",
            "Question: why does the test fail, and what is the smallest fix?",
        ]
    )


def command_for_client(
    client: str,
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str], Path]:
    if client == "codex":
        cmd, env = codex_command(gateway_base, model, prompt, meta)
        return cmd, env, Path("/tmp")
    if client == "claude_code":
        cmd, env = claude_command(gateway_base, prompt, meta)
        return cmd, env, Path("/tmp")
    if client == "opencode":
        cmd, env = opencode_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, Path("/tmp")
    if client == "qwen_code":
        cmd, env = qwen_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, qwen_workspace_path(log_dir, meta)
    if client == "pi_agent_harness":
        cmd, env = pi_agent_harness_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, Path("/tmp")
    if client == "openclaw":
        cmd, env = openclaw_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, Path("/tmp")
    if client == "nemo_agent_toolkit":
        cmd, env = nemo_agent_toolkit_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, Path("/tmp")
    if client == "hermes_agent":
        cmd, env = hermes_agent_command(gateway_base, model, prompt, meta, log_dir)
        return cmd, env, Path("/tmp")
    raise ValueError(f"unsupported real client probe: {client}")


def has_client_request(trace_path: Path, client: str, label: str) -> bool:
    return any(
        row.get("event") == "m27.request.end"
        and row.get("harness") == client
        and row.get("label") == label
        for row in read_jsonl(trace_path)
    )


def run_client(
    client: str,
    gateway_base: str,
    model: str,
    fixture: Path,
    trace_path: Path,
    log_dir: Path,
    timeout_secs: float,
) -> dict[str, Any]:
    label = f"{client}_real_client_probe"
    meta = {
        "harness": client,
        "mode": "e2e_priority_hints",
        "pressure_level": "p0_control",
        "session_id": f"{client}_real_client_session",
        "phase": "replay",
        "label": label,
        "task_index": "0",
        "prompt_hash": f"{client}_real_client_probe",
        "priority_label": "high",
        "deadline_offset_ms": 0,
        "high_priority": 100,
        "low_priority": -100,
        "max_tokens": 24,
    }
    prompt = realistic_prompt(fixture)
    client_log_dir = log_dir / client
    client_log_dir.mkdir(parents=True, exist_ok=True)
    cmd, extra_env, cwd = command_for_client(client, gateway_base, model, prompt, meta, client_log_dir)
    env = {**os.environ, **extra_env}
    log_path = client_log_dir / "client.log"
    start = time.monotonic()
    status = "timeout"
    error = ""
    return_code: int | None = None
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    deadline = time.monotonic() + timeout_secs
    try:
        while time.monotonic() < deadline:
            if has_client_request(trace_path, client, label):
                status = "gateway_completed"
                break
            if proc.poll() is not None:
                return_code = proc.returncode
                status = "client_exited"
                break
            time.sleep(0.25)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        return_code = proc.returncode
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        status = "error"
    captured = has_client_request(trace_path, client, label)
    if captured:
        status = "gateway_completed"
    return {
        "client": client,
        "label": label,
        "status": status,
        "captured": captured,
        "return_code": return_code,
        "elapsed_ms": round((time.monotonic() - start) * 1000.0, 3),
        "log_path": str(log_path),
        "error": error,
    }


def summarize_client(client: str, label: str, trace_rows: list[dict[str, Any]], gateway_rows: list[dict[str, Any]]) -> dict[str, Any]:
    end_rows = [
        row
        for row in trace_rows
        if row.get("event") == "m27.request.end"
        and row.get("harness") == client
        and row.get("label") == label
    ]
    forwarded_rows = [
        row
        for row in gateway_rows
        if row.get("event") == "gateway.forwarded_request"
        and row.get("harness") == client
        and row.get("label") == label
    ]
    source = forwarded_rows[-1] if forwarded_rows else (end_rows[-1] if end_rows else {})
    shape = source
    return {
        "client": client,
        "captured": bool(end_rows),
        "api_kind": source.get("api_kind", ""),
        "path": source.get("path", ""),
        "model_requested": shape.get("model_requested", ""),
        "body_size_bytes": shape.get("body_size_bytes", ""),
        "prompt_chars": shape.get("prompt_chars", ""),
        "stream_requested": shape.get("stream_requested", ""),
        "sglang_priority": source.get("sglang_priority", ""),
        "ttft_ms": source.get("ttft_ms", ""),
        "total_latency_ms": source.get("total_latency_ms", ""),
        "status": source.get("status", ""),
        "error": source.get("error", ""),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def render_html(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    columns = [
        "client",
        "captured",
        "api_kind",
        "path",
        "body_size_bytes",
        "prompt_chars",
        "stream_requested",
        "sglang_priority",
        "ttft_ms",
        "total_latency_ms",
        "status",
        "error",
    ]
    rows = []
    for row in report["summaries"]:
        rows.append("<tr>" + "".join(f"<td>{cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    run_rows = []
    for row in report["client_runs"]:
        run_rows.append(
            "<tr>"
            + "".join(f"<td>{cell(row.get(column, ''))}</td>" for column in ["client", "status", "captured", "return_code", "elapsed_ms", "log_path", "error"])
            + "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Real Client Wireability Probe</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #f8fafc; }}
main {{ max-width: 1400px; margin: 0 auto; padding: 32px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
h2 {{ margin-top: 28px; font-size: 21px; }}
p {{ color: #334155; line-height: 1.5; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-top: 16px; overflow-x: auto; }}
.note {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 16px; color: #1e3a8a; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
th {{ background: #f1f5f9; }}
code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
<h1>Real Client Wireability Probe</h1>
<p>Report label: <code>{cell(report["report_label"])}</code>. Backend: <code>{cell(report["target_base"])}</code>.</p>
<p class="note">This probe launches real client CLIs against the inspection / priority gateway and records the request shape the gateway observes. Prompt bodies are not written to the report; sizes and hashes are used instead.</p>
<h2>Observed Gateway Traffic</h2>
<div class="card"><table><thead><tr>{''.join(f'<th>{column}</th>' for column in columns)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Client Runs</h2>
<div class="card"><table><thead><tr>{''.join(f'<th>{column}</th>' for column in ["client", "status", "captured", "return_code", "elapsed_ms", "log_path", "error"])}</tr></thead><tbody>{''.join(run_rows)}</tbody></table></div>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe real client CLI traffic through the SGLang inspection gateway.")
    parser.add_argument("--clients", nargs="+", choices=CLIENTS, default=list(CLIENTS))
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--target-base", help="Existing SGLang base URL. If omitted, a fake local SGLang server is used.")
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--timeout-secs", type=float, default=180)
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="real_client_wireability_"))
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = write_fixture(out_dir / "fixtures")
    fake_server = None
    target_base = args.target_base
    if not target_base:
        fake_server, target_base = run_fake_sglang()

    gateway_port = args.gateway_port or free_port()
    gateway_base = f"http://{args.gateway_host}:{gateway_port}"
    trace = out_dir / "m27_trace.jsonl"
    gateway_events = out_dir / "gateway_events.jsonl"
    gateway_stdout = out_dir / "gateway.log"
    gateway_cmd = [
        args.python_bin,
        str(root / "scripts" / "harness_sglang_gateway.py"),
        "--listen-host",
        args.gateway_host,
        "--listen-port",
        str(gateway_port),
        "--target-base",
        target_base,
        "--trace",
        str(trace),
        "--log",
        str(gateway_events),
        "--model",
        args.model,
    ]

    print(f"probe: starting gateway {gateway_base} -> {target_base}")
    with gateway_stdout.open("w", encoding="utf-8") as handle:
        gateway_proc = subprocess.Popen(gateway_cmd, cwd=str(root), stdout=handle, stderr=subprocess.STDOUT, text=True)
    client_runs: list[dict[str, Any]] = []
    try:
        wait_for_url(f"{gateway_base}/health", timeout_secs=30)
        for client in args.clients:
            print(f"probe: running real client {client}")
            run = run_client(client, gateway_base, args.model, fixture, trace, out_dir / "client_logs", args.timeout_secs)
            client_runs.append(run)
            print(f"probe: {client} status={run['status']} captured={run['captured']} elapsed_ms={run['elapsed_ms']}")
    finally:
        gateway_proc.terminate()
        try:
            gateway_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gateway_proc.kill()
            gateway_proc.wait(timeout=5)
        if fake_server is not None:
            fake_server.shutdown()
            fake_server.server_close()

    trace_rows = read_jsonl(trace)
    gateway_rows = read_jsonl(gateway_events)
    summaries = [summarize_client(str(run["client"]), str(run["label"]), trace_rows, gateway_rows) for run in client_runs]
    report = {
        "report_label": out_dir.name,
        "target_base": target_base,
        "gateway_base": gateway_base,
        "fixture": str(fixture),
        "client_runs": client_runs,
        "summaries": summaries,
    }
    write_json(out_dir / "real_client_wireability_report.json", report)
    write_jsonl(out_dir / "real_client_wireability_summary.jsonl", summaries)
    (out_dir / "real_client_wireability_report.html").write_text(render_html(report), encoding="utf-8")
    print(f"probe: wrote {out_dir / 'real_client_wireability_report.html'}")
    if not all(row.get("captured") for row in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
