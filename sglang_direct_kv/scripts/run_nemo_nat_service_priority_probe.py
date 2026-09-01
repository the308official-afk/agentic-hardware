#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from run_multi_harness_replay_driver import marker
from run_real_prompt_controlled_replay import make_shared_prefix


def write_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts_ns", time.time_ns())
    row.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_agentic_kv(payload: dict[str, Any]) -> dict[str, Any]:
    return as_dict(as_dict(payload.get("custom_params")).get("agentic_kv"))


def chat_sse(text: str, model: str) -> bytes:
    chunk_id = f"chatcmpl_nat_probe_{int(time.time() * 1_000_000)}"
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    done = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (
        f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
        f"data: {json.dumps(done, separators=(',', ':'))}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


def make_fake_sglang_handler(trace_path: Path, backend_delay_ms: int, model: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FakeSGLangForNATServiceProbe/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(fmt % args, file=sys.stderr)

        def _send_json(self, status: int, value: dict[str, Any]) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/model_info"):
                return self._send_json(200, {"model_path": model, "is_generation": True})
            if self.path.startswith("/v1/models"):
                return self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": model, "object": "model"}],
                    },
                )
            return self._send_json(200, {"ok": True})

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("content-length", "0") or 0))
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                payload = {}
            payload_dict = as_dict(payload)
            agentic_kv = extract_agentic_kv(payload_dict)
            common = {
                "event": "m27.nat_service_probe.fake_sglang_receive",
                "request_id": agentic_kv.get("request_id", ""),
                "label": agentic_kv.get("label", agentic_kv.get("request_id", "")),
                "session_id": agentic_kv.get("session_id", ""),
                "phase": agentic_kv.get("phase", ""),
                "harness": agentic_kv.get("harness", ""),
                "mode": agentic_kv.get("mode", ""),
                "sglang_priority": agentic_kv.get("sglang_priority", ""),
                "backend_delay_ms": backend_delay_ms,
            }
            write_jsonl(trace_path, common)
            if backend_delay_ms > 0:
                time.sleep(backend_delay_ms / 1000.0)
            text = "nat-service-priority-probe-ok"
            body_out = chat_sse(text, str(payload_dict.get("model") or model))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body_out)))
            self.end_headers()
            self.wfile.write(body_out)

    return Handler


def start_fake_sglang(trace_path: Path, backend_delay_ms: int, model: str) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", free_port()), make_fake_sglang_handler(trace_path, backend_delay_ms, model))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{int(server.server_address[1])}"


def build_nat_config(
    gateway_base: str,
    model: str,
    provider: str,
    prefix_total_requests: int,
    prefix_osl: int,
    prefix_iat: int,
) -> dict[str, Any]:
    llm_config: dict[str, Any] = {
        "_type": "dynamo" if provider == "dynamo" else "openai",
        "api_key": "dummy",
        "base_url": f"{gateway_base.rstrip('/')}/v1",
        "model_name": model,
        "api_type": "chat_completion",
        "temperature": 0,
        "max_tokens": 8,
        "request_timeout": 900,
        "max_retries": 0,
    }
    if provider == "dynamo":
        llm_config.update(
            {
                "enable_nvext_hints": True,
                "nvext_prefix_total_requests": prefix_total_requests,
                "nvext_prefix_osl": prefix_osl,
                "nvext_prefix_iat": prefix_iat,
                "nvext_cache_pin_type": "ephemeral",
                "nvext_cache_control_mode": "always",
                "nvext_max_sensitivity": 1000,
            }
        )
    return {
        "general": {
            "front_end": {
                "_type": "fastapi",
                "workflow": {
                    "method": "POST",
                    "description": "Executes the NAT service-level priority probe workflow.",
                    "path": "/v1/workflow",
                    "openai_api_path": "/v1/chat",
                    "openai_api_v1_path": "/v1/chat/completions",
                },
                "disable_legacy_routes": False,
            }
        },
        "llms": {
            "harness_llm": llm_config,
        },
        "workflow": {
            "_type": "chat_completion",
            "llm_name": "harness_llm",
            "system_prompt": "You are a concise coding-agent service probe. Do not use tools.",
        },
    }


async def send_dynamo_transport_request(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    trace_path: Path,
    total_requests: int,
    osl: int,
    iat: int,
) -> dict[str, Any]:
    from nat.builder.context import Context
    from nat.llm.dynamo_llm import CacheControlMode
    from nat.llm.dynamo_llm import CachePinType
    from nat.llm.dynamo_llm import DynamoPrefixContext
    from nat.llm.dynamo_llm import _DynamoTransport

    request_id = str(meta["label"])
    priority_class = str(as_dict(meta.get("priority_intent")).get("class") or "")
    sensitivity = 100 if priority_class == "urgent" else 2
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{marker(meta)}"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
    }
    submit_ns = time.time_ns()
    write_jsonl(
        trace_path,
        {
            "event": "m27.nat_service_probe.client_submit",
            "request_id": request_id,
            "label": request_id,
            "session_id": meta.get("session_id", ""),
            "phase": meta.get("phase", ""),
            "harness": meta.get("harness", ""),
            "mode": meta.get("mode", ""),
            "priority_intent": compact_json(meta.get("priority_intent")),
            "harness_input_priority_signal": meta.get("harness_input_priority_signal", ""),
            "harness_input_priority_signal_source": meta.get("harness_input_priority_signal_source", ""),
            "dynamo_direct_latency_sensitivity": sensitivity,
        },
    )
    transport = _DynamoTransport(
        transport=httpx.AsyncHTTPTransport(),
        total_requests=total_requests,
        osl=osl,
        iat=iat,
        cache_pin_type=CachePinType.EPHEMERAL,
        cache_control_mode=CacheControlMode.ALWAYS,
        max_sensitivity=1000,
    )
    status = 0
    error = ""
    try:
        with DynamoPrefixContext.scope("nat-service-priority-probe"):
            with Context.get().push_latency_sensitivity(sensitivity):
                async with httpx.AsyncClient(transport=transport, timeout=None) as client:
                    response = await client.post(f"{gateway_base.rstrip('/')}/v1/chat/completions", json=payload)
                    status = response.status_code
                    response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    done_ns = time.time_ns()
    row = {
        "event": "m27.nat_service_probe.client_done",
        "request_id": request_id,
        "label": request_id,
        "session_id": meta.get("session_id", ""),
        "phase": meta.get("phase", ""),
        "harness": meta.get("harness", ""),
        "mode": meta.get("mode", ""),
        "status": status,
        "error": error,
        "client_latency_ms": round((done_ns - submit_ns) / 1_000_000.0, 3),
        "dynamo_direct_latency_sensitivity": sensitivity,
    }
    write_jsonl(trace_path, row)
    return row


def wait_for_http(url: str, proc: subprocess.Popen[Any] | None, timeout_s: int, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            raise RuntimeError(f"process exited while waiting for {url}; returncode={proc.returncode}\n{tail}")
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code < 500:
                return
            last_error = f"status={response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {url}; last_error={last_error}")


def start_gateway(
    script_dir: Path,
    trace_path: Path,
    log_path: Path,
    target_base: str,
    model: str,
) -> tuple[subprocess.Popen[Any], str]:
    port = free_port()
    cmd = [
        sys.executable,
        str(script_dir / "harness_sglang_gateway.py"),
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        str(port),
        "--target-base",
        target_base,
        "--trace",
        str(trace_path),
        "--log",
        str(log_path),
        "--model",
        model,
    ]
    stdout = log_path.with_suffix(".stdout.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    wait_for_http(f"{base}/health", proc, 30, log_path.with_suffix(".stdout.log"))
    return proc, base


def start_nat_server(
    nat_bin: str,
    config_path: Path,
    nat_home: Path,
    log_path: Path,
    workers: int,
) -> tuple[subprocess.Popen[Any], str]:
    port = free_port()
    cmd = [
        nat_bin,
        "serve",
        "--config_file",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
    ]
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "dummy")
    env["NAT_HOME"] = str(nat_home)
    stdout = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, text=True, cwd="/tmp", env=env)
    base = f"http://127.0.0.1:{port}"
    wait_for_http(f"{base}/docs", proc, 90, log_path)
    return proc, base


def compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


async def send_nat_request(
    nat_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    trace_path: Path,
) -> dict[str, Any]:
    request_id = str(meta["label"])
    priority_intent = as_dict(meta.get("priority_intent"))
    priority_class = str(priority_intent.get("class") or "")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{marker(meta)}"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
        "metadata": {
            "priority_class": priority_class,
            "priority_reason": str(priority_intent.get("reason") or ""),
            "priority_deadline_ms": str(priority_intent.get("deadline_ms") or ""),
        },
        "agentic_hints": {
            "priority_class": priority_class,
            "reason": str(priority_intent.get("reason") or ""),
            "deadline_ms": str(priority_intent.get("deadline_ms") or ""),
        },
    }
    submit_ns = time.time_ns()
    write_jsonl(
        trace_path,
        {
            "event": "m27.nat_service_probe.client_submit",
            "request_id": request_id,
            "label": request_id,
            "session_id": meta.get("session_id", ""),
            "phase": meta.get("phase", ""),
            "harness": meta.get("harness", ""),
            "mode": meta.get("mode", ""),
            "priority_intent": compact_json(meta.get("priority_intent")),
            "harness_input_priority_signal": meta.get("harness_input_priority_signal", ""),
            "harness_input_priority_signal_source": meta.get("harness_input_priority_signal_source", ""),
        },
    )
    status = 0
    error = ""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{nat_base.rstrip('/')}/v1/chat/completions", json=payload)
            status = response.status_code
            response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    done_ns = time.time_ns()
    row = {
        "event": "m27.nat_service_probe.client_done",
        "request_id": request_id,
        "label": request_id,
        "session_id": meta.get("session_id", ""),
        "phase": meta.get("phase", ""),
        "harness": meta.get("harness", ""),
        "mode": meta.get("mode", ""),
        "status": status,
        "error": error,
        "client_latency_ms": round((done_ns - submit_ns) / 1_000_000.0, 3),
    }
    write_jsonl(trace_path, row)
    return row


def terminate_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=8)


def build_probe_metas(low_count: int, urgent_count: int, deadline_ms: int) -> list[dict[str, Any]]:
    metas: list[dict[str, Any]] = []
    for idx in range(low_count):
        metas.append(
            {
                "harness": "nemo_agent_toolkit_service",
                "mode": "pre_harness_priority_hints",
                "phase": "service_probe_low",
                "session_id": f"nat_service_low_{idx:03d}",
                "label": f"nat_service_low_{idx:03d}",
                "task_index": str(idx),
                "tool_wait_ms": deadline_ms,
                "deadline_offset_ms": deadline_ms,
                "priority_label": "low",
                "priority_intent": {
                    "class": "background",
                    "reason": "older_background_service_work",
                    "deadline_ms": deadline_ms,
                    "source": "experiment_driver",
                },
                "harness_input_priority_signal": "metadata.priority_class=background; agentic_hints.priority_class=background",
                "harness_input_priority_signal_source": "nat_service_openai_frontend",
            }
        )
    for idx in range(urgent_count):
        metas.append(
            {
                "harness": "nemo_agent_toolkit_service",
                "mode": "pre_harness_priority_hints",
                "phase": "service_probe_urgent",
                "session_id": f"nat_service_urgent_{idx:03d}",
                "label": f"nat_service_urgent_{idx:03d}",
                "task_index": str(idx),
                "tool_wait_ms": deadline_ms,
                "deadline_offset_ms": deadline_ms,
                "priority_label": "high",
                "priority_intent": {
                    "class": "urgent",
                    "reason": "tool_replay_deadline",
                    "deadline_ms": deadline_ms,
                    "source": "experiment_driver",
                },
                "harness_input_priority_signal": "metadata.priority_class=urgent; agentic_hints.priority_class=urgent",
                "harness_input_priority_signal_source": "nat_service_openai_frontend",
            }
        )
    return metas


async def run_load(
    nat_base: str,
    model: str,
    trace_path: Path,
    low_count: int,
    urgent_count: int,
    low_lead_ms: int,
    low_stagger_ms: int,
    prompt_tokens: int,
    deadline_ms: int,
) -> list[dict[str, Any]]:
    lows = build_probe_metas(low_count, 0, deadline_ms)
    urgents = build_probe_metas(0, urgent_count, deadline_ms)
    prompt = (
        f"{make_shared_prefix('nat_service_priority_probe', prompt_tokens)}\n\n"
        "Service-level priority probe: answer with one short sentence."
    )
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    for idx, meta in enumerate(lows):
        tasks.append(asyncio.create_task(send_nat_request(nat_base, model, prompt, meta, trace_path)))
        if low_stagger_ms > 0 and idx != len(lows) - 1:
            await asyncio.sleep(low_stagger_ms / 1000.0)
    if low_lead_ms > 0:
        await asyncio.sleep(low_lead_ms / 1000.0)
    for meta in urgents:
        tasks.append(asyncio.create_task(send_nat_request(nat_base, model, prompt, meta, trace_path)))
    return await asyncio.gather(*tasks)


async def run_dynamo_direct_transport_load(
    gateway_base: str,
    model: str,
    trace_path: Path,
    low_count: int,
    urgent_count: int,
    low_lead_ms: int,
    low_stagger_ms: int,
    prompt_tokens: int,
    deadline_ms: int,
    total_requests: int,
    osl: int,
    iat: int,
) -> list[dict[str, Any]]:
    lows = build_probe_metas(low_count, 0, deadline_ms)
    urgents = build_probe_metas(0, urgent_count, deadline_ms)
    prompt = (
        f"{make_shared_prefix('nat_dynamo_direct_transport_probe', prompt_tokens)}\n\n"
        "Dynamo-provider wireability probe: answer with one short sentence."
    )
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    for idx, meta in enumerate(lows):
        tasks.append(
            asyncio.create_task(
                send_dynamo_transport_request(
                    gateway_base,
                    model,
                    prompt,
                    meta,
                    trace_path,
                    total_requests,
                    osl,
                    iat,
                )
            )
        )
        if low_stagger_ms > 0 and idx != len(lows) - 1:
            await asyncio.sleep(low_stagger_ms / 1000.0)
    if low_lead_ms > 0:
        await asyncio.sleep(low_lead_ms / 1000.0)
    for meta in urgents:
        tasks.append(
            asyncio.create_task(
                send_dynamo_transport_request(
                    gateway_base,
                    model,
                    prompt,
                    meta,
                    trace_path,
                    total_requests,
                    osl,
                    iat,
                )
            )
        )
    return await asyncio.gather(*tasks)


def write_run_config(path: Path, args: argparse.Namespace, report_label: str, run_root: Path, report_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"REPORT_LABEL={report_label}",
                f"MODEL={args.model}",
                "EXPERIMENT_KIND=nemo_nat_service_priority_probe",
                f"RUN_ROOT={run_root}",
                f"REPORT_DIR={report_dir}",
                "HARNESSES=nemo_agent_toolkit_service",
                "MODES=pre_harness_priority_hints",
                "PRESSURE_LEVELS=nat_service_probe",
                f"NAT_SERVICE_LOW_COUNT={args.low_count}",
                f"NAT_SERVICE_URGENT_COUNT={args.urgent_count}",
                f"NAT_SERVICE_LOW_LEAD_MS={args.low_lead_ms}",
                f"NAT_SERVICE_LOW_STAGGER_MS={args.low_stagger_ms}",
                f"NAT_SERVICE_FAKE_BACKEND_DELAY_MS={args.fake_backend_delay_ms}",
                f"NAT_SERVICE_WORKERS={args.nat_workers}",
                f"NAT_PROVIDER={args.nat_provider}",
                f"NAT_DYNAMO_ENABLE_NVEXT_HINTS={1 if args.nat_provider in {'dynamo', 'dynamo_direct'} else 0}",
                f"NAT_DYNAMO_PREFIX_TOTAL_REQUESTS={args.nvext_prefix_total_requests}",
                f"NAT_DYNAMO_PREFIX_OSL={args.nvext_prefix_osl}",
                f"NAT_DYNAMO_PREFIX_IAT={args.nvext_prefix_iat}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe NAT shared-service priority behavior without patching NAT.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--results-root", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--report-label", default=f"nemo_nat_service_priority_probe_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--nat-bin", default=os.environ.get("HARNESS_NAT_BIN") or "nat")
    parser.add_argument("--low-count", type=int, default=6)
    parser.add_argument("--urgent-count", type=int, default=1)
    parser.add_argument("--low-lead-ms", type=int, default=100)
    parser.add_argument("--low-stagger-ms", type=int, default=5)
    parser.add_argument("--deadline-ms", type=int, default=50)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--fake-backend-delay-ms", type=int, default=1200)
    parser.add_argument("--nat-workers", type=int, default=1)
    parser.add_argument("--nat-provider", choices=("openai", "dynamo", "dynamo_direct"), default="openai")
    parser.add_argument("--nvext-prefix-total-requests", type=int, default=10)
    parser.add_argument("--nvext-prefix-osl", type=int, default=512)
    parser.add_argument("--nvext-prefix-iat", type=int, default=50)
    parser.add_argument("--update-latest", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_root = args.results_root.resolve()
    run_root = results_root / "runs" / "controlled" / args.report_label
    report_dir = results_root / "reports" / args.report_label
    case_dir = run_root / "nemo_agent_toolkit_service_nat_service_probe_pre_harness_priority_hints"
    log_dir = case_dir / "harness_logs"
    trace_path = case_dir / "m27_trace.jsonl"
    gateway_events = case_dir / "harness_gateway_events.jsonl"
    gateway_log = case_dir / "harness_gateway.log"
    nat_home = log_dir / "nat_service"
    nat_config_path = nat_home / "workflow.json"
    nat_log_path = nat_home / "nat_serve.log"
    run_config_path = report_dir / "run_config.env"
    case_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    nat_home.mkdir(parents=True, exist_ok=True)
    for path in (trace_path, gateway_events, gateway_log, nat_log_path):
        path.unlink(missing_ok=True)

    write_jsonl(
        trace_path,
        {
            "event": "m27.nat_service_probe.start",
            "harness": "nemo_agent_toolkit_service",
            "mode": "pre_harness_priority_hints",
            "low_count": args.low_count,
            "urgent_count": args.urgent_count,
            "low_lead_ms": args.low_lead_ms,
            "low_stagger_ms": args.low_stagger_ms,
            "fake_backend_delay_ms": args.fake_backend_delay_ms,
            "nat_workers": args.nat_workers,
            "nat_provider": args.nat_provider,
            "nat_dynamo_enable_nvext_hints": args.nat_provider in {"dynamo", "dynamo_direct"},
            "nat_dynamo_prefix_total_requests": args.nvext_prefix_total_requests,
            "nat_dynamo_prefix_osl": args.nvext_prefix_osl,
            "nat_dynamo_prefix_iat": args.nvext_prefix_iat,
        },
    )

    fake_server = None
    gateway_proc = None
    nat_proc = None
    try:
        fake_server, fake_base = start_fake_sglang(trace_path, args.fake_backend_delay_ms, args.model)
        write_jsonl(trace_path, {"event": "m27.nat_service_probe.fake_sglang_ready", "base_url": fake_base})
        gateway_proc, gateway_base = start_gateway(script_dir, trace_path, gateway_events, fake_base, args.model)
        write_jsonl(trace_path, {"event": "m27.nat_service_probe.gateway_ready", "base_url": gateway_base})
        if args.nat_provider == "dynamo_direct":
            write_jsonl(
                trace_path,
                {
                    "event": "m27.nat_service_probe.dynamo_direct_transport_ready",
                    "gateway_base": gateway_base,
                    "nat_provider": args.nat_provider,
                    "nat_dynamo_enable_nvext_hints": True,
                },
            )
            done_rows = asyncio.run(
                run_dynamo_direct_transport_load(
                    gateway_base,
                    args.model,
                    trace_path,
                    args.low_count,
                    args.urgent_count,
                    args.low_lead_ms,
                    args.low_stagger_ms,
                    args.prompt_tokens,
                    args.deadline_ms,
                    args.nvext_prefix_total_requests,
                    args.nvext_prefix_osl,
                    args.nvext_prefix_iat,
                )
            )
        else:
            nat_config = build_nat_config(
                gateway_base,
                args.model,
                args.nat_provider,
                args.nvext_prefix_total_requests,
                args.nvext_prefix_osl,
                args.nvext_prefix_iat,
            )
            nat_config_path.write_text(json.dumps(nat_config, indent=2, sort_keys=True), encoding="utf-8")
            write_jsonl(
                trace_path,
                {
                    "event": "m27.nat_service_probe.config_written",
                    "nat_config_path": str(nat_config_path),
                    "gateway_base": gateway_base,
                    "nat_provider": args.nat_provider,
                    "nat_dynamo_enable_nvext_hints": args.nat_provider == "dynamo",
                },
            )
            nat_proc, nat_base = start_nat_server(args.nat_bin, nat_config_path, nat_home, nat_log_path, args.nat_workers)
            write_jsonl(trace_path, {"event": "m27.nat_service_probe.nat_ready", "base_url": nat_base})
            done_rows = asyncio.run(
                run_load(
                    nat_base,
                    args.model,
                    trace_path,
                    args.low_count,
                    args.urgent_count,
                    args.low_lead_ms,
                    args.low_stagger_ms,
                    args.prompt_tokens,
                    args.deadline_ms,
                )
            )
        errors = [row for row in done_rows if row.get("error")]
        write_jsonl(
            trace_path,
            {
                "event": "m27.nat_service_probe.end",
                "completed": len(done_rows),
                "errors": len(errors),
                "status": "error" if errors else "ok",
            },
        )
    finally:
        terminate_process(nat_proc)
        terminate_process(gateway_proc)
        if fake_server is not None:
            fake_server.shutdown()
            fake_server.server_close()

    write_run_config(run_config_path, args, args.report_label, run_root, report_dir)
    cmd = [
        sys.executable,
        str(script_dir / "build_multi_harness_deadline_summary.py"),
        "--root",
        str(run_root),
        "--out-dir",
        str(report_dir),
        "--latest-root",
        str(results_root),
        "--report-label",
        args.report_label,
        "--run-config",
        str(run_config_path),
    ]
    if args.update_latest:
        cmd.append("--update-latest")
    subprocess.run(cmd, check=True)
    print(f"Done. Report: {report_dir / 'master_report.html'}")


if __name__ == "__main__":
    main()
