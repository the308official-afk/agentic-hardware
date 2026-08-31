#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from run_real_prompt_controlled_replay import make_pressure_filler_prompt, make_shared_prefix, prompt_hash

MARKER = "HARNESS_REPLAY_EXPERIMENT_JSON:"
SUPPORTED_HARNESSES = (
    "hatcher",
    "codex",
    "claude_code",
    "opencode",
    "qwen_code",
    "nemo_agent_toolkit",
    "deepseek_harness",
    "pi_agent_harness",
    "openclaw",
    "hermes_agent",
)


@dataclass(frozen=True)
class HarnessPair:
    session_id: str
    prompt: str
    replay_prompt: str
    task_index: str
    prompt_tokens: int


def write_trace(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts_ns", time.time_ns())
    row.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def trace_has_event(path: Path, label: str, event: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == event and row.get("label") == label:
                    return True
    except OSError:
        return False
    return False


def marker(meta: dict[str, Any]) -> str:
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MARKER + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def estimate_tokens(prompt: str) -> int:
    return max(1, int(round(len(prompt.split()) * 1.35)))


def build_pairs(harness: str, pressure_level: str, count: int, prompt_tokens: int) -> list[HarnessPair]:
    pairs: list[HarnessPair] = []
    for idx in range(count):
        session_id = f"{harness}_{pressure_level}_session_{idx:03d}"
        shared = make_shared_prefix(session_id, prompt_tokens)
        prompt = (
            f"{shared}\n\n"
            "Initial turn: inspect this synthetic coding task context and answer briefly. "
            "The tool result will arrive later."
        )
        replay_prompt = (
            f"{shared}\n\n"
            "Replay turn after tool wait: the tool returned one failing assertion and a traceback. "
            "Continue from the same context and answer briefly."
        )
        pairs.append(
            HarnessPair(
                session_id=session_id,
                prompt=prompt,
                replay_prompt=replay_prompt,
                task_index=str(idx),
                prompt_tokens=estimate_tokens(prompt),
            )
        )
    return pairs


async def run_hatcher_request(gateway_base: str, model: str, prompt: str, meta: dict[str, Any]) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{marker(meta)}"}],
        "max_tokens": int(meta.get("max_tokens") or 8),
        "temperature": 0,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(f"{gateway_base.rstrip('/')}/v1/chat/completions", json=payload)
        response.raise_for_status()


def cli_or_npx(binary: str, package: str) -> list[str]:
    installed = shutil.which(binary)
    if installed:
        return [installed]
    return ["npx", "-y", package]


def codex_command(gateway_base: str, model: str, prompt: str, meta: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    cmd = [
        "npx",
        "-y",
        "@openai/codex@latest",
        "exec",
        "--ignore-user-config",
        "--strict-config",
        "-c",
        'model_providers.harness.name="Harness Gateway"',
        "-c",
        f'model_providers.harness.base_url="{provider_base}"',
        "-c",
        'model_providers.harness.env_key="DUMMY_KEY"',
        "-c",
        'model_providers.harness.wire_api="responses"',
        "-c",
        'model_provider="harness"',
        "-m",
        model,
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {"DUMMY_KEY": "dummy"}


def claude_command(gateway_base: str, prompt: str, meta: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    cmd = [
        "npx",
        "-y",
        "@anthropic-ai/claude-code@latest",
        "--bare",
        "-p",
        "--model",
        "claude-sonnet-4-5",
        "--output-format",
        "json",
        "--max-budget-usd",
        "0.05",
        "--no-session-persistence",
        "--prompt-suggestions",
        "false",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {
        "ANTHROPIC_BASE_URL": gateway_base.rstrip("/"),
        "ANTHROPIC_API_KEY": "dummy",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def opencode_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    config_dir = log_dir / "opencode_config" / str(meta["label"])
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "harness": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Harness Gateway",
                "options": {
                    "baseURL": provider_base,
                    "apiKey": "dummy",
                },
                "models": {
                    model: {
                        "name": model,
                    }
                },
            }
        },
    }
    (config_dir / "opencode.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        *cli_or_npx("opencode", "opencode-ai@latest"),
        "run",
        "--model",
        f"harness/{model}",
        "--format",
        "json",
        "--dir",
        "/tmp",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {
        "OPENCODE_CONFIG_DIR": str(config_dir),
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
        "OPENCODE_PERMISSION": json.dumps({"edit": "deny", "bash": "deny", "webfetch": "deny"}),
        "OPENAI_API_KEY": "dummy",
    }


def qwen_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    workspace = log_dir / "qwen_workspace" / str(meta["label"])
    settings_dir = workspace / ".qwen"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "$version": 3,
        "model": {
            "name": model,
            "maxSessionTurns": -1,
        },
        "modelProviders": {
            "openai": [
                {
                    "id": model,
                    "name": model,
                    "baseUrl": provider_base,
                    "envKey": "OPENAI_API_KEY",
                }
            ]
        },
        "security": {
            "auth": {
                "selectedType": "openai",
                "apiKey": "dummy",
                "baseUrl": provider_base,
            }
        },
        "tools": {
            "approvalMode": "yolo",
            "exclude": ["shell", "write_file", "edit"],
        },
    }
    (settings_dir / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        *cli_or_npx("qwen", "@qwen-code/qwen-code@latest"),
        "--model",
        model,
        "--output-format",
        "json",
        "--prompt",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {
        "OPENAI_API_KEY": "dummy",
        "OPENAI_BASE_URL": provider_base,
        "OPENAI_MODEL": model,
        "QWEN_MODEL": model,
    }


def openai_chat_probe_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
    adapter_name: str,
) -> tuple[list[str], dict[str, str]]:
    adapter_dir = log_dir / "wireability_adapters" / adapter_name / str(meta["label"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    request_path = adapter_dir / "request.json"
    script_path = adapter_dir / "post_chat_completion.py"
    payload = {
        "url": f"{gateway_base.rstrip('/')}/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": f"{prompt}\n\n{marker(meta)}"}],
            "max_tokens": int(meta.get("max_tokens") or 8),
            "temperature": 0,
            "stream": False,
        },
    }
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    script_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "import sys",
                "import urllib.request",
                "",
                "request = json.loads(open(sys.argv[1], encoding='utf-8').read())",
                "body = json.dumps(request['body']).encode('utf-8')",
                "http_request = urllib.request.Request(",
                "    request['url'],",
                "    data=body,",
                "    headers={'content-type': 'application/json'},",
                "    method='POST',",
                ")",
                "with urllib.request.urlopen(http_request, timeout=None) as response:",
                "    sys.stdout.write(response.read().decode('utf-8', 'replace'))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [sys.executable, str(script_path), str(request_path)], {}


def nemo_agent_toolkit_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    return openai_chat_probe_command(gateway_base, model, prompt, meta, log_dir, "nemo_agent_toolkit")


def deepseek_harness_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    return openai_chat_probe_command(gateway_base, model, prompt, meta, log_dir, "deepseek_harness")


def pi_agent_harness_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    return openai_chat_probe_command(gateway_base, model, prompt, meta, log_dir, "pi_agent_harness")


def openclaw_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    return openai_chat_probe_command(gateway_base, model, prompt, meta, log_dir, "openclaw")


def hermes_agent_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    return openai_chat_probe_command(gateway_base, model, prompt, meta, log_dir, "hermes_agent")


async def run_cli_request(
    harness: str,
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> None:
    if harness == "codex":
        cmd, extra_env = codex_command(gateway_base, model, prompt, meta)
    elif harness == "claude_code":
        cmd, extra_env = claude_command(gateway_base, prompt, meta)
    elif harness == "opencode":
        cmd, extra_env = opencode_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "qwen_code":
        cmd, extra_env = qwen_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "nemo_agent_toolkit":
        cmd, extra_env = nemo_agent_toolkit_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "deepseek_harness":
        cmd, extra_env = deepseek_harness_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "pi_agent_harness":
        cmd, extra_env = pi_agent_harness_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "openclaw":
        cmd, extra_env = openclaw_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "hermes_agent":
        cmd, extra_env = hermes_agent_command(gateway_base, model, prompt, meta, log_dir)
    else:
        raise ValueError(f"unsupported CLI harness: {harness}")
    env = os.environ.copy()
    env.update(extra_env)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{meta['label']}.log"
    trace_path = Path(str(meta.get("_trace_path") or ""))
    request_timeout_secs = float(os.environ.get("HARNESS_REQUEST_TIMEOUT_SECS", "900"))
    stop_when_gateway_done = str(os.environ.get("HARNESS_STOP_WHEN_GATEWAY_DONE", "1")) == "1"

    def run() -> None:
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(log_dir / "qwen_workspace" / str(meta["label"])) if harness == "qwen_code" else "/tmp",
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + request_timeout_secs
            while proc.poll() is None:
                if stop_when_gateway_done and trace_path and trace_has_event(trace_path, str(meta["label"]), "m27.request.end"):
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                    return
                if time.monotonic() >= deadline:
                    proc.kill()
                    proc.wait(timeout=5)
                    if trace_path and trace_has_event(trace_path, str(meta["label"]), "m27.request.end"):
                        return
                    raise TimeoutError(f"{harness} request timed out before gateway completion: {meta['label']}")
                time.sleep(0.25)
            if proc.returncode and not (trace_path and trace_has_event(trace_path, str(meta["label"]), "m27.request.end")):
                raise subprocess.CalledProcessError(proc.returncode, cmd)

    await asyncio.to_thread(run)


async def run_harness_request(
    harness: str,
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> None:
    if harness == "hatcher":
        await run_hatcher_request(gateway_base, model, prompt, meta)
        return
    await run_cli_request(harness, gateway_base, model, prompt, meta, log_dir)


async def run_filler(
    gateway_base: str,
    model: str,
    pair: HarnessPair,
    idx: int,
    meta_base: dict[str, Any],
    tokens: int,
) -> None:
    filler_session = f"{pair.session_id}_pressure_{idx:03d}"
    prompt = make_pressure_filler_prompt(filler_session, tokens)
    meta = {
        **meta_base,
        "session_id": filler_session,
        "phase": "pressure_filler",
        "label": f"{filler_session}_request",
        "task_index": pair.task_index,
        "prompt_hash": prompt_hash(prompt),
        "priority_label": "low",
        "max_tokens": 2,
    }
    await run_hatcher_request(gateway_base, model, prompt, meta)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run target replay traffic through the supported harness adapters.")
    parser.add_argument("--harness", choices=SUPPORTED_HARNESSES, required=True)
    parser.add_argument("--mode", choices=("no_prefetch", "e2e_priority_hints"), required=True)
    parser.add_argument("--pressure-level", required=True)
    parser.add_argument("--gateway-base", default="http://127.0.0.1:31080")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--tool-wait-ms", type=int, default=50)
    parser.add_argument("--target-prompt-tokens", type=int, default=4096)
    parser.add_argument("--filler-sessions", type=int, default=0)
    parser.add_argument("--filler-prompt-tokens", type=int, default=1536)
    parser.add_argument("--session-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--arrival-gap-ms", type=int, default=40)
    args = parser.parse_args()

    if args.harness in {"codex", "claude_code", "opencode", "qwen_code"} and shutil.which("npx") is None:
        missing_bins = {
            "codex": "codex",
            "claude_code": "claude",
            "opencode": "opencode",
            "qwen_code": "qwen",
        }
        if shutil.which(missing_bins[args.harness]) is None:
            raise SystemExit(f"npx or {missing_bins[args.harness]} is required for {args.harness} harness probes.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.harness, args.pressure_level, args.session_count, args.target_prompt_tokens)
    rows: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(args.concurrency)
    workload_start = time.perf_counter()

    def offset_ms() -> float:
        return (time.perf_counter() - workload_start) * 1000.0

    async def sleep_until(target_ms: float) -> None:
        delay = workload_start + target_ms / 1000.0 - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

    write_trace(
        args.trace,
        {
            "event": "m27.workload_start",
            "harness": args.harness,
            "mode": args.mode,
            "pressure_level": args.pressure_level,
            "model": args.model,
            "pairs": len(pairs),
            "tool_wait_list_ms": [args.tool_wait_ms],
            "filler_sessions": args.filler_sessions,
            "target_prompt_tokens": args.target_prompt_tokens,
            "filler_prompt_tokens": args.filler_prompt_tokens,
        },
    )

    async def bounded_request(prompt: str, meta: dict[str, Any]) -> None:
        async with sem:
            await run_harness_request(args.harness, args.gateway_base, args.model, prompt, meta, args.log_dir)

    async def run_pair(pair: HarnessPair, index: int) -> None:
        await sleep_until(index * args.arrival_gap_ms)
        write_trace(
            args.trace,
            {
                "event": "m27.session.start",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "task_index": pair.task_index,
                "tool_names": "synthetic_tool",
                "arrival_offset_ms": round(offset_ms(), 3),
                "tool_wait_ms": args.tool_wait_ms,
                "prompt_tokens": pair.prompt_tokens,
            },
        )
        base_meta = {
            "harness": args.harness,
            "mode": args.mode,
            "pressure_level": args.pressure_level,
            "high_priority": 100,
            "low_priority": -100,
            "_trace_path": str(args.trace),
        }
        initial_meta = {
            **base_meta,
            "session_id": pair.session_id,
            "phase": "initial_turn",
            "label": f"{pair.session_id}_initial",
            "task_index": pair.task_index,
            "prompt_hash": prompt_hash(pair.prompt),
            "priority_label": "high",
            "max_tokens": 8,
        }
        await bounded_request(pair.prompt, initial_meta)
        tool_start_ms = offset_ms()
        replay_due_ms = tool_start_ms + args.tool_wait_ms
        write_trace(
            args.trace,
            {
                "event": "m27.tool_wait.start",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "tool_start_offset_ms": round(tool_start_ms, 3),
                "replay_due_offset_ms": round(replay_due_ms, 3),
                "tool_wait_ms": args.tool_wait_ms,
                "prompt_hash": prompt_hash(pair.prompt),
            },
        )
        filler_tasks = [
            asyncio.create_task(run_filler(args.gateway_base, args.model, pair, idx, base_meta, args.filler_prompt_tokens))
            for idx in range(args.filler_sessions)
        ]
        await sleep_until(replay_due_ms)
        write_trace(
            args.trace,
            {
                "event": "m27.pre_replay.checkpoint",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "replay_due_offset_ms": round(replay_due_ms, 3),
                "expected_reuse": "intercepted_priority" if args.mode == "e2e_priority_hints" else "baseline",
                "gpu_resident_tokens": "unknown",
                "host_resident_tokens": "unknown",
                "missing_tokens": "unknown",
                "protected_tokens": "unknown",
            },
        )
        write_trace(
            args.trace,
            {
                "event": "m27.replay.due",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "replay_due_offset_ms": round(replay_due_ms, 3),
            },
        )
        replay_meta = {
            **base_meta,
            "session_id": pair.session_id,
            "phase": "replay",
            "label": f"{pair.session_id}_replay",
            "task_index": pair.task_index,
            "prompt_hash": prompt_hash(pair.replay_prompt),
            "priority_label": "high",
            "deadline_offset_ms": round(replay_due_ms, 3),
            "max_tokens": 8,
        }
        await bounded_request(pair.replay_prompt, replay_meta)
        write_trace(
            args.trace,
            {
                "event": "m27.tool_wait.end",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "replay_due_offset_ms": round(replay_due_ms, 3),
            },
        )
        await asyncio.gather(*filler_tasks, return_exceptions=True)
        rows.append(
            {
                "harness": args.harness,
                "mode": args.mode,
                "session_id": pair.session_id,
                "tool_wait_ms": args.tool_wait_ms,
                "filler_sessions": args.filler_sessions,
                "target_prompt_tokens": args.target_prompt_tokens,
                "pressure_level": args.pressure_level,
            }
        )

    await asyncio.gather(*(run_pair(pair, idx) for idx, pair in enumerate(pairs)))
    write_trace(args.trace, {"event": "m27.workload_end", "harness": args.harness, "mode": args.mode, "row_count": len(rows)})
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
