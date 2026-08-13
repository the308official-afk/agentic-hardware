#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


TIMING_ALIASES = {
    "early_before_pressure": "pre_pressure",
    "late_after_pressure": "near_resume",
}
DIRECT_LOAD_TRIGGER = "AGENTIC_KV_DIRECT_LOAD_TRIGGER"


def canonical_timing(timing: str) -> str:
    return TIMING_ALIASES.get(timing, timing)


def trace_path() -> Path | None:
    raw_path = os.environ.get("AGENTIC_KV_TRACE_PATH")
    if not raw_path:
        return None
    return Path(raw_path)


def write_trace_event(event: dict[str, Any]) -> None:
    path = trace_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    event.setdefault("ts_ns", time.time_ns())
    event.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def session_id_from_label(label: str) -> str:
    parts = label.split("_")
    if len(parts) >= 2 and parts[0] in {"target", "filler"}:
        return f"{parts[0]}_{parts[1]}"
    return label


def request_role_from_label(label: str) -> str:
    if label.startswith("target_"):
        return "target"
    if label.startswith("filler_"):
        return "filler"
    return "unknown"


def make_prompt(label: str, target_tokens: int) -> str:
    header = (
        f"You are a coding agent session {label}. "
        "You are debugging a SWE-bench style repository failure. "
        "Keep track of files inspected, tests run, edits attempted, and tool outputs. "
    )
    chunk = (
        "Repository note: failing pytest assertion, stack trace line, candidate source file, "
        "dependency graph, previous patch, reviewer comment, build output, and next hypothesis. "
    )
    return header + chunk * max(1, target_tokens // 24)


async def chat_once(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    label: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    first_token_time: float | None = None
    chunks = 0
    async with client.stream("POST", f"{base_url.rstrip('/')}/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            chunks += 1
            if first_token_time is None:
                first_token_time = time.perf_counter()
    end = time.perf_counter()
    if first_token_time is None:
        first_token_time = end
    return {
        "label": label,
        "prompt_chars": len(prompt),
        "ttft_ms": round((first_token_time - start) * 1000, 3),
        "total_latency_ms": round((end - start) * 1000, 3),
        "stream_chunks": chunks,
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run a KV pressure/resume workload against SGLang.")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--target-sessions", type=int, default=2)
    parser.add_argument("--filler-sessions", type=int, default=18)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--resume-extra-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--tool-wait-ms", type=int, default=500)
    parser.add_argument(
        "--mode",
        choices=("no_prefetch", "generic_prefetch", "hint_aware"),
        default="no_prefetch",
    )
    parser.add_argument(
        "--hint-prefetch-timing",
        choices=(
            "very_early_before_pressure",
            "pre_pressure",
            "early_before_pressure",
            "middle_during_pressure",
            "near_resume",
            "late_after_pressure",
        ),
        default="near_resume",
        help="When hint_aware mode sends target warm/prefetch requests.",
    )
    parser.add_argument("--prefetch-max-tokens", type=int, default=1)
    parser.add_argument(
        "--prefetch-action",
        choices=("request_warm", "direct_probe", "direct_load"),
        default="request_warm",
        help=(
            "request_warm sends a normal SGLang warm request; direct_probe records the intended direct KV load; "
            "direct_load sends a marked trigger request that exercises SGLang's natural init_load_back/load_back path."
        ),
    )
    parser.add_argument("--out", default="artifacts/results/pressure_resume_metrics.jsonl")
    args = parser.parse_args()
    args.hint_prefetch_timing = canonical_timing(args.hint_prefetch_timing)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_prompts = [
        make_prompt(f"target_agent_{idx}", args.prompt_tokens)
        for idx in range(args.target_sessions)
    ]
    filler_prompts = [
        make_prompt(f"pressure_filler_{idx}", args.prompt_tokens)
        for idx in range(args.filler_sessions)
    ]

    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=None) as client:
        async def run_labeled(label: str, prompt: str, phase: str) -> dict[str, Any]:
            async with sem:
                session_id = session_id_from_label(label)
                role = request_role_from_label(label)
                prefix_hash = prompt_hash(prompt)
                write_trace_event(
                    {
                        "event": "agent.request.start",
                        "label": label,
                        "session_id": session_id,
                        "request_role": role,
                        "phase": phase,
                        "prompt_hash": prefix_hash,
                        "prompt_chars": len(prompt),
                        "prompt_tokens_target": args.prompt_tokens,
                    }
                )
                row = await chat_once(client, args.base_url, args.model, prompt, args.max_tokens, label)
                row["phase"] = phase
                row["mode"] = args.mode
                row["hint_prefetch_timing"] = args.hint_prefetch_timing
                row["prefetch_action"] = args.prefetch_action
                row["session_id"] = session_id
                row["prompt_hash"] = prefix_hash
                row["filler_sessions"] = args.filler_sessions
                row["prompt_tokens"] = args.prompt_tokens
                write_trace_event(
                    {
                        "event": "agent.request.end",
                        "label": label,
                        "session_id": session_id,
                        "request_role": role,
                        "phase": phase,
                        "prompt_hash": prefix_hash,
                        "ttft_ms": row["ttft_ms"],
                        "total_latency_ms": row["total_latency_ms"],
                    }
                )
                print(json.dumps(row, sort_keys=True), flush=True)
                rows.append(row)
                return row

        async def prefetch_targets(event_prefix: str, timing: str, phase: str, label_suffix: str) -> None:
            for idx, prompt in enumerate(target_prompts):
                session_id = f"target_{idx}"
                prefix_hash = prompt_hash(prompt)
                write_trace_event(
                    {
                        "event": f"{event_prefix}_start",
                        "session_id": session_id,
                        "priority": "high",
                        "timing": timing,
                        "prefetch_action": args.prefetch_action,
                        "prompt_hash": prefix_hash,
                        "prompt_chars": len(prompt),
                    }
                )
                if args.prefetch_action == "direct_probe" and event_prefix == "agent.hint_prefetch":
                    write_trace_event(
                        {
                            "event": "agent.direct_kv_prefetch_probe",
                            "session_id": session_id,
                            "priority": "high",
                            "timing": timing,
                            "prompt_hash": prefix_hash,
                            "prompt_chars": len(prompt),
                            "probe_only": True,
                            "intended_action": "direct_host_to_gpu_kv_load",
                        }
                    )
                    write_trace_event(
                        {
                            "event": f"{event_prefix}_end",
                            "session_id": session_id,
                            "priority": "high",
                            "timing": timing,
                            "prefetch_action": args.prefetch_action,
                            "prompt_hash": prefix_hash,
                            "probe_only": True,
                        }
                    )
                    continue
                if args.prefetch_action == "direct_load" and event_prefix == "agent.hint_prefetch":
                    trigger_prompt = (
                        prompt
                        + "\n\n"
                        + f"{DIRECT_LOAD_TRIGGER} session_id={session_id} prompt_hash={prefix_hash}"
                    )
                    write_trace_event(
                        {
                            "event": "agent.direct_kv_load_attempt",
                            "session_id": session_id,
                            "priority": "high",
                            "timing": timing,
                            "prompt_hash": prefix_hash,
                            "prompt_chars": len(prompt),
                            "trigger_prompt_hash": prompt_hash(trigger_prompt),
                            "trigger_marker": DIRECT_LOAD_TRIGGER,
                            "intended_action": "exercise_sglang_init_load_back_path",
                        }
                    )
                    trigger_label = f"target_{idx}_direct_load_back"
                    write_trace_event(
                        {
                            "event": "agent.request.start",
                            "label": trigger_label,
                            "session_id": session_id,
                            "request_role": "target",
                            "phase": phase,
                            "prompt_hash": prefix_hash,
                            "trigger_prompt_hash": prompt_hash(trigger_prompt),
                            "prompt_chars": len(trigger_prompt),
                            "prompt_tokens_target": args.prompt_tokens,
                            "prefetch_action": args.prefetch_action,
                        }
                    )
                    row = await chat_once(
                        client,
                        args.base_url,
                        args.model,
                        trigger_prompt,
                        args.prefetch_max_tokens,
                        trigger_label,
                    )
                    row["phase"] = phase
                    row["mode"] = args.mode
                    row["hint_prefetch_timing"] = args.hint_prefetch_timing
                    row["prefetch_action"] = args.prefetch_action
                    row["session_id"] = session_id
                    row["prompt_hash"] = prefix_hash
                    row["trigger_prompt_hash"] = prompt_hash(trigger_prompt)
                    row["filler_sessions"] = args.filler_sessions
                    row["prompt_tokens"] = args.prompt_tokens
                    rows.append(row)
                    write_trace_event(
                        {
                            "event": "agent.request.end",
                            "label": trigger_label,
                            "session_id": session_id,
                            "request_role": "target",
                            "phase": phase,
                            "prompt_hash": prefix_hash,
                            "trigger_prompt_hash": prompt_hash(trigger_prompt),
                            "ttft_ms": row["ttft_ms"],
                            "total_latency_ms": row["total_latency_ms"],
                            "prefetch_action": args.prefetch_action,
                        }
                    )
                    write_trace_event(
                        {
                            "event": "agent.direct_kv_load_request.end",
                            "session_id": session_id,
                            "priority": "high",
                            "timing": timing,
                            "prefetch_action": args.prefetch_action,
                            "prompt_hash": prefix_hash,
                            "trigger_prompt_hash": prompt_hash(trigger_prompt),
                            "ttft_ms": row["ttft_ms"],
                            "total_latency_ms": row["total_latency_ms"],
                        }
                    )
                    write_trace_event(
                        {
                            "event": f"{event_prefix}_end",
                            "session_id": session_id,
                            "priority": "high",
                            "timing": timing,
                            "prefetch_action": args.prefetch_action,
                            "prompt_hash": prefix_hash,
                            "trigger_prompt_hash": prompt_hash(trigger_prompt),
                            "ttft_ms": row["ttft_ms"],
                            "total_latency_ms": row["total_latency_ms"],
                        }
                    )
                    continue
                row = await chat_once(
                    client,
                    args.base_url,
                    args.model,
                    prompt,
                    args.prefetch_max_tokens,
                    f"target_{idx}_{label_suffix}",
                )
                row["phase"] = phase
                row["mode"] = args.mode
                row["hint_prefetch_timing"] = args.hint_prefetch_timing
                row["prefetch_action"] = args.prefetch_action
                row["session_id"] = session_id
                row["prompt_hash"] = prefix_hash
                row["filler_sessions"] = args.filler_sessions
                row["prompt_tokens"] = args.prompt_tokens
                rows.append(row)
                write_trace_event(
                    {
                        "event": f"{event_prefix}_end",
                        "session_id": session_id,
                        "priority": "high",
                        "timing": timing,
                        "prefetch_action": args.prefetch_action,
                        "prompt_hash": prefix_hash,
                        "ttft_ms": row["ttft_ms"],
                        "total_latency_ms": row["total_latency_ms"],
                    }
                )

        async def run_fillers(prompts: list[str], start_idx: int) -> None:
            await asyncio.gather(
                *(
                    run_labeled(f"filler_{start_idx + idx}", prompt, "pressure_filler")
                    for idx, prompt in enumerate(prompts)
                )
            )

        print("Phase 1: warm target sessions", flush=True)
        write_trace_event(
            {
                "event": "agent.mode_start",
                "mode": args.mode,
                "hint_prefetch_timing": args.hint_prefetch_timing,
                "prefetch_action": args.prefetch_action,
                "filler_sessions": args.filler_sessions,
                "prompt_tokens": args.prompt_tokens,
            }
        )
        for idx, prompt in enumerate(target_prompts):
            prefix_hash = prompt_hash(prompt)
            write_trace_event(
                {
                    "event": "agent.session_warm",
                    "session_id": f"target_{idx}",
                    "priority": "high",
                    "prompt_hash": prefix_hash,
                    "prompt_chars": len(prompt),
                }
            )
            write_trace_event(
                {
                    "event": "agent.session_prefix_map",
                    "session_id": f"target_{idx}",
                    "request_role": "target",
                    "prompt_hash": prefix_hash,
                    "prompt_chars": len(prompt),
                    "prompt_tokens_target": args.prompt_tokens,
                }
            )
            await run_labeled(f"target_{idx}_warm", prompt, "target_warm")

        print(f"Tool wait: {args.tool_wait_ms} ms", flush=True)
        if args.mode == "hint_aware":
            for idx in range(args.target_sessions):
                write_trace_event(
                    {
                        "event": "agent.hint_submitted",
                        "session_id": f"target_{idx}",
                        "state": "tool_wait",
                        "priority": "high",
                        "expected_resume_ms": args.tool_wait_ms,
                        "reuse_confidence": 0.9,
                        "prefetch_timing": args.hint_prefetch_timing,
                        "prefetch_action": args.prefetch_action,
                        "prompt_hash": prompt_hash(target_prompts[idx]),
                    }
                )
            if args.hint_prefetch_timing == "very_early_before_pressure":
                print("Hint-aware prefetch: warm high-priority targets immediately after tool wait starts", flush=True)
                await prefetch_targets(
                    "agent.hint_prefetch",
                    "very_early_before_pressure",
                    "hint_prefetch",
                    "hint_prefetch",
                )
        await asyncio.sleep(args.tool_wait_ms / 1000)

        if args.mode == "generic_prefetch":
            print("Generic prefetch: warm targets before pressure", flush=True)
            await prefetch_targets(
                "agent.generic_prefetch",
                "pre_pressure",
                "generic_prefetch",
                "generic_prefetch",
            )

        if args.mode == "hint_aware" and args.hint_prefetch_timing == "pre_pressure":
            print("Hint-aware prefetch: warm high-priority targets before pressure", flush=True)
            await prefetch_targets(
                "agent.hint_prefetch",
                "pre_pressure",
                "hint_prefetch",
                "hint_prefetch",
            )

        print("Phase 2: create KV pressure with filler sessions", flush=True)
        write_trace_event(
            {
                "event": "agent.pressure_start",
                "filler_sessions": args.filler_sessions,
                "prompt_tokens_target": args.prompt_tokens,
            }
        )
        if args.mode == "hint_aware" and args.hint_prefetch_timing == "middle_during_pressure":
            midpoint = max(1, len(filler_prompts) // 2)
            await run_fillers(filler_prompts[:midpoint], 0)
            print("Hint-aware prefetch: warm high-priority targets during pressure", flush=True)
            await prefetch_targets(
                "agent.hint_prefetch",
                "middle_during_pressure",
                "hint_prefetch",
                "hint_prefetch",
            )
            await run_fillers(filler_prompts[midpoint:], midpoint)
        else:
            await run_fillers(filler_prompts, 0)

        if args.mode == "hint_aware" and args.hint_prefetch_timing == "near_resume":
            print("Hint-aware prefetch: warm high-priority targets near resume", flush=True)
            await prefetch_targets(
                "agent.hint_prefetch",
                "near_resume",
                "hint_prefetch",
                "hint_prefetch",
            )

        print("Phase 3: resume target sessions", flush=True)
        resume_suffix = "\nTool result: pytest failed again. " + make_prompt(
            "resume_tool_output", args.resume_extra_tokens
        )
        for idx, prompt in enumerate(target_prompts):
            write_trace_event(
                {
                    "event": "agent.resume_start",
                    "session_id": f"target_{idx}",
                    "priority": "high",
                    "hint_expected_resume_ms": args.tool_wait_ms,
                    "prompt_hash": prompt_hash(prompt),
                }
            )
            await run_labeled(f"target_{idx}_resume", prompt + resume_suffix, "target_resume")

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote pressure workload metrics to {output_path}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
