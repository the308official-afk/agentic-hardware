#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


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
    parser.add_argument("--out", default="artifacts/results/pressure_resume_metrics.jsonl")
    args = parser.parse_args()

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
                write_trace_event(
                    {
                        "event": "agent.request.start",
                        "label": label,
                        "phase": phase,
                        "prompt_chars": len(prompt),
                    }
                )
                row = await chat_once(client, args.base_url, args.model, prompt, args.max_tokens, label)
                row["phase"] = phase
                write_trace_event(
                    {
                        "event": "agent.request.end",
                        "label": label,
                        "phase": phase,
                        "ttft_ms": row["ttft_ms"],
                        "total_latency_ms": row["total_latency_ms"],
                    }
                )
                print(json.dumps(row, sort_keys=True), flush=True)
                rows.append(row)
                return row

        print("Phase 1: warm target sessions", flush=True)
        for idx, prompt in enumerate(target_prompts):
            write_trace_event(
                {
                    "event": "agent.session_warm",
                    "session_id": f"target_{idx}",
                    "priority": "high",
                    "prompt_chars": len(prompt),
                }
            )
            await run_labeled(f"target_{idx}_warm", prompt, "target_warm")

        print(f"Tool wait: {args.tool_wait_ms} ms", flush=True)
        for idx in range(args.target_sessions):
            write_trace_event(
                {
                    "event": "agent.hint_submitted",
                    "session_id": f"target_{idx}",
                    "state": "tool_wait",
                    "priority": "high",
                    "expected_resume_ms": args.tool_wait_ms,
                    "reuse_confidence": 0.9,
                }
            )
        await asyncio.sleep(args.tool_wait_ms / 1000)

        print("Phase 2: create KV pressure with filler sessions", flush=True)
        write_trace_event(
            {
                "event": "agent.pressure_start",
                "filler_sessions": args.filler_sessions,
                "prompt_tokens_target": args.prompt_tokens,
            }
        )
        await asyncio.gather(
            *(
                run_labeled(f"filler_{idx}", prompt, "pressure_filler")
                for idx, prompt in enumerate(filler_prompts)
            )
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
