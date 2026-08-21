#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agentic_kv.nvtx import mark, range_scope


DIRECT_LOAD_TRIGGER = "AGENTIC_KV_DIRECT_LOAD_TRIGGER"


@dataclass(frozen=True)
class AgentSession:
    session_id: str
    arrival_ms: int
    tool_wait_ms: int
    prompt_tokens: int
    priority: str
    prompt: str | None = None
    replay_prompt: str | None = None


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


def trace_and_mark(event: dict[str, Any]) -> None:
    write_trace_event(event)
    session_id = event.get("session_id")
    mode = event.get("mode")
    name = event.get("event", "event")
    if session_id and mode:
        mark(f"agentic_kv:{name}:session={session_id}:mode={mode}")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.replace(",", " ").split() if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_range(raw: str) -> tuple[int, int]:
    parts = [int(item) for item in raw.replace(",", " ").split() if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected two integers for range, got: {raw}")
    lo, hi = parts
    if lo > hi:
        raise ValueError(f"range lower bound is larger than upper bound: {raw}")
    return lo, hi


def make_prompt(session_id: str, target_tokens: int, *, replay: bool = False) -> str:
    header = (
        f"You are coding agent session {session_id}. "
        "You are working on a SWE-bench style bug fix. "
        "Track files inspected, tests run, failures observed, and the next patch hypothesis. "
    )
    if replay:
        header += (
            "Tool result: pytest returned a failing assertion after the previous command. "
            "Use the prior repository context and continue from the same task. "
        )
    chunk = (
        "Repository context: failing test name, stack trace frame, source file candidate, "
        "previous edit, build log, dependency note, reviewer hint, and next diagnostic step. "
    )
    return header + chunk * max(1, target_tokens // 24)


async def chat_once(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    label: str,
    custom_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if custom_params:
        payload["custom_params"] = custom_params
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


def build_sessions(
    *,
    session_count: int,
    arrival_gap_ms: int,
    tool_wait_values: list[int],
    prompt_token_values: list[int],
    randomize: bool = False,
    seed: int = 0,
    arrival_gap_range_ms: tuple[int, int] = (80, 240),
    tool_wait_range_ms: tuple[int, int] = (250, 2200),
) -> list[AgentSession]:
    sessions: list[AgentSession] = []
    rng = random.Random(seed)
    next_arrival_ms = 0
    for idx in range(session_count):
        priority = "high" if idx % 4 == 0 else "normal"
        if randomize:
            if idx == 0:
                next_arrival_ms = 0
            else:
                next_arrival_ms += rng.randint(*arrival_gap_range_ms)
            arrival_ms = next_arrival_ms
            tool_wait_ms = rng.randint(*tool_wait_range_ms)
            prompt_tokens = rng.choice(prompt_token_values)
        else:
            arrival_ms = idx * arrival_gap_ms
            tool_wait_ms = tool_wait_values[idx % len(tool_wait_values)]
            prompt_tokens = prompt_token_values[idx % len(prompt_token_values)]
        sessions.append(
            AgentSession(
                session_id=f"agent_{idx:03d}",
                arrival_ms=arrival_ms,
                tool_wait_ms=tool_wait_ms,
                prompt_tokens=prompt_tokens,
                priority=priority,
            )
        )
    return sessions


def read_replay_workload(path: Path) -> list[AgentSession]:
    sessions: list[AgentSession] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            session_id = str(row.get("session_id") or f"agentbench_{idx:03d}")
            prompt = str(row.get("prompt") or "")
            replay_prompt = str(row.get("replay_prompt") or prompt)
            prompt_tokens = int(row.get("prompt_tokens") or max(1, len(prompt.split())))
            sessions.append(
                AgentSession(
                    session_id=session_id,
                    arrival_ms=int(row.get("arrival_ms") or 0),
                    tool_wait_ms=int(row.get("tool_wait_ms") or 0),
                    prompt_tokens=prompt_tokens,
                    priority=str(row.get("priority") or "normal"),
                    prompt=prompt,
                    replay_prompt=replay_prompt,
                )
            )
    if not sessions:
        raise ValueError(f"no replay sessions found in {path}")
    return sessions


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run overlapping agent sessions with tool waits and replay hints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--mode",
        choices=("no_prefetch", "direct_load", "oracle_direct_load"),
        default="no_prefetch",
    )
    parser.add_argument("--session-count", type=int, default=12)
    parser.add_argument("--arrival-gap-ms", type=int, default=120)
    parser.add_argument("--tool-wait-list-ms", default="250 500 900 1600")
    parser.add_argument("--prompt-token-list", default="768 1024 1536")
    parser.add_argument("--randomize-traffic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-gap-range-ms", default="80 240")
    parser.add_argument("--tool-wait-range-ms", default="250 2200")
    parser.add_argument("--hint-delay-ms", type=int, default=120)
    parser.add_argument("--oracle-lead-ms", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prefetch-max-tokens", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default="artifacts/results/agentic_traffic_metrics.jsonl")
    parser.add_argument(
        "--workload-jsonl",
        type=Path,
        default=None,
        help="Optional replay workload built from real AgentBench/DeepAgents traces.",
    )
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    tool_wait_values = parse_int_list(args.tool_wait_list_ms)
    prompt_token_values = parse_int_list(args.prompt_token_list)
    arrival_gap_range_ms = parse_range(args.arrival_gap_range_ms)
    tool_wait_range_ms = parse_range(args.tool_wait_range_ms)
    if args.workload_jsonl is not None:
        sessions = read_replay_workload(args.workload_jsonl)
    else:
        sessions = build_sessions(
            session_count=args.session_count,
            arrival_gap_ms=args.arrival_gap_ms,
            tool_wait_values=tool_wait_values,
            prompt_token_values=prompt_token_values,
            randomize=args.randomize_traffic,
            seed=args.seed,
            arrival_gap_range_ms=arrival_gap_range_ms,
            tool_wait_range_ms=tool_wait_range_ms,
        )

    write_trace_event(
        {
            "event": "traffic.workload_start",
            "mode": args.mode,
            "session_count": args.session_count,
            "arrival_gap_ms": args.arrival_gap_ms,
            "tool_wait_list_ms": tool_wait_values,
            "randomize_traffic": args.randomize_traffic,
            "seed": args.seed,
            "arrival_gap_range_ms": arrival_gap_range_ms,
            "tool_wait_range_ms": tool_wait_range_ms,
            "sampled_sessions": [session.__dict__ for session in sessions],
            "prompt_token_list": prompt_token_values,
            "hint_delay_ms": args.hint_delay_ms,
            "oracle_lead_ms": args.oracle_lead_ms,
            "workload_jsonl": str(args.workload_jsonl) if args.workload_jsonl else "",
        }
    )

    sem = asyncio.Semaphore(args.concurrency)
    workload_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=None) as client:
        async def sleep_until(offset_ms: int) -> None:
            target = workload_start + offset_ms / 1000.0
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

        async def run_request(session: AgentSession, prompt: str, phase: str, label: str, max_tokens: int) -> dict[str, Any]:
            async with sem:
                p_hash = prompt_hash(prompt)
                trace_and_mark(
                    {
                        "event": "agent.request.start",
                        "label": label,
                        "session_id": session.session_id,
                        "request_role": "agent",
                        "phase": phase,
                        "mode": args.mode,
                        "priority": session.priority,
                        "prompt_hash": p_hash,
                        "prompt_chars": len(prompt),
                        "prompt_tokens_target": session.prompt_tokens,
                        "arrival_ms": session.arrival_ms,
                        "tool_wait_ms": session.tool_wait_ms,
                    }
                )
                with range_scope(f"agentic_kv:client_request:session={session.session_id}:phase={phase}:mode={args.mode}"):
                    row = await chat_once(
                        client,
                        args.base_url,
                        args.model,
                        prompt,
                        max_tokens,
                        label,
                        custom_params={
                            "agentic_kv": {
                                "session_id": session.session_id,
                                "phase": phase,
                                "label": label,
                                "mode": args.mode,
                                "prompt_hash": p_hash,
                                "priority": session.priority,
                            }
                        },
                    )
                row.update(
                    {
                        "phase": phase,
                        "mode": args.mode,
                        "session_id": session.session_id,
                        "priority": session.priority,
                        "prompt_hash": p_hash,
                        "arrival_ms": session.arrival_ms,
                        "tool_wait_ms": session.tool_wait_ms,
                        "prompt_tokens": session.prompt_tokens,
                    }
                )
                rows.append(row)
                trace_and_mark(
                    {
                        "event": "agent.request.end",
                        "label": label,
                        "session_id": session.session_id,
                        "request_role": "agent",
                        "phase": phase,
                        "mode": args.mode,
                        "priority": session.priority,
                        "prompt_hash": p_hash,
                        "ttft_ms": row["ttft_ms"],
                        "total_latency_ms": row["total_latency_ms"],
                    }
                )
                print(json.dumps(row, sort_keys=True), flush=True)
                return row

        async def issue_hint(session: AgentSession, base_prompt: str, replay_due_offset_ms: int) -> None:
            if args.mode == "no_prefetch":
                return
            p_hash = prompt_hash(base_prompt)
            action = "direct_load"
            trace_and_mark(
                {
                    "event": "agent.hint_prefetch_start",
                    "session_id": session.session_id,
                    "mode": args.mode,
                    "priority": session.priority,
                    "timing": "oracle_near_resume" if args.mode == "oracle_direct_load" else "frontend_predicted",
                    "prefetch_action": action,
                    "prompt_hash": p_hash,
                    "replay_due_offset_ms": replay_due_offset_ms,
                }
            )
            trigger_prompt = (
                base_prompt
                + "\n\n"
                + f"{DIRECT_LOAD_TRIGGER} session_id={session.session_id} prompt_hash={p_hash}"
            )
            trace_and_mark(
                {
                    "event": "agent.direct_kv_load_attempt",
                    "session_id": session.session_id,
                    "mode": args.mode,
                    "priority": session.priority,
                    "prefetch_action": action,
                    "prompt_hash": p_hash,
                    "trigger_prompt_hash": prompt_hash(trigger_prompt),
                    "trigger_marker": DIRECT_LOAD_TRIGGER,
                    "intended_action": "exercise_sglang_init_load_back_path",
                }
            )
            row = await run_request(
                session,
                trigger_prompt,
                "hint_prefetch",
                f"{session.session_id}_direct_load_hint",
                args.prefetch_max_tokens,
            )
            trace_and_mark(
                {
                    "event": "agent.direct_kv_load_request.end",
                    "session_id": session.session_id,
                    "mode": args.mode,
                    "priority": session.priority,
                    "prefetch_action": action,
                    "prompt_hash": p_hash,
                    "ttft_ms": row["ttft_ms"],
                    "total_latency_ms": row["total_latency_ms"],
                }
            )
            trace_and_mark(
                {
                    "event": "agent.hint_prefetch_end",
                    "session_id": session.session_id,
                    "mode": args.mode,
                    "priority": session.priority,
                    "prefetch_action": action,
                    "prompt_hash": p_hash,
                }
            )

        async def run_session(session: AgentSession) -> None:
            await sleep_until(session.arrival_ms)
            base_prompt = session.prompt or make_prompt(session.session_id, session.prompt_tokens)
            replay_prompt = session.replay_prompt or make_prompt(session.session_id, session.prompt_tokens, replay=True)
            trace_and_mark(
                {
                    "event": "agent.session_arrival",
                    "session_id": session.session_id,
                    "mode": args.mode,
                    "priority": session.priority,
                    "arrival_ms": session.arrival_ms,
                    "tool_wait_ms": session.tool_wait_ms,
                    "prompt_tokens": session.prompt_tokens,
                }
            )
            await run_request(session, base_prompt, "initial_turn", f"{session.session_id}_initial", args.max_tokens)
            tool_start_ns = time.time_ns()
            tool_start_offset_ms = int((time.perf_counter() - workload_start) * 1000)
            replay_due_offset_ms = tool_start_offset_ms + session.tool_wait_ms
            trace_and_mark(
                {
                    "event": "agent.tool_wait_start",
                    "session_id": session.session_id,
                    "mode": args.mode,
                    "priority": session.priority,
                    "expected_resume_ms": session.tool_wait_ms,
                    "tool_start_offset_ms": tool_start_offset_ms,
                    "replay_due_offset_ms": replay_due_offset_ms,
                    "prompt_hash": prompt_hash(base_prompt),
                }
            )
            hint_task: asyncio.Task[None] | None = None
            if args.mode != "no_prefetch":
                trace_and_mark(
                    {
                        "event": "agent.hint_submitted",
                        "session_id": session.session_id,
                        "mode": args.mode,
                        "priority": session.priority,
                        "expected_resume_ms": session.tool_wait_ms,
                        "tool_start_offset_ms": tool_start_offset_ms,
                        "reuse_confidence": 0.75,
                        "prefetch_action": "direct_load",
                        "prompt_hash": prompt_hash(base_prompt),
                    }
                )
                hint_offset = tool_start_offset_ms + args.hint_delay_ms
                if args.mode == "oracle_direct_load":
                    hint_offset = max(tool_start_offset_ms, replay_due_offset_ms - args.oracle_lead_ms)
                trace_and_mark(
                    {
                        "event": "agent.hint_task_scheduled",
                        "session_id": session.session_id,
                        "mode": args.mode,
                        "priority": session.priority,
                        "hint_offset_ms": hint_offset,
                        "replay_due_offset_ms": replay_due_offset_ms,
                    }
                )

                async def run_hint_task() -> None:
                    await sleep_until(hint_offset)
                    await issue_hint(session, base_prompt, replay_due_offset_ms)

                hint_task = asyncio.create_task(run_hint_task())
            await sleep_until(replay_due_offset_ms)
            trace_and_mark(
                {
                    "event": "agent.replay_due",
                    "session_id": session.session_id,
                    "mode": args.mode,
                    "priority": session.priority,
                    "tool_start_ts_ns": tool_start_ns,
                    "replay_due_offset_ms": replay_due_offset_ms,
                }
            )
            trace_and_mark(
                {
                    "event": "agent.resume_start",
                    "session_id": session.session_id,
                    "mode": args.mode,
                    "priority": session.priority,
                    "hint_expected_resume_ms": session.tool_wait_ms,
                    "prompt_hash": prompt_hash(base_prompt),
                }
            )
            await run_request(session, replay_prompt, "replay", f"{session.session_id}_replay", args.max_tokens)
            if hint_task is not None:
                await hint_task

        await asyncio.gather(*(run_session(session) for session in sessions))

    write_trace_event(
        {
            "event": "traffic.workload_end",
            "mode": args.mode,
            "session_count": args.session_count,
            "row_count": len(rows),
        }
    )
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote agentic traffic metrics to {output_path}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
