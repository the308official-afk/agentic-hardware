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

DIRECT_LOAD_TRIGGER = "AGENTIC_KV_DIRECT_LOAD_TRIGGER"
DYNAMO_PRIORITY_MODE = "dynamo_priority_hints"


class NoopAsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


@dataclass(frozen=True)
class ReplayPair:
    session_id: str
    prompt: str
    replay_prompt: str
    source: str
    task_index: str
    tool_names: str
    priority: str
    prompt_tokens: int


def trace_path() -> Path | None:
    raw_path = os.environ.get("AGENTIC_KV_TRACE_PATH")
    return Path(raw_path) if raw_path else None


def write_trace_event(event: dict[str, Any]) -> None:
    path = trace_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    event.setdefault("ts_ns", time.time_ns())
    event.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(prompt: str) -> int:
    return max(1, int(round(len(prompt.split()) * 1.35)))


def parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.replace(",", " ").split() if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def make_fallback_prompt(session_id: str, target_tokens: int, *, replay: bool = False) -> str:
    header = (
        f"You are coding agent session {session_id}. "
        "You are debugging a SWE-bench style repository failure. "
        "Remember files inspected, tool outputs, test failures, and patch hypotheses. "
    )
    if replay:
        header += (
            "Tool result: the previous command returned a stack trace and one failing assertion. "
            "Continue from the same repository state and decide the next step. "
        )
    chunk = (
        "Repository context includes failing pytest output, candidate source files, grep hits, "
        "recent edits, dependency notes, and reviewer comments. "
    )
    return header + chunk * max(1, target_tokens // 22)


def make_shared_prefix(session_id: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    lines = [
        f"SHARED_AGENT_CONTEXT_BEGIN session={session_id}",
        "This block is intentionally shared by the first turn and replay turn.",
        "It creates a large common prefix so SGLang has meaningful KV to reuse, load, or recompute.",
    ]
    estimated_words = sum(len(line.split()) for line in lines)
    idx = 0
    while max(1, int(round(estimated_words * 1.35))) < target_tokens:
        words = [
            "sharedcontext",
            session_id,
            f"block{idx:05d}",
            f"module{idx % 97}",
            f"symbol{idx % 211}",
            f"test{idx % 163}",
            "repository",
            "history",
            "patch",
            "failure",
            "context",
        ]
        lines.append(" ".join(words))
        estimated_words += len(words)
        idx += 1
    lines.append("SHARED_AGENT_CONTEXT_END")
    return "\n".join(lines)


def make_pressure_filler_prompt(session_id: str, target_tokens: int) -> str:
    lines = [
        f"PRESSURE_FILLER_UNIQUE_BEGIN session={session_id}",
        "This pressure request intentionally diverges immediately from target agent prompts.",
    ]
    estimated_words = sum(len(line.split()) for line in lines)
    idx = 0
    while max(1, int(round(estimated_words * 1.35))) < target_tokens:
        digest = hashlib.sha256(f"{session_id}:{idx}".encode("utf-8")).hexdigest()[:12]
        words = [
            "uniquepressure",
            digest,
            f"repo{idx % 53}",
            f"path{idx}",
            f"symbol{idx % 997}",
            "cache",
            "pressure",
            "unshared",
            "prefix",
        ]
        lines.append(" ".join(words))
        estimated_words += len(words)
        idx += 1
    lines.append("PRESSURE_FILLER_UNIQUE_END")
    return "\n".join(lines)


def pad_pair_with_shared_prefix(pair: ReplayPair, target_prompt_tokens: int) -> ReplayPair:
    if target_prompt_tokens <= 0:
        return pair
    current_min_tokens = min(estimate_tokens(pair.prompt), estimate_tokens(pair.replay_prompt))
    missing_tokens = max(0, target_prompt_tokens - current_min_tokens)
    if missing_tokens <= 0:
        return pair
    shared_prefix = make_shared_prefix(pair.session_id, missing_tokens)
    prompt = f"{shared_prefix}\n\n{pair.prompt}"
    replay_prompt = f"{shared_prefix}\n\n{pair.replay_prompt}"
    return ReplayPair(
        session_id=pair.session_id,
        prompt=prompt,
        replay_prompt=replay_prompt,
        source=f"{pair.source}+shared_prefix",
        task_index=pair.task_index,
        tool_names=pair.tool_names,
        priority=pair.priority,
        prompt_tokens=estimate_tokens(prompt),
    )


def load_workload_jsonl(path: Path, max_pairs: int) -> list[ReplayPair]:
    pairs: list[ReplayPair] = []
    if not path.exists() or path.stat().st_size == 0:
        return pairs
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(row.get("prompt") or "")
            replay_prompt = str(row.get("replay_prompt") or "")
            if not prompt or not replay_prompt:
                continue
            session_id = str(row.get("session_id") or f"real_gap_{idx:03d}")
            pairs.append(
                ReplayPair(
                    session_id=f"m27_{idx:03d}_{session_id}"[:96],
                    prompt=prompt,
                    replay_prompt=replay_prompt,
                    source=str(row.get("source") or path.name),
                    task_index=str(row.get("task_index") or ""),
                    tool_names=str(row.get("tool_names") or row.get("tools") or ""),
                    priority=str(row.get("priority") or "high"),
                    prompt_tokens=int(row.get("prompt_tokens") or estimate_tokens(prompt)),
                )
            )
            if len(pairs) >= max_pairs:
                break
    return pairs


def build_fallback_pairs(max_pairs: int, prompt_tokens: int) -> list[ReplayPair]:
    pairs: list[ReplayPair] = []
    tool_cycle = ["read_file", "grep", "execute", "edit_file"]
    for idx in range(max_pairs):
        session_id = f"m27_fallback_{idx:03d}"
        pairs.append(
            ReplayPair(
                session_id=session_id,
                prompt=make_fallback_prompt(session_id, prompt_tokens),
                replay_prompt=make_fallback_prompt(session_id, prompt_tokens, replay=True),
                source="fallback_realistic_prompt",
                task_index=str(idx // 4),
                tool_names=tool_cycle[idx % len(tool_cycle)],
                priority="high" if idx % 4 == 0 else "normal",
                prompt_tokens=prompt_tokens,
            )
        )
    return pairs


async def chat_once(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    label: str,
    custom_params: dict[str, Any],
    request_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "custom_params": custom_params,
    }
    if request_extra:
        payload.update(request_extra)
    start_perf = time.perf_counter()
    start_ns = time.time_ns()
    first_token_perf: float | None = None
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
            if first_token_perf is None:
                first_token_perf = time.perf_counter()
    end_perf = time.perf_counter()
    end_ns = time.time_ns()
    if first_token_perf is None:
        first_token_perf = end_perf
    return {
        "label": label,
        "request_start_ns": start_ns,
        "request_end_ns": end_ns,
        "prompt_chars": len(prompt),
        "prompt_hash": prompt_hash(prompt),
        "ttft_ms": round((first_token_perf - start_perf) * 1000.0, 3),
        "total_latency_ms": round((end_perf - start_perf) * 1000.0, 3),
        "stream_chunks": chunks,
    }


def agentic_params(pair: ReplayPair, phase: str, mode: str, label: str, p_hash: str) -> dict[str, Any]:
    correlation_id = f"{pair.session_id}:{phase}:{label}"
    return {
        "agentic_kv": {
            "session_id": pair.session_id,
            "phase": phase,
            "label": label,
            "mode": mode,
            "prompt_hash": p_hash,
            "priority": pair.priority,
            "task_index": pair.task_index,
            "request_id": label,
            "parent_run_id": pair.session_id,
            "correlation_id": correlation_id,
            "case_id": pair.session_id,
            "gap_id": str(pair.task_index),
        },
        "request_context": {
            "request_id": label,
            "parent_run_id": pair.session_id,
            "phase": phase,
            "task_index": pair.task_index,
            "correlation_id": correlation_id,
            "case_id": pair.session_id,
            "gap_id": str(pair.task_index),
        },
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled real-prompt replay workload with optional direct KV prefetch hints."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument(
        "--mode",
        choices=("no_prefetch", "direct_prefetch", DYNAMO_PRIORITY_MODE, "oracle_prefetch"),
        default="no_prefetch",
    )
    parser.add_argument("--workload-jsonl", type=Path)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--tool-wait-list-ms", default="100 250 500 1000")
    parser.add_argument("--prefetch-timing", choices=("pre_pressure", "mid_wait", "near_resume"), default="near_resume")
    parser.add_argument("--hint-delay-ms", type=int, default=20)
    parser.add_argument("--oracle-lead-ms", type=int, default=250)
    parser.add_argument("--filler-sessions", type=int, default=32)
    parser.add_argument("--filler-prompt-tokens", type=int, default=1024)
    parser.add_argument("--target-prompt-tokens", type=int, default=0)
    parser.add_argument("--filler-diverge-early", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filler-max-tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prefetch-max-tokens", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--priority-direct-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For direct_prefetch and dynamo_priority_hints, give the prefetch/replay "
            "path a local driver priority window over filler traffic."
        ),
    )
    parser.add_argument(
        "--dynamo-high-priority",
        type=int,
        default=100,
        help="SGLang priority value used for urgent Dynamo-style agent hints.",
    )
    parser.add_argument(
        "--dynamo-normal-priority",
        type=int,
        default=0,
        help="SGLang priority value used for normal requests.",
    )
    parser.add_argument(
        "--dynamo-low-priority",
        type=int,
        default=-100,
        help="SGLang priority value used for low-priority background filler requests.",
    )
    parser.add_argument(
        "--priority-prefetch-head-start-ms",
        type=int,
        default=50,
        help="Delay low-priority filler launch briefly after a direct prefetch hint is submitted.",
    )
    parser.add_argument(
        "--priority-replay-guard-ms",
        type=int,
        default=120,
        help="Pause new low-priority filler launches this many ms before replay is due.",
    )
    parser.add_argument(
        "--priority-replay-release-ms",
        type=int,
        default=80,
        help="Resume low-priority filler launches this many ms after replay is due.",
    )
    parser.add_argument(
        "--priority-filler-stagger-ms",
        type=int,
        default=2,
        help="Small stagger between low-priority filler launches in priority direct prefetch mode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/results/m27_metrics.jsonl"))
    args = parser.parse_args()

    pairs = load_workload_jsonl(args.workload_jsonl, args.max_pairs) if args.workload_jsonl else []
    if not pairs:
        pairs = build_fallback_pairs(args.max_pairs, args.filler_prompt_tokens)
    if args.target_prompt_tokens > 0:
        pairs = [pad_pair_with_shared_prefix(pair, args.target_prompt_tokens) for pair in pairs]
    tool_wait_values = parse_int_list(args.tool_wait_list_ms)
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(args.concurrency)
    workload_start = time.perf_counter()

    write_trace_event(
        {
            "event": "m27.workload_start",
            "mode": args.mode,
            "model": args.model,
            "pairs": len(pairs),
            "workload_jsonl": str(args.workload_jsonl or ""),
            "tool_wait_list_ms": tool_wait_values,
            "filler_sessions": args.filler_sessions,
            "filler_prompt_tokens": args.filler_prompt_tokens,
            "target_prompt_tokens": args.target_prompt_tokens,
            "filler_diverge_early": args.filler_diverge_early,
            "prefetch_timing": args.prefetch_timing,
            "oracle_lead_ms": args.oracle_lead_ms,
            "priority_direct_prefetch": args.priority_direct_prefetch,
            "priority_prefetch_head_start_ms": args.priority_prefetch_head_start_ms,
            "priority_replay_guard_ms": args.priority_replay_guard_ms,
            "priority_replay_release_ms": args.priority_replay_release_ms,
            "priority_filler_stagger_ms": args.priority_filler_stagger_ms,
            "dynamo_high_priority": args.dynamo_high_priority,
            "dynamo_normal_priority": args.dynamo_normal_priority,
            "dynamo_low_priority": args.dynamo_low_priority,
        }
    )

    async with httpx.AsyncClient(timeout=None) as client:
        async def sleep_until(offset_ms: float) -> None:
            target = workload_start + offset_ms / 1000.0
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

        def dynamo_mode_enabled() -> bool:
            return args.mode == DYNAMO_PRIORITY_MODE

        def priority_direct_enabled() -> bool:
            return args.mode in {"direct_prefetch", DYNAMO_PRIORITY_MODE} and args.priority_direct_prefetch

        def priority_policy_name() -> str:
            if dynamo_mode_enabled():
                return "dynamo_agent_hints_to_sglang_priority"
            if priority_direct_enabled():
                return "dynamo_like_direct_prefetch"
            return "none"

        def dynamo_agent_priority(phase: str, pair: ReplayPair) -> str:
            if phase in {"hint_prefetch", "replay"}:
                return "high"
            if phase == "pressure_filler" or pair.priority == "low":
                return "low"
            return "normal"

        def sglang_priority_value(agent_priority: str) -> int:
            if agent_priority == "high":
                return args.dynamo_high_priority
            if agent_priority == "low":
                return args.dynamo_low_priority
            return args.dynamo_normal_priority

        def dynamo_hint_bridge(
            pair: ReplayPair,
            phase: str,
            label: str,
            p_hash: str,
            deadline_ms: float | None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            if not dynamo_mode_enabled():
                return {}, {}
            agent_priority = dynamo_agent_priority(phase, pair)
            sglang_priority = sglang_priority_value(agent_priority)
            hint_action = {
                "hint_prefetch": "prefetch_kv",
                "replay": "consume_kv",
                "pressure_filler": "background_work",
            }.get(phase, "normal_request")
            hints = {
                "schema": "nvext.agent_hints",
                "session_id": pair.session_id,
                "request_id": label,
                "phase": phase,
                "task_index": pair.task_index,
                "prompt_hash": p_hash,
                "priority": agent_priority,
                "deadline_offset_ms": round(deadline_ms, 3) if deadline_ms is not None else "",
                "expected_action": hint_action,
                "reuse_confidence": 0.95 if phase in {"hint_prefetch", "replay"} else 0.2,
                "kv_cache_relevant": phase in {"hint_prefetch", "replay"},
            }
            bridge = {
                "hint_source": "custom_params.nvext.agent_hints",
                "dynamo_agent_priority": agent_priority,
                "sglang_priority": sglang_priority,
                "deadline_offset_ms": round(deadline_ms, 3) if deadline_ms is not None else "",
                "priority_translation": (
                    f"nvext.agent_hints.priority={agent_priority} -> "
                    f"SGLang priority={sglang_priority}"
                ),
                "nvext_agent_hints": hints,
            }
            request_extra = {"priority": sglang_priority}
            return request_extra, bridge

        async def run_request(
            pair: ReplayPair,
            prompt: str,
            phase: str,
            label: str,
            max_tokens: int,
            *,
            use_concurrency_limit: bool = True,
            deadline_ms: float | None = None,
        ) -> dict[str, Any]:
            p_hash = prompt_hash(prompt)
            request_extra, bridge_metadata = dynamo_hint_bridge(pair, phase, label, p_hash, deadline_ms)
            write_trace_event(
                {
                    "event": "m27.request.submitted",
                    "session_id": pair.session_id,
                    "phase": phase,
                    "mode": args.mode,
                    "label": label,
                    "prompt_hash": p_hash,
                    "prompt_chars": len(prompt),
                    "uses_driver_concurrency_gate": use_concurrency_limit,
                    **bridge_metadata,
                }
            )
            limiter = sem if use_concurrency_limit else NoopAsyncContext()
            async with limiter:
                write_trace_event(
                    {
                        "event": "m27.request.start",
                        "session_id": pair.session_id,
                        "phase": phase,
                        "mode": args.mode,
                        "label": label,
                        "prompt_hash": p_hash,
                        "prompt_chars": len(prompt),
                        "uses_driver_concurrency_gate": use_concurrency_limit,
                        **bridge_metadata,
                    }
                )
                params = agentic_params(pair, phase, args.mode, label, p_hash)
                if bridge_metadata:
                    params["nvext"] = {"agent_hints": bridge_metadata["nvext_agent_hints"]}
                    params["dynamo_priority_bridge"] = {
                        "hint_source": bridge_metadata["hint_source"],
                        "dynamo_agent_priority": bridge_metadata["dynamo_agent_priority"],
                        "sglang_priority": bridge_metadata["sglang_priority"],
                        "deadline_offset_ms": bridge_metadata["deadline_offset_ms"],
                        "priority_translation": bridge_metadata["priority_translation"],
                    }
                    params["agentic_kv"].update(
                        {
                            "hint_source": bridge_metadata["hint_source"],
                            "dynamo_agent_priority": bridge_metadata["dynamo_agent_priority"],
                            "sglang_priority": bridge_metadata["sglang_priority"],
                            "deadline_offset_ms": bridge_metadata["deadline_offset_ms"],
                            "priority_translation": bridge_metadata["priority_translation"],
                        }
                    )
                row = await chat_once(
                    client,
                    args.base_url,
                    args.model,
                    prompt,
                    max_tokens,
                    label,
                    params,
                    request_extra,
                )
                row.update(
                    {
                        "session_id": pair.session_id,
                        "phase": phase,
                        "mode": args.mode,
                        "task_index": pair.task_index,
                        "tool_names": pair.tool_names,
                        "priority": pair.priority,
                        "source": pair.source,
                        **{k: v for k, v in bridge_metadata.items() if k != "nvext_agent_hints"},
                    }
                )
                rows.append(row)
                write_trace_event(
                    {
                        "event": "m27.request.end",
                        "session_id": pair.session_id,
                        "phase": phase,
                        "mode": args.mode,
                        "label": label,
                        "prompt_hash": p_hash,
                        "ttft_ms": row["ttft_ms"],
                        "total_latency_ms": row["total_latency_ms"],
                        **bridge_metadata,
                    }
                )
                print(json.dumps(row, sort_keys=True), flush=True)
                return row

        async def wait_for_priority_filler_slot(
            pair: ReplayPair,
            idx: int,
            *,
            tool_start_offset_ms: float,
            replay_due_ms: float,
            hint_offset_ms: float | None,
        ) -> None:
            if not priority_direct_enabled():
                return
            earliest_offset_ms = tool_start_offset_ms
            if hint_offset_ms is not None:
                earliest_offset_ms = max(
                    earliest_offset_ms,
                    hint_offset_ms + args.priority_prefetch_head_start_ms,
                )
            earliest_offset_ms += idx * args.priority_filler_stagger_ms
            guard_start_ms = replay_due_ms - args.priority_replay_guard_ms
            guard_end_ms = replay_due_ms + args.priority_replay_release_ms
            if guard_start_ms <= earliest_offset_ms <= guard_end_ms:
                earliest_offset_ms = guard_end_ms + idx * args.priority_filler_stagger_ms
            write_trace_event(
                {
                    "event": "m27.priority.filler_scheduled",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "filler_index": idx,
                    "earliest_offset_ms": round(earliest_offset_ms, 3),
                    "guard_start_offset_ms": round(guard_start_ms, 3),
                    "guard_end_offset_ms": round(guard_end_ms, 3),
                    "priority_policy": priority_policy_name(),
                }
            )
            await sleep_until(earliest_offset_ms)

        async def run_filler(
            pair: ReplayPair,
            idx: int,
            *,
            tool_start_offset_ms: float | None = None,
            replay_due_ms: float | None = None,
            hint_offset_ms: float | None = None,
        ) -> None:
            if tool_start_offset_ms is not None and replay_due_ms is not None:
                await wait_for_priority_filler_slot(
                    pair,
                    idx,
                    tool_start_offset_ms=tool_start_offset_ms,
                    replay_due_ms=replay_due_ms,
                    hint_offset_ms=hint_offset_ms,
                )
            filler_session_id = f"{pair.session_id}_pressure_{idx}"
            if args.filler_diverge_early:
                prompt = make_pressure_filler_prompt(filler_session_id, args.filler_prompt_tokens)
            else:
                prompt = make_fallback_prompt(filler_session_id, args.filler_prompt_tokens)
            filler = ReplayPair(
                session_id=f"{pair.session_id}_pressure_{idx:03d}",
                prompt=prompt,
                replay_prompt=prompt,
                source="pressure_filler",
                task_index=pair.task_index,
                tool_names="pressure_filler",
                priority="low",
                prompt_tokens=args.filler_prompt_tokens,
            )
            await run_request(filler, prompt, "pressure_filler", f"{filler.session_id}_request", args.filler_max_tokens)

        async def issue_prefetch(pair: ReplayPair, replay_due_ms: float) -> None:
            base_prompt = pair.prompt
            base_hash = prompt_hash(base_prompt)
            trigger_prompt = base_prompt + "\n\n" + f"{DIRECT_LOAD_TRIGGER} session_id={pair.session_id} prompt_hash={base_hash}"
            write_trace_event(
                {
                    "event": "m27.prefetch.start",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "prefetch_action": "direct_load",
                    "prompt_hash": base_hash,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "priority_policy": priority_policy_name(),
                }
            )
            await run_request(
                pair,
                trigger_prompt,
                "hint_prefetch",
                f"{pair.session_id}_direct_prefetch",
                args.prefetch_max_tokens,
                use_concurrency_limit=not priority_direct_enabled(),
                deadline_ms=replay_due_ms,
            )
            write_trace_event(
                {
                    "event": "m27.prefetch.end",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "prefetch_action": "direct_load",
                    "prompt_hash": base_hash,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "priority_policy": priority_policy_name(),
                }
            )

        async def run_pair(pair: ReplayPair, index: int) -> None:
            arrival_ms = index * 40.0
            await sleep_until(arrival_ms)
            tool_wait_ms = tool_wait_values[index % len(tool_wait_values)]
            write_trace_event(
                {
                    "event": "m27.session.start",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "task_index": pair.task_index,
                    "tool_names": pair.tool_names,
                    "arrival_offset_ms": round(arrival_ms, 3),
                    "tool_wait_ms": tool_wait_ms,
                    "prompt_tokens": pair.prompt_tokens,
                }
            )
            await run_request(pair, pair.prompt, "initial_turn", f"{pair.session_id}_initial", args.max_tokens)
            tool_start_offset_ms = (time.perf_counter() - workload_start) * 1000.0
            replay_due_ms = tool_start_offset_ms + tool_wait_ms
            write_trace_event(
                {
                    "event": "m27.tool_wait.start",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "tool_start_offset_ms": round(tool_start_offset_ms, 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "tool_wait_ms": tool_wait_ms,
                    "prompt_hash": prompt_hash(pair.prompt),
                }
            )
            hint_task: asyncio.Task[None] | None = None
            hint_offset_ms: float | None = None
            if args.mode != "no_prefetch":
                if args.mode == "oracle_prefetch":
                    hint_offset_ms = max(tool_start_offset_ms, replay_due_ms - args.oracle_lead_ms)
                    timing = "oracle_before_resume"
                elif args.prefetch_timing == "pre_pressure":
                    hint_offset_ms = tool_start_offset_ms + args.hint_delay_ms
                    timing = "pre_pressure"
                elif args.prefetch_timing == "mid_wait":
                    hint_offset_ms = tool_start_offset_ms + tool_wait_ms / 2.0
                    timing = "mid_wait"
                else:
                    hint_offset_ms = max(tool_start_offset_ms, replay_due_ms - args.oracle_lead_ms)
                    timing = "near_resume"
                write_trace_event(
                    {
                        "event": "m27.hint.submitted",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "timing": timing,
                        "hint_offset_ms": round(hint_offset_ms, 3),
                        "tool_start_offset_ms": round(tool_start_offset_ms, 3),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "reuse_confidence": round(rng.uniform(0.72, 0.95), 3),
                        "priority_policy": priority_policy_name(),
                    }
                )

                async def hint_runner() -> None:
                    await sleep_until(hint_offset_ms)
                    await issue_prefetch(pair, replay_due_ms)

                hint_task = asyncio.create_task(hint_runner())
            pressure_tasks = [
                asyncio.create_task(
                    run_filler(
                        pair,
                        idx,
                        tool_start_offset_ms=tool_start_offset_ms,
                        replay_due_ms=replay_due_ms,
                        hint_offset_ms=hint_offset_ms,
                    )
                )
                for idx in range(args.filler_sessions)
            ]
            await sleep_until(replay_due_ms)
            if priority_direct_enabled():
                write_trace_event(
                    {
                        "event": "m27.priority.replay_lane",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "guard_ms": args.priority_replay_guard_ms,
                        "release_ms": args.priority_replay_release_ms,
                        "priority_policy": priority_policy_name(),
                    }
                )
            write_trace_event(
                {
                    "event": "m27.pre_replay.checkpoint",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "prefetch_hint_submitted": bool(hint_task is not None),
                    "expected_reuse": "high" if args.mode != "no_prefetch" else "baseline",
                    "gpu_resident_tokens": "unknown",
                    "host_resident_tokens": "unknown",
                    "missing_tokens": "unknown",
                    "protected_tokens": "unknown",
                }
            )
            write_trace_event(
                {
                    "event": "m27.replay.due",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                }
            )
            await run_request(
                pair,
                pair.replay_prompt,
                "replay",
                f"{pair.session_id}_replay",
                args.max_tokens,
                use_concurrency_limit=not priority_direct_enabled(),
                deadline_ms=replay_due_ms,
            )
            write_trace_event(
                {
                    "event": "m27.tool_wait.end",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                }
            )
            if hint_task is not None:
                await hint_task
            await asyncio.gather(*pressure_tasks, return_exceptions=True)

        await asyncio.gather(*(run_pair(pair, idx) for idx, pair in enumerate(pairs)))

    write_trace_event({"event": "m27.workload_end", "mode": args.mode, "row_count": len(rows)})
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote controlled replay metrics to {args.out}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
