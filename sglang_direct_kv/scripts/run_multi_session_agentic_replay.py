#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from run_real_prompt_controlled_replay import (
    DIRECT_LOAD_TRIGGER,
    ReplayPair,
    agentic_params,
    build_fallback_pairs,
    chat_once,
    estimate_tokens,
    load_workload_jsonl,
    make_pressure_filler_prompt,
    pad_pair_with_shared_prefix,
    parse_int_list,
    prompt_hash,
    write_trace_event,
)

PREFETCH_MODES = {"direct_prefetch", "priority_direct_prefetch", "deadline_priority_prefetch"}
PRIORITY_PREFETCH_MODES = {"direct_prefetch", "priority_direct_prefetch", "deadline_priority_prefetch"}


class NoopAsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def parse_range(raw: str) -> tuple[int, int]:
    values = parse_int_list(raw)
    if len(values) != 2:
        raise ValueError(f"expected exactly two integers, got {raw!r}")
    lo, hi = values
    if lo > hi:
        raise ValueError(f"range lower bound is larger than upper bound: {raw!r}")
    return lo, hi


def bounded_jitter(value: int, jitter: int, rng: random.Random) -> int:
    if jitter <= 0:
        return value
    return max(1, value + rng.randint(-jitter, jitter))


def arrival_offset_ms(index: int, args: argparse.Namespace, rng: random.Random) -> int:
    if args.arrival_shape == "burst":
        burst = max(1, args.burst_size)
        return (index // burst) * args.burst_gap_ms + (index % burst) * args.arrival_gap_ms
    if args.arrival_shape == "random":
        lo, hi = parse_range(args.arrival_gap_range_ms)
        if index == 0:
            return 0
        return sum(rng.randint(lo, hi) for _ in range(index))
    return index * args.arrival_gap_ms


def normalize_pairs(pairs: list[ReplayPair], session_count: int, target_prompt_tokens: int) -> list[ReplayPair]:
    selected = pairs[:session_count]
    if len(selected) < session_count:
        selected.extend(build_fallback_pairs(session_count - len(selected), target_prompt_tokens or 1024))
    normalized: list[ReplayPair] = []
    for idx, pair in enumerate(selected):
        session_id = f"m36_{idx:03d}_{pair.session_id}"[:96]
        current = replace(pair, session_id=session_id, task_index=str(pair.task_index or idx))
        if target_prompt_tokens > 0:
            current = pad_pair_with_shared_prefix(current, target_prompt_tokens)
        normalized.append(current)
    return normalized


def is_prefetch_mode(mode: str) -> bool:
    return mode in PREFETCH_MODES


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Run many overlapping agentic sessions with direct KV-prefetch hints."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument(
        "--mode",
        choices=("no_prefetch", "direct_prefetch", "priority_direct_prefetch", "deadline_priority_prefetch"),
        default="no_prefetch",
    )
    parser.add_argument("--workload-jsonl", type=Path)
    parser.add_argument("--session-count", type=int, default=16)
    parser.add_argument("--arrival-shape", choices=("staggered", "burst", "random"), default="staggered")
    parser.add_argument("--arrival-gap-ms", type=int, default=120)
    parser.add_argument("--arrival-gap-range-ms", default="60 240")
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument("--burst-gap-ms", type=int, default=800)
    parser.add_argument("--tool-wait-list-ms", default="100 250 500 1000")
    parser.add_argument("--tool-wait-jitter-ms", type=int, default=0)
    parser.add_argument("--prefetch-timing", choices=("early", "near_resume"), default="early")
    parser.add_argument("--hint-delay-ms", type=int, default=20)
    parser.add_argument("--prefetch-lead-ms", type=int, default=120)
    parser.add_argument(
        "--priority-prefetch-window-ms",
        type=int,
        default=500,
        help="In direct_prefetch, hold low-priority fillers while the urgent prefetch runs, up to this window.",
    )
    parser.add_argument(
        "--priority-post-prefetch-quiet-ms",
        type=int,
        default=0,
        help="Optional quiet period after a priority prefetch before low-priority fillers are released.",
    )
    parser.add_argument(
        "--deadline-reserve-window-ms",
        type=int,
        default=300,
        help="For direct_prefetch, reserve this window before replay by deferring low-priority fillers.",
    )
    parser.add_argument("--background-fillers-per-session", type=int, default=0)
    parser.add_argument("--filler-prompt-tokens", type=int, default=1024)
    parser.add_argument("--target-prompt-tokens", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prefetch-max-tokens", type=int, default=1)
    parser.add_argument("--filler-max-tokens", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/results/m36_metrics.jsonl"))
    args = parser.parse_args()

    pairs = load_workload_jsonl(args.workload_jsonl, args.session_count) if args.workload_jsonl else []
    if not pairs:
        pairs = build_fallback_pairs(args.session_count, args.target_prompt_tokens or args.filler_prompt_tokens)
    pairs = normalize_pairs(pairs, args.session_count, args.target_prompt_tokens)

    rng = random.Random(args.seed)
    tool_wait_values = parse_int_list(args.tool_wait_list_ms)
    session_specs: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        tool_wait_ms = bounded_jitter(tool_wait_values[idx % len(tool_wait_values)], args.tool_wait_jitter_ms, rng)
        session_specs.append(
            {
                "pair": pair,
                "arrival_ms": arrival_offset_ms(idx, args, rng),
                "tool_wait_ms": tool_wait_ms,
                "priority": pair.priority or ("high" if idx % 4 == 0 else "normal"),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(args.concurrency)
    workload_start = time.perf_counter()

    write_trace_event(
        {
            "event": "m36.workload_start",
            "mode": args.mode,
            "model": args.model,
            "session_count": len(session_specs),
            "arrival_shape": args.arrival_shape,
            "arrival_gap_ms": args.arrival_gap_ms,
            "tool_wait_list_ms": tool_wait_values,
            "tool_wait_jitter_ms": args.tool_wait_jitter_ms,
            "prefetch_timing": args.prefetch_timing,
            "priority_prefetch_window_ms": args.priority_prefetch_window_ms,
            "priority_post_prefetch_quiet_ms": args.priority_post_prefetch_quiet_ms,
            "deadline_reserve_window_ms": args.deadline_reserve_window_ms,
            "background_fillers_per_session": args.background_fillers_per_session,
            "sampled_sessions": [
                {
                    "session_id": spec["pair"].session_id,
                    "arrival_ms": spec["arrival_ms"],
                    "tool_wait_ms": spec["tool_wait_ms"],
                    "task_index": spec["pair"].task_index,
                    "tool_names": spec["pair"].tool_names,
                    "prompt_tokens": spec["pair"].prompt_tokens,
                }
                for spec in session_specs
            ],
        }
    )

    async with httpx.AsyncClient(timeout=None) as client:
        async def sleep_until(offset_ms: float) -> None:
            target = workload_start + offset_ms / 1000.0
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

        async def run_request(
            pair: ReplayPair,
            prompt: str,
            phase: str,
            label: str,
            max_tokens: int,
            *,
            use_concurrency_limit: bool = True,
        ) -> dict[str, Any]:
            p_hash = prompt_hash(prompt)
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
                    }
                )
                row = await chat_once(
                    client,
                    args.base_url,
                    args.model,
                    prompt,
                    max_tokens,
                    label,
                    agentic_params(pair, phase, args.mode, label, p_hash),
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
                        "prompt_tokens": pair.prompt_tokens,
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
                    }
                )
                print(json.dumps(row, sort_keys=True), flush=True)
                return row

        async def run_background_filler(pair: ReplayPair, index: int) -> None:
            session_id = f"{pair.session_id}_ambient_{index:03d}"
            prompt = make_pressure_filler_prompt(session_id, args.filler_prompt_tokens)
            filler = ReplayPair(
                session_id=session_id,
                prompt=prompt,
                replay_prompt=prompt,
                source="multi_session_background_filler",
                task_index=pair.task_index,
                tool_names="ambient_pressure",
                priority="low",
                prompt_tokens=estimate_tokens(prompt),
            )
            await run_request(filler, prompt, "pressure_filler", f"{session_id}_request", args.filler_max_tokens)

        async def issue_prefetch(pair: ReplayPair, replay_due_ms: float) -> None:
            base_hash = prompt_hash(pair.prompt)
            trigger_prompt = pair.prompt + "\n\n" + (
                f"{DIRECT_LOAD_TRIGGER} session_id={pair.session_id} prompt_hash={base_hash}"
            )
            write_trace_event(
                {
                    "event": "m27.prefetch.start",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "prefetch_action": "direct_load",
                    "prompt_hash": base_hash,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                }
            )
            await run_request(
                pair,
                trigger_prompt,
                "hint_prefetch",
                f"{pair.session_id}_direct_prefetch",
                args.prefetch_max_tokens,
                use_concurrency_limit=args.mode not in PRIORITY_PREFETCH_MODES,
            )
            write_trace_event(
                {
                    "event": "m27.prefetch.end",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "prefetch_action": "direct_load",
                    "prompt_hash": base_hash,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                }
            )

        async def run_session(index: int, spec: dict[str, Any]) -> None:
            pair: ReplayPair = spec["pair"]
            await sleep_until(spec["arrival_ms"])
            write_trace_event(
                {
                    "event": "m27.session.start",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "task_index": pair.task_index,
                    "tool_names": pair.tool_names,
                    "arrival_offset_ms": round(spec["arrival_ms"], 3),
                    "tool_wait_ms": spec["tool_wait_ms"],
                    "prompt_tokens": pair.prompt_tokens,
                    "traffic_shape": "multi_session",
                    "traffic_session_index": index,
                }
            )
            await run_request(pair, pair.prompt, "initial_turn", f"{pair.session_id}_initial", args.max_tokens)

            tool_start_offset_ms = (time.perf_counter() - workload_start) * 1000.0
            replay_due_ms = tool_start_offset_ms + spec["tool_wait_ms"]
            write_trace_event(
                {
                    "event": "m27.tool_wait.start",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "tool_start_offset_ms": round(tool_start_offset_ms, 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "tool_wait_ms": spec["tool_wait_ms"],
                    "prompt_hash": prompt_hash(pair.prompt),
                }
            )

            hint_task: asyncio.Task[None] | None = None
            replay_completed_event = asyncio.Event()
            if is_prefetch_mode(args.mode):
                if args.mode == "deadline_priority_prefetch":
                    hint_offset_ms = tool_start_offset_ms + args.hint_delay_ms
                    timing = "deadline_early"
                elif args.prefetch_timing == "near_resume":
                    hint_offset_ms = max(tool_start_offset_ms, replay_due_ms - args.prefetch_lead_ms)
                    timing = "near_resume"
                else:
                    hint_offset_ms = tool_start_offset_ms + args.hint_delay_ms
                    timing = "early"
                write_trace_event(
                    {
                        "event": "m27.hint.submitted",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "timing": timing,
                        "hint_offset_ms": round(hint_offset_ms, 3),
                        "tool_start_offset_ms": round(tool_start_offset_ms, 3),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "reuse_confidence": 0.82,
                        "traffic_shape": "multi_session",
                        "priority_policy": (
                            "deadline_reserved_prefetch_and_replay_lane"
                            if args.mode == "deadline_priority_prefetch"
                            else "dynamo_like_priority_hint_prefetch_and_replay_lane"
                            if args.mode in PRIORITY_PREFETCH_MODES
                            else "best_effort"
                        ),
                        "priority_prefetch_window_ms": args.priority_prefetch_window_ms
                        if args.mode in PRIORITY_PREFETCH_MODES
                        else 0,
                        "deadline_reserve_window_ms": args.deadline_reserve_window_ms
                        if args.mode in PRIORITY_PREFETCH_MODES
                        else 0,
                        "uses_driver_concurrency_gate": args.mode not in PRIORITY_PREFETCH_MODES,
                    }
                )
                if args.mode == "deadline_priority_prefetch":
                    write_trace_event(
                        {
                            "event": "m38.deadline_service.start",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "tool_start_offset_ms": round(tool_start_offset_ms, 3),
                            "hint_offset_ms": round(hint_offset_ms, 3),
                            "replay_due_offset_ms": round(replay_due_ms, 3),
                            "policy": "coordinate_prefetch_residency_and_replay_admission",
                        }
                    )
                    write_trace_event(
                        {
                            "event": "m38.deadline_prefetch_deadline.set",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "hint_offset_ms": round(hint_offset_ms, 3),
                            "replay_due_offset_ms": round(replay_due_ms, 3),
                            "deadline_reserve_window_ms": args.deadline_reserve_window_ms,
                            "priority_post_prefetch_quiet_ms": args.priority_post_prefetch_quiet_ms,
                            "policy": "early_hint_reserved_lane_hold_fillers_until_replay",
                        }
                    )

                async def run_hint_task() -> None:
                    await sleep_until(hint_offset_ms)
                    if args.mode in PRIORITY_PREFETCH_MODES:
                        write_trace_event(
                            {
                                "event": "m38.deadline_priority_prefetch_lane.start"
                                if args.mode == "deadline_priority_prefetch"
                                else "m38.dynamo_like_prefetch_lane.start",
                                "session_id": pair.session_id,
                                "mode": args.mode,
                                "hint_offset_ms": round(hint_offset_ms, 3),
                                "replay_due_offset_ms": round(replay_due_ms, 3),
                                "priority_prefetch_window_ms": args.priority_prefetch_window_ms,
                                "deadline_reserve_window_ms": args.deadline_reserve_window_ms
                                if args.mode == "deadline_priority_prefetch"
                                else 0,
                                "uses_driver_concurrency_gate": False,
                            }
                        )
                    await issue_prefetch(pair, replay_due_ms)
                    if args.mode == "deadline_priority_prefetch":
                        protection_start_ms = (time.perf_counter() - workload_start) * 1000.0
                        write_trace_event(
                            {
                                "event": "m38.deadline_residency_protection.start",
                                "session_id": pair.session_id,
                                "mode": args.mode,
                                "protection_start_offset_ms": round(protection_start_ms, 3),
                                "replay_due_offset_ms": round(replay_due_ms, 3),
                                "policy": "defer_low_priority_fillers_until_replay_completes",
                            }
                        )
                    if args.mode in PRIORITY_PREFETCH_MODES:
                        write_trace_event(
                            {
                                "event": "m38.deadline_priority_prefetch_lane.end"
                                if args.mode == "deadline_priority_prefetch"
                                else "m38.dynamo_like_prefetch_lane.end",
                                "session_id": pair.session_id,
                                "mode": args.mode,
                                "replay_due_offset_ms": round(replay_due_ms, 3),
                            }
                        )

                hint_task = asyncio.create_task(run_hint_task())

            async def run_filler_group() -> None:
                if args.mode == "deadline_priority_prefetch" and hint_task is not None:
                    reserve_start_ms = max(tool_start_offset_ms, replay_due_ms - args.deadline_reserve_window_ms)
                    write_trace_event(
                        {
                            "event": "m38.deadline_filler_admission.blocked",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "background_fillers_per_session": args.background_fillers_per_session,
                            "block_reason": "replay_critical_prefetch_active",
                            "replay_due_offset_ms": round(replay_due_ms, 3),
                        }
                    )
                    write_trace_event(
                        {
                            "event": "m38.deadline_low_priority_fillers.held",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "background_fillers_per_session": args.background_fillers_per_session,
                            "deadline_reserve_window_ms": args.deadline_reserve_window_ms,
                            "reserve_start_offset_ms": round(reserve_start_ms, 3),
                            "replay_due_offset_ms": round(replay_due_ms, 3),
                            "policy": "hold_low_priority_fillers_until_replay_completes",
                        }
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(hint_task),
                            timeout=max(1, args.priority_prefetch_window_ms) / 1000.0,
                        )
                        hint_release_reason = "prefetch_completed"
                    except asyncio.TimeoutError:
                        hint_release_reason = "priority_window_expired"
                    await replay_completed_event.wait()
                    if args.priority_post_prefetch_quiet_ms > 0:
                        await asyncio.sleep(args.priority_post_prefetch_quiet_ms / 1000.0)
                    write_trace_event(
                        {
                            "event": "m38.deadline_low_priority_fillers.released",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "hint_release_reason": hint_release_reason,
                            "release_reason": "replay_completed",
                            "background_fillers_per_session": args.background_fillers_per_session,
                            "priority_post_prefetch_quiet_ms": args.priority_post_prefetch_quiet_ms,
                        }
                    )
                    write_trace_event(
                        {
                            "event": "m38.deadline_filler_admission.released",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "background_fillers_per_session": args.background_fillers_per_session,
                            "release_reason": "replay_completed",
                        }
                    )
                elif args.mode in {"direct_prefetch", "priority_direct_prefetch"} and hint_task is not None:
                    reserve_start_ms = max(tool_start_offset_ms, replay_due_ms - args.deadline_reserve_window_ms)
                    write_trace_event(
                        {
                            "event": "m38.dynamo_like_low_priority_fillers.held",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "background_fillers_per_session": args.background_fillers_per_session,
                            "priority_prefetch_window_ms": args.priority_prefetch_window_ms,
                            "deadline_reserve_window_ms": args.deadline_reserve_window_ms,
                            "reserve_start_offset_ms": round(reserve_start_ms, 3),
                            "replay_due_offset_ms": round(replay_due_ms, 3),
                            "policy": "hold_low_priority_fillers_while_urgent_hint_and_replay_get_front_of_driver_queue",
                        }
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(hint_task),
                            timeout=max(1, args.priority_prefetch_window_ms) / 1000.0,
                        )
                        release_reason = "prefetch_completed"
                    except asyncio.TimeoutError:
                        release_reason = "priority_window_expired"
                    if args.priority_post_prefetch_quiet_ms > 0:
                        await asyncio.sleep(args.priority_post_prefetch_quiet_ms / 1000.0)
                    write_trace_event(
                        {
                            "event": "m38.dynamo_like_low_priority_fillers.released",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "release_reason": release_reason,
                            "background_fillers_per_session": args.background_fillers_per_session,
                        }
                    )
                filler_tasks = [
                    asyncio.create_task(run_background_filler(pair, filler_idx))
                    for filler_idx in range(args.background_fillers_per_session)
                ]
                if filler_tasks:
                    await asyncio.gather(*filler_tasks, return_exceptions=True)

            filler_group_task: asyncio.Task[None] | None = None
            if args.background_fillers_per_session > 0:
                filler_group_task = asyncio.create_task(run_filler_group())

            await sleep_until(replay_due_ms)
            write_trace_event(
                {
                    "event": "m27.pre_replay.checkpoint",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "prefetch_hint_submitted": bool(hint_task is not None),
                    "expected_reuse": "high" if hint_task is not None else "baseline",
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
            replay_admission_start_ms = (time.perf_counter() - workload_start) * 1000.0
            if args.mode in PRIORITY_PREFETCH_MODES and hint_task is not None:
                if not hint_task.done():
                    write_trace_event(
                        {
                            "event": "m38.deadline_replay_admission.waiting_for_prefetch",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "replay_due_offset_ms": round(replay_due_ms, 3),
                            "wait_start_offset_ms": round(replay_admission_start_ms, 3),
                            "reason": "prefetch_not_finished_at_replay_deadline",
                        }
                    )
                    await hint_task
                    replay_admission_start_ms = (time.perf_counter() - workload_start) * 1000.0
                write_trace_event(
                    {
                        "event": "m38.deadline_replay_admission.start"
                        if args.mode == "deadline_priority_prefetch"
                        else "m38.dynamo_like_replay_admission.start",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "admission_start_offset_ms": round(replay_admission_start_ms, 3),
                        "admission_lateness_ms": round(replay_admission_start_ms - replay_due_ms, 3),
                        "uses_driver_concurrency_gate": False,
                        "policy": "admit_replay_immediately_after_priority_prefetch"
                        if args.mode == "deadline_priority_prefetch"
                        else "front_of_driver_queue_after_direct_prefetch",
                    }
                )
                await run_request(
                    pair,
                    pair.replay_prompt,
                    "replay",
                    f"{pair.session_id}_replay",
                    args.max_tokens,
                    use_concurrency_limit=False,
                )
                replay_admission_end_ms = (time.perf_counter() - workload_start) * 1000.0
                write_trace_event(
                    {
                        "event": "m38.deadline_replay_admission.end"
                        if args.mode == "deadline_priority_prefetch"
                        else "m38.dynamo_like_replay_admission.end",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "admission_end_offset_ms": round(replay_admission_end_ms, 3),
                        "admitted_replay_duration_ms": round(replay_admission_end_ms - replay_admission_start_ms, 3),
                    }
                )
            else:
                await run_request(pair, pair.replay_prompt, "replay", f"{pair.session_id}_replay", args.max_tokens)
            replay_completed_event.set()
            if args.mode == "deadline_priority_prefetch" and hint_task is not None:
                protection_end_ms = (time.perf_counter() - workload_start) * 1000.0
                write_trace_event(
                    {
                        "event": "m38.deadline_residency_protection.end",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "protection_end_offset_ms": round(protection_end_ms, 3),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "release_reason": "replay_completed",
                    }
                )
                write_trace_event(
                    {
                        "event": "m38.deadline_service.end",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "service_end_offset_ms": round(protection_end_ms, 3),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                    }
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
            if filler_group_task is not None:
                await filler_group_task

        await asyncio.gather(*(run_session(idx, spec) for idx, spec in enumerate(session_specs)))

    write_trace_event({"event": "m36.workload_end", "mode": args.mode, "row_count": len(rows)})
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote multi-session metrics to {args.out}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
