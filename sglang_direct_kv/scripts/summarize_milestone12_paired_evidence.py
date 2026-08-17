#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    return float(value)


def as_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return 0
    return int(float(value))


def yes(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip() in {"1", "True", "true", "yes"}


def truthy(value: Any) -> bool:
    return str(value).strip() in {"1", "True", "true", "yes"}


def to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def fmt_ms(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "not attributed"
    return f"{number:.3f} ms"


def yes_no(value: Any) -> str:
    return "yes" if truthy(value) else "no"


def checkpoint_class(value: Any, good_when_yes: bool = True) -> str:
    is_yes = truthy(value)
    good = is_yes if good_when_yes else not is_yes
    return "good" if good else "bad"


def load_clean_rows(root: Path, modes: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        rows = read_csv(root / f"{mode}_outcomes" / "hint_outcomes.csv")
        if rows:
            out[mode] = rows
    return out


def outcome_counts(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("outcome", "unknown")) for row in rows)
    return ", ".join(f"{name}: {count}" for name, count in counts.most_common())


def build_clean_summary(clean_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    baseline_rows = clean_rows.get("no_prefetch", [])
    baseline_avg = mean(as_float(row, "replay_ttft_ms") for row in baseline_rows) if baseline_rows else 0.0
    rows_out: list[dict[str, Any]] = []
    for mode, rows in clean_rows.items():
        replay_ttfts = [as_float(row, "replay_ttft_ms") for row in rows]
        avg_replay = mean(replay_ttfts) if replay_ttfts else 0.0
        improvement = baseline_avg - avg_replay if baseline_avg else 0.0
        rows_out.append(
            {
                "mode": mode,
                "sessions": len(rows),
                "performance_status": "baseline"
                if mode == "no_prefetch"
                else ("faster" if improvement > 0 else "slower" if improvement < 0 else "same"),
                "avg_replay_ttft_ms": round(avg_replay, 3),
                "median_replay_ttft_ms": round(median(replay_ttfts), 3) if replay_ttfts else 0.0,
                "avg_improvement_vs_no_prefetch_ms": round(improvement, 3),
                "avg_improvement_vs_no_prefetch_pct": round(improvement * 100.0 / baseline_avg, 2)
                if baseline_avg
                else 0.0,
                "late_prefetch_sessions": sum(1 for row in rows if row.get("outcome") == "late_prefetch"),
                "reload_or_unprotected_sessions": sum(
                    1
                    for row in rows
                    if row.get("outcome") in {"too_early_or_unprotected", "resume_still_loaded_kv"}
                ),
                "total_resume_load_count": sum(as_int(row, "resume_load_count") for row in rows),
                "total_eviction_pressure_after_prefetch": sum(
                    as_int(row, "eviction_pressure_after_prefetch") for row in rows
                ),
                "outcomes": outcome_counts(rows),
            }
        )
    return rows_out


def build_attribution_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    visible_copy = sum(1 for row in rows if row.get("visible_copy_end_ms") not in ("", None))
    lightweight_copy = sum(1 for row in rows if as_int(row, "telemetry_h2d_copy_events") > 0)
    visible_h2d = sum(1 for row in rows if as_int(row, "torch_h2d_copy_events") > 0)
    kv_ready = sum(1 for row in rows if yes(row, "kv_copy_ready_before_replay"))
    cuda_ready = sum(1 for row in rows if yes(row, "cuda_copy_ready_before_replay"))
    hint_done = sum(1 for row in rows if yes(row, "full_hint_done_before_replay"))
    reloaded = sum(1 for row in rows if yes(row, "replay_reloaded_kv"))
    clean_success = sum(1 for row in rows if row.get("checkpoint_result") == "clean_success")
    return [
        {
            "profiled_sessions": len(rows),
            "evidence_status": "clean_success_seen"
            if clean_success
            else ("copy_visible_but_not_sufficient" if kv_ready and reloaded else "mechanism_visible"),
            "sessions_with_visible_copy_telemetry": visible_copy,
            "sessions_with_lightweight_kv_copy_telemetry": lightweight_copy,
            "sessions_with_profiler_cuda_h2d_copy": visible_h2d,
            "kv_copy_ready_before_replay": kv_ready,
            "cuda_copy_ready_before_replay": cuda_ready,
            "full_hint_done_before_replay": hint_done,
            "replay_reloaded_kv": reloaded,
            "clean_success": clean_success,
            "checkpoint_results": outcome_counts([{"outcome": row.get("checkpoint_result", "unknown")} for row in rows]),
        }
    ]


def build_paired_rows(
    clean_rows: dict[str, list[dict[str, Any]]],
    attribution_rows: list[dict[str, Any]],
    attribution_mode: str,
) -> list[dict[str, Any]]:
    clean_mode_rows = {
        row.get("session_id", ""): row
        for row in clean_rows.get(attribution_mode, [])
        if row.get("session_id")
    }
    baseline_rows = {
        row.get("session_id", ""): row
        for row in clean_rows.get("no_prefetch", [])
        if row.get("session_id")
    }
    rows_out: list[dict[str, Any]] = []
    for attr in attribution_rows:
        session_id = str(attr.get("session_id", ""))
        clean = clean_mode_rows.get(session_id, {})
        baseline = baseline_rows.get(session_id, {})
        clean_ttft = as_float(clean, "replay_ttft_ms") if clean else 0.0
        baseline_ttft = as_float(baseline, "replay_ttft_ms") if baseline else 0.0
        rows_out.append(
            {
                "session_id": session_id,
                "clean_mode": attribution_mode,
                "paired_takeaway": paired_takeaway(clean, baseline, attr),
                "clean_replay_ttft_ms": clean.get("replay_ttft_ms", ""),
                "no_prefetch_replay_ttft_ms": baseline.get("replay_ttft_ms", ""),
                "clean_delta_vs_no_prefetch_ms": round(baseline_ttft - clean_ttft, 3) if baseline and clean else "",
                "clean_hint_outcome": clean.get("outcome", ""),
                "profiled_cuda_copy_ready_before_replay": "yes"
                if yes(attr, "cuda_copy_ready_before_replay")
                else "no",
                "profiled_full_hint_done_before_replay": "yes"
                if yes(attr, "full_hint_done_before_replay")
                else "no",
                "profiled_replay_reloaded_kv": "yes" if yes(attr, "replay_reloaded_kv") else "no",
                "profiled_checkpoint_result": attr.get("checkpoint_result", ""),
                "profiled_kv_copy_ready_before_replay": "yes"
                if yes(attr, "kv_copy_ready_before_replay")
                else "no",
                "profiled_visible_copy_source": attr.get("visible_copy_source", ""),
                "profiled_telemetry_h2d_events": attr.get("telemetry_h2d_copy_events", ""),
                "profiled_h2d_events": attr.get("torch_h2d_copy_events", ""),
                "profiled_h2d_bytes": attr.get("torch_h2d_bytes", ""),
                "profiled_prefetch_margin_ms": attr.get("prefetch_margin_ms", ""),
            }
        )
    return rows_out


def paired_takeaway(clean: dict[str, Any], baseline: dict[str, Any], attr: dict[str, Any]) -> str:
    if not attr:
        return "no profiled row"
    kv_ready = yes(attr, "kv_copy_ready_before_replay")
    cuda_ready = yes(attr, "cuda_copy_ready_before_replay")
    hint_done = yes(attr, "full_hint_done_before_replay")
    reloaded = yes(attr, "replay_reloaded_kv")
    clean_ttft = as_float(clean, "replay_ttft_ms") if clean else 0.0
    baseline_ttft = as_float(baseline, "replay_ttft_ms") if baseline else 0.0
    faster = baseline and clean and clean_ttft < baseline_ttft
    if cuda_ready and hint_done and not reloaded and faster:
        return "clean success"
    if kv_ready and not cuda_ready and reloaded:
        return "KV copy telemetry visible, replay still reloaded"
    if kv_ready and not cuda_ready:
        return "KV copy telemetry visible, CUDA profiler not required"
    if cuda_ready and reloaded:
        return "copy was early, replay still reloaded"
    if cuda_ready and not hint_done:
        return "copy was early, full hint path was late"
    if not cuda_ready and hint_done:
        return "hint finished, CUDA copy was not clearly attributed"
    if clean and baseline and faster:
        return "clean TTFT improved, mechanism still unclear"
    return "no clean prefetch win"


def load_timeline_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    timeline = data.get("timeline", [])
    return timeline if isinstance(timeline, list) else []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ns_to_ms(ts_ns: int, start_ns: int) -> float:
    return round((ts_ns - start_ns) / 1_000_000, 3)


def first_agent_event(
    events: list[dict[str, Any]],
    name: str,
    session_id: str,
    phase: str | None = None,
) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") != name or event.get("session_id") != session_id:
            continue
        if phase is not None and event.get("phase") != phase:
            continue
        return event
    return None


def request_window_ms(
    events: list[dict[str, Any]],
    session_id: str,
    phase: str,
    trace_start_ns: int,
) -> tuple[float | None, float | None]:
    start = first_agent_event(events, "agent.request.start", session_id, phase)
    end = first_agent_event(events, "agent.request.end", session_id, phase)
    if not start or not end:
        return None, None
    return ns_to_ms(int(start["ts_ns"]), trace_start_ns), ns_to_ms(int(end["ts_ns"]), trace_start_ns)


def event_ms(
    events: list[dict[str, Any]],
    name: str,
    session_id: str,
    trace_start_ns: int,
) -> float | None:
    event = first_agent_event(events, name, session_id)
    if not event or not event.get("ts_ns"):
        return None
    return ns_to_ms(int(event["ts_ns"]), trace_start_ns)


def request_metric(events: list[dict[str, Any]], session_id: str, phase: str, key: str) -> Any:
    event = first_agent_event(events, "agent.request.end", session_id, phase)
    if not event:
        return ""
    return event.get(key, "")


def load_clean_timelines(
    clean_root: Path,
    modes: list[str],
    clean_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    timelines: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for mode in modes:
        trace_path = clean_root / f"{mode}_traffic_trace.jsonl"
        events = read_jsonl(trace_path)
        if not events:
            continue
        trace_start_ns = min(int(event["ts_ns"]) for event in events if event.get("ts_ns"))
        outcome_by_session = {
            str(row.get("session_id", "")): row
            for row in clean_rows.get(mode, [])
            if row.get("session_id")
        }
        session_ids = sorted(
            {
                str(event["session_id"])
                for event in events
                if isinstance(event.get("session_id"), str) and str(event.get("session_id")).startswith("agent_")
            }
        )
        rows: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []

        def add_bar(
            session_id: str,
            kind: str,
            start: float | None,
            end: float | None,
            label: str,
        ) -> None:
            if start is None or end is None:
                return
            timeline.append(
                {
                    "session_id": session_id,
                    "kind": kind,
                    "start_ms": round(float(start), 3),
                    "end_ms": round(float(end), 3),
                    "label": label,
                }
            )

        def add_marker(session_id: str, kind: str, at: float | None, label: str) -> None:
            if at is None:
                return
            timeline.append(
                {
                    "session_id": session_id,
                    "kind": kind,
                    "start_ms": round(float(at), 3),
                    "end_ms": round(float(at), 3),
                    "label": label,
                }
            )

        for session_id in session_ids:
            arrival = first_agent_event(events, "agent.session_arrival", session_id) or {}
            initial_start, initial_end = request_window_ms(events, session_id, "initial_turn", trace_start_ns)
            hint_start, hint_end = request_window_ms(events, session_id, "hint_prefetch", trace_start_ns)
            replay_start, replay_end = request_window_ms(events, session_id, "replay", trace_start_ns)
            tool_wait_start = event_ms(events, "agent.tool_wait_start", session_id, trace_start_ns)
            hint_submitted = event_ms(events, "agent.hint_submitted", session_id, trace_start_ns)
            replay_due = event_ms(events, "agent.replay_due", session_id, trace_start_ns)
            outcome = outcome_by_session.get(session_id, {})
            replay_delay = (
                round(float(replay_start) - float(replay_due), 3)
                if replay_start is not None and replay_due is not None
                else ""
            )
            replay_ttft = to_float(request_metric(events, session_id, "replay", "ttft_ms"))
            first_token_ms = (
                round(float(replay_start) + replay_ttft, 3)
                if replay_start is not None and replay_ttft is not None
                else None
            )
            effective_wait_after_due = (
                round(float(replay_delay) + replay_ttft, 3)
                if replay_delay != "" and replay_ttft is not None
                else ""
            )
            hint_done_before_due = (
                "yes"
                if hint_end is not None and replay_due is not None and hint_end <= replay_due
                else "no"
                if hint_end is not None and replay_due is not None
                else ""
            )
            rows.append(
                {
                    "mode": mode,
                    "session_id": session_id,
                    "priority": arrival.get("priority", ""),
                    "prompt_tokens": arrival.get("prompt_tokens", ""),
                    "tool_wait_ms": arrival.get("tool_wait_ms", ""),
                    "outcome": outcome.get("outcome", "no_hint" if mode == "no_prefetch" else ""),
                    "initial_ttft_ms": request_metric(events, session_id, "initial_turn", "ttft_ms"),
                    "hint_ttft_ms": request_metric(events, session_id, "hint_prefetch", "ttft_ms"),
                    "replay_ttft_ms": replay_ttft if replay_ttft is not None else "",
                    "replay_due_ms": replay_due if replay_due is not None else "",
                    "replay_start_ms": replay_start if replay_start is not None else "",
                    "replay_delay_after_due_ms": replay_delay,
                    "first_token_ms": first_token_ms if first_token_ms is not None else "",
                    "effective_wait_after_due_ms": effective_wait_after_due,
                    "hint_done_before_replay_due": hint_done_before_due,
                }
            )
            add_bar(session_id, "initial", initial_start, initial_end, "initial request")
            add_bar(session_id, "tool_wait", tool_wait_start, replay_due, "tool wait")
            add_marker(session_id, "hint_submitted", hint_submitted, "hint submitted")
            add_bar(session_id, "hint_request", hint_start, hint_end, "hint request")
            add_marker(session_id, "replay_due", replay_due, "replay due")
            add_bar(session_id, "replay", replay_start, replay_end, "replay request")
            add_marker(session_id, "first_token", first_token_ms, "first token")
        timelines[mode] = {"rows": rows, "timeline": timeline}
    return timelines


def choose_timeline_sessions(rows: list[dict[str, Any]], max_sessions: int) -> list[str]:
    late = [row for row in rows if truthy(row.get("late_prefetch"))]
    visible = [row for row in rows if row.get("visible_copy_end_ms") not in ("", None)]
    ordered: list[str] = []
    for group in (late, visible, rows):
        for row in sorted(group, key=lambda item: to_float(item.get("prefetch_margin_ms")) or 1_000_000):
            sid = str(row.get("session_id", ""))
            if sid and sid not in ordered:
                ordered.append(sid)
            if len(ordered) >= max_sessions:
                return ordered
    return ordered


def timeline_status(row: dict[str, Any]) -> tuple[str, str]:
    margin = to_float(row.get("prefetch_margin_ms"))
    checkpoint = row.get("checkpoint_result")
    if checkpoint == "no_prefetch_needed":
        return "NO LOAD", "#6b7280"
    if checkpoint == "clean_success":
        return f"CLEAN +{margin:.0f} ms" if margin is not None else "CLEAN", "#166534"
    if checkpoint == "copy_ready_but_replay_reloaded":
        return f"RELOAD +{margin:.0f} ms" if margin is not None else "RELOAD", "#92400e"
    if checkpoint == "copy_ready_but_hint_not_done":
        return f"HINT LATE +{margin:.0f} ms" if margin is not None else "HINT LATE", "#b45309"
    if truthy(row.get("late_prefetch")) and margin is not None:
        return f"LATE {margin:.0f} ms", "#b91c1c"
    if as_int(row, "torch_h2d_copy_events") > 0 and margin is not None:
        return f"SUCCESS +{margin:.0f} ms", "#166534"
    if as_int(row, "telemetry_h2d_copy_events") > 0 and margin is not None:
        return f"KV TRACE {margin:.0f} ms", "#15803d" if margin >= 0 else "#b91c1c"
    if as_int(row, "sglang_copy_events") > 0 and margin is not None:
        return f"SGLang OK +{margin:.0f} ms", "#92400e"
    return "NO VISIBLE COPY", "#6b7280"


def session_observation(row: dict[str, Any]) -> tuple[str, str, str]:
    margin = to_float(row.get("prefetch_margin_ms"))
    torch_copy_events = as_int(row, "torch_h2d_copy_events")
    telemetry_copy_events = as_int(row, "telemetry_h2d_copy_events")
    sglang_events = as_int(row, "sglang_copy_events")
    missing_reason = str(row.get("h2d_missing_reason") or "")
    cuda_ready = yes_no(row.get("cuda_copy_ready_before_replay"))
    kv_ready = yes_no(row.get("kv_copy_ready_before_replay"))
    hint_done = yes_no(row.get("full_hint_done_before_replay"))
    replay_reloaded = yes_no(row.get("replay_reloaded_kv"))

    if truthy(row.get("no_visible_prefetch")):
        status = "No visible prefetch"
        observation = "The trace did not show a SGLang KV load or CUDA HtoD copy for this hint."
        deduction = "This session is weak evidence for movement timing; use it mainly to show missing visibility."
    elif truthy(row.get("late_prefetch")) and margin is not None:
        status = "Late prefetch"
        if missing_reason == "profiler_stopped_before_kv_load":
            observation = (
                f"SGLang KV movement completed {abs(margin):.3f} ms after replay was already due. "
                "CUDA HtoD is not shown because torch.profiler had already stopped before this KV window."
            )
            deduction = "This is a real late SGLang prefetch, but the missing green bar is a profiler-coverage issue, not proof that no CUDA copy happened."
        else:
            observation = f"The hinted KV movement completed {abs(margin):.3f} ms after replay was already due."
            deduction = "The software hint existed, but the normal serving/KV path did not act early enough."
    elif torch_copy_events > 0 and margin is not None:
        status = "Clean useful prefetch"
        observation = f"SGLang KV load and CUDA HtoD copies were visible, and movement completed {margin:.3f} ms before replay."
        deduction = "This is the clean success case for movement timing: the hint moved KV early enough for the agent replay."
    elif telemetry_copy_events > 0 and margin is not None:
        status = "Lightweight KV telemetry visible"
        observation = f"SGLang KV-copy telemetry was visible and movement completed with margin {margin:.3f} ms."
        deduction = "This is scalable movement evidence from the exact SGLang KV load path; use torch-profiler runs separately to validate CUDA-level mapping."
    elif sglang_events > 0 and margin is not None:
        status = "SGLang-level useful prefetch"
        if missing_reason:
            observation = (
                f"SGLang KV load was visible and completed {margin:.3f} ms before replay, "
                f"but CUDA HtoD attribution is missing because: {missing_reason}."
            )
        else:
            observation = f"SGLang KV load was visible and completed {margin:.3f} ms before replay, but CUDA HtoD rows were not confidently attributed to this session."
        deduction = "This is useful SGLang-level evidence, but weaker CUDA-level evidence."
    else:
        status = "Incomplete timing"
        observation = "Some required timing fields were missing."
        deduction = "Do not use this row as a strong success or failure example."

    evidence = (
        f"tool wait {fmt_ms(row.get('tool_wait_ms'))}; "
        f"KV copy ready before replay {kv_ready}; "
        f"CUDA copy ready before replay {cuda_ready}; "
        f"full hint done before replay {hint_done}; "
        f"replay reloaded KV {replay_reloaded}; "
        f"SGLang load {fmt_ms(row.get('sglang_copy_start_ms'))} -> {fmt_ms(row.get('sglang_copy_end_ms'))}; "
        f"lightweight KV copy {fmt_ms(row.get('telemetry_copy_start_ms'))} -> {fmt_ms(row.get('telemetry_copy_end_ms'))}; "
        f"CUDA HtoD {fmt_ms(row.get('torch_copy_start_ms'))} -> {fmt_ms(row.get('torch_copy_end_ms'))}; "
        f"profiler window {fmt_ms(row.get('profiler_start_ms'))} -> {fmt_ms(row.get('profiler_end_ms'))}; "
        f"HtoD missing reason {fmt(row.get('h2d_missing_reason')) or 'none'}; "
        f"replay due {fmt_ms(row.get('replay_due_ms'))}; "
        f"margin {fmt_ms(row.get('prefetch_margin_ms'))}"
    )
    return status, observation, deduction + " " + evidence


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows.", ""]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.append("")
    return lines


def write_md(path: Path, sections: dict[str, list[dict[str, Any]]], metadata: dict[str, Any]) -> None:
    lines = [
        "# Milestone 12 Paired Evidence Report",
        "",
        "This report separates clean performance evidence from profiled attribution evidence.",
        "",
        "Use the clean run for TTFT/performance claims. Use the profiled run for CUDA HtoD/KV mechanism claims.",
        "",
        "## Manager Summary",
        "",
        *manager_summary_lines(sections),
        "",
        "## How To Read This Report",
        "",
        "- Clean performance rows come from runs with `torch.profiler` off. These are the right numbers for TTFT claims.",
        "- Profiled attribution rows come from runs with `torch.profiler` on. These are the right rows for mechanism evidence.",
        "- If CUDA copy is ready but replay reloads KV anyway, the copy happened but software did not preserve/reuse it predictably.",
        "",
        "## Key Deductions",
        "",
        *key_deduction_lines(sections),
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
    ]
    for title, rows in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(md_table(rows))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_timeline_md(
    path: Path,
    attribution_rows: list[dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
    max_timeline_sessions: int,
) -> None:
    if not attribution_rows:
        return
    selected_ids = choose_timeline_sessions(attribution_rows, max_timeline_sessions)
    row_by_session = {str(row.get("session_id", "")): row for row in attribution_rows}
    selected_rows = [row_by_session[sid] for sid in selected_ids if sid in row_by_session]
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.extend(
        [
            "",
            "## Timeline Summary",
            "",
            "This is the profiled mechanism view. The HTML report includes the visual timeline.",
            "",
            *md_table(timeline_summary_rows(attribution_rows)),
            "## Timeline Layers",
            "",
            *md_table(timeline_layers_rows()),
            "## Timeline Sanity Checks",
            "",
            "Green copy windows should usually be inside the purple hint request. Hint/replay overlap means the software hint path was still running when replay arrived.",
            "",
            *md_table(timeline_sanity_rows(selected_rows)),
            "## Prefetch Checkpoints",
            "",
            *md_table(prefetch_checkpoint_rows()),
            "## Checkpoint Results Per Session",
            "",
            *md_table(checkpoint_result_rows(selected_rows)),
            "## Key Observations Per Session",
            "",
            *md_table(key_observation_rows(selected_rows)),
            "## Session Details",
            "",
            *md_table(session_detail_rows(selected_rows)),
        ]
    )
    if timeline_rows:
        lines.extend(["", "Timeline rows are available in the paired report JSON and the HTML report."])
    path.write_text("\n".join(lines), encoding="utf-8")


def html_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    headers = list(rows[0].keys())
    out = ["<table><thead><tr>"]
    for header in headers:
        out.append(f"<th>{html.escape(header)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for header in headers:
            value = str(row.get(header, ""))
            cls = ""
            if value == "yes":
                cls = ' class="good"'
            elif value == "no":
                cls = ' class="bad"'
            elif value in {"faster", "clean success", "clean_success_seen"}:
                cls = ' class="good"'
            elif value in {
                "slower",
                "copy was early, replay still reloaded",
                "copy was early, full hint path was late",
                "KV copy telemetry visible, replay still reloaded",
            }:
                cls = ' class="warn"'
            out.append(f"<td{cls}>{html.escape(value)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def choose_clean_timeline_sessions(rows: list[dict[str, Any]], max_sessions: int) -> list[str]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -(to_float(row.get("replay_delay_after_due_ms")) or 0.0),
            -(to_float(row.get("replay_ttft_ms")) or 0.0),
            str(row.get("session_id", "")),
        ),
    )
    return [str(row.get("session_id", "")) for row in sorted_rows[:max_sessions] if row.get("session_id")]


def clean_timeline_status(row: dict[str, Any]) -> tuple[str, str]:
    outcome = str(row.get("outcome", ""))
    ttft = to_float(row.get("replay_ttft_ms"))
    delay = to_float(row.get("replay_delay_after_due_ms"))
    if outcome == "no_hint":
        return f"TTFT {ttft:.0f} ms" if ttft is not None else "baseline", "#374151"
    if delay is not None and delay > 5:
        return f"delayed +{delay:.0f} ms", "#b91c1c"
    if outcome:
        return outcome.replace("_", " "), "#92400e"
    return f"TTFT {ttft:.0f} ms" if ttft is not None else "clean run", "#374151"


def build_clean_timeline_svg(
    mode: str,
    rows: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    max_sessions: int,
) -> tuple[str, list[dict[str, Any]]]:
    selected = choose_clean_timeline_sessions(rows, max_sessions)
    row_by_session = {str(row.get("session_id", "")): row for row in rows}
    selected_rows = [row_by_session[sid] for sid in selected if sid in row_by_session]
    selected_timeline = [item for item in timeline if item.get("session_id") in selected]
    focus_values: list[float] = []
    start_values: list[float] = []
    for item in selected_timeline:
        kind = str(item.get("kind", ""))
        item_start = float(item["start_ms"])
        item_end = float(item["end_ms"])
        if kind in {"initial", "tool_wait", "hint_submitted", "hint_request", "replay_due", "first_token"}:
            start_values.append(item_start)
            focus_values.extend([item_start, item_end])
        elif kind == "replay":
            focus_values.append(item_start)
    for row in selected_rows:
        for key in ("replay_due_ms", "replay_start_ms", "first_token_ms"):
            value = to_float(row.get(key))
            if value is not None:
                focus_values.append(value)
    if focus_values:
        start = min(start_values or focus_values) - 120.0
        end = max(focus_values) + 500.0
    else:
        start, end = 0.0, 1.0
    span = max(1.0, end - start)
    width = 1500
    left = 225
    right = 55
    row_h = 72
    top = 78
    height = top + row_h * max(1, len(selected)) + 92
    plot_w = width - left - right
    colors = {
        "initial": "#2563eb",
        "tool_wait": "#d1d5db",
        "hint_submitted": "#7c3aed",
        "hint_request": "#a855f7",
        "replay_due": "#111827",
        "replay": "#dc2626",
        "first_token": "#f59e0b",
    }

    def x_pos(ms: float) -> float:
        return left + (ms - start) / span * plot_w

    def x_clamped(ms: float) -> float:
        return max(left, min(left + plot_w, x_pos(ms)))

    def layer_order(item: dict[str, Any]) -> int:
        order = {
            "tool_wait": 0,
            "initial": 1,
            "hint_submitted": 2,
            "hint_request": 3,
            "replay_due": 4,
            "replay": 5,
            "first_token": 6,
        }
        return order.get(str(item.get("kind", "")), 10)

    row_index = {sid: idx for idx, sid in enumerate(selected)}
    svg: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Clean performance timeline for {html.escape(mode)}">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    for tick in range(6):
        ms = start + span * tick / 5
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 30}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{ms:.0f} ms</text>')
    for sid in selected:
        y = top + row_index[sid] * row_h
        row = row_by_session.get(sid, {})
        status_label, status_color = clean_timeline_status(row)
        ttft = to_float(row.get("replay_ttft_ms"))
        effective_wait = to_float(row.get("effective_wait_after_due_ms"))
        ttft_label = f"replay TTFT {ttft:.0f} ms" if ttft is not None else "replay TTFT n/a"
        if effective_wait is not None:
            ttft_label += f"; due->token {effective_wait:.0f} ms"
        svg.append(f'<text x="10" y="{y + 15}" font-weight="700">{html.escape(sid)}</text>')
        svg.append(f'<text x="10" y="{y + 36}" font-size="13" fill="{status_color}" font-weight="700">{html.escape(status_label)}</text>')
        svg.append(f'<text x="10" y="{y + 55}" font-size="12" fill="#4b5563">{html.escape(ttft_label)}</text>')
        svg.append(f'<line x1="{left}" y1="{y + 12}" x2="{left + plot_w}" y2="{y + 12}" stroke="#f3f4f6"/>')

    for item in sorted(selected_timeline, key=layer_order):
        sid = str(item.get("session_id", ""))
        if sid not in row_index:
            continue
        y = top + row_index[sid] * row_h
        kind = str(item.get("kind", ""))
        color = colors.get(kind, "#6b7280")
        raw_start_ms = float(item.get("start_ms", 0.0))
        raw_end_ms = float(item.get("end_ms", 0.0))
        x1 = x_clamped(raw_start_ms)
        x2 = x_clamped(raw_end_ms)
        label = html.escape(str(item.get("label", kind)))
        if raw_start_ms == raw_end_ms:
            stroke_width = 6 if kind == "replay_due" else 4 if kind == "first_token" else 3
            dash = ' stroke-dasharray="5 3"' if kind == "first_token" else ""
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y + 1}" x2="{x1:.1f}" y2="{y + 34}" stroke="{color}" stroke-width="{stroke_width}"{dash}><title>{label}</title></line>'
            )
            if kind == "first_token":
                row = row_by_session.get(sid, {})
                ttft = to_float(row.get("replay_ttft_ms"))
                text = f"first token +{ttft:.0f} ms" if ttft is not None else "first token"
                svg.append(f'<text x="{x1:.1f}" y="{y + 44}" text-anchor="middle" font-size="10" fill="#92400e" font-weight="700">{html.escape(text)}</text>')
        else:
            display_x1 = x1
            display_x2 = x2
            bar_y = y + 4
            bar_h = 24
            opacity = "0.72" if kind in {"hint_request", "replay"} else "0.88"
            replay_continues = kind == "replay" and x_pos(raw_end_ms) > left + plot_w
            svg.append(
                f'<rect x="{display_x1:.1f}" y="{bar_y}" width="{max(2, display_x2 - display_x1):.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="{opacity}"><title>{label}</title></rect>'
            )
            if kind == "hint_request":
                svg.append(
                    f'<text x="{x1:.1f}" y="{bar_y - 6}" text-anchor="middle" font-size="9" fill="#6d28d9" font-weight="700">hint start</text>'
                )
                svg.append(
                    f'<text x="{x2:.1f}" y="{bar_y + bar_h + 13}" text-anchor="middle" font-size="9" fill="#6d28d9" font-weight="700">hint end</text>'
                )
            if replay_continues:
                arrow_x = left + plot_w - 7
                arrow_y = bar_y + bar_h / 2
                svg.append(
                    f'<path d="M {arrow_x - 7:.1f} {arrow_y - 7:.1f} L {arrow_x:.1f} {arrow_y:.1f} L {arrow_x - 7:.1f} {arrow_y + 7:.1f}" '
                    'fill="none" stroke="#991b1b" stroke-width="2"><title>replay continues beyond focused window</title></path>'
                )
                svg.append(f'<text x="{left + plot_w - 82:.1f}" y="{bar_y - 4}" font-size="10" fill="#991b1b" font-weight="700">continues</text>')

    legend_x = left
    legend_y = height - 30
    legend_labels = {
        "initial": "initial",
        "tool_wait": "tool wait",
        "hint_submitted": "hint submitted",
        "hint_request": "hint request",
        "replay_due": "replay due",
        "replay": "replay",
        "first_token": "first token",
    }
    for idx, (kind, color) in enumerate(colors.items()):
        lx = legend_x + idx * 130
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}">{html.escape(legend_labels.get(kind, kind))}</text>')
    svg.append("</svg>")
    return "\n".join(svg), selected_rows


def build_expanded_clean_timeline_svg(
    mode: str,
    rows: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    max_sessions: int,
) -> tuple[str, list[dict[str, Any]]]:
    selected = choose_clean_timeline_sessions(rows, max_sessions)
    row_by_session = {str(row.get("session_id", "")): row for row in rows}
    selected_rows = [row_by_session[sid] for sid in selected if sid in row_by_session]
    selected_timeline = [item for item in timeline if item.get("session_id") in selected]
    if not selected_rows:
        return "<p>No expanded clean timeline data was found.</p>", []

    rel_values: list[float] = []
    due_by_session = {
        sid: to_float(row_by_session.get(sid, {}).get("replay_due_ms"))
        for sid in selected
    }
    for item in selected_timeline:
        sid = str(item.get("session_id", ""))
        due = due_by_session.get(sid)
        if due is None:
            continue
        kind = str(item.get("kind", ""))
        start_ms = to_float(item.get("start_ms"))
        end_ms = to_float(item.get("end_ms"))
        if start_ms is None or end_ms is None:
            continue
        if kind == "replay":
            rel_values.append(start_ms - due)
            rel_values.append(min(600.0, end_ms - due))
        else:
            rel_values.extend([start_ms - due, end_ms - due])
    start = min(rel_values or [-500.0])
    end = max(rel_values or [500.0])
    start = min(start - 60.0, -120.0)
    end = max(end + 80.0, 220.0)
    span = max(1.0, end - start)
    width = 1500
    left = 250
    right = 60
    row_h = 76
    top = 86
    height = top + row_h * max(1, len(selected)) + 92
    plot_w = width - left - right
    colors = {
        "initial": "#2563eb",
        "tool_wait": "#d1d5db",
        "hint_submitted": "#7c3aed",
        "hint_request": "#a855f7",
        "replay_due": "#111827",
        "replay": "#dc2626",
        "first_token": "#f59e0b",
    }

    def x_pos(relative_ms: float) -> float:
        return left + (relative_ms - start) / span * plot_w

    def rel(item: dict[str, Any], due: float, key: str) -> float | None:
        value = to_float(item.get(key))
        if value is None:
            return None
        return value - due

    def layer_order(item: dict[str, Any]) -> int:
        order = {
            "tool_wait": 0,
            "initial": 1,
            "hint_submitted": 2,
            "hint_request": 3,
            "replay_due": 4,
            "replay": 5,
            "first_token": 6,
        }
        return order.get(str(item.get("kind", "")), 10)

    zero_x = x_pos(0.0)
    row_index = {sid: idx for idx, sid in enumerate(selected)}
    svg: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Expanded clean performance timeline for {html.escape(mode)}">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    for tick in range(7):
        ms = start + span * tick / 6
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 42}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{ms:.0f} ms</text>')
    svg.append(f'<line x1="{zero_x:.1f}" y1="{top - 40}" x2="{zero_x:.1f}" y2="{height - 42}" stroke="#111827" stroke-width="3"/>')
    svg.append(f'<text x="{zero_x + 7:.1f}" y="{top - 48}" font-size="12" font-weight="700">0 ms replay due</text>')
    svg.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="13" font-weight="700">local time around each synthetic gap: negative = before replay due, positive = after replay due</text>'
    )

    for sid in selected:
        y = top + row_index[sid] * row_h
        row = row_by_session.get(sid, {})
        status_label, status_color = clean_timeline_status(row)
        ttft = to_float(row.get("replay_ttft_ms"))
        wait = to_float(row.get("tool_wait_ms"))
        svg.append(f'<text x="10" y="{y + 16}" font-weight="700">{html.escape(sid)}</text>')
        svg.append(f'<text x="10" y="{y + 36}" font-size="13" fill="{status_color}" font-weight="700">{html.escape(status_label)}</text>')
        detail = []
        if wait is not None:
            detail.append(f"wait={wait:.0f} ms")
        if ttft is not None:
            detail.append(f"TTFT={ttft:.0f} ms")
        svg.append(f'<text x="10" y="{y + 55}" font-size="12" fill="#4b5563">{html.escape("; ".join(detail))}</text>')
        svg.append(f'<line x1="{left}" y1="{y + 9}" x2="{left + plot_w}" y2="{y + 9}" stroke="#f3f4f6"/>')

    for item in sorted(selected_timeline, key=layer_order):
        sid = str(item.get("session_id", ""))
        due = due_by_session.get(sid)
        if due is None or sid not in row_index:
            continue
        y = top + row_index[sid] * row_h
        kind = str(item.get("kind", ""))
        color = colors.get(kind, "#6b7280")
        start_rel = rel(item, due, "start_ms")
        end_rel = rel(item, due, "end_ms")
        if start_rel is None or end_rel is None:
            continue
        x1 = x_pos(start_rel)
        x2 = x_pos(end_rel)
        label = html.escape(str(item.get("label", kind)))
        if kind == "replay":
            clipped_end_rel = min(end_rel, start_rel + 520.0)
            x2 = x_pos(clipped_end_rel)
        if start_rel == end_rel:
            stroke_width = 6 if kind == "replay_due" else 4 if kind == "first_token" else 3
            dash = ' stroke-dasharray="5 3"' if kind == "first_token" else ""
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y + 1}" x2="{x1:.1f}" y2="{y + 34}" stroke="{color}" stroke-width="{stroke_width}"{dash}><title>{label}</title></line>'
            )
            if kind == "first_token":
                row = row_by_session.get(sid, {})
                ttft = to_float(row.get("replay_ttft_ms"))
                text = f"first token +{ttft:.0f} ms" if ttft is not None else "first token"
                svg.append(f'<text x="{x1:.1f}" y="{y + 47}" text-anchor="middle" font-size="10" fill="#92400e" font-weight="700">{html.escape(text)}</text>')
        else:
            bar_y = y + 4
            bar_h = 24
            opacity = "0.72" if kind in {"hint_request", "replay"} else "0.88"
            svg.append(
                f'<rect x="{x1:.1f}" y="{bar_y}" width="{max(2, x2 - x1):.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="{opacity}"><title>{label}</title></rect>'
            )
            if kind == "hint_request":
                svg.append(f'<text x="{x1:.1f}" y="{bar_y - 6}" text-anchor="middle" font-size="9" fill="#6d28d9" font-weight="700">hint start</text>')
                svg.append(f'<text x="{x2:.1f}" y="{bar_y + bar_h + 13}" text-anchor="middle" font-size="9" fill="#6d28d9" font-weight="700">hint end</text>')
            if kind == "replay" and end_rel > start_rel + 520.0:
                svg.append(f'<text x="{x2 - 55:.1f}" y="{bar_y - 4}" font-size="10" fill="#991b1b" font-weight="700">continues</text>')

    legend_x = left
    legend_y = height - 50
    legend_labels = {
        "initial": "initial",
        "tool_wait": "tool wait",
        "hint_submitted": "hint submitted",
        "hint_request": "hint request",
        "replay_due": "replay due",
        "replay": "replay",
        "first_token": "first token",
    }
    for idx, (kind, color) in enumerate(colors.items()):
        lx = legend_x + idx * 130
        if kind == "replay_due":
            svg.append(f'<line x1="{lx}" y1="{legend_y - 12}" x2="{lx}" y2="{legend_y + 4}" stroke="{color}" stroke-width="4"/>')
        else:
            svg.append(f'<rect x="{lx}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y}">{html.escape(legend_labels.get(kind, kind))}</text>')
    svg.append("</svg>")
    return "\n".join(svg), selected_rows


def build_timeline_svg(
    rows: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    max_sessions: int,
) -> tuple[str, list[dict[str, Any]]]:
    selected = choose_timeline_sessions(rows, max_sessions)
    row_by_session = {str(row.get("session_id", "")): row for row in rows}
    selected_rows = [row_by_session[sid] for sid in selected if sid in row_by_session]
    raw_selected_timeline = [item for item in timeline if item.get("session_id") in selected]
    copy_preference = {"torch_copy": 3, "telemetry_copy": 2, "sglang_copy": 1}
    best_copy_by_session: dict[str, dict[str, Any]] = {}
    for item in raw_selected_timeline:
        kind = str(item.get("kind", ""))
        if kind not in copy_preference:
            continue
        sid = str(item.get("session_id", ""))
        current = best_copy_by_session.get(sid)
        if current is None or copy_preference[kind] > copy_preference[str(current.get("kind", ""))]:
            best_copy_by_session[sid] = item

    selected_timeline: list[dict[str, Any]] = []
    for item in raw_selected_timeline:
        kind = str(item.get("kind", ""))
        sid = str(item.get("session_id", ""))
        if kind in copy_preference:
            if best_copy_by_session.get(sid) is not item:
                continue
            normalized = dict(item)
            normalized["kind"] = "copy_activity"
            if kind == "torch_copy":
                normalized["copy_source"] = "torch_profiler_h2d"
                normalized["label"] = "CUDA HtoD copy"
            elif kind == "telemetry_copy":
                normalized["copy_source"] = "sglang_lightweight_h2d_telemetry"
                normalized["label"] = "SGLang KV telemetry"
            else:
                normalized["copy_source"] = "sglang_kv_load"
                normalized["label"] = "SGLang KV load"
            selected_timeline.append(normalized)
        else:
            selected_timeline.append(item)
    focus_values: list[float] = []
    start_values: list[float] = []
    for item in selected_timeline:
        kind = str(item.get("kind", ""))
        item_start = float(item["start_ms"])
        item_end = float(item["end_ms"])
        if kind in {"initial", "tool_wait", "hint_submitted", "hint_request", "copy_activity", "replay_due"}:
            start_values.append(item_start)
            focus_values.extend([item_start, item_end])
        elif kind == "replay":
            focus_values.append(item_start)
    for row in selected_rows:
        for key in ("hint_request_end_ms", "visible_copy_end_ms", "replay_due_ms", "replay_start_ms"):
            value = to_float(row.get(key))
            if value is not None:
                focus_values.append(value)
    if focus_values:
        start = min(start_values or focus_values) - 120.0
        end = max(focus_values) + 500.0
    else:
        start, end = 0.0, 1.0
    span = max(1.0, end - start)
    width = 1500
    left = 225
    right = 55
    row_h = 84
    top = 78
    height = top + row_h * max(1, len(selected)) + 104
    plot_w = width - left - right
    colors = {
        "initial": "#2563eb",
        "tool_wait": "#d1d5db",
        "hint_submitted": "#7c3aed",
        "hint_request": "#a855f7",
        "copy_activity": "#16a34a",
        "replay_due": "#111827",
        "replay": "#dc2626",
    }

    def x_pos(ms: float) -> float:
        return left + (ms - start) / span * plot_w

    def x_clamped(ms: float) -> float:
        return max(left, min(left + plot_w, x_pos(ms)))

    def layer_order(item: dict[str, Any]) -> int:
        order = {
            "tool_wait": 0,
            "initial": 1,
            "hint_submitted": 2,
            "hint_request": 3,
            "copy_activity": 4,
            "replay_due": 5,
            "replay": 6,
        }
        return order.get(str(item.get("kind", "")), 10)

    row_index = {sid: idx for idx, sid in enumerate(selected)}
    svg: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Agentic prefetch timeline">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    for tick in range(6):
        ms = start + span * tick / 5
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 30}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{ms:.0f} ms</text>')
    for sid in selected:
        y = top + row_index[sid] * row_h
        row = row_by_session.get(sid, {})
        status_label, status_color = timeline_status(row)
        svg.append(f'<text x="10" y="{y + 15}" font-weight="700">{html.escape(sid)}</text>')
        svg.append(
            f'<text x="10" y="{y + 36}" font-size="13" fill="{status_color}" font-weight="700">{html.escape(status_label)}</text>'
        )
        overlap_ms = as_float(row, "hint_replay_overlap_ms")
        if overlap_ms > 0:
            svg.append(f'<text x="10" y="{y + 57}" font-size="12" fill="#b91c1c" font-weight="700">hint/replay overlap {overlap_ms:.0f} ms</text>')
        svg.append(f'<line x1="{left}" y1="{y + 12}" x2="{left + plot_w}" y2="{y + 12}" stroke="#f3f4f6"/>')
        svg.append(f'<line x1="{left}" y1="{y + 48}" x2="{left + plot_w}" y2="{y + 48}" stroke="#f9fafb"/>')
        prefetch_done = to_float(row.get("visible_copy_end_ms")) or to_float(row.get("sglang_copy_end_ms"))
        replay_due = to_float(row.get("replay_due_ms"))
        margin = to_float(row.get("prefetch_margin_ms"))
        hint_start_for_overlap = to_float(row.get("hint_request_start_ms"))
        hint_end_for_overlap = to_float(row.get("hint_request_end_ms"))
        replay_start_for_overlap = to_float(row.get("replay_start_ms"))
        replay_end_for_overlap = to_float(row.get("replay_end_ms"))
        if (
            hint_start_for_overlap is not None
            and hint_end_for_overlap is not None
            and replay_start_for_overlap is not None
            and replay_end_for_overlap is not None
        ):
            overlap_start = max(hint_start_for_overlap, replay_start_for_overlap)
            overlap_end = min(hint_end_for_overlap, replay_end_for_overlap)
            if overlap_end > overlap_start:
                overlap_x1 = x_clamped(overlap_start)
                overlap_x2 = x_clamped(overlap_end)
                svg.append(
                    f'<rect x="{overlap_x1:.1f}" y="{y + 3}" width="{max(2, overlap_x2 - overlap_x1):.1f}" '
                    'height="40" fill="#fecaca" opacity="0.55"><title>hint and replay overlap</title></rect>'
                )
        if prefetch_done is not None and replay_due is not None and margin is not None:
            x_done = x_clamped(prefetch_done)
            x_due = x_clamped(replay_due)
            y_margin = y + 48
            y_label = y + 70
            if margin >= 0:
                svg.append(
                    f'<line x1="{x_done:.1f}" y1="{y_margin}" x2="{x_due:.1f}" y2="{y_margin}" '
                    'stroke="#16a34a" stroke-width="4" stroke-dasharray="8 5"/>'
                )
                svg.append(
                    f'<circle cx="{x_done:.1f}" cy="{y_margin}" r="5" fill="#16a34a"><title>prefetch done</title></circle>'
                )
                svg.append(
                    f'<text x="{(x_done + x_due) / 2:.1f}" y="{y_label}" text-anchor="middle" font-size="12" fill="#166534" font-weight="700">ready +{margin:.0f} ms</text>'
                )
            else:
                x1 = min(x_due, x_done)
                x2 = max(x_due, x_done)
                svg.append(f'<rect x="{x1:.1f}" y="{y + 3}" width="{max(2, x2 - x1):.1f}" height="40" fill="#fee2e2" opacity="0.55"/>')
                svg.append(
                    f'<line x1="{x_due:.1f}" y1="{y_margin}" x2="{x_done:.1f}" y2="{y_margin}" '
                    'stroke="#dc2626" stroke-width="4" stroke-dasharray="8 5"/>'
                )
                svg.append(
                    f'<circle cx="{x_done:.1f}" cy="{y_margin}" r="5" fill="#dc2626"><title>prefetch done after replay due</title></circle>'
                )
                svg.append(
                    f'<text x="{(x_done + x_due) / 2:.1f}" y="{y_label}" text-anchor="middle" font-size="12" fill="#b91c1c" font-weight="700">{abs(margin):.0f} ms late</text>'
                )
    for item in sorted(selected_timeline, key=layer_order):
        sid = str(item.get("session_id", ""))
        if sid not in row_index:
            continue
        y = top + row_index[sid] * row_h
        kind = str(item.get("kind", ""))
        color = colors.get(kind, "#6b7280")
        raw_start_ms = float(item.get("start_ms", 0.0))
        raw_end_ms = float(item.get("end_ms", 0.0))
        x1 = x_clamped(raw_start_ms)
        x2 = x_clamped(raw_end_ms)
        label = html.escape(str(item.get("label", kind)))
        if raw_start_ms == raw_end_ms:
            stroke_width = 6 if kind == "replay_due" else 3
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y + 1}" x2="{x1:.1f}" y2="{y + 34}" stroke="{color}" stroke-width="{stroke_width}"><title>{label}</title></line>'
            )
        else:
            display_x2 = x2
            display_x1 = x1
            bar_y = y + 4
            bar_h = 24
            opacity = "0.72" if kind in {"hint_request", "replay"} else "0.88"
            stroke = ""
            replay_continues = kind == "replay" and x_pos(raw_end_ms) > left + plot_w
            if kind == "copy_activity":
                display_x2 = max(x2, x1 + 24)
                row = row_by_session.get(sid, {})
                hint_start = to_float(row.get("hint_request_start_ms"))
                hint_end = to_float(row.get("hint_request_end_ms"))
                if (
                    hint_start is not None
                    and hint_end is not None
                    and raw_start_ms >= hint_start
                    and raw_end_ms <= hint_end
                ):
                    hint_x1 = x_clamped(hint_start)
                    hint_x2 = x_clamped(hint_end)
                    if hint_x2 > hint_x1:
                        display_x1 = max(hint_x1, min(display_x1, hint_x2 - 2))
                        display_x2 = min(hint_x2, max(display_x2, display_x1 + 2))
                bar_y = y
                bar_h = 32
                opacity = "1"
                stroke = ' stroke="#f8fafc" stroke-width="3"'
                color = "#16a34a" if item.get("copy_source") == "torch_profiler_h2d" else "#22c55e"
            svg.append(
                f'<rect x="{display_x1:.1f}" y="{bar_y}" width="{max(2, display_x2 - display_x1):.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="{opacity}"{stroke}><title>{label}</title></rect>'
            )
            if kind == "hint_request":
                svg.append(
                    f'<line x1="{x1:.1f}" y1="{bar_y - 3}" x2="{x1:.1f}" y2="{bar_y + bar_h + 3}" stroke="#6d28d9" stroke-width="1.3"><title>hint request start</title></line>'
                )
                svg.append(
                    f'<line x1="{x2:.1f}" y1="{bar_y - 3}" x2="{x2:.1f}" y2="{bar_y + bar_h + 3}" stroke="#6d28d9" stroke-width="1.3"><title>hint request end</title></line>'
                )
                svg.append(
                    f'<text x="{x1:.1f}" y="{bar_y - 6}" text-anchor="middle" font-size="9" fill="#6d28d9" font-weight="700">hint start</text>'
                )
                svg.append(
                    f'<text x="{x2:.1f}" y="{bar_y + bar_h + 13}" text-anchor="middle" font-size="9" fill="#6d28d9" font-weight="700">hint end</text>'
                )
            if kind == "replay":
                svg.append(
                    f'<line x1="{x1:.1f}" y1="{bar_y - 3}" x2="{x1:.1f}" y2="{bar_y + bar_h + 3}" stroke="#991b1b" stroke-width="1.3"><title>replay start</title></line>'
                )
            if replay_continues:
                arrow_x = left + plot_w - 7
                arrow_y = bar_y + bar_h / 2
                svg.append(
                    f'<path d="M {arrow_x - 7:.1f} {arrow_y - 7:.1f} L {arrow_x:.1f} {arrow_y:.1f} L {arrow_x - 7:.1f} {arrow_y + 7:.1f}" '
                    'fill="none" stroke="#991b1b" stroke-width="2"><title>replay continues beyond focused window</title></path>'
                )
                svg.append(f'<text x="{left + plot_w - 82:.1f}" y="{bar_y - 4}" font-size="10" fill="#991b1b" font-weight="700">continues</text>')
            if kind == "copy_activity":
                text_label = "HtoD" if item.get("copy_source") == "torch_profiler_h2d" else "KV"
                svg.append(
                    f'<line x1="{x1:.1f}" y1="{bar_y - 3}" x2="{x1:.1f}" y2="{bar_y + bar_h + 3}" stroke="#064e3b" stroke-width="1.5"><title>exact copy start</title></line>'
                )
                svg.append(
                    f'<line x1="{x2:.1f}" y1="{bar_y - 3}" x2="{x2:.1f}" y2="{bar_y + bar_h + 3}" stroke="#064e3b" stroke-width="1.5"><title>exact copy end</title></line>'
                )
                svg.append(
                    f'<text x="{(display_x1 + display_x2) / 2:.1f}" y="{bar_y + 20}" text-anchor="middle" font-size="10" fill="white" font-weight="700">{text_label}</text>'
                )
    legend_x = left
    legend_y = height - 32
    legend_labels = {
        "initial": "initial",
        "tool_wait": "tool wait",
        "hint_submitted": "hint submitted",
        "hint_request": "hint request",
        "copy_activity": "copy activity",
        "replay_due": "replay due",
        "replay": "replay",
    }
    for idx, (kind, color) in enumerate(colors.items()):
        lx = legend_x + idx * 130
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}">{html.escape(legend_labels.get(kind, kind))}</text>')
    svg.append(
        f'<line x1="{left}" y1="{height - 8}" x2="{left + 90}" y2="{height - 8}" stroke="#16a34a" stroke-width="4" stroke-dasharray="8 5"/>'
    )
    svg.append(f'<text x="{left + 100}" y="{height - 4}" fill="#166534">green gap = prefetch done before replay</text>')
    svg.append(
        f'<line x1="{left + 420}" y1="{height - 8}" x2="{left + 510}" y2="{height - 8}" stroke="#dc2626" stroke-width="4" stroke-dasharray="8 5"/>'
    )
    svg.append(f'<text x="{left + 520}" y="{height - 4}" fill="#b91c1c">red gap = prefetch finished after replay was due</text>')
    svg.append("</svg>")
    return "\n".join(svg), selected_rows


def visible_copy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("visible_copy_end_ms") in ("", None):
            continue
        out.append(
            {
                "session_id": row.get("session_id", ""),
                "copy_start_ms": row.get("visible_copy_start_ms", ""),
                "copy_end_ms": row.get("visible_copy_end_ms", ""),
                "copy_source": row.get("visible_copy_source", ""),
                "telemetry_events": row.get("telemetry_h2d_copy_events", ""),
                "torch_h2d_events": row.get("torch_h2d_copy_events", ""),
                "torch_h2d_bytes": row.get("torch_h2d_bytes", ""),
                "copy_inside_hint": yes_no(row.get("visible_copy_inside_hint")),
                "replay_due_ms": row.get("replay_due_ms", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
            }
        )
    return out


def timeline_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    margins = [float(row["prefetch_margin_ms"]) for row in rows if row.get("prefetch_margin_ms") not in ("", None)]
    missing_reasons = sorted({str(row.get("h2d_missing_reason")) for row in rows if row.get("h2d_missing_reason")})
    return [
        {
            "sessions": len(rows),
            "sessions_with_visible_copy_telemetry": sum(1 for row in rows if row.get("visible_copy_end_ms") not in ("", None)),
            "sessions_with_lightweight_kv_copy_telemetry": sum(1 for row in rows if as_int(row, "telemetry_h2d_copy_events") > 0),
            "sessions_with_profiler_cuda_h2d_copy": sum(1 for row in rows if as_int(row, "torch_h2d_copy_events") > 0),
            "late_prefetch_sessions": sum(1 for row in rows if truthy(row.get("late_prefetch"))),
            "kv_copy_ready_before_replay": sum(1 for row in rows if truthy(row.get("kv_copy_ready_before_replay"))),
            "cuda_copy_ready_before_replay": sum(1 for row in rows if truthy(row.get("cuda_copy_ready_before_replay"))),
            "full_hint_done_before_replay": sum(1 for row in rows if truthy(row.get("full_hint_done_before_replay"))),
            "sessions_where_replay_reloaded_kv": sum(1 for row in rows if truthy(row.get("replay_reloaded_kv"))),
            "clean_success_sessions": sum(1 for row in rows if row.get("checkpoint_result") == "clean_success"),
            "h2d_missing_reasons": ", ".join(missing_reasons),
            "avg_prefetch_margin_ms": round(mean(margins), 3) if margins else "",
        }
    ]


def timeline_layers_rows() -> list[dict[str, Any]]:
    return [
        {
            "Layer": "gray tool_wait",
            "Meaning": "The agent is waiting for a tool result, such as tests, search, or build output.",
            "Why it matters": "This is the opportunity window where prefetch can happen before the next model turn.",
        },
        {
            "Layer": "purple hint_request",
            "Meaning": "The software request we currently send to SGLang to trigger KV load-back.",
            "Why it matters": "This is not pure DMA. It includes scheduling, prefix matching, KV load-back, model work, and request bookkeeping.",
        },
        {
            "Layer": "green copy activity",
            "Meaning": "The single visible KV copy-activity bar. Dark green is used when CUDA HtoD profiler evidence exists. Light green is used as a fallback when only lightweight SGLang KV telemetry exists.",
            "Why it matters": "This keeps the main chart simple while preserving the exact telemetry and torch timings in the tables.",
        },
        {
            "Layer": "red replay",
            "Meaning": "The real agent replay request.",
            "Why it matters": "If the purple hint request overlaps the red replay in time, the prefetch path was still running when the agent needed to resume.",
        },
        {
            "Layer": "black replay_due",
            "Meaning": "The time when the tool result is ready and the real replay request is due.",
            "Why it matters": "If prefetch finishes after this line, the agent can still stall waiting for KV.",
        },
        {
            "Layer": "green/red dashed gap",
            "Meaning": "The time between KV/copy completion and replay due.",
            "Why it matters": "Green means KV movement finished before replay. Red means replay was already due before movement finished.",
        },
    ]


def prefetch_checkpoint_rows() -> list[dict[str, Any]]:
    return [
        {
            "Checkpoint": "KV copy ready before replay",
            "Simple meaning": "The lightweight green KV-copy telemetry ended before the black replay-due line.",
            "Why it matters": "This is the scalable checkpoint for larger runs where torch.profiler is off.",
        },
        {
            "Checkpoint": "CUDA copy ready before replay",
            "Simple meaning": "The dark-green profiler HtoD copy ended before the black replay-due line.",
            "Why it matters": "This proves profiler-visible GPU copy work happened early enough, but only for the copied slice we attributed in smaller validation runs.",
        },
        {
            "Checkpoint": "Full hint done before replay",
            "Simple meaning": "The whole purple hint request finished before replay resumed.",
            "Why it matters": "This catches cases where the copy started early but the normal SGLang request path was still busy.",
        },
        {
            "Checkpoint": "Replay reloaded KV",
            "Simple meaning": "The real replay request still triggered SGLang KV load-back events.",
            "Why it matters": "This catches cases where prefetched KV was incomplete, evicted, or not enough for the replay.",
        },
    ]


def checkpoint_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "session_id",
        "kv_copy_ready_before_replay",
        "cuda_copy_ready_before_replay",
        "full_hint_done_before_replay",
        "replay_reloaded_kv",
        "resume_load_count",
        "hint_outcome",
        "checkpoint_result",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def timeline_sanity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "session_id": row.get("session_id", ""),
            "green_inside_purple": yes_no(row.get("visible_copy_inside_hint")),
            "light_green_inside_purple": yes_no(row.get("telemetry_copy_inside_hint")),
            "dark_green_inside_purple": yes_no(row.get("torch_copy_inside_hint")),
            "hint_overlaps_replay": yes_no(row.get("hint_overlaps_replay")),
            "hint_replay_overlap_ms": row.get("hint_replay_overlap_ms", ""),
            "replay_reloaded_kv": yes_no(row.get("replay_reloaded_kv")),
        }
        for row in rows
    ]


def key_observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        status, observation, deduction = session_observation(row)
        out.append(
            {
                "session_id": row.get("session_id", ""),
                "status": status,
                "what happened": observation,
                "deduction and evidence": deduction,
            }
        )
    return out


def session_detail_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [
        "session_id",
        "priority",
        "tool_wait_ms",
        "prompt_tokens",
        "hint_submitted_ms",
        "hint_request_start_ms",
        "hint_request_end_ms",
        "sglang_copy_start_ms",
        "sglang_copy_end_ms",
        "telemetry_copy_start_ms",
        "telemetry_copy_end_ms",
        "telemetry_h2d_copy_events",
        "telemetry_host_index_count",
        "torch_copy_start_ms",
        "torch_copy_end_ms",
        "torch_h2d_copy_events",
        "torch_h2d_bytes",
        "visible_copy_start_ms",
        "visible_copy_end_ms",
        "visible_copy_source",
        "sglang_kv_profiler_status",
        "h2d_missing_reason",
        "profiler_start_ms",
        "profiler_end_ms",
        "profiler_stop_reason",
        "replay_due_ms",
        "replay_start_ms",
        "replay_end_ms",
        "hint_replay_overlap_ms",
        "hint_overlaps_replay",
        "telemetry_copy_inside_hint",
        "torch_copy_inside_hint",
        "visible_copy_inside_hint",
        "prefetch_margin_ms",
        "late_prefetch",
        "kv_copy_ready_before_replay",
        "cuda_copy_ready_before_replay",
        "full_hint_done_before_replay",
        "replay_reloaded_kv",
        "resume_load_count",
        "resume_hicache_load_count",
        "eviction_pressure_after_prefetch",
        "hint_total_duration_ms",
        "hint_outcome",
        "checkpoint_result",
        "replay_ttft_ms",
    ]
    return [{column: row.get(column, "") for column in columns} for row in rows]


def synthetic_setup_diagram_svg() -> str:
    boxes = [
        (50, 65, 190, 74, "Agent Task", "synthetic session"),
        (280, 65, 190, 74, "First Turn", "model builds KV"),
        (510, 65, 190, 74, "Tool Call", "controlled trigger"),
        (740, 65, 190, 74, "Tool Wait Gap", "prefetch opportunity"),
        (970, 65, 190, 74, "Hint / Prefetch", "warm/direct-load/oracle"),
        (1200, 65, 190, 74, "Resume Turn", "measure TTFT + reloads"),
    ]
    arrows = [
        (240, 102, 280, 102),
        (470, 102, 510, 102),
        (700, 102, 740, 102),
        (930, 102, 970, 102),
        (1160, 102, 1200, 102),
    ]
    parts = [
        '<svg viewBox="0 0 1440 210" width="100%" role="img" aria-label="Simple synthetic experiment setup flow diagram">',
        "<defs>",
        '<marker id="synthetic-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#334155"/>',
        "</marker>",
        "</defs>",
        '<rect x="20" y="25" width="1400" height="150" rx="10" fill="#f8fafc" stroke="#e5e7eb"/>',
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="2" marker-end="url(#synthetic-arrow)"/>'
        )
    for idx, (x, y, w, h, title, subtitle) in enumerate(boxes):
        fill = "#fff7ed" if idx in (3, 4) else "#eff6ff" if idx == 5 else "#ffffff"
        stroke = "#ea580c" if idx in (3, 4) else "#2563eb" if idx == 5 else "#cbd5e1"
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
                f'<text x="{x + w / 2}" y="{y + 27}" text-anchor="middle" font-size="15" font-weight="700">{html.escape(title)}</text>',
                f'<text x="{x + w / 2}" y="{y + 48}" text-anchor="middle" font-size="12" fill="#475569">{html.escape(subtitle)}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="720" y="192" text-anchor="middle" font-size="13" fill="#475569">Core question: was the right KV ready before the resume turn arrived?</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def synthetic_setup_html(sections: dict[str, list[dict[str, Any]]]) -> str:
    clean = {str(row.get("mode", "")): row for row in sections.get("Clean Performance Summary", [])}
    attr = sections.get("Profiled Attribution Summary", [])
    attr_row = attr[0] if attr else {}
    best_mode = min(
        (row for row in clean.values() if row.get("mode") != "no_prefetch"),
        key=lambda row: float(row.get("avg_replay_ttft_ms") or 0.0),
        default={},
    )
    setup_rows = [
        {"part": "1. Request source", "simple meaning": "Synthetic agent sessions create controlled model turns.", "example": "synthetic task -> initial SGLang model request"},
        {"part": "2. Tool wait window", "simple meaning": "The driver inserts a controlled pause. That pause is the chance to prefetch KV.", "example": "emulated read_file(), grep(), or run_tests() wait"},
        {"part": "3. Resume request", "simple meaning": "The replay turn arrives, and we measure whether KV was ready before first token generation.", "example": "tool result arrives -> replay prompt starts"},
    ]
    mode_rows = [
        {"mode": "No prefetch", "what happens": "The system waits until replay arrives, then SGLang handles KV reuse/load normally."},
        {"mode": "Prefetch modes", "what happens": "During the tool wait, request-warm/direct-load/oracle modes try to prepare KV before replay."},
    ]
    metric_rows = [
        {"metric": "replay TTFT", "meaning": "Time from replay admission to first generated token in the clean run."},
        {"metric": "prefetch margin", "meaning": "Whether the copy/hint path finished before or after replay was due."},
        {"metric": "KV copy ready", "meaning": "Whether SGLang-level KV-copy telemetry finished before replay."},
        {"metric": "CUDA HtoD ready", "meaning": "Whether torch-profiler CUDA host-to-device copy evidence finished before replay."},
        {"metric": "replay reloaded KV", "meaning": "Whether the replay still triggered SGLang KV load-back work even after a hint."},
    ]
    observation_rows = [
        {"observation": "The synthetic setup gives controlled stress conditions.", "evidence": "We can make tool waits short, add cache pressure, and choose prefetch timing deliberately."},
        {"observation": "Performance and attribution are intentionally separated.", "evidence": "Clean runs support TTFT claims; profiled runs support DMA/KV mechanism claims."},
        {"observation": "The current best clean mode is recorded in the report.", "evidence": f"Best non-baseline mode shown here: {best_mode.get('mode', 'not available')}."},
        {"observation": "The mechanism run shows whether KV movement was actually visible.", "evidence": f"CUDA HtoD visible sessions: {attr_row.get('sessions_with_profiler_cuda_h2d_copy', '')} / {attr_row.get('profiled_sessions', '')}."},
        {"observation": "Replay reload behavior is a key failure signal.", "evidence": f"Replay reloaded KV in {attr_row.get('replay_reloaded_kv', '')} / {attr_row.get('profiled_sessions', '')} profiled sessions."},
    ]
    return "\n".join(
        [
            '<div class="panel"><h2>Experiment Setup And Manager Summary</h2>',
            '<p class="caption">This section is intended for slide-building. It explains how the controlled synthetic experiment was set up, what was measured, and how to read the hardware motivation.</p>',
            synthetic_setup_diagram_svg(),
            "<h3>Simple Setup</h3>",
            '<div class="table-wrap">',
            html_table(setup_rows),
            "</div>",
            "<h3>Modes Compared</h3>",
            '<div class="table-wrap">',
            html_table(mode_rows),
            "</div>",
            "<h3>What Was Measured</h3>",
            '<div class="table-wrap">',
            html_table(metric_rows),
            "</div>",
            "<h3>What Was Observed</h3>",
            '<div class="table-wrap">',
            html_table(observation_rows),
            "</div>",
            "<h3>Why This Supports The Hardware Proposal</h3>",
            '<p class="caption">The synthetic setup lets us stress the exact failure modes we care about: short tool gaps, cache pressure, late prefetch, replay reloads, and unclear residency. If a software hint path copies or loads KV too late, or if replay reloads KV anyway, that supports the need for deadline-aware KV movement, residency protection, and better hardware/runtime telemetry.</p>',
            "</div>",
        ]
    )


SECTION_THEMES = {
    "executive": "theme-summary",
    "setup": "theme-setup",
    "timeline-guide": "theme-guide",
    "global-prefetch": "theme-global",
    "clean-timelines": "theme-clean",
    "expanded-clean-timelines": "theme-clean",
    "clean-tables": "theme-clean-table",
    "profiled-timelines": "theme-profiled",
    "profiled-tables": "theme-profiled",
    "deductions": "theme-deductions",
    "observations": "theme-observations",
    "paired": "theme-paired",
    "reproduce": "theme-reproduce",
    "appendix": "theme-appendix",
}


def html_toc(items: list[tuple[str, str]]) -> str:
    links = "".join(
        f'<a class="{SECTION_THEMES.get(anchor, "theme-appendix")}" href="#{html.escape(anchor)}">{html.escape(label)}</a>'
        for anchor, label in items
    )
    return (
        '<div class="panel"><h2>Table of Contents</h2>'
        '<p class="section-color-legend">Colors group sections by evidence type: setup, clean performance, '
        'profiled mechanism, interpretation, and appendix.</p>'
        f'<div class="toc">{links}</div>'
        '<div class="toc-actions"><button type="button" data-action="expand-all">Expand All</button>'
        '<button type="button" data-action="collapse-all">Collapse All</button></div></div>'
    )


def report_script() -> str:
    return """
<script>
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".panel[id][class*='theme-']").forEach(function (panel) {
    const id = panel.id;
    const h2 = panel.querySelector(":scope > h2");
    if (!id || !h2) return;
    const details = document.createElement("details");
    details.id = id;
    details.className = panel.className.replace("panel", "section-card").trim();
    if (id === "executive") details.open = true;
    const summary = document.createElement("summary");
    summary.appendChild(h2);
    details.appendChild(summary);
    panel.removeAttribute("id");
    while (panel.firstChild) {
      details.appendChild(panel.firstChild);
    }
    panel.replaceWith(details);
  });
  const cards = Array.from(document.querySelectorAll("details.section-card"));
  document.querySelectorAll(".toc a[href^='#']").forEach(function (link) {
    link.addEventListener("click", function () {
      const id = link.getAttribute("href").slice(1);
      const target = document.getElementById(id);
      if (target && target.tagName.toLowerCase() === "details") {
        target.open = true;
      }
    });
  });
  document.querySelectorAll("[data-action='expand-all']").forEach(function (button) {
    button.addEventListener("click", function () {
      cards.forEach(function (card) { card.open = true; });
    });
  });
  document.querySelectorAll("[data-action='collapse-all']").forEach(function (button) {
    button.addEventListener("click", function () {
      cards.forEach(function (card) { card.open = false; });
    });
  });
});
</script>
"""


def html_code_block(text: str) -> str:
    return f"<pre><code>{html.escape(text.strip())}</code></pre>"


def synthetic_reproduce_html() -> str:
    build_existing = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=synthetic_manager_demo_1 \
UPDATE_LATEST=0 \
CLEAN_ROOT=artifacts/results/milestone15_targeted_dma_validation/clean_performance \
ATTRIBUTION_ROOT=artifacts/results/milestone15_targeted_dma_validation/profiled_attribution \
bash scripts/build_labeled_synthetic_master_report.sh
"""
    refresh_latest = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=synthetic_latest_refresh \
UPDATE_LATEST=1 \
CLEAN_ROOT=artifacts/results/milestone15_targeted_dma_validation/clean_performance \
ATTRIBUTION_ROOT=artifacts/results/milestone15_targeted_dma_validation/profiled_attribution \
bash scripts/build_labeled_synthetic_master_report.sh
"""
    rerun_full = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone12_rerun_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
SESSION_COUNT=12 \
RANDOMIZE_TRAFFIC=1 \
ORACLE_LEAD_MS=1000 \
bash scripts/run_milestone12_paired_evidence.sh Qwen/Qwen2.5-1.5B-Instruct
"""
    return "\n".join(
        [
            '<div class="panel theme-reproduce" id="reproduce"><h2>Reproduce This Report</h2>',
            '<p class="caption">These commands reproduce or archive the controlled synthetic master report. Labeled builds do not overwrite the current latest report unless <code>UPDATE_LATEST=1</code>.</p>',
            "<h3>Archive A Labeled Synthetic Report From Existing Runs</h3>",
            html_code_block(build_existing),
            '<p class="caption">Output:</p>',
            html_code_block("artifacts/results/labeled/synthetic/synthetic_manager_demo_1/master_report.html"),
            "<h3>Deliberately Refresh The Latest Synthetic Master Report</h3>",
            html_code_block(refresh_latest),
            "<h3>Rerun The Full Synthetic Experiment</h3>",
            '<p class="caption">This reruns the clean and profiled synthetic experiment, then regenerates the standard latest synthetic report.</p>',
            html_code_block(rerun_full),
            "</div>",
        ]
    )


def synthetic_timeline_guide_html() -> str:
    step_rows = [
        {"step": "1. Ask model what to do", "timeline color": "blue bar", "simple meaning": "model turn before the tool call"},
        {"step": "2. Model says to call a tool", "timeline color": "end of blue bar", "simple meaning": "the model turn hands work to a tool"},
        {"step": "3. Tool runs", "timeline color": "gray bar", "simple meaning": "the synthetic tool wait is happening"},
        {"step": "4. Agent waits", "timeline color": "gray bar", "simple meaning": "this wait is the prefetch opportunity"},
        {"step": "5. Tool returns", "timeline color": "black vertical line", "simple meaning": "the next model turn is due"},
        {"step": "6. Agent asks model again", "timeline color": "red bar", "simple meaning": "resume request after the tool result"},
        {"step": "During steps 3/4, if prefetch is enabled", "timeline color": "purple bar", "simple meaning": "our prefetch/direct-load attempt runs during the wait"},
    ]
    rows = [
        {
            "color": "blue",
            "meaning": "Initial model turn",
            "simple description": "The agent asks the model what to do next. In the synthetic report, this is the first request before the simulated tool wait.",
        },
        {
            "color": "gray",
            "meaning": "Tool wait window",
            "simple description": "The tool is assumed to be running, so the model is idle for this session. This pause is the opportunity to prepare that session's KV before the agent resumes.",
        },
        {
            "color": "purple",
            "meaning": "Prefetch attempt window",
            "simple description": "This includes detecting the tool-wait opportunity, creating a hint for that agent/session, calling our direct SGLang KV hook, letting SGLang check whether host-side KV exists, and if needed, asking SGLang to move KV back to GPU memory.",
        },
        {
            "color": "green",
            "meaning": "KV copy/load activity",
            "simple description": "This is evidence that KV load/copy work was observed for the hint. Dark green is CUDA HtoD profiler evidence; light green is lightweight SGLang KV telemetry fallback.",
        },
        {
            "color": "black",
            "meaning": "Replay due boundary",
            "simple description": "This is when the tool result is ready and the next model turn should be able to start. If purple or green finishes after this line, the prefetch path was late for that resume.",
        },
        {
            "color": "red",
            "meaning": "Replay request",
            "simple description": "The agent asks the model to continue after the tool result. This is the request we want to speed up by having the right KV ready before it arrives.",
        },
        {
            "color": "yellow",
            "meaning": "First token marker",
            "simple description": "This marks when the replay request starts producing output. It helps separate request arrival from first-token latency.",
        },
    ]
    return "\n".join(
        [
            '<div class="panel theme-guide" id="timeline-guide"><h2>How To Read The Timelines</h2>',
            '<p class="caption">The timelines are the main visual evidence. Read the clean timelines first for performance, then the profiled mechanism timeline for KV/copy attribution.</p>',
            "<h3>One Row In Plain English</h3>",
            '<p class="caption">Each timeline row is one tool-wait episode: model turn, tool call, wait, optional prefetch, then model resume.</p>',
            '<div class="table-wrap">',
            html_table(step_rows),
            "</div>",
            "<h3>Color Legend</h3>",
            '<div class="table-wrap">',
            html_table(rows),
            "</div></div>",
        ]
    )


def synthetic_margin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        margin = to_float(row.get("prefetch_margin_ms"))
        if margin is None:
            continue
        duration = to_float(row.get("hint_total_duration_ms"))
        if margin >= 50:
            verdict = "early"
        elif margin >= 0:
            verdict = "barely early"
        elif margin > -50:
            verdict = "near miss"
        else:
            verdict = "late"
        out.append(
            {
                "session_id": row.get("session_id", ""),
                "priority": row.get("priority", ""),
                "prompt_tokens": row.get("prompt_tokens", ""),
                "tool_wait_ms": row.get("tool_wait_ms", ""),
                "hint_total_duration_ms": duration,
                "prefetch_margin_ms": margin,
                "checkpoint_result": row.get("checkpoint_result", ""),
                "verdict": verdict,
            }
        )
    return out


def synthetic_margin_summary(rows: list[dict[str, Any]], total_sessions: int) -> list[dict[str, Any]]:
    margins = [float(row["prefetch_margin_ms"]) for row in rows]
    durations = [
        float(row["hint_total_duration_ms"])
        for row in rows
        if row.get("hint_total_duration_ms") is not None
    ]
    early = [value for value in margins if value >= 0]
    late = [value for value in margins if value < 0]
    return [
        {
            "profiled_sessions": total_sessions,
            "sessions_with_measured_margin": len(rows),
            "sessions_without_measured_margin": max(0, total_sessions - len(rows)),
            "finished_before_replay": len(early),
            "late": len(late),
            "late_pct": round(len(late) * 100.0 / len(rows), 2) if rows else "",
            "median_margin_ms": round(median(margins), 3) if margins else "",
            "worst_lateness_ms": round(abs(min(late)), 3) if late else "",
            "best_early_margin_ms": round(max(early), 3) if early else "",
            "avg_hint_duration_ms": round(mean(durations), 3) if durations else "",
        }
    ]


def synthetic_margin_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [
        ("> +500 ms early", lambda value: value > 500),
        ("+100 to +500 ms early", lambda value: 100 < value <= 500),
        ("0 to +100 ms early", lambda value: 0 <= value <= 100),
        ("0 to -100 ms late", lambda value: -100 <= value < 0),
        ("-100 to -500 ms late", lambda value: -500 <= value < -100),
        ("< -500 ms late", lambda value: value < -500),
    ]
    total = len(rows)
    output: list[dict[str, Any]] = []
    for label, predicate in buckets:
        count = sum(1 for row in rows if predicate(float(row["prefetch_margin_ms"])))
        output.append(
            {
                "bucket": label,
                "sessions": count,
                "pct": round(count * 100.0 / total, 2) if total else "",
            }
        )
    return output


def symlog_value(value: float, linear_width: float = 50.0) -> float:
    if value == 0:
        return 0.0
    return math.copysign(math.log1p(abs(value) / linear_width), value)


def symlog_tick_values(min_margin: float, max_margin: float) -> list[float]:
    candidates = [
        -10000.0,
        -5000.0,
        -1000.0,
        -500.0,
        -100.0,
        -50.0,
        0.0,
        50.0,
        100.0,
        500.0,
        1000.0,
        5000.0,
        10000.0,
    ]
    ticks = [value for value in candidates if min_margin <= value <= max_margin]
    for value in (min_margin, max_margin):
        if all(abs(value - tick) > 1 for tick in ticks):
            ticks.append(value)
    return sorted(ticks)


def synthetic_margin_dot_plot(rows: list[dict[str, Any]], scale: str = "linear") -> str:
    if not rows:
        return '<p class="caption">No synthetic prefetch margin rows were available.</p>'
    if scale not in {"linear", "symlog"}:
        raise ValueError(f"unsupported margin plot scale: {scale}")
    width = 1480
    height = 500
    left = 86
    right = 34
    top = 56
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    margins = [float(row["prefetch_margin_ms"]) for row in rows]
    min_margin = min(margins)
    max_margin = max(margins)
    pad = max(50.0, (max_margin - min_margin) * 0.08)
    y_min = min(min_margin - pad, -50.0)
    y_max = max(max_margin + pad, 50.0)
    if scale == "symlog":
        y_min = min(y_min, -50.0)
        y_max = max(y_max, 50.0)
        scaled_min = symlog_value(y_min)
        scaled_max = symlog_value(y_max)
    else:
        scaled_min = y_min
        scaled_max = y_max

    def x_pos(index: int) -> float:
        if len(rows) <= 1:
            return left + plot_w / 2
        return left + index * plot_w / (len(rows) - 1)

    def y_pos(value: float) -> float:
        scaled = symlog_value(value) if scale == "symlog" else value
        return top + (scaled_max - scaled) * plot_h / (scaled_max - scaled_min)

    zero_y = y_pos(0.0)
    scale_label = "symlog" if scale == "symlog" else "linear"
    axis_label = "prefetch margin ms (symlog)" if scale == "symlog" else "prefetch margin ms"
    svg = [
        f'<svg viewBox="0 0 1480 500" width="100%" role="img" aria-label="Synthetic global prefetch margin dot plot {scale_label} view">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="2"/>',
        f'<text x="{left + plot_w - 8}" y="{zero_y - 8:.1f}" text-anchor="end" font-size="12" font-weight="700">0 ms deadline</text>',
        f'<text x="18" y="250" transform="rotate(-90 18 250)" text-anchor="middle" font-size="13" font-weight="700">{axis_label}</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="13" font-weight="700">synthetic session order</text>',
        '<text x="94" y="34" font-size="13" fill="#166534" font-weight="700">above line = finished before replay</text>',
        '<text x="350" y="34" font-size="13" fill="#b91c1c" font-weight="700">below line = late prefetch</text>',
    ]
    tick_values = symlog_tick_values(y_min, y_max) if scale == "symlog" else [y_min, y_min / 2, 0.0, y_max / 2, y_max]
    seen_ticks: set[int] = set()
    for value in tick_values:
        rounded = int(round(value))
        if rounded in seen_ticks:
            continue
        seen_ticks.add(rounded)
        y = y_pos(value)
        svg.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{rounded} ms</text>')
    for index, row in enumerate(rows):
        x = x_pos(index)
        margin = float(row["prefetch_margin_ms"])
        duration = row.get("hint_total_duration_ms")
        color = "#16a34a" if margin >= 50 else "#84cc16" if margin >= 0 else "#f97316" if margin > -50 else "#dc2626"
        radius = 7
        if isinstance(duration, (int, float)):
            radius = max(5, min(10, 5 + float(duration) / 250.0))
        y = y_pos(margin)
        title = (
            f"{row.get('session_id')} | margin={margin:.3f} ms | "
            f"hint_duration={duration if duration is not None else 'n/a'} ms | "
            f"tool_wait={row.get('tool_wait_ms')} ms | prompt_tokens={row.get('prompt_tokens')}"
        )
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" opacity="0.88" stroke="#ffffff" stroke-width="1.4">'
            f'<title>{html.escape(title)}</title></circle>'
        )
        svg.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="10">{html.escape(str(row.get("session_id")))}</text>')
    legend = [
        ("early", "#16a34a"),
        ("barely early", "#84cc16"),
        ("near miss", "#f97316"),
        ("late", "#dc2626"),
    ]
    lx = left
    ly = height - 50
    for label, color in legend:
        svg.append(f'<circle cx="{lx}" cy="{ly}" r="6" fill="{color}"/>')
        svg.append(f'<text x="{lx + 12}" y="{ly + 4}" font-size="12">{html.escape(label)}</text>')
        lx += 145
    svg.append("</svg>")
    return "\n".join(svg)


def synthetic_global_prefetch_margin_html(attribution_rows: list[dict[str, Any]]) -> str:
    rows = synthetic_margin_rows(attribution_rows)
    return (
        '<div class="panel theme-global" id="global-prefetch"><h2>Global Prefetch Margin</h2>'
        '<p class="caption">This is the high-level view across profiled synthetic sessions with measured prefetch margins. Positive margin means prefetch finished before replay; negative margin means replay was already due before prefetch finished. Sessions without visible copy/margin attribution are counted in the summary table but are not plotted as dots.</p>'
        '<div class="table-wrap">'
        + html_table(synthetic_margin_summary(rows, len(attribution_rows)))
        + "</div>"
        + "<h3>Linear View</h3>"
        + '<p class="caption">The linear view preserves the true size of large early or late margins.</p>'
        + synthetic_margin_dot_plot(rows)
        + "<h3>Symlog View</h3>"
        + '<p class="caption">Same data as above, but compressed with a symmetric log-style scale so both small near-deadline misses and very large late prefetches are easier to see. Above zero still means early; below zero still means late.</p>'
        + synthetic_margin_dot_plot(rows, scale="symlog")
        + '<h3>Margin Buckets</h3><div class="table-wrap">'
        + html_table(synthetic_margin_bucket_rows(rows))
        + '</div><h3>Points Behind The Plot</h3><div class="table-wrap">'
        + html_table(rows)
        + "</div></div>"
    )


def write_html(
    path: Path,
    sections: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    attribution_rows: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    clean_timelines: dict[str, dict[str, list[dict[str, Any]]]],
    max_timeline_sessions: int,
) -> None:
    cards = summary_cards(sections)
    timeline_svg, selected_timeline_rows = build_timeline_svg(attribution_rows, timeline, max_timeline_sessions)
    toc = [
        ("executive", "Executive Summary"),
        ("setup", "Experiment Setup"),
        ("timeline-guide", "How To Read Timelines"),
        ("global-prefetch", "Global Prefetch Margin"),
        ("clean-timelines", "Clean Performance Timelines"),
        ("expanded-clean-timelines", "Expanded Per-Gap Timelines"),
        ("clean-tables", "Clean Performance Tables"),
        ("profiled-timelines", "Profiled Mechanism Timelines"),
        ("profiled-tables", "Profiled Mechanism Tables"),
        ("deductions", "Key Deductions"),
        ("observations", "Session Observations"),
        ("paired", "Paired Evidence"),
        ("reproduce", "Reproduce This Report"),
        ("appendix", "Appendix"),
    ]
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Milestone 12 Paired Evidence Report</title>",
        "<style>",
        ":root{--ink:#111827;--muted:#4b5563;--line:#e5e7eb;--soft:#f8fafc;--panel:#ffffff;--good:#166534;--bad:#b91c1c;--warn:#b45309}",
        "body{font-family:Arial,sans-serif;margin:28px;background:var(--soft);color:var(--ink)}",
        "h1{font-size:32px;margin:0 0 8px} h2{font-size:22px;margin:0 0 12px} h3{font-size:16px;margin:14px 0 8px}",
        ".subtle{color:var(--muted);line-height:1.5;margin:0}",
        ".panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px;margin:18px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:14px}",
        ".card{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fbfdff}",
        ".card .label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}",
        ".card .value{font-size:24px;font-weight:700;margin-top:6px}",
        ".theme-summary{--theme:#1e3a8a;--theme-bg:#eff6ff}",
        ".theme-setup{--theme:#2563eb;--theme-bg:#eff6ff}",
        ".theme-guide{--theme:#475569;--theme-bg:#f1f5f9}",
        ".theme-global{--theme:#dc2626;--theme-bg:#fef2f2}",
        ".theme-clean{--theme:#15803d;--theme-bg:#f0fdf4}",
        ".theme-clean-table{--theme:#65a30d;--theme-bg:#f7fee7}",
        ".theme-profiled{--theme:#7e22ce;--theme-bg:#faf5ff}",
        ".theme-deductions{--theme:#b45309;--theme-bg:#fffbeb}",
        ".theme-observations{--theme:#0f766e;--theme-bg:#f0fdfa}",
        ".theme-paired{--theme:#be123c;--theme-bg:#fff1f2}",
        ".theme-reproduce{--theme:#0891b2;--theme-bg:#ecfeff}",
        ".theme-appendix{--theme:#64748b;--theme-bg:#f8fafc}",
        ".panel[class*='theme-']{border-top:5px solid var(--theme)}",
        ".panel[class*='theme-']>h2{border-left:8px solid var(--theme);padding-left:10px;color:var(--theme)}",
        "details.section-card{background:var(--panel);border:1px solid var(--line);border-top:5px solid var(--theme);border-radius:8px;padding:18px;margin:18px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}",
        "details.section-card summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px}",
        "details.section-card summary::-webkit-details-marker{display:none}",
        "details.section-card summary h2{border-left:8px solid var(--theme);padding-left:10px;color:var(--theme);margin:0}",
        "details.section-card summary h2::before{content:'▶';display:inline-block;color:var(--theme);font-size:16px;margin-right:8px;transform:translateY(-1px)}",
        "details.section-card[open] summary h2::before{content:'▼'}",
        ".section-color-legend{color:#475569;font-size:14px;margin:8px 0 12px}",
        ".toc{display:flex;flex-wrap:wrap;gap:10px 12px;margin-top:14px}",
        ".toc a{background:var(--theme-bg,#f1f5f9);border:1px solid #e2e8f0;border-left:7px solid var(--theme,#64748b);border-radius:6px;padding:7px 10px;color:#0f172a;text-decoration:none;font-weight:650}",
        ".toc-actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}",
        ".toc-actions button{border:1px solid #cbd5e1;background:#fff;color:#0f172a;border-radius:6px;padding:7px 10px;font-weight:650;cursor:pointer}",
        ".toc-actions button:hover{background:#f8fafc}",
        ".table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}",
        "th{background:#f3f4f6;font-weight:700;white-space:nowrap}",
        "td{white-space:normal;min-width:90px}",
        ".wide td{min-width:180px}",
        ".caption{color:#374151;line-height:1.5}",
        ".good{color:var(--good);font-weight:700}",
        ".bad{color:var(--bad);font-weight:700}",
        ".warn{color:var(--warn);font-weight:700}",
        ".pill{display:inline-block;border-radius:999px;padding:3px 8px;font-size:12px;font-weight:700;background:#eef2ff;color:#3730a3}",
        "ul{line-height:1.55;margin:8px 0 0 20px;padding:0}",
        "code{background:#f3f4f6;border-radius:4px;padding:1px 4px}",
        "pre{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;border-radius:8px;padding:12px;overflow:auto}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Milestone 12 Paired Evidence Report</h1>",
        '<p class="subtle">Clean runs answer performance questions. Profiled runs answer mechanism questions. This keeps the TTFT story separate from profiler overhead.</p>',
        html_toc(toc),
        '<div class="panel theme-summary" id="executive"><h2>Executive Summary</h2>',
        '<p class="caption">This report is designed to answer two different questions without mixing them together.</p>',
        "<ul>",
        *[f"<li>{html.escape(line)}</li>" for line in manager_summary_lines(sections)],
        "</ul>",
        '<div class="cards">',
        *[
            f'<div class="card"><div class="label">{html.escape(card["label"])}</div><div class="value">{html.escape(card["value"])}</div><p class="subtle">{html.escape(card["detail"])}</p></div>'
            for card in cards
        ],
        "</div></div>",
        synthetic_setup_html(sections).replace('<div class="panel"><h2>', '<div class="panel theme-setup" id="setup"><h2>', 1),
        synthetic_timeline_guide_html(),
        synthetic_global_prefetch_margin_html(attribution_rows),
        '<div class="panel theme-guide"><h2>How To Read This Report</h2>',
        '<p class="caption"><span class="pill">Clean performance</span> comes from profiler-off runs. Use these rows for TTFT and latency claims.</p>',
        '<p class="caption"><span class="pill">Mechanism attribution</span> shows lightweight SGLang KV-copy telemetry, optional torch-profiler CUDA HtoD validation, hint completion, and replay reload behavior.</p>',
        '<p class="caption"><span class="pill">Paired evidence</span> joins the two views by session id, so we can say what improved and what mechanism was observed.</p>',
        "</div>",
        '<div class="panel theme-deductions" id="deductions"><h2>Key Deductions</h2><ul>',
        *[f"<li>{html.escape(line)}</li>" for line in key_deduction_lines(sections)],
        "</ul></div>",
    ]

    lines.append('<div class="panel theme-clean" id="clean-timelines"><h2>A. Clean Performance Timelines</h2>')
    lines.append(
        '<p class="caption"><span class="pill">Profiler OFF</span> Use these timelines for request-flow, replay timing, and TTFT/performance claims. They intentionally do not show CUDA HtoD bars.</p>'
    )
    if clean_timelines:
        for mode, data in clean_timelines.items():
            clean_svg, selected_clean_rows = build_clean_timeline_svg(
                mode,
                data.get("rows", []),
                data.get("timeline", []),
                max_timeline_sessions,
            )
            lines.append(f'<h3>{html.escape(mode)}</h3>')
            lines.append(
                '<p class="caption">Blue is the initial request, gray is the tool wait, purple is the hint request if this mode sends one, black is replay due, red is replay admission/execution, and yellow is first token. Long replay bars are clipped so the chart focuses on the resume boundary.</p>'
            )
            lines.append(clean_svg)
            lines.append('<div class="table-wrap">')
            lines.append(html_table(selected_clean_rows))
            lines.append("</div>")
    else:
        lines.append('<p class="caption">No clean performance timeline data was found for this run.</p>')
    lines.append("</div>")

    lines.append('<div class="panel theme-clean" id="expanded-clean-timelines"><h2>A.1 Expanded Per-Gap Timelines</h2>')
    lines.append(
        '<p class="caption"><span class="pill">Profiler OFF</span> This is the same clean-performance data, but each row has its own local clock. The black replay-due line is always <code>0 ms</code>; negative time is before replay due, and positive time is after the resume boundary.</p>'
    )
    lines.append(
        '<p class="caption">Use this view when the global timeline feels compressed. It stretches each synthetic tool gap independently so the hint window, replay boundary, replay request, and first-token marker are easier to compare.</p>'
    )
    if clean_timelines:
        for mode, data in clean_timelines.items():
            expanded_svg, selected_expanded_rows = build_expanded_clean_timeline_svg(
                mode,
                data.get("rows", []),
                data.get("timeline", []),
                max_timeline_sessions,
            )
            lines.append(f'<h3>{html.escape(mode)}</h3>')
            lines.append(expanded_svg)
            lines.append('<div class="table-wrap">')
            lines.append(html_table(selected_expanded_rows))
            lines.append("</div>")
    else:
        lines.append('<p class="caption">No expanded clean timeline data was found for this run.</p>')
    lines.append("</div>")

    lines.append('<div class="panel theme-clean-table" id="clean-tables"><h2>A.2 Clean Performance Tables</h2>')
    lines.append(f'<p class="caption">{html.escape(section_caption("Clean Performance Summary"))}</p>')
    lines.append('<div class="table-wrap">')
    lines.append(html_table(sections.get("Clean Performance Summary", [])))
    lines.append("</div></div>")

    if attribution_rows and timeline:
        lines.append('<div class="panel theme-profiled" id="profiled-timelines"><h2>B. Profiled Mechanism Timelines</h2>')
        lines.append(
            '<p class="caption"><span class="pill">Profiler / telemetry view</span> Use this for KV-copy and DMA-style attribution, not clean TTFT claims. Dark green means CUDA HtoD profiler evidence; light green means lightweight SGLang KV telemetry fallback.</p>'
        )
        lines.append(timeline_svg)
        lines.append("</div>")

        lines.append('<div class="panel theme-profiled" id="profiled-tables"><h2>B.1 Profiled Mechanism Tables</h2>')
        lines.append(f'<p class="caption">{html.escape(section_caption("Profiled Attribution Summary"))}</p>')
        lines.append('<div class="table-wrap">')
        lines.append(html_table(sections.get("Profiled Attribution Summary", [])))
        lines.append("</div>")
        lines.append("<h3>Timeline Summary</h3>")
        lines.append('<div class="table-wrap">')
        lines.append(html_table(timeline_summary_rows(attribution_rows)))
        lines.append("</div>")
        sanity_rows = timeline_sanity_rows(selected_timeline_rows)
        if sanity_rows:
            lines.append("<h3>Timeline Sanity Checks</h3>")
            lines.append(html_table(sanity_rows))
        copy_rows = visible_copy_rows(selected_timeline_rows)
        if copy_rows:
            lines.append("<h3>Visible KV Copy Telemetry</h3>")
            lines.append(
                '<p class="caption">These are the selected sessions represented by green copy bars in the timeline. `sglang_lightweight_h2d_telemetry` is the scalable source for larger runs; `torch_profiler_h2d` is the heavier CUDA-validation source for smaller runs.</p>'
            )
            lines.append('<div class="table-wrap">')
            lines.append(html_table(copy_rows))
            lines.append("</div>")
        lines.append("<h3>Timeline Layers</h3>")
        lines.append(html_table(timeline_layers_rows()))
        lines.append("<h3>Prefetch Checkpoints</h3>")
        lines.append(html_table(prefetch_checkpoint_rows()))
        lines.append("<h3>Checkpoint Results Per Session</h3>")
        lines.append(html_table(checkpoint_result_rows(selected_timeline_rows)))
        lines.append("</div>")

        timeline_sections = [
            (
                "Key Observations Per Session",
                key_observation_rows(selected_timeline_rows),
                "Plain-English interpretation of the selected sessions in the timeline.",
            ),
            (
                "Session Details",
                session_detail_rows(selected_timeline_rows),
                "Raw timing fields for the selected sessions. Use this to defend the visual conclusions.",
            ),
        ]
        for title, rows, caption in timeline_sections:
            section_id = "observations" if title == "Key Observations Per Session" else "appendix"
            theme_class = "theme-observations" if title == "Key Observations Per Session" else "theme-appendix"
            lines.append(f'<div class="panel {theme_class}" id="{section_id}"><h2>{html.escape(title)}</h2>')
            lines.append(f'<p class="caption">{html.escape(caption)}</p>')
            lines.append('<div class="table-wrap wide">')
            lines.append(html_table(rows))
            lines.append("</div></div>")
    else:
        lines.append(
            '<div class="panel theme-profiled" id="profiled-timelines"><h2>B. Profiled Mechanism Timelines</h2><p class="caption">No profiled timeline JSON was found for this report. Run the profiled attribution step to populate the visual timeline sections.</p></div>'
        )
    lines.append('<div class="panel theme-paired" id="paired"><h2>Paired Session Evidence</h2>')
    lines.append(f'<p class="caption">{html.escape(section_caption("Paired Session Evidence"))}</p>')
    lines.append('<div class="table-wrap">')
    lines.append(html_table(sections.get("Paired Session Evidence", [])))
    lines.append("</div></div>")
    lines.append(synthetic_reproduce_html())
    lines.append('<div class="panel theme-appendix" id="appendix-metadata"><h2>Appendix: Metadata</h2><pre>')
    lines.append(html.escape(json.dumps(metadata, indent=2, sort_keys=True)))
    lines.append("</pre></div>")
    lines.append(report_script())
    lines.extend(["</body>", "</html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_cards(sections: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    clean = sections.get("Clean Performance Summary", [])
    attr = sections.get("Profiled Attribution Summary", [])
    baseline = next((row for row in clean if row.get("mode") == "no_prefetch"), {})
    best = max(
        (row for row in clean if row.get("mode") != "no_prefetch"),
        key=lambda row: as_float(row, "avg_improvement_vs_no_prefetch_ms"),
        default={},
    )
    attr_row = attr[0] if attr else {}
    profiled_sessions = as_int(attr_row, "profiled_sessions") if attr_row else 0
    return [
        {
            "label": "Baseline Replay TTFT",
            "value": f'{baseline.get("avg_replay_ttft_ms", "n/a")} ms',
            "detail": "From the clean no-prefetch run.",
        },
        {
            "label": "Best Clean Delta",
            "value": f'{best.get("avg_improvement_vs_no_prefetch_ms", "n/a")} ms',
            "detail": f'Best mode: {best.get("mode", "n/a")}.',
        },
        {
            "label": "KV Copy Ready",
            "value": f'{attr_row.get("kv_copy_ready_before_replay", 0)} / {profiled_sessions}',
            "detail": "Sessions where scalable KV-copy telemetry finished before replay.",
        },
        {
            "label": "Replay Reloaded KV",
            "value": f'{attr_row.get("replay_reloaded_kv", 0)} / {profiled_sessions}',
            "detail": "Profiled sessions where replay still did KV load-back work.",
        },
    ]


def manager_summary_lines(sections: dict[str, list[dict[str, Any]]]) -> list[str]:
    clean = sections.get("Clean Performance Summary", [])
    attr = sections.get("Profiled Attribution Summary", [])
    best = max(
        (row for row in clean if row.get("mode") != "no_prefetch"),
        key=lambda row: as_float(row, "avg_improvement_vs_no_prefetch_ms"),
        default={},
    )
    attr_row = attr[0] if attr else {}
    lines = [
        "The clean run is the performance source of truth because it runs without torch.profiler overhead.",
        "The mechanism run is the movement source of truth because it exposes lightweight SGLang KV-copy telemetry, with torch-profiler CUDA HtoD as optional validation.",
    ]
    if best:
        lines.append(
            f'Best clean mode so far: {best.get("mode")} with {best.get("avg_improvement_vs_no_prefetch_ms")} ms average replay TTFT delta versus no_prefetch.'
        )
    if attr_row:
        lines.append(
            f'Mechanism attribution: KV copy ready before replay in {attr_row.get("kv_copy_ready_before_replay")} / {attr_row.get("profiled_sessions")} sessions; profiler CUDA HtoD visible in {attr_row.get("sessions_with_profiler_cuda_h2d_copy")} / {attr_row.get("profiled_sessions")} sessions; replay reloaded KV in {attr_row.get("replay_reloaded_kv")} / {attr_row.get("profiled_sessions")} sessions.'
        )
    return lines


def key_deduction_lines(sections: dict[str, list[dict[str, Any]]]) -> list[str]:
    attr = sections.get("Profiled Attribution Summary", [])
    paired = sections.get("Paired Session Evidence", [])
    attr_row = attr[0] if attr else {}
    cuda_ready = as_int(attr_row, "cuda_copy_ready_before_replay") if attr_row else 0
    kv_ready = as_int(attr_row, "kv_copy_ready_before_replay") if attr_row else 0
    reloaded = as_int(attr_row, "replay_reloaded_kv") if attr_row else 0
    hint_done = as_int(attr_row, "full_hint_done_before_replay") if attr_row else 0
    sessions = as_int(attr_row, "profiled_sessions") if attr_row else 0
    lines = [
        "We should judge TTFT using the clean run, then use the profiled run to explain why the result happened.",
    ]
    if kv_ready and reloaded:
        lines.append(
            "Important hardware argument: even when KV copy telemetry finishes before replay, replay can still reload KV. Copying memory earlier is not enough by itself; residency, protection, and reuse need to be enforceable."
        )
    elif cuda_ready and reloaded:
        lines.append(
            "Important hardware argument: even when CUDA HtoD copy is ready before replay, replay can still reload KV. Copying memory earlier is not enough by itself; residency, protection, and reuse need to be enforceable."
        )
    if sessions and hint_done < sessions:
        lines.append(
            "Some hint paths did not fully finish before replay. That supports the concern that software hints can be delayed by the normal serving path."
        )
    if paired:
        reload_rows = sum(1 for row in paired if row.get("profiled_replay_reloaded_kv") == "yes")
        if reload_rows:
            lines.append(
                f"{reload_rows} paired sessions show replay reload behavior, which is the exact failure mode eviction protection/residency hints are meant to address."
            )
    return lines


def section_caption(title: str) -> str:
    captions = {
        "Clean Performance Summary": "Profiler is off here. Use this table for TTFT and performance claims.",
        "Profiled Attribution Summary": "Use this table to understand scalable KV-copy telemetry, optional CUDA HtoD validation, hint completion, and replay reloads.",
        "Paired Session Evidence": "This joins the clean and profiled views by session id so each session has both a performance view and a mechanism view.",
    }
    return captions.get(title, "")


def copy_latest_reports(out_root: Path, latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    latest_synthetic = latest_root / "latest_synthetic"
    latest_synthetic.mkdir(parents=True, exist_ok=True)
    master = out_root / "paired_report.html"
    if master.exists():
        shutil.copyfile(master, latest_root / "latest_synthetic_master_report.html")
    copies = [
        ("paired_report.html", "master_report.html"),
        ("paired_report.md", "master_report.md"),
        ("paired_report.json", "master_report.json"),
        ("paired_clean_summary.csv", "clean_summary.csv"),
        ("paired_attribution_summary.csv", "attribution_summary.csv"),
        ("paired_session_evidence.csv", "session_evidence.csv"),
        ("paired_timeline_summary.csv", "timeline_summary.csv"),
        ("paired_timeline_sanity_checks.csv", "timeline_sanity_checks.csv"),
        ("paired_checkpoint_results.csv", "checkpoint_results.csv"),
        ("paired_key_observations.csv", "key_observations.csv"),
        ("paired_session_details.csv", "session_details.csv"),
    ]
    for src_name, dst_name in copies:
        src = out_root / src_name
        if src.exists():
            shutil.copyfile(src, latest_synthetic / dst_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join clean performance and profiled attribution runs.")
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--attribution-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--modes", default="no_prefetch direct_load oracle_direct_load")
    parser.add_argument("--attribution-mode", default="oracle_direct_load")
    parser.add_argument("--latest-root", default="")
    parser.add_argument("--timeline-json", default="")
    parser.add_argument("--max-timeline-sessions", type=int, default=12)
    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    attribution_root = Path(args.attribution_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    modes = [mode for mode in args.modes.replace(",", " ").split() if mode]

    clean_rows = load_clean_rows(clean_root, modes)
    clean_timelines = load_clean_timelines(clean_root, modes, clean_rows)
    attribution_rows = read_csv(attribution_root / f"{args.attribution_mode}_agentic_prefetch_timeline.csv")
    timeline_json = Path(args.timeline_json) if args.timeline_json else attribution_root / f"{args.attribution_mode}_agentic_prefetch_timeline.json"
    timeline_rows = load_timeline_json(timeline_json)
    selected_ids = choose_timeline_sessions(attribution_rows, args.max_timeline_sessions)
    attr_by_session = {str(row.get("session_id", "")): row for row in attribution_rows}
    selected_timeline_rows = [attr_by_session[sid] for sid in selected_ids if sid in attr_by_session]

    sections = {
        "Clean Performance Summary": build_clean_summary(clean_rows),
        "Profiled Attribution Summary": build_attribution_summary(attribution_rows),
        "Paired Session Evidence": build_paired_rows(clean_rows, attribution_rows, args.attribution_mode),
    }
    metadata = {
        "clean_root": str(clean_root),
        "attribution_root": str(attribution_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "modes": modes,
        "attribution_mode": args.attribution_mode,
        "timeline_json": str(timeline_json),
        "max_timeline_sessions": args.max_timeline_sessions,
        "latest_root": args.latest_root,
        "note": "Profiled attribution rows should not be used for TTFT performance claims.",
    }

    write_csv(out_root / "paired_clean_summary.csv", sections["Clean Performance Summary"])
    write_csv(out_root / "paired_attribution_summary.csv", sections["Profiled Attribution Summary"])
    write_csv(out_root / "paired_session_evidence.csv", sections["Paired Session Evidence"])
    write_csv(out_root / "paired_timeline_summary.csv", timeline_summary_rows(attribution_rows))
    write_csv(out_root / "paired_timeline_sanity_checks.csv", timeline_sanity_rows(selected_timeline_rows))
    write_csv(out_root / "paired_checkpoint_results.csv", checkpoint_result_rows(selected_timeline_rows))
    write_csv(out_root / "paired_key_observations.csv", key_observation_rows(selected_timeline_rows))
    write_csv(out_root / "paired_session_details.csv", session_detail_rows(selected_timeline_rows))
    for mode, data in clean_timelines.items():
        selected_clean_ids = choose_clean_timeline_sessions(data.get("rows", []), args.max_timeline_sessions)
        clean_by_session = {str(row.get("session_id", "")): row for row in data.get("rows", [])}
        write_csv(
            out_root / f"clean_{mode}_timeline_details.csv",
            [clean_by_session[sid] for sid in selected_clean_ids if sid in clean_by_session],
        )
    (out_root / "paired_report.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "sections": sections,
                "timeline": {
                    "clean_performance_timelines": clean_timelines,
                    "summary": timeline_summary_rows(attribution_rows),
                    "layers": timeline_layers_rows(),
                    "sanity_checks": timeline_sanity_rows(selected_timeline_rows),
                    "prefetch_checkpoints": prefetch_checkpoint_rows(),
                    "checkpoint_results": checkpoint_result_rows(selected_timeline_rows),
                    "key_observations": key_observation_rows(selected_timeline_rows),
                    "session_details": session_detail_rows(selected_timeline_rows),
                    "timeline_rows": timeline_rows,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_md(out_root / "paired_report.md", sections, metadata)
    write_timeline_md(out_root / "paired_report.md", attribution_rows, timeline_rows, args.max_timeline_sessions)
    write_html(
        out_root / "paired_report.html",
        sections,
        metadata,
        attribution_rows,
        timeline_rows,
        clean_timelines,
        args.max_timeline_sessions,
    )
    if args.latest_root:
        copy_latest_reports(out_root, Path(args.latest_root))

    print(f"Wrote paired report under {out_root}")
    if args.latest_root:
        print(f"Wrote latest synthetic master report copies under {args.latest_root}")
    for row in sections["Clean Performance Summary"]:
        print(
            f"clean {row['mode']}: avg_replay_ttft_ms={row['avg_replay_ttft_ms']}, "
            f"outcomes={row['outcomes']}"
        )
    for row in sections["Profiled Attribution Summary"]:
        print(
            "profiled attribution: "
            f"kv_ready={row.get('kv_copy_ready_before_replay', 0)}/{row['profiled_sessions']}, "
            f"cuda_ready={row['cuda_copy_ready_before_replay']}/{row['profiled_sessions']}, "
            f"hint_done={row['full_hint_done_before_replay']}/{row['profiled_sessions']}, "
            f"reloaded={row['replay_reloaded_kv']}/{row['profiled_sessions']}, "
            f"clean_success={row['clean_success']}/{row['profiled_sessions']}"
        )


if __name__ == "__main__":
    main()
