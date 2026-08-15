#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def ms_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() * 1000.0, 3)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def short(value: Any, limit: int = 180) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def first_event(events: list[dict[str, Any]], stage: str, phase: str | None = None) -> dict[str, Any] | None:
    for event in events:
        if event.get("stage") != stage:
            continue
        if phase is not None and event.get("phase") != phase:
            continue
        return event
    return None


def pair_phase_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = [
        event
        for event in events
        if str(event.get("stage", "")).endswith("_request_dispatched")
        and event.get("event_kind") == "request_dispatch"
    ]
    ends: list[dict[str, Any]] = [
        event
        for event in events
        if str(event.get("stage", "")).endswith("_response_received")
        and event.get("event_kind") == "response"
    ]
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for start in starts:
        phase = str(start.get("phase") or "unknown")
        start_ts = parse_ts(start.get("timestamp"))
        candidates = [
            (idx, end)
            for idx, end in enumerate(ends)
            if idx not in used
            and str(end.get("phase") or "unknown") == phase
            and (parse_ts(end.get("timestamp")) or datetime.max.replace(tzinfo=start_ts.tzinfo if start_ts else None))
            >= (start_ts or datetime.min.replace(tzinfo=None))
        ]
        if not candidates:
            continue
        idx, end = candidates[0]
        used.add(idx)
        end_ts = parse_ts(end.get("timestamp"))
        measurement = end.get("measurement") if isinstance(end.get("measurement"), dict) else {}
        prompt = start.get("prompt") or ""
        request_context = start.get("request_context") if isinstance(start.get("request_context"), dict) else {}
        pairs.append(
            {
                "phase": phase,
                "sequence_index": start.get("sequence_index", ""),
                "request_id": request_context.get("request_id", ""),
                "start": start,
                "end": end,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_ms": ms_between(start_ts, end_ts),
                "prompt_chars": start.get("prompt_chars", len(str(prompt))),
                "prompt_lines": start.get("prompt_lines", ""),
                "latency_ms": measurement.get("latency_ms", ""),
                "ttft_ms": measurement.get("ttft_ms", ""),
                "input_tokens": measurement.get("input_tokens", ""),
                "output_tokens": measurement.get("output_tokens", ""),
                "tool_call_count": (end.get("tool_progress") or {}).get("tool_call_count", ""),
                "tool_call_names": ", ".join((end.get("tool_progress") or {}).get("tool_call_names") or []),
                "prompt": prompt,
            }
        )
    return pairs


def run_rows(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in index_rows:
        run_dir = Path(str(item.get("result_dir", "")))
        result = read_json(run_dir / "others" / "result.json") or {}
        lifecycle = read_json(run_dir / "others" / "stage_lifecycle_trace_raw.json") or []
        if not isinstance(lifecycle, list):
            lifecycle = []
        phase_pairs = pair_phase_events(lifecycle)
        tool_calls = sum(int(float(row.get("tool_call_count") or 0)) for row in phase_pairs)
        total_latency = sum(float(row.get("latency_ms") or 0) for row in phase_pairs if row.get("latency_ms") not in ("", None))
        rows.append(
            {
                "task_index": item.get("task_index", ""),
                "run_id": item.get("run_id", run_dir.name),
                "repo": item.get("repo", result.get("task", {}).get("repo", "")),
                "instance_id": result.get("task", {}).get("instance_id", ""),
                "model": result.get("model", ""),
                "frontend_url": result.get("frontend_url", ""),
                "phase_model_turns": len(phase_pairs),
                "tool_calls_seen": tool_calls,
                "total_model_latency_ms": round(total_latency, 3),
                "result_dir": str(run_dir),
                "prompt_evolution_report": str(run_dir / "prompt_evolution_report.md"),
                "tool_call_details": str(Path(str(item.get("report_dir", ""))) / "tool_call_details.md")
                if item.get("report_dir")
                else "",
            }
        )
    return rows


def phase_rows(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in index_rows:
        run_dir = Path(str(item.get("result_dir", "")))
        lifecycle = read_json(run_dir / "others" / "stage_lifecycle_trace_raw.json") or []
        if not isinstance(lifecycle, list):
            continue
        for pair in pair_phase_events(lifecycle):
            rows.append(
                {
                    "task_index": item.get("task_index", ""),
                    "run_id": item.get("run_id", run_dir.name),
                    "phase": pair["phase"],
                    "request_id": pair.get("request_id", ""),
                    "duration_ms": pair.get("duration_ms", ""),
                    "reported_latency_ms": pair.get("latency_ms", ""),
                    "ttft_ms": pair.get("ttft_ms", ""),
                    "prompt_chars": pair.get("prompt_chars", ""),
                    "input_tokens": pair.get("input_tokens", ""),
                    "output_tokens": pair.get("output_tokens", ""),
                    "tool_call_count": pair.get("tool_call_count", ""),
                    "tool_call_names": pair.get("tool_call_names", ""),
                }
            )
    return rows


def kv_summary(trace_path: Path, copy_path: Path | None) -> list[dict[str, Any]]:
    trace = read_jsonl(trace_path)
    copies = read_jsonl(copy_path) if copy_path else []
    event_counts = Counter(str(row.get("event", "")) for row in trace)
    agent_context_rows = [
        row
        for row in trace
        if row.get("agent_session_id") or row.get("agent_sessions") or (isinstance(row.get("request"), dict) and row["request"].get("agent_session_id"))
    ]
    h2d_events = [
        row
        for row in trace
        if row.get("direction") == "host_to_device"
        or (isinstance(row.get("context"), dict) and row["context"].get("direction") == "host_to_device")
    ]
    copy_h2d = [row for row in copies if row.get("direction") == "host_to_device"]
    return [
        {
            "trace_file": str(trace_path),
            "trace_events": len(trace),
            "unique_event_names": len(event_counts),
            "top_events": ", ".join(f"{name}: {count}" for name, count in event_counts.most_common(8)),
            "sglang_h2d_like_events": len(h2d_events),
            "trace_events_with_agent_context": len(agent_context_rows),
            "copy_telemetry_file": str(copy_path or ""),
            "copy_telemetry_h2d_events": len(copy_h2d),
        }
    ]


def copy_summary(copy_path: Path | None) -> list[dict[str, Any]]:
    copies = read_jsonl(copy_path) if copy_path else []
    direction_counts = Counter(str(row.get("direction", "unknown")) for row in copies)
    agent_rows = [row for row in copies if row.get("agent_session_id")]
    durations = [
        float(row.get("duration_ms"))
        for row in copies
        if row.get("event") == "kv_telemetry.copy.end" and row.get("duration_ms") not in ("", None)
    ]
    return [
        {
            "copy_events": len(copies),
            "agent_context_copy_events": len(agent_rows),
            "unique_agent_sessions": len({str(row.get("agent_session_id")) for row in agent_rows}),
            "device_to_host_events": direction_counts.get("device_to_host", 0),
            "host_to_device_events": direction_counts.get("host_to_device", 0),
            "device_evict_events": direction_counts.get("device_evict", 0),
            "avg_copy_end_duration_ms": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "max_copy_end_duration_ms": round(max(durations), 3) if durations else 0.0,
        }
    ]


def load_replay_workload(out_root: Path) -> list[dict[str, Any]]:
    candidates = [
        out_root.parent / "agentbench_replay_workload.jsonl",
        out_root.parent.parent / "latest_agentbench_replay_workload.jsonl",
    ]
    for path in candidates:
        rows = read_jsonl(path)
        if rows:
            return rows
    return []


def replay_session_rows(workload_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in workload_rows:
        wait_ms = float(row.get("tool_wait_ms") or 0)
        prompt_tokens = int(float(row.get("prompt_tokens") or 0))
        replay_tokens = int(float(row.get("replay_prompt_tokens") or 0))
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "from_phase": row.get("from_phase", ""),
                "to_phase": row.get("to_phase", ""),
                "tool_wait_ms": round(wait_ms, 3),
                "wait_class": "very_short" if wait_ms < 100 else "short" if wait_ms < 500 else "long",
                "current_latency_ms": row.get("current_latency_ms", ""),
                "next_latency_ms": row.get("next_latency_ms", ""),
                "prompt_tokens": prompt_tokens,
                "replay_prompt_tokens": replay_tokens,
                "token_delta": replay_tokens - prompt_tokens,
                "next_tool_calls": row.get("next_tool_call_count", ""),
            }
        )
    return rows


def high_level_summary(
    runs: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    kv: list[dict[str, Any]],
    copy: list[dict[str, Any]],
    replay: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    total_tool_calls = sum(int(float(row.get("tool_calls_seen") or 0)) for row in runs)
    total_latency = sum(float(row.get("total_model_latency_ms") or 0) for row in runs)
    kv_row = kv[0] if kv else {}
    copy_row = copy[0] if copy else {}
    return [
        {
            "real_agentbench_runs": len(runs),
            "phase_model_turns": len(phases),
            "tool_calls_observed": total_tool_calls,
            "total_model_latency_ms": round(total_latency, 3),
            "sglang_trace_events": kv_row.get("trace_events", 0),
            "kv_copy_telemetry_events": copy_row.get("copy_events", 0),
            "copy_events_with_agent_context": copy_row.get("agent_context_copy_events", 0),
            "replay_sessions_extracted": len(replay),
        }
    ]


def timeline_layer_rows() -> list[dict[str, Any]]:
    return [
        {
            "layer": "blue current_turn",
            "meaning": "the real AgentBench/DeepAgents turn before the next resume",
            "why_it_matters": "this is the context whose KV may be useful for the next turn",
        },
        {
            "layer": "gray tool_wait",
            "meaning": "the observed gap between the current turn and the next AgentBench turn",
            "why_it_matters": "this is the opportunity window where a future hint-aware prefetch could run",
        },
        {
            "layer": "black replay_due",
            "meaning": "the boundary where the next AgentBench turn is due",
            "why_it_matters": "prefetch must finish before this boundary to avoid resume stalls",
        },
        {
            "layer": "red replay",
            "meaning": "the next real prompt extracted from AgentBench",
            "why_it_matters": "this is the request replayed in Milestone 18 prefetch-mode experiments",
        },
        {
            "layer": "purple/green not shown here",
            "meaning": "the live direct run does not inject hint requests, so there is no purple hint bar or green prefetch-copy bar in this timeline",
            "why_it_matters": "those bars appear in the controlled replay reports where we compare prefetch modes",
        },
    ]


def key_observation_rows(
    runs: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    copy: list[dict[str, Any]],
    replay: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if runs:
        run = runs[0]
        rows.append(
            {
                "observation": "Real harness is connected",
                "evidence": f"{run.get('repo')} run {run.get('run_id')} produced {run.get('phase_model_turns')} model turns through {run.get('frontend_url')}",
                "deduction": "This is no longer a synthetic-only experiment; real AgentBench traffic reached SGLang directly.",
            }
        )
    if phases:
        longest = max(phases, key=lambda row: float(row.get("reported_latency_ms") or 0))
        rows.append(
            {
                "observation": "Agent phases have very different serving costs",
                "evidence": f"{longest.get('phase')} took {longest.get('reported_latency_ms')} ms with {longest.get('input_tokens')} input tokens and {longest.get('output_tokens')} output tokens",
                "deduction": "A hint policy should not treat all turns equally; phase and prompt size matter.",
            }
        )
    copy_row = copy[0] if copy else {}
    if copy_row:
        rows.append(
            {
                "observation": "KV/copy telemetry is visible",
                "evidence": f"{copy_row.get('copy_events')} copy telemetry events, {copy_row.get('agent_context_copy_events')} carrying agent/session context",
                "deduction": "The SGLang hooks can connect memory movement to agent sessions in the live path.",
            }
        )
    if replay:
        short_waits = sum(1 for row in replay if str(row.get("wait_class")) in {"very_short", "short"})
        rows.append(
            {
                "observation": "Many extracted waits are short",
                "evidence": f"{short_waits}/{len(replay)} replay sessions have wait_class short or very_short",
                "deduction": "This is exactly where normal software scheduling can miss the prefetch window.",
            }
        )
    return rows


def build_phase_timeline_svg(rows: list[dict[str, Any]], index_rows: list[dict[str, Any]]) -> str:
    spans: list[dict[str, Any]] = []
    for item in index_rows:
        run_id = str(item.get("run_id", ""))
        lifecycle = read_json(Path(str(item.get("result_dir", ""))) / "others" / "stage_lifecycle_trace_raw.json") or []
        if not isinstance(lifecycle, list):
            continue
        pairs = pair_phase_events(lifecycle)
        if not pairs:
            continue
        base_ts = pairs[0]["start_ts"]
        for pair in pairs:
            start_ms = ms_between(base_ts, pair["start_ts"])
            end_ms = ms_between(base_ts, pair["end_ts"])
            if start_ms is None or end_ms is None:
                continue
            spans.append(
                {
                    "run_id": run_id,
                    "phase": pair["phase"],
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "label": pair["phase"],
                }
            )
    if not spans:
        return "<p>No phase timeline available.</p>"

    selected_runs = []
    for span in spans:
        run_id = span["run_id"]
        if run_id not in selected_runs:
            selected_runs.append(run_id)
        if len(selected_runs) >= 6:
            break
    spans = [span for span in spans if span["run_id"] in selected_runs]
    start = min(float(span["start_ms"]) for span in spans)
    end = max(float(span["end_ms"]) for span in spans)
    span_ms = max(1.0, end - start)
    width = 1500
    left = 250
    right = 50
    top = 70
    row_h = 54
    height = top + len(selected_runs) * row_h + 72
    plot_w = width - left - right
    colors = {
        "planning": "#2563eb",
        "execution": "#a855f7",
        "patch_generation": "#f59e0b",
        "review": "#16a34a",
        "baseline_execution": "#64748b",
    }

    def x_pos(ms: float) -> float:
        return left + (ms - start) / span_ms * plot_w

    run_index = {run_id: idx for idx, run_id in enumerate(selected_runs)}
    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="AgentBench direct SGLang phase timeline">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    for tick in range(6):
        ms = start + span_ms * tick / 5
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 32}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{ms:.0f} ms</text>')
    for run_id in selected_runs:
        y = top + run_index[run_id] * row_h
        svg.append(f'<text x="10" y="{y + 18}" font-weight="700">{html.escape(run_id)}</text>')
        svg.append(f'<line x1="{left}" y1="{y + 8}" x2="{left + plot_w}" y2="{y + 8}" stroke="#f3f4f6"/>')
    for span in spans:
        y = top + run_index[span["run_id"]] * row_h
        x1 = x_pos(float(span["start_ms"]))
        x2 = x_pos(float(span["end_ms"]))
        phase = str(span["phase"])
        color = colors.get(phase, "#64748b")
        svg.append(
            f'<rect x="{x1:.1f}" y="{y}" width="{max(3, x2 - x1):.1f}" height="24" rx="3" '
            f'fill="{color}" opacity="0.82"><title>{html.escape(phase)}</title></rect>'
        )
        if x2 - x1 > 65:
            svg.append(
                f'<text x="{(x1 + x2) / 2:.1f}" y="{y + 16}" text-anchor="middle" '
                f'font-size="11" fill="white" font-weight="700">{html.escape(phase)}</text>'
            )
    legend_y = height - 28
    for idx, (phase, color) in enumerate(colors.items()):
        lx = left + idx * 170
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}">{html.escape(phase)}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def build_replay_timeline_svg(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No replay-session timeline available.</p>"
    spans: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:8]):
        session_id = str(row.get("session_id", f"session_{idx}"))
        arrival = float(row.get("arrival_ms") or idx * 80)
        current_latency = float(row.get("current_latency_ms") or 0)
        wait_ms = float(row.get("tool_wait_ms") or 0)
        next_latency = float(row.get("next_latency_ms") or 0)
        current_end = arrival + max(current_latency, 1.0)
        replay_due = current_end + wait_ms
        replay_end = replay_due + max(next_latency, 1.0)
        rel_current_start = arrival - replay_due
        rel_current_end = current_end - replay_due
        rel_replay_end = replay_end - replay_due
        spans.extend(
            [
                {
                    "session_id": session_id,
                    "kind": "current_turn",
                    "start_ms": rel_current_start,
                    "end_ms": rel_current_end,
                    "label": str(row.get("from_phase", "current")),
                    "wait_ms": wait_ms,
                },
                {
                    "session_id": session_id,
                    "kind": "tool_wait",
                    "start_ms": rel_current_end,
                    "end_ms": 0.0,
                    "label": f"wait {wait_ms:.0f} ms",
                    "wait_ms": wait_ms,
                },
                {
                    "session_id": session_id,
                    "kind": "replay_turn",
                    "start_ms": 0.0,
                    "end_ms": rel_replay_end,
                    "label": str(row.get("to_phase", "replay")),
                    "wait_ms": wait_ms,
                },
            ]
        )
        markers.append({"session_id": session_id, "kind": "replay_due", "time_ms": 0.0, "label": "replay due"})
    start = -1200.0
    end = 1800.0
    span_ms = max(1.0, end - start)
    sessions = []
    for span in spans:
        if span["session_id"] not in sessions:
            sessions.append(span["session_id"])
    width = 1500
    left = 330
    right = 50
    top = 78
    row_h = 64
    height = top + len(sessions) * row_h + 76
    plot_w = width - left - right
    colors = {
        "current_turn": "#2563eb",
        "tool_wait": "#d1d5db",
        "replay_due": "#111827",
        "replay_turn": "#ef4444",
    }

    def x_pos(ms: float) -> float:
        return left + (ms - start) / span_ms * plot_w

    def x_clamped(ms: float) -> float:
        return max(left, min(left + plot_w, x_pos(ms)))

    session_index = {session_id: idx for idx, session_id in enumerate(sessions)}
    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="AgentBench replay-boundary timeline">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    ticks = [-1200, -600, 0, 600, 1200, 1800]
    for ms in ticks:
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 36}" stroke="#e5e7eb"/>')
        label = "replay due" if ms == 0 else f"{ms:+.0f} ms"
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{html.escape(label)}</text>')
    for session_id in sessions:
        y = top + session_index[session_id] * row_h
        short_id = session_id.replace("agentbench_", "ab_")
        wait_span = next((span for span in spans if span["session_id"] == session_id), {})
        wait_ms = float(wait_span.get("wait_ms") or 0)
        status = "VERY SHORT WAIT" if wait_ms < 100 else "SHORT WAIT" if wait_ms < 500 else "LONG WAIT"
        status_color = "#b45309" if wait_ms < 100 else "#166534"
        svg.append(f'<text x="10" y="{y + 15}" font-weight="700">{html.escape(short_id)}</text>')
        svg.append(
            f'<text x="10" y="{y + 36}" font-size="13" fill="{status_color}" font-weight="700">{status} {wait_ms:.0f} ms</text>'
        )
        svg.append(f'<line x1="{left}" y1="{y + 8}" x2="{left + plot_w}" y2="{y + 8}" stroke="#f3f4f6"/>')
        svg.append(f'<line x1="{left}" y1="{y + 44}" x2="{left + plot_w}" y2="{y + 44}" stroke="#f9fafb"/>')
    for span in sorted(spans, key=lambda item: {"tool_wait": 0, "current_turn": 1, "replay_turn": 3}.get(str(item["kind"]), 2)):
        y = top + session_index[span["session_id"]] * row_h
        raw_start = float(span["start_ms"])
        raw_end = float(span["end_ms"])
        x1 = x_clamped(raw_start)
        x2 = x_clamped(raw_end)
        kind = str(span["kind"])
        color = colors.get(kind, "#64748b")
        opacity = "0.82" if kind == "replay_turn" else "0.88" if kind != "tool_wait" else "0.72"
        bar_y = y + 4
        bar_h = 24
        if raw_end < start or raw_start > end:
            continue
        display_x1 = x1
        display_x2 = x2
        if kind == "tool_wait" and display_x2 - display_x1 < 10:
            display_x1 = max(left, display_x2 - 10)
        continues = x_pos(raw_end) > left + plot_w or x_pos(raw_start) < left
        svg.append(
            f'<rect x="{display_x1:.1f}" y="{bar_y}" width="{max(3, display_x2 - display_x1):.1f}" height="{bar_h}" rx="3" '
            f'fill="{color}" opacity="{opacity}"><title>{html.escape(str(span["label"]))}</title></rect>'
        )
        if kind == "replay_turn":
            svg.append(
                f'<line x1="{x1:.1f}" y1="{bar_y - 3}" x2="{x1:.1f}" y2="{bar_y + bar_h + 3}" stroke="#991b1b" stroke-width="1.3"><title>replay start</title></line>'
            )
        if continues and kind in {"current_turn", "replay_turn"}:
            if kind == "replay_turn":
                arrow_x = left + plot_w - 7
                arrow_y = bar_y + bar_h / 2
                svg.append(
                    f'<path d="M {arrow_x - 7:.1f} {arrow_y - 7:.1f} L {arrow_x:.1f} {arrow_y:.1f} L {arrow_x - 7:.1f} {arrow_y + 7:.1f}" '
                    'fill="none" stroke="#991b1b" stroke-width="2"><title>replay continues beyond focused window</title></path>'
                )
                svg.append(f'<text x="{left + plot_w - 82:.1f}" y="{bar_y - 4}" font-size="10" fill="#991b1b" font-weight="700">continues</text>')
            elif kind == "current_turn":
                svg.append(f'<text x="{left + 8}" y="{bar_y - 4}" font-size="10" fill="#1d4ed8" font-weight="700">continues from earlier</text>')
        if display_x2 - display_x1 > 80:
            fill = "#111827" if kind == "tool_wait" else "white"
            svg.append(
                f'<text x="{(display_x1 + display_x2) / 2:.1f}" y="{bar_y + 16}" text-anchor="middle" '
                f'font-size="11" fill="{fill}" font-weight="700">{html.escape(str(span["label"]))}</text>'
            )
    for marker in markers:
        y = top + session_index[marker["session_id"]] * row_h
        x = x_pos(float(marker["time_ms"]))
        svg.append(
            f'<line x1="{x:.1f}" y1="{y + 1}" x2="{x:.1f}" y2="{y + 36}" stroke="#111827" stroke-width="6"><title>replay due</title></line>'
        )
        svg.append(f'<text x="{x:.1f}" y="{y - 4}" text-anchor="middle" font-size="10" fill="#111827" font-weight="700">due</text>')
    legend_y = height - 30
    for idx, (name, color) in enumerate(colors.items()):
        lx = left + idx * 170
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}">{html.escape(name)}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


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
            cls = (
                ' class="wrap"'
                if header
                in {
                    "result_dir",
                    "prompt_evolution_report",
                    "tool_call_details",
                    "top_events",
                    "evidence",
                    "deduction",
                    "why_it_matters",
                    "session_id",
                    "trace_file",
                    "copy_telemetry_file",
                }
                else ""
            )
            out.append(f"<td{cls}>{fmt(row.get(header, ''))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


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
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    lines.append("")
    return lines


def write_outputs(
    *,
    out_root: Path,
    latest_root: Path | None,
    index_rows: list[dict[str, Any]],
    trace_path: Path,
    copy_path: Path | None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    runs = run_rows(index_rows)
    phases = phase_rows(index_rows)
    kv = kv_summary(trace_path, copy_path)
    copy = copy_summary(copy_path)
    replay_workload = load_replay_workload(out_root)
    replay = replay_session_rows(replay_workload)
    summary = high_level_summary(runs, phases, kv, copy, replay)
    layers = timeline_layer_rows()
    observations = key_observation_rows(runs, phases, copy, replay)
    write_csv(out_root / "agentbench_sglang_runs.csv", runs)
    write_csv(out_root / "agentbench_sglang_phase_turns.csv", phases)
    write_csv(out_root / "agentbench_sglang_kv_summary.csv", kv)
    write_csv(out_root / "agentbench_sglang_copy_summary.csv", copy)
    write_csv(out_root / "agentbench_sglang_high_level_summary.csv", summary)
    write_csv(out_root / "agentbench_sglang_timeline_layers.csv", layers)
    write_csv(out_root / "agentbench_sglang_key_observations.csv", observations)
    write_csv(out_root / "agentbench_sglang_replay_sessions.csv", replay)

    timeline_svg = build_phase_timeline_svg(phases, index_rows)
    replay_svg = build_replay_timeline_svg(replay_workload)
    html_lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>AgentBench Direct SGLang Report</title>",
        "<style>",
        ":root{--ink:#111827;--muted:#4b5563;--line:#e5e7eb;--soft:#f8fafc;--panel:#ffffff;--good:#166534;--bad:#b91c1c;--warn:#b45309}",
        "body{font-family:Arial,sans-serif;margin:28px;background:var(--soft);color:var(--ink)}",
        "h1{font-size:32px;margin:0 0 8px} h2{font-size:22px;margin:0 0 12px} h3{font-size:16px;margin:14px 0 8px}",
        ".subtle{color:var(--muted);line-height:1.5;margin:0}",
        ".panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px 20px;margin:18px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}",
        ".caption{margin:0 0 12px;color:#374151;line-height:1.5;font-size:15px}",
        ".grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}",
        ".callout{border-left:4px solid #2563eb;background:#eff6ff;padding:12px 14px;margin:12px 0;color:#1f2937}",
        ".warn{border-left-color:#b45309;background:#fffbeb}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;white-space:nowrap;vertical-align:top}",
        "th{background:#f3f4f6;font-weight:700}",
        ".wrap{white-space:normal;line-height:1.35;min-width:260px;max-width:520px}",
        ".good{color:var(--good);font-weight:700}.bad{color:var(--bad);font-weight:700}.warnText{color:var(--warn);font-weight:700}",
        "@media(max-width:900px){.grid{grid-template-columns:1fr} body{margin:14px}}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>AgentBench Direct SGLang Report</h1>",
        '<p class="subtle">A manager-readable report for the real SWE-bench Pro -> AgentBench -> Deep Agents -> SGLang direct path.</p>',
        '<div class="panel"><h2>What This Report Shows</h2>',
        '<p class="caption">This run uses real SWE-bench tasks, the existing AgentBench/DeepAgents harness, and a direct SGLang OpenAI-compatible endpoint. Dynamo is removed. The runtime path is SWE-bench Pro -> AgentBench -> Deep Agents -> SGLang -> direct KV trace/reporting.</p>',
        '<div class="callout"><strong>Main point:</strong> this is the realism proof. It shows real agent traffic reaching SGLang and producing KV/cache/copy telemetry. The controlled prefetch comparison is done later using the replay sessions extracted from this live run.</div>',
        "</div>",
        '<div class="panel"><h2>High-Level Summary</h2>',
        html_table(summary),
        "</div>",
        '<div class="panel"><h2>Run Details</h2>',
        html_table(runs),
        "</div>",
        '<div class="panel"><h2>AgentBench Resume-Boundary Timeline</h2>',
        '<p class="caption">How to read this: this uses the same visual grammar as the paired report, but with real AgentBench/SWE-bench prompts. Each row is aligned at the black replay-due boundary. Blue is the previous real AgentBench turn, gray is the observed wait gap, and red is the next real AgentBench turn. Purple hint bars and green prefetch-copy bars are absent here because this live direct report does not inject prefetch hints; those appear in the controlled replay-mode reports.</p>',
        replay_svg,
        "</div>",
        '<div class="panel"><h2>Timeline Layers</h2>',
        html_table(layers),
        "</div>",
        '<div class="panel"><h2>Live Phase Overview</h2>',
        '<p class="caption">Supporting context: each colored bar is one real DeepAgents model turn served by SGLang. This overview explains where the replay sessions came from, but it is not the main prefetch-boundary view.</p>',
        timeline_svg,
        "</div>",
        '<div class="panel"><h2>Key Observations</h2>',
        html_table(observations),
        "</div>",
        '<div class="grid">',
        '<div class="panel"><h2>SGLang KV Trace Summary</h2>',
        html_table(kv),
        "</div>",
        '<div class="panel"><h2>KV Copy Telemetry Summary</h2>',
        html_table(copy),
        "</div>",
        "</div>",
        '<div class="panel"><h2>Extracted Replay Sessions</h2>',
        '<p class="caption">These are the real prompts extracted from the live AgentBench phase trace. Milestone 18 reuses these sessions to compare prefetch modes under controlled timing.</p>',
        html_table(replay),
        "</div>",
        '<div class="panel"><h2>Phase-Level Model Turns</h2>',
        html_table(phases),
        "</div>",
        '<div class="panel"><h2>Important Interpretation</h2>',
        '<div class="callout"><strong>What we can claim:</strong> real AgentBench/DeepAgents traffic reached SGLang directly, SGLang served the model turns, and KV/copy telemetry was captured with agent/session context.</div>',
        '<div class="callout warn"><strong>What we should not claim from this report alone:</strong> this live report does not isolate the performance value of prefetching. DeepAgents controls the live tool-loop timing, so controlled mode comparison belongs in the AgentBench replay-mode report.</div>',
        '<p class="caption">Use this report as the realism bridge. Use <code>latest_agentbench_replay_mode_summary.html</code> for the controlled no-prefetch/request-warm/direct-load/oracle-direct-load comparison using these real prompts.</p>',
        "</div>",
        "</body></html>",
    ]
    html_path = out_root / "agentbench_sglang_direct_report.html"
    md_path = out_root / "agentbench_sglang_direct_report.md"
    html_path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# AgentBench Direct SGLang Report",
        "",
        "Runtime path:",
        "",
        "```text",
        "SWE-bench Pro -> AgentBench -> Deep Agents -> SGLang direct -> KV trace/reporting",
        "```",
        "",
        "Dynamo is intentionally removed in this experiment.",
        "",
        "## High-Level Summary",
        "",
        *md_table(summary),
        "## Run Summary",
        "",
        *md_table(runs),
        "## Key Observations",
        "",
        *md_table(observations),
        "## SGLang KV Trace Summary",
        "",
        *md_table(kv),
        "## KV Copy Telemetry Summary",
        "",
        *md_table(copy),
        "## Extracted Replay Sessions",
        "",
        *md_table(replay),
        "## Phase-Level Model Turns",
        "",
        *md_table(phases),
        "## Interpretation",
        "",
        "This live path proves the realistic harness is connected to SGLang directly.",
        "Use trace replay for controlled prefetch comparisons because DeepAgents owns the internal tool loop timing.",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "out_root": str(out_root),
        "html": str(html_path),
        "markdown": str(md_path),
        "run_csv": str(out_root / "agentbench_sglang_runs.csv"),
        "phase_csv": str(out_root / "agentbench_sglang_phase_turns.csv"),
        "kv_summary_csv": str(out_root / "agentbench_sglang_kv_summary.csv"),
        "copy_summary_csv": str(out_root / "agentbench_sglang_copy_summary.csv"),
        "high_level_summary_csv": str(out_root / "agentbench_sglang_high_level_summary.csv"),
        "key_observations_csv": str(out_root / "agentbench_sglang_key_observations.csv"),
        "replay_sessions_csv": str(out_root / "agentbench_sglang_replay_sessions.csv"),
        "trace_path": str(trace_path),
        "copy_telemetry_path": str(copy_path or ""),
        "sections": {
            "high_level_summary": summary,
            "runs": runs,
            "phases": phases,
            "kv_summary": kv,
            "copy_summary": copy,
            "timeline_layers": layers,
            "key_observations": observations,
            "replay_sessions": replay,
        },
    }
    (out_root / "agentbench_sglang_direct_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if latest_root is not None:
        latest_root.mkdir(parents=True, exist_ok=True)
        for src, name in [
            (html_path, "latest_agentbench_sglang_direct_report.html"),
            (md_path, "latest_agentbench_sglang_direct_report.md"),
            (out_root / "agentbench_sglang_direct_report.json", "latest_agentbench_sglang_direct_report.json"),
            (out_root / "agentbench_sglang_runs.csv", "latest_agentbench_sglang_runs.csv"),
            (out_root / "agentbench_sglang_phase_turns.csv", "latest_agentbench_sglang_phase_turns.csv"),
            (out_root / "agentbench_sglang_kv_summary.csv", "latest_agentbench_sglang_kv_summary.csv"),
            (out_root / "agentbench_sglang_copy_summary.csv", "latest_agentbench_sglang_copy_summary.csv"),
            (out_root / "agentbench_sglang_high_level_summary.csv", "latest_agentbench_sglang_high_level_summary.csv"),
            (out_root / "agentbench_sglang_timeline_layers.csv", "latest_agentbench_sglang_timeline_layers.csv"),
            (out_root / "agentbench_sglang_key_observations.csv", "latest_agentbench_sglang_key_observations.csv"),
            (out_root / "agentbench_sglang_replay_sessions.csv", "latest_agentbench_sglang_replay_sessions.csv"),
        ]:
            if src.exists():
                shutil.copyfile(src, latest_root / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize direct-SGLang AgentBench runs.")
    parser.add_argument("--index-csv", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--copy-telemetry", type=Path, default=None)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--latest-root", type=Path, default=None)
    args = parser.parse_args()
    index_rows = read_csv(args.index_csv)
    write_outputs(
        out_root=args.out_root,
        latest_root=args.latest_root,
        index_rows=index_rows,
        trace_path=args.trace,
        copy_path=args.copy_telemetry,
    )
    print(f"Wrote AgentBench direct SGLang report to {args.out_root}")


if __name__ == "__main__":
    main()
