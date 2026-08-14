#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any

from correlate_torch_profile_with_agent_trace import KV_EVENTS
from correlate_torch_profile_with_agent_trace import annotate_kv_windows_with_agent
from correlate_torch_profile_with_agent_trace import annotate_kv_windows_with_request_maps
from correlate_torch_profile_with_agent_trace import build_request_maps
from correlate_torch_profile_with_agent_trace import kv_columns
from correlate_torch_profile_with_agent_trace import paired_method_windows
from correlate_torch_profile_with_agent_trace import read_jsonl
from correlate_torch_profile_with_agent_trace import request_windows


def ns_to_ms(ts_ns: int, start_ns: int) -> float:
    return round((ts_ns - start_ns) / 1_000_000, 3)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def first_event(events: list[dict[str, Any]], name: str, session_id: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") == name and event.get("session_id") == session_id:
            return event
    return None


def request_window(events: list[dict[str, Any]], session_id: str, phase: str) -> tuple[int | None, int | None]:
    start = first_event([event for event in events if event.get("phase") == phase], "agent.request.start", session_id)
    end = first_event([event for event in events if event.get("phase") == phase], "agent.request.end", session_id)
    if not start or not end:
        return None, None
    return int(start["ts_ns"]), int(end["ts_ns"])


def build_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows = request_windows(events)
    windows.extend(paired_method_windows(events, KV_EVENTS))
    annotate_kv_windows_with_agent(windows)
    node_map, index_map = build_request_maps(windows)
    annotate_kv_windows_with_request_maps(windows, node_map, index_map)
    return windows


def session_ids(events: list[dict[str, Any]]) -> list[str]:
    ids = {
        str(event["session_id"])
        for event in events
        if isinstance(event.get("session_id"), str) and str(event.get("session_id")).startswith("agent_")
    }
    return sorted(ids)


def selected_kv_windows(windows: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    preferred_events = {"hostpool.load_to_device_per_layer"}
    fallback_events = {"hicache.start_loading", "hicache.load"}
    for event_names in (preferred_events, fallback_events):
        for window in windows:
            cols = kv_columns(window)
            if (
                window.get("window_type") == "sglang_kv_method"
                and window.get("event") in event_names
                and cols.get("kv_direction") == "host_to_device"
                and cols.get("kv_agent_session_id") == session_id
                and cols.get("kv_agent_phase") == "hint_prefetch"
            ):
                out.append(window)
        if out:
            break
    return sorted(out, key=lambda item: int(item["start_ns"]))


def read_copy_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def overlaps_ms(row: dict[str, Any], start_ms: float, end_ms: float) -> bool:
    row_start = to_float(row.get("start_ms_from_trace_start"))
    row_end = to_float(row.get("end_ms_from_trace_start"))
    if row_start is None or row_end is None:
        return False
    return row_start < end_ms and row_end > start_ms


def selected_copy_rows(
    rows: list[dict[str, Any]],
    session_id: str,
    kv_windows: list[dict[str, Any]],
    trace_start_ns: int,
) -> list[dict[str, Any]]:
    kv_windows_ms = [
        (ns_to_ms(int(window["start_ns"]), trace_start_ns), ns_to_ms(int(window["end_ns"]), trace_start_ns))
        for window in kv_windows
    ]

    def matches_session(row: dict[str, Any]) -> bool:
        if row.get("overlap_kv_agent_session_id") == session_id and row.get("overlap_kv_agent_phase") == "hint_prefetch":
            return True
        if row.get("enclosing_agent_session_id") == session_id and row.get("enclosing_agent_phase") == "hint_prefetch":
            return True
        if row.get("overlap_kv_agent_session_id"):
            return False
        return any(overlaps_ms(row, start_ms, end_ms) for start_ms, end_ms in kv_windows_ms)

    out = [
        row
        for row in rows
        if row.get("direction") == "h2d"
        and row.get("overlap_event") == "hostpool.load_to_device_per_layer"
        and matches_session(row)
    ]
    return sorted(out, key=lambda row: float(row.get("start_ms_from_trace_start") or 0.0))


def event_ms(events: list[dict[str, Any]], name: str, session_id: str, trace_start_ns: int) -> float | None:
    event = first_event(events, name, session_id)
    if not event or not event.get("ts_ns"):
        return None
    return ns_to_ms(int(event["ts_ns"]), trace_start_ns)


def phase_metric(events: list[dict[str, Any]], session_id: str, phase: str, key: str) -> Any:
    for event in events:
        if event.get("event") == "agent.request.end" and event.get("session_id") == session_id and event.get("phase") == phase:
            return event.get(key)
    return ""


def build_rows(events: list[dict[str, Any]], copy_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_start_ns = min(int(event["ts_ns"]) for event in events if event.get("ts_ns"))
    windows = build_windows(events)
    rows: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []

    for session_id in session_ids(events):
        arrival = first_event(events, "agent.session_arrival", session_id) or {}
        initial_start, initial_end = request_window(events, session_id, "initial_turn")
        hint_start, hint_end = request_window(events, session_id, "hint_prefetch")
        replay_start, replay_end = request_window(events, session_id, "replay")
        hint_submitted_ms = event_ms(events, "agent.hint_submitted", session_id, trace_start_ns)
        tool_wait_start_ms = event_ms(events, "agent.tool_wait_start", session_id, trace_start_ns)
        replay_due_ms = event_ms(events, "agent.replay_due", session_id, trace_start_ns)
        resume_start_ms = event_ms(events, "agent.resume_start", session_id, trace_start_ns)

        kv = selected_kv_windows(windows, session_id)
        sglang_start_ms = ns_to_ms(int(kv[0]["start_ns"]), trace_start_ns) if kv else None
        sglang_end_ms = ns_to_ms(max(int(item["end_ns"]) for item in kv), trace_start_ns) if kv else None
        sglang_event_count = len(kv)
        sglang_bytes_like = ""
        if kv:
            counts = [kv_columns(item).get("host_index_count") for item in kv if kv_columns(item).get("host_index_count")]
            if counts:
                sglang_bytes_like = str(counts[0])

        copies = selected_copy_rows(copy_rows, session_id, kv, trace_start_ns)
        torch_start_ms = to_float(copies[0].get("start_ms_from_trace_start")) if copies else None
        torch_end_ms = max((to_float(row.get("end_ms_from_trace_start")) or 0.0 for row in copies), default=None)
        torch_copy_count = len(copies)
        torch_bytes = sum(int(float(row.get("bytes") or 0)) for row in copies)

        prefetch_done_ms = torch_end_ms if torch_end_ms is not None else sglang_end_ms
        prefetch_margin_ms = (
            round(float(replay_due_ms) - float(prefetch_done_ms), 3)
            if replay_due_ms is not None and prefetch_done_ms is not None
            else None
        )
        late_prefetch = prefetch_margin_ms is not None and prefetch_margin_ms < 0
        no_visible_prefetch = prefetch_done_ms is None

        rows.append(
            {
                "session_id": session_id,
                "priority": arrival.get("priority", ""),
                "arrival_ms": arrival.get("arrival_ms", ""),
                "tool_wait_ms": arrival.get("tool_wait_ms", ""),
                "prompt_tokens": arrival.get("prompt_tokens", ""),
                "hint_submitted_ms": hint_submitted_ms if hint_submitted_ms is not None else "",
                "sglang_copy_start_ms": sglang_start_ms if sglang_start_ms is not None else "",
                "sglang_copy_end_ms": sglang_end_ms if sglang_end_ms is not None else "",
                "sglang_copy_events": sglang_event_count,
                "sglang_host_index_count": sglang_bytes_like,
                "torch_copy_start_ms": torch_start_ms if torch_start_ms is not None else "",
                "torch_copy_end_ms": torch_end_ms if torch_end_ms is not None else "",
                "torch_h2d_copy_events": torch_copy_count,
                "torch_h2d_bytes": torch_bytes,
                "replay_due_ms": replay_due_ms if replay_due_ms is not None else "",
                "resume_start_ms": resume_start_ms if resume_start_ms is not None else "",
                "prefetch_margin_ms": prefetch_margin_ms if prefetch_margin_ms is not None else "",
                "late_prefetch": late_prefetch,
                "no_visible_prefetch": no_visible_prefetch,
                "replay_ttft_ms": phase_metric(events, session_id, "replay", "ttft_ms"),
            }
        )

        def add_bar(kind: str, start: int | float | None, end: int | float | None, label: str) -> None:
            if start is None or end is None:
                return
            timeline.append({"session_id": session_id, "kind": kind, "start_ms": round(float(start), 3), "end_ms": round(float(end), 3), "label": label})

        def add_marker(kind: str, at: int | float | None, label: str) -> None:
            if at is None:
                return
            timeline.append({"session_id": session_id, "kind": kind, "start_ms": round(float(at), 3), "end_ms": round(float(at), 3), "label": label})

        add_bar("initial", ns_to_ms(initial_start, trace_start_ns) if initial_start else None, ns_to_ms(initial_end, trace_start_ns) if initial_end else None, "initial")
        add_bar("tool_wait", tool_wait_start_ms, replay_due_ms, "tool wait")
        add_marker("hint_submitted", hint_submitted_ms, "hint")
        add_bar("hint_request", ns_to_ms(hint_start, trace_start_ns) if hint_start else None, ns_to_ms(hint_end, trace_start_ns) if hint_end else None, "hint request")
        add_bar("sglang_copy", sglang_start_ms, sglang_end_ms, "SGLang KV load")
        add_bar("torch_copy", torch_start_ms, torch_end_ms, "CUDA HtoD copies")
        add_marker("replay_due", replay_due_ms, "replay due")
        add_bar("replay", ns_to_ms(replay_start, trace_start_ns) if replay_start else None, ns_to_ms(replay_end, trace_start_ns) if replay_end else None, "replay")

    return rows, timeline


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sessions": rows, "timeline": timeline}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def choose_sessions(rows: list[dict[str, Any]], max_sessions: int) -> list[str]:
    late = [row for row in rows if row.get("late_prefetch") is True]
    visible = [row for row in rows if row.get("torch_h2d_copy_events", 0)]
    ordered: list[str] = []
    for group in (late, visible, rows):
        for row in sorted(group, key=lambda item: float(item.get("prefetch_margin_ms") or 1_000_000)):
            sid = str(row["session_id"])
            if sid not in ordered:
                ordered.append(sid)
            if len(ordered) >= max_sessions:
                return ordered
    return ordered


def write_html(path: Path, rows: list[dict[str, Any]], timeline: list[dict[str, Any]], max_sessions: int) -> None:
    selected = choose_sessions(rows, max_sessions)
    selected_rows = [row for row in rows if row["session_id"] in selected]
    selected_timeline = [item for item in timeline if item["session_id"] in selected]
    if selected_timeline:
        start = min(float(item["start_ms"]) for item in selected_timeline)
        end = max(float(item["end_ms"]) for item in selected_timeline)
    else:
        start, end = 0.0, 1.0
    span = max(1.0, end - start)
    width = 1200
    left = 150
    right = 40
    row_h = 44
    top = 60
    height = top + row_h * max(1, len(selected)) + 70
    plot_w = width - left - right
    colors = {
        "initial": "#2563eb",
        "tool_wait": "#d1d5db",
        "hint_submitted": "#7c3aed",
        "hint_request": "#a855f7",
        "sglang_copy": "#f59e0b",
        "torch_copy": "#16a34a",
        "replay_due": "#111827",
        "replay": "#dc2626",
    }

    def x_pos(ms: float) -> float:
        return left + (ms - start) / span * plot_w

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
        svg.append(f'<text x="10" y="{y + 18}" font-weight="700">{html.escape(sid)}</text>')
        svg.append(f'<line x1="{left}" y1="{y + 12}" x2="{left + plot_w}" y2="{y + 12}" stroke="#f3f4f6"/>')
    for item in selected_timeline:
        sid = item["session_id"]
        y = top + row_index[sid] * row_h
        kind = item["kind"]
        color = colors.get(kind, "#6b7280")
        x1 = x_pos(float(item["start_ms"]))
        x2 = x_pos(float(item["end_ms"]))
        if x1 == x2:
            svg.append(f'<line x1="{x1:.1f}" y1="{y - 2}" x2="{x1:.1f}" y2="{y + 26}" stroke="{color}" stroke-width="3"/>')
        else:
            svg.append(f'<rect x="{x1:.1f}" y="{y}" width="{max(2, x2 - x1):.1f}" height="24" rx="3" fill="{color}" opacity="0.88"><title>{html.escape(item["label"])}</title></rect>')
    legend_x = left
    legend_y = height - 32
    for idx, (kind, color) in enumerate(colors.items()):
        lx = legend_x + idx * 130
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}">{html.escape(kind)}</text>')
    svg.append("</svg>")

    late_count = sum(1 for row in rows if row.get("late_prefetch") is True)
    visible_count = sum(1 for row in rows if row.get("torch_h2d_copy_events", 0))
    margins = [float(row["prefetch_margin_ms"]) for row in rows if row.get("prefetch_margin_ms") not in ("", None)]
    summary = {
        "sessions": len(rows),
        "sessions_with_visible_h2d_copy": visible_count,
        "late_prefetch_sessions": late_count,
        "avg_prefetch_margin_ms": round(mean(margins), 3) if margins else "",
    }
    columns = [
        "session_id",
        "priority",
        "tool_wait_ms",
        "prompt_tokens",
        "hint_submitted_ms",
        "sglang_copy_start_ms",
        "sglang_copy_end_ms",
        "torch_copy_start_ms",
        "torch_copy_end_ms",
        "torch_h2d_copy_events",
        "torch_h2d_bytes",
        "replay_due_ms",
        "prefetch_margin_ms",
        "late_prefetch",
        "replay_ttft_ms",
    ]
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Agentic Prefetch Timeline</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}",
        "h1,h2{margin:0 0 12px}",
        ".panel{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin:16px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;white-space:nowrap}",
        "th{background:#f3f4f6;font-weight:700}",
        ".bad{color:#b91c1c;font-weight:700}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Agentic Prefetch Timeline</h1>",
        '<div class="panel"><h2>Summary</h2><table><tbody>',
    ]
    for key, value in summary.items():
        lines.append(f"<tr><th>{html.escape(key)}</th><td>{fmt(value)}</td></tr>")
    lines.extend(["</tbody></table></div>", '<div class="panel"><h2>Timeline</h2>', *svg, "</div>"])
    lines.append('<div class="panel"><h2>Session Details</h2><table><thead><tr>')
    for col in columns:
        lines.append(f"<th>{html.escape(col)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in selected_rows:
        cls = ' class="bad"' if row.get("late_prefetch") else ""
        lines.append(f"<tr{cls}>")
        for col in columns:
            lines.append(f"<td>{fmt(row.get(col, ''))}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody></table></div>", "</body>", "</html>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-session agentic KV prefetch timeline.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--copy-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--max-sessions", type=int, default=12)
    args = parser.parse_args()

    events = read_jsonl(Path(args.trace))
    copy_rows = read_copy_rows(Path(args.copy_csv))
    rows, timeline = build_rows(events, copy_rows)
    write_csv(Path(args.out_csv), rows)
    write_json(Path(args.out_json), rows, timeline)
    write_html(Path(args.out_html), rows, timeline, args.max_sessions)
    print(f"Wrote timeline CSV to {args.out_csv}")
    print(f"Wrote timeline JSON to {args.out_json}")
    print(f"Wrote timeline HTML to {args.out_html}")
    print(f"Sessions: {len(rows)}")
    print(f"Late prefetch sessions: {sum(1 for row in rows if row.get('late_prefetch') is True)}")


if __name__ == "__main__":
    main()
