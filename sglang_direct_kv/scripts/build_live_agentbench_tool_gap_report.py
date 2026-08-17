#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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


def read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def short(value: Any, limit: int = 160) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def maybe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def tool_names(row: dict[str, Any]) -> list[str]:
    names = as_list(row.get("normalized_tool_call_names")) or as_list(row.get("response_tool_call_names"))
    return [str(name) for name in names if str(name)]


def tool_count(row: dict[str, Any]) -> int:
    normalized = int(maybe_float(row.get("normalized_tool_call_count")) or 0)
    response = int(maybe_float(row.get("response_tool_call_count")) or 0)
    return max(normalized, response, len(tool_names(row)))


def likely_tool_intent_without_call(row: dict[str, Any]) -> bool:
    if tool_count(row) > 0:
        return False
    if int(maybe_float(row.get("tools_count")) or 0) <= 0:
        return False
    if str(row.get("finish_reason") or "").lower() != "stop":
        return False
    preview = str(row.get("content_preview") or "").lower()
    intent_terms = (
        "inspect",
        "read the",
        "read_file",
        "open the",
        "look at",
        "grep",
        "search",
        "run ",
        "execute",
        "edit",
        "write_file",
        "list",
        "ls ",
        "test",
    )
    return any(term in preview for term in intent_terms)


def request_start(row: dict[str, Any]) -> float:
    value = maybe_float(row.get("request_start_ts"))
    if value is not None:
        return value
    end = maybe_float(row.get("request_end_ts")) or maybe_float(row.get("ts")) or 0.0
    elapsed = (maybe_float(row.get("elapsed_ms")) or 0.0) / 1000.0
    return end - elapsed


def request_end(row: dict[str, Any]) -> float:
    return maybe_float(row.get("request_end_ts")) or maybe_float(row.get("ts")) or request_start(row)


def group_key(row: dict[str, Any], ordinal: int) -> str:
    parent = str(row.get("parent_run_id") or "")
    if parent:
        return parent
    session = str(row.get("agent_session_id") or "")
    if session:
        return session.split("::")[0]
    task = str(row.get("task_instance_id") or "")
    if task:
        return task
    return f"unknown_{ordinal // 1000}"


def normalize_requests(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    base_ts: float | None = None
    for ordinal, row in enumerate(raw_rows):
        if row.get("method") != "POST" or row.get("path") != "/v1/chat/completions":
            continue
        if int(maybe_float(row.get("status")) or 0) >= 400:
            continue
        start = request_start(row)
        end = request_end(row)
        if base_ts is None or start < base_ts:
            base_ts = start
        request_context = as_dict(row.get("request_context"))
        agentic_kv = as_dict(row.get("agentic_kv"))
        agent_hints = as_dict(row.get("agent_hints"))
        context_request_id = str(row.get("request_id") or request_context.get("request_id") or "")
        proxy_request_id = f"{context_request_id}#proxy_row_{ordinal}" if context_request_id else f"proxy_row_{ordinal}"
        requests.append(
            {
                "ordinal": ordinal,
                "request_id": proxy_request_id,
                "context_request_id": context_request_id,
                "parent_run_id": row.get("parent_run_id") or request_context.get("parent_run_id") or "",
                "task_instance_id": row.get("task_instance_id") or request_context.get("task_instance_id") or "",
                "phase": row.get("phase") or request_context.get("phase") or agentic_kv.get("phase") or "",
                "step_title": row.get("step_title") or request_context.get("step_title") or "",
                "sequence_index": row.get("sequence_index", request_context.get("sequence_index", "")),
                "agent_session_id": row.get("agent_session_id") or agentic_kv.get("session_id") or "",
                "model": row.get("model") or "",
                "message_count": row.get("message_count") or "",
                "tools_count": row.get("tools_count") or 0,
                "tool_count": tool_count(row),
                "tool_names": ",".join(tool_names(row)),
                "finish_reason": row.get("finish_reason") or "",
                "content_preview": row.get("content_preview") or "",
                "tool_intent_without_call": likely_tool_intent_without_call(row),
                "elapsed_ms": round(maybe_float(row.get("elapsed_ms")) or max(0.0, (end - start) * 1000.0), 3),
                "start_ts": start,
                "end_ts": end,
                "request_context_present": bool(request_context),
                "agentic_kv_present": bool(agentic_kv),
                "agent_hints_present": bool(agent_hints),
                "hint_priority": row.get("agent_priority") or agent_hints.get("priority") or "",
                "reuse_likelihood": row.get("agent_reuse_likelihood") or agent_hints.get("reuse_likelihood") or "",
            }
        )
    if base_ts is None:
        return []
    for request in requests:
        request["start_ms"] = round((float(request["start_ts"]) - base_ts) * 1000.0, 3)
        request["end_ms"] = round((float(request["end_ts"]) - base_ts) * 1000.0, 3)
        request["group_key"] = group_key(request, int(request["ordinal"]))
    return sorted(requests, key=lambda item: (float(item["start_ts"]), int(item["ordinal"])))


def build_tool_gaps(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        by_group[str(request.get("group_key") or "unknown")].append(request)

    gaps: list[dict[str, Any]] = []
    gap_index = 0
    for group, items in by_group.items():
        items = sorted(items, key=lambda item: (float(item["start_ts"]), int(item["ordinal"])))
        for index, current in enumerate(items[:-1]):
            if int(current.get("tool_count") or 0) <= 0:
                continue
            resume = items[index + 1]
            gap_ms = max(0.0, (float(resume["start_ts"]) - float(current["end_ts"])) * 1000.0)
            session_id = f"live_gap_{gap_index:03d}"
            phase = str(current.get("phase") or "turn")
            next_phase = str(resume.get("phase") or "resume")
            task_id = current.get("task_instance_id") or resume.get("task_instance_id") or ""
            gaps.append(
                {
                    "session_id": session_id,
                    "group_key": group,
                    "parent_run_id": current.get("parent_run_id") or resume.get("parent_run_id") or "",
                    "task_instance_id": task_id,
                    "from_request_id": current.get("request_id"),
                    "to_request_id": resume.get("request_id"),
                    "from_proxy_ordinal": current.get("ordinal"),
                    "to_proxy_ordinal": resume.get("ordinal"),
                    "from_context_request_id": current.get("context_request_id"),
                    "to_context_request_id": resume.get("context_request_id"),
                    "from_phase": phase,
                    "to_phase": next_phase,
                    "tool_names": current.get("tool_names") or "",
                    "tool_call_count": current.get("tool_count") or 0,
                    "tool_gap_ms": round(gap_ms, 3),
                    "current_latency_ms": current.get("elapsed_ms") or "",
                    "resume_latency_ms": resume.get("elapsed_ms") or "",
                    "current_start_ms": current.get("start_ms"),
                    "current_end_ms": current.get("end_ms"),
                    "tool_gap_start_ms": current.get("end_ms"),
                    "tool_gap_end_ms": resume.get("start_ms"),
                    "resume_start_ms": resume.get("start_ms"),
                    "resume_end_ms": resume.get("end_ms"),
                    "current_message_count": current.get("message_count") or "",
                    "resume_message_count": resume.get("message_count") or "",
                    "current_preview": current.get("content_preview") or "",
                    "resume_preview": resume.get("content_preview") or "",
                    "hint_priority": current.get("hint_priority") or "",
                    "reuse_likelihood": current.get("reuse_likelihood") or "",
                }
            )
            gap_index += 1
    return sorted(gaps, key=lambda item: float(item.get("current_start_ms") or 0.0))


def is_preflight_request(row: dict[str, Any]) -> bool:
    parent = str(row.get("parent_run_id") or row.get("group_key") or "")
    task = str(row.get("task_instance_id") or "")
    return parent == "tool_loop_preflight" or task == "tool-loop-diagnostic"


def summary_rows(
    requests: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    captured_request_count: int,
    excluded_preflight_count: int,
    hint_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tool_counter: Counter[str] = Counter()
    for request in requests:
        for name in str(request.get("tool_names") or "").split(","):
            if name:
                tool_counter[name] += 1
    gap_values = [float(row.get("tool_gap_ms") or 0.0) for row in gaps]
    request_context_count = sum(1 for row in requests if row.get("request_context_present"))
    tool_intent_misses = sum(1 for row in requests if row.get("tool_intent_without_call"))
    prefetch_attempts = [
        row
        for row in gaps
        if row.get("prefetch_start_ms") not in ("", None) or row.get("prefetch_end_ms") not in ("", None)
    ]
    margins = [float(row["prefetch_margin_ms"]) for row in gaps if row.get("prefetch_margin_ms") not in ("", None)]
    return [
        {
            "captured_model_requests": captured_request_count,
            "analyzed_model_requests": len(requests),
            "excluded_preflight_requests": excluded_preflight_count,
            "requests_with_tools": sum(1 for row in requests if int(row.get("tool_count") or 0) > 0),
            "total_tool_calls": sum(int(row.get("tool_count") or 0) for row in requests),
            "observed_tool_gaps": len(gaps),
            "avg_observed_tool_gap_ms": round(sum(gap_values) / len(gap_values), 3) if gap_values else 0.0,
            "max_observed_tool_gap_ms": round(max(gap_values), 3) if gap_values else 0.0,
            "requests_with_context": request_context_count,
            "agentbench_tasks_in_index": len(task_rows),
            "tool_intent_without_structured_call": tool_intent_misses,
            "live_hints_submitted": len(hint_rows),
            "controller_events": len(controller_rows),
            "prefetch_attempts_matched_to_gaps": len(prefetch_attempts),
            "prefetch_done_before_resume": sum(1 for row in gaps if row.get("prefetch_done_before_resume") == 1),
            "avg_prefetch_margin_ms": round(sum(margins) / len(margins), 3) if margins else "",
            "top_tools": ", ".join(f"{name}: {count}" for name, count in tool_counter.most_common(8)),
        }
    ]


def seconds_to_ms(ts: Any, base_ts: float) -> str | float:
    value = maybe_float(ts)
    if value is None:
        return ""
    return round((value - base_ts) * 1000.0, 3)


def controller_windows(controller_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    out: dict[str, dict[str, Any]] = {}
    for row in controller_rows:
        hint_id = str(row.get("hint_id") or "")
        if not hint_id:
            continue
        if row.get("event") == "live_prefetch.start":
            starts[hint_id] = row
        elif row.get("event") in {"live_prefetch.end", "live_prefetch.error"}:
            start = starts.get(hint_id, {})
            merged = dict(start)
            merged.update(row)
            out[hint_id] = merged
    return out


def augment_gaps_with_prefetch(
    gaps: list[dict[str, Any]],
    hint_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    base_ts: float,
) -> list[dict[str, Any]]:
    hints_by_ordinal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hint_rows:
        if row.get("event") != "live_hint.submitted":
            continue
        hints_by_ordinal[str(row.get("source_proxy_ordinal"))].append(row)

    windows = controller_windows(controller_rows)
    out: list[dict[str, Any]] = []
    for gap in gaps:
        row = dict(gap)
        hints = hints_by_ordinal.get(str(gap.get("from_proxy_ordinal")), [])
        if hints:
            hint = hints[0]
            hint_id = str(hint.get("hint_id") or "")
            window = windows.get(hint_id, {})
            row["hint_id"] = hint_id
            row["hint_submitted_ms"] = seconds_to_ms(hint.get("ts"), base_ts)
            row["hint_payload_path"] = hint.get("payload_path", "")
            row["prefetch_status"] = "submitted"
            row["prefetch_start_ms"] = seconds_to_ms(window.get("request_start_ts") or window.get("ts"), base_ts)
            row["prefetch_end_ms"] = seconds_to_ms(window.get("request_end_ts"), base_ts)
            row["prefetch_duration_ms"] = window.get("duration_ms", "")
            row["prefetch_error"] = window.get("error", "")
            if window:
                row["prefetch_status"] = "error" if window.get("event") == "live_prefetch.error" else "done"
            prefetch_end = maybe_float(row.get("prefetch_end_ms"))
            resume_start = maybe_float(row.get("resume_start_ms"))
            prefetch_start = maybe_float(row.get("prefetch_start_ms"))
            if prefetch_end is not None and resume_start is not None:
                margin = round(resume_start - prefetch_end, 3)
                row["prefetch_margin_ms"] = margin
                row["prefetch_done_before_resume"] = 1 if margin >= 0 else 0
            if prefetch_start is not None and prefetch_end is not None and resume_start is not None:
                row["prefetch_resume_overlap_ms"] = round(max(0.0, prefetch_end - max(prefetch_start, resume_start)), 3)
        else:
            row["hint_id"] = ""
            row["hint_submitted_ms"] = ""
            row["prefetch_status"] = "no_hint"
            row["prefetch_start_ms"] = ""
            row["prefetch_end_ms"] = ""
            row["prefetch_duration_ms"] = ""
            row["prefetch_error"] = ""
            row["prefetch_margin_ms"] = ""
            row["prefetch_done_before_resume"] = ""
            row["prefetch_resume_overlap_ms"] = ""
        out.append(row)
    return out


def table_html(rows: list[dict[str, Any]], columns: list[str] | None = None, limit: int | None = None) -> str:
    if not rows:
        return "<p>No rows.</p>"
    selected = rows[:limit] if limit is not None else rows
    if columns is None:
        columns = list(selected[0].keys())
    out = ["<div class=\"table-wrap\"><table>", "<thead><tr>"]
    out.extend(f"<th>{fmt(column)}</th>" for column in columns)
    out.append("</tr></thead><tbody>")
    for row in selected:
        out.append("<tr>")
        for column in columns:
            value = row.get(column, "")
            if column.endswith("preview"):
                value = short(value, 140)
            out.append(f"<td>{fmt(value)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def build_timeline_svg(gaps: list[dict[str, Any]], max_rows: int, show_prefetch_legend: bool = True) -> str:
    rows = gaps[:max_rows]
    if not rows:
        return "<p>No live tool-gap timeline available.</p>"
    start_candidates: list[float] = []
    end_candidates: list[float] = []
    for row in rows:
        for key in ("current_start_ms", "prefetch_start_ms", "direct_kv_h2d_start_ms"):
            value = maybe_float(row.get(key))
            if value is not None:
                start_candidates.append(value)
        for key in ("resume_end_ms", "prefetch_end_ms", "direct_kv_h2d_end_ms"):
            value = maybe_float(row.get(key))
            if value is not None:
                end_candidates.append(value)
    start = min(start_candidates or [0.0])
    end = max(end_candidates or [0.0])
    padding = max(100.0, (end - start) * 0.05)
    start -= padding
    end += padding
    span = max(1.0, end - start)
    width = 1580
    left = 380
    right = 60
    top = 78
    row_h = 72
    height = top + len(rows) * row_h + 78
    plot_w = width - left - right

    def x_pos(ms: Any) -> float:
        value = float(ms or 0.0)
        return left + (value - start) / span * plot_w

    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Live AgentBench tool-gap timeline">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    for tick in range(6):
        ms = start + span * tick / 5
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 38}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{ms:.0f} ms</text>')

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = str(row.get("session_id") or f"gap_{idx}")
        title = f"{row.get('from_phase') or 'turn'} -> {row.get('to_phase') or 'resume'}"
        tools = str(row.get("tool_names") or "")
        gap_ms = float(row.get("tool_gap_ms") or 0.0)
        margin = maybe_float(row.get("prefetch_margin_ms"))
        if margin is not None:
            status = f"PREFETCH READY {margin:.0f} ms" if margin >= 0 else f"PREFETCH LATE {abs(margin):.0f} ms"
            status_color = "#166534" if margin >= 0 else "#b91c1c"
        else:
            status = "SHORT GAP" if gap_ms < 500 else "MEDIUM GAP" if gap_ms < 1500 else "LONG GAP"
            status_color = "#b45309" if gap_ms < 500 else "#166534"
        svg.append(f'<text x="10" y="{y + 16}" font-weight="700">{fmt(label)}</text>')
        svg.append(f'<text x="10" y="{y + 36}" font-size="12" fill="#334155">{fmt(title)}</text>')
        svg.append(
            f'<text x="10" y="{y + 54}" font-size="12" fill="{status_color}" font-weight="700">{status} {gap_ms:.0f} ms; tools={fmt(tools)}</text>'
        )
        svg.append(f'<line x1="{left}" y1="{y + 8}" x2="{left + plot_w}" y2="{y + 8}" stroke="#f1f5f9"/>')

        current_x1 = x_pos(row.get("current_start_ms"))
        current_x2 = x_pos(row.get("current_end_ms"))
        wait_x1 = x_pos(row.get("tool_gap_start_ms"))
        wait_x2 = x_pos(row.get("tool_gap_end_ms"))
        replay_x1 = x_pos(row.get("resume_start_ms"))
        replay_x2 = x_pos(row.get("resume_end_ms"))
        prefetch_start = row.get("prefetch_start_ms")
        prefetch_end = row.get("prefetch_end_ms")
        direct_h2d_start = row.get("direct_kv_h2d_start_ms")
        direct_h2d_end = row.get("direct_kv_h2d_end_ms")

        svg.append(
            f'<rect x="{wait_x1:.1f}" y="{y + 4}" width="{max(2, wait_x2 - wait_x1):.1f}" height="30" rx="3" fill="#d1d5db" opacity="0.72">'
            f'<title>observed tool gap {gap_ms:.3f} ms</title></rect>'
        )
        svg.append(
            f'<rect x="{current_x1:.1f}" y="{y}" width="{max(3, current_x2 - current_x1):.1f}" height="26" rx="3" fill="#2563eb" opacity="0.88">'
            f'<title>model turn emitted tool call</title></rect>'
        )
        svg.append(
            f'<rect x="{replay_x1:.1f}" y="{y + 36}" width="{max(3, replay_x2 - replay_x1):.1f}" height="26" rx="3" fill="#ef4444" opacity="0.86">'
            f'<title>next model turn after tool execution</title></rect>'
        )
        svg.append(
            f'<line x1="{replay_x1:.1f}" y1="{y - 4}" x2="{replay_x1:.1f}" y2="{y + 68}" stroke="#111827" stroke-width="3">'
            '<title>resume request starts</title></line>'
        )
        if prefetch_start not in ("", None) and prefetch_end not in ("", None):
            prefetch_x1 = x_pos(prefetch_start)
            prefetch_x2 = x_pos(prefetch_end)
            svg.append(
                f'<rect x="{prefetch_x1:.1f}" y="{y + 23}" width="{max(3, prefetch_x2 - prefetch_x1):.1f}" height="16" rx="3" fill="#a855f7" opacity="0.78">'
                f'<title>live controller prefetch attempt; status={fmt(row.get("prefetch_status"))}</title></rect>'
            )
        if direct_h2d_start not in ("", None) and direct_h2d_end not in ("", None):
            h2d_x1 = x_pos(direct_h2d_start)
            h2d_x2 = x_pos(direct_h2d_end)
            h2d_events = row.get("direct_kv_h2d_events") or ""
            h2d_duration = row.get("direct_kv_h2d_duration_ms") or ""
            svg.append(
                f'<rect x="{h2d_x1:.1f}" y="{y + 20}" width="{max(5, h2d_x2 - h2d_x1):.1f}" height="22" rx="3" fill="#16a34a" opacity="0.92" stroke="#065f46" stroke-width="1">'
                f'<title>direct KV host-to-device movement; events={fmt(h2d_events)}; duration_ms={fmt(h2d_duration)}</title></rect>'
            )
            svg.append(
                f'<text x="{max(left + 2, h2d_x1 + 3):.1f}" y="{y + 35}" font-size="10" fill="#ffffff" font-weight="700">HtoD</text>'
            )
    legend_y = height - 28
    legend = [
        ("model turn with tool call", "#2563eb"),
        ("observed tool gap", "#d1d5db"),
        ("resume model turn", "#ef4444"),
        ("resume start boundary", "#111827"),
    ]
    if show_prefetch_legend:
        legend.insert(2, ("live prefetch attempt", "#a855f7"))
        legend.insert(3, ("direct KV HtoD copy", "#16a34a"))
    lx = left
    for label, color in legend:
        svg.append(f'<rect x="{lx}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y}">{fmt(label)}</text>')
        lx += 220
    svg.append("</svg>")
    return "\n".join(svg)


def build_expanded_gap_timeline_svg(
    gaps: list[dict[str, Any]],
    max_rows: int,
    show_prefetch_legend: bool = True,
) -> str:
    rows = gaps[:max_rows]
    if not rows:
        return "<p>No expanded live tool-gap timeline available.</p>"

    rel_values: list[float] = []
    for row in rows:
        due = maybe_float(row.get("resume_start_ms")) or maybe_float(row.get("tool_gap_end_ms"))
        if due is None:
            continue
        for key in (
            "current_start_ms",
            "current_end_ms",
            "tool_gap_start_ms",
            "tool_gap_end_ms",
            "prefetch_start_ms",
            "prefetch_end_ms",
            "direct_kv_h2d_start_ms",
            "direct_kv_h2d_end_ms",
        ):
            value = maybe_float(row.get(key))
            if value is not None:
                rel_values.append(value - due)
        replay_start = maybe_float(row.get("resume_start_ms"))
        replay_end = maybe_float(row.get("resume_end_ms"))
        if replay_start is not None:
            rel_values.append(replay_start - due)
        if replay_end is not None:
            rel_values.append(min(600.0, replay_end - due))

    start = min(rel_values or [-500.0])
    end = max(rel_values or [500.0])
    start = min(start - 60.0, -120.0)
    end = max(end + 80.0, 220.0)
    span = max(1.0, end - start)
    width = 1580
    left = 380
    right = 70
    top = 86
    row_h = 76
    height = top + len(rows) * row_h + 92
    plot_w = width - left - right

    def rel(row: dict[str, Any], key: str, due: float) -> float | None:
        value = maybe_float(row.get(key))
        if value is None:
            return None
        return value - due

    def x_pos(relative_ms: float) -> float:
        return left + (relative_ms - start) / span * plot_w

    def rect(
        svg: list[str],
        x1: float,
        x2: float,
        y: float,
        h: float,
        color: str,
        title: str,
        opacity: float = 0.88,
        min_w: float = 3.0,
    ) -> None:
        width_px = max(min_w, x2 - x1)
        svg.append(
            f'<rect x="{x1:.1f}" y="{y:.1f}" width="{width_px:.1f}" height="{h:.1f}" rx="3" '
            f'fill="{color}" opacity="{opacity}"><title>{fmt(title)}</title></rect>'
        )

    zero_x = x_pos(0.0)
    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Expanded per-gap timeline aligned at replay due">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]

    tick_count = 7
    for tick in range(tick_count):
        ms = start + span * tick / (tick_count - 1)
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 42}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{ms:.0f} ms</text>')
    svg.append(f'<line x1="{zero_x:.1f}" y1="{top - 40}" x2="{zero_x:.1f}" y2="{height - 42}" stroke="#111827" stroke-width="3"/>')
    svg.append(f'<text x="{zero_x + 7:.1f}" y="{top - 48}" font-size="12" font-weight="700">0 ms replay due</text>')
    svg.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="13" font-weight="700">local time around each gap: negative = before replay due, positive = after replay due</text>'
    )

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        due = maybe_float(row.get("resume_start_ms")) or maybe_float(row.get("tool_gap_end_ms"))
        if due is None:
            continue
        label = str(row.get("session_id") or f"gap_{idx}")
        tools = str(row.get("tool_names") or "")
        gap_ms = maybe_float(row.get("tool_gap_ms")) or 0.0
        margin = maybe_float(row.get("prefetch_margin_ms"))
        if margin is not None:
            status = f"READY +{margin:.0f} ms" if margin >= 0 else f"LATE -{abs(margin):.0f} ms"
            status_color = "#166534" if margin >= 0 else "#b91c1c"
        else:
            status = "NO PREFETCH"
            status_color = "#64748b"
        svg.append(f'<text x="10" y="{y + 16}" font-weight="700">{fmt(label)}</text>')
        svg.append(f'<text x="10" y="{y + 36}" font-size="12" fill="{status_color}" font-weight="700">{fmt(status)}</text>')
        svg.append(f'<text x="10" y="{y + 55}" font-size="12" fill="#334155">gap={gap_ms:.0f} ms; tools={fmt(tools)}</text>')
        svg.append(f'<line x1="{left}" y1="{y + 9}" x2="{left + plot_w}" y2="{y + 9}" stroke="#f1f5f9"/>')

        current_start = rel(row, "current_start_ms", due)
        current_end = rel(row, "current_end_ms", due)
        wait_start = rel(row, "tool_gap_start_ms", due)
        wait_end = rel(row, "tool_gap_end_ms", due)
        prefetch_start = rel(row, "prefetch_start_ms", due)
        prefetch_end = rel(row, "prefetch_end_ms", due)
        h2d_start = rel(row, "direct_kv_h2d_start_ms", due)
        h2d_end = rel(row, "direct_kv_h2d_end_ms", due)
        replay_start = rel(row, "resume_start_ms", due)
        replay_end = rel(row, "resume_end_ms", due)

        if wait_start is not None and wait_end is not None:
            rect(svg, x_pos(wait_start), x_pos(wait_end), y + 5, 32, "#d1d5db", f"tool wait window {gap_ms:.3f} ms", 0.72, 2)
        if current_start is not None and current_end is not None:
            rect(svg, x_pos(current_start), x_pos(current_end), y + 1, 26, "#2563eb", "initial model turn that emitted tool call", 0.88, 3)
        if show_prefetch_legend and prefetch_start is not None and prefetch_end is not None:
            rect(
                svg,
                x_pos(prefetch_start),
                x_pos(prefetch_end),
                y + 27,
                18,
                "#a855f7",
                f"prefetch attempt; status={row.get('prefetch_status', '')}; duration_ms={row.get('prefetch_duration_ms', '')}",
                0.78,
                3,
            )
            svg.append(f'<text x="{x_pos(prefetch_start):.1f}" y="{y + 24}" text-anchor="middle" font-size="9" fill="#6d28d9" font-weight="700">hint start</text>')
            svg.append(f'<text x="{x_pos(prefetch_end):.1f}" y="{y + 58}" text-anchor="middle" font-size="9" fill="#6d28d9" font-weight="700">hint end</text>')
        if show_prefetch_legend and h2d_start is not None and h2d_end is not None:
            rect(
                svg,
                x_pos(h2d_start),
                x_pos(h2d_end),
                y + 21,
                30,
                "#16a34a",
                f"direct KV HtoD movement; events={row.get('direct_kv_h2d_events', '')}; duration_ms={row.get('direct_kv_h2d_duration_ms', '')}",
                0.94,
                14,
            )
            svg.append(f'<text x="{x_pos(h2d_start) + 4:.1f}" y="{y + 40}" font-size="10" fill="#ffffff" font-weight="700">HtoD</text>')
        if replay_start is not None:
            replay_display_end = replay_start + 260.0
            continues = False
            if replay_end is not None:
                replay_display_end = min(replay_end, replay_start + 520.0)
                continues = replay_end > replay_display_end
            rect(svg, x_pos(replay_start), x_pos(replay_display_end), y + 48, 22, "#ef4444", "resume model request after tool result", 0.82, 4)
            if continues:
                svg.append(f'<text x="{x_pos(replay_display_end) - 52:.1f}" y="{y + 44}" font-size="10" fill="#991b1b" font-weight="700">continues</text>')
        if margin is not None and prefetch_end is not None:
            y_margin = y + 68
            x_done = x_pos(prefetch_end)
            color = "#16a34a" if margin >= 0 else "#dc2626"
            label = f"{margin:.0f} ms early" if margin >= 0 else f"{abs(margin):.0f} ms late"
            svg.append(
                f'<line x1="{min(x_done, zero_x):.1f}" y1="{y_margin:.1f}" x2="{max(x_done, zero_x):.1f}" y2="{y_margin:.1f}" '
                f'stroke="{color}" stroke-width="3" stroke-dasharray="7 5"/>'
            )
            svg.append(f'<circle cx="{x_done:.1f}" cy="{y_margin:.1f}" r="4" fill="{color}"/>')
            svg.append(f'<text x="{(x_done + zero_x) / 2:.1f}" y="{y_margin + 13:.1f}" text-anchor="middle" font-size="10" fill="{color}" font-weight="700">{label}</text>')

    legend_y = height - 48
    legend = [
        ("initial model turn", "#2563eb"),
        ("tool wait", "#d1d5db"),
        ("replay due", "#111827"),
        ("resume request", "#ef4444"),
    ]
    if show_prefetch_legend:
        legend.insert(2, ("prefetch attempt", "#a855f7"))
        legend.insert(3, ("direct KV HtoD", "#16a34a"))
    lx = left
    for label, color in legend:
        if label == "replay due":
            svg.append(f'<line x1="{lx}" y1="{legend_y - 12}" x2="{lx}" y2="{legend_y + 4}" stroke="{color}" stroke-width="4"/>')
        else:
            svg.append(f'<rect x="{lx}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y}">{fmt(label)}</text>')
        lx += 185
    svg.append("</svg>")
    return "\n".join(svg)


def key_observations(requests: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if requests:
        rows.append(
            {
                "observation": "Real Deep Agents traffic reached direct SGLang.",
                "evidence": f"{len(requests)} live OpenAI-compatible chat requests were captured by the proxy.",
                "deduction": "The feed is no longer synthetic; it is generated by the real AgentBench/Deep Agents harness.",
            }
        )
    if gaps:
        shortest = min(gaps, key=lambda row: float(row.get("tool_gap_ms") or 0.0))
        longest = max(gaps, key=lambda row: float(row.get("tool_gap_ms") or 0.0))
        rows.append(
            {
                "observation": "Tool calls create measurable prefetch windows.",
                "evidence": f"{len(gaps)} model-to-model gaps after tool calls were found; shortest={shortest.get('tool_gap_ms')} ms, longest={longest.get('tool_gap_ms')} ms.",
                "deduction": "These gaps are the live opportunity windows where a future hint-guided KV prefetch path would act.",
            }
        )
        short_count = sum(1 for row in gaps if float(row.get("tool_gap_ms") or 0.0) < 500.0)
        rows.append(
            {
                "observation": "Some tool gaps can be too short for best-effort software.",
                "evidence": f"{short_count}/{len(gaps)} observed tool gaps were under 500 ms.",
                "deduction": "Short gaps make deadline-aware scheduling important because prefetch work must start and finish quickly.",
            }
        )
    misses = sum(1 for row in requests if row.get("tool_intent_without_call"))
    if misses:
        rows.append(
            {
                "observation": "Some model turns looked like tool intent but did not produce structured tool calls.",
                "evidence": f"{misses} tool-capable turns returned plain text with inspect/read/run/edit-style intent.",
                "deduction": "This is a parser/model/tool-loop health signal; if this number is high, the run is not yet a faithful tool-heavy AgentBench workload.",
            }
        )
    if any(row.get("request_context_present") for row in requests):
        rows.append(
            {
                "observation": "Request context is preserved in the live path.",
                "evidence": "The proxy captured request_context / agentic_kv / agent_hints fields from direct SGLang requests.",
                "deduction": "Future experiments can attach prefetch decisions to task, phase, session, priority, and reuse-likelihood metadata.",
            }
        )
    prefetch_attempts = [row for row in gaps if row.get("prefetch_start_ms") not in ("", None)]
    if prefetch_attempts:
        ready = sum(1 for row in prefetch_attempts if row.get("prefetch_done_before_resume") == 1)
        rows.append(
            {
                "observation": "Live software prefetch attempts were issued from tool-call hints.",
                "evidence": f"{len(prefetch_attempts)} prefetch attempts matched live tool gaps; {ready}/{len(prefetch_attempts)} finished before the resume request.",
                "deduction": "This is the first live intervention path: tool-call hint -> controller prefetch request -> next real agent turn.",
            }
        )
    return rows


def render_markdown(summary: list[dict[str, Any]], observations: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> str:
    lines = [
        "# Live AgentBench Tool-Gap Report",
        "",
        "This report is the observe-only bridge between real Deep Agents tool calls and the KV-prefetch analysis path.",
        "",
        "## Summary",
        "",
    ]
    for key, value in (summary[0] if summary else {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Key Observations", ""])
    for row in observations:
        lines.append(f"- **{row['observation']}** {row['evidence']} {row['deduction']}")
    lines.extend(["", "## Tool Gaps", ""])
    lines.append("| session | phases | tools | gap_ms | prefetch_status | prefetch_margin_ms | current_ms | resume_ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in gaps:
        lines.append(
            "| {session} | {from_phase} -> {to_phase} | {tools} | {gap} | {status} | {margin} | {current} | {resume} |".format(
                session=row.get("session_id", ""),
                from_phase=row.get("from_phase", ""),
                to_phase=row.get("to_phase", ""),
                tools=row.get("tool_names", ""),
                gap=row.get("tool_gap_ms", ""),
                status=row.get("prefetch_status", ""),
                margin=row.get("prefetch_margin_ms", ""),
                current=row.get("current_latency_ms", ""),
                resume=row.get("resume_latency_ms", ""),
            )
        )
    return "\n".join(lines) + "\n"


def render_html(
    summary: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    max_timeline_gaps: int,
) -> str:
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
    main { max-width: 1760px; margin: 0 auto; padding: 24px; }
    section { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px 24px; margin: 18px 0; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
    h1 { font-size: 34px; margin: 0 0 8px; }
    h2 { font-size: 26px; margin: 0 0 12px; }
    p { color: #334155; line-height: 1.45; }
    .note { background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 14px; color: #1e3a8a; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th { text-align: left; background: #f1f5f9; color: #111827; padding: 9px 10px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; }
    td { padding: 9px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }
    code { background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }
    svg text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }
    """
    gap_columns = [
        "session_id",
        "parent_run_id",
        "from_phase",
        "to_phase",
        "tool_names",
        "tool_gap_ms",
        "current_latency_ms",
        "resume_latency_ms",
        "prefetch_status",
        "prefetch_duration_ms",
        "prefetch_margin_ms",
        "prefetch_done_before_resume",
        "prefetch_resume_overlap_ms",
        "hint_priority",
        "reuse_likelihood",
    ]
    request_columns = [
        "ordinal",
        "parent_run_id",
        "context_request_id",
        "phase",
        "message_count",
        "tool_count",
        "tool_names",
        "tool_intent_without_call",
        "elapsed_ms",
        "request_context_present",
        "agentic_kv_present",
        "agent_hints_present",
        "content_preview",
    ]
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live AgentBench Tool-Gap Report</title>
  <style>{css}</style>
</head>
<body>
<main>
  <section>
    <h1>Live AgentBench Tool-Gap Report</h1>
    <p>This is the live bridge from real SWE-bench/Deep Agents traffic into the KV-prefetch analysis infrastructure.</p>
    <p class="note">Mode: live analysis. Blue bars are real model turns that emitted tool calls. Gray bars are observed tool execution / harness wait gaps. Red bars are the next live model turns. If live intervention is enabled, purple bars show the controller's prefetch/direct-load request. Deep Agents preflight rows are excluded by default so the timeline focuses on the SWE-bench task traffic.</p>
  </section>
  <section>
    <h2>Summary</h2>
    {table_html(summary)}
  </section>
  <section>
    <h2>Timeline</h2>
    <p>How to read this: blue is the model turn that produced one or more structured tool calls. Gray is the tool wait window, where the tool or harness is running and the model is idle for this session. Purple is the prefetch attempt window: detect the tool-call gap, create a hint for that agent/session, call our direct SGLang KV hook, let SGLang check whether host-side KV exists, and if needed, ask SGLang to move KV back to GPU memory. Green is direct KV host-to-device movement observed for that hint. Black is the resume boundary, when the next model turn is due. Red is the resumed model request after the tool result. If purple or green ends after the black boundary, the prefetch path was late.</p>
    {build_timeline_svg(gaps, max_timeline_gaps)}
    <h3>Expanded Per-Gap View</h3>
    <p>This is the same data with each row aligned to its own replay boundary. The black line is always 0 ms. Negative time is before the agent resumes; positive time is after the resume boundary.</p>
    {build_expanded_gap_timeline_svg(gaps, max_timeline_gaps)}
  </section>
  <section>
    <h2>Key Observations</h2>
    {table_html(observations, ["observation", "evidence", "deduction"])}
  </section>
  <section>
    <h2>Observed Tool Gaps</h2>
    {table_html(gaps, gap_columns)}
  </section>
  <section>
    <h2>Live Request Details</h2>
    {table_html(requests, request_columns, limit=80)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a live AgentBench tool-gap report from the direct-SGLang proxy log.")
    parser.add_argument("--proxy-jsonl", type=Path, required=True)
    parser.add_argument("--task-index-csv", type=Path)
    parser.add_argument("--hint-log", type=Path)
    parser.add_argument("--controller-log", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--max-timeline-gaps", type=int, default=16)
    parser.add_argument(
        "--include-preflight",
        action="store_true",
        help="Include Deep Agents tool-loop preflight rows in the timeline and summary.",
    )
    args = parser.parse_args()

    raw_rows = read_jsonl(args.proxy_jsonl)
    hint_rows = read_jsonl(args.hint_log) if args.hint_log else []
    controller_rows = read_jsonl(args.controller_log) if args.controller_log else []
    task_rows = read_csv(args.task_index_csv)
    all_requests = normalize_requests(raw_rows)
    if args.include_preflight:
        requests = all_requests
        excluded_preflight_count = 0
    else:
        requests = [row for row in all_requests if not is_preflight_request(row)]
        excluded_preflight_count = len(all_requests) - len(requests)
    gaps = build_tool_gaps(requests)
    base_ts = min((float(row["start_ts"]) for row in all_requests), default=0.0)
    gaps = augment_gaps_with_prefetch(gaps, hint_rows, controller_rows, base_ts)
    summary = summary_rows(
        requests,
        gaps,
        task_rows,
        len(all_requests),
        excluded_preflight_count,
        hint_rows,
        controller_rows,
    )
    observations = key_observations(requests, gaps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "live_requests.csv", requests)
    write_csv(args.out_dir / "live_tool_gaps.csv", gaps)
    write_csv(args.out_dir / "live_summary.csv", summary)
    write_csv(args.out_dir / "live_key_observations.csv", observations)
    with (args.out_dir / "live_tool_gaps.jsonl").open("w", encoding="utf-8") as handle:
        for row in gaps:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_json = {
        "proxy_jsonl": str(args.proxy_jsonl),
        "task_index_csv": str(args.task_index_csv or ""),
        "hint_log": str(args.hint_log or ""),
        "controller_log": str(args.controller_log or ""),
        "summary": summary[0] if summary else {},
        "key_observations": observations,
        "tool_gaps": gaps,
        "requests": requests,
    }
    (args.out_dir / "live_agentbench_tool_gap_report.json").write_text(
        json.dumps(report_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    md = render_markdown(summary, observations, gaps)
    html_text = render_html(summary, observations, gaps, requests, args.max_timeline_gaps)
    md_path = args.out_dir / "live_agentbench_tool_gap_report.md"
    html_path = args.out_dir / "live_agentbench_tool_gap_report.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")

    if args.latest_root:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        latest_real = args.latest_root / "latest_real"
        latest_real.mkdir(parents=True, exist_ok=True)
        latest_pairs = [
            (html_path, "m22_live_tool_gap_report.html"),
            (md_path, "m22_live_tool_gap_report.md"),
            (args.out_dir / "live_agentbench_tool_gap_report.json", "m22_live_tool_gap_report.json"),
            (args.out_dir / "live_tool_gaps.csv", "m22_live_tool_gaps.csv"),
            (args.out_dir / "live_requests.csv", "m22_live_requests.csv"),
        ]
        if hint_rows or controller_rows:
            latest_pairs.extend(
                [
                    (html_path, "m23_live_prefetch_report.html"),
                    (md_path, "m23_live_prefetch_report.md"),
                    (
                        args.out_dir / "live_agentbench_tool_gap_report.json",
                        "m23_live_prefetch_report.json",
                    ),
                    (args.out_dir / "live_tool_gaps.csv", "m23_live_tool_gaps.csv"),
                    (args.out_dir / "live_requests.csv", "m23_live_requests.csv"),
                ]
            )
        for source, name in latest_pairs:
            if source.exists():
                shutil.copyfile(source, latest_real / name)

    print(f"Wrote live AgentBench tool-gap report: {html_path}")
    print(f"Live model requests: {len(requests)}")
    print(f"Observed tool gaps: {len(gaps)}")


if __name__ == "__main__":
    main()
