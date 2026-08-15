#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
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
    visible_h2d = sum(1 for row in rows if as_int(row, "torch_h2d_copy_events") > 0)
    cuda_ready = sum(1 for row in rows if yes(row, "cuda_copy_ready_before_replay"))
    hint_done = sum(1 for row in rows if yes(row, "full_hint_done_before_replay"))
    reloaded = sum(1 for row in rows if yes(row, "replay_reloaded_kv"))
    clean_success = sum(1 for row in rows if row.get("checkpoint_result") == "clean_success")
    return [
        {
            "profiled_sessions": len(rows),
            "evidence_status": "clean_success_seen"
            if clean_success
            else ("copy_visible_but_not_sufficient" if cuda_ready and reloaded else "mechanism_visible"),
            "sessions_with_visible_h2d_copy": visible_h2d,
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
                "profiled_h2d_events": attr.get("torch_h2d_copy_events", ""),
                "profiled_h2d_bytes": attr.get("torch_h2d_bytes", ""),
                "profiled_prefetch_margin_ms": attr.get("prefetch_margin_ms", ""),
            }
        )
    return rows_out


def paired_takeaway(clean: dict[str, Any], baseline: dict[str, Any], attr: dict[str, Any]) -> str:
    if not attr:
        return "no profiled row"
    cuda_ready = yes(attr, "cuda_copy_ready_before_replay")
    hint_done = yes(attr, "full_hint_done_before_replay")
    reloaded = yes(attr, "replay_reloaded_kv")
    clean_ttft = as_float(clean, "replay_ttft_ms") if clean else 0.0
    baseline_ttft = as_float(baseline, "replay_ttft_ms") if baseline else 0.0
    faster = baseline and clean and clean_ttft < baseline_ttft
    if cuda_ready and hint_done and not reloaded and faster:
        return "clean success"
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


def choose_timeline_sessions(rows: list[dict[str, Any]], max_sessions: int) -> list[str]:
    late = [row for row in rows if truthy(row.get("late_prefetch"))]
    visible = [row for row in rows if as_int(row, "torch_h2d_copy_events") > 0]
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
    if as_int(row, "sglang_copy_events") > 0 and margin is not None:
        return f"SGLang OK +{margin:.0f} ms", "#92400e"
    return "INCOMPLETE", "#6b7280"


def session_observation(row: dict[str, Any]) -> tuple[str, str, str]:
    margin = to_float(row.get("prefetch_margin_ms"))
    torch_copy_events = as_int(row, "torch_h2d_copy_events")
    sglang_events = as_int(row, "sglang_copy_events")
    missing_reason = str(row.get("h2d_missing_reason") or "")
    cuda_ready = yes_no(row.get("cuda_copy_ready_before_replay"))
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
        f"CUDA copy ready before replay {cuda_ready}; "
        f"full hint done before replay {hint_done}; "
        f"replay reloaded KV {replay_reloaded}; "
        f"SGLang load {fmt_ms(row.get('sglang_copy_start_ms'))} -> {fmt_ms(row.get('sglang_copy_end_ms'))}; "
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
            elif value in {"slower", "copy was early, replay still reloaded", "copy was early, full hint path was late"}:
                cls = ' class="warn"'
            out.append(f"<td{cls}>{html.escape(value)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def build_timeline_svg(
    rows: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    max_sessions: int,
) -> tuple[str, list[dict[str, Any]]]:
    selected = choose_timeline_sessions(rows, max_sessions)
    row_by_session = {str(row.get("session_id", "")): row for row in rows}
    selected_rows = [row_by_session[sid] for sid in selected if sid in row_by_session]
    selected_timeline = [item for item in timeline if item.get("session_id") in selected]
    if selected_timeline:
        start = min(float(item["start_ms"]) for item in selected_timeline)
        end = max(float(item["end_ms"]) for item in selected_timeline)
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
        "sglang_copy": "#f59e0b",
        "torch_copy": "#16a34a",
        "replay_due": "#111827",
        "replay": "#dc2626",
    }

    def x_pos(ms: float) -> float:
        return left + (ms - start) / span * plot_w

    def layer_order(item: dict[str, Any]) -> int:
        order = {
            "tool_wait": 0,
            "initial": 1,
            "hint_submitted": 2,
            "hint_request": 3,
            "replay": 4,
            "sglang_copy": 5,
            "torch_copy": 6,
            "replay_due": 7,
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
        svg.append(f'<line x1="{left}" y1="{y + 12}" x2="{left + plot_w}" y2="{y + 12}" stroke="#f3f4f6"/>')
        svg.append(f'<line x1="{left}" y1="{y + 48}" x2="{left + plot_w}" y2="{y + 48}" stroke="#f9fafb"/>')
        prefetch_done = to_float(row.get("torch_copy_end_ms")) or to_float(row.get("sglang_copy_end_ms"))
        replay_due = to_float(row.get("replay_due_ms"))
        margin = to_float(row.get("prefetch_margin_ms"))
        if prefetch_done is not None and replay_due is not None and margin is not None:
            x_done = x_pos(prefetch_done)
            x_due = x_pos(replay_due)
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
                svg.append(f'<rect x="{x1:.1f}" y="{y + 3}" width="{max(2, x2 - x1):.1f}" height="40" fill="#fee2e2" opacity="0.7"/>')
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
        x1 = x_pos(float(item.get("start_ms", 0.0)))
        x2 = x_pos(float(item.get("end_ms", 0.0)))
        label = html.escape(str(item.get("label", kind)))
        if x1 == x2:
            stroke_width = 6 if kind == "replay_due" else 3
            svg.append(
                f'<line x1="{x1:.1f}" y1="{y + 1}" x2="{x1:.1f}" y2="{y + 34}" stroke="{color}" stroke-width="{stroke_width}"><title>{label}</title></line>'
            )
            if kind == "replay_due":
                svg.append(
                    f'<text x="{x1:.1f}" y="{y + 43}" text-anchor="middle" font-size="11" fill="#111827" font-weight="700">due</text>'
                )
        else:
            display_x2 = x2
            bar_y = y + 4
            bar_h = 24
            opacity = "0.88"
            stroke = ""
            if kind == "torch_copy":
                display_x2 = max(x2, x1 + 24)
                bar_y = y
                bar_h = 32
                opacity = "1"
                stroke = ' stroke="#f8fafc" stroke-width="3"'
            svg.append(
                f'<rect x="{x1:.1f}" y="{bar_y}" width="{max(2, display_x2 - x1):.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="{opacity}"{stroke}><title>{label}</title></rect>'
            )
            if kind == "torch_copy":
                svg.append(
                    f'<text x="{(x1 + display_x2) / 2:.1f}" y="{y + 20}" text-anchor="middle" font-size="10" fill="white" font-weight="700">HtoD</text>'
                )
            elif kind == "sglang_copy" and x2 - x1 > 45:
                text_label = "KV load" if kind == "sglang_copy" else "HtoD"
                svg.append(
                    f'<text x="{(x1 + x2) / 2:.1f}" y="{y + 20}" text-anchor="middle" font-size="11" fill="white" font-weight="700">{text_label}</text>'
                )
    legend_x = left
    legend_y = height - 32
    for idx, (kind, color) in enumerate(colors.items()):
        lx = legend_x + idx * 130
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}">{html.escape(kind)}</text>')
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


def visible_h2d_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        events = as_int(row, "torch_h2d_copy_events")
        if events <= 0:
            continue
        out.append(
            {
                "session_id": row.get("session_id", ""),
                "h2d_start_ms": row.get("torch_copy_start_ms", ""),
                "h2d_end_ms": row.get("torch_copy_end_ms", ""),
                "h2d_events": events,
                "h2d_bytes": row.get("torch_h2d_bytes", ""),
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
            "sessions_with_visible_h2d_copy": sum(1 for row in rows if as_int(row, "torch_h2d_copy_events") > 0),
            "late_prefetch_sessions": sum(1 for row in rows if truthy(row.get("late_prefetch"))),
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
            "Layer": "green torch_copy",
            "Meaning": "Profiler-attributed CUDA host-to-device copy activity inside the hint request. On the chart, short green bars are visually widened so they are easy to see.",
            "Why it matters": "This is the closest signal we have for actual GPU-side KV movement. It should usually live inside the purple hint request. Use the HtoD table for exact start/end times.",
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
            "Checkpoint": "CUDA copy ready before replay",
            "Simple meaning": "The green HtoD copy ended before the black replay-due line.",
            "Why it matters": "This proves profiler-visible GPU copy work happened early enough, but only for the copied slice we attributed.",
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
        "cuda_copy_ready_before_replay",
        "full_hint_done_before_replay",
        "replay_reloaded_kv",
        "resume_load_count",
        "hint_outcome",
        "checkpoint_result",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


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
        "sglang_copy_start_ms",
        "sglang_copy_end_ms",
        "torch_copy_start_ms",
        "torch_copy_end_ms",
        "torch_h2d_copy_events",
        "torch_h2d_bytes",
        "sglang_kv_profiler_status",
        "h2d_missing_reason",
        "profiler_start_ms",
        "profiler_end_ms",
        "profiler_stop_reason",
        "replay_due_ms",
        "prefetch_margin_ms",
        "late_prefetch",
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


def write_html(
    path: Path,
    sections: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    attribution_rows: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    max_timeline_sessions: int,
) -> None:
    cards = summary_cards(sections)
    timeline_svg, selected_timeline_rows = build_timeline_svg(attribution_rows, timeline, max_timeline_sessions)
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
        '<div class="panel"><h2>Manager Summary</h2>',
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
        '<div class="panel"><h2>How To Read This Report</h2>',
        '<p class="caption"><span class="pill">Clean performance</span> comes from profiler-off runs. Use these rows for TTFT and latency claims.</p>',
        '<p class="caption"><span class="pill">Profiled attribution</span> comes from profiler-on runs. Use these rows to show CUDA HtoD copies, hint completion, and replay reload behavior.</p>',
        '<p class="caption"><span class="pill">Paired evidence</span> joins the two views by session id, so we can say what improved and what mechanism was observed.</p>',
        "</div>",
        '<div class="panel"><h2>Key Deductions</h2><ul>',
        *[f"<li>{html.escape(line)}</li>" for line in key_deduction_lines(sections)],
        "</ul></div>",
        '<div class="panel"><h2>Metadata</h2><pre>',
        html.escape(json.dumps(metadata, indent=2, sort_keys=True)),
        "</pre></div>",
    ]
    for title in ("Clean Performance Summary", "Profiled Attribution Summary"):
        rows = sections.get(title, [])
        lines.append(f'<div class="panel"><h2>{html.escape(title)}</h2>')
        lines.append(f'<p class="caption">{html.escape(section_caption(title))}</p>')
        lines.append('<div class="table-wrap">')
        lines.append(html_table(rows))
        lines.append("</div></div>")
    if attribution_rows and timeline:
        lines.append('<div class="panel"><h2>Timeline Summary</h2>')
        lines.append('<p class="caption">This is the profiled mechanism view. It shows what happened inside the hinted path, not clean TTFT performance.</p>')
        lines.append('<div class="table-wrap">')
        lines.append(html_table(timeline_summary_rows(attribution_rows)))
        lines.append("</div></div>")
        lines.append('<div class="panel"><h2>Timeline</h2>')
        lines.append(
            '<p class="caption">How to read this: gray is the tool-wait window. Purple is the software hint request that runs during the tool wait. Green is the profiler-attributed CUDA HtoD copy observed inside that hint request. Green HtoD bars are drawn on top with a minimum visual width so short copy windows are not lost at full timeline scale. The black line is replay due. A green dashed gap means KV movement finished before replay. A red dashed gap means replay was already due before KV movement finished.</p>'
        )
        lines.append(timeline_svg)
        lines.append("</div>")
        h2d_rows = visible_h2d_rows(selected_timeline_rows)
        if h2d_rows:
            lines.append('<div class="panel"><h2>Visible CUDA HtoD Copies</h2>')
            lines.append(
                '<p class="caption">These are the selected sessions where the profiler attributed host-to-device CUDA copy activity. The green bars in the timeline correspond to these rows.</p>'
            )
            lines.append('<div class="table-wrap">')
            lines.append(html_table(h2d_rows))
            lines.append("</div></div>")
        timeline_sections = [
            ("Timeline Layers", timeline_layers_rows(), "These rows explain what each visual layer in the timeline means."),
            ("Prefetch Checkpoints", prefetch_checkpoint_rows(), "These checkpoints separate copy readiness, full hint completion, and replay reuse."),
            (
                "Checkpoint Results Per Session",
                checkpoint_result_rows(selected_timeline_rows),
                "This table shows whether each selected session passed or failed each checkpoint.",
            ),
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
            lines.append(f'<div class="panel"><h2>{html.escape(title)}</h2>')
            lines.append(f'<p class="caption">{html.escape(caption)}</p>')
            lines.append('<div class="table-wrap wide">')
            lines.append(html_table(rows))
            lines.append("</div></div>")
    else:
        lines.append(
            '<div class="panel"><h2>Timeline</h2><p class="caption">No profiled timeline JSON was found for this report. Run the profiled attribution step to populate the visual timeline sections.</p></div>'
        )
    lines.append('<div class="panel"><h2>Paired Session Evidence</h2>')
    lines.append(f'<p class="caption">{html.escape(section_caption("Paired Session Evidence"))}</p>')
    lines.append('<div class="table-wrap">')
    lines.append(html_table(sections.get("Paired Session Evidence", [])))
    lines.append("</div></div>")
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
            "label": "CUDA Ready",
            "value": f'{attr_row.get("cuda_copy_ready_before_replay", 0)} / {profiled_sessions}',
            "detail": "Profiled sessions where CUDA HtoD copy finished before replay.",
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
        "The profiled run is the mechanism source of truth because it exposes CUDA HtoD and SGLang KV movement evidence.",
    ]
    if best:
        lines.append(
            f'Best clean mode so far: {best.get("mode")} with {best.get("avg_improvement_vs_no_prefetch_ms")} ms average replay TTFT delta versus no_prefetch.'
        )
    if attr_row:
        lines.append(
            f'Profiled attribution: CUDA copy ready before replay in {attr_row.get("cuda_copy_ready_before_replay")} / {attr_row.get("profiled_sessions")} sessions; replay reloaded KV in {attr_row.get("replay_reloaded_kv")} / {attr_row.get("profiled_sessions")} sessions.'
        )
    return lines


def key_deduction_lines(sections: dict[str, list[dict[str, Any]]]) -> list[str]:
    attr = sections.get("Profiled Attribution Summary", [])
    paired = sections.get("Paired Session Evidence", [])
    attr_row = attr[0] if attr else {}
    cuda_ready = as_int(attr_row, "cuda_copy_ready_before_replay") if attr_row else 0
    reloaded = as_int(attr_row, "replay_reloaded_kv") if attr_row else 0
    hint_done = as_int(attr_row, "full_hint_done_before_replay") if attr_row else 0
    sessions = as_int(attr_row, "profiled_sessions") if attr_row else 0
    lines = [
        "We should judge TTFT using the clean run, then use the profiled run to explain why the result happened.",
    ]
    if cuda_ready and reloaded:
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
        "Profiled Attribution Summary": "Profiler is on here. Use this table to understand CUDA HtoD copy visibility, hint completion, and replay reloads.",
        "Paired Session Evidence": "This joins the clean and profiled views by session id so each session has both a performance view and a mechanism view.",
    }
    return captions.get(title, "")


def copy_latest_reports(out_root: Path, latest_root: Path) -> None:
    latest_root.mkdir(parents=True, exist_ok=True)
    copies = {
        "paired_report.html": "latest_paired_report.html",
        "paired_report.md": "latest_paired_report.md",
        "paired_report.json": "latest_paired_report.json",
        "paired_clean_summary.csv": "latest_paired_clean_summary.csv",
        "paired_attribution_summary.csv": "latest_paired_attribution_summary.csv",
        "paired_session_evidence.csv": "latest_paired_session_evidence.csv",
        "paired_timeline_summary.csv": "latest_paired_timeline_summary.csv",
        "paired_checkpoint_results.csv": "latest_paired_checkpoint_results.csv",
        "paired_key_observations.csv": "latest_paired_key_observations.csv",
        "paired_session_details.csv": "latest_paired_session_details.csv",
    }
    for src_name, dst_name in copies.items():
        src = out_root / src_name
        if src.exists():
            shutil.copyfile(src, latest_root / dst_name)


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
    modes = [mode for mode in args.modes.split() if mode]

    clean_rows = load_clean_rows(clean_root, modes)
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
    write_csv(out_root / "paired_checkpoint_results.csv", checkpoint_result_rows(selected_timeline_rows))
    write_csv(out_root / "paired_key_observations.csv", key_observation_rows(selected_timeline_rows))
    write_csv(out_root / "paired_session_details.csv", session_detail_rows(selected_timeline_rows))
    (out_root / "paired_report.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "sections": sections,
                "timeline": {
                    "summary": timeline_summary_rows(attribution_rows),
                    "layers": timeline_layers_rows(),
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
        args.max_timeline_sessions,
    )
    if args.latest_root:
        copy_latest_reports(out_root, Path(args.latest_root))

    print(f"Wrote paired report under {out_root}")
    if args.latest_root:
        print(f"Wrote latest paired report copies under {args.latest_root}")
    for row in sections["Clean Performance Summary"]:
        print(
            f"clean {row['mode']}: avg_replay_ttft_ms={row['avg_replay_ttft_ms']}, "
            f"outcomes={row['outcomes']}"
        )
    for row in sections["Profiled Attribution Summary"]:
        print(
            "profiled attribution: "
            f"cuda_ready={row['cuda_copy_ready_before_replay']}/{row['profiled_sessions']}, "
            f"hint_done={row['full_hint_done_before_replay']}/{row['profiled_sessions']}, "
            f"reloaded={row['replay_reloaded_kv']}/{row['profiled_sessions']}, "
            f"clean_success={row['clean_success']}/{row['profiled_sessions']}"
        )


if __name__ == "__main__":
    main()
