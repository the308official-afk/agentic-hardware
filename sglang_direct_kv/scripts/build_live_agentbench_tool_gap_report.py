#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
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


def maybe_int(value: Any) -> int | None:
    parsed = maybe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def has_positive(value: Any) -> bool:
    parsed = maybe_float(value)
    return parsed is not None and parsed > 0


def timeline_kv_outcome(row: dict[str, Any]) -> tuple[str, str, str]:
    hint_h2d = has_positive(row.get("direct_kv_h2d_events")) or has_positive(row.get("hint_host_load_tokens"))
    replay_h2d = has_positive(row.get("replay_kv_h2d_events")) or has_positive(row.get("replay_host_load_tokens"))
    recomputed_tokens = maybe_int(row.get("recomputed_tokens_est"))
    if recomputed_tokens is None:
        recomputed_tokens = maybe_int(row.get("replay_new_prefill_tokens_est"))
    recompute = recomputed_tokens is not None and recomputed_tokens >= 128
    margin = maybe_float(row.get("prefetch_margin_ms"))
    hit_ratio = maybe_float(row.get("replay_cache_hit_ratio_pct"))
    mode = str(row.get("mode") or "")

    if hint_h2d and (replay_h2d or recompute):
        return "WASTED PREFETCH", "#92400e", "Prefetch moved or touched KV, but replay still loaded or rebuilt KV."
    if hint_h2d and margin is not None and margin >= 0:
        return "PREFETCH HIT", "#166534", "Hint-side KV movement finished before replay and replay did not need visible recovery."
    if replay_h2d and recompute:
        return "MIXED LOAD+RECOMPUTE", "#7c3aed", "Replay loaded some KV from host and also rebuilt missing prefix work."
    if replay_h2d:
        return "REPLAY HOST LOAD", "#0e7490", "Replay itself loaded KV from host to GPU."
    if recompute:
        return "RECOMPUTE", "#be185d", "Replay rebuilt/prefilled missing prefix tokens instead of cleanly loading old KV."
    if mode == "no_prefetch":
        if hit_ratio is not None and hit_ratio >= 90:
            return "FULL REUSE", "#475569", "Replay mostly reused already available KV."
        return "NO PREFETCH", "#64748b", "No hint path ran for this row."
    if margin is not None and margin < 0:
        return "LATE PREFETCH", "#b91c1c", "The prefetch attempt finished after replay was due."
    if hit_ratio is not None and hit_ratio >= 90:
        return "FULL REUSE", "#475569", "Replay mostly reused already available KV."
    return "NO VISIBLE KV MOVE", "#64748b", "No host-to-device KV movement was visible for this row."


def replay_recompute_segment_ms(row: dict[str, Any]) -> float:
    recomputed_tokens = maybe_int(row.get("recomputed_tokens_est"))
    if recomputed_tokens is None:
        recomputed_tokens = maybe_int(row.get("replay_new_prefill_tokens_est"))
    if recomputed_tokens is None or recomputed_tokens < 128:
        return 0.0
    prefill_compute = maybe_float(row.get("prefill_compute_ms_est"))
    model_forward = maybe_float(row.get("model_forward_ms"))
    ttft = maybe_float(row.get("resume_ttft_ms"))
    replay_h2d = maybe_float(row.get("replay_kv_h2d_duration_ms")) or 0.0
    if prefill_compute is not None and prefill_compute > 0:
        return prefill_compute
    if model_forward is not None and model_forward > 0:
        return model_forward
    if ttft is not None:
        return max(0.0, ttft - replay_h2d)
    return 0.0


def duration_between(row: dict[str, Any], start_key: str, end_key: str) -> float | None:
    start = maybe_float(row.get(start_key))
    end = maybe_float(row.get(end_key))
    if start is None or end is None:
        return None
    return max(0.0, end - start)


def display_ms(value: Any) -> str:
    parsed = maybe_float(value)
    if parsed is None:
        return ""
    if abs(parsed) >= 1000:
        return f"{parsed / 1000.0:.1f} s"
    return f"{parsed:.0f} ms"


def timeline_case_fillers(row: dict[str, Any]) -> str:
    explicit = row.get("fillers") or row.get("filler_count")
    if explicit not in ("", None):
        return str(explicit)
    case_dir = str(row.get("case_dir") or "")
    case_name = Path(case_dir).name
    if "_f" not in case_name:
        return ""
    return case_name.rsplit("_f", 1)[-1]


def append_timeline_legend(
    svg: list[str],
    legend: list[tuple[str, str]],
    x: float,
    y: float,
    step: float,
    line_labels: set[str] | None = None,
) -> None:
    line_labels = line_labels or set()
    lx = x
    for label, color in legend:
        if label in line_labels:
            svg.append(
                f'<line x1="{lx:.1f}" y1="{y - 12:.1f}" x2="{lx:.1f}" y2="{y + 4:.1f}" '
                f'stroke="{color}" stroke-width="4"/>'
            )
        else:
            svg.append(f'<rect x="{lx:.1f}" y="{y - 12:.1f}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20:.1f}" y="{y:.1f}" font-size="12">{fmt(label)}</text>')
        lx += step


def replay_phase_segments(row: dict[str, Any]) -> dict[str, float]:
    ttft = maybe_float(row.get("resume_ttft_ms")) or duration_between(row, "replay_prefill_start_ms", "replay_prefill_end_ms") or 0.0
    replay_h2d = maybe_float(row.get("replay_kv_h2d_duration_ms")) or maybe_float(row.get("host_load_ms")) or 0.0
    recompute = replay_recompute_segment_ms(row)
    known = min(ttft, replay_h2d + recompute)
    normal_prefill = max(0.0, ttft - known)
    latency = maybe_float(row.get("resume_latency_ms")) or duration_between(row, "resume_start_ms", "resume_end_ms") or 0.0
    decode = max(0.0, latency - ttft)
    return {
        "ttft": ttft,
        "replay_h2d": replay_h2d,
        "recompute": recompute,
        "normal_prefill": normal_prefill,
        "decode": decode,
    }


def build_readable_phase_timeline_svg(
    gaps: list[dict[str, Any]],
    max_rows: int,
    show_prefetch_legend: bool = True,
) -> str:
    rows = gaps[:max_rows]
    if not rows:
        return "<p>No readable phase timeline available.</p>"

    width = 1580
    left = 205
    right = 40
    top = 112
    row_h = 172
    header_h = 40
    gap = 16
    turn_w = 220
    wait_w = 190
    prefetch_w = 330 if show_prefetch_legend else 170
    replay_w = width - left - right - turn_w - wait_w - prefetch_w - (3 * gap)
    x_turn = left
    x_wait = x_turn + turn_w + gap
    x_prefetch = x_wait + wait_w + gap
    x_replay = x_prefetch + prefetch_w + gap
    plot_bottom = top + header_h + len(rows) * row_h
    legend_y = plot_bottom + 42
    height = legend_y + 56

    def rect(
        svg: list[str],
        x: float,
        y: float,
        w: float,
        h: float,
        color: str,
        label: str,
        title: str,
        opacity: float = 0.9,
        text_color: str = "#ffffff",
        dashed: bool = False,
    ) -> None:
        stroke = ' stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5 4"' if dashed else ""
        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
            f'fill="{color}" opacity="{opacity}"{stroke}><title>{fmt(title)}</title></rect>'
        )
        if label:
            svg.append(
                f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="10" fill="{text_color}" font-weight="700">{fmt(label)}</text>'
            )

    def missing(svg: list[str], x: float, y: float, w: float, h: float, label: str) -> None:
        rect(svg, x, y, w, h, "#f8fafc", label, label, 1.0, "#64748b", dashed=True)

    def active_bar(svg: list[str], x: float, y: float, w: float, h: float, color: str, name: str, duration: Any, title: str, text_color: str = "#ffffff") -> None:
        duration_label = display_ms(duration)
        label = f"{name}: {duration_label}" if duration_label else name
        rect(svg, x, y, w, h, color, label, title, 0.9, text_color)

    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Readable phase timeline without global time scale">',
        '<text x="10" y="26" font-size="18" font-weight="700" fill="#0f172a">Readable phase timeline</text>',
        '<text x="10" y="48" font-size="12" fill="#475569">Each bar is stretched for readability. The printed duration is the measured value.</text>',
    ]
    legend = [
        ("initial model turn", "#2563eb"),
        ("tool wait", "#d1d5db"),
        ("prefetch attempt", "#a855f7"),
        ("hint-side KV HtoD", "#16a34a"),
        ("replay-side KV HtoD", "#06b6d4"),
        ("recompute/rebuild", "#db2777"),
        ("remaining TTFT", "#eab308"),
        ("decode", "#ef4444"),
    ]
    append_timeline_legend(svg, legend, left, 76, 160)
    headers = [
        ("initial turn", x_turn, turn_w),
        ("tool wait", x_wait, wait_w),
        ("prefetch", x_prefetch, prefetch_w),
        ("replay path", x_replay, replay_w),
    ]
    for label, x, w in headers:
        svg.append(f'<text x="{x + w / 2:.1f}" y="{top - 8}" text-anchor="middle" font-size="13" fill="#334155" font-weight="700">{fmt(label)}</text>')
        svg.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" height="{plot_bottom - top:.1f}" fill="none" stroke="#e5e7eb"/>')

    for idx, row in enumerate(rows):
        y = top + header_h + idx * row_h
        band_fill = "#ffffff" if idx % 2 == 0 else "#e8eef6"
        svg.append(f'<rect x="0" y="{y:.1f}" width="{width}" height="{row_h:.1f}" fill="{band_fill}" opacity="0.96"/>')
        svg.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#cbd5e1"/>')

        label = str(row.get("timeline_label") or f"G{idx:02d}")
        margin = maybe_float(row.get("prefetch_margin_ms"))
        if margin is None:
            timing = "NO PREFETCH"
            timing_color = "#64748b"
        elif margin >= 0:
            timing = f"READY +{display_ms(margin)}"
            timing_color = "#166534"
        else:
            timing = f"LATE -{display_ms(abs(margin))}"
            timing_color = "#b91c1c"
        outcome, outcome_color, outcome_title = timeline_kv_outcome(row)
        svg.append(f'<text x="12" y="{y + 27}" font-size="16" fill="#0f172a" font-weight="800">{fmt(label)}</text>')
        svg.append(f'<text x="12" y="{y + 49}" font-size="12" fill="{timing_color}" font-weight="800">{fmt(timing)}</text>')
        svg.append(f'<text x="12" y="{y + 70}" font-size="12" fill="{outcome_color}" font-weight="800"><title>{fmt(outcome_title)}</title>{fmt(outcome)}</text>')
        svg.append(f'<text x="12" y="{y + 91}" font-size="10" fill="#64748b">wait {fmt(display_ms(row.get("tool_gap_ms")))}</text>')

        lane_y = y + 24
        small_h = 22
        current_duration = maybe_float(row.get("current_latency_ms")) or duration_between(row, "current_start_ms", "current_end_ms")
        active_bar(svg, x_turn + 12, lane_y, turn_w - 24, 28, "#2563eb", "turn", current_duration, "initial model turn")
        active_bar(svg, x_wait + 12, lane_y, wait_w - 24, 28, "#d1d5db", "wait", row.get("tool_gap_ms"), "tool/tool-wait window", "#334155")

        if show_prefetch_legend:
            prefetch_duration = maybe_float(row.get("prefetch_duration_ms"))
            direct_h2d_duration = maybe_float(row.get("direct_kv_h2d_duration_ms"))
            if prefetch_duration is not None:
                active_bar(svg, x_prefetch + 12, y + 16, prefetch_w - 24, small_h, "#a855f7", "attempt", prefetch_duration, "software/direct prefetch attempt")
            else:
                missing(svg, x_prefetch + 12, y + 16, prefetch_w - 24, small_h, "no prefetch attempt")
            if direct_h2d_duration is not None and direct_h2d_duration > 0:
                active_bar(svg, x_prefetch + 12, y + 46, prefetch_w - 24, small_h, "#16a34a", "hint HtoD", direct_h2d_duration, "hint-side direct KV host-to-device movement")
            else:
                missing(svg, x_prefetch + 12, y + 46, prefetch_w - 24, small_h, "no hint HtoD observed")

        segments = replay_phase_segments(row)
        replay_h2d_duration = maybe_float(row.get("replay_kv_h2d_duration_ms")) or maybe_float(row.get("host_load_ms"))
        replay_rows = [
            ("replay HtoD", replay_h2d_duration, "#06b6d4", "replay-side KV host-to-device load", "#ffffff"),
            ("recompute", segments["recompute"] if segments["recompute"] > 0 else None, "#db2777", "replay recompute / rebuilt missing prefix", "#ffffff"),
            ("other TTFT", segments["normal_prefill"] if segments["normal_prefill"] > 0 else None, "#eab308", "remaining before-first-token work", "#713f12"),
            ("decode", segments["decode"] if segments["decode"] > 0 else None, "#ef4444", "decode/generation after first token", "#ffffff"),
        ]
        for lane_idx, (name, duration, color, title, text_color) in enumerate(replay_rows):
            ry = y + 8 + lane_idx * 35
            if duration is not None and duration > 0:
                active_bar(svg, x_replay + 12, ry, replay_w - 24, small_h, color, name, duration, title, text_color)
            else:
                missing(svg, x_replay + 12, ry, replay_w - 24, small_h, f"no {name}")

    svg.append(f'<line x1="0" y1="{plot_bottom:.1f}" x2="{width}" y2="{plot_bottom:.1f}" stroke="#cbd5e1"/>')
    append_timeline_legend(svg, legend, left, legend_y, 160)
    svg.append("</svg>")
    return "\n".join(svg)


def build_local_timing_phase_timeline_svg(
    gaps: list[dict[str, Any]],
    max_rows: int,
    show_prefetch_legend: bool = True,
) -> str:
    rows = gaps[:max_rows]
    if not rows:
        return "<p>No local-timing readable phase timeline available.</p>"

    width = 1580
    left = 215
    right = 40
    top = 116
    row_h = 214
    header_h = 42
    gap = 16
    turn_w = 220
    wait_w = 190
    prefetch_w = 340 if show_prefetch_legend else 170
    replay_w = width - left - right - turn_w - wait_w - prefetch_w - (3 * gap)
    x_turn = left
    x_wait = x_turn + turn_w + gap
    x_prefetch = x_wait + wait_w + gap
    x_replay = x_prefetch + prefetch_w + gap
    plot_bottom = top + header_h + len(rows) * row_h
    legend_y = plot_bottom + 42
    height = legend_y + 56

    legend = [
        ("initial model turn", "#2563eb"),
        ("tool wait", "#d1d5db"),
        ("prefetch attempt", "#a855f7"),
        ("hint-side KV HtoD", "#16a34a"),
        ("replay-side KV HtoD", "#06b6d4"),
        ("recompute/rebuild", "#db2777"),
        ("remaining TTFT", "#eab308"),
        ("decode", "#ef4444"),
    ]

    def rect(
        svg: list[str],
        x: float,
        y: float,
        w: float,
        h: float,
        color: str,
        label: str,
        title: str,
        opacity: float = 0.9,
        text_color: str = "#ffffff",
        dashed: bool = False,
        min_w: float = 12.0,
        max_x: float | None = None,
    ) -> None:
        stroke = ' stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5 4"' if dashed else ""
        display_w = max(min_w, w)
        if max_x is not None and x + display_w > max_x:
            x = max(x, max_x - display_w)
        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{display_w:.1f}" height="{h:.1f}" rx="4" '
            f'fill="{color}" opacity="{opacity}"{stroke}><title>{fmt(title)}</title></rect>'
        )
        if label and display_w >= 64:
            svg.append(
                f'<text x="{x + display_w / 2:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="10" fill="{text_color}" font-weight="700">{fmt(label)}</text>'
            )

    def missing(svg: list[str], x: float, y: float, w: float, h: float, label: str) -> None:
        svg.append(
            f'<rect x="{x:.1f}" y="{y + h / 2 - 2:.1f}" width="{w:.1f}" height="4" rx="2" '
            f'fill="#cbd5e1" opacity="0.45"><title>{fmt(label)}</title></rect>'
        )
        if w >= 110:
            svg.append(
                f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="9" fill="#64748b">{fmt(label)}</text>'
            )

    def local_x(value: float, start: float, end: float, x: float, w: float) -> float:
        span = max(1e-6, end - start)
        clamped = min(max(value, start), end)
        return x + (clamped - start) / span * w

    def local_bar(
        svg: list[str],
        start_value: float | None,
        end_value: float | None,
        local_start: float,
        local_end: float,
        x: float,
        y: float,
        w: float,
        h: float,
        color: str,
        name: str,
        duration: Any,
        title: str,
        text_color: str = "#ffffff",
        opacity: float = 0.9,
        dashed: bool = False,
    ) -> None:
        if start_value is None or end_value is None:
            missing(svg, x, y, w, h, f"no {name}")
            return
        x1 = local_x(start_value, local_start, local_end, x, w)
        x2 = local_x(end_value, local_start, local_end, x, w)
        duration_label = display_ms(duration)
        label = duration_label if duration_label else name
        rect(svg, x1, y, max(0.0, x2 - x1), h, color, label, title, opacity, text_color, dashed, 12.0, x + w)

    def bounds(*pairs: tuple[float | None, float | None]) -> tuple[float, float] | None:
        values: list[float] = []
        for start_value, end_value in pairs:
            if start_value is not None:
                values.append(start_value)
            if end_value is not None:
                values.append(end_value)
        if not values:
            return None
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Readable phase timeline with local timing inside each column">',
        '<text x="10" y="26" font-size="18" font-weight="700" fill="#0f172a">Readable phase timeline with local timing</text>',
        '<text x="10" y="48" font-size="12" fill="#475569">Each big column has its own local timing. Bars keep left-to-right order inside that phase, but columns do not share one global time scale.</text>',
    ]
    append_timeline_legend(svg, legend, left, 78, 160)
    prefetch_label_w = 58
    replay_label_w = 78
    prefetch_bar_x = x_prefetch + prefetch_label_w + 14
    prefetch_bar_w = prefetch_w - prefetch_label_w - 26
    replay_bar_x = x_replay + replay_label_w + 14
    replay_bar_w = replay_w - replay_label_w - 26

    headers = [
        ("initial turn", x_turn, turn_w),
        ("tool wait", x_wait, wait_w),
        ("prefetch", x_prefetch, prefetch_w),
        ("replay path", x_replay, replay_w),
    ]
    for label, x, w in headers:
        svg.append(f'<text x="{x + w / 2:.1f}" y="{top - 8}" text-anchor="middle" font-size="13" fill="#334155" font-weight="700">{fmt(label)}</text>')
        svg.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" height="{plot_bottom - top:.1f}" fill="none" stroke="#e5e7eb"/>')

    for idx, row in enumerate(rows):
        y = top + header_h + idx * row_h
        band_fill = "#ffffff" if idx % 2 == 0 else "#e8eef6"
        svg.append(f'<rect x="0" y="{y:.1f}" width="{width}" height="{row_h:.1f}" fill="{band_fill}" opacity="0.96"/>')
        svg.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#cbd5e1"/>')

        label = str(row.get("timeline_label") or f"G{idx:02d}")
        margin = maybe_float(row.get("prefetch_margin_ms"))
        if margin is None:
            timing = "NO PREFETCH"
            timing_color = "#64748b"
        elif margin >= 0:
            timing = f"READY +{display_ms(margin)}"
            timing_color = "#166534"
        else:
            timing = f"LATE -{display_ms(abs(margin))}"
            timing_color = "#b91c1c"
        outcome, outcome_color, outcome_title = timeline_kv_outcome(row)
        fillers = timeline_case_fillers(row)
        task_index = row.get("task_index", "")
        gap_index = row.get("gap_order_in_task", "")
        mode = str(row.get("mode") or "")
        svg.append(f'<text x="12" y="{y + 27}" font-size="16" fill="#0f172a" font-weight="800">{fmt(label)}</text>')
        svg.append(f'<text x="12" y="{y + 49}" font-size="12" fill="{timing_color}" font-weight="800">{fmt(timing)}</text>')
        svg.append(f'<text x="12" y="{y + 70}" font-size="12" fill="{outcome_color}" font-weight="800"><title>{fmt(outcome_title)}</title>{fmt(outcome)}</text>')
        svg.append(f'<text x="12" y="{y + 91}" font-size="10" fill="#64748b">wait {fmt(display_ms(row.get("tool_gap_ms")))}</text>')
        if fillers:
            svg.append(f'<text x="12" y="{y + 109}" font-size="10" fill="#64748b">fillers {fmt(fillers)}</text>')
        if task_index not in ("", None) or gap_index not in ("", None):
            svg.append(f'<text x="12" y="{y + 127}" font-size="10" fill="#64748b">task {fmt(task_index)} / gap {fmt(gap_index)}</text>')
        if mode:
            svg.append(f'<text x="12" y="{y + 145}" font-size="10" fill="#64748b">mode {fmt(mode)}</text>')

        lane_h = 22
        lane_step = 38
        for lane_idx, lane in enumerate(["attempt", "HtoD"] if show_prefetch_legend else [""]):
            if lane:
                svg.append(
                    f'<text x="{x_prefetch + prefetch_label_w:.1f}" y="{y + 29 + lane_idx * lane_step:.1f}" '
                    f'font-size="9" fill="#475569" text-anchor="end" font-weight="700">{lane}</text>'
                )
        for lane_idx, lane in enumerate(["HtoD", "recompute", "prefill", "decode"]):
            svg.append(
                f'<text x="{x_replay + replay_label_w:.1f}" y="{y + 29 + lane_idx * lane_step:.1f}" '
                f'font-size="9" fill="#475569" text-anchor="end" font-weight="700">{lane}</text>'
            )

        bar_h = lane_h
        current_start = maybe_float(row.get("current_start_ms"))
        current_end = maybe_float(row.get("current_end_ms"))
        current_bounds = bounds((current_start, current_end)) or (0.0, 1.0)
        current_duration = maybe_float(row.get("current_latency_ms")) or duration_between(row, "current_start_ms", "current_end_ms")
        local_bar(
            svg,
            current_start,
            current_end,
            current_bounds[0],
            current_bounds[1],
            x_turn + 12,
            y + 20,
            turn_w - 24,
            bar_h,
            "#2563eb",
            "turn",
            current_duration,
            "initial model turn",
        )

        wait_start = maybe_float(row.get("tool_gap_start_ms"))
        wait_end = maybe_float(row.get("tool_gap_end_ms"))
        wait_bounds = bounds((wait_start, wait_end)) or (0.0, 1.0)
        local_bar(
            svg,
            wait_start,
            wait_end,
            wait_bounds[0],
            wait_bounds[1],
            x_wait + 12,
            y + 20,
            wait_w - 24,
            bar_h,
            "#d1d5db",
            "wait",
            row.get("tool_gap_ms"),
            "tool/tool-wait window",
            "#334155",
        )

        if show_prefetch_legend:
            prefetch_start = maybe_float(row.get("prefetch_start_ms"))
            prefetch_end = maybe_float(row.get("prefetch_end_ms"))
            h2d_start = maybe_float(row.get("direct_kv_h2d_start_ms"))
            h2d_end = maybe_float(row.get("direct_kv_h2d_end_ms"))
            prefetch_bounds = bounds((prefetch_start, prefetch_end), (h2d_start, h2d_end)) or (0.0, 1.0)
            local_bar(
                svg,
                prefetch_start,
                prefetch_end,
                prefetch_bounds[0],
                prefetch_bounds[1],
                prefetch_bar_x,
                y + 14,
                prefetch_bar_w,
                bar_h,
                "#a855f7",
                "attempt",
                row.get("prefetch_duration_ms"),
                "software/direct prefetch attempt; locally positioned inside the prefetch column",
                "#ffffff",
                0.82,
            )
            if h2d_start is not None and h2d_end is not None:
                local_bar(
                    svg,
                    h2d_start,
                    h2d_end,
                    prefetch_bounds[0],
                    prefetch_bounds[1],
                    prefetch_bar_x,
                    y + 14 + lane_step,
                    prefetch_bar_w,
                    bar_h,
                    "#16a34a",
                    "hint HtoD",
                    row.get("direct_kv_h2d_duration_ms"),
                    "hint-side direct KV host-to-device movement; locally positioned relative to the prefetch attempt",
                    "#ffffff",
                    0.94,
                )
            else:
                missing(svg, prefetch_bar_x, y + 14 + lane_step, prefetch_bar_w, bar_h, "no hint HtoD")

        replay_start = maybe_float(row.get("resume_start_ms"))
        replay_end = maybe_float(row.get("resume_end_ms"))
        replay_prefill_start = maybe_float(row.get("replay_prefill_start_ms"))
        replay_prefill_end = maybe_float(row.get("replay_prefill_end_ms"))
        replay_h2d_start = maybe_float(row.get("replay_kv_h2d_start_ms"))
        replay_h2d_end = maybe_float(row.get("replay_kv_h2d_end_ms"))
        replay_bounds = bounds(
            (replay_start, replay_end),
            (replay_prefill_start, replay_prefill_end),
            (replay_h2d_start, replay_h2d_end),
        ) or (0.0, 1.0)
        replay_h2d_duration = maybe_float(row.get("replay_kv_h2d_duration_ms")) or maybe_float(row.get("host_load_ms"))
        if replay_h2d_start is not None and replay_h2d_end is not None:
            local_bar(
                svg,
                replay_h2d_start,
                replay_h2d_end,
                replay_bounds[0],
                replay_bounds[1],
                replay_bar_x,
                y + 14,
                replay_bar_w,
                bar_h,
                "#06b6d4",
                "replay HtoD",
                replay_h2d_duration,
                "replay-side KV host-to-device load; locally positioned inside replay request time",
            )
        else:
            missing(svg, replay_bar_x, y + 14, replay_bar_w, bar_h, "no replay HtoD")

        segments = replay_phase_segments(row)
        ttft = segments["ttft"]
        recompute = segments["recompute"]
        normal_prefill = segments["normal_prefill"]
        replay_prefill_start = replay_prefill_start if replay_prefill_start is not None else replay_start
        first_token = replay_prefill_end
        if first_token is None and replay_start is not None:
            first_token = replay_start + ttft
        cursor = replay_prefill_start
        if cursor is not None and recompute > 0:
            recompute_end = min(first_token if first_token is not None else cursor + recompute, cursor + recompute)
            local_bar(
                svg,
                cursor,
                recompute_end,
                replay_bounds[0],
                replay_bounds[1],
                replay_bar_x,
                y + 14 + lane_step,
                replay_bar_w,
                bar_h,
                "#db2777",
                "recompute",
                recompute,
                "estimated replay recompute / rebuilt missing prefix",
            )
            cursor = recompute_end
        else:
            missing(svg, replay_bar_x, y + 14 + lane_step, replay_bar_w, bar_h, "no recompute")

        if cursor is None:
            cursor = replay_prefill_start
        if cursor is not None and normal_prefill > 0:
            prefill_end = min(first_token if first_token is not None else cursor + normal_prefill, cursor + normal_prefill)
            local_bar(
                svg,
                cursor,
                prefill_end,
                replay_bounds[0],
                replay_bounds[1],
                replay_bar_x,
                y + 14 + 2 * lane_step,
                replay_bar_w,
                bar_h,
                "#eab308",
                "prefill",
                normal_prefill,
                "estimated remaining replay prefill or before-first-token work",
                "#713f12",
                0.9,
            )
        else:
            missing(svg, replay_bar_x, y + 14 + 2 * lane_step, replay_bar_w, bar_h, "no extra prefill")

        decode_start = first_token
        if decode_start is not None and replay_end is not None and replay_end > decode_start:
            local_bar(
                svg,
                decode_start,
                replay_end,
                replay_bounds[0],
                replay_bounds[1],
                replay_bar_x,
                y + 14 + 3 * lane_step,
                replay_bar_w,
                bar_h,
                "#ef4444",
                "decode",
                max(0.0, replay_end - decode_start),
                "decode/generation after first token",
                "#ffffff",
                0.72,
            )
        else:
            missing(svg, replay_bar_x, y + 14 + 3 * lane_step, replay_bar_w, bar_h, "no decode")

    svg.append(f'<line x1="0" y1="{plot_bottom:.1f}" x2="{width}" y2="{plot_bottom:.1f}" stroke="#cbd5e1"/>')
    append_timeline_legend(svg, legend, left, legend_y, 160)
    svg.append("</svg>")
    return "\n".join(svg)


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
        for key in ("current_start_ms", "prefetch_start_ms", "direct_kv_h2d_start_ms", "replay_kv_h2d_start_ms"):
            value = maybe_float(row.get(key))
            if value is not None:
                start_candidates.append(value)
        for key in ("resume_end_ms", "prefetch_end_ms", "direct_kv_h2d_end_ms", "replay_kv_h2d_end_ms"):
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
    top = 104
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
    legend = [
        ("model turn with tool call", "#2563eb"),
        ("observed tool gap", "#d1d5db"),
        ("resume model turn", "#ef4444"),
        ("resume start boundary", "#111827"),
    ]
    if show_prefetch_legend:
        legend.insert(2, ("live prefetch attempt", "#a855f7"))
        legend.insert(3, ("direct KV HtoD copy", "#16a34a"))
        legend.insert(4, ("replay-side KV HtoD", "#06b6d4"))
    append_timeline_legend(svg, legend, left, 30, 220, {"resume start boundary"})
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
        replay_h2d_start = row.get("replay_kv_h2d_start_ms")
        replay_h2d_end = row.get("replay_kv_h2d_end_ms")

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
        if replay_h2d_start not in ("", None) and replay_h2d_end not in ("", None):
            replay_h2d_x1 = x_pos(replay_h2d_start)
            replay_h2d_x2 = x_pos(replay_h2d_end)
            replay_h2d_events = row.get("replay_kv_h2d_events") or ""
            replay_h2d_duration = row.get("replay_kv_h2d_duration_ms") or ""
            svg.append(
                f'<rect x="{replay_h2d_x1:.1f}" y="{y + 42}" width="{max(5, replay_h2d_x2 - replay_h2d_x1):.1f}" height="20" rx="3" fill="#06b6d4" opacity="0.94" stroke="#0e7490" stroke-width="1">'
                f'<title>replay-side KV host-to-device movement; events={fmt(replay_h2d_events)}; duration_ms={fmt(replay_h2d_duration)}</title></rect>'
            )
            svg.append(
                f'<text x="{max(left + 2, replay_h2d_x1 + 3):.1f}" y="{y + 56}" font-size="9" fill="#ffffff" font-weight="700">replay KV</text>'
            )
    legend_y = height - 28
    append_timeline_legend(svg, legend, left, legend_y, 220, {"resume start boundary"})
    svg.append("</svg>")
    return "\n".join(svg)


def build_expanded_gap_timeline_svg(
    gaps: list[dict[str, Any]],
    max_rows: int,
    show_prefetch_legend: bool = True,
    scale: str = "linear",
) -> str:
    if scale not in {"linear", "symlog"}:
        raise ValueError(f"unsupported expanded timeline scale: {scale}")
    rows = gaps[:max_rows]
    if not rows:
        return "<p>No expanded live tool-gap timeline available.</p>"

    rel_values: list[float] = []
    for row in rows:
        due = maybe_float(row.get("tool_gap_end_ms")) or maybe_float(row.get("resume_start_ms"))
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
            "replay_kv_h2d_start_ms",
            "replay_kv_h2d_end_ms",
            "replay_prefill_start_ms",
            "replay_prefill_end_ms",
        ):
            value = maybe_float(row.get(key))
            if value is not None:
                rel_values.append(value - due)
        replay_start = maybe_float(row.get("resume_start_ms"))
        replay_end = maybe_float(row.get("resume_end_ms"))
        if replay_start is not None:
            rel_values.append(replay_start - due)
        if replay_end is not None:
            if scale == "symlog":
                rel_values.append(replay_end - due)
            else:
                rel_values.append(min(600.0, replay_end - due))

    start = min(rel_values or [-500.0])
    end = max(rel_values or [500.0])
    start = min(start - 60.0, -120.0)
    end = max(end + 80.0, 220.0)
    width = 1580
    left = 178
    right = 70
    top = 112
    row_h = 118
    plot_w = width - left - right
    plot_bottom = top + len(rows) * row_h + 10
    legend_y = plot_bottom + 48
    axis_label_y = legend_y + 42
    height = axis_label_y + 28

    def symlog(value: float, linear_width: float = 50.0) -> float:
        if value == 0:
            return 0.0
        return math.copysign(math.log1p(abs(value) / linear_width), value)

    if scale == "symlog":
        scaled_start = symlog(start)
        scaled_end = symlog(end)
    else:
        scaled_start = start
        scaled_end = end
    scaled_span = max(1e-9, scaled_end - scaled_start)

    def rel(row: dict[str, Any], key: str, due: float) -> float | None:
        value = maybe_float(row.get(key))
        if value is None:
            return None
        return value - due

    def x_pos(relative_ms: float) -> float:
        scaled = symlog(relative_ms) if scale == "symlog" else relative_ms
        return left + (scaled - scaled_start) / scaled_span * plot_w

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
    scale_label = "symlog full replay" if scale == "symlog" else "linear focused"
    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Expanded per-gap timeline aligned at replay due, {scale_label} view">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    legend = [
        ("initial model turn", "#2563eb"),
        ("tool wait", "#d1d5db"),
        ("replay due", "#111827"),
        ("replay decode", "#ef4444"),
        ("normal prefill", "#eab308"),
        ("recompute", "#db2777"),
    ]
    if show_prefetch_legend:
        legend.insert(2, ("prefetch attempt", "#a855f7"))
        legend.insert(3, ("direct KV HtoD", "#16a34a"))
        legend.insert(4, ("replay-side KV HtoD", "#06b6d4"))
    append_timeline_legend(svg, legend, left, 34, 185, {"replay due"})
    for idx in range(len(rows)):
        y = top + idx * row_h
        band_fill = "#ffffff" if idx % 2 == 0 else "#eef2f7"
        svg.append(
            f'<rect x="0" y="{y:.1f}" width="{width}" height="{row_h:.1f}" '
            f'fill="{band_fill}" opacity="0.92"/>'
        )
        svg.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
    svg.append(f'<line x1="0" y1="{plot_bottom:.1f}" x2="{width}" y2="{plot_bottom:.1f}" stroke="#e5e7eb"/>')

    if scale == "symlog":
        tick_candidates = [
            -100000.0,
            -50000.0,
            -10000.0,
            -5000.0,
            -1000.0,
            -500.0,
            -100.0,
            -50.0,
            -10.0,
            0.0,
            10.0,
            50.0,
            100.0,
            500.0,
            1000.0,
            5000.0,
            10000.0,
            50000.0,
            100000.0,
        ]
        tick_values = [value for value in tick_candidates if start <= value <= end]
        for value in (start, end):
            if all(abs(value - tick) > 1 for tick in tick_values):
                tick_values.append(value)
        tick_values = sorted(tick_values)
    else:
        tick_values = [start + (end - start) * tick / 6 for tick in range(7)]
    seen_ticks: set[int] = set()
    last_labeled_x = -10_000.0
    for ms in tick_values:
        rounded = int(round(ms))
        if rounded in seen_ticks:
            continue
        seen_ticks.add(rounded)
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{plot_bottom:.1f}" stroke="#e5e7eb"/>')
        should_label = rounded == 0 or abs(x - last_labeled_x) >= 58
        if should_label:
            svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle" font-size="10">{ms:.0f} ms</text>')
            last_labeled_x = x
    svg.append(f'<line x1="{zero_x:.1f}" y1="{top - 40}" x2="{zero_x:.1f}" y2="{plot_bottom:.1f}" stroke="#111827" stroke-width="3"/>')
    svg.append(f'<text x="{zero_x + 7:.1f}" y="{top - 48}" font-size="12" font-weight="700">0 ms replay due</text>')
    svg.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{axis_label_y:.1f}" text-anchor="middle" font-size="13" font-weight="700">local time around each gap ({scale_label}): negative = before replay due, positive = after replay due</text>'
    )

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        due = maybe_float(row.get("tool_gap_end_ms")) or maybe_float(row.get("resume_start_ms"))
        if due is None:
            continue
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        gap_ms = maybe_float(row.get("tool_gap_ms")) or 0.0
        margin = maybe_float(row.get("prefetch_margin_ms"))
        if margin is not None:
            status = f"READY +{margin:.0f} ms" if margin >= 0 else f"LATE -{abs(margin):.0f} ms"
            status_color = "#166534" if margin >= 0 else "#b91c1c"
        else:
            status = "NO PREFETCH"
            status_color = "#64748b"
        kv_outcome, kv_outcome_color, kv_outcome_title = timeline_kv_outcome(row)
        svg.append(f'<line x1="{left}" y1="{y + 39:.1f}" x2="{left + plot_w}" y2="{y + 39:.1f}" stroke="#f1f5f9"/>')
        svg.append(f'<line x1="{left}" y1="{y + 74:.1f}" x2="{left + plot_w}" y2="{y + 74:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="10" y="{y + 27}" font-size="15" font-weight="700">{fmt(label)}</text>')
        svg.append(f'<text x="10" y="{y + 49}" font-size="12" fill="{status_color}" font-weight="700">{fmt(status)}</text>')
        svg.append(f'<text x="10" y="{y + 68}" font-size="11" fill="{kv_outcome_color}" font-weight="700"><title>{fmt(kv_outcome_title)}</title>{fmt(kv_outcome)}</text>')
        svg.append(f'<text x="10" y="{y + 87}" font-size="10" fill="#64748b">wait {gap_ms:.0f} ms</text>')
        svg.append(f'<text x="{left - 52}" y="{y + 25}" font-size="10" fill="#64748b" text-anchor="end">turn</text>')
        if show_prefetch_legend:
            svg.append(f'<text x="{left - 52}" y="{y + 58}" font-size="10" fill="#64748b" text-anchor="end">prefetch</text>')
        svg.append(f'<text x="{left - 52}" y="{y + 94}" font-size="10" fill="#64748b" text-anchor="end">replay</text>')

        upper_y = y + 12
        mid_y = y + 45
        lower_y = y + 82
        bar_h = 20
        overlay_h = 13

        current_start = rel(row, "current_start_ms", due)
        current_end = rel(row, "current_end_ms", due)
        wait_start = rel(row, "tool_gap_start_ms", due)
        wait_end = rel(row, "tool_gap_end_ms", due)
        prefetch_start = rel(row, "prefetch_start_ms", due)
        prefetch_end = rel(row, "prefetch_end_ms", due)
        h2d_start = rel(row, "direct_kv_h2d_start_ms", due)
        h2d_end = rel(row, "direct_kv_h2d_end_ms", due)
        replay_h2d_start = rel(row, "replay_kv_h2d_start_ms", due)
        replay_h2d_end = rel(row, "replay_kv_h2d_end_ms", due)
        replay_prefill_start = rel(row, "replay_prefill_start_ms", due)
        replay_prefill_end = rel(row, "replay_prefill_end_ms", due)
        replay_start = rel(row, "resume_start_ms", due)
        replay_end = rel(row, "resume_end_ms", due)

        if wait_start is not None and wait_end is not None:
            rect(
                svg,
                x_pos(wait_start),
                x_pos(wait_end),
                upper_y,
                bar_h,
                "#d1d5db",
                f"tool wait window {gap_ms:.3f} ms",
                0.68,
                12,
            )
        if current_start is not None and current_end is not None:
            rect(
                svg,
                x_pos(current_start),
                x_pos(current_end),
                upper_y,
                bar_h,
                "#2563eb",
                "initial model turn that emitted tool call",
                0.9,
                12,
            )
        if show_prefetch_legend and prefetch_start is not None and prefetch_end is not None:
            rect(
                svg,
                x_pos(prefetch_start),
                x_pos(prefetch_end),
                mid_y,
                bar_h,
                "#a855f7",
                f"prefetch attempt; status={row.get('prefetch_status', '')}; duration_ms={row.get('prefetch_duration_ms', '')}",
                0.76,
                12,
            )
        if show_prefetch_legend and h2d_start is not None and h2d_end is not None:
            rect(
                svg,
                x_pos(h2d_start),
                x_pos(h2d_end),
                mid_y,
                bar_h,
                "#16a34a",
                f"direct KV HtoD movement; events={row.get('direct_kv_h2d_events', '')}; duration_ms={row.get('direct_kv_h2d_duration_ms', '')}",
                0.92,
                14,
            )
        first_token_rel = replay_start
        if replay_start is not None:
            if replay_prefill_end is not None and replay_prefill_end >= replay_start:
                first_token_rel = replay_prefill_end
            replay_display_end = max(first_token_rel or replay_start, replay_start + 260.0)
            continues = False
            if replay_end is not None:
                if scale == "symlog":
                    replay_display_end = replay_end
                    continues = False
                else:
                    replay_display_end = min(replay_end, replay_start + 520.0)
                    continues = replay_end > replay_display_end
            rect(
                svg,
                x_pos(first_token_rel or replay_start),
                x_pos(replay_display_end),
                lower_y,
                bar_h,
                "#ef4444",
                "resume decode/generation after first token",
                0.68,
                12,
            )
            if first_token_rel is not None and replay_display_end > first_token_rel:
                label_x = x_pos(first_token_rel) + 4
                svg.append(
                    f'<text x="{label_x:.1f}" y="{lower_y - 4:.1f}" font-size="9" fill="#991b1b" font-weight="700">'
                    f'{fmt(kv_outcome)}</text>'
                )
        if replay_prefill_start is not None and replay_prefill_end is not None:
            replay_h2d_duration = maybe_float(row.get("replay_kv_h2d_duration_ms")) or 0.0
            recompute_ms = replay_recompute_segment_ms(row)
            ttft_ms = max(0.0, replay_prefill_end - replay_prefill_start)
            known_ms = min(ttft_ms, replay_h2d_duration + recompute_ms)
            gold_ms = max(0.0, ttft_ms - known_ms)
            cursor_ms = replay_prefill_start
            if recompute_ms > 0 and (replay_h2d_start is None or recompute_ms >= ttft_ms * 0.5):
                recompute_end = min(replay_prefill_end, cursor_ms + recompute_ms)
                rect(
                    svg,
                    x_pos(cursor_ms),
                    x_pos(recompute_end),
                    lower_y + 3,
                    overlay_h,
                    "#db2777",
                    f"replay recompute / rebuilt prefix; recomputed_tokens_est={row.get('recomputed_tokens_est', row.get('replay_new_prefill_tokens_est', ''))}",
                    0.9,
                    14,
                )
                cursor_ms = recompute_end
            if gold_ms > 0:
                gold_end = min(replay_prefill_end, cursor_ms + gold_ms)
                rect(
                    svg,
                    x_pos(cursor_ms),
                    x_pos(gold_end),
                    lower_y + 3,
                    overlay_h,
                    "#eab308",
                    f"normal remaining replay prefill / queue work; estimated_ms={gold_ms:.3f}",
                    0.88,
                    14,
                )
                cursor_ms = gold_end
            if recompute_ms > 0 and cursor_ms < replay_prefill_end and replay_h2d_start is not None and recompute_ms < ttft_ms * 0.5:
                rect(
                    svg,
                    x_pos(cursor_ms),
                    x_pos(replay_prefill_end),
                    lower_y + 3,
                    overlay_h,
                    "#db2777",
                    f"replay recompute / rebuilt prefix; recomputed_tokens_est={row.get('recomputed_tokens_est', row.get('replay_new_prefill_tokens_est', ''))}",
                    0.9,
                    14,
                )
        if replay_prefill_start is not None and replay_prefill_end is not None and replay_prefill_end - replay_prefill_start >= 12:
            svg.append(
                f'<text x="{max(left + 2, x_pos(replay_prefill_start) + 3):.1f}" y="{lower_y + 13:.1f}" '
                f'font-size="9" fill="#713f12" font-weight="700">TTFT</text>'
            )
        if show_prefetch_legend and replay_h2d_start is not None and replay_h2d_end is not None:
            rect(
                svg,
                x_pos(replay_h2d_start),
                x_pos(replay_h2d_end),
                lower_y + 4,
                overlay_h,
                "#06b6d4",
                f"replay-side KV HtoD movement; events={row.get('replay_kv_h2d_events', '')}; duration_ms={row.get('replay_kv_h2d_duration_ms', '')}",
                0.92,
                14,
            )
        if margin is not None and prefetch_end is not None:
            y_margin = lower_y + bar_h + 9
            x_done = x_pos(prefetch_end)
            color = "#16a34a" if margin >= 0 else "#dc2626"
            svg.append(
                f'<line x1="{min(x_done, zero_x):.1f}" y1="{y_margin:.1f}" x2="{max(x_done, zero_x):.1f}" y2="{y_margin:.1f}" '
                f'stroke="{color}" stroke-width="2" stroke-dasharray="7 5"/>'
            )
            svg.append(f'<circle cx="{x_done:.1f}" cy="{y_margin:.1f}" r="3.5" fill="{color}"/>')

    svg.append(f'<line x1="{left}" y1="{plot_bottom + 18:.1f}" x2="{left + plot_w}" y2="{plot_bottom + 18:.1f}" stroke="#e5e7eb"/>')
    append_timeline_legend(svg, legend, left, legend_y, 185, {"replay due"})
    svg.append("</svg>")
    return "\n".join(svg)


def build_replay_execution_timeline_svg(
    gaps: list[dict[str, Any]],
    max_rows: int,
    show_prefetch_legend: bool = True,
) -> str:
    rows = gaps[:max_rows]
    if not rows:
        return "<p>No replay execution timeline available.</p>"

    rel_values: list[float] = [0.0]
    for row in rows:
        replay_start = maybe_float(row.get("resume_start_ms"))
        if replay_start is None:
            continue
        for key in (
            "resume_end_ms",
            "replay_prefill_end_ms",
            "replay_kv_h2d_start_ms",
            "replay_kv_h2d_end_ms",
        ):
            value = maybe_float(row.get(key))
            if value is not None:
                rel_values.append(max(0.0, value - replay_start))

    end = max(max(rel_values or [1000.0]) + 250.0, 1000.0)
    width = 1580
    left = 178
    right = 70
    top = 112
    row_h = 78
    plot_w = width - left - right
    plot_bottom = top + len(rows) * row_h + 10
    legend_y = plot_bottom + 48
    axis_label_y = legend_y + 42
    height = axis_label_y + 28

    def x_pos(relative_ms: float) -> float:
        return left + relative_ms / end * plot_w

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

    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Replay execution timeline aligned at actual resume start">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    legend = [
        ("replay decode", "#ef4444"),
        ("normal prefill", "#eab308"),
        ("recompute", "#db2777"),
        ("replay-side KV HtoD", "#06b6d4"),
        ("resume start", "#111827"),
    ]
    append_timeline_legend(svg, legend, left, 34, 220, {"resume start"})
    for idx in range(len(rows)):
        y = top + idx * row_h
        band_fill = "#ffffff" if idx % 2 == 0 else "#eef2f7"
        svg.append(f'<rect x="0" y="{y:.1f}" width="{width}" height="{row_h:.1f}" fill="{band_fill}" opacity="0.92"/>')
        svg.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
    svg.append(f'<line x1="0" y1="{plot_bottom:.1f}" x2="{width}" y2="{plot_bottom:.1f}" stroke="#e5e7eb"/>')

    tick_count = 6
    for tick in range(tick_count + 1):
        value = end * tick / tick_count
        x = x_pos(value)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{plot_bottom:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle" font-size="10">{value:.0f} ms</text>')
    zero_x = x_pos(0.0)
    svg.append(f'<line x1="{zero_x:.1f}" y1="{top - 40}" x2="{zero_x:.1f}" y2="{plot_bottom:.1f}" stroke="#111827" stroke-width="3"/>')
    svg.append(f'<text x="{zero_x + 7:.1f}" y="{top - 48}" font-size="12" font-weight="700">0 ms resume starts</text>')
    svg.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{axis_label_y:.1f}" text-anchor="middle" font-size="13" font-weight="700">time inside actual resume request: cyan = host KV load, magenta = recompute, gold = remaining TTFT, red = decode after first token</text>'
    )

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        replay_start = maybe_float(row.get("resume_start_ms"))
        replay_end = maybe_float(row.get("resume_end_ms"))
        ttft = maybe_float(row.get("resume_ttft_ms"))
        due = maybe_float(row.get("tool_gap_end_ms"))
        start_delay = replay_start - due if replay_start is not None and due is not None else None
        margin = maybe_float(row.get("prefetch_margin_ms"))
        if margin is not None:
            status = f"LATE -{abs(margin):.0f} ms" if margin < 0 else f"READY +{margin:.0f} ms"
            status_color = "#b91c1c" if margin < 0 else "#166534"
        else:
            status = "NO PREFETCH"
            status_color = "#64748b"
        delay_text = f"start delay {start_delay:.0f} ms" if start_delay is not None else "start delay unknown"
        kv_outcome, kv_outcome_color, kv_outcome_title = timeline_kv_outcome(row)

        svg.append(f'<text x="10" y="{y + 23}" font-size="15" font-weight="700">{fmt(label)}</text>')
        svg.append(f'<text x="10" y="{y + 43}" font-size="12" fill="{status_color}" font-weight="700">{fmt(status)}</text>')
        svg.append(f'<text x="10" y="{y + 60}" font-size="10" fill="{kv_outcome_color}" font-weight="700"><title>{fmt(kv_outcome_title)}</title>{fmt(kv_outcome)}</text>')
        svg.append(f'<text x="10" y="{y + 73}" font-size="9" fill="#64748b">{fmt(delay_text)}</text>')

        bar_y = y + 22
        bar_h = 24
        overlay_y = y + 27
        overlay_h = 14
        first_token_ms = ttft if ttft is not None else 0.0
        if replay_start is not None and replay_end is not None:
            decode_end_ms = max(first_token_ms, replay_end - replay_start)
            rect(
                svg,
                x_pos(max(0.0, first_token_ms)),
                x_pos(max(0.0, decode_end_ms)),
                bar_y,
                bar_h,
                "#ef4444",
                f"resume decode/generation after first token; latency_ms={row.get('resume_latency_ms', '')}",
                0.56,
                16,
            )
        if ttft is not None:
            replay_h2d_duration = maybe_float(row.get("replay_kv_h2d_duration_ms")) or 0.0
            recompute_ms = replay_recompute_segment_ms(row)
            known_ms = min(ttft, replay_h2d_duration + recompute_ms)
            gold_ms = max(0.0, ttft - known_ms)
            cursor_ms = 0.0
            replay_h2d_start = maybe_float(row.get("replay_kv_h2d_start_ms"))
            replay_h2d_end = maybe_float(row.get("replay_kv_h2d_end_ms"))
            if recompute_ms > 0 and (replay_h2d_start is None or recompute_ms >= ttft * 0.5):
                recompute_end = min(ttft, cursor_ms + recompute_ms)
                rect(
                    svg,
                    x_pos(cursor_ms),
                    x_pos(recompute_end),
                    overlay_y,
                    overlay_h,
                    "#db2777",
                    f"replay recompute / rebuilt prefix; recomputed_tokens_est={row.get('recomputed_tokens_est', row.get('replay_new_prefill_tokens_est', ''))}",
                    0.92,
                    16,
                )
                cursor_ms = recompute_end
            if gold_ms > 0:
                gold_end = min(ttft, cursor_ms + gold_ms)
                rect(
                    svg,
                    x_pos(cursor_ms),
                    x_pos(gold_end),
                    overlay_y,
                    overlay_h,
                    "#eab308",
                    f"normal remaining replay prefill / queue work; estimated_ms={gold_ms:.3f}",
                    0.9,
                    16,
                )
                cursor_ms = gold_end
            if recompute_ms > 0 and cursor_ms < ttft and replay_h2d_start is not None and recompute_ms < ttft * 0.5:
                rect(
                    svg,
                    x_pos(cursor_ms),
                    x_pos(ttft),
                    overlay_y,
                    overlay_h,
                    "#db2777",
                    f"replay recompute / rebuilt prefix; recomputed_tokens_est={row.get('recomputed_tokens_est', row.get('replay_new_prefill_tokens_est', ''))}",
                    0.92,
                    16,
                )
            if x_pos(max(0.0, ttft)) - x_pos(0.0) >= 64:
                svg.append(
                    f'<text x="{x_pos(max(0.0, ttft)) - 6:.1f}" y="{overlay_y + 11:.1f}" '
                    f'font-size="10" fill="#78350f" text-anchor="end" font-weight="700">TTFT {ttft:.0f} ms</text>'
                )
        replay_h2d_start = maybe_float(row.get("replay_kv_h2d_start_ms"))
        replay_h2d_end = maybe_float(row.get("replay_kv_h2d_end_ms"))
        if show_prefetch_legend and replay_start is not None and replay_h2d_start is not None and replay_h2d_end is not None:
            rect(
                svg,
                x_pos(max(0.0, replay_h2d_start - replay_start)),
                x_pos(max(0.0, replay_h2d_end - replay_start)),
                y + 49,
                12,
                "#06b6d4",
                f"replay-side KV HtoD movement; events={row.get('replay_kv_h2d_events', '')}; duration_ms={row.get('replay_kv_h2d_duration_ms', '')}",
                0.95,
                14,
            )

    svg.append(f'<line x1="{left}" y1="{plot_bottom + 18:.1f}" x2="{left + plot_w}" y2="{plot_bottom + 18:.1f}" stroke="#e5e7eb"/>')
    append_timeline_legend(svg, legend, left, legend_y, 220, {"resume start"})
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
    <h3>Symlog Full Replay View</h3>
    <p>This view uses a symmetric log-style time axis. It keeps detail around 0 ms while compressing very long replay requests, so red replay bars can extend to their true end without hiding the prefetch/copy boundary.</p>
    {build_expanded_gap_timeline_svg(gaps, max_timeline_gaps, scale="symlog")}
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
