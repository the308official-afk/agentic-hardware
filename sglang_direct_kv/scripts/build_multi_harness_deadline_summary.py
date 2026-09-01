#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HARNESS_LABELS = {
    "hatcher": "Hatcher",
    "codex": "Codex",
    "claude_code": "Claude Code",
    "opencode": "OpenCode",
    "qwen_code": "Qwen Code",
    "nemo_agent_toolkit": "NeMo Agent Toolkit / NAT",
    "deepseek_harness": "DeepSeek Harness",
    "pi_agent_harness": "Pi Agent Harness",
    "openclaw": "OpenClaw",
    "hermes_agent": "Hermes Agent",
}

MODE_LABELS = {
    "no_prefetch": "NP = No prefetch",
    "e2e_priority_hints": "E2E = End-to-end priority hints",
}

MODE_COLORS = {
    "no_prefetch": "#2563eb",
    "e2e_priority_hints": "#0f766e",
}

HARNESS_SYMBOLS = {
    "hatcher": "circle",
    "codex": "square",
    "claude_code": "triangle",
    "opencode": "diamond",
    "qwen_code": "cross",
    "nemo_agent_toolkit": "plus",
    "deepseek_harness": "star",
    "pi_agent_harness": "hexagon",
    "openclaw": "triangle-down",
    "hermes_agent": "ring",
}

PRESSURE_LABELS = {
    "p0_control": "P0 Control",
    "p1_mild": "P1 Mild",
    "p2_medium": "P2 Medium",
    "p3_high": "P3 Queue Pressure",
    "p4_cliff": "P4 Cliff",
    "p5_boss_queue": "P5 Boss Queue",
}

PRESSURE_ORDER = (
    "p0_control",
    "p1_mild",
    "p2_medium",
    "p3_high",
    "p4_cliff",
    "p5_boss_queue",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def case_key_from_name(name: str) -> tuple[str, str, str]:
    for harness in sorted(HARNESS_LABELS, key=len, reverse=True):
        prefix = f"{harness}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        for pressure in sorted(PRESSURE_LABELS, key=len, reverse=True):
            prefix_pressure = f"{pressure}_"
            if not rest.startswith(prefix_pressure):
                continue
            mode_part = rest[len(prefix_pressure) :]
            mode = "e2e_priority_hints" if mode_part.startswith("e2e_priority_hints") else "no_prefetch"
            return harness, pressure, mode
    return "", "", ""


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_rows(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        harness, pressure, mode = case_key_from_name(case_dir.name)
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        due_by_session: dict[str, dict[str, Any]] = {}
        replay_starts: list[dict[str, Any]] = []
        replay_ends: list[dict[str, Any]] = []
        for row in trace_rows:
            event = row.get("event")
            phase = row.get("phase")
            if event == "m27.replay.due":
                due_by_session[str(row.get("session_id") or "")] = row
            elif event == "m27.request.start" and phase == "replay":
                replay_starts.append(row)
            elif event == "m27.request.end" and phase == "replay":
                replay_ends.append(row)
        start_by_label = {str(row.get("label") or row.get("request_id") or ""): row for row in replay_starts}
        for end in replay_ends:
            label = str(end.get("label") or end.get("request_id") or "")
            session_id = str(end.get("session_id") or "")
            start = start_by_label.get(label, {})
            due = due_by_session.get(session_id, {})
            start_ts_ns = int(float_value(start.get("ts_ns")))
            due_ts_ns = int(float_value(due.get("ts_ns")))
            ttft_ms = float_value(end.get("ttft_ms"))
            first_token_ts_ns = start_ts_ns + int(round(ttft_ms * 1_000_000))
            lateness_ms = ((first_token_ts_ns - due_ts_ns) / 1_000_000.0) if due_ts_ns and start_ts_ns else float("nan")
            out.append(
                {
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                    "harness": str(end.get("harness") or harness),
                    "harness_label": HARNESS_LABELS.get(str(end.get("harness") or harness), str(end.get("harness") or harness)),
                    "mode": str(end.get("mode") or mode),
                    "mode_label": MODE_LABELS.get(str(end.get("mode") or mode), str(end.get("mode") or mode)),
                    "pressure_level": pressure,
                    "pressure_level_label": PRESSURE_LABELS.get(pressure, pressure),
                    "session_id": session_id,
                    "request_id": label,
                    "first_token_lateness_ms": round(lateness_ms, 3) if math.isfinite(lateness_ms) else "",
                    "ttft_ms": round(ttft_ms, 3),
                    "request_start_ts_ns": start_ts_ns,
                    "replay_due_ts_ns": due_ts_ns,
                    "request_end_ts_ns": int(float_value(end.get("ts_ns"))),
                    "sglang_priority": end.get("sglang_priority", ""),
                    "status": end.get("status", ""),
                    "error": end.get("error", ""),
                }
            )
    return out


RAW_COLUMNS = [
    "harness",
    "harness_label",
    "pressure_level",
    "pressure_level_label",
    "mode",
    "mode_label",
    "session_id",
    "request_id",
    "first_token_lateness_ms",
    "ttft_ms",
    "sglang_priority",
    "status",
    "error",
    "case_id",
    "case_dir",
    "replay_due_ts_ns",
    "request_start_ts_ns",
    "request_end_ts_ns",
]

SUMMARY_COLUMNS = [
    "harness",
    "harness_label",
    "pressure_level",
    "pressure_level_label",
    "mode",
    "mode_label",
    "samples",
    "median_first_token_lateness_ms",
    "min_first_token_lateness_ms",
    "max_first_token_lateness_ms",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("first_token_lateness_ms")
        if value == "":
            continue
        grouped[(str(row["harness"]), str(row["pressure_level"]), str(row["mode"]))].append(float(value))
    out: list[dict[str, Any]] = []
    for (harness, pressure, mode), values in sorted(grouped.items(), key=lambda item: (item[0][0], PRESSURE_ORDER.index(item[0][1]) if item[0][1] in PRESSURE_ORDER else 999, item[0][2])):
        out.append(
            {
                "harness": harness,
                "harness_label": HARNESS_LABELS.get(harness, harness),
                "pressure_level": pressure,
                "pressure_level_label": PRESSURE_LABELS.get(pressure, pressure),
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "samples": len(values),
                "median_first_token_lateness_ms": round(statistics.median(values), 3),
                "min_first_token_lateness_ms": round(min(values), 3),
                "max_first_token_lateness_ms": round(max(values), 3),
            }
        )
    return out


def symlog(value: float, linear_threshold: float = 50.0) -> float:
    sign = -1.0 if value < 0 else 1.0
    value = abs(value)
    if value <= linear_threshold:
        return sign * (value / linear_threshold)
    return sign * (1.0 + math.log10(value / linear_threshold))


def svg_symbol(kind: str, x: float, y: float, color: str, title: str) -> str:
    escaped_title = html.escape(title)
    common = f'fill="{color}" stroke="{color}" stroke-width="2" opacity="0.88"'
    if kind == "square":
        shape = f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" rx="2" {common}/>'
    elif kind == "triangle":
        points = f"{x:.1f},{y-7:.1f} {x-6.5:.1f},{y+5.5:.1f} {x+6.5:.1f},{y+5.5:.1f}"
        shape = f'<polygon points="{points}" {common}/>'
    elif kind == "triangle-down":
        points = f"{x:.1f},{y+7:.1f} {x-6.5:.1f},{y-5.5:.1f} {x+6.5:.1f},{y-5.5:.1f}"
        shape = f'<polygon points="{points}" {common}/>'
    elif kind == "diamond":
        points = f"{x:.1f},{y-7:.1f} {x+7:.1f},{y:.1f} {x:.1f},{y+7:.1f} {x-7:.1f},{y:.1f}"
        shape = f'<polygon points="{points}" {common}/>'
    elif kind == "cross":
        shape = (
            f'<line x1="{x-6:.1f}" x2="{x+6:.1f}" y1="{y-6:.1f}" y2="{y+6:.1f}" {common}/>'
            f'<line x1="{x-6:.1f}" x2="{x+6:.1f}" y1="{y+6:.1f}" y2="{y-6:.1f}" {common}/>'
        )
    elif kind == "plus":
        shape = (
            f'<line x1="{x-7:.1f}" x2="{x+7:.1f}" y1="{y:.1f}" y2="{y:.1f}" {common}/>'
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y-7:.1f}" y2="{y+7:.1f}" {common}/>'
        )
    elif kind == "star":
        points = []
        for i in range(10):
            radius = 7 if i % 2 == 0 else 3.2
            angle = -math.pi / 2 + i * math.pi / 5
            points.append(f"{x + math.cos(angle) * radius:.1f},{y + math.sin(angle) * radius:.1f}")
        shape = f'<polygon points="{" ".join(points)}" {common}/>'
    elif kind == "hexagon":
        points = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            points.append(f"{x + math.cos(angle) * 7:.1f},{y + math.sin(angle) * 7:.1f}")
        shape = f'<polygon points="{" ".join(points)}" {common}/>'
    elif kind == "ring":
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.2" fill="#ffffff" stroke="{color}" stroke-width="2.4" opacity="0.95"/>'
    else:
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" {common}/>'
    return f'<g><title>{escaped_title}</title>{shape}</g>'


def render_pressure_chart(rows: list[dict[str, Any]]) -> str:
    pressures = [pressure for pressure in PRESSURE_ORDER if any(row["pressure_level"] == pressure for row in rows)]
    harnesses = [harness for harness in HARNESS_LABELS if any(row["harness"] == harness for row in rows)]
    if not pressures or not harnesses:
        return "<p>No replay rows found.</p>"

    values = [float(row["first_token_lateness_ms"]) for row in rows if row.get("first_token_lateness_ms") != ""]
    transformed = [symlog(value) for value in values] + [symlog(0.0), symlog(50.0), symlog(500.0), symlog(10_000.0)]
    y_min = min(transformed)
    y_max = max(transformed)
    pad = max(0.2, (y_max - y_min) * 0.08)
    y_min -= pad
    y_max += pad

    pressure_w = max(420, len(harnesses) * 44 + 110)
    width = max(1400, pressure_w * len(pressures) + 220)
    height = 820
    left = 120
    right = 40
    top = 60
    bottom = 260
    plot_w = width - left - right
    plot_h = height - top - bottom
    pressure_group_w = plot_w / len(pressures)

    def y_pos(value: float) -> float:
        mapped = symlog(value)
        return top + (y_max - mapped) / (y_max - y_min) * plot_h

    def x_pos(pressure_index: int, harness_index: int, mode: str, sample_index: int, sample_count: int) -> float:
        pressure_left = left + pressure_index * pressure_group_w
        harness_step = pressure_group_w / max(1, len(harnesses))
        base = pressure_left + harness_step * (harness_index + 0.5)
        mode_offset = -9 if mode == "no_prefetch" else 9
        jitter = 0.0 if sample_count <= 1 else (sample_index - (sample_count - 1) / 2) * 3.2
        return base + mode_offset + jitter

    lines = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Replay Deadline Pressure Chart">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    tick_values = [-1000, -500, -100, 0, 50, 500, 1000, 5000, 10000, 60000]
    for tick in tick_values:
        y = y_pos(float(tick))
        stroke = "#111827" if tick == 0 else "#e5e7eb"
        width_attr = "1.5" if tick == 0 else "1"
        lines.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{width_attr}"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#374151">{tick} ms</text>')
    lines.append(f'<text x="{width-right-4}" y="{y_pos(0)-8:.1f}" text-anchor="end" font-size="13" font-weight="700">0 ms deadline</text>')

    rows_by_group_mode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_group_mode[(str(row["pressure_level"]), str(row["harness"]), str(row["mode"]))].append(row)

    for pressure_index, pressure in enumerate(pressures):
        x = left + pressure_index * pressure_group_w
        lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height-bottom}" stroke="#cbd5e1" stroke-dasharray="5 6"/>')
        if pressure_index % 2 == 1:
            lines.append(f'<rect x="{x:.1f}" y="{top}" width="{pressure_group_w:.1f}" height="{plot_h}" fill="#f8fafc" opacity="0.62"/>')
        cx = x + pressure_group_w / 2
        lines.append(f'<text x="{cx:.1f}" y="{height-bottom+36}" text-anchor="middle" font-size="16" font-weight="800" fill="#111827">{html.escape(PRESSURE_LABELS.get(pressure, pressure))}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{height-bottom+56}" text-anchor="middle" font-size="11" fill="#64748b">all harnesses overlaid; blue = baseline, green = E2E</text>')
        for harness_index, harness in enumerate(harnesses):
            harness_x = x_pos(pressure_index, harness_index, "no_prefetch", 0, 1) + 9
            lines.append(f'<line x1="{harness_x:.1f}" x2="{harness_x:.1f}" y1="{top}" y2="{height-bottom}" stroke="#f1f5f9" stroke-width="1"/>')
            for mode in MODE_LABELS:
                sample_rows = rows_by_group_mode.get((pressure, harness, mode), [])
                sample_rows = [row for row in sample_rows if row.get("first_token_lateness_ms") != ""]
                if not sample_rows:
                    continue
                med = statistics.median(float(row["first_token_lateness_ms"]) for row in sample_rows)
                mx = x_pos(pressure_index, harness_index, mode, 0, 1)
                y = y_pos(med)
                lines.append(f'<line x1="{mx-9:.1f}" x2="{mx+9:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{MODE_COLORS[mode]}" stroke-width="3" stroke-linecap="round"/>')
                for sample_index, row in enumerate(sample_rows):
                    value = float(row["first_token_lateness_ms"])
                    dot_x = x_pos(pressure_index, harness_index, mode, sample_index, len(sample_rows))
                    dot_y = y_pos(value)
                    title = (
                        f"{PRESSURE_LABELS.get(pressure, pressure)} | "
                        f"{HARNESS_LABELS.get(harness, harness)} | "
                        f"{MODE_LABELS[mode]} | {value:.1f} ms late"
                    )
                    lines.append(svg_symbol(HARNESS_SYMBOLS.get(harness, "circle"), dot_x, dot_y, MODE_COLORS[mode], title))

    lines.append(f'<line x1="{width-right:.1f}" x2="{width-right:.1f}" y1="{top}" y2="{height-bottom}" stroke="#cbd5e1" stroke-dasharray="5 6"/>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height-40}" text-anchor="middle" font-size="14" font-weight="700">pressure level</text>')
    lines.append(f'<text transform="translate(32 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="14" font-weight="700">lateness vs replay deadline ms (symlog)</text>')
    legend_y = height - 172
    lines.append(f'<rect x="{left}" y="{legend_y-28}" width="{min(plot_w, 1180):.1f}" height="116" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>')
    legend_x = left + 28
    lines.append(f'<text x="{legend_x}" y="{legend_y-2}" font-size="13" font-weight="800" fill="#111827">Mode color</text>')
    legend_x += 96
    for mode in MODE_LABELS:
        lines.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="6" fill="{MODE_COLORS[mode]}"/>')
        lines.append(f'<text x="{legend_x+14}" y="{legend_y+4}" font-size="13" fill="#111827">{html.escape(MODE_LABELS[mode])}</text>')
        legend_x += 240
    harness_y = legend_y + 48
    harness_x = left + 28
    lines.append(f'<text x="{harness_x}" y="{harness_y+4}" font-size="13" font-weight="800" fill="#111827">Harness symbol</text>')
    harness_x += 126
    for index, harness in enumerate(harnesses):
        if index == 5:
            harness_x = left + 154
            harness_y += 34
        lines.append(svg_symbol(HARNESS_SYMBOLS.get(harness, "circle"), harness_x, harness_y, "#334155", HARNESS_LABELS.get(harness, harness)))
        lines.append(f'<text x="{harness_x+14}" y="{harness_y+4}" font-size="12" fill="#111827">{html.escape(HARNESS_LABELS.get(harness, harness))}</text>')
        harness_x += 190
    lines.append("</svg>")
    return "\n".join(lines)


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_lines = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body_lines.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_lines)}</tbody></table>"


def render_html(rows: list[dict[str, Any]], summary: list[dict[str, Any]], report_label: str) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    chart = render_pressure_chart(rows)
    summary_table = render_table(
        summary,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "samples",
            "median_first_token_lateness_ms",
            "min_first_token_lateness_ms",
            "max_first_token_lateness_ms",
        ],
    )
    raw_table = render_table(
        rows,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "session_id",
            "first_token_lateness_ms",
            "ttft_ms",
            "sglang_priority",
            "status",
            "error",
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Replay Deadline Pressure Chart</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #f8fafc; }}
main {{ max-width: 1600px; margin: 0 auto; padding: 32px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
h2 {{ margin-top: 32px; font-size: 22px; }}
p {{ line-height: 1.5; color: #334155; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-top: 18px; overflow-x: auto; }}
.note {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 16px; color: #1e3a8a; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
th {{ background: #f1f5f9; font-weight: 700; }}
code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
<h1>Replay Deadline Pressure Chart</h1>
<p>Report label: <code>{html.escape(report_label)}</code>. Generated {generated}.</p>
<p class="note">This lightweight all-harness report uses the completed workload traces directly. Each symbol is one replay request. Pressure levels are grouped on the x-axis; harnesses are encoded by shape; mode is encoded by color. Higher means later. Values above <code>0 ms</code> missed the replay deadline.</p>
<div class="card">{chart}</div>
<h2>Summary</h2>
<div class="card">{summary_table}</div>
<h2>Raw Replay Proof</h2>
<div class="card">{raw_table}</div>
</main>
</body>
</html>
"""


def write_manifest(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_kind": "multi_harness_deadline_pressure",
        "report_label": args.report_label,
        "script": "scripts/build_multi_harness_deadline_summary.py",
        "root": str(args.root),
        "report_dir": str(args.out_dir),
        "row_count": len(rows),
        "summary_row_count": len(summary),
        "harnesses": sorted({row["harness"] for row in rows}),
        "pressure_levels": sorted({row["pressure_level"] for row in rows}),
        "modes": sorted({row["mode"] for row in rows}),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight all-harness replay deadline report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--report-label", default=os.environ.get("REPORT_LABEL") or f"multi_harness_deadline_summary_{int(time.time())}")
    parser.add_argument("--update-latest", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.root)
    summary = summarize(rows)
    write_csv(args.out_dir / "global_kv_readiness_by_mode.csv", rows, RAW_COLUMNS)
    write_csv(args.out_dir / "global_kv_readiness_by_mode_summary.csv", summary, SUMMARY_COLUMNS)
    html_text = render_html(rows, summary, args.report_label)
    report_path = args.out_dir / "master_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    write_manifest(args.out_dir / "manifest.json", args, rows, summary)
    if args.latest_root and args.update_latest:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        (args.latest_root / "latest_master_report.html").write_text(html_text, encoding="utf-8")
        write_manifest(args.latest_root / "latest_manifest.json", args, rows, summary)
    print(f"wrote {report_path}")
    print(f"rows={len(rows)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
