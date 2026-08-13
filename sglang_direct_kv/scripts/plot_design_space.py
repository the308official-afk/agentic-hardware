#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path
from typing import Any


TIMING_LABELS = {
    "very_early_before_pressure": "very early",
    "early_before_pressure": "early",
    "middle_during_pressure": "middle",
    "late_after_pressure": "late",
}

COLORS = {
    "very_early_before_pressure": "#2563eb",
    "early_before_pressure": "#16a34a",
    "middle_during_pressure": "#d97706",
    "late_after_pressure": "#dc2626",
    "no_prefetch": "#111827",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in (
            "filler_sessions",
            "prompt_tokens",
            "warm_ttft_avg_ms",
            "resume_ttft_avg_ms",
            "benefit_vs_no_prefetch_ms",
            "benefit_vs_no_prefetch_pct",
            "prefetch_ttft_avg_ms",
            "hicache_load",
            "hicache_evict_device",
        ):
            row[key] = float(row[key])
    return rows


def nice_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = max(1.0, abs(lo) * 0.2)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.12
    return min(0.0, lo - pad), hi + pad


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def render_chart(
    *,
    title: str,
    rows: list[dict[str, Any]],
    y_key: str,
    y_label: str,
    out_path: Path,
    include_baseline: bool = False,
) -> Path:
    width = 960
    height = 560
    left = 90
    right = 220
    top = 70
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    fillers = sorted({row["filler_sessions"] for row in rows})
    y_values = [row[y_key] for row in rows if include_baseline or row["mode"] == "hint_aware"]
    y_min, y_max = nice_range(y_values)
    x_min, x_max = min(fillers), max(fillers)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["mode"] == "no_prefetch":
            if include_baseline:
                grouped["no_prefetch"].append(row)
            continue
        grouped[str(row["hint_timing"])].append(row)

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append(f'<text x="{left}" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{title}</text>')
    lines.append(f'<text x="{left}" y="{height - 28}" font-family="Arial" font-size="14" fill="#374151">Cache pressure: filler sessions</text>')
    lines.append(f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" font-family="Arial" font-size="14" fill="#374151">{y_label}</text>')

    for i in range(5):
        y_val = y_min + (y_max - y_min) * i / 4
        y = scale(y_val, y_min, y_max, top + plot_h, top)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{y_val:.1f}</text>')

    for filler in fillers:
        x = scale(filler, x_min, x_max, left, left + plot_w)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{int(filler)}</text>')

    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>')

    legend_x = left + plot_w + 36
    legend_y = top + 18
    for idx, (name, series) in enumerate(sorted(grouped.items())):
        series = sorted(series, key=lambda row: row["filler_sessions"])
        points = [
            (
                scale(row["filler_sessions"], x_min, x_max, left, left + plot_w),
                scale(row[y_key], y_min, y_max, top + plot_h, top),
            )
            for row in series
        ]
        color = COLORS.get(name, "#6b7280")
        label = TIMING_LABELS.get(name, "no prefetch" if name == "no_prefetch" else name)
        if len(points) >= 2:
            lines.append(f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"/>')
        y_leg = legend_y + idx * 28
        lines.append(f'<line x1="{legend_x}" y1="{y_leg}" x2="{legend_x + 26}" y2="{y_leg}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 36}" y="{y_leg + 5}" font-family="Arial" font-size="13" fill="#111827">{label}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def row_for(rows: list[dict[str, Any]], *, mode: str, filler: float, timing: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        if row["mode"] != mode or row["filler_sessions"] != filler:
            continue
        if timing is not None and row["hint_timing"] != timing:
            continue
        return row
    return None


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}%"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["<table>", "<thead>", "<tr>"]
    for header in headers:
        lines.append(f"<th>{html.escape(header)}</th>")
    lines.extend(["</tr>", "</thead>", "<tbody>"])
    for row in rows:
        lines.append("<tr>")
        for value in row:
            lines.append(f"<td>{html.escape(value)}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def chart_table(section: str, rows: list[dict[str, Any]]) -> str:
    fillers = sorted({row["filler_sessions"] for row in rows})
    table_rows: list[list[str]] = []

    if section == "Benefit vs Cache Pressure":
        for filler in fillers:
            baseline = row_for(rows, mode="no_prefetch", filler=filler)
            early = row_for(rows, mode="hint_aware", filler=filler, timing="early_before_pressure")
            late = row_for(rows, mode="hint_aware", filler=filler, timing="late_after_pressure")
            table_rows.append(
                [
                    str(int(filler)),
                    fmt_ms(baseline["warm_ttft_avg_ms"] if baseline else None),
                    fmt_ms(baseline["resume_ttft_avg_ms"] if baseline else None),
                    fmt_ms(early["resume_ttft_avg_ms"] if early else None),
                    fmt_ms(late["resume_ttft_avg_ms"] if late else None),
                    fmt_ms(early["benefit_vs_no_prefetch_ms"] if early else None),
                    fmt_ms(late["benefit_vs_no_prefetch_ms"] if late else None),
                    fmt_pct(late["benefit_vs_no_prefetch_pct"] if late else None),
                ]
            )
        return render_table(
            ["fillers", "first TTFT", "resume base", "resume early", "resume late", "early benefit", "late benefit", "late %"],
            table_rows,
        )

    if section == "Resume TTFT vs Cache Pressure":
        for filler in fillers:
            baseline = row_for(rows, mode="no_prefetch", filler=filler)
            early = row_for(rows, mode="hint_aware", filler=filler, timing="early_before_pressure")
            late = row_for(rows, mode="hint_aware", filler=filler, timing="late_after_pressure")
            table_rows.append(
                [
                    str(int(filler)),
                    fmt_ms(baseline["warm_ttft_avg_ms"] if baseline else None),
                    fmt_ms(baseline["resume_ttft_avg_ms"] if baseline else None),
                    fmt_ms(early["resume_ttft_avg_ms"] if early else None),
                    fmt_ms(late["resume_ttft_avg_ms"] if late else None),
                ]
            )
        return render_table(
            ["fillers", "first TTFT", "resume base", "resume early", "resume late"],
            table_rows,
        )

    if section == "Prefetch Cost vs Cache Pressure":
        for filler in fillers:
            baseline = row_for(rows, mode="no_prefetch", filler=filler)
            early = row_for(rows, mode="hint_aware", filler=filler, timing="early_before_pressure")
            late = row_for(rows, mode="hint_aware", filler=filler, timing="late_after_pressure")
            table_rows.append(
                [
                    str(int(filler)),
                    fmt_ms(baseline["warm_ttft_avg_ms"] if baseline else None),
                    fmt_ms(early["prefetch_ttft_avg_ms"] if early else None),
                    fmt_ms(late["prefetch_ttft_avg_ms"] if late else None),
                ]
            )
        return render_table(
            ["fillers", "first TTFT", "early prefetch", "late prefetch"],
            table_rows,
        )

    return ""


def render_dashboard(charts: list[tuple[str, Path, list[dict[str, Any]]]], out_path: Path) -> None:
    grouped: dict[str, list[tuple[Path, list[dict[str, Any]]]]] = defaultdict(list)
    for section, path, rows in charts:
        grouped[section].append((path, rows))

    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        "  <title>Milestone 6 Design-Space Charts</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 32px; color: #111827; background: #f9fafb; }",
        "    h1 { margin: 0 0 8px; font-size: 28px; }",
        "    p { margin: 0 0 24px; color: #4b5563; line-height: 1.45; }",
        "    section { margin: 28px 0 42px; }",
        "    h2 { font-size: 20px; margin: 0 0 14px; }",
        "    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }",
        "    .panel { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }",
        "    .chart { min-width: 0; }",
        "    .numbers { overflow-x: auto; margin-top: 10px; }",
        "    img { display: block; width: 100%; height: auto; }",
        "    table { width: 100%; border-collapse: collapse; font-size: 13px; }",
        "    th { text-align: right; background: #f3f4f6; }",
        "    th, td { border-bottom: 1px solid #e5e7eb; padding: 7px 8px; white-space: nowrap; }",
        "    td { text-align: right; }",
        "    td:first-child, th:first-child { text-align: left; }",
        "    code { background: #eef2ff; padding: 2px 5px; border-radius: 4px; }",
        "    @media (max-width: 1180px) { .grid { grid-template-columns: 1fr; } }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Milestone 6 Design-Space Charts</h1>",
        "  <p>Each chart uses cache pressure on the x-axis. Lines represent prefetch timing choices. Separate panels represent prompt sizes. Positive benefit means hint-aware prefetch reduced resume TTFT compared with no_prefetch.</p>",
    ]

    for section in ("Benefit vs Cache Pressure", "Resume TTFT vs Cache Pressure", "Prefetch Cost vs Cache Pressure"):
        panels = grouped.get(section, [])
        if not panels:
            continue
        lines.append("  <section>")
        lines.append(f"    <h2>{html.escape(section)}</h2>")
        lines.append('    <div class="grid">')
        for path, rows in sorted(panels, key=lambda item: item[0].name):
            rel = path.relative_to(out_path.parent)
            alt = path.stem.replace("_", " ")
            lines.append('      <div class="panel">')
            lines.append('        <div class="chart">')
            lines.append(f'          <img src="{html.escape(str(rel))}" alt="{html.escape(alt)}">')
            lines.append("        </div>")
            lines.append('        <div class="numbers">')
            lines.append(chart_table(section, rows))
            lines.append("        </div>")
            lines.append("      </div>")
        lines.append("    </div>")
        lines.append("  </section>")

    lines.extend(
        [
            "  <p>Raw data: <code>summary.csv</code> and <code>summary.json</code> in the parent results directory.</p>",
            "</body>",
            "</html>",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SVG charts for the Milestone 6 design-space sweep.")
    parser.add_argument("--root", default="artifacts/results/milestone6_design_space")
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_rows(root / "summary.csv")
    charts_dir = root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts: list[tuple[str, Path, list[dict[str, Any]]]] = []

    for prompt_tokens in sorted({row["prompt_tokens"] for row in rows}):
        prompt_rows = [row for row in rows if row["prompt_tokens"] == prompt_tokens]
        suffix = f"p{int(prompt_tokens)}"
        charts.append(
            (
                "Benefit vs Cache Pressure",
                render_chart(
                    title=f"Hint-Aware Benefit vs Cache Pressure, prompt_tokens={int(prompt_tokens)}",
                    rows=prompt_rows,
                    y_key="benefit_vs_no_prefetch_ms",
                    y_label="Benefit vs no_prefetch (ms)",
                    out_path=charts_dir / f"benefit_vs_pressure_{suffix}.svg",
                ),
                prompt_rows,
            )
        )
        charts.append(
            (
                "Resume TTFT vs Cache Pressure",
                render_chart(
                    title=f"Resume TTFT vs Cache Pressure, prompt_tokens={int(prompt_tokens)}",
                    rows=prompt_rows,
                    y_key="resume_ttft_avg_ms",
                    y_label="Resume TTFT (ms)",
                    out_path=charts_dir / f"resume_ttft_vs_pressure_{suffix}.svg",
                    include_baseline=True,
                ),
                prompt_rows,
            )
        )
        charts.append(
            (
                "Prefetch Cost vs Cache Pressure",
                render_chart(
                    title=f"Prefetch Cost vs Cache Pressure, prompt_tokens={int(prompt_tokens)}",
                    rows=prompt_rows,
                    y_key="prefetch_ttft_avg_ms",
                    y_label="Prefetch request TTFT (ms)",
                    out_path=charts_dir / f"prefetch_cost_vs_pressure_{suffix}.svg",
                ),
                prompt_rows,
            )
        )

    render_dashboard(charts, charts_dir / "all_charts.html")
    print(f"Wrote SVG charts to {charts_dir}")
    print(f"Wrote combined dashboard to {charts_dir / 'all_charts.html'}")


if __name__ == "__main__":
    main()
