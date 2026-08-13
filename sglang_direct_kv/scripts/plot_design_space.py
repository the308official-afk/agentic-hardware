#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SVG charts for the Milestone 6 design-space sweep.")
    parser.add_argument("--root", default="artifacts/results/milestone6_design_space")
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_rows(root / "summary.csv")
    charts_dir = root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    for prompt_tokens in sorted({row["prompt_tokens"] for row in rows}):
        prompt_rows = [row for row in rows if row["prompt_tokens"] == prompt_tokens]
        suffix = f"p{int(prompt_tokens)}"
        render_chart(
            title=f"Hint-Aware Benefit vs Cache Pressure, prompt_tokens={int(prompt_tokens)}",
            rows=prompt_rows,
            y_key="benefit_vs_no_prefetch_ms",
            y_label="Benefit vs no_prefetch (ms)",
            out_path=charts_dir / f"benefit_vs_pressure_{suffix}.svg",
        )
        render_chart(
            title=f"Resume TTFT vs Cache Pressure, prompt_tokens={int(prompt_tokens)}",
            rows=prompt_rows,
            y_key="resume_ttft_avg_ms",
            y_label="Resume TTFT (ms)",
            out_path=charts_dir / f"resume_ttft_vs_pressure_{suffix}.svg",
            include_baseline=True,
        )
        render_chart(
            title=f"Prefetch Cost vs Cache Pressure, prompt_tokens={int(prompt_tokens)}",
            rows=prompt_rows,
            y_key="prefetch_ttft_avg_ms",
            y_label="Prefetch request TTFT (ms)",
            out_path=charts_dir / f"prefetch_cost_vs_pressure_{suffix}.svg",
        )

    print(f"Wrote SVG charts to {charts_dir}")


if __name__ == "__main__":
    main()
