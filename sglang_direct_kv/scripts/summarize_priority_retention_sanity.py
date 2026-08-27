#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CONTROL_MODE = "no_prefetch"
PROTECTED_MODE = "dynamo_priority_hints"


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def parse_int(value: Any) -> int | None:
    val = parse_float(value)
    if val is None:
        return None
    return int(round(val))


def fmt_ms(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) >= 1000:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.1f} ms"


def fmt_num(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def find_gaps_csv(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"gaps CSV not found: {explicit}")
        return explicit
    candidates = [
        root / "controlled_replay_report" / "controlled_replay_gaps.csv",
        root / "report" / "controlled_replay_gaps.csv",
        root / "controlled_replay_gaps.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = sorted(root.glob("**/controlled_replay_gaps.csv"))
    if found:
        return found[0]
    raise SystemExit(f"could not find controlled_replay_gaps.csv under {root}")


def distractor_count(row: dict[str, str]) -> int:
    for key in ("filler_sessions", "fillers", "distractors", "distractor_count"):
        val = parse_int(row.get(key))
        if val is not None:
            return val
    for key in ("case_id", "case_dir", "session_id"):
        text = row.get(key) or ""
        match = re.search(r"(?:^|[_-])f(\d+)(?:$|[_-])", text)
        if match:
            return int(match.group(1))
    return 0


def arm_for_mode(mode: str) -> str:
    if mode == CONTROL_MODE:
        return "control"
    if mode == PROTECTED_MODE:
        return "protected"
    return "other"


def summarize_effect(control: dict[str, Any] | None, protected: dict[str, Any] | None) -> dict[str, Any]:
    if not control or not protected:
        return {
            "protected_replay_ttft_delta_ms": "",
            "protected_cached_prefix_delta_tokens": "",
            "protected_h2d_token_delta": "",
            "effect_status": "missing_control_or_protected",
            "simple_meaning": "Need both control and protected rows for this distractor count.",
        }

    control_ttft = parse_float(control.get("avg_replay_ttft_ms"))
    protected_ttft = parse_float(protected.get("avg_replay_ttft_ms"))
    control_cached = parse_float(control.get("avg_replay_cached_prefix_tokens"))
    protected_cached = parse_float(protected.get("avg_replay_cached_prefix_tokens"))
    control_h2d = parse_float(control.get("avg_replay_h2d_tokens"))
    protected_h2d = parse_float(protected.get("avg_replay_h2d_tokens"))

    ttft_delta = None if control_ttft is None or protected_ttft is None else protected_ttft - control_ttft
    cached_delta = None if control_cached is None or protected_cached is None else protected_cached - control_cached
    h2d_delta = None if control_h2d is None or protected_h2d is None else protected_h2d - control_h2d

    status = "no_clear_retention_benefit"
    meaning = "Protected priority did not clearly improve replay retention in this run."
    if ttft_delta is not None and ttft_delta < -0.05 * max(control_ttft or 1.0, 1.0):
        status = "protected_replay_faster"
        meaning = "Protected priority replay was meaningfully faster than control."
    elif cached_delta is not None and cached_delta > 128:
        status = "protected_reused_more_prefix"
        meaning = "Protected priority preserved or reused more prefix KV than control."
    elif h2d_delta is not None and h2d_delta > 128:
        status = "protected_loaded_more_host_kv"
        meaning = "Protected priority had more host-backed KV available to reload."
    elif ttft_delta is not None and ttft_delta > 0.05 * max(control_ttft or 1.0, 1.0):
        status = "protected_worse"
        meaning = "Protected priority was slower here; priority did not help this retention case."

    return {
        "protected_replay_ttft_delta_ms": "" if ttft_delta is None else round(ttft_delta, 3),
        "protected_cached_prefix_delta_tokens": "" if cached_delta is None else round(cached_delta, 3),
        "protected_h2d_token_delta": "" if h2d_delta is None else round(h2d_delta, 3),
        "effect_status": status,
        "simple_meaning": meaning,
    }


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    head = "".join(f"<th>{html.escape(label)}</th>" for key, label in columns)
    body = []
    for row in shown:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not body:
        body.append(f"<tr><td colspan='{len(columns)}'>No rows</td></tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def make_svg(by_distractor: list[dict[str, Any]], metric: str, title: str, ylabel: str) -> str:
    values = []
    for row in by_distractor:
        for key in (f"control_{metric}", f"protected_{metric}"):
            val = parse_float(row.get(key))
            if val is not None:
                values.append(val)
    if not values:
        return "<p>No chart data available.</p>"
    width = 980
    height = 360
    left = 72
    right = 32
    top = 44
    bottom = 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min = 0.0
    y_max = max(values) * 1.12
    if y_max <= y_min:
        y_max = y_min + 1.0
    xs = [parse_int(row.get("distractors")) or idx for idx, row in enumerate(by_distractor)]
    x_min = min(xs)
    x_max = max(xs)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def x_pos(x: float) -> float:
        return left + ((x - x_min) / (x_max - x_min)) * plot_w

    def y_pos(y: float) -> float:
        return top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_h

    def point_path(prefix: str) -> str:
        pts = []
        for row in by_distractor:
            x = parse_float(row.get("distractors"))
            y = parse_float(row.get(f"{prefix}_{metric}"))
            if x is None or y is None:
                continue
            pts.append(f"{x_pos(x):.1f},{y_pos(y):.1f}")
        return " ".join(pts)

    control_points = point_path("control")
    protected_points = point_path("protected")
    circles = []
    for row in by_distractor:
        x = parse_float(row.get("distractors"))
        if x is None:
            continue
        cx = x_pos(x)
        for prefix, color in (("control", "#2563eb"), ("protected", "#f59e0b")):
            y = parse_float(row.get(f"{prefix}_{metric}"))
            if y is None:
                continue
            circles.append(
                f"<circle cx='{cx:.1f}' cy='{y_pos(y):.1f}' r='6' fill='{color}' />"
                f"<title>{prefix}: {fmt_num(y)}</title>"
            )
        circles.append(
            f"<text x='{cx:.1f}' y='{height - 22}' text-anchor='middle' class='axis'>{int(x)}</text>"
        )

    y_ticks = [0, y_max / 4, y_max / 2, 3 * y_max / 4, y_max]
    grid = []
    for y in y_ticks:
        yy = y_pos(y)
        grid.append(f"<line x1='{left}' x2='{width-right}' y1='{yy:.1f}' y2='{yy:.1f}' class='grid' />")
        grid.append(f"<text x='{left-10}' y='{yy+4:.1f}' text-anchor='end' class='axis'>{fmt_num(y)}</text>")

    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <style>
    .axis {{ font: 13px system-ui, -apple-system, Segoe UI, sans-serif; fill: #334155; }}
    .title {{ font: 700 16px system-ui, -apple-system, Segoe UI, sans-serif; fill: #0f172a; }}
    .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    .line-control {{ fill: none; stroke: #2563eb; stroke-width: 3; }}
    .line-protected {{ fill: none; stroke: #f59e0b; stroke-width: 3; }}
  </style>
  <text x='{left}' y='24' class='title'>{html.escape(title)}</text>
  {''.join(grid)}
  <line x1='{left}' x2='{left}' y1='{top}' y2='{height-bottom}' stroke='#94a3b8' />
  <line x1='{left}' x2='{width-right}' y1='{height-bottom}' y2='{height-bottom}' stroke='#94a3b8' />
  <text x='{width/2:.1f}' y='{height-4}' text-anchor='middle' class='axis'>distractor requests</text>
  <text transform='translate(20 {height/2:.1f}) rotate(-90)' text-anchor='middle' class='axis'>{html.escape(ylabel)}</text>
  <polyline points='{control_points}' class='line-control' />
  <polyline points='{protected_points}' class='line-protected' />
  {''.join(circles)}
  <rect x='{left+12}' y='{top+10}' width='14' height='14' rx='3' fill='#2563eb'/><text x='{left+34}' y='{top+22}' class='axis'>control: no priority</text>
  <rect x='{left+180}' y='{top+10}' width='14' height='14' rx='3' fill='#f59e0b'/><text x='{left+202}' y='{top+22}' class='axis'>protected: Dynamo priority hints</text>
</svg>
"""


def build_html(
    out_path: Path,
    root: Path,
    gaps_csv: Path,
    summary_rows: list[dict[str, Any]],
    by_distractor_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
) -> None:
    summary_cols = [
        ("distractors", "distractors"),
        ("control_avg_replay_ttft_ms", "control replay TTFT"),
        ("protected_avg_replay_ttft_ms", "protected replay TTFT"),
        ("protected_replay_ttft_delta_ms", "protected delta"),
        ("control_avg_replay_cached_prefix_tokens", "control cached prefix"),
        ("protected_avg_replay_cached_prefix_tokens", "protected cached prefix"),
        ("protected_cached_prefix_delta_tokens", "cached prefix delta"),
        ("effect_status", "effect"),
        ("simple_meaning", "simple meaning"),
    ]
    detail_cols = [
        ("arm", "arm"),
        ("distractors", "distractors"),
        ("mode", "mode"),
        ("session_id", "session"),
        ("first_latency_ms", "first latency ms"),
        ("replay_ttft_ms", "replay TTFT ms"),
        ("replay_cached_prefix_tokens", "cached prefix tok"),
        ("replay_h2d_tokens", "replay H2D tok"),
        ("recomputed_tokens_est", "recomputed tok"),
        ("lifecycle_verdict", "lifecycle verdict"),
        ("final_path", "final path"),
    ]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dynamo Priority KV Retention Sanity</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #0f172a; background: #f8fafc; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    section {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin: 18px 0; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ color: #334155; line-height: 1.55; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 8px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
    th {{ background: #f1f5f9; font-weight: 700; }}
    .flow {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; align-items: center; }}
    .box {{ background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 8px; padding: 12px; text-align: center; font-weight: 700; }}
    .arrow {{ text-align: center; color: #64748b; font-weight: 700; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }}
    .card b {{ display: block; font-size: 24px; margin-top: 6px; }}
  </style>
</head>
<body>
<main>
  <h1>Dynamo Priority KV Retention Sanity</h1>
  <p>This milestone checks one narrow question: when the same important request is replayed after distractor pressure, does Dynamo-style priority help its KV survive or reload better than the no-priority control?</p>

  <section>
    <h2>Experiment Shape</h2>
    <div class="flow">
      <div class="box">A_first</div><div class="arrow">-></div>
      <div class="box">many distractors</div><div class="arrow">-></div>
      <div class="box">A_replay</div>
    </div>
    <p><b>Control arm:</b> A_first and A_replay use normal/no priority. <b>Protected arm:</b> A_first and A_replay carry Dynamo-style high priority; distractors carry low priority.</p>
    <p>This is not a deadline-prefetch test. It is a retention sanity test: can priority preserve or recover important KV under cache pressure?</p>
  </section>

  <section>
    <h2>At A Glance</h2>
    <div class="cards">
      <div class="card">Distractor settings<b>{len(by_distractor_rows)}</b></div>
      <div class="card">Detailed rows<b>{len(detail_rows)}</b></div>
      <div class="card">Source CSV<b>{html.escape(gaps_csv.name)}</b></div>
    </div>
  </section>

  <section>
    <h2>Replay TTFT vs Distractors</h2>
    <p>Lower is better. If priority protects KV, the protected line should stay flatter or lower as distractors increase.</p>
    {make_svg(by_distractor_rows, "avg_replay_ttft_ms", "Replay TTFT under distractor pressure", "replay TTFT ms")}
  </section>

  <section>
    <h2>Cached Prefix Tokens vs Distractors</h2>
    <p>Higher means more of the replay prefix was already reusable.</p>
    {make_svg(by_distractor_rows, "avg_replay_cached_prefix_tokens", "Cached prefix reuse under distractor pressure", "cached prefix tokens")}
  </section>

  <section>
    <h2>Retention Summary</h2>
    {html_table(summary_rows, summary_cols)}
  </section>

  <section>
    <h2>Detailed Evidence Rows</h2>
    {html_table(detail_rows, detail_cols)}
  </section>

  <section>
    <h2>Inputs</h2>
    <p>Root: <code>{html.escape(str(root))}</code></p>
    <p>Gaps CSV: <code>{html.escape(str(gaps_csv))}</code></p>
  </section>
</main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Dynamo priority KV retention sanity results.")
    parser.add_argument("--root", type=Path, required=True, help="Milestone 40/Milestone 27 result root.")
    parser.add_argument("--gaps-csv", type=Path, default=None, help="Optional controlled_replay_gaps.csv path.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--latest-root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    gaps_csv = find_gaps_csv(root, args.gaps_csv)
    out_dir = (args.out_dir or (root / "priority_retention_report")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = csv_rows(gaps_csv)
    detail_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        mode = row.get("mode", "")
        if mode not in {CONTROL_MODE, PROTECTED_MODE}:
            continue
        arm = arm_for_mode(mode)
        replay_h2d_tokens = (
            parse_float(row.get("lifecycle_replay_h2d_tokens"))
            or parse_float(row.get("replay_host_load_tokens"))
            or 0.0
        )
        detail_rows.append(
            {
                "arm": arm,
                "distractors": distractor_count(row),
                "mode": mode,
                "session_id": row.get("session_id", ""),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "first_latency_ms": fmt_num(parse_float(row.get("current_latency_ms"))),
                "replay_latency_ms": fmt_num(parse_float(row.get("resume_latency_ms"))),
                "replay_ttft_ms": fmt_num(parse_float(row.get("resume_ttft_ms"))),
                "replay_cached_prefix_tokens": fmt_num(parse_float(row.get("replay_cached_prefix_tokens"))),
                "replay_initial_cached_prefix_tokens": fmt_num(parse_float(row.get("replay_initial_cached_prefix_tokens"))),
                "replay_final_cached_prefix_tokens": fmt_num(parse_float(row.get("replay_final_cached_prefix_tokens"))),
                "replay_cache_hit_ratio_pct": fmt_num(parse_float(row.get("replay_cache_hit_ratio_pct"))),
                "replay_h2d_tokens": fmt_num(replay_h2d_tokens),
                "replay_h2d_events": fmt_num(parse_float(row.get("replay_kv_h2d_events"))),
                "recomputed_tokens_est": fmt_num(parse_float(row.get("recomputed_tokens_est"))),
                "lifecycle_gpu_evict_tokens": fmt_num(parse_float(row.get("lifecycle_gpu_evict_tokens"))),
                "lifecycle_host_evict_tokens": fmt_num(parse_float(row.get("lifecycle_host_evict_tokens"))),
                "lifecycle_verdict": row.get("lifecycle_verdict", ""),
                "final_path": row.get("final_path", ""),
            }
        )

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(int(row["distractors"]), row["arm"])].append(row)

    arm_rows: dict[tuple[int, str], dict[str, Any]] = {}
    for (distractors, arm), rows in sorted(grouped.items()):
        arm_rows[(distractors, arm)] = {
            "distractors": distractors,
            "arm": arm,
            "mode": rows[0].get("mode", ""),
            "rows": len(rows),
            "avg_first_latency_ms": mean([parse_float(r.get("first_latency_ms")) for r in rows]),
            "avg_replay_ttft_ms": mean([parse_float(r.get("replay_ttft_ms")) for r in rows]),
            "avg_replay_latency_ms": mean([parse_float(r.get("replay_latency_ms")) for r in rows]),
            "avg_replay_cached_prefix_tokens": mean([parse_float(r.get("replay_cached_prefix_tokens")) for r in rows]),
            "avg_replay_h2d_tokens": mean([parse_float(r.get("replay_h2d_tokens")) for r in rows]),
            "avg_recomputed_tokens_est": mean([parse_float(r.get("recomputed_tokens_est")) for r in rows]),
        }

    by_distractor_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for distractors in sorted({int(r["distractors"]) for r in detail_rows}):
        control = arm_rows.get((distractors, "control"))
        protected = arm_rows.get((distractors, "protected"))
        effect = summarize_effect(control, protected)
        row: dict[str, Any] = {"distractors": distractors}
        for prefix, arm_row in (("control", control), ("protected", protected)):
            for metric in (
                "rows",
                "avg_first_latency_ms",
                "avg_replay_ttft_ms",
                "avg_replay_latency_ms",
                "avg_replay_cached_prefix_tokens",
                "avg_replay_h2d_tokens",
                "avg_recomputed_tokens_est",
            ):
                value = None if arm_row is None else arm_row.get(metric)
                row[f"{prefix}_{metric}"] = "" if value is None else round(value, 3) if isinstance(value, float) else value
        row.update(effect)
        by_distractor_rows.append(row)
        summary_rows.append(row)

    detail_fields = [
        "arm",
        "distractors",
        "mode",
        "session_id",
        "tool_wait_ms",
        "first_latency_ms",
        "replay_latency_ms",
        "replay_ttft_ms",
        "replay_cached_prefix_tokens",
        "replay_initial_cached_prefix_tokens",
        "replay_final_cached_prefix_tokens",
        "replay_cache_hit_ratio_pct",
        "replay_h2d_tokens",
        "replay_h2d_events",
        "recomputed_tokens_est",
        "lifecycle_gpu_evict_tokens",
        "lifecycle_host_evict_tokens",
        "lifecycle_verdict",
        "final_path",
    ]
    summary_fields = [
        "distractors",
        "control_rows",
        "protected_rows",
        "control_avg_first_latency_ms",
        "protected_avg_first_latency_ms",
        "control_avg_replay_ttft_ms",
        "protected_avg_replay_ttft_ms",
        "protected_replay_ttft_delta_ms",
        "control_avg_replay_cached_prefix_tokens",
        "protected_avg_replay_cached_prefix_tokens",
        "protected_cached_prefix_delta_tokens",
        "control_avg_replay_h2d_tokens",
        "protected_avg_replay_h2d_tokens",
        "protected_h2d_token_delta",
        "effect_status",
        "simple_meaning",
    ]
    write_csv(out_dir / "priority_retention_detail.csv", detail_rows, detail_fields)
    write_csv(out_dir / "priority_retention_summary.csv", summary_rows, summary_fields)
    (out_dir / "priority_retention_summary.json").write_text(
        json.dumps({"summary": summary_rows, "detail": detail_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    build_html(
        out_dir / "priority_retention_report.html",
        root=root,
        gaps_csv=gaps_csv,
        summary_rows=summary_rows,
        by_distractor_rows=by_distractor_rows,
        detail_rows=detail_rows,
    )

    if args.latest_root is not None:
        latest_root = args.latest_root.resolve()
        latest_root.mkdir(parents=True, exist_ok=True)
        (latest_root / "latest_priority_retention_sanity_report.html").write_text(
            (out_dir / "priority_retention_report.html").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    print(f"Wrote priority retention report to {out_dir / 'priority_retention_report.html'}")
    print(f"Wrote summary CSV to {out_dir / 'priority_retention_summary.csv'}")


if __name__ == "__main__":
    main()
