#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_MODES = ["no_prefetch", "request_warm", "direct_load", "oracle_direct_load"]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        return 0.0
    return float(raw)


def as_int(row: dict[str, Any], key: str) -> int:
    raw = row.get(key, "")
    if raw == "":
        return 0
    return int(float(raw))


def fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def outcome_counts(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("outcome", "unknown")) for row in rows)
    return ", ".join(f"{name}: {count}" for name, count in counts.most_common())


def load_mode_rows(root: Path, modes: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        path = root / f"{mode}_outcomes" / "hint_outcomes.csv"
        rows = read_csv(path)
        if rows:
            result[mode] = rows
    return result


def build_mode_summary(mode_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    baseline_avg = 0.0
    baseline_rows = mode_rows.get("no_prefetch", [])
    if baseline_rows:
        baseline_avg = mean(as_float(row, "replay_ttft_ms") for row in baseline_rows)

    summary: list[dict[str, Any]] = []
    for mode, rows in mode_rows.items():
        replay_values = [as_float(row, "replay_ttft_ms") for row in rows]
        improvement = baseline_avg - mean(replay_values) if baseline_avg and replay_values else 0.0
        summary.append(
            {
                "mode": mode,
                "sessions": len(rows),
                "avg_replay_ttft_ms": round(mean(replay_values), 3) if replay_values else 0.0,
                "median_replay_ttft_ms": round(median(replay_values), 3) if replay_values else 0.0,
                "min_replay_ttft_ms": round(min(replay_values), 3) if replay_values else 0.0,
                "max_replay_ttft_ms": round(max(replay_values), 3) if replay_values else 0.0,
                "avg_improvement_vs_no_prefetch_ms": round(improvement, 3),
                "avg_improvement_vs_no_prefetch_pct": (
                    round(improvement * 100.0 / baseline_avg, 2) if baseline_avg else 0.0
                ),
                "total_prefetch_load_count": sum(as_int(row, "prefetch_load_count") for row in rows),
                "total_resume_load_count": sum(as_int(row, "resume_load_count") for row in rows),
                "total_eviction_pressure_after_prefetch": sum(
                    as_int(row, "eviction_pressure_after_prefetch") for row in rows
                ),
                "outcomes": outcome_counts(rows),
            }
        )
    return summary


def build_tool_wait_summary(mode_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for mode, rows in mode_rows.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[as_int(row, "tool_wait_ms")].append(row)
        for tool_wait_ms, group in sorted(grouped.items()):
            replay_values = [as_float(row, "replay_ttft_ms") for row in group]
            rows_out.append(
                {
                    "mode": mode,
                    "tool_wait_ms": tool_wait_ms,
                    "sessions": len(group),
                    "avg_replay_ttft_ms": round(mean(replay_values), 3) if replay_values else 0.0,
                    "avg_prefetch_load_count": round(mean(as_int(row, "prefetch_load_count") for row in group), 3),
                    "avg_resume_load_count": round(mean(as_int(row, "resume_load_count") for row in group), 3),
                    "outcomes": outcome_counts(group),
                }
            )
    return rows_out


def build_failure_rows(mode_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    good = {"no_hint", "no_prefetch_needed", "useful_prefetch"}
    rows_out: list[dict[str, Any]] = []
    for mode, rows in mode_rows.items():
        for row in rows:
            if row.get("outcome") in good:
                continue
            rows_out.append(
                {
                    "mode": mode,
                    "session_id": row.get("session_id", ""),
                    "tool_wait_ms": as_int(row, "tool_wait_ms"),
                    "prompt_tokens": as_int(row, "prompt_tokens"),
                    "replay_ttft_ms": as_float(row, "replay_ttft_ms"),
                    "prefetch_load_count": as_int(row, "prefetch_load_count"),
                    "resume_load_count": as_int(row, "resume_load_count"),
                    "eviction_pressure_after_prefetch": as_int(row, "eviction_pressure_after_prefetch"),
                    "hint_end_ms_before_replay": row.get("hint_end_ms_before_replay", ""),
                    "outcome": row.get("outcome", ""),
                }
            )
    return rows_out


def build_session_delta_rows(mode_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    baseline = {
        row["session_id"]: row for row in mode_rows.get("no_prefetch", []) if row.get("session_id")
    }
    rows_out: list[dict[str, Any]] = []
    for session_id, base in sorted(baseline.items()):
        out: dict[str, Any] = {
            "session_id": session_id,
            "tool_wait_ms": as_int(base, "tool_wait_ms"),
            "prompt_tokens": as_int(base, "prompt_tokens"),
            "no_prefetch_replay_ttft_ms": as_float(base, "replay_ttft_ms"),
        }
        for mode in [mode for mode in DEFAULT_MODES if mode != "no_prefetch"]:
            row = next((item for item in mode_rows.get(mode, []) if item.get("session_id") == session_id), None)
            if row is None:
                out[f"{mode}_replay_ttft_ms"] = ""
                out[f"{mode}_delta_ms"] = ""
                out[f"{mode}_outcome"] = ""
                continue
            value = as_float(row, "replay_ttft_ms")
            out[f"{mode}_replay_ttft_ms"] = value
            out[f"{mode}_delta_ms"] = round(as_float(base, "replay_ttft_ms") - value, 3)
            out[f"{mode}_outcome"] = row.get("outcome", "")
        rows_out.append(out)
    return rows_out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
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


def write_md(path: Path, sections: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Milestone 9 Agentic Traffic Summary", ""]
    for title, rows in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(md_table(rows))
    path.write_text("\n".join(lines), encoding="utf-8")


def html_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    headers = list(rows[0].keys())
    lines = ["<table>", "<thead>", "<tr>"]
    for header in headers:
        lines.append(f"<th>{html.escape(header)}</th>")
    lines.extend(["</tr>", "</thead>", "<tbody>"])
    for row in rows:
        lines.append("<tr>")
        for header in headers:
            lines.append(f"<td>{html.escape(str(row.get(header, '')))}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def write_html(path: Path, sections: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        "  <title>Milestone 9 Agentic Traffic Summary</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 32px; color: #111827; background: #f9fafb; }",
        "    h1 { margin: 0 0 18px; }",
        "    section { margin: 28px 0 42px; }",
        "    table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }",
        "    th, td { border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: right; white-space: nowrap; }",
        "    th { background: #f3f4f6; }",
        "    th:first-child, td:first-child, th:last-child, td:last-child { text-align: left; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Milestone 9 Agentic Traffic Summary</h1>",
    ]
    for title, rows in sections.items():
        lines.append("  <section>")
        lines.append(f"    <h2>{html.escape(title)}</h2>")
        lines.append(html_table(rows))
        lines.append("  </section>")
    lines.extend(["</body>", "</html>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Milestone 9 agentic traffic results across modes.")
    parser.add_argument("--root", default="artifacts/results/milestone9_agentic_traffic")
    parser.add_argument("--modes", default=" ".join(DEFAULT_MODES))
    args = parser.parse_args()

    root = Path(args.root)
    modes = [mode for mode in args.modes.split() if mode]
    mode_rows = load_mode_rows(root, modes)
    if not mode_rows:
        print(f"No hint_outcomes.csv files found under {root}")
        return

    mode_summary = build_mode_summary(mode_rows)
    tool_wait_summary = build_tool_wait_summary(mode_rows)
    failure_rows = build_failure_rows(mode_rows)
    session_delta_rows = build_session_delta_rows(mode_rows)
    sections = {
        "Mode Summary": mode_summary,
        "Tool Wait Breakdown": tool_wait_summary,
        "Failure Mode Rows": failure_rows,
        "Per-Session Delta vs Baseline": session_delta_rows,
    }

    json_path = root / "traffic_summary.json"
    csv_path = root / "traffic_summary.csv"
    md_path = root / "traffic_summary.md"
    html_path = root / "traffic_summary.html"
    json_path.write_text(json.dumps(sections, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, mode_summary)
    write_md(md_path, sections)
    write_html(html_path, sections)

    print(f"Wrote summary to {json_path}")
    print(f"Wrote summary to {csv_path}")
    print(f"Wrote summary to {md_path}")
    print(f"Wrote summary to {html_path}")
    print()
    for row in mode_summary:
        print(
            f"{row['mode']}: avg_replay_ttft_ms={fmt_ms(row['avg_replay_ttft_ms'])}, "
            f"improvement_ms={fmt_ms(row['avg_improvement_vs_no_prefetch_ms'])}, "
            f"outcomes={row['outcomes']}"
        )


if __name__ == "__main__":
    main()
