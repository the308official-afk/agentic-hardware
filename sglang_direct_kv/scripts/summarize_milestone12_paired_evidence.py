#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
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
    return [
        {
            "profiled_sessions": len(rows),
            "sessions_with_visible_h2d_copy": sum(1 for row in rows if as_int(row, "torch_h2d_copy_events") > 0),
            "cuda_copy_ready_before_replay": sum(1 for row in rows if yes(row, "cuda_copy_ready_before_replay")),
            "full_hint_done_before_replay": sum(1 for row in rows if yes(row, "full_hint_done_before_replay")),
            "replay_reloaded_kv": sum(1 for row in rows if yes(row, "replay_reloaded_kv")),
            "clean_success": sum(1 for row in rows if row.get("checkpoint_result") == "clean_success"),
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
        "Use the clean run for TTFT/performance claims.",
        "Use the profiled run for CUDA HtoD/KV mechanism claims.",
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
            out.append(f"<td{cls}>{html.escape(value)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def write_html(path: Path, sections: dict[str, list[dict[str, Any]]], metadata: dict[str, Any]) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Milestone 12 Paired Evidence Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:28px;background:#f8fafc;color:#111827}",
        ".panel{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin:16px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border-bottom:1px solid #e5e7eb;padding:7px 8px;text-align:left;white-space:nowrap}",
        "th{background:#f3f4f6;font-weight:700}",
        ".caption{color:#374151;line-height:1.45}",
        ".good{color:#166534;font-weight:700}",
        ".bad{color:#b91c1c;font-weight:700}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Milestone 12 Paired Evidence Report</h1>",
        '<div class="panel"><p class="caption">Clean runs answer performance questions. Profiled runs answer mechanism questions. Do not use profiled TTFT values for performance claims.</p></div>',
        '<div class="panel"><h2>Metadata</h2><pre>',
        html.escape(json.dumps(metadata, indent=2, sort_keys=True)),
        "</pre></div>",
    ]
    for title, rows in sections.items():
        lines.append(f'<div class="panel"><h2>{html.escape(title)}</h2>')
        lines.append(html_table(rows))
        lines.append("</div>")
    lines.extend(["</body>", "</html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Join clean performance and profiled attribution runs.")
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--attribution-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--modes", default="no_prefetch direct_load oracle_direct_load")
    parser.add_argument("--attribution-mode", default="oracle_direct_load")
    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    attribution_root = Path(args.attribution_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    modes = [mode for mode in args.modes.split() if mode]

    clean_rows = load_clean_rows(clean_root, modes)
    attribution_rows = read_csv(attribution_root / f"{args.attribution_mode}_agentic_prefetch_timeline.csv")

    sections = {
        "Clean Performance Summary": build_clean_summary(clean_rows),
        "Profiled Attribution Summary": build_attribution_summary(attribution_rows),
        "Paired Session Evidence": build_paired_rows(clean_rows, attribution_rows, args.attribution_mode),
    }
    metadata = {
        "clean_root": str(clean_root),
        "attribution_root": str(attribution_root),
        "modes": modes,
        "attribution_mode": args.attribution_mode,
        "note": "Profiled attribution rows should not be used for TTFT performance claims.",
    }

    write_csv(out_root / "paired_clean_summary.csv", sections["Clean Performance Summary"])
    write_csv(out_root / "paired_attribution_summary.csv", sections["Profiled Attribution Summary"])
    write_csv(out_root / "paired_session_evidence.csv", sections["Paired Session Evidence"])
    (out_root / "paired_report.json").write_text(
        json.dumps({"metadata": metadata, "sections": sections}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_md(out_root / "paired_report.md", sections, metadata)
    write_html(out_root / "paired_report.html", sections, metadata)

    print(f"Wrote paired report under {out_root}")
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
