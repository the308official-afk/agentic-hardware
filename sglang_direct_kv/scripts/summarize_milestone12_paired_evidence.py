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


def write_html(path: Path, sections: dict[str, list[dict[str, Any]]], metadata: dict[str, Any]) -> None:
    cards = summary_cards(sections)
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
    for title, rows in sections.items():
        lines.append(f'<div class="panel"><h2>{html.escape(title)}</h2>')
        lines.append(f'<p class="caption">{html.escape(section_caption(title))}</p>')
        lines.append('<div class="table-wrap">')
        lines.append(html_table(rows))
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
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "modes": modes,
        "attribution_mode": args.attribution_mode,
        "latest_root": args.latest_root,
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
