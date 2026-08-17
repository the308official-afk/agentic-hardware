#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from build_live_agentbench_tool_gap_report import (
    augment_gaps_with_prefetch,
    build_timeline_svg,
    build_tool_gaps,
    is_preflight_request,
    maybe_float,
    normalize_requests,
    read_csv,
    read_jsonl,
    table_html,
    write_csv,
)


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def short(value: Any, limit: int = 120) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def avg(values: list[float]) -> float | str:
    return round(mean(values), 3) if values else ""


def med(values: list[float]) -> float | str:
    return round(median(values), 3) if values else ""


def pct(delta: float | None, baseline: float | None) -> float | str:
    if delta is None or baseline in (None, 0):
        return ""
    return round(delta * 100.0 / baseline, 2)


def task_index_map(task_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run_id") or ""): row for row in task_rows if row.get("run_id")}


def add_task_and_pair_fields(
    requests: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    mode: str,
) -> None:
    by_run = task_index_map(task_rows)
    for request in requests:
        run_id = str(request.get("parent_run_id") or "")
        task = by_run.get(run_id, {})
        request["mode"] = mode
        request["task_index"] = task.get("task_index", "")
        request["task_status"] = task.get("status", "")

    per_task_counter: defaultdict[str, int] = defaultdict(int)
    for gap in sorted(gaps, key=lambda row: float(row.get("current_start_ms") or 0.0)):
        run_id = str(gap.get("parent_run_id") or "")
        task = by_run.get(run_id, {})
        task_index = str(task.get("task_index", ""))
        task_key = task_index if task_index else run_id
        gap_order = per_task_counter[task_key]
        per_task_counter[task_key] += 1
        gap["mode"] = mode
        gap["task_index"] = task_index
        gap["task_status"] = task.get("status", "")
        gap["gap_order_in_task"] = gap_order
        gap["pair_key"] = f"task_{task_key}:gap_{gap_order}"


def load_live_run(root: Path, mode: str, include_preflight: bool) -> dict[str, Any]:
    proxy_jsonl = root / "tool_normalizer_proxy.jsonl"
    task_index_csv = root / "exp6_direct_sglang_task_index.csv"
    hint_log = root / "live_hint_events.jsonl"
    controller_log = root / "live_prefetch_controller.jsonl"

    raw_rows = read_jsonl(proxy_jsonl)
    hint_rows = read_jsonl(hint_log)
    controller_rows = read_jsonl(controller_log)
    task_rows = read_csv(task_index_csv)
    all_requests = normalize_requests(raw_rows)
    if include_preflight:
        requests = all_requests
        excluded_preflight = 0
    else:
        requests = [row for row in all_requests if not is_preflight_request(row)]
        excluded_preflight = len(all_requests) - len(requests)

    gaps = build_tool_gaps(requests)
    base_ts = min((float(row["start_ts"]) for row in all_requests), default=0.0)
    gaps = augment_gaps_with_prefetch(gaps, hint_rows, controller_rows, base_ts)
    add_task_and_pair_fields(requests, gaps, task_rows, mode)

    return {
        "mode": mode,
        "root": str(root),
        "proxy_jsonl": str(proxy_jsonl),
        "task_index_csv": str(task_index_csv),
        "hint_log": str(hint_log) if hint_log.exists() else "",
        "controller_log": str(controller_log) if controller_log.exists() else "",
        "raw_request_count": len(raw_rows),
        "captured_model_requests": len(all_requests),
        "excluded_preflight_requests": excluded_preflight,
        "requests": requests,
        "gaps": gaps,
        "task_rows": task_rows,
        "hint_rows": hint_rows,
        "controller_rows": controller_rows,
    }


def mode_summary(run: dict[str, Any]) -> dict[str, Any]:
    requests = run["requests"]
    gaps = run["gaps"]
    hint_rows = run["hint_rows"]
    controller_rows = run["controller_rows"]
    gap_values = [float(row.get("tool_gap_ms") or 0.0) for row in gaps]
    resume_values = [float(row.get("resume_latency_ms") or 0.0) for row in gaps if row.get("resume_latency_ms") not in ("", None)]
    prefetch_attempts = [
        row
        for row in gaps
        if row.get("prefetch_start_ms") not in ("", None) or row.get("prefetch_end_ms") not in ("", None)
    ]
    margins = [float(row["prefetch_margin_ms"]) for row in gaps if row.get("prefetch_margin_ms") not in ("", None)]
    durations = [
        float(row["prefetch_duration_ms"])
        for row in gaps
        if row.get("prefetch_duration_ms") not in ("", None)
    ]
    tool_counter: Counter[str] = Counter()
    for request in requests:
        for name in str(request.get("tool_names") or "").split(","):
            if name:
                tool_counter[name] += 1
    return {
        "mode": run["mode"],
        "tasks_in_index": len(run["task_rows"]),
        "captured_model_requests": run["captured_model_requests"],
        "analyzed_model_requests": len(requests),
        "excluded_preflight_requests": run["excluded_preflight_requests"],
        "requests_with_tools": sum(1 for row in requests if int(row.get("tool_count") or 0) > 0),
        "total_tool_calls": sum(int(row.get("tool_count") or 0) for row in requests),
        "observed_tool_gaps": len(gaps),
        "avg_tool_gap_ms": avg(gap_values),
        "median_tool_gap_ms": med(gap_values),
        "avg_resume_request_latency_ms": avg(resume_values),
        "median_resume_request_latency_ms": med(resume_values),
        "live_hints_submitted": len(hint_rows),
        "controller_events": len(controller_rows),
        "prefetch_attempts_matched_to_gaps": len(prefetch_attempts),
        "prefetch_done_before_resume": sum(1 for row in prefetch_attempts if row.get("prefetch_done_before_resume") == 1),
        "late_prefetch_attempts": sum(1 for row in prefetch_attempts if as_float(row.get("prefetch_margin_ms")) is not None and float(row["prefetch_margin_ms"]) < 0),
        "avg_prefetch_duration_ms": avg(durations),
        "avg_prefetch_margin_ms": avg(margins),
        "top_tools": ", ".join(f"{name}: {count}" for name, count in tool_counter.most_common(8)),
    }


def pair_gaps(no_gaps: list[dict[str, Any]], prefetch_gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    no_by_key = {str(row.get("pair_key") or ""): row for row in no_gaps}
    prefetch_by_key = {str(row.get("pair_key") or ""): row for row in prefetch_gaps}
    keys = sorted(
        set(no_by_key) | set(prefetch_by_key),
        key=lambda key: (
            int(key.split(":")[0].replace("task_", "")) if key.startswith("task_") and key.split(":")[0].replace("task_", "").isdigit() else 10**9,
            int(key.split(":gap_")[-1]) if ":gap_" in key and key.split(":gap_")[-1].isdigit() else 10**9,
            key,
        ),
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        base = no_by_key.get(key, {})
        pref = prefetch_by_key.get(key, {})
        no_latency = as_float(base.get("resume_latency_ms"))
        pref_latency = as_float(pref.get("resume_latency_ms"))
        delta = round(no_latency - pref_latency, 3) if no_latency is not None and pref_latency is not None else None
        margin = as_float(pref.get("prefetch_margin_ms"))
        if delta is None:
            perf = "unpaired"
        elif delta > 0:
            perf = "prefetch run faster"
        elif delta < 0:
            perf = "prefetch run slower"
        else:
            perf = "same"
        if margin is None:
            prefetch_verdict = "no matched hint"
        elif margin >= 0:
            prefetch_verdict = "prefetch done before resume"
        else:
            prefetch_verdict = "prefetch late"
        rows.append(
            {
                "pair_key": key,
                "task_index": pref.get("task_index") or base.get("task_index") or "",
                "gap_order_in_task": pref.get("gap_order_in_task") if pref else base.get("gap_order_in_task", ""),
                "tool_names": pref.get("tool_names") or base.get("tool_names") or "",
                "phases": f"{pref.get('from_phase') or base.get('from_phase') or ''} -> {pref.get('to_phase') or base.get('to_phase') or ''}",
                "no_prefetch_tool_gap_ms": base.get("tool_gap_ms", ""),
                "prefetch_tool_gap_ms": pref.get("tool_gap_ms", ""),
                "no_prefetch_resume_request_latency_ms": base.get("resume_latency_ms", ""),
                "prefetch_resume_request_latency_ms": pref.get("resume_latency_ms", ""),
                "prefetch_gain_ms": delta if delta is not None else "",
                "prefetch_gain_pct": pct(delta, no_latency),
                "performance_verdict": perf,
                "prefetch_status": pref.get("prefetch_status", ""),
                "prefetch_duration_ms": pref.get("prefetch_duration_ms", ""),
                "prefetch_margin_ms": pref.get("prefetch_margin_ms", ""),
                "prefetch_verdict": prefetch_verdict,
                "prefetch_resume_overlap_ms": pref.get("prefetch_resume_overlap_ms", ""),
                "pairing_note": "paired by same SWE-bench task index and same gap order; live runs can still diverge",
                "no_prefetch_preview": short(base.get("resume_preview", "")),
                "prefetch_preview": short(pref.get("resume_preview", "")),
            }
        )
    return rows


def paired_summary(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired = [
        row
        for row in pair_rows
        if row.get("no_prefetch_resume_request_latency_ms") not in ("", None)
        and row.get("prefetch_resume_request_latency_ms") not in ("", None)
    ]
    gains = [float(row["prefetch_gain_ms"]) for row in paired if row.get("prefetch_gain_ms") not in ("", None)]
    no_latencies = [float(row["no_prefetch_resume_request_latency_ms"]) for row in paired]
    pre_latencies = [float(row["prefetch_resume_request_latency_ms"]) for row in paired]
    late = [row for row in pair_rows if row.get("prefetch_verdict") == "prefetch late"]
    ready = [row for row in pair_rows if row.get("prefetch_verdict") == "prefetch done before resume"]
    return [
        {
            "paired_tool_gaps": len(paired),
            "avg_no_prefetch_resume_request_latency_ms": avg(no_latencies),
            "avg_prefetch_resume_request_latency_ms": avg(pre_latencies),
            "avg_prefetch_gain_ms": avg(gains),
            "median_prefetch_gain_ms": med(gains),
            "prefetch_faster_pairs": sum(1 for row in paired if float(row["prefetch_gain_ms"]) > 0),
            "prefetch_slower_pairs": sum(1 for row in paired if float(row["prefetch_gain_ms"]) < 0),
            "prefetch_done_before_resume_pairs": len(ready),
            "prefetch_late_pairs": len(late),
        }
    ]


def key_deductions(mode_rows: list[dict[str, Any]], pair_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode = {row["mode"]: row for row in mode_rows}
    no = by_mode.get("no_prefetch", {})
    pref = by_mode.get("live_prefetch", {})
    paired = pair_summary_rows[0] if pair_summary_rows else {}
    return [
        {
            "finding": "This report uses real live SWE-bench / Deep Agents traffic.",
            "evidence": f"no-prefetch analyzed requests={no.get('analyzed_model_requests', '')}; live-prefetch analyzed requests={pref.get('analyzed_model_requests', '')}.",
            "why_it_matters": "The request stream is no longer synthetic; tool calls, tool gaps, and resumes come from the real harness talking to SGLang.",
        },
        {
            "finding": "Tool gaps are real prefetch opportunities.",
            "evidence": f"no-prefetch gaps={no.get('observed_tool_gaps', '')}; live-prefetch gaps={pref.get('observed_tool_gaps', '')}; avg prefetch-run gap={pref.get('avg_tool_gap_ms', '')} ms.",
            "why_it_matters": "These gaps are where a runtime would issue KV residency hints after a tool call.",
        },
        {
            "finding": "Best-effort software prefetch can miss short live windows.",
            "evidence": f"live-prefetch matched attempts={pref.get('prefetch_attempts_matched_to_gaps', '')}; late attempts={pref.get('late_prefetch_attempts', '')}; avg prefetch duration={pref.get('avg_prefetch_duration_ms', '')} ms.",
            "why_it_matters": "If the controller/SGLang path takes longer than the tool gap, the next agent turn still arrives before the prefetch path finishes.",
        },
        {
            "finding": "Prefetch can help or hurt depending on timing and interference.",
            "evidence": f"paired gaps={paired.get('paired_tool_gaps', '')}; faster pairs={paired.get('prefetch_faster_pairs', '')}; slower pairs={paired.get('prefetch_slower_pairs', '')}; avg gain={paired.get('avg_prefetch_gain_ms', '')} ms.",
            "why_it_matters": "This supports the hardware story: policy hints alone are not enough; movement needs deadline-aware scheduling, protection, and telemetry.",
        },
        {
            "finding": "Live pairing is useful, but not perfectly deterministic.",
            "evidence": "Pairs are matched by SWE-bench task index plus gap order inside that task.",
            "why_it_matters": "The two live agent runs can choose slightly different actions, so exact per-gap comparisons should be read alongside aggregate trends.",
        },
    ]


def timeline_summary(gaps: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps[:limit]:
        margin = as_float(row.get("prefetch_margin_ms"))
        overlap = as_float(row.get("prefetch_resume_overlap_ms")) or 0.0
        if margin is None:
            status = "no matched hint"
        elif margin >= 0:
            status = "prefetch finished before resume"
        else:
            status = "prefetch late"
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "task_index": row.get("task_index", ""),
                "gap_order_in_task": row.get("gap_order_in_task", ""),
                "tools": row.get("tool_names", ""),
                "tool_gap_ms": row.get("tool_gap_ms", ""),
                "prefetch_status": row.get("prefetch_status", ""),
                "prefetch_duration_ms": row.get("prefetch_duration_ms", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "prefetch_resume_overlap_ms": round(overlap, 3),
                "timeline_verdict": status,
            }
        )
    return rows


def checkpoint_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps:
        margin = as_float(row.get("prefetch_margin_ms"))
        has_hint = row.get("prefetch_status") not in ("", "no_hint", None)
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "task_index": row.get("task_index", ""),
                "hint_was_submitted": "yes" if row.get("hint_id") else "no",
                "controller_started_prefetch": "yes" if row.get("prefetch_start_ms") not in ("", None) else "no",
                "controller_finished_prefetch": "yes" if row.get("prefetch_end_ms") not in ("", None) else "no",
                "prefetch_finished_before_resume": "yes" if margin is not None and margin >= 0 else "no" if has_hint else "",
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "resume_request_latency_ms": row.get("resume_latency_ms", ""),
            }
        )
    return rows


def session_observations(gaps: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gaps[:limit]:
        margin = as_float(row.get("prefetch_margin_ms"))
        if margin is None:
            status = "No matched hint"
            what = "This live gap did not have a matched controller prefetch attempt."
            deduction = "Good for observing the tool gap, but not an intervention example."
        elif margin >= 0:
            status = "Useful prefetch timing"
            what = f"The live software prefetch request completed {margin:.3f} ms before the real resume request."
            deduction = "The hint path met this deadline, so this is a success timing example."
        else:
            status = "Late prefetch"
            what = f"The live software prefetch request completed {abs(margin):.3f} ms after the real resume request started."
            deduction = "The runtime had the semantic hint, but the ordinary software/SGLang path did not finish in time."
        rows.append(
            {
                "session_id": row.get("session_id", ""),
                "status": status,
                "what_happened": what,
                "deduction_and_evidence": (
                    f"{deduction} tool gap={row.get('tool_gap_ms')} ms; "
                    f"prefetch duration={row.get('prefetch_duration_ms')} ms; "
                    f"resume request latency={row.get('resume_latency_ms')} ms; "
                    f"tools={row.get('tool_names')}."
                ),
            }
        )
    return rows


def request_details(run: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run["requests"][:limit]:
        rows.append(
            {
                "mode": run["mode"],
                "task_index": row.get("task_index", ""),
                "parent_run_id": row.get("parent_run_id", ""),
                "phase": row.get("phase", ""),
                "message_count": row.get("message_count", ""),
                "tool_count": row.get("tool_count", ""),
                "tool_names": row.get("tool_names", ""),
                "elapsed_ms": row.get("elapsed_ms", ""),
                "context_present": row.get("request_context_present", ""),
                "preview": short(row.get("content_preview", ""), 160),
            }
        )
    return rows


def render_markdown(
    mode_rows: list[dict[str, Any]],
    pair_summary_rows: list[dict[str, Any]],
    deductions: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Live Paired AgentBench Report",
        "",
        "This report compares a live no-prefetch AgentBench/SGLang run against a live prefetch-intervention run.",
        "",
        "## Performance Summary",
        "",
    ]
    for row in mode_rows:
        lines.append(
            f"- `{row['mode']}`: requests={row['analyzed_model_requests']}, tool_calls={row['total_tool_calls']}, "
            f"tool_gaps={row['observed_tool_gaps']}, avg_resume_latency_ms={row['avg_resume_request_latency_ms']}"
        )
    if pair_summary_rows:
        row = pair_summary_rows[0]
        lines.extend(
            [
                "",
                "## Paired Summary",
                "",
                f"- paired_tool_gaps: `{row.get('paired_tool_gaps')}`",
                f"- avg_prefetch_gain_ms: `{row.get('avg_prefetch_gain_ms')}`",
                f"- prefetch_faster_pairs: `{row.get('prefetch_faster_pairs')}`",
                f"- prefetch_slower_pairs: `{row.get('prefetch_slower_pairs')}`",
            ]
        )
    lines.extend(["", "## Key Deductions", ""])
    for row in deductions:
        lines.append(f"- **{row['finding']}** {row['evidence']} {row['why_it_matters']}")
    lines.extend(["", "## Paired Session Evidence", ""])
    lines.append("| pair_key | task | tools | no_prefetch_ms | live_prefetch_ms | gain_ms | prefetch_margin_ms | verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in pair_rows:
        lines.append(
            f"| {row.get('pair_key','')} | {row.get('task_index','')} | {row.get('tool_names','')} | "
            f"{row.get('no_prefetch_resume_request_latency_ms','')} | {row.get('prefetch_resume_request_latency_ms','')} | "
            f"{row.get('prefetch_gain_ms','')} | {row.get('prefetch_margin_ms','')} | {row.get('performance_verdict','')} |"
        )
    return "\n".join(lines) + "\n"


def css() -> str:
    return """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
    main { max-width: 1760px; margin: 0 auto; padding: 24px; }
    section, details.section-card { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px 24px; margin: 18px 0; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
    h1 { font-size: 34px; margin: 0 0 8px; }
    h2 { font-size: 26px; margin: 0 0 12px; }
    h3 { font-size: 18px; margin: 18px 0 8px; }
    p { color: #334155; line-height: 1.45; }
    a { color: #2563eb; text-decoration: none; }
    .note { background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 14px; color: #1e3a8a; }
    .warn { background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; color: #7c2d12; }
    .theme-summary { --theme: #1e3a8a; --theme-bg: #eff6ff; }
    .theme-setup { --theme: #2563eb; --theme-bg: #eff6ff; }
    .theme-guide { --theme: #475569; --theme-bg: #f1f5f9; }
    .theme-clean { --theme: #15803d; --theme-bg: #f0fdf4; }
    .theme-clean-table { --theme: #65a30d; --theme-bg: #f7fee7; }
    .theme-profiled { --theme: #7e22ce; --theme-bg: #faf5ff; }
    .theme-deductions { --theme: #b45309; --theme-bg: #fffbeb; }
    .theme-checkpoints { --theme: #ea580c; --theme-bg: #fff7ed; }
    .theme-observations { --theme: #0f766e; --theme-bg: #f0fdfa; }
    .theme-paired { --theme: #be123c; --theme-bg: #fff1f2; }
    .theme-appendix { --theme: #64748b; --theme-bg: #f8fafc; }
    section[class*="theme-"], details.section-card[class*="theme-"] { border-top: 5px solid var(--theme); }
    section[class*="theme-"] > h2, details.section-card summary h2 { border-left: 8px solid var(--theme); padding-left: 10px; color: var(--theme); }
    details.section-card summary { cursor: pointer; list-style: none; display: flex; align-items: center; gap: 8px; }
    details.section-card summary::-webkit-details-marker { display: none; }
    details.section-card summary h2 { margin: 0; }
    details.section-card summary h2::before { content: "▶"; display: inline-block; color: var(--theme); font-size: 18px; margin-right: 8px; transform: translateY(-1px); }
    details.section-card[open] summary h2::before { content: "▼"; }
    .section-color-legend { color: #475569; font-size: 14px; margin: 8px 0 12px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .timeline-stack { display: grid; grid-template-columns: 1fr; gap: 26px; }
    .timeline-panel { min-width: 0; }
    .setup-diagram { margin: 12px 0 18px; }
    .toc { display: flex; flex-wrap: wrap; gap: 10px 16px; }
    .toc a { background: var(--theme-bg, #f1f5f9); border: 1px solid #e2e8f0; border-left: 7px solid var(--theme, #64748b); border-radius: 6px; padding: 7px 10px; color: #0f172a; font-weight: 650; }
    .toc-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
    .toc-actions button { border: 1px solid #cbd5e1; background: #fff; color: #0f172a; border-radius: 6px; padding: 7px 10px; font-weight: 650; cursor: pointer; }
    .toc-actions button:hover { background: #f8fafc; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #f8fafc; }
    .card .label { color: #64748b; font-size: 13px; }
    .card .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th { text-align: left; background: #f1f5f9; color: #111827; padding: 9px 10px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; }
    td { padding: 9px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }
    code { background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }
    svg text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }
    @media (max-width: 1000px) { .grid, .cards { grid-template-columns: 1fr; } }
    """


def metric_cards(mode_rows: list[dict[str, Any]], pair_summary_rows: list[dict[str, Any]]) -> str:
    by_mode = {row["mode"]: row for row in mode_rows}
    pref = by_mode.get("live_prefetch", {})
    paired = pair_summary_rows[0] if pair_summary_rows else {}
    cards = [
        ("paired tool gaps", paired.get("paired_tool_gaps", "")),
        ("avg live prefetch gain", f"{paired.get('avg_prefetch_gain_ms', '')} ms"),
        ("late prefetch pairs", paired.get("prefetch_late_pairs", "")),
        ("live tool calls", pref.get("total_tool_calls", "")),
    ]
    return "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{fmt(label)}</div><div class=\"value\">{fmt(value)}</div></div>"
        for label, value in cards
    ) + "</div>"


def setup_diagram_svg() -> str:
    boxes = [
        (50, 65, 190, 74, "Agent Task", "SWE-bench / DeepAgents"),
        (280, 65, 190, 74, "First Turn", "model builds KV"),
        (510, 65, 190, 74, "Tool Call", "read_file / run_tests"),
        (740, 65, 190, 74, "Tool Wait Gap", "prefetch opportunity"),
        (970, 65, 190, 74, "Hint / Prefetch", "try to prepare KV"),
        (1200, 65, 190, 74, "Resume Turn", "measure latency"),
    ]
    arrows = [
        (240, 102, 280, 102),
        (470, 102, 510, 102),
        (700, 102, 740, 102),
        (930, 102, 970, 102),
        (1160, 102, 1200, 102),
    ]
    parts = [
        '<svg viewBox="0 0 1440 210" width="100%" role="img" aria-label="Simple experiment setup flow diagram">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#334155"/>',
        "</marker>",
        "</defs>",
        '<rect x="20" y="25" width="1400" height="150" rx="10" fill="#f8fafc" stroke="#e5e7eb"/>',
    ]
    for x1, y1, x2, y2 in arrows:
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    for idx, (x, y, w, h, title, subtitle) in enumerate(boxes):
        fill = "#fff7ed" if idx in (3, 4) else "#eff6ff" if idx == 5 else "#ffffff"
        stroke = "#ea580c" if idx in (3, 4) else "#2563eb" if idx == 5 else "#cbd5e1"
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
                f'<text x="{x + w / 2}" y="{y + 27}" text-anchor="middle" font-size="15" font-weight="700">{html.escape(title)}</text>',
                f'<text x="{x + w / 2}" y="{y + 48}" text-anchor="middle" font-size="12" fill="#475569">{html.escape(subtitle)}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="720" y="192" text-anchor="middle" font-size="13" fill="#475569">Core question: was the right KV ready before the resume turn arrived?</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def experiment_setup_html(mode_rows: list[dict[str, Any]], pair_summary_rows: list[dict[str, Any]]) -> str:
    by_mode = {row["mode"]: row for row in mode_rows}
    no_prefetch = by_mode.get("no_prefetch", {})
    live_prefetch = by_mode.get("live_prefetch", {})
    paired = pair_summary_rows[0] if pair_summary_rows else {}
    setup_rows = [
        {"part": "1. Request source", "simple meaning": "Real SWE-bench / DeepAgents tasks create model turns.", "example": "SWE-bench task -> DeepAgents -> SGLang model request"},
        {"part": "2. Tool wait window", "simple meaning": "The agent calls a tool, then pauses. That pause is the chance to prefetch KV.", "example": "read_file(), grep(), or run_tests() is running"},
        {"part": "3. Resume request", "simple meaning": "The tool returns, then the agent asks the model to continue.", "example": "tests failed with error X; what should I do next?"},
    ]
    mode_rows_simple = [
        {"mode": "No prefetch", "what happens": "The system waits until the resume turn arrives, then SGLang handles KV reuse/load normally."},
        {"mode": "Live prefetch", "what happens": "During the tool wait, the hint path tries to prepare KV before the resume turn arrives."},
    ]
    metric_rows = [
        {"metric": "tool gap", "meaning": "Time between a tool-call response and the next model request from the same live agent run."},
        {"metric": "prefetch duration", "meaning": "How long the live software prefetch/controller path took to complete."},
        {"metric": "prefetch margin", "meaning": "Whether prefetch finished before or after the real resume request boundary."},
        {"metric": "resume request latency", "meaning": "End-to-end latency of the next model request after the tool gap."},
        {"metric": "late prefetch count", "meaning": "How often the hint path missed the available tool-gap window."},
    ]
    observation_rows = [
        {"observation": "The traffic was live agent traffic, not synthetic prompts.", "evidence": f"{no_prefetch.get('analyzed_model_requests', '')} no-prefetch requests and {live_prefetch.get('analyzed_model_requests', '')} live-prefetch requests were analyzed."},
        {"observation": "The run produced real tool-call gaps.", "evidence": f"{no_prefetch.get('observed_tool_gaps', '')} no-prefetch gaps and {live_prefetch.get('observed_tool_gaps', '')} live-prefetch gaps were observed."},
        {"observation": "Tool gaps were often very short.", "evidence": f"Median live-prefetch tool gap was {live_prefetch.get('median_tool_gap_ms', '')} ms."},
        {"observation": "Software prefetch was usually too slow for those windows.", "evidence": f"{live_prefetch.get('late_prefetch_attempts', '')} late prefetch attempts; average prefetch duration was {live_prefetch.get('avg_prefetch_duration_ms', '')} ms."},
        {"observation": "Hints alone did not guarantee a win.", "evidence": f"Paired gaps={paired.get('paired_tool_gaps', '')}; faster pairs={paired.get('prefetch_faster_pairs', '')}; slower pairs={paired.get('prefetch_slower_pairs', '')}."},
    ]
    return f"""
    <div class="setup-diagram">{setup_diagram_svg()}</div>
    <h3>Simple Setup</h3>
    {table_html(setup_rows, ["part", "simple meaning", "example"])}
    <h3>Modes Compared</h3>
    {table_html(mode_rows_simple, ["mode", "what happens"])}
    <h3>What Was Measured</h3>
    {table_html(metric_rows, ["metric", "meaning"])}
    <h3>What Was Observed</h3>
    {table_html(observation_rows, ["observation", "evidence"])}
    <h3>Why This Supports The Hardware Proposal</h3>
    <p>Current GPU/runtime data-movement paths can move memory, but they do not know that a transfer is urgent KV for a soon-resuming agent session. This experiment shows that when prefetch is routed through ordinary software and SGLang request paths, it can miss short live tool gaps. A hint-aware hardware/runtime path could make these movements more predictable by prioritizing urgent KV, protecting prefetched KV, and exposing telemetry for late or wasted prefetches.</p>
    """


def timeline_guide_html(profiled_available: bool) -> str:
    rows = [
        {"color": "blue", "meaning": "Initial model turn", "where_used": "Clean performance timelines"},
        {"color": "gray", "meaning": "Tool wait / prefetch opportunity", "where_used": "Clean and profiled timelines"},
        {"color": "purple", "meaning": "Software hint or prefetch request", "where_used": "Prefetch timelines"},
        {"color": "black", "meaning": "Replay due / resume boundary", "where_used": "Clean and profiled timelines"},
        {"color": "red", "meaning": "Replay request after the tool returns", "where_used": "Clean and profiled timelines"},
        {"color": "yellow", "meaning": "First token marker", "where_used": "Synthetic clean timelines"},
        {"color": "green", "meaning": "KV/copy activity; dark green means CUDA HtoD profiler evidence", "where_used": "Profiled mechanism timelines"},
    ]
    note = (
        "This live master report currently has clean live request timing only. CUDA HtoD / dark-green bars will appear after we add the live profiled attribution run."
        if not profiled_available
        else "This report includes profiled KV/DMA attribution."
    )
    return f"""
    <p class="note">{html.escape(note)}</p>
    {table_html(rows, ["color", "meaning", "where_used"])}
    """


SECTION_THEMES = {
    "summary": "theme-summary",
    "setup": "theme-setup",
    "timeline-guide": "theme-guide",
    "timelines": "theme-clean",
    "performance": "theme-clean-table",
    "profiled": "theme-profiled",
    "deductions": "theme-deductions",
    "checkpoints": "theme-checkpoints",
    "observations": "theme-observations",
    "paired": "theme-paired",
    "appendix": "theme-appendix",
}


def toc_html(items: list[tuple[str, str]]) -> str:
    links = "".join(
        f'<a class="{SECTION_THEMES.get(anchor, "theme-appendix")}" href="#{anchor}">{html.escape(label)}</a>'
        for anchor, label in items
    )
    return (
        '<p class="section-color-legend">Colors group sections by evidence type: setup, clean performance, '
        'profiled mechanism, interpretation, and appendix.</p>'
        f'<div class="toc">{links}</div>'
        '<div class="toc-actions"><button type="button" data-action="expand-all">Expand All</button>'
        '<button type="button" data-action="collapse-all">Collapse All</button></div>'
    )


def report_script() -> str:
    return """
<script>
document.addEventListener("DOMContentLoaded", function () {
  const cards = Array.from(document.querySelectorAll("details.section-card"));
  document.querySelectorAll(".toc a[href^='#']").forEach(function (link) {
    link.addEventListener("click", function () {
      const id = link.getAttribute("href").slice(1);
      const target = document.getElementById(id);
      if (target && target.tagName.toLowerCase() === "details") {
        target.open = true;
      }
    });
  });
  document.querySelectorAll("[data-action='expand-all']").forEach(function (button) {
    button.addEventListener("click", function () {
      cards.forEach(function (card) { card.open = true; });
    });
  });
  document.querySelectorAll("[data-action='collapse-all']").forEach(function (button) {
    button.addEventListener("click", function () {
      cards.forEach(function (card) { card.open = false; });
    });
  });
});
</script>
"""


def render_html(
    no_run: dict[str, Any],
    pref_run: dict[str, Any],
    mode_rows: list[dict[str, Any]],
    pair_summary_rows: list[dict[str, Any]],
    deductions: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    max_timeline_gaps: int,
) -> str:
    pref_gaps = pref_run["gaps"]
    no_gaps = no_run["gaps"]
    checkpoint = checkpoint_rows(pref_gaps)
    observations = session_observations(pref_gaps, max_timeline_gaps)
    timeline_rows = timeline_summary(pref_gaps, max_timeline_gaps)
    detail_rows = [
        {
            "mode": "no_prefetch",
            "root": no_run["root"],
            "proxy_jsonl": no_run["proxy_jsonl"],
            "task_index_csv": no_run["task_index_csv"],
        },
        {
            "mode": "live_prefetch",
            "root": pref_run["root"],
            "proxy_jsonl": pref_run["proxy_jsonl"],
            "task_index_csv": pref_run["task_index_csv"],
            "hint_log": pref_run["hint_log"],
            "controller_log": pref_run["controller_log"],
        },
    ]
    toc = [
        ("summary", "Summary"),
        ("setup", "Experiment Setup"),
        ("timeline-guide", "How To Read Timelines"),
        ("timelines", "Clean Performance Timelines"),
        ("performance", "Clean Performance Tables"),
        ("profiled", "Profiled Mechanism Evidence"),
        ("deductions", "Key Deductions"),
        ("checkpoints", "Prefetch Checkpoints"),
        ("observations", "Key Observations Per Session"),
        ("paired", "Paired Session Evidence"),
        ("appendix", "Appendix"),
    ]
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Paired AgentBench Report</title>
  <style>{css()}</style>
</head>
<body>
<main>
  <section>
    <h1>Live Paired AgentBench Report</h1>
    <p>This is the live version of the paired evidence report. The request generator is real SWE-bench / Deep Agents traffic; the backend is direct SGLang; the comparison is no-prefetch versus live software prefetch intervention.</p>
    <p class="note">Important metric note: this report uses full resume request latency captured by the proxy, not streaming TTFT. It still shows whether the next real agent turn became faster or slower after the prefetch intervention.</p>
    <h2>Table of Contents</h2>
    {toc_html(toc)}
  </section>

  <details id="summary" class="section-card theme-summary" open>
    <summary><h2>Summary</h2></summary>
    <p>This report uses real SWE-bench / DeepAgents traffic. It answers whether the live software hint path finishes before the next real agent turn resumes.</p>
    {metric_cards(mode_rows, pair_summary_rows)}
  </details>

  <details id="setup" class="section-card theme-setup">
    <summary><h2>Experiment Setup And Manager Summary</h2></summary>
    <p>This section is intended for slide-building: it shows the live request path, the hint/prefetch path, how the experiment was conducted, and the main evidence collected.</p>
    {experiment_setup_html(mode_rows, pair_summary_rows)}
  </details>

  <details id="timeline-guide" class="section-card theme-guide">
    <summary><h2>How To Read The Timelines</h2></summary>
    <p>The timelines are the primary visual evidence. Tables below each timeline provide the exact numbers behind the picture.</p>
    {timeline_guide_html(profiled_available=False)}
  </details>

  <details id="timelines" class="section-card theme-clean">
    <summary><h2>A. Clean Performance Timelines</h2></summary>
    <p class="note">Profiler is off. Use this section for live request-flow and latency claims.</p>
    <p>Blue is a live model turn that emitted tool calls. Gray is the observed tool/harness gap. Red is the next live model turn. Purple appears only in the live-prefetch run and shows the software prefetch request.</p>
    <div class="timeline-stack">
      <div class="timeline-panel">
        <h3>No Prefetch</h3>
        {build_timeline_svg(no_gaps, max_timeline_gaps)}
      </div>
      <div class="timeline-panel">
        <h3>Live Prefetch Intervention</h3>
        {build_timeline_svg(pref_gaps, max_timeline_gaps)}
      </div>
    </div>
  </details>

  <details id="performance" class="section-card theme-clean-table">
    <summary><h2>A.1 Clean Performance Tables</h2></summary>
    <p>These tables provide the exact request counts, tool-gap counts, latency values, and paired aggregate numbers behind the clean timelines.</p>
    <h3>By Mode</h3>
    {table_html(mode_rows)}
    <h3>Paired Aggregate</h3>
    {table_html(pair_summary_rows)}
  </details>

  <details id="profiled" class="section-card theme-profiled">
    <summary><h2>B. Profiled Mechanism Evidence</h2></summary>
    <p class="warn">Not available yet for the real SWE-bench / DeepAgents master report. This report currently shows clean live request timing and live hint-controller timing, but it does not yet include torch-profiler CUDA HtoD attribution for live SWE-bench traffic.</p>
    <p>After we add the live profiled attribution run, this section will show dark-green CUDA HtoD copy bars, KV/copy telemetry, replay reload evidence, and checkpoint tables for the real traffic path.</p>
  </details>

  <details id="deductions" class="section-card theme-deductions">
    <summary><h2>Key Deductions</h2></summary>
    {table_html(deductions, ["finding", "evidence", "why_it_matters"])}
  </details>

  <details id="checkpoints" class="section-card theme-checkpoints">
    <summary><h2>Prefetch Checkpoints</h2></summary>
    <p>These checkpoints make the live intervention path explicit: hint submitted, controller started, controller finished, and whether it finished before resume.</p>
    {table_html(checkpoint[:max_timeline_gaps])}
  </details>

  <details id="observations" class="section-card theme-observations">
    <summary><h2>Key Observations Per Session</h2></summary>
    {table_html(observations, ["session_id", "status", "what_happened", "deduction_and_evidence"])}
  </details>

  <details id="paired" class="section-card theme-paired">
    <summary><h2>Paired Session Evidence</h2></summary>
    {table_html(pair_rows)}
  </details>

  <details id="appendix" class="section-card theme-appendix">
    <summary><h2>Appendix: Detailed Evidence</h2></summary>
    <h3>Timeline Summary</h3>
    {table_html(timeline_rows)}
    <h3>Input Runs</h3>
    {table_html(detail_rows)}
    <h3>Live Request Details</h3>
    {table_html(request_details(pref_run, 80))}
  </details>
</main>
{report_script()}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired live AgentBench/SGLang report.")
    parser.add_argument("--no-prefetch-root", type=Path, required=True)
    parser.add_argument("--prefetch-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--max-timeline-gaps", type=int, default=16)
    parser.add_argument("--include-preflight", action="store_true")
    args = parser.parse_args()

    no_run = load_live_run(args.no_prefetch_root, "no_prefetch", args.include_preflight)
    pref_run = load_live_run(args.prefetch_root, "live_prefetch", args.include_preflight)
    mode_rows = [mode_summary(no_run), mode_summary(pref_run)]
    pair_rows = pair_gaps(no_run["gaps"], pref_run["gaps"])
    pair_summary_rows = paired_summary(pair_rows)
    deductions = key_deductions(mode_rows, pair_summary_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "live_paired_mode_summary.csv", mode_rows)
    write_csv(args.out_dir / "live_paired_summary.csv", pair_summary_rows)
    write_csv(args.out_dir / "live_paired_key_deductions.csv", deductions)
    write_csv(args.out_dir / "live_paired_session_evidence.csv", pair_rows)
    write_csv(args.out_dir / "live_prefetch_checkpoint_results.csv", checkpoint_rows(pref_run["gaps"]))
    write_csv(args.out_dir / "live_prefetch_timeline_summary.csv", timeline_summary(pref_run["gaps"], 10_000))
    write_csv(args.out_dir / "live_prefetch_session_observations.csv", session_observations(pref_run["gaps"], 10_000))
    write_csv(args.out_dir / "live_prefetch_request_details.csv", request_details(pref_run, 10_000))
    write_csv(args.out_dir / "live_no_prefetch_tool_gaps.csv", no_run["gaps"])
    write_csv(args.out_dir / "live_prefetch_tool_gaps.csv", pref_run["gaps"])

    report = {
        "no_prefetch": {key: value for key, value in no_run.items() if key not in {"requests", "gaps"}},
        "live_prefetch": {key: value for key, value in pref_run.items() if key not in {"requests", "gaps"}},
        "mode_summary": mode_rows,
        "paired_summary": pair_summary_rows,
        "key_deductions": deductions,
        "paired_session_evidence": pair_rows,
        "no_prefetch_gaps": no_run["gaps"],
        "prefetch_gaps": pref_run["gaps"],
    }
    json_path = args.out_dir / "live_paired_agentbench_report.json"
    md_path = args.out_dir / "live_paired_agentbench_report.md"
    html_path = args.out_dir / "live_paired_agentbench_report.html"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(mode_rows, pair_summary_rows, deductions, pair_rows), encoding="utf-8")
    html_path.write_text(
        render_html(no_run, pref_run, mode_rows, pair_summary_rows, deductions, pair_rows, args.max_timeline_gaps),
        encoding="utf-8",
    )

    if args.latest_root:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        latest_real = args.latest_root / "latest_real"
        latest_real.mkdir(parents=True, exist_ok=True)
        latest_pairs = [
            (html_path, "latest_master_report.html"),
        ]
        for source, name in latest_pairs:
            if source.exists():
                shutil.copyfile(source, args.latest_root / name)
        detail_pairs = [
            (html_path, "master_report.html"),
            (md_path, "master_report.md"),
            (json_path, "master_report.json"),
            (html_path, "m24_live_paired_report.html"),
            (md_path, "m24_live_paired_report.md"),
            (json_path, "m24_live_paired_report.json"),
            (args.out_dir / "live_paired_session_evidence.csv", "session_evidence.csv"),
            (args.out_dir / "live_paired_mode_summary.csv", "mode_summary.csv"),
            (args.out_dir / "live_paired_summary.csv", "paired_summary.csv"),
            (args.out_dir / "live_paired_key_deductions.csv", "key_deductions.csv"),
            (args.out_dir / "live_prefetch_checkpoint_results.csv", "prefetch_checkpoint_results.csv"),
            (args.out_dir / "live_prefetch_timeline_summary.csv", "prefetch_timeline_summary.csv"),
            (args.out_dir / "live_prefetch_session_observations.csv", "prefetch_session_observations.csv"),
        ]
        for source, name in detail_pairs:
            if source.exists():
                shutil.copyfile(source, latest_real / name)

    print(f"Wrote live paired AgentBench report: {html_path}")
    print(f"No-prefetch gaps: {len(no_run['gaps'])}")
    print(f"Live-prefetch gaps: {len(pref_run['gaps'])}")
    print(f"Paired gaps: {pair_summary_rows[0].get('paired_tool_gaps', 0) if pair_summary_rows else 0}")


if __name__ == "__main__":
    main()
