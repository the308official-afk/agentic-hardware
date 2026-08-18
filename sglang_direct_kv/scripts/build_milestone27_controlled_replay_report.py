#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from build_live_agentbench_tool_gap_report import (
    build_expanded_gap_timeline_svg,
    read_jsonl,
    table_html,
    write_csv,
)
from build_live_paired_agentbench_report import (
    css as master_css,
    global_prefetch_margin_html,
    load_live_run as load_live_agentbench_run,
    mode_summary as live_mode_summary,
    movement_events_by_session,
    report_script,
    setup_diagram_svg,
    timeline_guide_html,
    toc_html,
)


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def rel_ms(ts_ns: Any, base_ts: float) -> float | str:
    value = as_float(ts_ns)
    if value is None:
        return ""
    return round((value / 1_000_000_000.0 - base_ts) * 1000.0, 3)


def has_events(value: Any) -> bool:
    try:
        return int(value or 0) > 0
    except (TypeError, ValueError):
        return False


def summarize_movement(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"start_ms": "", "end_ms": "", "duration_ms": "", "events": "0", "categories": ""}
    starts: list[float] = []
    ends: list[float] = []
    total = 0.0
    counted = 0
    categories: dict[str, int] = defaultdict(int)
    for event in events:
        ts = as_float(event.get("start_or_end_ms"))
        if ts is None:
            continue
        duration = as_float(event.get("duration_ms"))
        if duration is not None:
            starts.append(ts - duration)
            ends.append(ts)
            total += duration
            counted += 1
        else:
            starts.append(ts)
            ends.append(ts)
        categories[str(event.get("category") or event.get("event") or "event")] += 1
    if not starts or not ends:
        return {"start_ms": "", "end_ms": "", "duration_ms": "", "events": str(len(events)), "categories": ""}
    return {
        "start_ms": round(min(starts), 3),
        "end_ms": round(max(ends), 3),
        "duration_ms": round(total, 3) if counted else "",
        "events": str(len(events)),
        "categories": ", ".join(f"{key}:{value}" for key, value in sorted(categories.items())),
    }


def trace_request_windows(trace_rows: list[dict[str, Any]], base_ts: float) -> dict[tuple[str, str], dict[str, Any]]:
    windows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in trace_rows:
        if row.get("event") not in {"m27.request.start", "m27.request.end"}:
            continue
        session = str(row.get("session_id") or "")
        phase = str(row.get("phase") or "")
        if not session or not phase:
            continue
        item = windows.setdefault(
            (session, phase),
            {
                "session_id": session,
                "phase": phase,
                "mode": row.get("mode", ""),
                "label": row.get("label", ""),
                "prompt_hash": row.get("prompt_hash", ""),
            },
        )
        if row.get("event") == "m27.request.start":
            item["start_ms"] = rel_ms(row.get("ts_ns"), base_ts)
        else:
            item["end_ms"] = rel_ms(row.get("ts_ns"), base_ts)
            item["ttft_ms"] = row.get("ttft_ms", "")
            item["total_latency_ms"] = row.get("total_latency_ms", "")
    return windows


def first_event_by_session(trace_rows: list[dict[str, Any]], event_name: str, base_ts: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        if row.get("event") != event_name:
            continue
        session = str(row.get("session_id") or "")
        if not session or session in out:
            continue
        copied = dict(row)
        copied["ms"] = rel_ms(row.get("ts_ns"), base_ts)
        out[session] = copied
    return out


def latest_event_by_session(trace_rows: list[dict[str, Any]], event_name: str, base_ts: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        if row.get("event") != event_name:
            continue
        session = str(row.get("session_id") or "")
        if not session:
            continue
        copied = dict(row)
        copied["ms"] = rel_ms(row.get("ts_ns"), base_ts)
        out[session] = copied
    return out


def events_in_window(
    events_by_session: dict[str, list[dict[str, Any]]],
    session_id: str,
    start_ms: Any,
    end_ms: Any,
) -> list[dict[str, Any]]:
    start = as_float(start_ms)
    end = as_float(end_ms)
    if start is None or end is None:
        return []
    matched: list[dict[str, Any]] = []
    for event in events_by_session.get(session_id, []):
        ts = as_float(event.get("start_or_end_ms"))
        if ts is not None and start <= ts <= end:
            matched.append(event)
    return matched


def build_gaps_for_case(case_dir: Path, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
    telemetry_rows = read_jsonl(case_dir / "m27_copy_telemetry.jsonl")
    if not trace_rows:
        return [], []
    base_ts = min((float(row["ts_ns"]) / 1_000_000_000.0 for row in trace_rows if row.get("ts_ns")), default=0.0)
    requests = trace_request_windows(trace_rows, base_ts)
    session_meta = first_event_by_session(trace_rows, "m27.session.start", base_ts)
    tool_starts = first_event_by_session(trace_rows, "m27.tool_wait.start", base_ts)
    replay_due = first_event_by_session(trace_rows, "m27.replay.due", base_ts)
    hint_submitted = first_event_by_session(trace_rows, "m27.hint.submitted", base_ts)
    prefetch_starts = first_event_by_session(trace_rows, "m27.prefetch.start", base_ts)
    prefetch_ends = latest_event_by_session(trace_rows, "m27.prefetch.end", base_ts)
    movement_by_session = movement_events_by_session(trace_rows, telemetry_rows, base_ts)
    sessions = sorted(session_meta)
    gaps: list[dict[str, Any]] = []
    for idx, session in enumerate(sessions):
        meta = session_meta.get(session, {})
        current = requests.get((session, "initial_turn"), {})
        replay = requests.get((session, "replay"), {})
        tool = tool_starts.get(session, {})
        due = replay_due.get(session, {})
        hint = hint_submitted.get(session, {})
        p_start = prefetch_starts.get(session, {})
        p_end = prefetch_ends.get(session, {})
        if not current or not replay:
            continue
        prefetch_start_ms = p_start.get("ms", "")
        prefetch_end_ms = p_end.get("ms", "")
        hint_events = events_in_window(movement_by_session, session, prefetch_start_ms, prefetch_end_ms)
        replay_events = events_in_window(movement_by_session, session, replay.get("start_ms"), replay.get("end_ms"))
        hint_summary = summarize_movement(hint_events)
        replay_summary = summarize_movement(replay_events)
        margin = ""
        if prefetch_end_ms not in ("", None) and replay.get("start_ms") not in ("", None):
            margin = round(float(replay["start_ms"]) - float(prefetch_end_ms), 3)
        gap = {
            "session_id": session,
            "mode": mode,
            "task_index": meta.get("task_index", ""),
            "gap_order_in_task": idx,
            "tool_names": meta.get("tool_names", ""),
            "tool_call_count": 1 if meta.get("tool_names") else "",
            "tool_gap_ms": meta.get("tool_wait_ms", ""),
            "current_start_ms": current.get("start_ms", ""),
            "current_end_ms": current.get("end_ms", ""),
            "current_latency_ms": current.get("total_latency_ms", ""),
            "tool_gap_start_ms": tool.get("ms", ""),
            "tool_gap_end_ms": replay.get("start_ms", ""),
            "hint_submitted_ms": hint.get("ms", ""),
            "prefetch_start_ms": prefetch_start_ms,
            "prefetch_end_ms": prefetch_end_ms,
            "prefetch_duration_ms": (
                round(float(prefetch_end_ms) - float(prefetch_start_ms), 3)
                if prefetch_start_ms not in ("", None) and prefetch_end_ms not in ("", None)
                else ""
            ),
            "prefetch_margin_ms": margin,
            "prefetch_done_before_resume": 1 if isinstance(margin, (int, float)) and margin >= 0 else 0 if margin != "" else "",
            "prefetch_status": "done" if prefetch_end_ms not in ("", None) else "no_hint",
            "resume_start_ms": replay.get("start_ms", ""),
            "resume_end_ms": replay.get("end_ms", ""),
            "resume_latency_ms": replay.get("total_latency_ms", ""),
            "resume_ttft_ms": replay.get("ttft_ms", ""),
            "direct_kv_h2d_start_ms": hint_summary["start_ms"],
            "direct_kv_h2d_end_ms": hint_summary["end_ms"],
            "direct_kv_h2d_duration_ms": hint_summary["duration_ms"],
            "direct_kv_h2d_events": hint_summary["events"],
            "direct_kv_h2d_categories": hint_summary["categories"],
            "replay_kv_h2d_start_ms": replay_summary["start_ms"],
            "replay_kv_h2d_end_ms": replay_summary["end_ms"],
            "replay_kv_h2d_duration_ms": replay_summary["duration_ms"],
            "replay_kv_h2d_events": replay_summary["events"],
            "replay_kv_h2d_categories": replay_summary["categories"],
        }
        if has_events(gap["direct_kv_h2d_events"]) and has_events(gap["replay_kv_h2d_events"]):
            gap["movement_class"] = "hint and replay both moved KV"
        elif has_events(gap["direct_kv_h2d_events"]):
            gap["movement_class"] = "hint-side KV movement observed"
        elif has_events(gap["replay_kv_h2d_events"]):
            gap["movement_class"] = "replay loaded KV"
        else:
            gap["movement_class"] = "no visible HtoD"
        gaps.append(gap)
    return gaps, trace_rows


def mode_summary_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gap in gaps:
        by_mode[str(gap.get("mode") or "")].append(gap)
    rows: list[dict[str, Any]] = []
    for mode, items in sorted(by_mode.items()):
        margins = [float(row["prefetch_margin_ms"]) for row in items if row.get("prefetch_margin_ms") not in ("", None)]
        replay_ttfts = [float(row["resume_ttft_ms"]) for row in items if row.get("resume_ttft_ms") not in ("", None)]
        rows.append(
            {
                "mode": mode,
                "controlled_gaps": len(items),
                "prefetch_attempts": len(margins),
                "late_prefetches": sum(1 for value in margins if value < 0),
                "median_prefetch_margin_ms": round(median(margins), 3) if margins else "",
                "avg_resume_ttft_ms": round(mean(replay_ttfts), 3) if replay_ttfts else "",
                "hint_h2d_gaps": sum(1 for row in items if has_events(row.get("direct_kv_h2d_events"))),
                "replay_h2d_gaps": sum(1 for row in items if has_events(row.get("replay_kv_h2d_events"))),
            }
        )
    return rows


def selected_timeline_gaps(gaps: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    interesting = [
        row
        for row in gaps
        if has_events(row.get("direct_kv_h2d_events")) or has_events(row.get("replay_kv_h2d_events"))
    ]
    if len(interesting) >= max_rows:
        return interesting[:max_rows]
    seen = {id(row) for row in interesting}
    for row in gaps:
        if id(row) not in seen:
            interesting.append(row)
        if len(interesting) >= max_rows:
            break
    return interesting


def timeline_rows_with_labels(rows: list[dict[str, Any]], prefix: str = "G") -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        copied = dict(row)
        copied["timeline_label"] = f"{prefix}{idx:02d}"
        labeled.append(copied)
    return labeled


def case_fillers(row: dict[str, Any]) -> str:
    name = Path(str(row.get("case_dir") or "")).name
    if "_f" not in name:
        return ""
    return name.rsplit("_f", 1)[-1]


def timeline_mapping_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        output.append(
            {
                "row": row.get("timeline_label") or f"G{idx:02d}",
                "mode": row.get("mode", ""),
                "fillers": case_fillers(row),
                "task": row.get("task_index", ""),
                "gap": row.get("gap_order_in_task", ""),
                "tool_wait_ms": row.get("tool_gap_ms", ""),
                "prefetch_margin_ms": row.get("prefetch_margin_ms", ""),
                "movement": row.get("movement_class", ""),
            }
        )
    return output


def observation_status(row: dict[str, Any]) -> tuple[str, str]:
    mode = str(row.get("mode") or "")
    margin = as_float(row.get("prefetch_margin_ms"))
    hint_h2d = has_events(row.get("direct_kv_h2d_events"))
    replay_h2d = has_events(row.get("replay_kv_h2d_events"))
    if mode == "no_prefetch":
        if replay_h2d:
            return "No prefetch; replay loaded KV", "#b45309"
        return "No prefetch; no visible H2D", "#64748b"
    if margin is None:
        if replay_h2d:
            return "No completed prefetch; replay loaded KV", "#b45309"
        return "No completed prefetch", "#64748b"
    if margin < 0:
        if replay_h2d:
            return "Late prefetch; replay loaded KV", "#b91c1c"
        return "Late prefetch", "#b91c1c"
    if hint_h2d and replay_h2d:
        return "Prefetch ready, but replay also loaded KV", "#b45309"
    if hint_h2d:
        return "Useful direct KV prefetch", "#166534"
    if replay_h2d:
        return "Prefetch finished, replay loaded KV", "#b45309"
    return "Prefetch finished; no visible H2D", "#166534"


def key_observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        label = str(row.get("timeline_label") or f"G{idx:02d}")
        mode = str(row.get("mode") or "")
        margin = as_float(row.get("prefetch_margin_ms"))
        gap_ms = as_float(row.get("tool_gap_ms"))
        ttft_ms = as_float(row.get("resume_ttft_ms"))
        hint_h2d = has_events(row.get("direct_kv_h2d_events"))
        replay_h2d = has_events(row.get("replay_kv_h2d_events"))
        status, _ = observation_status(row)

        if mode == "no_prefetch":
            if replay_h2d:
                what = "No hint was issued. When the resume request arrived, SGLang performed replay-side KV HtoD movement."
                why = "This is the baseline: the real request path had to handle KV movement at resume time."
            else:
                what = "No hint was issued, and this row did not show replay-side HtoD movement."
                why = "The KV may already have been resident/reusable, or this row did not trigger observable host-to-device movement."
        elif margin is None:
            what = "A prefetch mode was selected, but the trace did not show a completed prefetch window for this row."
            why = "This is useful as a control/coverage warning, but it is weaker evidence than rows with measured margins."
        elif margin < 0:
            what = f"The prefetch attempt finished {abs(margin):.0f} ms after the resume request was already due."
            if replay_h2d:
                what += " The resume request also showed replay-side KV HtoD movement."
                why = "This is the failure case: the normal request path had to move KV because the hint path did not finish in time."
            else:
                why = "The hint path was late, so software prefetch did not meet the agent resume deadline."
        else:
            what = f"The prefetch attempt finished {margin:.0f} ms before the resume request was due."
            if hint_h2d and not replay_h2d:
                what += " Direct KV HtoD movement was visible on the hint side, with no replay-side HtoD in this row."
                why = "This is the clean success case: the hint appears to have prepared KV before the agent resumed."
            elif hint_h2d and replay_h2d:
                what += " But replay-side KV HtoD was also visible later."
                why = "This is an important hardware argument: moving KV early is not enough unless residency and reuse are also protected."
            elif replay_h2d:
                what += " Replay-side KV HtoD was still visible."
                why = "The software hint completed early, but the resume path still had to move KV, so reuse was not fully predictable."
            else:
                why = "The hint completed before replay, but this row did not show visible HtoD movement."

        output.append(
            {
                "row": label,
                "mode": mode,
                "status": status,
                "what happened": what,
                "why it matters": why,
                "tool_wait_ms": round(gap_ms, 3) if gap_ms is not None else "",
                "resume_ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else "",
            }
        )
    return output


def manager_setup_html() -> str:
    setup_rows = [
        {"part": "1. Real prompt pair", "simple meaning": "Use two adjacent model turns from real AgentBench/DeepAgents traces."},
        {"part": "2. Controlled wait", "simple meaning": "Replay Turn A, then impose a chosen tool-wait window such as 100 ms or 500 ms."},
        {"part": "3. Cache pressure", "simple meaning": "Send filler requests during the wait to make KV residency harder."},
        {"part": "4. Direct KV hint", "simple meaning": "In prefetch modes, issue the marked direct-load request during the wait."},
        {"part": "5. Resume", "simple meaning": "Send Turn B at the scheduled resume time, even if the hint is still running."},
    ]
    return f"""
    <div class="setup-diagram">{setup_diagram_svg()}</div>
    <p>This milestone keeps real SWE-bench/DeepAgents prompt content, but controls the timing. That gives us a cleaner hardware-style experiment: same kind of agent text, known wait windows, known cache pressure, and direct SGLang KV-hook attempts.</p>
    {table_html(setup_rows, ["part", "simple meaning"])}
    """


def metric_cards_html(mode_rows: list[dict[str, Any]]) -> str:
    by_mode = {str(row.get("mode") or ""): row for row in mode_rows}
    no_prefetch = by_mode.get("no_prefetch", {})
    direct = by_mode.get("direct_prefetch", {})
    cards = [
        ("controlled gaps", sum(int(row.get("controlled_gaps") or 0) for row in mode_rows)),
        ("no-prefetch avg TTFT", f"{no_prefetch.get('avg_resume_ttft_ms', '')} ms"),
        ("direct-prefetch avg TTFT", f"{direct.get('avg_resume_ttft_ms', '')} ms"),
        ("direct late prefetches", direct.get("late_prefetches", "")),
        ("direct H2D gaps", direct.get("hint_h2d_gaps", "")),
    ]
    if "oracle_prefetch" in by_mode:
        oracle = by_mode["oracle_prefetch"]
        cards.extend(
            [
                ("oracle-prefetch avg TTFT", f"{oracle.get('avg_resume_ttft_ms', '')} ms"),
                ("oracle late prefetches", oracle.get("late_prefetches", "")),
                ("oracle H2D gaps", oracle.get("hint_h2d_gaps", "")),
            ]
        )
    return "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{html.escape(str(label))}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
        for label, value in cards
    ) + "</div>"


def code_block(text: str) -> str:
    return f"<pre><code>{html.escape(text.strip())}</code></pre>"


def reproduce_controlled_replay_html(result_root: Path) -> str:
    current_root = str(result_root)
    rebuild_current = f"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/build_milestone27_controlled_replay_report.py \\
  --root {current_root} \\
  --out-dir {current_root}/controlled_replay_report \\
  --latest-root artifacts/results \\
  --max-timeline-gaps 18
"""
    run_labeled = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

export REPORT_LABEL=manager_demo_1
export RESULT_ROOT=artifacts/results/labeled/controlled_replay/${REPORT_LABEL}
export LATEST_REPORT_ROOT=${RESULT_ROOT}/latest

WORKLOAD_JSONL=/path/to/real_prompt_pairs.jsonl \
MAX_PAIRS=8 \
MODES="no_prefetch direct_prefetch" \
TOOL_WAIT_LIST_MS="100 250 500 1000" \
FILLER_LIST="16 32" \
REQUEST_CONCURRENCY=4 \
MAX_TOTAL_TOKENS=8192 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.72 \
bash scripts/run_milestone27_real_prompt_controlled_replay.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
"""
    run_from_trace_index = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

export REPORT_LABEL=trace_index_demo_1
export RESULT_ROOT=artifacts/results/labeled/controlled_replay/${REPORT_LABEL}
export LATEST_REPORT_ROOT=${RESULT_ROOT}/latest

TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
MAX_PAIRS=8 \
MODES="no_prefetch direct_prefetch" \
TOOL_WAIT_LIST_MS="100 250 500 1000" \
FILLER_LIST="16 32" \
REQUEST_CONCURRENCY=4 \
MAX_TOTAL_TOKENS=8192 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.72 \
bash scripts/run_milestone27_real_prompt_controlled_replay.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
"""
    refresh_latest = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone27_real_prompt_controlled_replay_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
WORKLOAD_JSONL=/path/to/real_prompt_pairs.jsonl \
MAX_PAIRS=8 \
MODES="no_prefetch direct_prefetch" \
TOOL_WAIT_LIST_MS="100 250 500 1000" \
FILLER_LIST="16 32" \
REQUEST_CONCURRENCY=4 \
MAX_TOTAL_TOKENS=8192 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.72 \
bash scripts/run_milestone27_real_prompt_controlled_replay.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
"""
    run_live_direct = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

# This runs the real SWE-bench / DeepAgents live path and generates
# the live direct-prefetch timeline inside latest_master_report.html.
RESULT_ROOT=artifacts/results/milestone26_live_direct_only_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
START_INDEX=1 \
END_INDEX=4 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=6 \
AGENTBENCH_DIRECT_SGLANG_MAX_TOKENS=512 \
MAX_TOTAL_TOKENS=16384 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.72 \
bash scripts/run_milestone26_live_direct_kv_load_intervention.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct

# Rebuild the master report so it includes the latest controlled replay
# section plus the latest live direct-prefetch section.
bash scripts/build_latest_master_with_live_direct.sh
"""
    run_live_paired = r"""
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

# This runs both live no-prefetch and live direct-prefetch, then builds
# the real-workload master report. It takes longer than the direct-only path.
RESULT_ROOT=artifacts/results/milestone26_live_paired_direct_kv_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
START_INDEX=0 \
END_INDEX=15 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
MAX_TOTAL_TOKENS=32768 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.45 \
bash scripts/run_milestone26_live_paired_direct_kv_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
"""
    return "\n".join(
        [
            "<p>This section gives copy-paste commands for reproducing the direct-KV controlled replay master report. The default experiment compares only <code>no_prefetch</code> and <code>direct_prefetch</code>. Labeled runs write to their own folder and do not overwrite the normal latest report.</p>",
            "<h3>Rebuild This Exact Report</h3>",
            "<p>Use this when the run folders already exist and you only want to regenerate the HTML, tables, and timelines.</p>",
            code_block(rebuild_current),
            "<p>Output:</p>",
            code_block(f"{current_root}/controlled_replay_report/controlled_replay_report.html\nartifacts/results/latest_master_report.html"),
            "<h3>Run A New Labeled Controlled Replay Experiment</h3>",
            "<p>Use this for a new manager-demo run from an existing real prompt-pair workload. This keeps the output under a label-specific folder.</p>",
            code_block(run_labeled),
            "<p>Output:</p>",
            code_block("artifacts/results/labeled/controlled_replay/manager_demo_1/controlled_replay_report/controlled_replay_report.html\nartifacts/results/labeled/controlled_replay/manager_demo_1/latest/latest_master_report.html"),
            "<h3>Generate The Live Direct-Prefetch Timeline</h3>",
            "<p>Use this when you want the real SWE-bench / DeepAgents live timeline in <code>latest_master_report.html</code>. This is the current preferred live path: real tool calls create live gaps, and the controller tries direct SGLang KV load-back during those gaps.</p>",
            code_block(run_live_direct),
            "<p>Output:</p>",
            code_block("artifacts/results/latest_master_report.html"),
            "<h3>Run A Full Live Paired Experiment</h3>",
            "<p>Use this when you want both live no-prefetch and live direct-prefetch runs. This is heavier, but it gives a fuller real-workload comparison.</p>",
            code_block(run_live_paired),
            "<h3>Build Prompt Pairs From An AgentBench Trace Index</h3>",
            "<p>Use this when you have an AgentBench trace index and want the script to extract real Turn A / Turn B prompt pairs first.</p>",
            code_block(run_from_trace_index),
            "<h3>Deliberately Refresh The Latest Master Report</h3>",
            "<p>Only use this when you want to replace <code>artifacts/results/latest_master_report.html</code>.</p>",
            code_block(refresh_latest),
        ]
    )


def live_direct_prefetch_html(live_run: dict[str, Any] | None, max_timeline_gaps: int) -> str:
    if not live_run:
        return """
  <details id="live-direct" class="section-card theme-profiled">
    <summary><h2>Live AgentBench Direct Prefetch</h2></summary>
    <p>No live direct-prefetch run was attached to this report.</p>
  </details>
"""
    gaps = live_run.get("gaps", [])
    mode_rows = [live_mode_summary(live_run)]
    interesting = timeline_rows_with_labels(selected_timeline_gaps(gaps, max_timeline_gaps), prefix="L")
    summary = mode_rows[0] if mode_rows else {}
    live_cards = [
        ("live analyzed requests", summary.get("analyzed_model_requests", "")),
        ("live tool calls", summary.get("total_tool_calls", "")),
        ("live tool gaps", summary.get("observed_tool_gaps", "")),
        ("late live prefetches", summary.get("late_prefetch_attempts", "")),
        ("avg live prefetch duration", f"{summary.get('avg_prefetch_duration_ms', '')} ms"),
        ("hint H2D gaps", sum(1 for row in gaps if has_events(row.get("direct_kv_h2d_events")))),
        ("replay H2D gaps", sum(1 for row in gaps if has_events(row.get("replay_kv_h2d_events")))),
    ]
    cards_html = "<div class=\"cards\">" + "\n".join(
        f"<div class=\"card\"><div class=\"label\">{html.escape(str(label))}</div><div class=\"value\">{html.escape(str(value))}</div></div>"
        for label, value in live_cards
    ) + "</div>"
    live_setup_rows = [
        {
            "part": "Request source",
            "simple meaning": "Real SWE-bench / DeepAgents task execution creates live model turns and real tool calls.",
        },
        {
            "part": "Hint trigger",
            "simple meaning": "When the live proxy sees a model turn produce tool calls, it emits a hint for that session.",
        },
        {
            "part": "Direct prefetch attempt",
            "simple meaning": "The controller sends a marked direct-load request so SGLang can exercise its KV load-back path.",
        },
        {
            "part": "Resume",
            "simple meaning": "The real agent continues after the tool work; the report checks whether prefetch finished before that resume.",
        },
    ]
    detail_columns = [
        "session_id",
        "task_index",
        "gap_order_in_task",
        "tool_names",
        "tool_gap_ms",
        "prefetch_duration_ms",
        "prefetch_margin_ms",
        "resume_latency_ms",
        "direct_kv_h2d_events",
        "replay_kv_h2d_events",
    ]
    return f"""
  <details id="live-direct" class="section-card theme-profiled" open>
    <summary><h2>Live AgentBench Direct Prefetch</h2></summary>
    <p class="note">This section is the real live workload check. It uses only direct prefetch mode: real AgentBench/DeepAgents tool calls create live gaps, and the controller tries to trigger direct SGLang KV load-back during those gaps.</p>
    <h3>Live Summary</h3>
    {cards_html}
    <h3>How This Live Run Works</h3>
    {table_html(live_setup_rows, ["part", "simple meaning"])}
    <h3>Global Live Prefetch Margin</h3>
    <p>Positive margin means the live direct-prefetch path finished before the real agent resumed. Negative margin means the agent resumed first.</p>
    {global_prefetch_margin_html(gaps)}
    <h3>Live Direct-Prefetch Timeline</h3>
    <p class="note">Rows with green or cyan bars are shown first. Green is hint-side direct KV HtoD evidence; cyan is replay-side HtoD evidence from the real resume request.</p>
    {build_expanded_gap_timeline_svg(interesting, max_timeline_gaps, show_prefetch_legend=True, scale="symlog")}
    <h3>Live Row Map</h3>
    {table_html(timeline_mapping_rows(interesting))}
    <h3>Live Key Observations</h3>
    {table_html(key_observation_rows(interesting), ["row", "mode", "status", "what happened", "why it matters", "tool_wait_ms", "resume_ttft_ms"])}
    <h3>Live Direct KV Evidence</h3>
    {table_html(gaps, detail_columns, limit=200)}
  </details>
"""


def render_html(
    gaps: list[dict[str, Any]],
    result_root: Path,
    max_timeline_gaps: int,
    live_run: dict[str, Any] | None = None,
) -> str:
    mode_rows = mode_summary_rows(gaps)
    interesting = timeline_rows_with_labels(selected_timeline_gaps(gaps, max_timeline_gaps))
    gap_columns = [
        "session_id",
        "mode",
        "task_index",
        "tool_names",
        "tool_gap_ms",
        "prefetch_duration_ms",
        "prefetch_margin_ms",
        "resume_ttft_ms",
        "movement_class",
        "direct_kv_h2d_events",
        "replay_kv_h2d_events",
    ]
    toc = [
        ("summary", "Summary"),
        ("setup", "Experiment Setup"),
        ("global-prefetch", "Global Prefetch Margin"),
        ("timeline-guide", "How To Read Timelines"),
        ("timelines", "Controlled Replay Timeline"),
        ("live-direct", "Live Direct Prefetch"),
        ("observations", "Key Observations"),
        ("performance", "Mode Tables"),
        ("direct-kv", "Direct KV Evidence"),
        ("appendix", "Gap Details"),
        ("reproduce", "Reproduce This Report"),
    ]
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Real-Prompt Controlled Replay Master Report</title>
  <style>{master_css()}</style>
</head>
<body>
<main>
  <section>
    <h1>Real-Prompt Controlled Replay Master Report</h1>
    <p>This is the controlled real-prompt version of the master report. The prompts come from real SWE-bench / DeepAgents traces, while the experiment controls tool-wait time, cache pressure, and direct KV prefetch timing.</p>
    <p class="note">Use this report for the clean hardware-story experiment: real prompt content, known resume deadlines, known pressure levels, direct SGLang KV-hook attempts, and replay-side KV movement evidence.</p>
    <h2>Table of Contents</h2>
    {toc_html(toc)}
  </section>

  <details id="summary" class="section-card theme-summary" open>
    <summary><h2>Summary</h2></summary>
    <p>This section gives the headline numbers across no-prefetch and direct-prefetch modes. Oracle prefetch remains supported for optional sensitivity studies, but it is intentionally excluded from the default report to keep the main story simple.</p>
    {metric_cards_html(mode_rows)}
  </details>

  <details id="setup" class="section-card theme-setup">
    <summary><h2>Experiment Setup And Manager Summary</h2></summary>
    {manager_setup_html()}
  </details>

  <details id="global-prefetch" class="section-card theme-global">
    <summary><h2>Global Prefetch Margin</h2></summary>
    <p>Positive margin means the hint path finished before replay. Negative margin means replay arrived first.</p>
    {global_prefetch_margin_html(gaps)}
  </details>

  <details id="timeline-guide" class="section-card theme-guide">
    <summary><h2>How To Read The Timelines</h2></summary>
    {timeline_guide_html(profiled_available=True)}
  </details>

  <details id="timelines" class="section-card theme-clean">
    <summary><h2>Controlled Replay Timeline</h2></summary>
    <p class="note">Rows with green or cyan bars are shown first. Green is hint-side direct KV HtoD evidence. Cyan is replay-side HtoD evidence.</p>
    {build_expanded_gap_timeline_svg(interesting, max_timeline_gaps, show_prefetch_legend=True, scale="symlog")}
    <h3>Timeline Row Map</h3>
    <p>This table maps the compact row names in the chart back to the full experiment details.</p>
    {table_html(timeline_mapping_rows(interesting))}
  </details>

  {live_direct_prefetch_html(live_run, max_timeline_gaps)}

  <details id="observations" class="section-card theme-observations">
    <summary><h2>Key Observations Per Gap/Session</h2></summary>
    <p>This section translates the timeline rows into plain English. It uses the same compact row names as the chart, so <code>G00</code> here means the same <code>G00</code> in the timeline.</p>
    {table_html(key_observation_rows(interesting), ["row", "mode", "status", "what happened", "why it matters", "tool_wait_ms", "resume_ttft_ms"])}
  </details>

  <details id="performance" class="section-card theme-clean-table">
    <summary><h2>Mode Tables</h2></summary>
    <h3>Mode Summary</h3>
    {table_html(mode_rows)}
  </details>

  <details id="direct-kv" class="section-card theme-directkv">
    <summary><h2>Direct KV Load Evidence</h2></summary>
    <p>Green bars and <code>direct_kv_h2d_*</code> columns come from SGLang-level KV movement hooks and lightweight copy telemetry during the prefetch attempt. Cyan/replay columns show KV movement performed by the real resume request.</p>
    {table_html(gaps, ["session_id", "mode", "tool_gap_ms", "prefetch_margin_ms", "movement_class", "direct_kv_h2d_events", "direct_kv_h2d_duration_ms", "replay_kv_h2d_events", "replay_kv_h2d_duration_ms"])}
  </details>

  <details id="appendix" class="section-card theme-appendix">
    <summary><h2>Gap Details</h2></summary>
    {table_html(gaps, gap_columns)}
  </details>

  <details id="reproduce" class="section-card theme-reproduce">
    <summary><h2>Reproduce This Report</h2></summary>
    {reproduce_controlled_replay_html(result_root)}
  </details>
</main>
{report_script()}
</body>
</html>
"""


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_cases(root: Path) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir():
            continue
        trace = child / "m27_trace.jsonl"
        if trace.exists():
            mode = child.name.split("_tw", 1)[0]
            cases.append((mode, child))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Milestone 27 controlled replay report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--live-direct-root", type=Path)
    parser.add_argument("--max-timeline-gaps", type=int, default=18)
    args = parser.parse_args()

    all_gaps: list[dict[str, Any]] = []
    for mode, case_dir in discover_cases(args.root):
        gaps, _ = build_gaps_for_case(case_dir, mode)
        for gap in gaps:
            gap["case_dir"] = str(case_dir)
        all_gaps.extend(gaps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "controlled_replay_gaps.csv", all_gaps)
    write_json(args.out_dir / "controlled_replay_report.json", {"gaps": all_gaps, "summary": mode_summary_rows(all_gaps)})
    live_run = None
    if args.live_direct_root:
        live_run = load_live_agentbench_run(args.live_direct_root, "live_direct_prefetch", include_preflight=False)

    html_text = render_html(all_gaps, args.root, args.max_timeline_gaps, live_run=live_run)
    report_path = args.out_dir / "controlled_replay_report.html"
    report_path.write_text(html_text, encoding="utf-8")

    if args.latest_root:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, args.latest_root / "latest_controlled_replay_report.html")
        shutil.copy2(report_path, args.latest_root / "latest_master_report.html")

    print(f"Wrote Milestone 27 report to {report_path}")


if __name__ == "__main__":
    main()
