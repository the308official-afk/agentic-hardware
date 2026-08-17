#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_live_agentbench_tool_gap_report import (
    augment_gaps_with_prefetch,
    build_tool_gaps,
    is_preflight_request,
    normalize_requests,
    read_csv,
    read_jsonl,
    table_html,
    write_csv,
)


DIRECT_EVENTS = {
    "hiradix.init_load_back.end": "init_load_back",
    "hiradix.load_back.end": "load_back",
    "hicache.load.end": "hicache_load",
    "hicache.start_loading.end": "hicache_start_loading",
    "hostpool.load_to_device_per_layer.end": "hostpool_h2d",
}


def maybe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    return "" if value is None else html.escape(str(value))


def context_from_event(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("kv_context")
    if isinstance(context, dict):
        return context
    return row


def agent_session_from_context(context: dict[str, Any]) -> str:
    for key in ("agent_session_id", "session_id"):
        value = context.get(key)
        if value not in ("", None):
            return str(value)
    req = context.get("request")
    if isinstance(req, dict):
        for key in ("agent_session_id", "session_id"):
            value = req.get(key)
            if value not in ("", None):
                return str(value)
    sessions = context.get("agent_sessions")
    if isinstance(sessions, list):
        for item in sessions:
            if isinstance(item, dict):
                value = item.get("agent_session_id")
                if value not in ("", None):
                    return str(value)
    return ""


def hint_id_from_session(session_id: str) -> str:
    marker = "::live_prefetch::"
    if marker not in session_id:
        return ""
    return session_id.split(marker, 1)[1]


def relative_ms_from_ns(ts_ns: Any, base_ts: float) -> float | str:
    value = maybe_float(ts_ns)
    if value is None:
        return ""
    return round((value / 1_000_000_000.0 - base_ts) * 1000.0, 3)


def direct_trace_events(trace_rows: list[dict[str, Any]], base_ts: float) -> dict[str, list[dict[str, Any]]]:
    by_hint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        event = str(row.get("event") or "")
        if event not in DIRECT_EVENTS:
            continue
        context = context_from_event(row)
        session_id = agent_session_from_context(context)
        hint_id = hint_id_from_session(session_id)
        if not hint_id:
            continue
        by_hint[hint_id].append(
            {
                "hint_id": hint_id,
                "source": "sglang_trace",
                "event": event,
                "category": DIRECT_EVENTS[event],
                "agent_session_id": session_id,
                "direction": context.get("direction", ""),
                "duration_ms": row.get("duration_ms", ""),
                "start_or_end_ms": relative_ms_from_ns(row.get("ts_ns"), base_ts),
                "host_indices": json.dumps(context.get("host_indices", {}), sort_keys=True)[:240],
                "device_indices": json.dumps(context.get("device_indices", {}), sort_keys=True)[:240],
            }
        )
    return by_hint


def telemetry_events(telemetry_rows: list[dict[str, Any]], base_ts: float) -> dict[str, list[dict[str, Any]]]:
    by_hint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in telemetry_rows:
        event = str(row.get("event") or "")
        if event not in {"kv_telemetry.copy.end", "kv_telemetry.copy.start"}:
            continue
        direction = str(row.get("direction") or "")
        if direction != "host_to_device":
            continue
        session_id = str(row.get("agent_session_id") or "")
        hint_id = hint_id_from_session(session_id)
        if not hint_id:
            continue
        by_hint[hint_id].append(
            {
                "hint_id": hint_id,
                "source": "kv_copy_telemetry",
                "event": event,
                "category": "telemetry_h2d",
                "agent_session_id": session_id,
                "direction": direction,
                "duration_ms": row.get("duration_ms", ""),
                "start_or_end_ms": relative_ms_from_ns(row.get("ts_ns"), base_ts),
                "host_indices": str(row.get("host_index_count") or ""),
                "device_indices": str(row.get("device_index_count") or ""),
            }
        )
    return by_hint


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
            merged = dict(starts.get(hint_id, {}))
            merged.update(row)
            out[hint_id] = merged
    return out


def build_gap_rows(
    proxy_rows: list[dict[str, Any]],
    hint_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    include_preflight: bool,
) -> tuple[list[dict[str, Any]], float]:
    all_requests = normalize_requests(proxy_rows)
    requests = all_requests if include_preflight else [row for row in all_requests if not is_preflight_request(row)]
    gaps = build_tool_gaps(requests)
    base_ts = min((float(row["start_ts"]) for row in all_requests), default=0.0)
    return augment_gaps_with_prefetch(gaps, hint_rows, controller_rows, base_ts), base_ts


def evidence_rows(
    hint_rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    trace_by_hint: dict[str, list[dict[str, Any]]],
    telemetry_by_hint: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    gaps_by_hint = {str(row.get("hint_id") or ""): row for row in gaps if row.get("hint_id")}
    windows = controller_windows(controller_rows)
    rows: list[dict[str, Any]] = []
    for hint in hint_rows:
        if hint.get("event") != "live_hint.submitted":
            continue
        hint_id = str(hint.get("hint_id") or "")
        window = windows.get(hint_id, {})
        gap = gaps_by_hint.get(hint_id, {})
        trace_events = trace_by_hint.get(hint_id, [])
        telemetry = telemetry_by_hint.get(hint_id, [])
        counts = defaultdict(int)
        for item in [*trace_events, *telemetry]:
            counts[str(item.get("category") or "")] += 1
        direct_load_observed = any(
            counts[key] > 0
            for key in ("init_load_back", "load_back", "hicache_load", "hostpool_h2d", "telemetry_h2d")
        )
        rows.append(
            {
                "hint_id": hint_id,
                "source_proxy_ordinal": hint.get("source_proxy_ordinal", ""),
                "task_instance_id": hint.get("source_task_instance_id", ""),
                "tool_names": ",".join(str(item) for item in (hint.get("tool_names") or [])),
                "controller_action": window.get("prefetch_action", ""),
                "trigger_injected": window.get("direct_load_trigger_injected", ""),
                "controller_status": "error" if window.get("event") == "live_prefetch.error" else "done" if window else "not_finished",
                "controller_duration_ms": window.get("duration_ms", ""),
                "prefetch_margin_ms": gap.get("prefetch_margin_ms", ""),
                "init_load_back_events": counts["init_load_back"],
                "load_back_events": counts["load_back"],
                "hicache_load_events": counts["hicache_load"],
                "hostpool_h2d_events": counts["hostpool_h2d"],
                "telemetry_h2d_events": counts["telemetry_h2d"],
                "direct_kv_load_observed": "yes" if direct_load_observed else "no",
                "interpretation": (
                    "SGLang direct load-back/copy evidence observed"
                    if direct_load_observed
                    else "hint request ran, but no matching load-back/copy event was attributed"
                ),
            }
        )
    return rows


def summary_rows(rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], telemetry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    observed = sum(1 for row in rows if row.get("direct_kv_load_observed") == "yes")
    late_values = [maybe_float(row.get("prefetch_margin_ms")) for row in rows]
    late_values = [value for value in late_values if value is not None and value < 0]
    return [
        {
            "live_hints": total,
            "hints_with_direct_kv_load_evidence": observed,
            "hints_without_attributed_load": total - observed,
            "init_load_back_hints": sum(1 for row in rows if int(row.get("init_load_back_events") or 0) > 0),
            "load_back_hints": sum(1 for row in rows if int(row.get("load_back_events") or 0) > 0),
            "h2d_copy_hints": sum(
                1
                for row in rows
                if int(row.get("hostpool_h2d_events") or 0) > 0 or int(row.get("telemetry_h2d_events") or 0) > 0
            ),
            "late_prefetch_hints": len(late_values),
            "trace_events_total": len(trace_rows),
            "copy_telemetry_events_total": len(telemetry_rows),
        }
    ]


def render_html(summary: list[dict[str, Any]], evidence: list[dict[str, Any]], event_rows: list[dict[str, Any]]) -> str:
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
    main { max-width: 1760px; margin: 0 auto; padding: 24px; }
    section { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px 24px; margin: 18px 0; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
    h1 { font-size: 34px; margin: 0 0 8px; }
    h2 { font-size: 24px; margin: 0 0 12px; }
    p { color: #334155; line-height: 1.45; }
    .note { background: #ecfeff; border-left: 4px solid #0891b2; padding: 12px 14px; color: #164e63; }
    """
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Direct KV Load Evidence</title>
  <style>{css}</style>
</head>
<body>
<main>
  <section>
    <h1>Live Direct KV Load Evidence</h1>
    <p>This report checks whether live DeepAgents tool-call hints triggered SGLang's direct KV load-back path.</p>
    <p class="note">Important: the controller is outside the SGLang worker process. It cannot directly call an in-process Python cache object. Instead it sends a marked direct-load request, then the SGLang hooks verify whether the real <code>init_load_back</code>, <code>load_back</code>, <code>hicache.load</code>, or host-to-device copy telemetry fired.</p>
  </section>
  <section>
    <h2>Summary</h2>
    {table_html(summary)}
  </section>
  <section>
    <h2>Per-Hint Direct KV Evidence</h2>
    {table_html(evidence)}
  </section>
  <section>
    <h2>Matched SGLang KV / Copy Events</h2>
    {table_html(event_rows, limit=200)}
  </section>
</main>
</body>
</html>
"""


def render_markdown(summary: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> str:
    lines = [
        "# Live Direct KV Load Evidence",
        "",
        "This report checks whether live DeepAgents tool-call hints triggered SGLang's direct KV load-back path.",
        "",
        "Important: the controller is outside the SGLang worker process. It sends a marked direct-load request, then the SGLang hooks verify whether `init_load_back`, `load_back`, `hicache.load`, or host-to-device copy telemetry fired.",
        "",
        "## Summary",
        "",
    ]
    for key, value in (summary[0] if summary else {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Per-Hint Evidence", ""])
    lines.append("| hint_id | action | margin_ms | direct_kv_load_observed | init_load_back | load_back | h2d_copy | interpretation |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in evidence:
        h2d = int(row.get("hostpool_h2d_events") or 0) + int(row.get("telemetry_h2d_events") or 0)
        lines.append(
            f"| {row.get('hint_id','')} | {row.get('controller_action','')} | {row.get('prefetch_margin_ms','')} | "
            f"{row.get('direct_kv_load_observed','')} | {row.get('init_load_back_events','')} | "
            f"{row.get('load_back_events','')} | {h2d} | {row.get('interpretation','')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build live direct KV load evidence from SGLang hooks.")
    parser.add_argument("--proxy-jsonl", type=Path, required=True)
    parser.add_argument("--hint-log", type=Path, required=True)
    parser.add_argument("--controller-log", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--copy-telemetry", type=Path)
    parser.add_argument("--task-index-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--include-preflight", action="store_true")
    args = parser.parse_args()

    proxy_rows = read_jsonl(args.proxy_jsonl)
    hint_rows = read_jsonl(args.hint_log)
    controller_rows = read_jsonl(args.controller_log)
    trace_rows = read_jsonl(args.trace)
    telemetry_rows = read_jsonl(args.copy_telemetry) if args.copy_telemetry else []
    read_csv(args.task_index_csv)

    gaps, base_ts = build_gap_rows(proxy_rows, hint_rows, controller_rows, args.include_preflight)
    trace_by_hint = direct_trace_events(trace_rows, base_ts)
    telemetry_by_hint = telemetry_events(telemetry_rows, base_ts)
    evidence = evidence_rows(hint_rows, controller_rows, gaps, trace_by_hint, telemetry_by_hint)
    event_rows = [item for values in trace_by_hint.values() for item in values] + [
        item for values in telemetry_by_hint.values() for item in values
    ]
    summary = summary_rows(evidence, trace_rows, telemetry_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "live_direct_kv_load_summary.csv", summary)
    write_csv(args.out_dir / "live_direct_kv_load_evidence.csv", evidence)
    write_csv(args.out_dir / "live_direct_kv_load_events.csv", event_rows)
    report_json = {
        "summary": summary[0] if summary else {},
        "evidence": evidence,
        "events": event_rows,
        "inputs": {
            "proxy_jsonl": str(args.proxy_jsonl),
            "hint_log": str(args.hint_log),
            "controller_log": str(args.controller_log),
            "trace": str(args.trace),
            "copy_telemetry": str(args.copy_telemetry or ""),
        },
    }
    (args.out_dir / "live_direct_kv_load_report.json").write_text(json.dumps(report_json, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "live_direct_kv_load_report.md").write_text(render_markdown(summary, evidence), encoding="utf-8")
    (args.out_dir / "live_direct_kv_load_report.html").write_text(render_html(summary, evidence, event_rows), encoding="utf-8")

    if args.latest_root:
        latest_real = args.latest_root / "latest_real"
        latest_real.mkdir(parents=True, exist_ok=True)
        for source, target in (
            (args.out_dir / "live_direct_kv_load_report.html", "m26_live_direct_kv_load_report.html"),
            (args.out_dir / "live_direct_kv_load_report.md", "m26_live_direct_kv_load_report.md"),
            (args.out_dir / "live_direct_kv_load_report.json", "m26_live_direct_kv_load_report.json"),
            (args.out_dir / "live_direct_kv_load_summary.csv", "m26_live_direct_kv_load_summary.csv"),
            (args.out_dir / "live_direct_kv_load_evidence.csv", "m26_live_direct_kv_load_evidence.csv"),
            (args.out_dir / "live_direct_kv_load_events.csv", "m26_live_direct_kv_load_events.csv"),
        ):
            if source.exists():
                shutil.copyfile(source, latest_real / target)

    print(f"Wrote live direct KV load evidence report: {args.out_dir / 'live_direct_kv_load_report.html'}")
    print(f"Live hints: {summary[0].get('live_hints', 0) if summary else 0}")
    print(f"Hints with direct KV evidence: {summary[0].get('hints_with_direct_kv_load_evidence', 0) if summary else 0}")


if __name__ == "__main__":
    main()
