#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def ms_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() * 1000.0, 3)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def short(value: Any, limit: int = 180) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def first_event(events: list[dict[str, Any]], stage: str, phase: str | None = None) -> dict[str, Any] | None:
    for event in events:
        if event.get("stage") != stage:
            continue
        if phase is not None and event.get("phase") != phase:
            continue
        return event
    return None


def pair_phase_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = [
        event
        for event in events
        if str(event.get("stage", "")).endswith("_request_dispatched")
        and event.get("event_kind") == "request_dispatch"
    ]
    ends: list[dict[str, Any]] = [
        event
        for event in events
        if str(event.get("stage", "")).endswith("_response_received")
        and event.get("event_kind") == "response"
    ]
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for start in starts:
        phase = str(start.get("phase") or "unknown")
        start_ts = parse_ts(start.get("timestamp"))
        candidates = [
            (idx, end)
            for idx, end in enumerate(ends)
            if idx not in used
            and str(end.get("phase") or "unknown") == phase
            and (parse_ts(end.get("timestamp")) or datetime.max.replace(tzinfo=start_ts.tzinfo if start_ts else None))
            >= (start_ts or datetime.min.replace(tzinfo=None))
        ]
        if not candidates:
            continue
        idx, end = candidates[0]
        used.add(idx)
        end_ts = parse_ts(end.get("timestamp"))
        measurement = end.get("measurement") if isinstance(end.get("measurement"), dict) else {}
        prompt = start.get("prompt") or ""
        request_context = start.get("request_context") if isinstance(start.get("request_context"), dict) else {}
        pairs.append(
            {
                "phase": phase,
                "sequence_index": start.get("sequence_index", ""),
                "request_id": request_context.get("request_id", ""),
                "start": start,
                "end": end,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_ms": ms_between(start_ts, end_ts),
                "prompt_chars": start.get("prompt_chars", len(str(prompt))),
                "prompt_lines": start.get("prompt_lines", ""),
                "latency_ms": measurement.get("latency_ms", ""),
                "ttft_ms": measurement.get("ttft_ms", ""),
                "input_tokens": measurement.get("input_tokens", ""),
                "output_tokens": measurement.get("output_tokens", ""),
                "tool_call_count": (end.get("tool_progress") or {}).get("tool_call_count", ""),
                "tool_call_names": ", ".join((end.get("tool_progress") or {}).get("tool_call_names") or []),
                "prompt": prompt,
            }
        )
    return pairs


def run_rows(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in index_rows:
        run_dir = Path(str(item.get("result_dir", "")))
        result = read_json(run_dir / "others" / "result.json") or {}
        lifecycle = read_json(run_dir / "others" / "stage_lifecycle_trace_raw.json") or []
        if not isinstance(lifecycle, list):
            lifecycle = []
        phase_pairs = pair_phase_events(lifecycle)
        tool_calls = sum(int(float(row.get("tool_call_count") or 0)) for row in phase_pairs)
        total_latency = sum(float(row.get("latency_ms") or 0) for row in phase_pairs if row.get("latency_ms") not in ("", None))
        rows.append(
            {
                "task_index": item.get("task_index", ""),
                "run_id": item.get("run_id", run_dir.name),
                "repo": item.get("repo", result.get("task", {}).get("repo", "")),
                "instance_id": result.get("task", {}).get("instance_id", ""),
                "model": result.get("model", ""),
                "frontend_url": result.get("frontend_url", ""),
                "phase_model_turns": len(phase_pairs),
                "tool_calls_seen": tool_calls,
                "total_model_latency_ms": round(total_latency, 3),
                "result_dir": str(run_dir),
                "prompt_evolution_report": str(run_dir / "prompt_evolution_report.md"),
                "tool_call_details": str(Path(str(item.get("report_dir", ""))) / "tool_call_details.md")
                if item.get("report_dir")
                else "",
            }
        )
    return rows


def phase_rows(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in index_rows:
        run_dir = Path(str(item.get("result_dir", "")))
        lifecycle = read_json(run_dir / "others" / "stage_lifecycle_trace_raw.json") or []
        if not isinstance(lifecycle, list):
            continue
        for pair in pair_phase_events(lifecycle):
            rows.append(
                {
                    "task_index": item.get("task_index", ""),
                    "run_id": item.get("run_id", run_dir.name),
                    "phase": pair["phase"],
                    "request_id": pair.get("request_id", ""),
                    "duration_ms": pair.get("duration_ms", ""),
                    "reported_latency_ms": pair.get("latency_ms", ""),
                    "ttft_ms": pair.get("ttft_ms", ""),
                    "prompt_chars": pair.get("prompt_chars", ""),
                    "input_tokens": pair.get("input_tokens", ""),
                    "output_tokens": pair.get("output_tokens", ""),
                    "tool_call_count": pair.get("tool_call_count", ""),
                    "tool_call_names": pair.get("tool_call_names", ""),
                }
            )
    return rows


def kv_summary(trace_path: Path, copy_path: Path | None) -> list[dict[str, Any]]:
    trace = read_jsonl(trace_path)
    copies = read_jsonl(copy_path) if copy_path else []
    event_counts = Counter(str(row.get("event", "")) for row in trace)
    agent_context_rows = [
        row
        for row in trace
        if row.get("agent_session_id") or row.get("agent_sessions") or (isinstance(row.get("request"), dict) and row["request"].get("agent_session_id"))
    ]
    h2d_events = [
        row
        for row in trace
        if row.get("direction") == "host_to_device"
        or (isinstance(row.get("context"), dict) and row["context"].get("direction") == "host_to_device")
    ]
    copy_h2d = [row for row in copies if row.get("direction") == "host_to_device"]
    return [
        {
            "trace_file": str(trace_path),
            "trace_events": len(trace),
            "unique_event_names": len(event_counts),
            "top_events": ", ".join(f"{name}: {count}" for name, count in event_counts.most_common(8)),
            "sglang_h2d_like_events": len(h2d_events),
            "trace_events_with_agent_context": len(agent_context_rows),
            "copy_telemetry_file": str(copy_path or ""),
            "copy_telemetry_h2d_events": len(copy_h2d),
        }
    ]


def build_phase_timeline_svg(rows: list[dict[str, Any]], index_rows: list[dict[str, Any]]) -> str:
    spans: list[dict[str, Any]] = []
    for item in index_rows:
        run_id = str(item.get("run_id", ""))
        lifecycle = read_json(Path(str(item.get("result_dir", ""))) / "others" / "stage_lifecycle_trace_raw.json") or []
        if not isinstance(lifecycle, list):
            continue
        pairs = pair_phase_events(lifecycle)
        if not pairs:
            continue
        base_ts = pairs[0]["start_ts"]
        for pair in pairs:
            start_ms = ms_between(base_ts, pair["start_ts"])
            end_ms = ms_between(base_ts, pair["end_ts"])
            if start_ms is None or end_ms is None:
                continue
            spans.append(
                {
                    "run_id": run_id,
                    "phase": pair["phase"],
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "label": pair["phase"],
                }
            )
    if not spans:
        return "<p>No phase timeline available.</p>"

    selected_runs = []
    for span in spans:
        run_id = span["run_id"]
        if run_id not in selected_runs:
            selected_runs.append(run_id)
        if len(selected_runs) >= 6:
            break
    spans = [span for span in spans if span["run_id"] in selected_runs]
    start = min(float(span["start_ms"]) for span in spans)
    end = max(float(span["end_ms"]) for span in spans)
    span_ms = max(1.0, end - start)
    width = 1500
    left = 250
    right = 50
    top = 70
    row_h = 54
    height = top + len(selected_runs) * row_h + 72
    plot_w = width - left - right
    colors = {
        "planning": "#2563eb",
        "execution": "#a855f7",
        "patch_generation": "#f59e0b",
        "review": "#16a34a",
        "baseline_execution": "#64748b",
    }

    def x_pos(ms: float) -> float:
        return left + (ms - start) / span_ms * plot_w

    run_index = {run_id: idx for idx, run_id in enumerate(selected_runs)}
    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="AgentBench direct SGLang phase timeline">',
        f'<line x1="{left}" y1="{top - 24}" x2="{left + plot_w}" y2="{top - 24}" stroke="#111827"/>',
    ]
    for tick in range(6):
        ms = start + span_ms * tick / 5
        x = x_pos(ms)
        svg.append(f'<line x1="{x:.1f}" y1="{top - 30}" x2="{x:.1f}" y2="{height - 32}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{x:.1f}" y="{top - 38}" text-anchor="middle">{ms:.0f} ms</text>')
    for run_id in selected_runs:
        y = top + run_index[run_id] * row_h
        svg.append(f'<text x="10" y="{y + 18}" font-weight="700">{html.escape(run_id)}</text>')
        svg.append(f'<line x1="{left}" y1="{y + 8}" x2="{left + plot_w}" y2="{y + 8}" stroke="#f3f4f6"/>')
    for span in spans:
        y = top + run_index[span["run_id"]] * row_h
        x1 = x_pos(float(span["start_ms"]))
        x2 = x_pos(float(span["end_ms"]))
        phase = str(span["phase"])
        color = colors.get(phase, "#64748b")
        svg.append(
            f'<rect x="{x1:.1f}" y="{y}" width="{max(3, x2 - x1):.1f}" height="24" rx="3" '
            f'fill="{color}" opacity="0.82"><title>{html.escape(phase)}</title></rect>'
        )
        if x2 - x1 > 65:
            svg.append(
                f'<text x="{(x1 + x2) / 2:.1f}" y="{y + 16}" text-anchor="middle" '
                f'font-size="11" fill="white" font-weight="700">{html.escape(phase)}</text>'
            )
    legend_y = height - 28
    for idx, (phase, color) in enumerate(colors.items()):
        lx = left + idx * 170
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 20}" y="{legend_y + 12}">{html.escape(phase)}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


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
            cls = ' class="wrap"' if header in {"result_dir", "prompt_evolution_report", "tool_call_details", "top_events"} else ""
            out.append(f"<td{cls}>{fmt(row.get(header, ''))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


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
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    lines.append("")
    return lines


def write_outputs(
    *,
    out_root: Path,
    latest_root: Path | None,
    index_rows: list[dict[str, Any]],
    trace_path: Path,
    copy_path: Path | None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    runs = run_rows(index_rows)
    phases = phase_rows(index_rows)
    kv = kv_summary(trace_path, copy_path)
    write_csv(out_root / "agentbench_sglang_runs.csv", runs)
    write_csv(out_root / "agentbench_sglang_phase_turns.csv", phases)
    write_csv(out_root / "agentbench_sglang_kv_summary.csv", kv)

    timeline_svg = build_phase_timeline_svg(phases, index_rows)
    html_lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>AgentBench Direct SGLang Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}",
        "h1,h2{margin:0 0 12px}",
        ".panel{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin:16px 0}",
        ".caption{margin:0 0 12px;color:#374151;line-height:1.45}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;white-space:nowrap;vertical-align:top}",
        "th{background:#f3f4f6;font-weight:700}",
        ".wrap{white-space:normal;line-height:1.35;min-width:260px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>AgentBench Direct SGLang Report</h1>",
        '<div class="panel"><h2>What This Proves</h2>',
        '<p class="caption">This run uses real SWE-bench tasks, the existing AgentBench/DeepAgents harness, and a direct SGLang OpenAI-compatible endpoint. Dynamo is removed. The runtime path is SWE-bench Pro -> AgentBench -> Deep Agents -> SGLang -> direct KV trace/reporting.</p>',
        "</div>",
        '<div class="panel"><h2>Run Summary</h2>',
        html_table(runs),
        "</div>",
        '<div class="panel"><h2>Model-Turn Timeline</h2>',
        '<p class="caption">This phase-level timeline shows real DeepAgents model turns. Tool calls happen inside these turns; the next milestone extracts these prompts into replayable sessions so we can compare no-prefetch, request-warm, direct-load, and oracle-direct-load under controlled timing.</p>',
        timeline_svg,
        "</div>",
        '<div class="panel"><h2>SGLang KV Trace Summary</h2>',
        html_table(kv),
        "</div>",
        '<div class="panel"><h2>Phase-Level Model Turns</h2>',
        html_table(phases),
        "</div>",
        '<div class="panel"><h2>Important Interpretation</h2>',
        '<p class="caption">This live path is the realism proof: real agent, real tools, real prompts, real SGLang. It is not yet the controlled prefetch comparison. For the controlled comparison, use the trace-replay workload generated from these real AgentBench prompts.</p>',
        "</div>",
        "</body></html>",
    ]
    html_path = out_root / "agentbench_sglang_direct_report.html"
    md_path = out_root / "agentbench_sglang_direct_report.md"
    html_path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# AgentBench Direct SGLang Report",
        "",
        "Runtime path:",
        "",
        "```text",
        "SWE-bench Pro -> AgentBench -> Deep Agents -> SGLang direct -> KV trace/reporting",
        "```",
        "",
        "Dynamo is intentionally removed in this experiment.",
        "",
        "## Run Summary",
        "",
        *md_table(runs),
        "## SGLang KV Trace Summary",
        "",
        *md_table(kv),
        "## Phase-Level Model Turns",
        "",
        *md_table(phases),
        "## Interpretation",
        "",
        "This live path proves the realistic harness is connected to SGLang directly.",
        "Use trace replay for controlled prefetch comparisons because DeepAgents owns the internal tool loop timing.",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "out_root": str(out_root),
        "html": str(html_path),
        "markdown": str(md_path),
        "run_csv": str(out_root / "agentbench_sglang_runs.csv"),
        "phase_csv": str(out_root / "agentbench_sglang_phase_turns.csv"),
        "kv_summary_csv": str(out_root / "agentbench_sglang_kv_summary.csv"),
        "trace_path": str(trace_path),
        "copy_telemetry_path": str(copy_path or ""),
    }
    (out_root / "agentbench_sglang_direct_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if latest_root is not None:
        latest_root.mkdir(parents=True, exist_ok=True)
        for src, name in [
            (html_path, "latest_agentbench_sglang_direct_report.html"),
            (md_path, "latest_agentbench_sglang_direct_report.md"),
            (out_root / "agentbench_sglang_direct_report.json", "latest_agentbench_sglang_direct_report.json"),
            (out_root / "agentbench_sglang_runs.csv", "latest_agentbench_sglang_runs.csv"),
            (out_root / "agentbench_sglang_phase_turns.csv", "latest_agentbench_sglang_phase_turns.csv"),
        ]:
            if src.exists():
                shutil.copyfile(src, latest_root / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize direct-SGLang AgentBench runs.")
    parser.add_argument("--index-csv", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--copy-telemetry", type=Path, default=None)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--latest-root", type=Path, default=None)
    args = parser.parse_args()
    index_rows = read_csv(args.index_csv)
    write_outputs(
        out_root=args.out_root,
        latest_root=args.latest_root,
        index_rows=index_rows,
        trace_path=args.trace,
        copy_path=args.copy_telemetry,
    )
    print(f"Wrote AgentBench direct SGLang report to {args.out_root}")


if __name__ == "__main__":
    main()
