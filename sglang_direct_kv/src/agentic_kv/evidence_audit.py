from __future__ import annotations

from collections import Counter
from typing import Any

from agentic_kv.evidence_schema import movement_kind_display, movement_kind_from_row


EvidenceRows = list[dict[str, Any]]


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def pct(count: int, total: int) -> float:
    return round(count * 100.0 / total, 2) if total else 0.0


def has_value(row: dict[str, Any], key: str) -> bool:
    return row.get(key) not in ("", None, [], {})


def has_any(row: dict[str, Any], keys: list[str]) -> bool:
    return any(has_value(row, key) for key in keys)


def count_rows_with(rows: EvidenceRows, key: str) -> int:
    return sum(1 for row in rows if has_value(row, key))


def count_rows_with_any(rows: EvidenceRows, keys: list[str]) -> int:
    return sum(1 for row in rows if has_any(row, keys))


def coverage_text(count: int, total: int) -> str:
    return f"{count}/{total} ({pct(count, total)}%)"


def chart_inventory_rows() -> EvidenceRows:
    return [
        {
            "report_item": "Readable KV Lifecycle Timeline: initial turn",
            "visual_element": "blue bar",
            "source_artifact": "controlled_replay_gaps.csv / controlled_replay_report.json:gaps",
            "raw_hook_or_source": "driver timing events",
            "identity_carried": "session_id, task_index, gap_order",
            "evidence_level": "DIRECT",
            "audit_rule": "current_start_ms and current_end_ms exist",
            "limitation": "driver-observed request timing, not GPU work",
        },
        {
            "report_item": "Readable KV Lifecycle Timeline: tool wait",
            "visual_element": "gray bar",
            "source_artifact": "controlled_replay_gaps.csv / controlled_replay_report.json:gaps",
            "raw_hook_or_source": "driver timing events",
            "identity_carried": "session_id, task_index, gap_order",
            "evidence_level": "DERIVED",
            "audit_rule": "tool_gap_start_ms and tool_gap_end_ms exist",
            "limitation": "computed wait window, not memory movement",
        },
        {
            "report_item": "Readable KV Lifecycle Timeline: prefetch attempt",
            "visual_element": "purple bar",
            "source_artifact": "controlled_replay_gaps.csv / controlled_replay_report.json:gaps",
            "raw_hook_or_source": "driver/controller timing events",
            "identity_carried": "session_id, mode, phase",
            "evidence_level": "DIRECT",
            "audit_rule": "prefetch_start_ms and prefetch_end_ms exist for prefetch modes",
            "limitation": "attempt window can include queueing and bookkeeping, not only copy",
        },
        {
            "report_item": "Readable KV Lifecycle Timeline: hint-side KV H2D",
            "visual_element": "green bar",
            "source_artifact": "exact_kv_movement_attribution.csv",
            "raw_hook_or_source": "hostpool.load_to_device_per_layer.end / hicache.load.end / hiradix load-back hooks",
            "identity_carried": "session_id, phase, request_id, node_id, host/device index signatures",
            "evidence_level": "DIRECT",
            "audit_rule": "phase=hint_prefetch, direction=host_to_device, copy_start_ms/end_ms exist",
            "limitation": "SGLang-visible copy window, not raw DMA queue occupancy",
        },
        {
            "report_item": "Readable KV Lifecycle Timeline: replay-side KV H2D",
            "visual_element": "cyan bar",
            "source_artifact": "exact_kv_movement_attribution.csv",
            "raw_hook_or_source": "hostpool.load_to_device_per_layer.end / hicache.load.end / hiradix load-back hooks",
            "identity_carried": "session_id, phase, request_id, node_id, host/device index signatures",
            "evidence_level": "DIRECT",
            "audit_rule": "phase=replay, direction=host_to_device, copy_start_ms/end_ms exist",
            "limitation": "SGLang-visible copy window, not raw DMA queue occupancy",
        },
        {
            "report_item": "Readable KV Lifecycle Timeline: recompute/rebuild",
            "visual_element": "magenta bar",
            "source_artifact": "replay_path_ledger.csv / gaps fields",
            "raw_hook_or_source": "cache counters, prefix-match counters, TTFT/model-forward evidence",
            "identity_carried": "session_id, request_id when available",
            "evidence_level": "INFERRED",
            "audit_rule": "replay_new_prefill_tokens_est or recomputed_tokens_est exists",
            "limitation": "not a physical per-token recompute hook yet",
        },
        {
            "report_item": "Global Replay H2D Readiness",
            "visual_element": "readiness dots and request/H2D stage markers",
            "source_artifact": "replay_h2d_readiness.csv, replay_queue_timing.csv",
            "raw_hook_or_source": "driver timestamps plus SGLang request-stage and H2D hooks",
            "identity_carried": "row, session_id, case_id",
            "evidence_level": "DIRECT/DERIVED",
            "audit_rule": "H2D start/end direct; margins derived",
            "limitation": "only rows with replay-side H2D have readiness dots",
        },
        {
            "report_item": "KV H2D Bandwidth Pressure",
            "visual_element": "H2D activity windows",
            "source_artifact": "h2d_activity_events.csv, h2d_activity_windows.csv",
            "raw_hook_or_source": "aligned exact H2D movement events",
            "identity_carried": "row, session_id, case_id, block_key",
            "evidence_level": "DIRECT/DERIVED",
            "audit_rule": "H2D events direct; windows and concurrency derived",
            "limitation": "H2D-only view unless paired with Client Dispatch KV Movement",
        },
        {
            "report_item": "Client Dispatch KV Movement",
            "visual_element": "H2D/D2H/GPU-evict bars inside blue dispatch window",
            "source_artifact": "all_aligned_kv_movement_events.csv, client_dispatch_kv_movement_events.csv",
            "raw_hook_or_source": "hostpool/hicache H2D, D2H/write-host, GPU eviction hooks",
            "identity_carried": "owner row/session, case_id, phase, source_event, indices",
            "evidence_level": "DIRECT/DERIVED",
            "audit_rule": "movement event direct; dispatch-window overlap derived",
            "limitation": "SGLang-visible KV movement, not full hardware DMA saturation",
        },
        {
            "report_item": "Replay Delay Breakdown",
            "visual_element": "delay waterfall",
            "source_artifact": "replay_delay_breakdown.csv, replay_delay_stage_trace.csv",
            "raw_hook_or_source": "driver timestamps plus SGLang request-stage hooks",
            "identity_carried": "row, session_id, case_id, request_id where available",
            "evidence_level": "DIRECT/DERIVED/INFERRED",
            "audit_rule": "stage timestamps direct; segment durations derived; recompute/cache-first may be inferred",
            "limitation": "client dispatch says replay was not inside SGLang yet; it does not prove DMA saturation",
        },
        {
            "report_item": "Detailed KV Block Lifecycle Table",
            "visual_element": "per-block lifecycle rows",
            "source_artifact": "kv_block_ledger.csv, kv_block_lifecycle_by_gap.csv",
            "raw_hook_or_source": "normalized KV_WRITE_HOST/EVICT_GPU/EVICT_HOST/LOAD_GPU events",
            "identity_carried": "block_id, session_id, node_id, host/device index signatures",
            "evidence_level": "DIRECT/DERIVED",
            "audit_rule": "block_id exists; source movement events carry copy windows and identities",
            "limitation": "block_id is stable logical identity, not a hardware page table ID",
        },
        {
            "report_item": "Hardware DMA saturation claim",
            "visual_element": "not directly plotted",
            "source_artifact": "requires CUPTI/Nsight or lower-level copy-engine telemetry",
            "raw_hook_or_source": "not currently available in master report",
            "identity_carried": "none",
            "evidence_level": "NOT_YET_PROVEN",
            "audit_rule": "must not be claimed from SGLang-visible movement alone",
            "limitation": "current report can motivate, but not prove, DMA-engine saturation",
        },
    ]


def artifact_inventory_rows() -> EvidenceRows:
    return [
        {
            "artifact": "m27_trace.jsonl",
            "role": "raw trace",
            "contains": "SGLang hook events, request-stage events, cache/memory movement context",
            "evidence_strength": "strongest software-visible source",
        },
        {
            "artifact": "controlled_replay_report.json",
            "role": "report bundle",
            "contains": "gaps, exact movement rows, ledger rows, delay rows, audit rows",
            "evidence_strength": "structured report source",
        },
        {
            "artifact": "exact_kv_movement_attribution.csv",
            "role": "direct movement attribution",
            "contains": "session, phase, movement, direction, indices, copy start/end, source hook",
            "evidence_strength": "direct SGLang-visible KV movement",
        },
        {
            "artifact": "kv_block_ledger.csv",
            "role": "block lifecycle ledger",
            "contains": "block_id, lifecycle verdict, node/index identity, first write/evict/load times",
            "evidence_strength": "logical block lifecycle from normalized direct events",
        },
        {
            "artifact": "replay_delay_stage_trace.csv",
            "role": "request stage trace",
            "contains": "SGLang receive, scheduler queue/admit, cache/load-back, H2D stage times",
            "evidence_strength": "direct request-stage hooks when rows are present",
        },
        {
            "artifact": "replay_path_ledger.csv",
            "role": "replay path classification",
            "contains": "host-load, recompute estimate, cache-hit, scheduler/model evidence",
            "evidence_strength": "mixed direct, derived, and inferred evidence",
        },
        {
            "artifact": "client_dispatch_kv_movement_events.csv",
            "role": "dispatch-window movement audit",
            "contains": "all visible H2D/D2H/GPU-evict movement overlapping each dispatch window",
            "evidence_strength": "direct movement events filtered by derived window overlap",
        },
    ]


def audit_exact_movement_rows(exact_rows: EvidenceRows) -> EvidenceRows:
    total = len(exact_rows)
    if total == 0:
        return [
            {
                "area": "exact KV movement",
                "check": "rows present",
                "coverage": "0/0 (0.0%)",
                "evidence_level": "DIRECT",
                "status": "missing",
                "meaning": "No exact KV movement rows were available for this report.",
            }
        ]
    movement_kinds = Counter(movement_kind_display(movement_kind_from_row(row)) for row in exact_rows)
    identity_keys = ["request_id", "agent_request_id", "correlation_id"]
    output: EvidenceRows = [
        {
            "area": "exact KV movement",
            "check": "rows present",
            "coverage": coverage_text(total, total),
            "evidence_level": "DIRECT",
            "status": "strong",
            "meaning": "The report contains SGLang-visible KV movement rows.",
        },
        {
            "area": "exact KV movement",
            "check": "copy start/end coverage",
            "coverage": coverage_text(
                sum(1 for row in exact_rows if has_value(row, "copy_start_ms") and has_value(row, "copy_end_ms")),
                total,
            ),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(
                sum(1 for row in exact_rows if has_value(row, "copy_start_ms") and has_value(row, "copy_end_ms")),
                total,
            ),
            "meaning": "Rows with both start and end can support exact movement bars.",
        },
        {
            "area": "exact KV movement",
            "check": "session identity coverage",
            "coverage": coverage_text(count_rows_with(exact_rows, "session_id"), total),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(count_rows_with(exact_rows, "session_id"), total),
            "meaning": "Movement rows can be tied back to agent/session identity.",
        },
        {
            "area": "exact KV movement",
            "check": "request/correlation identity coverage",
            "coverage": coverage_text(count_rows_with_any(exact_rows, identity_keys), total),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(count_rows_with_any(exact_rows, identity_keys), total, weak_below=0.25),
            "meaning": "Movement rows can be tied to an SGLang request/correlation identity when these fields are present.",
        },
        {
            "area": "exact KV movement",
            "check": "node ID coverage",
            "coverage": coverage_text(count_rows_with(exact_rows, "node_id"), total),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(count_rows_with(exact_rows, "node_id"), total, weak_below=0.25),
            "meaning": "Movement rows can be tied to SGLang radix/cache nodes when node_id is present.",
        },
        {
            "area": "exact KV movement",
            "check": "host/device index signature coverage",
            "coverage": coverage_text(
                sum(
                    1
                    for row in exact_rows
                    if has_value(row, "host_index_signature") and has_value(row, "device_index_signature")
                ),
                total,
            ),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(
                sum(
                    1
                    for row in exact_rows
                    if has_value(row, "host_index_signature") and has_value(row, "device_index_signature")
                ),
                total,
                weak_below=0.25,
            ),
            "meaning": "Rows with both host and device signatures are the strongest software-visible movement evidence.",
        },
        {
            "area": "exact KV movement",
            "check": "source hook coverage",
            "coverage": coverage_text(count_rows_with(exact_rows, "source_event"), total),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(count_rows_with(exact_rows, "source_event"), total),
            "meaning": "Each movement row should show the SGLang hook that emitted it.",
        },
        {
            "area": "exact KV movement",
            "check": "movement type counts",
            "coverage": ", ".join(f"{name}={count}" for name, count in sorted(movement_kinds.items())),
            "evidence_level": "DIRECT",
            "status": "informational",
            "meaning": "Breakdown of direct movement/residency event types visible in this run.",
        },
    ]
    return output


def status_for_fraction(count: int, total: int, weak_below: float = 0.5) -> str:
    if total <= 0:
        return "missing"
    fraction = count / total
    if fraction >= 0.9:
        return "strong"
    if fraction >= weak_below:
        return "partial"
    return "weak"


def audit_block_ledger_rows(block_rows: EvidenceRows) -> EvidenceRows:
    total = len(block_rows)
    if total == 0:
        return [
            {
                "area": "KV block lifecycle",
                "check": "block ledger rows present",
                "coverage": "0/0 (0.0%)",
                "evidence_level": "DIRECT/DERIVED",
                "status": "missing",
                "meaning": "No KV block ledger rows were available for this report.",
            }
        ]
    exact_hd = sum(1 for row in block_rows if as_int(row.get("has_exact_host_device_indices")) == 1)
    with_history = count_rows_with(block_rows, "lifecycle_steps")
    with_write = sum(1 for row in block_rows if (as_int(row.get("write_host_events")) or 0) > 0)
    with_gpu_evict = sum(1 for row in block_rows if (as_int(row.get("evict_gpu_events")) or 0) > 0)
    with_host_evict = sum(1 for row in block_rows if (as_int(row.get("evict_host_events")) or 0) > 0)
    with_load = sum(1 for row in block_rows if (as_int(row.get("load_gpu_events")) or 0) > 0)
    return [
        {
            "area": "KV block lifecycle",
            "check": "block IDs present",
            "coverage": coverage_text(count_rows_with(block_rows, "block_id"), total),
            "evidence_level": "DERIVED",
            "status": status_for_fraction(count_rows_with(block_rows, "block_id"), total),
            "meaning": "Every logical KV block should have a stable report-level block_id.",
        },
        {
            "area": "KV block lifecycle",
            "check": "lifecycle history present",
            "coverage": coverage_text(with_history, total),
            "evidence_level": "DIRECT/DERIVED",
            "status": status_for_fraction(with_history, total),
            "meaning": "Lifecycle steps show the direct source events behind each block record.",
        },
        {
            "area": "KV block lifecycle",
            "check": "exact host+device index identity",
            "coverage": coverage_text(exact_hd, total),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(exact_hd, total, weak_below=0.25),
            "meaning": "Blocks with both host and device signatures have the strongest software-visible identity.",
        },
        {
            "area": "KV block lifecycle",
            "check": "lifecycle event counts",
            "coverage": f"write_host={with_write}, gpu_evict={with_gpu_evict}, host_evict={with_host_evict}, load_gpu={with_load}",
            "evidence_level": "DIRECT",
            "status": "informational",
            "meaning": "Direct movement/residency events represented in the block ledger.",
        },
    ]


def audit_request_stage_rows(stage_rows: EvidenceRows, queue_rows: EvidenceRows) -> EvidenceRows:
    total = len(stage_rows)
    stage_names = Counter(str(row.get("stage") or "") for row in stage_rows if has_value(row, "stage"))
    queue_total = len(queue_rows)
    identity_keys = ["request_id", "agent_request_id", "correlation_id"]
    output: EvidenceRows = [
        {
            "area": "request lifecycle",
            "check": "request stage rows present",
            "coverage": coverage_text(total, total),
            "evidence_level": "DIRECT",
            "status": "strong" if total else "missing",
            "meaning": "SGLang request-stage hooks are present when this count is nonzero.",
        },
        {
            "area": "request lifecycle",
            "check": "request/correlation identity coverage in stage rows",
            "coverage": coverage_text(count_rows_with_any(stage_rows, identity_keys), total),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(count_rows_with_any(stage_rows, identity_keys), total, weak_below=0.25),
            "meaning": "Stage rows can be tied to SGLang request/correlation identities when present.",
        },
        {
            "area": "request lifecycle",
            "check": "agent session coverage in stage rows",
            "coverage": coverage_text(count_rows_with(stage_rows, "session_id"), total),
            "evidence_level": "DIRECT",
            "status": status_for_fraction(count_rows_with(stage_rows, "session_id"), total),
            "meaning": "Stage rows can be tied back to timeline sessions.",
        },
        {
            "area": "request lifecycle",
            "check": "queue timing rows present",
            "coverage": coverage_text(queue_total, queue_total),
            "evidence_level": "DIRECT/DERIVED",
            "status": "strong" if queue_total else "missing",
            "meaning": "Replay queue timing rows feed the queue-vs-H2D chart.",
        },
        {
            "area": "request lifecycle",
            "check": "observed stage names",
            "coverage": ", ".join(f"{name}:{count}" for name, count in sorted(stage_names.items())[:16]),
            "evidence_level": "DIRECT",
            "status": "informational",
            "meaning": "Direct SGLang stage categories observed in this run.",
        },
    ]
    return output


def audit_dispatch_kv_rows(summary_rows: EvidenceRows, event_rows: EvidenceRows) -> EvidenceRows:
    total = len(summary_rows)
    events = len(event_rows)
    movement_kinds = Counter(movement_kind_display(movement_kind_from_row(row)) for row in event_rows)
    pressure = sum(1 for row in event_rows if str(row.get("owner_kind") or "") == "pressure/filler")
    target = sum(1 for row in event_rows if str(row.get("owner_kind") or "") == "target row")
    return [
        {
            "area": "client dispatch KV movement",
            "check": "dispatch summary rows present",
            "coverage": coverage_text(total, total),
            "evidence_level": "DERIVED",
            "status": "strong" if total else "missing",
            "meaning": "Each row summarizes visible KV movement during one replay dispatch window.",
        },
        {
            "area": "client dispatch KV movement",
            "check": "movement events inside dispatch windows",
            "coverage": str(events),
            "evidence_level": "DIRECT/DERIVED",
            "status": "strong" if events else "missing",
            "meaning": "Direct movement events after filtering to dispatch windows.",
        },
        {
            "area": "client dispatch KV movement",
            "check": "movement type counts",
            "coverage": ", ".join(f"{name}={count}" for name, count in sorted(movement_kinds.items())),
            "evidence_level": "DIRECT",
            "status": "informational",
            "meaning": "What kind of SGLang-visible KV movement happened while rows were dispatching.",
        },
        {
            "area": "client dispatch KV movement",
            "check": "owner attribution",
            "coverage": f"target={target}, pressure/filler={pressure}, other={events - target - pressure}",
            "evidence_level": "DERIVED",
            "status": "informational",
            "meaning": "Whether movement belonged to the target row, filler pressure, or other sessions.",
        },
    ]


def audit_report_data(data: dict[str, Any]) -> dict[str, EvidenceRows]:
    exact_rows = list(data.get("exact_kv_movement_attribution") or [])
    block_rows = list(data.get("kv_block_ledger") or [])
    stage_rows = list(data.get("replay_delay_stage_trace") or [])
    queue_rows = list(data.get("replay_queue_timing") or [])
    dispatch_summary = list(data.get("client_dispatch_kv_movement_summary") or [])
    dispatch_events = list(data.get("client_dispatch_kv_movement_events") or [])

    matrix: EvidenceRows = []
    matrix.extend(audit_exact_movement_rows(exact_rows))
    matrix.extend(audit_block_ledger_rows(block_rows))
    matrix.extend(audit_request_stage_rows(stage_rows, queue_rows))
    matrix.extend(audit_dispatch_kv_rows(dispatch_summary, dispatch_events))
    matrix.extend(inference_boundary_rows(data))

    summary = audit_summary_rows(matrix)
    return {
        "summary": summary,
        "matrix": matrix,
        "chart_inventory": chart_inventory_rows(),
        "artifact_inventory": artifact_inventory_rows(),
    }


def inference_boundary_rows(data: dict[str, Any]) -> EvidenceRows:
    gaps = list(data.get("gaps") or [])
    recompute_rows = [
        row
        for row in gaps
        if as_int(row.get("replay_new_prefill_tokens_est")) not in (None, 0)
        or as_int(row.get("recomputed_tokens_est")) not in (None, 0)
    ]
    return [
        {
            "area": "inference boundaries",
            "check": "recompute/rebuild evidence",
            "coverage": coverage_text(len(recompute_rows), len(gaps)),
            "evidence_level": "INFERRED",
            "status": "explicitly marked inferred",
            "meaning": "Recompute bars are estimates from cache/prefix/model-forward evidence, not direct physical recompute hooks.",
        },
        {
            "area": "inference boundaries",
            "check": "hardware DMA saturation",
            "coverage": "0 direct hardware counter rows",
            "evidence_level": "NOT_YET_PROVEN",
            "status": "intentionally limited",
            "meaning": "SGLang-visible movement can show runtime/cache activity, but not full physical DMA-engine occupancy.",
        },
    ]


def audit_summary_rows(matrix: EvidenceRows) -> EvidenceRows:
    by_level = Counter(str(row.get("evidence_level") or "") for row in matrix)
    by_status = Counter(str(row.get("status") or "") for row in matrix)
    rows: EvidenceRows = []
    for level, count in sorted(by_level.items()):
        if not level:
            continue
        rows.append({"summary_type": "evidence_level", "label": level, "count": count})
    for status, count in sorted(by_status.items()):
        if not status:
            continue
        rows.append({"summary_type": "status", "label": status, "count": count})
    return rows


def markdown_table(rows: EvidenceRows, columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = str(row.get(column, ""))
            value = value.replace("\n", " ").replace("|", "\\|")
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def audit_markdown(audit: dict[str, EvidenceRows]) -> str:
    return "\n".join(
        [
            "# Instrumentation Evidence Audit",
            "",
            "This audit maps the master report back to raw SGLang/direct-driver evidence.",
            "It separates direct evidence, derived values, inferred values, and claims that are not yet proven by this report.",
            "",
            "## Summary",
            "",
            markdown_table(audit["summary"], ["summary_type", "label", "count"]),
            "## Audit Matrix",
            "",
            markdown_table(audit["matrix"], ["area", "check", "coverage", "evidence_level", "status", "meaning"]),
            "## Chart Inventory",
            "",
            markdown_table(
                audit["chart_inventory"],
                [
                    "report_item",
                    "visual_element",
                    "source_artifact",
                    "raw_hook_or_source",
                    "identity_carried",
                    "evidence_level",
                    "limitation",
                ],
            ),
            "## Artifact Inventory",
            "",
            markdown_table(audit["artifact_inventory"], ["artifact", "role", "contains", "evidence_strength"]),
        ]
    )
