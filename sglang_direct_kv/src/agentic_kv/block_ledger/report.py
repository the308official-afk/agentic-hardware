from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .ledger import KVBlockLedger
from .events import KVEventType, NormalizedKVEvent


def block_ledger_rows(ledger: KVBlockLedger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in ledger.records.values():
        rows.append(
            {
                "block_id": record.block_id,
                "session_id": record.session_id,
                "lifecycle_verdict": block_lifecycle_verdict(record),
                "lifecycle_explanation": block_lifecycle_explanation(record),
                "token_start": record.token_start if record.token_start is not None else "",
                "token_end": record.token_end if record.token_end is not None else "",
                "token_count": record.token_count,
                "node_id": record.node_id,
                "current_state": record.current_state,
                "first_seen_ms": record.first_seen_ms if record.first_seen_ms is not None else "",
                "last_seen_ms": record.last_seen_ms if record.last_seen_ms is not None else "",
                "host_index_signature": record.host_index_signature,
                "host_index_start": record.host_index_start if record.host_index_start is not None else "",
                "host_index_end": record.host_index_end if record.host_index_end is not None else "",
                "host_index_count": record.host_index_count,
                "device_index_signature": record.device_index_signature,
                "device_index_start": record.device_index_start if record.device_index_start is not None else "",
                "device_index_end": record.device_index_end if record.device_index_end is not None else "",
                "device_index_count": record.device_index_count,
                "first_write_host_ms": record.first_write_host_ms if record.first_write_host_ms is not None else "",
                "first_evict_gpu_ms": record.first_evict_gpu_ms if record.first_evict_gpu_ms is not None else "",
                "first_evict_host_ms": record.first_evict_host_ms if record.first_evict_host_ms is not None else "",
                "first_load_gpu_ms": record.first_load_gpu_ms if record.first_load_gpu_ms is not None else "",
                "last_load_gpu_ms": record.last_load_gpu_ms if record.last_load_gpu_ms is not None else "",
                "write_host_events": record.write_host_events,
                "evict_gpu_events": record.evict_gpu_events,
                "evict_host_events": record.evict_host_events,
                "load_gpu_events": record.load_gpu_events,
                "hint_load_gpu_events": record.hint_load_gpu_events,
                "replay_load_gpu_events": record.replay_load_gpu_events,
                "loaded_by_hint": int(record.loaded_by_hint),
                "loaded_by_replay": int(record.loaded_by_replay),
                "lost_before_replay": int(record.lost_before_replay),
                "confidence": record.confidence,
                "exact_attribution": record.exact_attribution,
                "has_exact_host_device_indices": int(
                    bool(record.host_index_signature and record.device_index_signature)
                ),
                "history_event_count": len(record.history),
                "lifecycle_steps": block_lifecycle_steps(record),
            }
        )
    return sorted(rows, key=lambda row: (str(row["session_id"]), int(row["token_start"] or -1), str(row["block_id"])))


def ledger_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_blocks = len(rows)
    written = [row for row in rows if int(row.get("write_host_events") or 0) > 0]
    gpu_evicted = [row for row in rows if int(row.get("evict_gpu_events") or 0) > 0]
    host_evicted = [row for row in rows if int(row.get("evict_host_events") or 0) > 0]
    loaded = [row for row in rows if int(row.get("load_gpu_events") or 0) > 0]
    replay_loaded = [row for row in rows if int(row.get("replay_load_gpu_events") or 0) > 0]
    hint_loaded = [row for row in rows if int(row.get("hint_load_gpu_events") or 0) > 0]
    lost = [row for row in rows if int(row.get("lost_before_replay") or 0) > 0]
    exact_host_device = [row for row in rows if int(row.get("has_exact_host_device_indices") or 0) > 0]
    full_history = [row for row in rows if row.get("lifecycle_steps")]
    return [
        {"metric": "total logical KV blocks tracked", "value": total_blocks},
        {"metric": "blocks with exact host+device index signatures", "value": len(exact_host_device)},
        {"metric": "blocks with lifecycle history", "value": len(full_history)},
        {"metric": "blocks written to host HiCache", "value": len(written)},
        {"metric": "blocks evicted from GPU", "value": len(gpu_evicted)},
        {"metric": "blocks evicted from host HiCache", "value": len(host_evicted)},
        {"metric": "blocks loaded back to GPU", "value": len(loaded)},
        {"metric": "blocks loaded by hint/prefetch path", "value": len(hint_loaded)},
        {"metric": "blocks loaded by replay path", "value": len(replay_loaded)},
        {"metric": "blocks lost before replay", "value": len(lost)},
        {"metric": "tokens lost before replay", "value": sum_int(lost, "token_count")},
    ]


def gap_lifecycle_summary_rows(gaps: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get("session_id") or "")].append(row)
    output: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        session_id = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        display_session_id = str(gap.get("session_id") or "")
        blocks = by_session.get(session_id, [])
        lost = [row for row in blocks if int(row.get("lost_before_replay") or 0) > 0]
        loaded = [row for row in blocks if int(row.get("load_gpu_events") or 0) > 0]
        replay_loaded = [row for row in blocks if int(row.get("replay_load_gpu_events") or 0) > 0]
        hint_loaded = [row for row in blocks if int(row.get("hint_load_gpu_events") or 0) > 0]
        states = Counter(str(row.get("current_state") or "UNKNOWN") for row in blocks)
        recomputed = gap.get("replay_new_prefill_tokens_est", "")
        output.append(
            {
                "row": gap.get("timeline_label") or f"G{idx:02d}",
                "session_id": display_session_id,
                "ledger_session_id": session_id,
                "mode": gap.get("mode", ""),
                "tool_wait_ms": gap.get("tool_gap_ms", ""),
                "tracked_blocks": len(blocks),
                "lost_blocks": len(lost),
                "lost_tokens": sum_int(lost, "token_count"),
                "loaded_blocks": len(loaded),
                "loaded_tokens": sum_int(loaded, "token_count"),
                "hint_loaded_blocks": len(hint_loaded),
                "replay_loaded_blocks": len(replay_loaded),
                "replay_recomputed_tokens": recomputed,
                "state_counts": ", ".join(f"{key}:{value}" for key, value in sorted(states.items())),
                "simple_meaning": gap_block_simple_meaning(gap, len(blocks), len(lost), sum_int(lost, "token_count"), len(loaded)),
            }
        )
    return output


def block_lifecycle_by_gap_rows(gaps: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get("session_id") or "")].append(row)

    output: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        session_id = str(gap.get("ledger_session_id") or gap.get("session_id") or "")
        display_session_id = str(gap.get("session_id") or "")
        label = gap.get("timeline_label") or f"G{idx:02d}"
        blocks = sorted(
            by_session.get(session_id, []),
            key=lambda row: (
                int(as_int(row.get("loaded_by_replay")) or 0) * -1,
                int(as_int(row.get("lost_before_replay")) or 0) * -1,
                int(as_int(row.get("token_start")) or -1),
                str(row.get("block_id") or ""),
            ),
        )
        for block in blocks:
            output.append(
                {
                    "row": label,
                    "session_id": display_session_id,
                    "ledger_session_id": session_id,
                    "mode": gap.get("mode", ""),
                    "lifecycle_verdict": block.get("lifecycle_verdict", ""),
                    "lifecycle_explanation": block.get("lifecycle_explanation", ""),
                    "block_id": block.get("block_id", ""),
                    "node_id": block.get("node_id", ""),
                    "token_start": block.get("token_start", ""),
                    "token_end": block.get("token_end", ""),
                    "token_range": token_range_text(block),
                    "token_count": block.get("token_count", ""),
                    "current_state": block.get("current_state", ""),
                    "tool_wait_ms": gap.get("tool_gap_ms", ""),
                    "fillers": gap.get("fillers", ""),
                    "replay_path": gap.get("replay_path", ""),
                    "replay_lifecycle_verdict": gap.get("lifecycle_verdict", ""),
                    "replay_initial_match_tokens": gap.get("replay_initial_cached_prefix_tokens", ""),
                    "replay_final_cached_tokens": gap.get("replay_final_cached_prefix_tokens", ""),
                    "replay_host_load_tokens": gap.get("replay_host_load_tokens", ""),
                    "replay_new_prefill_tokens_est": gap.get("replay_new_prefill_tokens_est", ""),
                    "loaded_by_hint": block.get("loaded_by_hint", ""),
                    "loaded_by_replay": block.get("loaded_by_replay", ""),
                    "lost_before_replay": block.get("lost_before_replay", ""),
                    "first_write_host_ms": block.get("first_write_host_ms", ""),
                    "first_evict_gpu_ms": block.get("first_evict_gpu_ms", ""),
                    "first_evict_host_ms": block.get("first_evict_host_ms", ""),
                    "h2d_start_ms": block.get("first_load_gpu_ms", ""),
                    "h2d_end_ms": block.get("last_load_gpu_ms", ""),
                    "h2d_duration_ms": block_h2d_duration(block),
                    "recompute_start_ms_est": replay_recompute_start_ms(gap),
                    "recompute_end_ms_est": replay_recompute_end_ms(gap),
                    "recompute_duration_ms_est": gap.get("prefill_compute_ms_est", ""),
                    "replay_due_ms": gap.get("tool_gap_end_ms", ""),
                    "replay_start_ms": gap.get("resume_start_ms", ""),
                    "first_token_ms": gap.get("replay_prefill_end_ms", ""),
                    "replay_end_ms": gap.get("resume_end_ms", ""),
                    "write_host_events": block.get("write_host_events", ""),
                    "evict_gpu_events": block.get("evict_gpu_events", ""),
                    "evict_host_events": block.get("evict_host_events", ""),
                    "load_gpu_events": block.get("load_gpu_events", ""),
                    "hint_load_gpu_events": block.get("hint_load_gpu_events", ""),
                    "replay_load_gpu_events": block.get("replay_load_gpu_events", ""),
                    "host_index_signature": block.get("host_index_signature", ""),
                    "device_index_signature": block.get("device_index_signature", ""),
                    "exact_attribution": block.get("exact_attribution", ""),
                    "confidence": block.get("confidence", ""),
                    "lifecycle_steps": block.get("lifecycle_steps", ""),
                    "evidence_summary": block_gap_evidence_summary(gap, block),
                }
            )
    return output


def block_lifecycle_focus_rows(rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    priority = {
        "replay_loaded_from_host": 0,
        "hint_loaded_from_host": 1,
        "lost_before_replay": 2,
        "host_copy_evicted": 3,
        "host_resident_after_gpu_eviction": 4,
    }
    selected = sorted(
        rows,
        key=lambda row: (
            priority.get(str(row.get("lifecycle_verdict") or ""), 9),
            str(row.get("row") or ""),
            int(as_int(row.get("token_start")) or -1),
            str(row.get("block_id") or ""),
        ),
    )
    if limit is not None:
        selected = selected[:limit]
    columns = [
        "row",
        "mode",
        "lifecycle_verdict",
        "lifecycle_explanation",
        "block_id",
        "node_id",
        "token_range",
        "token_count",
        "loaded_by_replay",
        "loaded_by_hint",
        "lost_before_replay",
        "first_write_host_ms",
        "first_evict_gpu_ms",
        "first_evict_host_ms",
        "h2d_start_ms",
        "h2d_end_ms",
        "h2d_duration_ms",
        "recompute_start_ms_est",
        "recompute_end_ms_est",
        "recompute_duration_ms_est",
        "replay_due_ms",
        "replay_start_ms",
        "first_token_ms",
        "replay_end_ms",
        "exact_attribution",
        "confidence",
        "lifecycle_steps",
        "evidence_summary",
    ]
    return [{column: row.get(column, "") for column in columns} for row in selected]


def block_lifecycle_verdict_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("lifecycle_verdict") or "unknown") for row in rows)
    tokens: defaultdict[str, int] = defaultdict(int)
    exact_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        verdict = str(row.get("lifecycle_verdict") or "unknown")
        tokens[verdict] += as_int(row.get("token_count")) or 0
        if row.get("exact_attribution") == "host_and_device_indices":
            exact_counts[verdict] += 1
    return [
        {
            "lifecycle_verdict": verdict,
            "blocks": count,
            "tokens": tokens[verdict],
            "exact_host_device_blocks": exact_counts[verdict],
            "simple_meaning": lifecycle_verdict_meaning(verdict),
        }
        for verdict, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def gap_block_simple_meaning(
    gap: dict[str, Any],
    tracked_blocks: int,
    lost_blocks: int,
    lost_tokens: int,
    loaded_blocks: int,
) -> str:
    recomputed = as_int(gap.get("replay_new_prefill_tokens_est")) or 0
    if lost_blocks and not loaded_blocks and recomputed >= 128:
        return (
            f"{lost_blocks}/{tracked_blocks} tracked KV blocks were lost before replay "
            f"({lost_tokens} tokens), and replay rebuilt/prefilled about {recomputed} tokens."
        )
    if loaded_blocks:
        return f"Replay or prefetch loaded {loaded_blocks} tracked KV blocks back to GPU."
    if tracked_blocks and recomputed >= 128:
        return f"{tracked_blocks} KV blocks were tracked, but replay still rebuilt/prefilled about {recomputed} tokens."
    if tracked_blocks:
        return "Tracked KV blocks stayed reusable or did not require visible load-back."
    return "No logical KV blocks were tracked for this row."


def write_ledger_artifacts(out_dir: Path, rows: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = ledger_summary_rows(rows)
    gap_summary = gap_lifecycle_summary_rows(gaps, rows)
    block_gap_rows = block_lifecycle_by_gap_rows(gaps, rows)
    write_csv(out_dir / "kv_block_ledger.csv", rows)
    write_csv(out_dir / "kv_block_lifecycle_summary.csv", summary)
    write_csv(out_dir / "kv_block_gap_summary.csv", gap_summary)
    write_csv(out_dir / "kv_block_lifecycle_by_gap.csv", block_gap_rows)
    write_csv(out_dir / "kv_block_lifecycle_verdict_counts.csv", block_lifecycle_verdict_counts(block_gap_rows))
    (out_dir / "kv_block_ledger.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "gap_summary": gap_summary,
                "block_lifecycle_by_gap": block_gap_rows,
                "verdict_counts": block_lifecycle_verdict_counts(block_gap_rows),
                "blocks": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def exact_movement_rows(events: list[NormalizedKVEvent]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.event_type not in {KVEventType.WRITE_HOST, KVEventType.EVICT_GPU, KVEventType.EVICT_HOST, KVEventType.LOAD_GPU}:
            continue
        rows.append(
            {
                "session_id": event.session_id,
                "phase": event.phase,
                "movement": movement_label(event),
                "direction": event.direction,
                "request_id": event.request_id,
                "node_id": event.node_id,
                "layer_id": event.layer_id,
                "copy_start_ms": event.copy_start_ms if event.copy_start_ms is not None else "",
                "copy_end_ms": event.copy_end_ms if event.copy_end_ms is not None else "",
                "duration_ms": event.duration_ms if event.duration_ms is not None else "",
                "host_index_start": event.host_index_start if event.host_index_start is not None else "",
                "host_index_end": event.host_index_end if event.host_index_end is not None else "",
                "host_index_count": event.host_index_count,
                "host_index_signature": event.host_index_signature,
                "device_index_start": event.device_index_start if event.device_index_start is not None else "",
                "device_index_end": event.device_index_end if event.device_index_end is not None else "",
                "device_index_count": event.device_index_count,
                "device_index_signature": event.device_index_signature,
                "token_or_index_count": event.token_count,
                "source_event": event.source_event,
                "confidence": event.confidence,
                "simple_meaning": movement_simple_meaning(event),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("session_id") or ""),
            float(row.get("copy_start_ms") or row.get("copy_end_ms") or -1),
            str(row.get("source_event") or ""),
        ),
    )


def exact_movement_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(rows)
    by_movement = Counter(str(row.get("movement") or "unknown") for row in rows)
    high_conf = sum(1 for row in rows if row.get("confidence") == "high")
    exact_host_device = sum(1 for row in rows if row.get("host_index_signature") and row.get("device_index_signature"))
    return [
        {"metric": "exact movement rows", "value": total},
        {"metric": "high-confidence rows", "value": high_conf},
        {"metric": "rows with both host and device index signatures", "value": exact_host_device},
        {"metric": "host -> GPU load rows", "value": by_movement.get("host_to_gpu_load", 0)},
        {"metric": "GPU -> host write rows", "value": by_movement.get("gpu_to_host_write", 0)},
        {"metric": "GPU eviction rows", "value": by_movement.get("gpu_evict", 0)},
        {"metric": "host eviction rows", "value": by_movement.get("host_evict", 0)},
    ]


def movement_label(event: NormalizedKVEvent) -> str:
    if event.event_type == KVEventType.LOAD_GPU:
        return "host_to_gpu_load"
    if event.event_type == KVEventType.WRITE_HOST:
        return "gpu_to_host_write"
    if event.event_type == KVEventType.EVICT_GPU:
        return "gpu_evict"
    if event.event_type == KVEventType.EVICT_HOST:
        return "host_evict"
    return event.event_type.value


def movement_simple_meaning(event: NormalizedKVEvent) -> str:
    count = event.host_index_count or event.device_index_count or event.token_count
    if event.event_type == KVEventType.LOAD_GPU:
        return f"SGLang loaded {count} KV indices from host memory into GPU memory."
    if event.event_type == KVEventType.WRITE_HOST:
        return f"SGLang wrote {count} KV indices from GPU memory into host HiCache."
    if event.event_type == KVEventType.EVICT_GPU:
        return f"SGLang removed {count} KV indices from GPU residency."
    if event.event_type == KVEventType.EVICT_HOST:
        return f"SGLang removed {count} KV indices from host HiCache."
    return "SGLang emitted a KV lifecycle event."


def block_lifecycle_verdict(record: Any) -> str:
    if getattr(record, "loaded_by_replay", False):
        return "replay_loaded_from_host"
    if getattr(record, "loaded_by_hint", False):
        return "hint_loaded_from_host"
    if getattr(record, "lost_before_replay", False):
        return "lost_before_replay"
    if getattr(record, "evict_host_events", 0) > 0 and getattr(record, "load_gpu_events", 0) == 0:
        return "host_copy_evicted"
    if getattr(record, "write_host_events", 0) > 0 and getattr(record, "evict_gpu_events", 0) > 0:
        return "host_resident_after_gpu_eviction"
    if getattr(record, "write_host_events", 0) > 0:
        return "gpu_and_host_resident"
    if getattr(record, "load_gpu_events", 0) > 0:
        return "loaded_to_gpu"
    if getattr(record, "recompute_events", 0) > 0:
        return "recomputed"
    return "unknown"


def block_lifecycle_explanation(record: Any) -> str:
    count = getattr(record, "token_count", 0) or getattr(record, "host_index_count", 0) or getattr(record, "device_index_count", 0)
    verdict = block_lifecycle_verdict(record)
    if verdict == "replay_loaded_from_host":
        return f"Replay loaded this logical KV block back from host to GPU ({count} indices/tokens tracked)."
    if verdict == "hint_loaded_from_host":
        return f"The hint/direct-prefetch path loaded this KV block from host to GPU before replay."
    if verdict == "lost_before_replay":
        return f"This KV block was written to host, evicted from GPU, then evicted from host before any load-back."
    if verdict == "host_copy_evicted":
        return f"The host-side copy of this KV block was evicted; no later GPU load-back was observed."
    if verdict == "host_resident_after_gpu_eviction":
        return f"This KV block was evicted from GPU but remained visible in host HiCache; no replay load-back was observed."
    if verdict == "gpu_and_host_resident":
        return f"SGLang wrote this KV block to host while it also remained GPU-resident in the observed trace."
    if verdict == "loaded_to_gpu":
        return f"This KV block was loaded to GPU, but the trace did not label it as hint-side or replay-side."
    if verdict == "recomputed":
        return "This block was marked recomputed by normalized cache evidence."
    return "The trace did not expose enough lifecycle evidence for this block."


def block_lifecycle_steps(record: Any) -> str:
    grouped: dict[str, dict[str, Any]] = {}
    for event in getattr(record, "history", []):
        event_type = str(event.get("event_type") or "")
        phase = str(event.get("phase") or "")
        start = as_float(event.get("copy_start_ms"))
        end = as_float(event.get("copy_end_ms") or event.get("time_ms"))
        label = lifecycle_step_label(event_type)
        if phase:
            label = f"{label}[{phase}]"
        timestamp = end if end is not None else start
        if timestamp is None:
            continue
        item = grouped.setdefault(
            label,
            {
                "first": timestamp,
                "start": start if start is not None else timestamp,
                "end": end if end is not None else timestamp,
                "count": 0,
            },
        )
        item["count"] += 1
        item["first"] = min(float(item["first"]), timestamp)
        item["start"] = min(float(item["start"]), start if start is not None else timestamp)
        item["end"] = max(float(item["end"]), end if end is not None else timestamp)
    ordered = sorted(grouped.items(), key=lambda pair: float(pair[1]["first"]))
    labels: list[str] = []
    for label, item in ordered:
        count = int(item["count"])
        suffix = f" x{count}" if count > 1 else ""
        start = round(float(item["start"]), 3)
        end = round(float(item["end"]), 3)
        if abs(end - start) > 0.001:
            labels.append(f"{label}{suffix}@{start}-{end}ms")
        else:
            labels.append(f"{label}{suffix}@{end}ms")
    return " -> ".join(labels)


def lifecycle_step_label(event_type: str) -> str:
    return {
        KVEventType.WRITE_HOST.value: "write_host",
        KVEventType.EVICT_GPU.value: "evict_gpu",
        KVEventType.EVICT_HOST.value: "evict_host",
        KVEventType.LOAD_GPU.value: "load_gpu",
        KVEventType.MATCH_PREFIX.value: "match_prefix",
        KVEventType.RECOMPUTE.value: "recompute",
    }.get(event_type, event_type.replace("KV_", "").lower())


def block_gap_evidence_summary(gap: dict[str, Any], block: dict[str, Any]) -> str:
    row = gap.get("timeline_label", "")
    verdict = str(block.get("lifecycle_verdict") or "")
    block_id = str(block.get("block_id") or "")
    if verdict == "replay_loaded_from_host":
        return f"{row}: {block_id} was loaded from host during replay; this is direct replay-side KV HtoD evidence."
    if verdict == "hint_loaded_from_host":
        return f"{row}: {block_id} was loaded by the hint path; this is direct prefetch-side KV HtoD evidence."
    if verdict == "lost_before_replay":
        return f"{row}: {block_id} was backed up to host but later lost from both GPU and host before replay."
    if verdict == "host_resident_after_gpu_eviction":
        return f"{row}: {block_id} left GPU residency but remained available in host cache in the observed trace."
    if verdict == "gpu_and_host_resident":
        return f"{row}: {block_id} was written to host; no eviction/load-back was observed for this block."
    return f"{row}: lifecycle evidence for {block_id} is limited or ambiguous."


def lifecycle_verdict_meaning(verdict: str) -> str:
    return {
        "replay_loaded_from_host": "Replay needed old KV and SGLang loaded it from host to GPU.",
        "hint_loaded_from_host": "The direct-prefetch path loaded old KV from host to GPU.",
        "lost_before_replay": "The KV block was evicted from both GPU and host before reuse.",
        "host_copy_evicted": "The host copy disappeared before any visible load-back.",
        "host_resident_after_gpu_eviction": "The block left GPU memory but stayed in host cache.",
        "gpu_and_host_resident": "The block was backed up to host and no later eviction/load was observed.",
        "loaded_to_gpu": "A load to GPU was observed, but phase attribution was not specific.",
        "recomputed": "Replay rebuilt the block instead of loading it.",
        "unknown": "The trace did not expose enough lifecycle events.",
    }.get(verdict, "Lifecycle verdict emitted by the block ledger.")


def token_range_text(row: dict[str, Any]) -> str:
    start = row.get("token_start", "")
    end = row.get("token_end", "")
    if start in ("", None) or end in ("", None):
        count = row.get("token_count", "")
        return f"unknown ({count} tracked)" if count not in ("", None, 0, "0") else "unknown"
    return f"{start}..{end}"


def block_h2d_duration(row: dict[str, Any]) -> float | str:
    start = as_float(row.get("first_load_gpu_ms"))
    end = as_float(row.get("last_load_gpu_ms"))
    if start is None or end is None:
        return ""
    return round(max(0.0, end - start), 3)


def replay_recompute_start_ms(gap: dict[str, Any]) -> float | str:
    recompute_tokens = as_int(gap.get("recomputed_tokens_est")) or as_int(gap.get("replay_new_prefill_tokens_est")) or 0
    if recompute_tokens < 128:
        return ""
    return gap.get("replay_prefill_start_ms", "") or gap.get("resume_start_ms", "")


def replay_recompute_end_ms(gap: dict[str, Any]) -> float | str:
    start = as_float(replay_recompute_start_ms(gap))
    duration = as_float(gap.get("prefill_compute_ms_est"))
    first_token = as_float(gap.get("replay_prefill_end_ms"))
    if start is None:
        return ""
    if duration is not None and duration > 0:
        return round(start + duration, 3)
    return first_token if first_token is not None else ""


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return sum(as_int(row.get(key)) or 0 for row in rows)


def as_int(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
