from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .ledger import KVBlockLedger


def block_ledger_rows(ledger: KVBlockLedger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in ledger.records.values():
        rows.append(
            {
                "block_id": record.block_id,
                "session_id": record.session_id,
                "token_start": record.token_start if record.token_start is not None else "",
                "token_end": record.token_end if record.token_end is not None else "",
                "token_count": record.token_count,
                "node_id": record.node_id,
                "current_state": record.current_state,
                "first_seen_ms": record.first_seen_ms if record.first_seen_ms is not None else "",
                "last_seen_ms": record.last_seen_ms if record.last_seen_ms is not None else "",
                "write_host_events": record.write_host_events,
                "evict_gpu_events": record.evict_gpu_events,
                "evict_host_events": record.evict_host_events,
                "load_gpu_events": record.load_gpu_events,
                "lost_before_replay": int(record.lost_before_replay),
                "confidence": record.confidence,
            }
        )
    return sorted(rows, key=lambda row: (str(row["session_id"]), int(row["token_start"] or -1), str(row["block_id"])))


def ledger_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_blocks = len(rows)
    written = [row for row in rows if int(row.get("write_host_events") or 0) > 0]
    gpu_evicted = [row for row in rows if int(row.get("evict_gpu_events") or 0) > 0]
    host_evicted = [row for row in rows if int(row.get("evict_host_events") or 0) > 0]
    loaded = [row for row in rows if int(row.get("load_gpu_events") or 0) > 0]
    lost = [row for row in rows if int(row.get("lost_before_replay") or 0) > 0]
    return [
        {"metric": "total logical KV blocks tracked", "value": total_blocks},
        {"metric": "blocks written to host HiCache", "value": len(written)},
        {"metric": "blocks evicted from GPU", "value": len(gpu_evicted)},
        {"metric": "blocks evicted from host HiCache", "value": len(host_evicted)},
        {"metric": "blocks loaded back to GPU", "value": len(loaded)},
        {"metric": "blocks lost before replay", "value": len(lost)},
        {"metric": "tokens lost before replay", "value": sum_int(lost, "token_count")},
    ]


def gap_lifecycle_summary_rows(gaps: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get("session_id") or "")].append(row)
    output: list[dict[str, Any]] = []
    for idx, gap in enumerate(gaps):
        session_id = str(gap.get("session_id") or "")
        blocks = by_session.get(session_id, [])
        lost = [row for row in blocks if int(row.get("lost_before_replay") or 0) > 0]
        loaded = [row for row in blocks if int(row.get("load_gpu_events") or 0) > 0]
        states = Counter(str(row.get("current_state") or "UNKNOWN") for row in blocks)
        recomputed = gap.get("replay_new_prefill_tokens_est", "")
        output.append(
            {
                "row": gap.get("timeline_label") or f"G{idx:02d}",
                "session_id": session_id,
                "mode": gap.get("mode", ""),
                "tool_wait_ms": gap.get("tool_gap_ms", ""),
                "tracked_blocks": len(blocks),
                "lost_blocks": len(lost),
                "lost_tokens": sum_int(lost, "token_count"),
                "loaded_blocks": len(loaded),
                "loaded_tokens": sum_int(loaded, "token_count"),
                "replay_recomputed_tokens": recomputed,
                "state_counts": ", ".join(f"{key}:{value}" for key, value in sorted(states.items())),
                "simple_meaning": gap_block_simple_meaning(gap, len(blocks), len(lost), sum_int(lost, "token_count"), len(loaded)),
            }
        )
    return output


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
    write_csv(out_dir / "kv_block_ledger.csv", rows)
    write_csv(out_dir / "kv_block_lifecycle_summary.csv", summary)
    write_csv(out_dir / "kv_block_gap_summary.csv", gap_summary)
    (out_dir / "kv_block_ledger.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "gap_summary": gap_summary,
                "blocks": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return sum(as_int(row.get(key)) or 0 for row in rows)


def as_int(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
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

