#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentic_kv.block_ledger import KVEventType, NormalizedKVEvent, block_ledger_rows, build_block_ledger


def main() -> None:
    events = [
        NormalizedKVEvent(
            event_type=KVEventType.WRITE_HOST,
            session_id="agent_001",
            phase="initial_turn",
            token_start=2062,
            token_end=4109,
            token_count=2048,
            node_id="5",
            time_ms=10.0,
            source_event="hicache.write.end",
            confidence="high",
        ),
        NormalizedKVEvent(
            event_type=KVEventType.EVICT_GPU,
            session_id="agent_001",
            phase="initial_turn",
            token_start=2062,
            token_end=4109,
            token_count=2048,
            time_ms=20.0,
            source_event="hicache.evict_device.end",
        ),
        NormalizedKVEvent(
            event_type=KVEventType.EVICT_HOST,
            session_id="agent_001",
            phase="initial_turn",
            token_start=2061,
            token_end=4108,
            token_count=2048,
            time_ms=30.0,
            source_event="hicache.evict_host.end",
        ),
    ]
    ledger = build_block_ledger(events)
    rows = block_ledger_rows(ledger)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["write_host_events"] == 1, row
    assert row["evict_gpu_events"] == 1, row
    assert row["evict_host_events"] == 1, row
    assert row["lost_before_replay"] == 1, row
    assert row["current_state"] == "MISSING", row
    print("KV block ledger validation passed.")


if __name__ == "__main__":
    main()

