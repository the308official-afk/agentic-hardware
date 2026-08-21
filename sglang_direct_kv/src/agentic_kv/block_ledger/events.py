from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class KVEventType(str, Enum):
    WRITE_HOST = "KV_WRITE_HOST"
    EVICT_GPU = "KV_EVICT_GPU"
    EVICT_HOST = "KV_EVICT_HOST"
    LOAD_GPU = "KV_LOAD_GPU"
    MATCH_PREFIX = "KV_MATCH_PREFIX"
    RECOMPUTE = "KV_RECOMPUTE"


@dataclass(frozen=True)
class NormalizedKVEvent:
    event_type: KVEventType
    session_id: str
    phase: str
    time_ms: float | None = None
    duration_ms: float | None = None
    token_start: int | None = None
    token_end: int | None = None
    token_count: int = 0
    node_id: str = ""
    request_id: str = ""
    agent_request_id: str = ""
    correlation_id: str = ""
    case_id: str = ""
    gap_id: str = ""
    layer_id: str = ""
    direction: str = ""
    movement_kind: str = ""
    host_index_start: int | None = None
    host_index_end: int | None = None
    host_index_count: int = 0
    host_index_signature: str = ""
    device_index_start: int | None = None
    device_index_end: int | None = None
    device_index_count: int = 0
    device_index_signature: str = ""
    copy_start_ms: float | None = None
    copy_end_ms: float | None = None
    source_event: str = ""
    confidence: str = "medium"
    evidence_level: str = ""
    exact_correlation_source: str = ""
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        if self.raw is not None:
            data["raw"] = dict(self.raw)
        return data
