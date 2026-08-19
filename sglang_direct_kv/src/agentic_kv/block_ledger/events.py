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
    source_event: str = ""
    confidence: str = "medium"
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        if self.raw is not None:
            data["raw"] = dict(self.raw)
        return data
