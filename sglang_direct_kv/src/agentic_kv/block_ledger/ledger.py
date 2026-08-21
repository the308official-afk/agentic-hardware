from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .block_id import BlockIdentity, make_block_id, nearby_range_score
from .events import KVEventType, NormalizedKVEvent


@dataclass
class KVBlockRecord:
    block_id: str
    session_id: str
    token_start: int | None
    token_end: int | None
    token_count: int
    node_id: str = ""
    request_id: str = ""
    agent_request_id: str = ""
    correlation_id: str = ""
    case_id: str = ""
    gap_id: str = ""
    current_state: str = "UNKNOWN"
    first_seen_ms: float | None = None
    last_seen_ms: float | None = None
    host_index_signature: str = ""
    host_index_start: int | None = None
    host_index_end: int | None = None
    host_index_count: int = 0
    device_index_signature: str = ""
    device_index_start: int | None = None
    device_index_end: int | None = None
    device_index_count: int = 0
    first_write_host_ms: float | None = None
    first_evict_gpu_ms: float | None = None
    first_evict_host_ms: float | None = None
    first_load_gpu_ms: float | None = None
    last_load_gpu_ms: float | None = None
    write_host_events: int = 0
    evict_gpu_events: int = 0
    evict_host_events: int = 0
    load_gpu_events: int = 0
    replay_load_gpu_events: int = 0
    hint_load_gpu_events: int = 0
    match_prefix_events: int = 0
    recompute_events: int = 0
    confidence: str = "medium"
    exact_attribution: str = "range_only"
    evidence_level: str = ""
    exact_correlation_source: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def lost_before_replay(self) -> bool:
        return self.write_host_events > 0 and self.evict_gpu_events > 0 and self.evict_host_events > 0 and self.load_gpu_events == 0

    @property
    def loaded_by_replay(self) -> bool:
        return self.replay_load_gpu_events > 0

    @property
    def loaded_by_hint(self) -> bool:
        return self.hint_load_gpu_events > 0


class KVBlockLedger:
    def __init__(self) -> None:
        self.records: dict[str, KVBlockRecord] = {}

    def apply(self, event: NormalizedKVEvent) -> None:
        if event.event_type == KVEventType.MATCH_PREFIX:
            return
        record = self._find_or_create_record(event)
        self._apply_state(record, event)

    def _find_or_create_record(self, event: NormalizedKVEvent) -> KVBlockRecord:
        if event.node_id:
            for record in self.records.values():
                if record.session_id == event.session_id and record.node_id == event.node_id:
                    return record
        for signature_attr, record_attr in (
            ("host_index_signature", "host_index_signature"),
            ("device_index_signature", "device_index_signature"),
        ):
            event_signature = getattr(event, signature_attr, "")
            if not event_signature:
                continue
            for record in self.records.values():
                if record.session_id != event.session_id:
                    continue
                if getattr(record, record_attr) == event_signature:
                    return record
        best_record: KVBlockRecord | None = None
        best_score = 0.0
        for record in self.records.values():
            if record.session_id != event.session_id:
                continue
            score = nearby_range_score(
                record.token_start,
                record.token_end,
                record.token_count,
                event.token_start,
                event.token_end,
                event.token_count,
            )
            if score > best_score:
                best_score = score
                best_record = record
        if best_record is not None and best_score >= 0.75:
            if not best_record.node_id and event.node_id:
                best_record.node_id = event.node_id
                best_record.confidence = "high"
            return best_record
        identity = BlockIdentity(
            session_id=event.session_id,
            token_start=event.token_start,
            token_end=event.token_end,
            token_count=event.token_count,
            node_id=event.node_id,
        )
        block_id = make_block_id(identity)
        record = KVBlockRecord(
            block_id=block_id,
            session_id=event.session_id,
            token_start=event.token_start,
            token_end=event.token_end,
            token_count=event.token_count,
            node_id=event.node_id,
            request_id=event.request_id,
            agent_request_id=event.agent_request_id,
            correlation_id=event.correlation_id,
            case_id=event.case_id,
            gap_id=event.gap_id,
            host_index_signature=event.host_index_signature,
            host_index_start=event.host_index_start,
            host_index_end=event.host_index_end,
            host_index_count=event.host_index_count,
            device_index_signature=event.device_index_signature,
            device_index_start=event.device_index_start,
            device_index_end=event.device_index_end,
            device_index_count=event.device_index_count,
            first_seen_ms=event.time_ms,
            last_seen_ms=event.time_ms,
            confidence=event.confidence,
            exact_attribution=exact_attribution_level(event),
            evidence_level=event.evidence_level,
            exact_correlation_source=event.exact_correlation_source,
        )
        self.records[block_id] = record
        return record

    def _apply_state(self, record: KVBlockRecord, event: NormalizedKVEvent) -> None:
        record.last_seen_ms = event.time_ms
        if record.first_seen_ms is None:
            record.first_seen_ms = event.time_ms
        merge_identity_fields(record, event)
        record.history.append(event.to_dict())
        if event.event_type == KVEventType.WRITE_HOST:
            record.write_host_events += 1
            if record.first_write_host_ms is None:
                record.first_write_host_ms = event.time_ms
            record.current_state = "GPU_AND_HOST"
        elif event.event_type == KVEventType.EVICT_GPU:
            record.evict_gpu_events += 1
            if record.first_evict_gpu_ms is None:
                record.first_evict_gpu_ms = event.time_ms
            record.current_state = "HOST_RESIDENT" if record.write_host_events else "MISSING"
        elif event.event_type == KVEventType.EVICT_HOST:
            record.evict_host_events += 1
            if record.first_evict_host_ms is None:
                record.first_evict_host_ms = event.time_ms
            record.current_state = "MISSING" if record.evict_gpu_events else "GPU_RESIDENT"
        elif event.event_type == KVEventType.LOAD_GPU:
            record.load_gpu_events += 1
            if event.phase == "replay":
                record.replay_load_gpu_events += 1
            if event.phase == "hint_prefetch":
                record.hint_load_gpu_events += 1
            if record.first_load_gpu_ms is None:
                record.first_load_gpu_ms = event.time_ms
            record.last_load_gpu_ms = event.time_ms
            record.current_state = "RELOADED_TO_GPU"
        elif event.event_type == KVEventType.RECOMPUTE:
            record.recompute_events += 1
            record.current_state = "RECOMPUTED"

    def rows(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in sorted(self.records.values(), key=record_sort_key)]


def record_sort_key(record: KVBlockRecord) -> tuple[str, int, int, str]:
    start = record.token_start if record.token_start is not None else -1
    end = record.token_end if record.token_end is not None else -1
    return (record.session_id, start, end, record.block_id)


def build_block_ledger(events: list[NormalizedKVEvent]) -> KVBlockLedger:
    ledger = KVBlockLedger()
    for event in events:
        ledger.apply(event)
    return ledger


def merge_identity_fields(record: KVBlockRecord, event: NormalizedKVEvent) -> None:
    if not record.node_id and event.node_id:
        record.node_id = event.node_id
    if not record.request_id and event.request_id:
        record.request_id = event.request_id
    if not record.agent_request_id and event.agent_request_id:
        record.agent_request_id = event.agent_request_id
    if not record.correlation_id and event.correlation_id:
        record.correlation_id = event.correlation_id
    if not record.case_id and event.case_id:
        record.case_id = event.case_id
    if not record.gap_id and event.gap_id:
        record.gap_id = event.gap_id
    if not record.host_index_signature and event.host_index_signature:
        record.host_index_signature = event.host_index_signature
    if record.host_index_start is None and event.host_index_start is not None:
        record.host_index_start = event.host_index_start
    if record.host_index_end is None and event.host_index_end is not None:
        record.host_index_end = event.host_index_end
    if not record.host_index_count and event.host_index_count:
        record.host_index_count = event.host_index_count
    if not record.device_index_signature and event.device_index_signature:
        record.device_index_signature = event.device_index_signature
    if record.device_index_start is None and event.device_index_start is not None:
        record.device_index_start = event.device_index_start
    if record.device_index_end is None and event.device_index_end is not None:
        record.device_index_end = event.device_index_end
    if not record.device_index_count and event.device_index_count:
        record.device_index_count = event.device_index_count
    record.exact_attribution = strongest_attribution(record.exact_attribution, exact_attribution_level(event))
    record.evidence_level = strongest_evidence_level(record.evidence_level, event.evidence_level)
    if not record.exact_correlation_source and event.exact_correlation_source:
        record.exact_correlation_source = event.exact_correlation_source
    if event.confidence == "high":
        record.confidence = "high"


def exact_attribution_level(event: NormalizedKVEvent) -> str:
    if event.host_index_signature and event.device_index_signature:
        return "host_and_device_indices"
    if event.host_index_signature:
        return "host_indices"
    if event.device_index_signature:
        return "device_indices"
    if event.node_id:
        return "node_id"
    return "range_only"


def strongest_attribution(current: str, candidate: str) -> str:
    rank = {
        "range_only": 0,
        "node_id": 1,
        "device_indices": 2,
        "host_indices": 2,
        "host_and_device_indices": 3,
    }
    return candidate if rank.get(candidate, 0) > rank.get(current, 0) else current


def strongest_evidence_level(current: str, candidate: str) -> str:
    rank = {
        "": 0,
        "DERIVED_OR_INFERRED": 1,
        "DIRECT_TIMED": 2,
        "DIRECT_PARTIAL_ID": 3,
        "DIRECT_EXACT_INDEXED": 4,
    }
    return candidate if rank.get(candidate, 0) > rank.get(current, 0) else current
