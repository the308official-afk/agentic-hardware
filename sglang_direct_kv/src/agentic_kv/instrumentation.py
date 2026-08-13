from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .hints import PrefetchHint


@dataclass
class KVTelemetry:
    prefetch_submitted: int = 0
    prefetch_completed: int = 0
    prefetch_hit: int = 0
    prefetch_late: int = 0
    prefetch_wasted: int = 0
    evicted_before_use: int = 0
    protected_sessions: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class DirectKVAdapter:
    """Interface for direct SGLang KV instrumentation.

    The probe-only implementation is intentionally conservative. On EC2, after
    identifying SGLang's KV/cache/offload internals, replace these methods with
    calls into the real KV block manager.
    """

    def tag_session_kv(self, hint: PrefetchHint) -> None:
        raise NotImplementedError

    async def prefetch_session_kv(self, hint: PrefetchHint) -> bool:
        raise NotImplementedError

    def protect_session_kv(self, session_id: str, protect_ms: int) -> None:
        raise NotImplementedError

    def release_session_kv(self, session_id: str) -> None:
        raise NotImplementedError

    def collect_kv_telemetry(self) -> KVTelemetry:
        raise NotImplementedError


class ProbeOnlySGLangKVAdapter(DirectKVAdapter):
    """No-op adapter used until direct SGLang KV paths are wired."""

    def __init__(self) -> None:
        self.telemetry = KVTelemetry()
        self.protected_until: dict[str, float] = {}

    def tag_session_kv(self, hint: PrefetchHint) -> None:
        self.telemetry.extra.setdefault("tagged_sessions", []).append(hint.session_id)

    async def prefetch_session_kv(self, hint: PrefetchHint) -> bool:
        self.telemetry.prefetch_submitted += 1
        # Placeholder: direct SGLang KV prefetch will be wired after probing.
        self.telemetry.prefetch_late += 1
        return False

    def protect_session_kv(self, session_id: str, protect_ms: int) -> None:
        self.protected_until[session_id] = time.perf_counter() + protect_ms / 1000.0
        self.telemetry.protected_sessions = len(self.protected_until)

    def release_session_kv(self, session_id: str) -> None:
        self.protected_until.pop(session_id, None)
        self.telemetry.protected_sessions = len(self.protected_until)

    def collect_kv_telemetry(self) -> KVTelemetry:
        now = time.perf_counter()
        expired = [sid for sid, until in self.protected_until.items() if until <= now]
        for sid in expired:
            self.protected_until.pop(sid, None)
        self.telemetry.protected_sessions = len(self.protected_until)
        return self.telemetry
