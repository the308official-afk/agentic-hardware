from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefetchHint:
    session_id: str
    priority: str
    expected_resume_ms: int
    reuse_confidence: float
    protect_ms: int
