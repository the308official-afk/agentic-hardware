"""Portable request-local encoding contracts. No serving-engine dependencies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

VERSION = "prompt_codec.v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Segment:
    path: tuple[str | int, ...]
    text: str
    role: str
    eligible: bool = True


@dataclass(frozen=True)
class Document:
    payload: Any
    segments: tuple[Segment, ...]
    api_kind: str


@dataclass(frozen=True)
class CodecConfig:
    codec: str = "identity"
    min_saved_tokens: int = 16
    min_saved_ratio: float = 0.05
    max_input_chars: int = 100_000
    max_encode_ms: float = 100.0
    max_dictionary_entries: int = 8
    max_candidates: int = 48
    min_phrase_words: int = 4
    max_phrase_words: int = 12

    def __post_init__(self) -> None:
        if self.min_saved_tokens < 0 or not 0 <= self.min_saved_ratio < 1:
            raise ValueError("invalid savings threshold")
        if min(self.max_input_chars, self.max_encode_ms, self.max_dictionary_entries,
               self.max_candidates, self.min_phrase_words) <= 0:
            raise ValueError("encoding limits must be positive")
        if self.max_phrase_words < self.min_phrase_words:
            raise ValueError("invalid phrase lengths")

    @property
    def fingerprint(self) -> str:
        return digest({"version": VERSION, **asdict(self)})[:16]


@dataclass(frozen=True)
class EncodedSegment:
    text: str
    body: str
    dictionary: tuple[tuple[str, str], ...] = ()
    legend: str = ""
    reversible: bool = True


@dataclass(frozen=True)
class EncodingResult:
    payload: Any
    status: str
    reason: str
    codec: str
    config_hash: str
    original_hash: str
    encoded_hash: str
    started_ns: int
    finished_ns: int
    elapsed_ms: float
    tokenizer_id: str = ""
    original_tokens: int | None = None
    encoded_tokens: int | None = None
    candidate_tokens: int | None = None
    legend_tokens: int | None = None
    changed_segments: int = 0
    validation: str = "not_run"
    schema_version: str = VERSION

    def evidence(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("payload")
        values["saved_tokens"] = (self.original_tokens - self.encoded_tokens
                                  if self.original_tokens is not None and self.encoded_tokens is not None else None)
        return {"encoding_" + key: value for key, value in values.items()}
