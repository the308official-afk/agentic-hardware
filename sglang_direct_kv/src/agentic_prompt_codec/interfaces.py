from __future__ import annotations

from typing import Any, Callable, Protocol
from .models import CodecConfig, Document, EncodedSegment


class TokenCounter(Protocol):
    identity: str

    def count_text(self, text: str) -> int: ...

    def count_request(self, payload: Any, api_kind: str) -> int: ...


class Codec(Protocol):
    name: str

    def encode(self, text: str, config: CodecConfig, counter: TokenCounter,
               check_budget: Callable[[], None]) -> EncodedSegment: ...


class RequestAdapter(Protocol):
    def extract(self, payload: Any) -> Document: ...

    def replace(self, document: Document, replacements: dict[tuple, str]) -> Any: ...
