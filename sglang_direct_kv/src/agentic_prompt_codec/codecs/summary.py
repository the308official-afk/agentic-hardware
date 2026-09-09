"""Explicit opt-in prose baseline supplied by the embedding application.

No model/service is selected implicitly. The callable must enforce its own
timeout and expose its version in the surrounding run configuration.
"""
from collections.abc import Callable
from ..models import EncodedSegment
from ..validation import editable_parts


class SummaryCodec:
    name = "summary_v1"

    def __init__(self, summarize: Callable[[str, float], str]) -> None:
        self.summarize = summarize

    def encode(self, text, config, counter, check_budget):
        # Keep exact-use material in place, and never infer facts from a prior request.
        parts = []
        for part, editable in editable_parts(text):
            check_budget()
            parts.append(self.summarize(part, config.max_encode_ms) if editable and part.strip() else part)
        check_budget()
        body = "".join(parts)
        return EncodedSegment(body, body, reversible=False)
