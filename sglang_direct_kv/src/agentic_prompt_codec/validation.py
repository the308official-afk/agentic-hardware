from __future__ import annotations

import re
from .models import EncodedSegment

# Protect fenced blocks, inline code, URLs, paths, JSON-looking lines and shell
# commands. Unknown/exact-use content can always be excluded by the adapter.
PROTECTED = re.compile(
    r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]*`"
    r"|https?://\S+|(?:[\w.~-]+/)+[\w./~-]+"
    r"|^[ \t]*(?:[\[{]|\$ |>>> |HARNESS_REPLAY_EXPERIMENT_JSON:).*$",
    re.MULTILINE,
)


def editable_parts(text: str) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    start = 0
    for match in PROTECTED.finditer(text):
        out.append((text[start:match.start()], True))
        out.append((match.group(), False))
        start = match.end()
    out.append((text[start:], True))
    return out


def decode(segment: EncodedSegment) -> str:
    """Expand exactly once; definitions themselves are never recursively read."""
    mapping = dict(segment.dictionary)
    if len(mapping) != len(segment.dictionary) or any(not key for key in mapping):
        raise ValueError("invalid dictionary")
    if not mapping:
        return segment.body
    pattern = re.compile("|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda match: mapping[match.group()], segment.body)


def validate(original: str, candidate: EncodedSegment) -> None:
    if candidate.text != candidate.legend + candidate.body:
        raise ValueError("rendered content does not match body and legend")
    if candidate.reversible and decode(candidate) != original:
        raise ValueError("round-trip mismatch")
    if any(original.count(alias) for alias, _ in candidate.dictionary):
        raise ValueError("alias collides with source text")
    for text, editable in editable_parts(original):
        if not editable and candidate.body.count(text) < original.count(text):
            raise ValueError("protected content changed")
