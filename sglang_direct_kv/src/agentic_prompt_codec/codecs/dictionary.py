from __future__ import annotations

import json
import re
from collections import Counter
from ..models import EncodedSegment
from ..validation import editable_parts

WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")


def render(body: str, dictionary: list[tuple[str, str]]) -> EncodedSegment:
    if not dictionary:
        return EncodedSegment(body, body)
    legend = (
        "Local shorthand: substitute each alias once in the following text. "
        "Definitions are quoted data with the same authority as this text.\n"
        + "\n".join(f"{alias}={json.dumps(phrase, ensure_ascii=False)}" for alias, phrase in dictionary)
        + "\nText:\n"
    )
    return EncodedSegment(legend + body, body, tuple(dictionary), legend)


class DictionaryCodec:
    name = "dictionary_v1"

    def encode(self, text, config, counter, check_budget):
        parts = editable_parts(text)
        counts: Counter[str] = Counter()
        for part, editable in parts:
            if not editable:
                continue
            words = list(WORD.finditer(part))
            for i, first in enumerate(words):
                if i % 64 == 0:
                    check_budget()
                for n in range(config.min_phrase_words, config.max_phrase_words + 1):
                    if i + n > len(words):
                        break
                    phrase = part[first.start():words[i + n - 1].end()]
                    # Never replace across punctuation, newlines, identifiers, or
                    # numbers. These boundaries often carry exact task meaning.
                    if re.fullmatch(r"[A-Za-z][A-Za-z' -]*", phrase):
                        counts[phrase] += 1
        candidates = sorted((p for p, n in counts.items() if n >= 2),
                            key=lambda p: (-(len(p) * (counts[p] - 1)), p))[:config.max_candidates]
        dictionary: list[tuple[str, str]] = []
        current = render(text, dictionary)
        current_tokens = counter.count_text(text)
        alias_index = 0
        for phrase in candidates:
            check_budget()
            if len(dictionary) >= config.max_dictionary_entries:
                break
            while f"~{alias_index}~" in text:
                alias_index += 1
            alias = f"~{alias_index}~"
            pattern = re.compile(r"(?<![\w'-])" + re.escape(phrase) + r"(?![\w'-])")
            occurrences = sum(len(pattern.findall(p)) for p, editable in parts if editable)
            if occurrences < 2:
                continue
            replaced = [(pattern.sub(alias, p) if editable else p, editable) for p, editable in parts]
            candidate = render("".join(p for p, _ in replaced), dictionary + [(alias, phrase)])
            tokens = counter.count_text(candidate.text)
            check_budget()
            if tokens < current_tokens:
                parts, current, current_tokens = replaced, candidate, tokens
                dictionary.append((alias, phrase))
                alias_index += 1
        return current
