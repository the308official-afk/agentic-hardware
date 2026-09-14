from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol


CheckBudget = Callable[[], None]


class ShorthandRule(Protocol):
    name: str

    def apply(self, text: str, state: "RuleState", check_budget: CheckBudget) -> str: ...


@dataclass
class RuleState:
    source_text: str
    max_rules: int
    expansions: list[tuple[str, str]] = field(default_factory=list)
    expansion_by_encoded: dict[str, str] = field(default_factory=dict)
    entity_aliases: dict[str, str] = field(default_factory=dict)
    counts: Counter[str] = field(default_factory=Counter)

    def add(self, rule_name: str, encoded: str, original: str, *, entity_alias: bool = False) -> str:
        if encoded == original:
            return original
        if encoded in self.expansion_by_encoded:
            if self.expansion_by_encoded[encoded] != original:
                return original
            if entity_alias:
                self.entity_aliases[encoded] = original
            self.counts[rule_name] += 1
            return encoded
        if encoded in self.source_text or len(self.expansions) >= self.max_rules:
            return original
        self.expansion_by_encoded[encoded] = original
        self.expansions.append((encoded, original))
        if entity_alias:
            self.entity_aliases[encoded] = original
        self.counts[rule_name] += 1
        return encoded

    def expand_entity_aliases(self, text: str) -> str:
        if not self.entity_aliases:
            return text
        out = text
        for alias, original in sorted(self.entity_aliases.items(), key=lambda item: len(item[0]), reverse=True):
            out = out.replace(alias, original)
        return out


@dataclass(frozen=True)
class RegexRule:
    name: str
    pattern: re.Pattern[str]
    make: Callable[[re.Match[str]], str]

    def apply(self, text: str, state: RuleState, check_budget: CheckBudget) -> str:
        def replace(match: re.Match[str]) -> str:
            check_budget()
            return state.add(self.name, self.make(match), state.expand_entity_aliases(match.group(0)))

        return self.pattern.sub(replace, text)


@dataclass(frozen=True)
class FullLineRule:
    name: str
    pattern: re.Pattern[str]
    make: Callable[[re.Match[str]], str]

    def apply(self, text: str, state: RuleState, check_budget: CheckBudget) -> str:
        stripped = text.strip()
        match = self.pattern.fullmatch(stripped)
        if not match:
            return text
        check_budget()
        leading = text[: len(text) - len(text.lstrip())]
        trailing = text[len(text.rstrip()) :]
        return leading + state.add(self.name, self.make(match), state.expand_entity_aliases(stripped)) + trailing
