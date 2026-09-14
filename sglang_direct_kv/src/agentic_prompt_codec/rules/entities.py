from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .base import CheckBudget, RuleState


BACKTICK_FILE = re.compile(r"`((?:[\w.~-]+/)+[\w./~-]+)`")
ABSOLUTE_REPO_ROOT = re.compile(r"/(?:[\w.~-]+/){2,}agentbench/repos/[\w.~-]+")
FUNCTION_NAME = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)?(?:\.[A-Za-z_][A-Za-z0-9_]*)?(?=\()")


def _best_repeated(matches: list[str], *, min_count: int, min_len: int) -> list[tuple[str, int]]:
    counts = Counter(value for value in matches if len(value) >= min_len)
    return sorted(
        [(value, count) for value, count in counts.items() if count >= min_count],
        key=lambda item: (item[1] * len(item[0]), item[1], len(item[0]), item[0]),
        reverse=True,
    )


@dataclass(frozen=True)
class RepeatedEntityAliasRule:
    name: str
    pattern: re.Pattern[str]
    prefix: str
    min_count: int = 2
    min_len: int = 12

    def apply_to_parts(
        self,
        parts: list[tuple[str, bool]],
        state: RuleState,
        check_budget: CheckBudget,
    ) -> list[tuple[str, bool]]:
        editable_text = "\n".join(part for part, editable in parts if editable)
        raw_matches = [match.group(0) for match in self.pattern.finditer(editable_text)]
        candidates = _best_repeated(raw_matches, min_count=self.min_count, min_len=self.min_len)
        aliases: dict[str, str] = {}
        next_index = 1
        for original, _count in candidates:
            check_budget()
            while f"{self.prefix}{next_index}" in state.source_text:
                next_index += 1
            alias = f"{self.prefix}{next_index}"
            encoded = state.add(self.name, alias, state.expand_entity_aliases(original), entity_alias=True)
            if encoded != original:
                aliases[original] = encoded
                state.counts[self.name] -= 1
                next_index += 1
            if len(state.expansions) >= state.max_rules:
                break
        if not aliases:
            return parts

        def replace(match: re.Match[str]) -> str:
            check_budget()
            replacement = aliases.get(match.group(0), match.group(0))
            if replacement != match.group(0):
                state.counts[self.name] += 1
            return replacement

        return [
            (self.pattern.sub(replace, part) if editable else part, editable)
            for part, editable in parts
        ]


@dataclass(frozen=True)
class WorkspaceRootAliasRule:
    name: str = "workspace_root_alias"
    prefix: str = "WR"
    min_count: int = 2
    min_len: int = 40

    def apply_to_parts(
        self,
        parts: list[tuple[str, bool]],
        state: RuleState,
        check_budget: CheckBudget,
    ) -> list[tuple[str, bool]]:
        editable_text = "\n".join(part for part, editable in parts if editable)
        candidates = _best_repeated(
            ABSOLUTE_REPO_ROOT.findall(editable_text),
            min_count=self.min_count,
            min_len=self.min_len,
        )
        aliases: dict[str, str] = {}
        next_index = 1
        for original, _count in candidates:
            check_budget()
            while f"{self.prefix}{next_index}" in state.source_text:
                next_index += 1
            alias = f"{self.prefix}{next_index}"
            encoded = state.add(self.name, alias, original, entity_alias=True)
            if encoded != original:
                aliases[original] = encoded
                state.counts[self.name] -= 1
                next_index += 1
            if len(state.expansions) >= state.max_rules:
                break
        if not aliases:
            return parts

        def replace(match: re.Match[str]) -> str:
            check_budget()
            replacement = aliases.get(match.group(0), match.group(0))
            if replacement != match.group(0):
                state.counts[self.name] += 1
            return replacement

        return [
            (ABSOLUTE_REPO_ROOT.sub(replace, part) if editable else part, editable)
            for part, editable in parts
        ]


def rules():
    return (
        WorkspaceRootAliasRule(),
        RepeatedEntityAliasRule("file_path_alias", BACKTICK_FILE, "F", min_count=2, min_len=14),
        RepeatedEntityAliasRule("function_alias", FUNCTION_NAME, "FN", min_count=3, min_len=10),
    )
