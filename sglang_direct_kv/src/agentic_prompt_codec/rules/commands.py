from __future__ import annotations

import re
from dataclasses import dataclass

from .base import CheckBudget, RegexRule, RuleState
from .entities import _best_repeated


VALIDATION_COMMAND = re.compile(
    r"(?:python(?:3)? -m pytest|pytest|npx mocha|npm test|pnpm test|yarn test)[^`\n\"']{6,180}"
)


@dataclass(frozen=True)
class ValidationCommandAliasRule:
    name: str = "validation_command_alias"
    prefix: str = "VC"
    min_count: int = 2
    min_len: int = 20

    def apply_to_parts(
        self,
        parts: list[tuple[str, bool]],
        state: RuleState,
        check_budget: CheckBudget,
    ) -> list[tuple[str, bool]]:
        editable_text = "\n".join(part for part, editable in parts if editable)
        candidates = _best_repeated(
            [match.group(0).strip() for match in VALIDATION_COMMAND.finditer(editable_text)],
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
            original = match.group(0).strip()
            replacement = aliases.get(original)
            if not replacement:
                return match.group(0)
            state.counts[self.name] += 1
            leading = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip())]
            trailing = match.group(0)[len(match.group(0).rstrip()) :]
            return leading + replacement + trailing

        return [
            (VALIDATION_COMMAND.sub(replace, part) if editable else part, editable)
            for part, editable in parts
        ]


def rules():
    return (
        ValidationCommandAliasRule(),
        RegexRule(
            "execute_tool_command",
            re.compile(r"(?<!`)\bexecute\(\"([^\"\n]{3,240})\"\)(?!`)"),
            lambda match: f"TC({match.group(1)})",
        ),
        RegexRule(
            "tool_name",
            re.compile(r"\b(read_file|edit_file|write_file|execute)\b"),
            lambda match: f"TL({match.group(1)})",
        ),
    )
