"""Lossless, rule-based shorthand for common coding-agent trace relations."""
import re
from ..models import EncodedSegment
from ..rules.base import RuleState
from ..rules.registry import DEFAULT_AGENT_TRACE_RULES, LEGEND_BY_RULE, build_rules

PROTECTED_BLOCKS = re.compile(
    r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)"
    r"|^[ \t]*(?:[\[{]|\$ |>>> |HARNESS_REPLAY_EXPERIMENT_JSON:).*$",
    re.MULTILINE,
)


def editable_parts(text: str) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    start = 0
    for match in PROTECTED_BLOCKS.finditer(text):
        out.append((text[start:match.start()], True))
        out.append((match.group(), False))
        start = match.end()
    out.append((text[start:], True))
    return out


def render(body: str, state: RuleState, enabled_rules: tuple[str, ...]) -> EncodedSegment:
    expansions = state.expansions
    if not expansions:
        return EncodedSegment(body, body)
    alias_lines = [f"{alias}={original}" for alias, original in state.entity_aliases.items()]
    legend_lines = [LEGEND_BY_RULE[name] for name in enabled_rules if name in state.counts]
    if alias_lines:
        legend_lines = ["Request-local entity aliases:"] + alias_lines + legend_lines
    legend = (
        "Relational shorthand legend. Expand these request-local forms exactly:\n"
        + "\n".join(legend_lines)
        + "\nText:\n"
    )
    return EncodedSegment(
        legend + body,
        body,
        tuple(expansions),
        legend,
        True,
        tuple(sorted(state.counts.items())),
    )


class AgentTraceRelationsCodec:
    name = "agent_trace_relations_v1"

    def encode(self, text, config, counter, check_budget):
        parts = editable_parts(text)
        body_parts: list[str] = []
        enabled_rules = tuple(config.enabled_rules or DEFAULT_AGENT_TRACE_RULES)
        rules = build_rules(enabled_rules)
        state = RuleState(text, config.max_shorthand_rules)
        for rule in rules:
            apply_to_parts = getattr(rule, "apply_to_parts", None)
            if apply_to_parts is not None:
                parts = apply_to_parts(parts, state, check_budget)

        for part, editable in parts:
            check_budget()
            if not editable:
                body_parts.append(part)
                continue
            lines = []
            for line in part.splitlines(keepends=True):
                ending = ""
                core = line
                if core.endswith("\r\n"):
                    core, ending = core[:-2], "\r\n"
                elif core.endswith("\n"):
                    core, ending = core[:-1], "\n"
                replacement = core
                for rule in rules:
                    if getattr(rule, "apply_to_parts", None) is not None:
                        continue
                    updated = rule.apply(replacement, state, check_budget)
                    check_budget()
                    if updated != replacement:
                        replacement = updated
                        break
                lines.append(replacement + ending)
            body_parts.append("".join(lines))

        return render("".join(body_parts), state, enabled_rules)
