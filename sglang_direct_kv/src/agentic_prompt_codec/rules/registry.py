from __future__ import annotations

from . import action, commands, entities, phase, runtime, workflow

DEFAULT_AGENT_TRACE_RULES = (
    "workspace_root_alias",
    "file_path_alias",
    "function_alias",
    "validation_command_alias",
    "execute_tool_command",
    "markdown_phase_heading",
    "markdown_step_heading",
    "numbered_bold_step",
    "phase",
    "guard_reason",
    "tool_calls",
    "read_for_purpose",
    "run_for_purpose",
    "modify_to_goal",
    "do_not",
    "ensure",
)

LEGEND_BY_RULE = {
    "workspace_root_alias": "WR#=request-local alias for the exact workspace root path shown.",
    "file_path_alias": "F#=request-local alias for the exact file path shown.",
    "function_alias": "FN#=request-local alias for the exact function or method name shown.",
    "validation_command_alias": "VC#=request-local alias for the exact validation command shown.",
    "execute_tool_command": "TC(command)=execute(\"command\")",
    "tool_name": "TL(name)=the literal tool name.",
    "markdown_phase_heading": "PH(x)=markdown phase heading: ## x",
    "markdown_step_heading": "SH(n; x)=markdown step heading: ### Step n: x",
    "numbered_bold_step": "SB(n; x)=numbered bold step heading: n. **x**",
    "phase": "P(x)=Phase: x",
    "do_not": "N(x)=Do not x.",
    "ensure": "E(x)=Ensure x.",
    "guard_reason": "G(x)=Guard retry reason: x",
    "tool_calls": "T(x)=Tool calls: x",
    "read_for_purpose": "R(file; purpose)=Read file to purpose.",
    "run_for_purpose": "X(command; purpose)=Run command to purpose.",
    "modify_to_goal": "M(entity; goal)=Modify entity to goal.",
}


def _all_rules():
    out = {}
    for module in (entities, commands, workflow, phase, runtime, action):
        for rule in module.rules():
            out[rule.name] = rule
    return out


def build_rules(enabled: tuple[str, ...] | list[str] | None = None):
    available = _all_rules()
    names = tuple(enabled or DEFAULT_AGENT_TRACE_RULES)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError("unknown shorthand rules: " + ", ".join(missing))
    return tuple(available[name] for name in names)
