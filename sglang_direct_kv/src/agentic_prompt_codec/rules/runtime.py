from __future__ import annotations

import re
from .base import FullLineRule


def rules():
    return (
        FullLineRule(
            "guard_reason",
            re.compile(r"Guard retry reason: ([A-Za-z_][A-Za-z0-9_]*)", re.I),
            lambda match: f"G({match.group(1)})",
        ),
        FullLineRule(
            "tool_calls",
            re.compile(r"Tool calls: ([A-Za-z0-9_ -]+)", re.I),
            lambda match: f"T({match.group(1)})",
        ),
    )
