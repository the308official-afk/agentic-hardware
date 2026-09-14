from __future__ import annotations

import re
from .base import FullLineRule


def rules():
    return (
        FullLineRule(
            "phase",
            re.compile(r"Phase: ([A-Za-z_][A-Za-z0-9_]*)"),
            lambda match: f"P({match.group(1)})",
        ),
    )
