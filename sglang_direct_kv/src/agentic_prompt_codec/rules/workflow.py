from __future__ import annotations

import re

from .base import FullLineRule


def rules():
    return (
        FullLineRule(
            "markdown_phase_heading",
            re.compile(r"## (planning|execution|patch_generation|review|synthesis)", re.I),
            lambda match: f"PH({match.group(1)})",
        ),
        FullLineRule(
            "markdown_step_heading",
            re.compile(r"#{2,4} Step ([0-9]+): ([^\n]{2,120})", re.I),
            lambda match: f"SH({match.group(1)}; {match.group(2)})",
        ),
        FullLineRule(
            "numbered_bold_step",
            re.compile(r"([0-9]+)\. \*\*([^*\n]{2,100})\*\*"),
            lambda match: f"SB({match.group(1)}; {match.group(2)})",
        ),
    )
