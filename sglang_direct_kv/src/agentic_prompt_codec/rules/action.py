from __future__ import annotations

import re
from .base import RegexRule


def rules():
    return (
        RegexRule(
            "read_for_purpose",
            re.compile(r"\b(?:Read|Inspect|Open) (`[^`\n]+`|F\d+) to ([^.\n]+)\.", re.I),
            lambda match: f"R({match.group(1)}; {match.group(2)})",
        ),
        RegexRule(
            "run_for_purpose",
            re.compile(r"\b(?:Run|Execute) (`[^`\n]+`) to ([^.\n]+)\.", re.I),
            lambda match: f"X({match.group(1)}; {match.group(2)})",
        ),
        RegexRule(
            "modify_to_goal",
            re.compile(r"\b(?:Modify|Update|Change) ([^.\n`]{2,90}?) to ([^.\n]+)\.", re.I),
            lambda match: f"M({match.group(1)}; {match.group(2)})",
        ),
        RegexRule(
            "do_not",
            re.compile(r"\bDo not ([^.\n]+)\.", re.I),
            lambda match: f"N({match.group(1)})",
        ),
        RegexRule(
            "ensure",
            re.compile(r"\bEnsure ([^.\n]+)\.", re.I),
            lambda match: f"E({match.group(1)})",
        ),
    )
