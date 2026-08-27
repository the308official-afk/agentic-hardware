from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SGLangHookTarget:
    """A version-specific SGLang class/method group to trace."""

    module: str
    class_name: str
    methods: Mapping[str, str]
    scheduler_required: bool = False


RawEventMap = Mapping[str, str]
