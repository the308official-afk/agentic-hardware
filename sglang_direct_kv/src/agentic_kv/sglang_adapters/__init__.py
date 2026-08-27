from __future__ import annotations

import re
from importlib import metadata
from typing import Iterable

from .base import SGLangHookTarget


def installed_sglang_version() -> str:
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return ""


def select_adapter_name(version: str | None = None) -> str:
    parsed = version or installed_sglang_version()
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", parsed)
    if not match:
        return "v0510"
    major, minor, patch = (int(part) for part in match.groups())
    if (major, minor, patch) >= (0, 5, 11):
        return "v0511"
    return "v0510"


def _adapter_module(version: str | None = None):
    adapter_name = select_adapter_name(version)
    if adapter_name == "v0511":
        from . import v0511

        return v0511
    from . import v0510

    return v0510


def get_hook_targets(
    *,
    include_scheduler: bool = False,
    version: str | None = None,
) -> tuple[SGLangHookTarget, ...]:
    module = _adapter_module(version)
    targets: Iterable[SGLangHookTarget] = module.HOOK_TARGETS
    if include_scheduler:
        return tuple(targets)
    return tuple(target for target in targets if not target.scheduler_required)


def get_raw_event_map(version: str | None = None) -> dict[str, str]:
    module = _adapter_module(version)
    return dict(module.RAW_EVENT_MAP)
