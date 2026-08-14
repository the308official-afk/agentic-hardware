from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any


def enabled() -> bool:
    return os.environ.get("AGENTIC_KV_NVTX_ENABLE", "0") == "1"


def _torch_nvtx() -> Any | None:
    if not enabled():
        return None
    try:
        import torch

        return torch.cuda.nvtx
    except Exception:
        return None


@contextlib.contextmanager
def range_scope(message: str) -> Iterator[None]:
    nvtx = _torch_nvtx()
    if nvtx is None:
        yield
        return

    nvtx.range_push(message)
    try:
        yield
    finally:
        nvtx.range_pop()


def mark(message: str) -> None:
    nvtx = _torch_nvtx()
    if nvtx is None:
        return
    try:
        nvtx.mark(message)
    except Exception:
        pass
