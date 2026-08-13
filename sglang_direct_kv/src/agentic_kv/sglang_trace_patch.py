from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


_INSTALLED = False


def _trace_path() -> Path:
    return Path(os.environ.get("AGENTIC_KV_TRACE_PATH", "artifacts/kv_movement_trace.jsonl"))


def _tensor_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            summary["shape"] = [int(dim) for dim in shape]
        except Exception:
            summary["shape"] = str(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        summary["device"] = str(device)
    try:
        summary["numel"] = int(value.numel())
    except Exception:
        pass
    try:
        if "numel" in summary and summary["numel"] <= 16:
            summary["values"] = value.detach().cpu().tolist()
    except Exception:
        pass
    return summary


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _safe_summary(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            return {"type": type(value).__name__, "len": len(value)}
        return [_safe_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _safe_summary(v) for k, v in list(value.items())[:16]}
    if hasattr(value, "shape") and hasattr(value, "numel"):
        return _tensor_summary(value)
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _interesting_attr_summary(obj: Any) -> dict[str, Any]:
    """Collect small, stable-looking cache metadata without walking large graphs."""

    attrs = (
        "size",
        "page_size",
        "dtype",
        "device",
        "mem_layout",
        "hicache_size",
        "write_policy",
        "disable",
        "enable_hierarchical_cache",
    )
    nested_attrs = (
        "device_pool",
        "host_pool",
        "token_to_kv_pool",
        "token_to_kv_pool_host",
        "req_to_token_pool",
        "tree_cache",
        "hicache_controller",
    )
    out: dict[str, Any] = {
        "object_id": hex(id(obj)),
        "object_type": type(obj).__name__,
    }
    for attr in attrs:
        if hasattr(obj, attr):
            try:
                out[attr] = _safe_summary(getattr(obj, attr))
            except Exception:
                pass
    for attr in nested_attrs:
        if not hasattr(obj, attr):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        nested: dict[str, Any] = {
            "type": type(value).__name__,
            "object_id": hex(id(value)),
        }
        value_len = _safe_len(value)
        if value_len is not None:
            nested["len"] = value_len
        for nested_attr in ("size", "page_size", "dtype", "device"):
            if hasattr(value, nested_attr):
                try:
                    nested[nested_attr] = _safe_summary(getattr(value, nested_attr))
                except Exception:
                    pass
        out[attr] = nested
    return out


def _result_metadata(result: Any) -> dict[str, Any]:
    if result is None:
        return {"result_type": "None"}
    metadata = {"result_type": type(result).__name__}
    result_len = _safe_len(result)
    if result_len is not None:
        metadata["result_len"] = result_len
    if isinstance(result, tuple):
        metadata["tuple_item_types"] = [type(item).__name__ for item in result[:8]]
    return metadata


def _write_event(event: dict[str, Any]) -> None:
    event.setdefault("ts_ns", time.time_ns())
    event.setdefault("pid", os.getpid())
    path = _trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _wrap_method(cls: type, method_name: str, event_name: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_agentic_kv_wrapped", False):
        return

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_ns = time.perf_counter_ns()
        start_event = {
            "event": f"{event_name}.start",
            "class": cls.__name__,
            "method": method_name,
            "self": _interesting_attr_summary(self),
            "args": [_safe_summary(arg) for arg in args],
            "kwargs": {key: _safe_summary(value) for key, value in kwargs.items()},
        }
        _write_event(start_event)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            _write_event(
                {
                    "event": f"{event_name}.error",
                    "class": cls.__name__,
                    "method": method_name,
                    "duration_ms": (time.perf_counter_ns() - start_ns) / 1_000_000,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

        _write_event(
            {
                "event": f"{event_name}.end",
                "class": cls.__name__,
                "method": method_name,
                "duration_ms": (time.perf_counter_ns() - start_ns) / 1_000_000,
                "self": _interesting_attr_summary(self),
                "result_metadata": _result_metadata(result),
                "result": _safe_summary(result),
            }
        )
        return result

    wrapper._agentic_kv_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapper)


def _try_patch(importer: Callable[[], Any], class_name: str, methods: dict[str, str]) -> None:
    try:
        module = importer()
        cls = getattr(module, class_name)
    except Exception as exc:
        if os.environ.get("AGENTIC_KV_TRACE_DEBUG", "0") == "1":
            _write_event(
                {
                    "event": "trace.patch_skip",
                    "class": class_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        return

    for method_name, event_name in methods.items():
        _wrap_method(cls, method_name, event_name)


def install_sglang_kv_trace() -> None:
    """Install non-invasive SGLang KV movement trace hooks."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _write_event({"event": "trace.install.start"})

    _try_patch(
        lambda: __import__("sglang.srt.managers.cache_controller", fromlist=["HiCacheController"]),
        "HiCacheController",
        {
            "load": "hicache.load",
            "write": "hicache.write",
            "evict_device": "hicache.evict_device",
            "evict_host": "hicache.evict_host",
            "prefetch": "hicache.prefetch",
        },
    )
    _try_patch(
        lambda: __import__("sglang.srt.mem_cache.hiradix_cache", fromlist=["HiRadixCache"]),
        "HiRadixCache",
        {
            "match_prefix": "hiradix.match_prefix",
            "cache_finished_req": "hiradix.cache_finished_req",
            "cache_unfinished_req": "hiradix.cache_unfinished_req",
            "evict": "hiradix.evict",
            "ready_to_load_host_cache": "hiradix.ready_to_load_host_cache",
        },
    )
    _try_patch(
        lambda: __import__("sglang.srt.mem_cache.radix_cache", fromlist=["RadixCache"]),
        "RadixCache",
        {
            "match_prefix": "radix.match_prefix",
            "cache_finished_req": "radix.cache_finished_req",
            "cache_unfinished_req": "radix.cache_unfinished_req",
            "evict": "radix.evict",
        },
    )

    _write_event({"event": "trace.install.end"})
