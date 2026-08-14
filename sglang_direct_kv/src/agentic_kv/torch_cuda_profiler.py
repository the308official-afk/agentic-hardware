from __future__ import annotations

import atexit
import os
import signal
import time
from pathlib import Path
from typing import Any


_PROFILER: Any | None = None
_STARTED = False
_STOPPED = False
_EVENT_COUNT = 0
_SIGNALS_INSTALLED = False
_PREVIOUS_SIGNAL_HANDLERS: dict[int, Any] = {}


def enabled() -> bool:
    return os.environ.get("AGENTIC_KV_TORCH_PROFILER_ENABLE", "0") == "1"


def _profile_dir() -> Path:
    return Path(os.environ.get("AGENTIC_KV_TORCH_PROFILER_DIR", "artifacts/torch_cuda_profiles"))


def _stop_after_events() -> int:
    try:
        return int(os.environ.get("AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS", "0"))
    except ValueError:
        return 0


def _write_status(event: dict[str, Any]) -> None:
    path = _profile_dir() / f"torch_profiler_status_pid{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event.setdefault("ts_ns", time.time_ns())
    event.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as f:
        import json

        f.write(json.dumps(event, sort_keys=True) + "\n")


def maybe_start(label: str) -> None:
    global _PROFILER, _STARTED, _STOPPED
    if not enabled() or _STARTED or _STOPPED:
        return
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile

        if not torch.cuda.is_available():
            _write_status({"event": "torch_profiler.skip", "reason": "cuda_unavailable", "label": label})
            _STOPPED = True
            return

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        _PROFILER = profile(
            activities=activities,
            record_shapes=os.environ.get("AGENTIC_KV_TORCH_PROFILER_RECORD_SHAPES", "0") == "1",
            profile_memory=os.environ.get("AGENTIC_KV_TORCH_PROFILER_PROFILE_MEMORY", "1") == "1",
            with_stack=os.environ.get("AGENTIC_KV_TORCH_PROFILER_WITH_STACK", "0") == "1",
        )
        _PROFILER.start()
        _STARTED = True
        _install_signal_handlers()
        _write_status({"event": "torch_profiler.start", "label": label})
    except Exception as exc:
        _STOPPED = True
        _write_status(
            {
                "event": "torch_profiler.error",
                "phase": "start",
                "label": label,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def record_event(label: str) -> None:
    global _EVENT_COUNT
    if not _STARTED or _STOPPED or _PROFILER is None:
        return
    _EVENT_COUNT += 1
    try:
        _PROFILER.step()
    except Exception:
        pass
    stop_after = _stop_after_events()
    if stop_after > 0 and _EVENT_COUNT >= stop_after:
        stop_and_export(f"event_limit_{stop_after}", label)


def stop_and_export(reason: str = "manual", label: str = "") -> None:
    global _STOPPED
    if not _STARTED or _STOPPED or _PROFILER is None:
        return
    _STOPPED = True
    out_dir = _profile_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"torch_cuda_profile_pid{os.getpid()}_{int(time.time())}_{reason}"
    trace_path = out_dir / f"{stem}.json"
    table_path = out_dir / f"{stem}_key_averages.txt"
    try:
        _PROFILER.stop()
        _PROFILER.export_chrome_trace(str(trace_path))
        try:
            table = _PROFILER.key_averages().table(sort_by="cuda_time_total", row_limit=80)
            table_path.write_text(table + "\n", encoding="utf-8")
        except Exception as exc:
            _write_status(
                {
                    "event": "torch_profiler.error",
                    "phase": "key_averages",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        _write_status(
            {
                "event": "torch_profiler.export",
                "reason": reason,
                "label": label,
                "event_count": _EVENT_COUNT,
                "trace_path": str(trace_path),
                "table_path": str(table_path),
            }
        )
    except Exception as exc:
        _write_status(
            {
                "event": "torch_profiler.error",
                "phase": "stop_export",
                "reason": reason,
                "label": label,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def _handle_signal(signum: int, frame: Any) -> None:
    stop_and_export(f"signal_{signum}", "shutdown")
    previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum)
    if callable(previous):
        previous(signum, frame)
        return
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def _install_signal_handlers() -> None:
    global _SIGNALS_INSTALLED
    if _SIGNALS_INSTALLED:
        return
    signums = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGQUIT"):
        signums.append(signal.SIGQUIT)
    for signum in signums:
        try:
            _PREVIOUS_SIGNAL_HANDLERS[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        except Exception:
            pass
    _SIGNALS_INSTALLED = True


atexit.register(lambda: stop_and_export("atexit"))
