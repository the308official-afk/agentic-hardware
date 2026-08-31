"""Optional process-wide hooks for the direct KV testbed.

Python imports ``sitecustomize`` automatically when it is present on
``PYTHONPATH``. The SGLang launch scripts add this project's ``src`` directory
to ``PYTHONPATH`` so we can enable trace hooks without editing site-packages.
"""

from __future__ import annotations

import os


if (
    os.environ.get("AGENTIC_KV_TRACE_ENABLE", "0") == "1"
    or os.environ.get("AGENTIC_RUNTIME_TELEMETRY", "0") == "1"
    or os.environ.get("AGENTIC_RUNTIME_TELEMETRY_ENABLE", "0") == "1"
):
    try:
        from agentic_kv.sglang_trace_patch import install_sglang_kv_trace

        install_sglang_kv_trace()
    except Exception as exc:  # pragma: no cover - defensive startup hook
        if os.environ.get("AGENTIC_KV_TRACE_DEBUG", "0") == "1":
            print(f"[agentic-kv-trace] failed to install: {exc}", flush=True)
