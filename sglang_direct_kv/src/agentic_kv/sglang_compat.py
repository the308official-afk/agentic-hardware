from __future__ import annotations

import os
from typing import Any


def enable_priority_radix_eviction_choice() -> bool:
    """Expose SGLang's built-in PriorityStrategy as a launch choice.

    SGLang 0.5.10.post1 includes `PriorityStrategy` in the radix cache, but its
    public `--radix-eviction-policy` choices omit `priority`. This registers the
    choice at process startup without editing site-packages or changing the
    installed SGLang version.
    """

    try:
        from sglang.srt import server_args
        from sglang.srt.mem_cache import evict_policy, radix_cache
    except Exception:
        return False

    if not hasattr(evict_policy, "PriorityStrategy"):
        return False
    if "PriorityStrategy" not in str(getattr(radix_cache, "PriorityStrategy", "")):
        return False

    choices: Any = getattr(server_args, "RADIX_EVICTION_POLICY_CHOICES", None)
    if not isinstance(choices, list):
        return False
    if "priority" not in choices:
        choices.append("priority")
    return "priority" in choices


def maybe_enable_priority_radix_eviction_choice() -> None:
    if os.environ.get("AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE", "0") != "1":
        return
    enabled = enable_priority_radix_eviction_choice()
    if not enabled and os.environ.get("AGENTIC_KV_TRACE_DEBUG", "0") == "1":
        print("[agentic-kv-compat] priority radix eviction choice not enabled", flush=True)
