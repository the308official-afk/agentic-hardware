#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from run_agentbench_sglang_task import install_sglang_chat_model_patch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the existing Deep Agents tool-loop preflight against direct SGLang."
    )
    parser.add_argument(
        "--agentbench-root",
        type=Path,
        default=Path(os.environ.get("AGENTBENCH_ROOT", "../kv_cache_offloading")),
    )
    parser.add_argument("preflight_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = list(args.preflight_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    agentbench_root = args.agentbench_root.expanduser().resolve()
    if not (agentbench_root / "agentbench" / "diagnose_deepagents_tool_loop.py").exists():
        raise SystemExit(
            "Could not find diagnose_deepagents_tool_loop.py under "
            f"{agentbench_root}. Set AGENTBENCH_ROOT or pass --agentbench-root."
        )

    install_sglang_chat_model_patch(agentbench_root)
    if str(agentbench_root) not in sys.path:
        sys.path.insert(0, str(agentbench_root))

    from agentbench import diagnose_deepagents_tool_loop

    sys.argv = [str(agentbench_root / "agentbench" / "diagnose_deepagents_tool_loop.py"), *forwarded]
    diagnose_deepagents_tool_loop.main()


if __name__ == "__main__":
    main()
