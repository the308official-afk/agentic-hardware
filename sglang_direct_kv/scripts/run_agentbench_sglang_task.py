#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing kv_cache_offloading AgentBench single-task harness "
            "against direct SGLang instead of Dynamo."
        )
    )
    parser.add_argument(
        "--agentbench-root",
        type=Path,
        default=Path(os.environ.get("AGENTBENCH_ROOT", "../kv_cache_offloading")),
        help="Path to the kv_cache_offloading repo that contains agentbench/.",
    )
    parser.add_argument(
        "agentbench_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to agentbench/deepagents_swebench_single_host.py.",
    )
    args = parser.parse_args()
    forwarded = list(args.agentbench_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return args, forwarded


def install_sglang_chat_model_patch(agentbench_root: Path) -> None:
    root = agentbench_root.resolve()
    cloned_deepagents = root / "upstream" / "deepagents" / "libs" / "deepagents"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if cloned_deepagents.exists() and str(cloned_deepagents) not in sys.path:
        sys.path.insert(0, str(cloned_deepagents))

    from langchain_openai import ChatOpenAI

    from agentbench.deepagents_app.src import agent as agent_module
    from agentbench.deepagents_app.src.hint_providers import (
        build_agent_context,
        build_annotations,
        supported_agent_hints,
    )

    def apply_tool_choice_override(llm: ChatOpenAI) -> ChatOpenAI:
        existing_override = getattr(agent_module, "apply_tool_choice_override", None)
        if callable(existing_override):
            return existing_override(llm)

        forced_choice = os.environ.get("AGENTBENCH_FORCE_TOOL_CHOICE", "").strip()
        if not forced_choice or forced_choice.lower() in {"0", "false", "none", "off", "auto"}:
            return llm

        original_bind_tools = llm.bind_tools

        @wraps(original_bind_tools)
        def bind_tools_with_forced_choice(tools, *args, **kwargs):
            existing_choice = kwargs.get("tool_choice")
            if existing_choice is None or str(existing_choice).strip().lower() == "auto":
                kwargs["tool_choice"] = forced_choice
            return original_bind_tools(tools, *args, **kwargs)

        object.__setattr__(llm, "bind_tools", bind_tools_with_forced_choice)
        return llm

    def build_sglang_chat_model(
        *,
        frontend_url: str,
        model: str,
        hint_payload: dict[str, Any] | None = None,
        request_context: dict[str, Any] | None = None,
        max_tokens: int = 2048,
    ) -> ChatOpenAI:
        full_hint_payload = hint_payload or {}
        hints = supported_agent_hints(full_hint_payload)
        context = request_context or {}
        phase = str(context.get("phase") or full_hint_payload.get("agent_phase") or "agentbench")
        request_id = str(context.get("request_id") or "")
        parent_run_id = str(context.get("parent_run_id") or "agentbench")
        task_instance_id = str(context.get("task_instance_id") or "")
        sequence_index = context.get("sequence_index", context.get("step_index", ""))
        session_parts = [parent_run_id, phase]
        if sequence_index not in ("", None):
            session_parts.append(str(sequence_index))
        session_id = request_id or "::".join(session_parts)
        agentic_kv = {
            "session_id": session_id,
            "phase": phase,
            "label": f"{parent_run_id}:{phase}",
            "mode": os.environ.get("AGENTBENCH_SGLANG_PREFETCH_MODE", "live_direct"),
            "priority": hints.get("priority", full_hint_payload.get("priority", "normal")),
            "task_id": task_instance_id,
            "parent_run_id": parent_run_id,
            "hint_probe_id": full_hint_payload.get("hint_probe_id"),
            "reuse_likelihood": full_hint_payload.get("reuse_likelihood"),
        }
        agentic_kv = {key: value for key, value in agentic_kv.items() if value not in (None, "", [], {})}

        extra_body: dict[str, Any] = {
            "custom_params": {
                "agentic_kv": agentic_kv,
                "request_context": context,
                "agent_hints": hints,
            }
        }
        if os.environ.get("AGENTBENCH_SGLANG_SEND_NVEXT", "0") == "1":
            extra_body["nvext"] = {
                "request_context": context,
                "agent_context": build_agent_context(context),
                "annotations": build_annotations(context, full_hint_payload),
            }
            if hints:
                extra_body["nvext"]["agent_hints"] = hints

        llm = ChatOpenAI(
            model=model,
            base_url=agent_module.frontend_base_url(frontend_url),
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=300,
            extra_body=extra_body,
        )
        return apply_tool_choice_override(llm)

    agent_module.build_dynamo_chat_model = build_sglang_chat_model


def main() -> None:
    args, forwarded = parse_wrapper_args()
    agentbench_root = args.agentbench_root.expanduser().resolve()
    if not (agentbench_root / "agentbench" / "deepagents_swebench_single_host.py").exists():
        raise SystemExit(
            "Could not find AgentBench single-task runner under "
            f"{agentbench_root}. Set AGENTBENCH_ROOT or pass --agentbench-root."
        )

    install_sglang_chat_model_patch(agentbench_root)
    if str(agentbench_root) not in sys.path:
        sys.path.insert(0, str(agentbench_root))

    from agentbench import deepagents_swebench_single_host

    sys.argv = [str(agentbench_root / "agentbench" / "deepagents_swebench_single_host.py"), *forwarded]
    deepagents_swebench_single_host.main()


if __name__ == "__main__":
    main()
