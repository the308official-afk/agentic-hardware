#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    def install_safe_edit_guard() -> None:
        if not env_flag("AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD", default=True):
            return

        try:
            from deepagents.backends import filesystem as filesystem_backend
            from deepagents.backends import state as state_backend
            from deepagents.backends import store as store_backend
            from deepagents.backends import utils as backend_utils
        except Exception:
            if env_flag("AGENTBENCH_DIRECT_SGLANG_REQUIRE_SAFE_EDIT_GUARD", default=False):
                raise
            return

        if getattr(backend_utils, "_direct_sglang_safe_edit_guard_installed", False):
            return

        original_replacement = backend_utils.perform_string_replacement

        @wraps(original_replacement)
        def guarded_replacement(
            content: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ):
            if not isinstance(old_string, str):
                return (
                    "Error: edit_file old_string must be a string. First call read_file, "
                    "then provide a non-empty exact text snippet from that file."
                )
            if old_string == "":
                return (
                    "Error: empty old_string is forbidden. First call read_file on the "
                    "target file, then retry edit_file with a non-empty exact snippet "
                    "copied from that file. Do not use replace_all=True with an empty "
                    "old_string."
                )
            return original_replacement(content, old_string, new_string, replace_all)

        backend_utils.perform_string_replacement = guarded_replacement
        filesystem_backend.perform_string_replacement = guarded_replacement
        state_backend.perform_string_replacement = guarded_replacement
        store_backend.perform_string_replacement = guarded_replacement
        backend_utils._direct_sglang_safe_edit_guard_installed = True

    def install_direct_sglang_harness_profile() -> None:
        if getattr(agent_module, "_direct_sglang_harness_profile_installed", False):
            return

        excluded_tools: set[str] = set()
        if env_flag("AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS", default=True):
            excluded_tools.add("write_todos")

        try:
            from deepagents.profiles import HarnessProfile, register_harness_profile

            register_harness_profile(
                "openai",
                HarnessProfile(
                    excluded_tools=frozenset(excluded_tools),
                    tool_description_overrides={
                        "ls": (
                            "List directory contents. Use this only for directories such as `/`, "
                            "`/src`, or `src/database`. Do not use ls to inspect a file path; "
                            "use read_file for files."
                        ),
                        "read_file": (
                            "Read a file from the repository. Use this for paths such as "
                            "`src/database/mongo/main.js`, `src/user/email.js`, or any path "
                            "that looks like a file."
                        ),
                        "edit_file": (
                            "Edit an existing file by replacing exact text. First read the file. "
                            "The old_string argument must be non-empty exact text copied from "
                            "that file. Never use an empty old_string and never use replace_all "
                            "with an empty old_string."
                        ),
                    },
                ),
            )
        except Exception:
            if env_flag("AGENTBENCH_DIRECT_SGLANG_REQUIRE_PROFILE", default=False):
                raise

        agent_module._direct_sglang_harness_profile_installed = True

    def install_virtual_tool_root_prompt_patch() -> None:
        if not env_flag("AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT", default=True):
            return
        if getattr(agent_module, "_direct_sglang_virtual_tool_root_patch_installed", False):
            return

        original_format_task_prompt = agent_module.format_swebench_task_prompt

        @wraps(original_format_task_prompt)
        def format_swebench_task_prompt_with_virtual_root(task: dict) -> str:
            prompt = original_format_task_prompt(task)
            workspace_path = str(task.get("workspace_path", "")).strip()
            if not workspace_path:
                return prompt

            old_notes = (
                f"You have a writable local workspace at:\n{workspace_path}\n\n"
                "Use the available filesystem and shell tools to inspect the repo, edit files when needed, "
                "run focused validation, and leave the workspace in a state where a git diff can be captured."
            )
            new_notes = (
                "The SWE-bench repository is mounted at `/` inside the available filesystem and shell tools.\n\n"
                "Use `/` or relative paths such as `src/...` when calling ls, read_file, grep, "
                "edit_file, write_file, or execute. Do not pass the host checkout path "
                f"`{workspace_path}` to tools; that path is only run metadata.\n\n"
                "Use `ls` only for directories. If a path looks like a file, such as "
                "`src/database/mongo/main.js`, call `read_file` on that path instead of `ls`. "
                "An empty `ls` result on a file path is not proof that the file is missing.\n\n"
                "Before calling `edit_file` or `write_file`, first call `read_file` on the exact file. "
                "For `edit_file`, `old_string` must be a non-empty exact snippet copied from the file. "
                "Never use an empty `old_string`, and never use `replace_all=True` with an empty string.\n\n"
                "Use the available filesystem and shell tools to inspect the repo, edit files when needed, "
                "run focused validation, and leave the workspace in a state where a git diff can be captured."
            )
            if old_notes in prompt:
                return prompt.replace(old_notes, new_notes)
            if "Workspace:" in prompt:
                return (
                    f"{prompt.rstrip()}\n\n"
                    "Direct-SGLang workspace path note:\n"
                    "- The SWE-bench repository is mounted at `/` inside tools.\n"
                    "- Use `/` or relative paths such as `src/...`.\n"
                    "- Use `ls` for directories and `read_file` for files.\n"
                    "- Empty `ls` output on a file path is not a blocker; call `read_file`.\n"
                    "- Read a file before editing it.\n"
                    "- Never call `edit_file` with an empty `old_string`.\n"
                    "- Never use `replace_all=True` with an empty `old_string`.\n"
                    f"- Do not pass the host checkout path `{workspace_path}` to tools.\n"
                )
            return prompt

        agent_module.format_swebench_task_prompt = format_swebench_task_prompt_with_virtual_root
        agent_module._direct_sglang_virtual_tool_root_patch_installed = True

    def install_tool_rich_prompt_patch() -> None:
        if not env_flag("AGENTBENCH_DIRECT_SGLANG_TOOL_RICH", default=False):
            return
        if getattr(agent_module, "_direct_sglang_tool_rich_patch_installed", False):
            return

        note = (
            "Direct-SGLang tool-use requirement:\n"
            "- If you need to inspect, read, search, edit, write, run, or validate, "
            "emit the actual structured tool call in this assistant turn.\n"
            "- Do not say that you will call ls, read_file, grep, edit_file, write_file, "
            "or execute next; call the tool now.\n"
            "- The repo root inside tools is `/`; use `/` or relative paths such as `src/...`, "
            "not the host checkout path.\n"
            "- Use ls only for directories. For file paths, use read_file; do not treat empty "
            "ls output on a file path as a blocker.\n"
            "- Before edit_file or write_file, read the exact file first.\n"
            "- For edit_file, old_string must be non-empty exact file text; never use empty "
            "old_string or replace_all=True with empty old_string.\n"
            "- A prose-only response that says you should inspect or run a command is not useful."
        )

        original_system_prompt = str(getattr(agent_module, "SYSTEM_PROMPT", ""))
        if note not in original_system_prompt:
            agent_module.SYSTEM_PROMPT = f"{original_system_prompt.rstrip()}\n\n{note}"

        def append_note(prompt: str, *, must_call: bool) -> str:
            if note in prompt:
                return prompt
            extra = note
            if must_call:
                extra += "\n- This step should produce a tool call unless a previous tool result shows a concrete blocker."
            return f"{prompt.rstrip()}\n\n{extra}"

        original_build_phase_prompt = agent_module.build_phase_prompt

        @wraps(original_build_phase_prompt)
        def build_phase_prompt_with_tool_rich_note(*args, **kwargs):
            prompt = original_build_phase_prompt(*args, **kwargs)
            phase = str(kwargs.get("phase") or "")
            if phase in {"execution", "patch_generation", "review"}:
                return append_note(prompt, must_call=True)
            return prompt

        agent_module.build_phase_prompt = build_phase_prompt_with_tool_rich_note

        original_build_execution_loop_prompt = agent_module.build_execution_loop_prompt

        @wraps(original_build_execution_loop_prompt)
        def build_execution_loop_prompt_with_tool_rich_note(*args, **kwargs):
            prompt = original_build_execution_loop_prompt(*args, **kwargs)
            return append_note(prompt, must_call=True)

        agent_module.build_execution_loop_prompt = build_execution_loop_prompt_with_tool_rich_note

        original_build_execution_retry_prompt = agent_module.build_execution_retry_prompt

        @wraps(original_build_execution_retry_prompt)
        def build_execution_retry_prompt_with_tool_rich_note(*args, **kwargs):
            prompt = original_build_execution_retry_prompt(*args, **kwargs)
            return append_note(prompt, must_call=True)

        agent_module.build_execution_retry_prompt = build_execution_retry_prompt_with_tool_rich_note
        agent_module._direct_sglang_tool_rich_patch_installed = True

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
        max_tokens_cap = os.environ.get("AGENTBENCH_DIRECT_SGLANG_MAX_TOKENS", "").strip()
        if max_tokens_cap:
            max_tokens = min(max_tokens, int(max_tokens_cap))
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
    install_safe_edit_guard()
    install_direct_sglang_harness_profile()
    install_virtual_tool_root_prompt_patch()
    install_tool_rich_prompt_patch()


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
