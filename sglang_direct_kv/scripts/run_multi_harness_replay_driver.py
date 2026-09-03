#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from run_real_prompt_controlled_replay import make_pressure_filler_prompt, make_shared_prefix, prompt_hash

MARKER = "HARNESS_REPLAY_EXPERIMENT_JSON:"
SUPPORTED_HARNESSES = (
    "hatcher",
    "codex",
    "claude_code",
    "opencode",
    "qwen_code",
    "nemo_agent_toolkit",
    "deepseek_harness",
    "pi_agent_harness",
    "openclaw",
    "hermes_agent",
)
SUPPORTED_MODES = (
    "no_prefetch",
    "e2e_priority_hints",
    "pre_harness_priority_hints",
    "nat_inferred_priority_hints",
    "e2e_priority_hints_speculative_prefill",
    "no_cache_signal",
    "harness_native_cache_lowered",
    "harness_emitted_signals",
)

NAT_INFERRED_PRIORITY_MODE = "nat_inferred_priority_hints"
HARNESS_NATIVE_CACHE_MODE = "harness_native_cache_lowered"
HARNESS_EMITTED_SIGNAL_MODE = "harness_emitted_signals"
NAT_INFERRED_PRIORITY_NODES = (
    {
        "workflow_node": "initial_turn",
        "workflow_node_goal": "normal first model turn before the tool wait",
        "expected_inferred_priority": 2,
    },
    {
        "workflow_node": "pressure_filler_background",
        "workflow_node_goal": "background pressure work used to occupy the backend",
        "expected_inferred_priority": 2,
    },
    {
        "workflow_node": "replay_after_tool_wait",
        "workflow_node_goal": "deadline-sensitive replay after a tool wait",
        "expected_inferred_priority": 100,
    },
)


@dataclass(frozen=True)
class HarnessPair:
    session_id: str
    prompt: str
    warmup_prompt: str
    replay_prompt: str
    task_index: str
    prompt_tokens: int


def write_trace(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts_ns", time.time_ns())
    row.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def trace_has_event(path: Path, label: str, event: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == event and row.get("label") == label:
                    return True
    except OSError:
        return False
    return False


def trace_event_row(path: Path, label: str, event: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == event and row.get("label") == label:
                    return row
    except OSError:
        return {}
    return {}


def marker(meta: dict[str, Any]) -> str:
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MARKER + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def pre_harness_priority_enabled(mode: str) -> bool:
    return mode == "pre_harness_priority_hints"


def nat_inferred_priority_enabled(mode: str) -> bool:
    return mode == NAT_INFERRED_PRIORITY_MODE


def harness_emitted_signal_mode(mode: str) -> bool:
    return mode == HARNESS_EMITTED_SIGNAL_MODE


def nat_inferred_node_for_phase(phase: str) -> dict[str, Any]:
    if phase == "replay":
        name = "replay_after_tool_wait"
    elif phase == "pressure_filler":
        name = "pressure_filler_background"
    else:
        name = "initial_turn"
    return next(node for node in NAT_INFERRED_PRIORITY_NODES if node["workflow_node"] == name)


def nat_inferred_priority_profile() -> dict[str, Any]:
    return {
        "schema": "nat_inferred_priority_profile.v1",
        "frontend_priority_intent": "absent",
        "inference_source": "nat_prediction_trie.workflow_path.latency_sensitivity",
        "priority_semantics": {
            "2": "background or low sensitivity workflow step",
            "100": "urgent or user-visible workflow step",
        },
        "workflow_nodes": [
            {
                "workflow_node": str(node["workflow_node"]),
                "workflow_path": [str(node["workflow_node"])],
                "workflow_node_goal": str(node["workflow_node_goal"]),
                "latency_sensitivity": int(node["expected_inferred_priority"]),
                "expected_emitted_nvext_priority": int(node["expected_inferred_priority"]),
            }
            for node in NAT_INFERRED_PRIORITY_NODES
        ],
    }


def attach_nat_inferred_priority_profile(meta: dict[str, Any]) -> dict[str, Any]:
    mode = str(meta.get("mode") or "")
    harness = str(meta.get("harness") or "")
    if not (nat_inferred_priority_enabled(mode) or (harness_emitted_signal_mode(mode) and harness == "nemo_agent_toolkit")):
        return meta
    node = nat_inferred_node_for_phase(str(meta.get("phase") or ""))
    out = dict(meta)
    out.pop("priority_intent", None)
    out["workflow_node"] = node["workflow_node"]
    out["workflow_node_goal"] = node["workflow_node_goal"]
    out["expected_inferred_priority"] = node["expected_inferred_priority"]
    out["inference_source"] = "nat_prediction_trie.workflow_path.latency_sensitivity"
    out["nat_inferred_priority_profile"] = nat_inferred_priority_profile()
    out["harness_input_priority_signal"] = f"workflow_path={node['workflow_node']}; no frontend priority intent"
    out["harness_input_priority_signal_source"] = "nat_workflow_profile_only"
    return out


def harness_priority_signal(harness: str, priority_class: str) -> tuple[str, str]:
    if priority_class == "background":
        return "adapter_metadata.priority_class=background", "adapter_metadata"
    if harness in {"codex", "opencode", "qwen_code", "deepseek_harness", "hatcher"}:
        return "service_tier=priority; metadata.priority_class=urgent", "openai_compatible"
    if harness == "nemo_agent_toolkit":
        return "service_tier=priority; agentic_hints.priority_class=urgent", "nat_openai_pass_through"
    if harness == "claude_code":
        return "service_tier=auto; metadata.priority_class=urgent", "anthropic_service_tier"
    return "adapter_metadata.priority_class=urgent", "adapter_metadata"


def attach_pre_harness_priority_intent(meta: dict[str, Any]) -> dict[str, Any]:
    if not pre_harness_priority_enabled(str(meta.get("mode") or "")):
        return meta
    phase = str(meta.get("phase") or "")
    priority_class = "background" if phase == "pressure_filler" else "urgent"
    reason = "tool_replay_deadline" if phase == "replay" else "session_priority_seed"
    if phase == "pressure_filler":
        reason = "pressure_filler_background_load"
    signal, source = harness_priority_signal(str(meta.get("harness") or ""), priority_class)
    out = dict(meta)
    out["priority_intent"] = {
        "class": priority_class,
        "reason": reason,
        "deadline_ms": meta.get("tool_wait_ms", meta.get("deadline_offset_ms", "")),
        "source": "experiment_driver",
    }
    out["harness_input_priority_signal"] = signal
    out["harness_input_priority_signal_source"] = source
    return out


def attach_harness_priority_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    meta = attach_pre_harness_priority_intent(meta)
    return attach_nat_inferred_priority_profile(meta)


def harness_native_cache_enabled(meta: dict[str, Any]) -> bool:
    return str(meta.get("mode") or "") in {HARNESS_NATIVE_CACHE_MODE, HARNESS_EMITTED_SIGNAL_MODE}


def native_cache_label(meta: dict[str, Any]) -> str:
    raw = f"{meta.get('harness', 'harness')}:{meta.get('pressure_level', 'pressure')}:{meta.get('session_id', 'session')}"
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:24]


def outbound_priority_fields(meta: dict[str, Any], api_kind: str) -> dict[str, Any]:
    if not pre_harness_priority_enabled(str(meta.get("mode") or "")):
        return {}
    intent = meta.get("priority_intent")
    if not isinstance(intent, dict):
        return {}
    priority_class = str(intent.get("class") or "")
    if priority_class != "urgent":
        return {
            "metadata": {
                "priority_class": priority_class,
                "priority_reason": str(intent.get("reason") or ""),
            }
        }
    metadata = {
        "priority_class": "urgent",
        "priority_reason": str(intent.get("reason") or "tool_replay_deadline"),
        "priority_deadline_ms": str(intent.get("deadline_ms") or ""),
    }
    if api_kind == "anthropic":
        return {"service_tier": "auto", "metadata": metadata}
    return {
        "service_tier": "priority",
        "metadata": metadata,
        "extra_body": {"agentic_hints": {"priority_class": "urgent", "reason": metadata["priority_reason"]}},
    }


def estimate_tokens(prompt: str) -> int:
    return max(1, int(round(len(prompt.split()) * 1.35)))


def build_pairs(harness: str, pressure_level: str, count: int, prompt_tokens: int) -> list[HarnessPair]:
    pairs: list[HarnessPair] = []
    for idx in range(count):
        session_id = f"{harness}_{pressure_level}_session_{idx:03d}"
        shared = make_shared_prefix(session_id, prompt_tokens)
        known_next_turn_prefix = (
            f"{shared}\n\n"
            "User task: fix the synthetic failing test in this repository.\n"
            "Assistant response: I will inspect the failing test by calling synthetic_tool."
        )
        prompt = (
            f"{shared}\n\n"
            "Initial turn: inspect this synthetic coding task context and answer briefly. "
            "The tool result will arrive later."
        )
        replay_prompt = (
            f"{known_next_turn_prefix}\n\n"
            "Replay turn after tool wait: the tool returned one failing assertion and a traceback. "
            "Continue from the same context and answer briefly."
        )
        pairs.append(
            HarnessPair(
                session_id=session_id,
                prompt=prompt,
                warmup_prompt=known_next_turn_prefix,
                replay_prompt=replay_prompt,
                task_index=str(idx),
                prompt_tokens=estimate_tokens(prompt),
            )
        )
    return pairs


async def run_hatcher_request(gateway_base: str, model: str, prompt: str, meta: dict[str, Any]) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{marker(meta)}"}],
        "max_tokens": int(meta.get("max_tokens") or 8),
        "temperature": 0,
        "stream": False,
    }
    payload.update(outbound_priority_fields(meta, "openai_chat"))
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(f"{gateway_base.rstrip('/')}/v1/chat/completions", json=payload)
        response.raise_for_status()


async def run_gateway_background_warmup(gateway_base: str, model: str, prompt: str, meta: dict[str, Any]) -> None:
    """Send a Dynamo-like frontend warmup without launching a full harness CLI."""

    await run_hatcher_request(gateway_base, model, prompt, meta)


def cli_or_npx(binary: str, package: str) -> list[str]:
    installed = shutil.which(binary)
    if installed:
        return [installed]
    return ["npx", "-y", package]


def qwen_workspace_path(log_dir: Path, meta: dict[str, Any]) -> Path:
    label = str(meta["label"])
    safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    digest = hashlib.sha1(str(log_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return Path(os.environ.get("HARNESS_QWEN_WORKSPACE_ROOT", "/tmp/agentic_hardware_qwen")) / f"{safe_label[:58]}_{digest}"


def codex_command(gateway_base: str, model: str, prompt: str, meta: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    cmd = [
        "npx",
        "-y",
        "@openai/codex@latest",
        "exec",
        "--ignore-user-config",
        "--strict-config",
        "-c",
        'model_providers.harness.name="Harness Gateway"',
        "-c",
        f'model_providers.harness.base_url="{provider_base}"',
        "-c",
        'model_providers.harness.env_key="DUMMY_KEY"',
        "-c",
        'model_providers.harness.wire_api="responses"',
        "-c",
        'model_provider="harness"',
        "-m",
        model,
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {"DUMMY_KEY": "dummy"}


def claude_command(gateway_base: str, prompt: str, meta: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    cmd = [
        "npx",
        "-y",
        "@anthropic-ai/claude-code@latest",
        "--bare",
        "-p",
        "--model",
        "claude-sonnet-4-5",
        "--output-format",
        "json",
        "--max-budget-usd",
        "0.05",
        "--no-session-persistence",
        "--prompt-suggestions",
        "false",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {
        "ANTHROPIC_BASE_URL": gateway_base.rstrip("/"),
        "ANTHROPIC_API_KEY": "dummy",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def opencode_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    opencode_model_id = model
    config_dir = (log_dir / "opencode_config" / str(meta["label"])).resolve()
    data_dir = (log_dir / "opencode_data" / str(meta["label"])).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    auth_dir = data_dir / "opencode"
    auth_dir.mkdir(parents=True, exist_ok=True)
    (auth_dir / "auth.json").write_text(
        json.dumps({"harness": {"type": "api", "key": "dummy"}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "harness": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Harness Gateway",
                "options": {
                    "baseURL": provider_base,
                    "apiKey": "dummy",
                    **({"setCacheKey": True} if harness_native_cache_enabled(meta) else {}),
                },
                "models": {
                    opencode_model_id: {
                        "name": model,
                        "limit": {
                            "context": 32768,
                            "output": 4096,
                        },
                        **(
                            {
                                "options": {
                                    "setCacheKey": True,
                                    "promptCacheKey": native_cache_label(meta),
                                }
                            }
                            if harness_native_cache_enabled(meta)
                            else {}
                        ),
                    }
                },
            }
        },
    }
    (config_dir / "opencode.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        *cli_or_npx("opencode", "opencode-ai@latest"),
        "run",
        "--print-logs",
        "--log-level",
        "DEBUG",
        "--model",
        f"harness/{opencode_model_id}",
        "--format",
        "json",
        "--dir",
        "/tmp",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {
        "OPENCODE_CONFIG_DIR": str(config_dir),
        "OPENCODE_CONFIG": str(config_dir / "opencode.json"),
        "XDG_DATA_HOME": str(data_dir),
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
        "OPENCODE_PERMISSION": json.dumps({"edit": "deny", "bash": "deny", "webfetch": "deny"}),
        "OPENAI_API_KEY": "dummy",
    }


def qwen_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    cache_signal_mode = str(meta.get("mode") or "") in {
        "no_cache_signal",
        HARNESS_NATIVE_CACHE_MODE,
        HARNESS_EMITTED_SIGNAL_MODE,
    }
    qwen_auth_type = "anthropic" if harness_native_cache_enabled(meta) else "openai"
    cache_enabled = harness_native_cache_enabled(meta)
    workspace = qwen_workspace_path(log_dir, meta)
    settings_dir = workspace / ".qwen"
    settings_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{meta['label']}.qwen_workspace.txt").write_text(str(workspace) + "\n", encoding="utf-8")
    settings = {
        "$version": 3,
        "model": {
            "name": model,
            "maxSessionTurns": -1,
            **(
                {
                    "generationConfig": {
                        "enableCacheControl": cache_enabled,
                        "forceGlobalCacheScope": cache_enabled,
                        "cacheRetention": "1h" if cache_enabled else "ephemeral",
                    }
                }
                if cache_signal_mode
                else {}
            ),
        },
        "modelProviders": {
            qwen_auth_type: [
                {
                    "id": model,
                    "name": model,
                    "baseUrl": provider_base,
                    "envKey": "ANTHROPIC_API_KEY" if qwen_auth_type == "anthropic" else "OPENAI_API_KEY",
                }
            ]
        },
        "security": {
            "auth": {
                "selectedType": qwen_auth_type,
                "apiKey": "dummy",
                "baseUrl": provider_base,
            }
        },
        "tools": {
            "approvalMode": "yolo",
            "exclude": ["shell", "write_file", "edit"],
        },
    }
    (settings_dir / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        *cli_or_npx("qwen", "@qwen-code/qwen-code@latest"),
        "--model",
        model,
        "--output-format",
        "json",
        "--prompt",
        f"{prompt}\n\n{marker(meta)}",
    ]
    env = {
        "OPENAI_API_KEY": "dummy",
        "OPENAI_BASE_URL": provider_base,
        "OPENAI_MODEL": model,
        "ANTHROPIC_API_KEY": "dummy",
        "ANTHROPIC_BASE_URL": provider_base,
        "ANTHROPIC_MODEL": model,
        "QWEN_MODEL": model,
    }
    return cmd, env


def openai_chat_probe_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
    adapter_name: str,
) -> tuple[list[str], dict[str, str]]:
    adapter_dir = log_dir / "wireability_adapters" / adapter_name / str(meta["label"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    request_path = adapter_dir / "request.json"
    script_path = adapter_dir / "post_chat_completion.py"
    payload = {
        "url": f"{gateway_base.rstrip('/')}/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": f"{prompt}\n\n{marker(meta)}"}],
            "max_tokens": int(meta.get("max_tokens") or 8),
            "temperature": 0,
            "stream": False,
        },
    }
    payload["body"].update(outbound_priority_fields(meta, "openai_chat"))
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    script_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "import sys",
                "import urllib.request",
                "",
                "request = json.loads(open(sys.argv[1], encoding='utf-8').read())",
                "body = json.dumps(request['body']).encode('utf-8')",
                "http_request = urllib.request.Request(",
                "    request['url'],",
                "    data=body,",
                "    headers={'content-type': 'application/json'},",
                "    method='POST',",
                ")",
                "with urllib.request.urlopen(http_request, timeout=None) as response:",
                "    sys.stdout.write(response.read().decode('utf-8', 'replace'))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [sys.executable, str(script_path), str(request_path)], {}


def nemo_agent_toolkit_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    nat_home = (log_dir / "nemo_agent_toolkit_config" / str(meta["label"])).resolve()
    nat_home.mkdir(parents=True, exist_ok=True)
    config_path = nat_home / "workflow.json"
    nat = os.environ.get("HARNESS_NAT_BIN") or shutil.which("nat")
    if not nat:
        raise FileNotFoundError("nat CLI not found; set HARNESS_NAT_BIN or install nvidia-nat.")
    wrapper_python = sys.executable
    if nat_inferred_priority_enabled(str(meta.get("mode") or "")):
        nat_python = os.environ.get("HARNESS_NAT_PYTHON")
        if nat_python:
            wrapper_python = nat_python
        else:
            sibling_python = Path(nat).resolve().parent / "python"
            if sibling_python.exists():
                wrapper_python = str(sibling_python)
    wrapper_path = Path(__file__).with_name("nemo_agent_toolkit_wrapper.py")
    request_path = nat_home / "wrapper_request.json"
    nat_log_path = nat_home / "nat_run.log"
    request = {
        "gateway_base": gateway_base,
        "model": model,
        "prompt": f"{prompt}\n\n{marker(meta)}",
        "meta": meta,
        "config_path": str(config_path),
        "nat_log_path": str(nat_log_path),
        "nat_bin": nat,
        "cwd": "/tmp",
        "env": {
            "OPENAI_API_KEY": "dummy",
            "NAT_CONFIG_FILE": str(config_path),
            "NAT_HOME": str(nat_home),
        },
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [wrapper_python, str(wrapper_path), "--request-json", str(request_path)]
    return cmd, {
        "OPENAI_API_KEY": "dummy",
        "NAT_CONFIG_FILE": str(config_path),
        "NAT_HOME": str(nat_home),
    }


def deepseek_harness_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    return openai_chat_probe_command(gateway_base, model, prompt, meta, log_dir, "deepseek_harness")


def pi_agent_harness_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    pi_dir = (log_dir / "pi_agent_config" / str(meta["label"])).resolve()
    extension_dir = pi_dir / "extensions"
    session_dir = pi_dir / "sessions"
    extension_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    extension_path = extension_dir / "harness-gateway-provider.mjs"
    extension_path.write_text(
        "\n".join(
            [
                "export default function(pi) {",
                "  pi.registerProvider('harness', {",
                "    name: 'Harness Gateway',",
                f"    baseUrl: {json.dumps(provider_base)},",
                "    apiKey: '$HARNESS_GATEWAY_API_KEY',",
                "    api: 'openai-completions',",
                "    models: [{",
                f"      id: {json.dumps(model)},",
                f"      name: {json.dumps(model)},",
                "      reasoning: false,",
                "      input: ['text'],",
                "      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },",
                "      contextWindow: 32768,",
                "      maxTokens: 4096,",
                "      compat: {",
                f"        cacheControlFormat: {json.dumps('anthropic') if harness_native_cache_enabled(meta) else 'undefined'},",
                f"        supportsLongCacheRetention: {str(bool(harness_native_cache_enabled(meta))).lower()},",
                f"        sendSessionAffinityHeaders: {str(bool(harness_native_cache_enabled(meta))).lower()},",
                "        sessionAffinityFormat: 'openai'",
                "      }",
                "    }]",
                "  });",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cmd = [
        *cli_or_npx("pi", "@earendil-works/pi-coding-agent@latest"),
        "--provider",
        "harness",
        "--model",
        model,
        "--api-key",
        "dummy",
        "--system-prompt",
        "You are a concise coding-agent wireability probe. Do not use tools.",
        "--mode",
        "json",
        "--print",
        "--no-tools",
        *(["--session-id", str(meta.get("session_id") or native_cache_label(meta))] if harness_native_cache_enabled(meta) else ["--no-session"]),
        "--session-dir",
        str(session_dir),
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-extensions",
        "--extension",
        str(extension_path),
        "--approve",
        "--offline",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {
        "HARNESS_GATEWAY_API_KEY": "dummy",
        "OPENAI_API_KEY": "dummy",
        "PI_CODING_AGENT_DIR": str(pi_dir),
        "PI_CODING_AGENT_SESSION_DIR": str(session_dir),
        "PI_OFFLINE": "1",
        "PI_TELEMETRY": "0",
        **({"PI_CACHE_RETENTION": "long"} if harness_native_cache_enabled(meta) else {}),
    }


def openclaw_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    openclaw_dir = (log_dir / "openclaw_config" / str(meta["label"])).resolve()
    state_dir = openclaw_dir / "state"
    openclaw_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = openclaw_dir / "config.json"
    config = {
        "models": {
            "mode": "merge",
            "providers": {
                "harness": {
                    "baseUrl": provider_base,
                    "apiKey": "dummy",
                    "auth": "api-key",
                    "api": "openai-completions",
                    "timeoutSeconds": 900,
                    "models": [
                        {
                            "id": model,
                            "name": model,
                            "api": "openai-completions",
                            "baseUrl": provider_base,
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": 32768,
                            "maxTokens": 4096,
                            **(
                                {
                                    "compat": {
                                        "supportsPromptCacheKey": True,
                                        "supportsLongCacheRetention": True,
                                    },
                                }
                                if harness_native_cache_enabled(meta)
                                else {}
                            ),
                        }
                    ],
                }
            },
        }
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        *cli_or_npx("openclaw", "openclaw@latest"),
        "agent",
        "exec",
        "--config",
        str(config_path),
        "--state-dir",
        str(state_dir),
        "--model",
        f"harness/{model}",
        "--code-mode",
        "direct",
        "--local-model-lean",
        "--timeout",
        "900",
        "--json",
        f"{prompt}\n\n{marker(meta)}",
    ]
    return cmd, {
        "OPENAI_API_KEY": "dummy",
        "HARNESS_GATEWAY_API_KEY": "dummy",
        "OPENCLAW_STATE_DIR": str(state_dir),
        "OPENCLAW_CONFIG_PATH": str(config_path),
        **(
            {
                "OPENCLAW_CACHE_RETENTION": "long",
                "OPENCLAW_CACHE_TRACE": "1",
            }
            if harness_native_cache_enabled(meta)
            else {}
        ),
    }


def hermes_agent_command(
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    provider_base = f"{gateway_base.rstrip('/')}/v1"
    hermes_home = (log_dir / "hermes_agent_config" / str(meta["label"])).resolve()
    hermes_home.mkdir(parents=True, exist_ok=True)
    config_path = hermes_home / "config.yaml"
    env_path = hermes_home / ".env"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  provider: harness",
                f"  default: {json.dumps(model)}",
                f"  model: {json.dumps(model)}",
                f"  base_url: {json.dumps(provider_base)}",
                '  api_key: "$HARNESS_GATEWAY_API_KEY"',
                "  api_mode: chat_completions",
                "  context_length: 65536",
                "providers:",
                "  harness:",
                "    name: Harness Gateway",
                f"    base_url: {json.dumps(provider_base)}",
                '    api_key: "$HARNESS_GATEWAY_API_KEY"',
                "    api_mode: chat_completions",
                f"    model: {json.dumps(model)}",
                "    models:",
                f"      {json.dumps(model)}:",
                "        context_length: 65536",
                "toolsets: []",
                "agent:",
                "  max_turns: 1",
                "  api_max_retries: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env_path.write_text("HARNESS_GATEWAY_API_KEY=dummy\nOPENAI_API_KEY=dummy\n", encoding="utf-8")
    hermes = os.environ.get("HARNESS_HERMES_BIN") or shutil.which("hermes") or shutil.which("hermes-agent")
    if hermes:
        cmd = [hermes]
    else:
        cmd = [sys.executable, "-m", "hermes_cli"]
    cmd.extend(
        [
            "--ignore-rules",
            "--accept-hooks",
            "--yolo",
            "--provider",
            "harness",
            "--model",
            model,
            "--toolsets",
            "",
            "--oneshot",
            f"{prompt}\n\n{marker(meta)}",
        ]
    )
    return cmd, {
        "HERMES_HOME": str(hermes_home),
        "HERMES_CONFIG": str(config_path),
        "HERMES_ENV": str(env_path),
        "HERMES_ACCEPT_HOOKS": "1",
        "HERMES_YOLO_MODE": "1",
        "HERMES_INFERENCE_PROVIDER": "harness",
        "HERMES_INFERENCE_MODEL": model,
        "HARNESS_GATEWAY_API_KEY": "dummy",
        "OPENAI_API_KEY": "dummy",
        "OPENAI_BASE_URL": provider_base,
    }


async def run_cli_request(
    harness: str,
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> None:
    if harness == "codex":
        cmd, extra_env = codex_command(gateway_base, model, prompt, meta)
    elif harness == "claude_code":
        cmd, extra_env = claude_command(gateway_base, prompt, meta)
    elif harness == "opencode":
        cmd, extra_env = opencode_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "qwen_code":
        cmd, extra_env = qwen_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "nemo_agent_toolkit":
        cmd, extra_env = nemo_agent_toolkit_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "deepseek_harness":
        cmd, extra_env = deepseek_harness_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "pi_agent_harness":
        cmd, extra_env = pi_agent_harness_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "openclaw":
        cmd, extra_env = openclaw_command(gateway_base, model, prompt, meta, log_dir)
    elif harness == "hermes_agent":
        cmd, extra_env = hermes_agent_command(gateway_base, model, prompt, meta, log_dir)
    else:
        raise ValueError(f"unsupported CLI harness: {harness}")
    env = os.environ.copy()
    env.update(extra_env)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{meta['label']}.log"
    trace_path = Path(str(meta.get("_trace_path") or ""))
    request_timeout_secs = float(os.environ.get("HARNESS_REQUEST_TIMEOUT_SECS", "900"))
    stop_when_gateway_done = str(os.environ.get("HARNESS_STOP_WHEN_GATEWAY_DONE", "1")) == "1"
    if harness == "nemo_agent_toolkit":
        stop_when_gateway_done = False

    def run() -> None:
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(qwen_workspace_path(log_dir, meta)) if harness == "qwen_code" else "/tmp",
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + request_timeout_secs
            while proc.poll() is None:
                if stop_when_gateway_done and trace_path and trace_has_event(trace_path, str(meta["label"]), "m27.request.end"):
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                    return
                if time.monotonic() >= deadline:
                    proc.kill()
                    proc.wait(timeout=5)
                    if trace_path and trace_has_event(trace_path, str(meta["label"]), "m27.request.end"):
                        return
                    raise TimeoutError(f"{harness} request timed out before gateway completion: {meta['label']}")
                time.sleep(0.25)
            if proc.returncode and not (trace_path and trace_has_event(trace_path, str(meta["label"]), "m27.request.end")):
                raise subprocess.CalledProcessError(proc.returncode, cmd)

    await asyncio.to_thread(run)


async def run_harness_request(
    harness: str,
    gateway_base: str,
    model: str,
    prompt: str,
    meta: dict[str, Any],
    log_dir: Path,
) -> None:
    if harness == "hatcher":
        await run_hatcher_request(gateway_base, model, prompt, meta)
        return
    await run_cli_request(harness, gateway_base, model, prompt, meta, log_dir)


async def run_filler(
    gateway_base: str,
    model: str,
    pair: HarnessPair,
    idx: int,
    meta_base: dict[str, Any],
    tokens: int,
) -> None:
    filler_session = f"{pair.session_id}_pressure_{idx:03d}"
    prompt = make_pressure_filler_prompt(filler_session, tokens)
    meta = {
        **meta_base,
        "session_id": filler_session,
        "phase": "pressure_filler",
        "label": f"{filler_session}_request",
        "task_index": pair.task_index,
        "prompt_hash": prompt_hash(prompt),
        "priority_label": "low",
        "max_tokens": 2,
    }
    meta = attach_pre_harness_priority_intent(meta)
    await run_hatcher_request(gateway_base, model, prompt, meta)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run target replay traffic through the supported harness adapters.")
    parser.add_argument("--harness", choices=SUPPORTED_HARNESSES, required=True)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--pressure-level", required=True)
    parser.add_argument("--gateway-base", default="http://127.0.0.1:31080")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--tool-wait-ms", type=int, default=50)
    parser.add_argument("--target-prompt-tokens", type=int, default=4096)
    parser.add_argument("--filler-sessions", type=int, default=0)
    parser.add_argument("--filler-prompt-tokens", type=int, default=1536)
    parser.add_argument("--session-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--arrival-gap-ms", type=int, default=40)
    parser.add_argument("--nat-inferred-profile-out", type=Path)
    args = parser.parse_args()

    if args.harness in {"codex", "claude_code", "opencode", "qwen_code"} and shutil.which("npx") is None:
        missing_bins = {
            "codex": "codex",
            "claude_code": "claude",
            "opencode": "opencode",
            "qwen_code": "qwen",
        }
        if shutil.which(missing_bins[args.harness]) is None:
            raise SystemExit(f"npx or {missing_bins[args.harness]} is required for {args.harness} harness probes.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.harness, args.pressure_level, args.session_count, args.target_prompt_tokens)
    rows: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(args.concurrency)
    workload_start = time.perf_counter()
    if args.mode == NAT_INFERRED_PRIORITY_MODE or (
        args.mode == HARNESS_EMITTED_SIGNAL_MODE and args.harness == "nemo_agent_toolkit"
    ):
        if args.harness != "nemo_agent_toolkit":
            raise SystemExit("nat_inferred_priority_hints is currently supported only for nemo_agent_toolkit.")
        if args.nat_inferred_profile_out is not None:
            args.nat_inferred_profile_out.parent.mkdir(parents=True, exist_ok=True)
            profile = {
                **nat_inferred_priority_profile(),
                "report_label": os.environ.get("REPORT_LABEL", ""),
                "created_unix_s": int(time.time()),
                "nat_provider": "dynamo_inferred",
            }
            args.nat_inferred_profile_out.write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def offset_ms() -> float:
        return (time.perf_counter() - workload_start) * 1000.0

    async def sleep_until(target_ms: float) -> None:
        delay = workload_start + target_ms / 1000.0 - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

    write_trace(
        args.trace,
        {
            "event": "m27.workload_start",
            "harness": args.harness,
            "mode": args.mode,
            "pressure_level": args.pressure_level,
            "model": args.model,
            "pairs": len(pairs),
            "tool_wait_list_ms": [args.tool_wait_ms],
            "filler_sessions": args.filler_sessions,
            "target_prompt_tokens": args.target_prompt_tokens,
            "filler_prompt_tokens": args.filler_prompt_tokens,
        },
    )

    async def bounded_request(prompt: str, meta: dict[str, Any]) -> None:
        async with sem:
            write_trace(
                args.trace,
                {
                    "event": "m27.harness.request_input",
                    "session_id": meta.get("session_id", ""),
                    "phase": meta.get("phase", ""),
                    "mode": meta.get("mode", ""),
                    "harness": meta.get("harness", args.harness),
                    "label": meta.get("label", ""),
                    "request_id": meta.get("label", ""),
                    "prompt_hash": meta.get("prompt_hash", ""),
                    "offset_ms": round(offset_ms(), 3),
                    "priority_intent": meta.get("priority_intent", ""),
                    "workflow_node": meta.get("workflow_node", ""),
                    "workflow_node_goal": meta.get("workflow_node_goal", ""),
                    "inference_source": meta.get("inference_source", ""),
                    "expected_inferred_priority": meta.get("expected_inferred_priority", ""),
                    "harness_input_priority_signal": meta.get("harness_input_priority_signal", ""),
                    "harness_input_priority_signal_source": meta.get("harness_input_priority_signal_source", ""),
                },
            )
            await run_harness_request(args.harness, args.gateway_base, args.model, prompt, meta, args.log_dir)
            write_trace(
                args.trace,
                {
                    "event": "m27.harness.request_done",
                    "session_id": meta.get("session_id", ""),
                    "phase": meta.get("phase", ""),
                    "mode": meta.get("mode", ""),
                    "harness": meta.get("harness", args.harness),
                    "label": meta.get("label", ""),
                    "request_id": meta.get("label", ""),
                    "offset_ms": round(offset_ms(), 3),
                },
            )

    async def run_pair(pair: HarnessPair, index: int) -> None:
        await sleep_until(index * args.arrival_gap_ms)
        write_trace(
            args.trace,
            {
                "event": "m27.session.start",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "task_index": pair.task_index,
                "tool_names": "synthetic_tool",
                "arrival_offset_ms": round(offset_ms(), 3),
                "tool_wait_ms": args.tool_wait_ms,
                "prompt_tokens": pair.prompt_tokens,
            },
        )
        base_meta = {
            "harness": args.harness,
            "mode": args.mode,
            "pressure_level": args.pressure_level,
            "high_priority": 100,
            "speculative_prefill_priority": 50,
            "low_priority": -100,
            "tool_wait_ms": args.tool_wait_ms,
            "_trace_path": str(args.trace),
            "nat_inferred_prefix_total_requests": 10,
            "nat_inferred_prefix_osl": 512,
            "nat_inferred_prefix_iat": 50,
            "native_cache_profile": {
                "enabled": harness_native_cache_enabled({"mode": args.mode}),
                "policy": "harness_decides_gateway_translates_only",
                "cache_key_seed": f"{args.harness}:{args.pressure_level}",
            },
        }
        initial_meta = {
            **base_meta,
            "session_id": pair.session_id,
            "phase": "initial_turn",
            "label": f"{pair.session_id}_initial",
            "task_index": pair.task_index,
            "prompt_hash": prompt_hash(pair.prompt),
            "priority_label": "high",
            "max_tokens": 8,
            "speculative_prefill": args.mode == "e2e_priority_hints_speculative_prefill",
        }
        initial_meta = attach_harness_priority_metadata(initial_meta)
        await bounded_request(pair.prompt, initial_meta)
        tool_start_ms = offset_ms()
        replay_due_ms = tool_start_ms + args.tool_wait_ms
        write_trace(
            args.trace,
            {
                "event": "m27.tool_wait.start",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "tool_start_offset_ms": round(tool_start_ms, 3),
                "replay_due_offset_ms": round(replay_due_ms, 3),
                "tool_wait_ms": args.tool_wait_ms,
                "prompt_hash": prompt_hash(pair.prompt),
            },
        )
        warmup_task: asyncio.Task[None] | None = None
        initial_gateway_row = trace_event_row(args.trace, str(initial_meta["label"]), "m27.request.start")
        cache_signal_driven_preload = (
            args.mode == HARNESS_EMITTED_SIGNAL_MODE
            and str(initial_gateway_row.get("harness_native_cache_signal_seen") or "").lower() == "yes"
        )
        direct_gateway_speculative_prefill = args.mode == "e2e_priority_hints_speculative_prefill"
        if direct_gateway_speculative_prefill or cache_signal_driven_preload:
            warmup_label = f"{pair.session_id}_speculative_prefill"
            replay_label = f"{pair.session_id}_replay"
            warmup_role = (
                "gateway_speculative_kv_preload"
                if cache_signal_driven_preload
                else "dynamo_like_background_warmup"
            )
            warmup_strategy = (
                "harness_cache_signal_gateway_speculative_kv_preload"
                if cache_signal_driven_preload
                else "known_next_turn_prefix"
            )
            warmup_trigger = "harness_cache_signal" if cache_signal_driven_preload else "gateway_injected_speculative_prefill"
            warmup_meta = {
                **base_meta,
                "session_id": pair.session_id,
                "phase": "speculative_prefill",
                "label": warmup_label,
                "task_index": pair.task_index,
                "prompt_hash": prompt_hash(pair.warmup_prompt),
                "priority_label": "background",
                "deadline_offset_ms": round(replay_due_ms, 3),
                "max_tokens": 1,
                "speculative_prefill": True,
                "speculative_prefill_role": warmup_role,
                "speculative_prefill_strategy": warmup_strategy,
                "parent_request_id": str(initial_meta["label"]),
                "expected_replay_request_id": replay_label,
                "warmup_prompt_tokens": estimate_tokens(pair.warmup_prompt),
                "harness_cache_signal_source": initial_gateway_row.get("harness_native_cache_signal_source", ""),
                "harness_cache_signal": initial_gateway_row.get("harness_native_cache_signal", ""),
            }
            warmup_meta = attach_harness_priority_metadata(warmup_meta)
            write_trace(
                args.trace,
                {
                    "event": "m27.speculative_prefill.hint_seen",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "request_id": initial_meta["label"],
                    "warmup_request_id": warmup_label,
                    "expected_replay_request_id": replay_label,
                    "strategy": warmup_strategy,
                    "role": warmup_role,
                    "trigger": warmup_trigger,
                    "harness_cache_signal_source": initial_gateway_row.get("harness_native_cache_signal_source", ""),
                    "warmup_prompt_hash": warmup_meta["prompt_hash"],
                    "warmup_prompt_tokens": warmup_meta["warmup_prompt_tokens"],
                    "tool_start_offset_ms": round(tool_start_ms, 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                },
            )

            async def run_warmup() -> None:
                write_trace(
                    args.trace,
                    {
                        "event": "m27.speculative_prefill.warmup_start",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "request_id": warmup_label,
                        "expected_replay_request_id": replay_label,
                        "strategy": warmup_strategy,
                        "role": warmup_role,
                        "trigger": warmup_trigger,
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
                try:
                    await run_gateway_background_warmup(args.gateway_base, args.model, pair.warmup_prompt, warmup_meta)
                    write_trace(
                        args.trace,
                        {
                            "event": "m27.speculative_prefill.warmup_end",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "harness": args.harness,
                            "request_id": warmup_label,
                            "expected_replay_request_id": replay_label,
                            "strategy": warmup_strategy,
                            "role": warmup_role,
                            "trigger": warmup_trigger,
                            "offset_ms": round(offset_ms(), 3),
                            "status": "ok",
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    write_trace(
                        args.trace,
                        {
                            "event": "m27.speculative_prefill.warmup_error",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "harness": args.harness,
                            "request_id": warmup_label,
                            "expected_replay_request_id": replay_label,
                            "strategy": warmup_strategy,
                            "role": warmup_role,
                            "trigger": warmup_trigger,
                            "offset_ms": round(offset_ms(), 3),
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

            warmup_task = asyncio.create_task(run_warmup())
        filler_tasks = [
            asyncio.create_task(run_filler(args.gateway_base, args.model, pair, idx, base_meta, args.filler_prompt_tokens))
            for idx in range(args.filler_sessions)
        ]
        await sleep_until(replay_due_ms)
        write_trace(
            args.trace,
            {
                "event": "m27.pre_replay.checkpoint",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "replay_due_offset_ms": round(replay_due_ms, 3),
                "expected_reuse": (
                    "gateway_speculative_kv_preload"
                    if cache_signal_driven_preload
                    else
                    "dynamo_like_speculative_prefill"
                    if args.mode == "e2e_priority_hints_speculative_prefill"
                    else "intercepted_priority"
                    if args.mode == "e2e_priority_hints"
                    else "baseline"
                ),
                "gpu_resident_tokens": "unknown",
                "host_resident_tokens": "unknown",
                "missing_tokens": "unknown",
                "protected_tokens": "unknown",
            },
        )
        write_trace(
            args.trace,
            {
                "event": "m27.replay.due",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "replay_due_offset_ms": round(replay_due_ms, 3),
            },
        )
        replay_meta = {
            **base_meta,
            "session_id": pair.session_id,
            "phase": "replay",
            "label": f"{pair.session_id}_replay",
            "task_index": pair.task_index,
            "prompt_hash": prompt_hash(pair.replay_prompt),
            "priority_label": "high",
            "deadline_offset_ms": round(replay_due_ms, 3),
            "max_tokens": 8,
        }
        replay_meta = attach_harness_priority_metadata(replay_meta)
        await bounded_request(pair.replay_prompt, replay_meta)
        write_trace(
            args.trace,
            {
                "event": "m27.tool_wait.end",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "replay_due_offset_ms": round(replay_due_ms, 3),
            },
        )
        if warmup_task is not None:
            await asyncio.gather(warmup_task, return_exceptions=True)
        await asyncio.gather(*filler_tasks, return_exceptions=True)
        rows.append(
            {
                "harness": args.harness,
                "mode": args.mode,
                "session_id": pair.session_id,
                "tool_wait_ms": args.tool_wait_ms,
                "filler_sessions": args.filler_sessions,
                "target_prompt_tokens": args.target_prompt_tokens,
                "pressure_level": args.pressure_level,
            }
        )

    await asyncio.gather(*(run_pair(pair, idx) for idx, pair in enumerate(pairs)))
    write_trace(args.trace, {"event": "m27.workload_end", "harness": args.harness, "mode": args.mode, "row_count": len(rows)})
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
