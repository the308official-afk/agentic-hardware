#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import base64
import heapq
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from agentic_kv.controller import (
    AIConfiguratorRuntimeCalibrator,
    BackendCapabilities,
    ControllerEvent,
    ControllerPolicy,
    ControllerStateStore,
    EventType,
    GatewayAdmissionControlBackendAdapter,
    GatewayDemoteRestoreBackendAdapter,
    GatewayFullControllerBackendAdapter,
    GatewayPriorityBackendAdapter,
    GatewaySpeculativePreloadBackendAdapter,
    ObserveOnlyBackendAdapter,
    PolicyConfig,
    SGLangTargetedKVPrefetchBackendAdapter,
    build_harness_controller_signal,
)
from agentic_kv.controller.modes import (
    CONTROLLER_ADMISSION_CONTROL_MODE,
    CONTROLLER_DEADLINE_FAIR_MODE,
    CONTROLLER_DEMOTE_RESTORE_MODE,
    CONTROLLER_FULL_CHUNKED_PREFILL_MODE,
    CONTROLLER_FULL_MODE,
    CONTROLLER_OBSERVE_ONLY_MODE,
    CONTROLLER_ORACLE_SAFE_SJF_MODES,
    CONTROLLER_ORACLE_SAFE_SJF_MODE,
    CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE,
    CONTROLLER_ORACLE_TIMELINE_MODE,
    CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_MODE,
    CONTROLLER_PROACTIVE_KV_MANAGEMENT_MODE,
    CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE,
    CONTROLLER_PRIORITY_DEMOTE_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_EARLYPREPARE_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_HARD_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MEDIUM_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SHORTHAND_MODE,
    CONTROLLER_PRIORITY_DEMOTION_ADMISSION_SOFT_MODE,
    CONTROLLER_SCHEDULER_PRIORITY_MODE,
    CONTROLLER_SPECULATIVE_PRELOAD_MODE,
    CONTROLLER_TARGETED_KV_PREFETCH_MODE,
    CONTROLLER_MEMORY_ADMISSION_MODE,
    CONTROLLER_VALUE_AWARE_EVICTION_MODE,
    HARNESS_EMITTED_SIGNAL_MODE,
    HARNESS_NATIVE_CACHE_MODE,
    NAT_INFERRED_PRIORITY_MODE,
    STORAGE_HICACHE_BASELINE_MODE,
    STORAGE_HICACHE_CONTROLLER_PREFETCH_MODE,
    SUPPORTED_MODES,
    controller_admission_aggressiveness,
    controller_admission_control_mode,
    controller_admission_lead_ms,
    controller_demote_restore_mode,
    controller_full_mode,
    controller_memory_admission_mode,
    controller_mode,
    controller_observe_only_mode,
    controller_proactive_kv_management_mode,
    controller_priority_demotion_admission_mode,
    controller_safe_sjf_degree,
    controller_scheduler_priority_mode,
    controller_speculative_preload_mode,
    controller_targeted_kv_prefetch_mode,
    controller_value_aware_eviction_mode,
    storage_hicache_mode,
)
from agentic_kv.controller.runtime_calibration import OracleExactRuntimeTable, RuntimeCalibrator, oracle_runtime_key
from agentic_kv.controller.sjf import SafeFillerAdmissionScheduler
from agentic_kv.controller.workload import (
    REALISTIC_AGENTIC_PROFILE,
    SYNTHETIC_PRESSURE_PROFILE,
    ToolWaitSpec,
    WorkloadShape,
    estimate_request_runtime_ms,
    normalize_agentic_workload_profile,
    realistic_initial_shape,
    realistic_replay_shape,
    sample_tool_wait_specs,
    tool_wait_distribution,
    workload_meta,
)
from run_real_prompt_controlled_replay import make_pressure_filler_prompt, make_shared_prefix, prompt_hash

MARKER = "HARNESS_REPLAY_EXPERIMENT_JSON:"
AUTHORIZED_DIRECT_LOAD_MECHANISM = "prepared_prefix_control"
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


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def make_agentic_workload_prompt(
    *,
    session_id: str,
    shape: WorkloadShape,
    role: str,
    stage: str,
    step_index: int,
    total_steps: int,
    previous_prompt: str = "",
) -> str:
    header = (
        "Synthetic but trajectory-shaped coding-agent request.\n"
        f"Session: {session_id}\n"
        f"Role: {role}\n"
        f"Stage: {stage}\n"
        f"Workflow phase: {shape.phase}\n"
        f"Request kind: {shape.kind}\n"
        f"Step: {step_index}/{total_steps}\n"
        f"Meaning: {shape.description}.\n"
    )
    if shape.kind == "planning_routing":
        body = (
            "The user asked for a repository change. Decide which files and tools are needed, "
            "then produce a concise plan for the next model call."
        )
    elif shape.kind == "file_search_inspect":
        body = (
            "Tool output contains file paths, snippets, and search hits. Identify the relevant "
            "files, preserve constraints, and choose the next safe action."
        )
    elif shape.kind == "patch_reasoning":
        body = (
            "The task now includes code context, an intended patch direction, and constraints. "
            "Reason about the edit while preserving behavior outside the requested scope."
        )
    elif shape.kind == "test_build_reasoning":
        body = (
            "A command or test returned logs. Interpret the failure, decide whether it is caused "
            "by the patch or environment, and choose the next action."
        )
    elif shape.kind == "review_final":
        body = (
            "Review the current state, summarize what changed, mention verification, and keep the "
            "response useful for a developer reading the final report."
        )
    else:
        body = (
            "The session resumed after a slower external step. Reconstruct the state from the "
            "stable context and continue without losing earlier constraints."
        )
    context_hint = (
        "\n\nRepresentative trace fragments:\n"
        "- system and developer constraints stay stable across turns\n"
        "- tools include file reads, repo search, commands, tests, patch application, and review\n"
        "- request priority is independent from this prompt shape unless the active mode supplies it\n"
    )
    prefix_budget = max(128, shape.prompt_tokens - estimate_tokens(header + body + context_hint) - 64)
    stable_prefix = make_shared_prefix(f"{session_id}:{shape.kind}:{stage}:{step_index}", prefix_budget)
    if previous_prompt and stage == "replay":
        previous_summary = "\n\nPrior turn summary: " + previous_prompt[:300]
    else:
        previous_summary = ""
    return f"{header}\n{stable_prefix}\n\n{body}{context_hint}{previous_summary}"


def replay_label_for(session_id: str, step_index: int, total_steps: int) -> str:
    if total_steps <= 1:
        return f"{session_id}_replay"
    return f"{session_id}_replay_{step_index:02d}"


def warmup_label_for(session_id: str, step_index: int, total_steps: int) -> str:
    if total_steps <= 1:
        return f"{session_id}_speculative_prefill"
    return f"{session_id}_speculative_prefill_{step_index:02d}"


def controller_prepare_lead_ms(wait_ms: int) -> int:
    if wait_ms < 500:
        return 0
    if wait_ms <= 3_000:
        return min(wait_ms, 500)
    if wait_ms <= 10_000:
        return min(wait_ms, 1_000)
    return min(wait_ms, 1_500)


def controller_targeted_prefetch_lead_ms(wait_ms: int) -> int:
    override = os.environ.get("CONTROLLER_TARGETED_PREFETCH_LEAD_MS")
    if override not in (None, ""):
        try:
            return max(0, int(float(override)))
        except ValueError:
            pass
    return controller_prepare_lead_ms(wait_ms)


def controller_direct_load_estimate_ms(prompt_tokens: int) -> int:
    fixed_ms = float(os.environ.get("CONTROLLER_DIRECT_LOAD_FIXED_MS", "1000") or "1000")
    ms_per_token = float(os.environ.get("CONTROLLER_DIRECT_LOAD_MS_PER_TOKEN", "2.0") or "2.0")
    multiplier = float(os.environ.get("CONTROLLER_DIRECT_LOAD_ESTIMATE_MULTIPLIER", "1.0") or "1.0")
    base_estimate_ms = fixed_ms + (max(0, prompt_tokens) * ms_per_token)
    return max(0, int(round(base_estimate_ms * max(0.0, multiplier))))


def controller_direct_load_allowed_wait_classes() -> set[str]:
    raw = os.environ.get("CONTROLLER_DIRECT_LOAD_ALLOWED_WAIT_CLASSES", "all")
    classes = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not classes or "all" in classes or "*" in classes:
        return set()
    return classes


def controller_direct_load_contract() -> str:
    if env_truthy("CONTROLLER_DIRECT_LOAD_REQUIRE_COMPLETION", default=False):
        return "requires_direct_load_completion_before_replay"
    return "best_effort_direct_load_before_replay"


def controller_direct_load_safety_margin_ms() -> int:
    return max(0, int(float(os.environ.get("CONTROLLER_DIRECT_LOAD_SAFETY_MARGIN_MS", "750") or "750")))


def controller_direct_load_latest_finish_ms(replay_due_ms: float) -> float:
    return replay_due_ms - controller_direct_load_safety_margin_ms()


def controller_direct_load_mechanism() -> str:
    mechanism = os.environ.get("CONTROLLER_DIRECT_LOAD_MECHANISM", AUTHORIZED_DIRECT_LOAD_MECHANISM).strip()
    mechanism = mechanism or AUTHORIZED_DIRECT_LOAD_MECHANISM
    if mechanism != AUTHORIZED_DIRECT_LOAD_MECHANISM:
        raise ValueError(
            "Unsupported controller direct-load mechanism "
            f"{mechanism!r}. The legacy synthetic-request KV warmup path has been removed; "
            f"use {AUTHORIZED_DIRECT_LOAD_MECHANISM!r}, which calls SGLang/HiCache prepare-prefix control."
        )
    return mechanism


def controller_direct_load_window(
    *,
    tool_start_ms: float,
    replay_due_ms: float,
    prompt_tokens: int,
    wait_class: str | None = None,
) -> dict[str, Any]:
    estimated_ms = controller_direct_load_estimate_ms(prompt_tokens)
    safety_margin_ms = controller_direct_load_safety_margin_ms()
    latest_finish_ms = replay_due_ms - safety_margin_ms
    available_slack_ms = latest_finish_ms - tool_start_ms
    allowed_wait_classes = controller_direct_load_allowed_wait_classes()
    normalized_wait_class = (wait_class or "").strip().lower()
    contract = controller_direct_load_contract()
    if allowed_wait_classes and normalized_wait_class not in allowed_wait_classes:
        return {
            "admitted": False,
            "reason": "skip_prefetch_wait_class_not_allowed_by_guarantee_contract",
            "contract": contract,
            "allowed_wait_classes": ",".join(sorted(allowed_wait_classes)),
            "estimated_ms": estimated_ms,
            "safety_margin_ms": safety_margin_ms,
            "available_slack_ms": available_slack_ms,
            "start_ms": tool_start_ms,
            "latest_finish_ms": latest_finish_ms,
        }
    if latest_finish_ms <= tool_start_ms:
        return {
            "admitted": False,
            "reason": "skip_prefetch_no_safe_window_before_replay",
            "contract": contract,
            "allowed_wait_classes": ",".join(sorted(allowed_wait_classes)) if allowed_wait_classes else "all",
            "estimated_ms": estimated_ms,
            "safety_margin_ms": safety_margin_ms,
            "available_slack_ms": available_slack_ms,
            "start_ms": tool_start_ms,
            "latest_finish_ms": latest_finish_ms,
        }
    if estimated_ms > available_slack_ms:
        return {
            "admitted": False,
            "reason": "skip_prefetch_not_enough_eta_slack",
            "contract": contract,
            "allowed_wait_classes": ",".join(sorted(allowed_wait_classes)) if allowed_wait_classes else "all",
            "estimated_ms": estimated_ms,
            "safety_margin_ms": safety_margin_ms,
            "available_slack_ms": available_slack_ms,
            "start_ms": tool_start_ms,
            "latest_finish_ms": latest_finish_ms,
        }
    # Start at the latest safe point that still leaves the estimated load time
    # plus the replay safety margin. Starting too early can observe the prefix
    # before pressure evicts it, then falsely conclude no H2D work is needed.
    start_ms = max(tool_start_ms, latest_finish_ms - estimated_ms)
    return {
        "admitted": True,
        "reason": "admit_prefetch_latest_safe_start_before_replay",
        "contract": contract,
        "allowed_wait_classes": ",".join(sorted(allowed_wait_classes)) if allowed_wait_classes else "all",
        "estimated_ms": estimated_ms,
        "safety_margin_ms": safety_margin_ms,
        "available_slack_ms": available_slack_ms,
        "start_ms": start_ms,
        "latest_finish_ms": latest_finish_ms,
    }


def deadline_fair_priority_for_due(replay_due_ms: float) -> int:
    base = int(os.environ.get("CONTROLLER_DEADLINE_FAIR_BASE_PRIORITY", "100000") or "100000")
    bucket_ms = max(1, int(os.environ.get("CONTROLLER_DEADLINE_FAIR_BUCKET_MS", "10") or "10"))
    due_bucket = int(max(0.0, replay_due_ms) // bucket_ms)
    return max(1, base - due_bucket)


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
    priority_class = "background" if phase.startswith("pressure_filler") else "urgent"
    reason = "tool_replay_deadline" if phase == "replay" else "session_priority_seed"
    if phase.startswith("pressure_filler"):
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


def _stable_percent(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _float_meta(meta: dict[str, Any], key: str, default: float) -> float:
    try:
        value = meta.get(key)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def value_aware_eviction_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    if not controller_value_aware_eviction_mode(str(meta.get("mode") or "")):
        return {}

    session_id = str(meta.get("session_id") or "")
    prefix_id = str(meta.get("prefix_id") or f"{session_id}:prefix")
    phase = str(meta.get("phase") or "")
    step = str(meta.get("tool_wait_step") or "0")
    bucket = _stable_percent(f"{session_id}|{prefix_id}|{step}")
    prompt_tokens = int(_float_meta(meta, "prompt_tokens", 0.0))

    if bucket < 40:
        value_class = "protected_high_value"
        reuse_probability = _float_meta(meta, "reuse_probability", 0.95)
        recompute_cost_tokens = int(_float_meta(meta, "recompute_cost_tokens", float(max(prompt_tokens, 2048))))
        priority = int(float(meta.get("high_priority") or 100))
        work_value = "critical_path"
        criticality = "high"
        cancelable = False
    elif bucket < 75:
        value_class = "normal_value"
        reuse_probability = _float_meta(meta, "reuse_probability", 0.65)
        recompute_cost_tokens = int(_float_meta(meta, "recompute_cost_tokens", float(max(prompt_tokens // 2, 1024))))
        priority = int(float(os.environ.get("CONTROLLER_EVICTION_NORMAL_PRIORITY", "0") or "0"))
        work_value = "normal"
        criticality = "normal"
        cancelable = False
    else:
        value_class = "evictable_low_value"
        reuse_probability = _float_meta(
            meta,
            "reuse_probability",
            float(os.environ.get("CONTROLLER_EVICTION_LOW_REUSE_PROBABILITY", "0.15") or "0.15"),
        )
        recompute_cost_tokens = int(
            _float_meta(
                meta,
                "recompute_cost_tokens",
                float(os.environ.get("CONTROLLER_EVICTION_LOW_RECOMPUTE_TOKENS", "256") or "256"),
            )
        )
        priority = int(float(meta.get("low_priority") or -100))
        work_value = "low_value_replay"
        criticality = "low"
        cancelable = True

    urgency_factor = 1.0
    try:
        eta_ms = float(meta.get("expected_tool_return_ms") or meta.get("next_ready_eta_ms") or meta.get("tool_wait_ms") or 0)
        if eta_ms > 0:
            urgency_factor = max(0.25, min(4.0, 60_000.0 / eta_ms))
    except (TypeError, ValueError):
        urgency_factor = 1.0
    value_score = int(round(reuse_probability * max(1, recompute_cost_tokens) * urgency_factor))
    return {
        "reuse_probability": reuse_probability,
        "recompute_cost_tokens": recompute_cost_tokens,
        "work_value": work_value,
        "criticality": criticality,
        "cancelable": cancelable,
        "priority_label": "low"
        if value_class == "evictable_low_value"
        else "normal"
        if value_class == "normal_value"
        else "high",
        "controller_sglang_priority": priority,
        "controller_eviction_policy": "sglang_radix_priority_eviction",
        "controller_eviction_value_class": value_class,
        "controller_eviction_value_score": value_score,
        "controller_eviction_bucket": bucket,
        "controller_eviction_signal_source": "session_id,prefix_id,phase,expected_tool_return_ms,reuse_probability,recompute_cost_tokens,deadline_after_ready_ms",
        "controller_eviction_translation": f"sglang.priority={priority};radix_eviction_policy=priority",
        "controller_eviction_scope": "all_replay_capable_requests",
        "controller_eviction_phase_seen": phase,
    }


def controller_event_from_meta(
    event_id: str,
    event_type: EventType,
    meta: dict[str, Any],
    *,
    monotonic_ms: int,
    expected_completion_ms: int | None = None,
    deadline_after_completion_ms: int | None = None,
    eta_uncertainty_ms: int | None = None,
) -> ControllerEvent:
    harness_controller_signal = build_harness_controller_signal(
        meta,
        monotonic_ms=monotonic_ms,
        expected_completion_ms=expected_completion_ms,
        deadline_after_completion_ms=deadline_after_completion_ms,
        eta_uncertainty_ms=eta_uncertainty_ms,
    )
    return ControllerEvent(
        event_id=event_id,
        event=event_type,
        session_id=str(meta.get("session_id") or ""),
        prefix_id=str(meta.get("prefix_id") or meta.get("session_id") or ""),
        session_generation=int(meta.get("session_generation") or 0),
        monotonic_ms=monotonic_ms,
        expected_completion_ms=expected_completion_ms,
        eta_uncertainty_ms=eta_uncertainty_ms,
        deadline_after_completion_ms=deadline_after_completion_ms,
        execution_priority=int(meta.get("high_priority") or 0),
        hbm_residency_priority=int(meta.get("hbm_residency_priority") or 0),
        host_retention_priority=int(meta.get("host_retention_priority") or 0),
        metadata={
            "harness": meta.get("harness", ""),
            "mode": meta.get("mode", ""),
            "pressure_level": meta.get("pressure_level", ""),
            "phase": meta.get("phase", ""),
            "label": meta.get("label", ""),
            "harness_controller_signal": harness_controller_signal,
        },
    )


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


def replay_prompt_for_step(pair: HarnessPair, step_index: int, total_steps: int) -> str:
    if total_steps <= 1:
        return pair.replay_prompt
    return (
        f"{pair.replay_prompt}\n\n"
        f"Agent loop replay step {step_index}/{total_steps}: continue the same coding task after "
        f"tool result {step_index}. Keep the response concise."
    )


def warmup_prompt_for_step(pair: HarnessPair, step_index: int, total_steps: int) -> str:
    if total_steps <= 1:
        return pair.warmup_prompt
    return (
        f"{pair.warmup_prompt}\n\n"
        f"Likely next replay step {step_index}/{total_steps}: prepare the stable task prefix and "
        f"the expected tool-result continuation."
    )


def filler_replay_prompt_for_step(prompt: str, filler_session: str, step_index: int, total_steps: int) -> str:
    suffix = "" if total_steps <= 1 else f" step {step_index}/{total_steps}"
    return (
        f"{prompt}\n\n"
        f"Tool result for {filler_session}{suffix}: background work completed. "
        "Resume the filler task and answer with one concise sentence."
    )


async def run_hatcher_request(gateway_base: str, model: str, prompt: str, meta: dict[str, Any]) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{marker(meta)}"}],
        "max_tokens": int(meta.get("max_tokens") or 8),
        "temperature": 0,
        "stream": False,
    }
    payload.update(outbound_priority_fields(meta, "openai_chat"))
    attempts = max(1, int(os.environ.get("HARNESS_HTTP_RETRIES", "2") or "2"))
    retry_delay_ms = max(0, int(os.environ.get("HARNESS_HTTP_RETRY_DELAY_MS", "250") or "250"))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_keepalive_connections=0)) as client:
                response = await client.post(f"{gateway_base.rstrip('/')}/v1/chat/completions", json=payload)
                response.raise_for_status()
                return
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            await asyncio.sleep(retry_delay_ms / 1000.0)
    assert last_error is not None
    raise last_error


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
    mode = str(meta.get("mode") or "")
    if nat_inferred_priority_enabled(mode) or harness_emitted_signal_mode(mode):
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
    *,
    wait_specs: list[ToolWaitSpec],
    trace: Path | None = None,
    workload_start: float | None = None,
    filler_replay_deadlines: bool = False,
    demotion_state: dict[str, Any] | None = None,
    admission_gate_state: dict[str, Any] | None = None,
    mode: str = "",
    oracle_safety_margin_ms: int = 150,
    safe_filler_scheduler: SafeFillerAdmissionScheduler | None = None,
    agentic_workload_profile: str = SYNTHETIC_PRESSURE_PROFILE,
    tool_wait_seed: int = 42,
    trace_controller_completion_linkage: bool = False,
    submit_request: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    deadline_fair_priority: Callable[[dict[str, Any], float, str, str], dict[str, Any]] | None = None,
    predictive_deadline_priority: Callable[[dict[str, Any], float, str, str], dict[str, Any]] | None = None,
    targeted_kv_prefetch: Callable[
        [dict[str, Any], str, str, float, float, ToolWaitSpec, int, str],
        asyncio.Task[None] | None,
    ]
    | None = None,
    memory_admission_register: Callable[[dict[str, Any], float, int, str, ToolWaitSpec, str], None] | None = None,
) -> None:
    def offset_ms() -> float:
        if workload_start is None:
            return 0.0
        return (time.perf_counter() - workload_start) * 1000.0

    def filler_stream_key(filler_session: str, stage: str, step_index: int | None = None) -> str:
        mode_part = "" if env_truthy("WORKLOAD_SHAPE_MODE_INDEPENDENT") else f":{meta_base.get('mode', '')}"
        suffix = f":{stage}" if step_index is None else f":{stage}:{step_index}"
        return (
            f"{meta_base.get('harness', '')}:{meta_base.get('pressure_level', '')}"
            f"{mode_part}:{filler_session}{suffix}"
        )

    def current_meta_base() -> dict[str, Any]:
        if not demotion_state or not demotion_state.get("active"):
            return dict(meta_base)
        demote_meta = demotion_state.get("meta")
        if not isinstance(demote_meta, dict):
            return dict(meta_base)
        return {
            **meta_base,
            **demote_meta,
            "controller_demotion_applied_to": "pressure_filler",
        }

    def write_background_reshape_signal(meta: dict[str, Any], *, request_id: str, stage: str) -> None:
        if trace is None or str(meta.get("controller_demote_restore_active") or "").lower() != "yes":
            return
        write_trace(
            trace,
            {
                "event": "m27.controller_traffic_reshape.background_request_lowered",
                "session_id": meta.get("session_id", ""),
                "mode": meta.get("mode", ""),
                "harness": meta.get("harness", ""),
                "phase": meta.get("phase", ""),
                "request_id": request_id,
                "label": request_id,
                "task_index": meta.get("task_index", ""),
                "tool_wait_step": meta.get("tool_wait_step", ""),
                "task_replay_steps": meta.get("task_replay_steps", ""),
                "stage": stage,
                "controller_demote_decision_id": meta.get("controller_demote_decision_id", ""),
                "controller_demote_command_id": meta.get("controller_demote_command_id", ""),
                "controller_demote_trigger": meta.get("controller_demote_trigger", ""),
                "controller_demote_translation": meta.get("controller_demote_translation", ""),
                "effective_sglang_priority": meta.get("controller_demote_priority", ""),
                "offset_ms": round(offset_ms(), 3),
            },
        )

    def write_completion_linkage(
        meta: dict[str, Any],
        admission_result: dict[str, Any] | None,
        *,
        request_id: str,
        stage: str,
        actual_start_offset_ms: float,
        actual_finish_offset_ms: float,
    ) -> None:
        if not trace_controller_completion_linkage or trace is None or not admission_result:
            return
        if not admission_result.get("admitted_by_safe_sjf"):
            return
        target_due_offset_ms = optional_float(admission_result.get("target_replay_due_offset_ms"))
        actual_runtime_ms = actual_finish_offset_ms - actual_start_offset_ms
        actual_overshoot_ms = (
            actual_finish_offset_ms - target_due_offset_ms
            if target_due_offset_ms is not None
            else None
        )
        estimated_runtime_ms = optional_float(admission_result.get("estimated_runtime_ms"))
        estimation_error_ms = (
            actual_runtime_ms - estimated_runtime_ms
            if estimated_runtime_ms is not None
            else None
        )
        write_trace(
            trace,
            {
                "event": "m27.controller_completion_linkage",
                "session_id": meta.get("session_id", ""),
                "mode": meta.get("mode", ""),
                "harness": meta.get("harness", ""),
                "pressure_level": meta.get("pressure_level", ""),
                "phase": meta.get("phase", ""),
                "request_id": request_id,
                "label": request_id,
                "request_group": "filler",
                "task_index": meta.get("task_index", ""),
                "tool_wait_step": meta.get("tool_wait_step", ""),
                "task_replay_steps": meta.get("task_replay_steps", ""),
                "stage": stage,
                "decision": admission_result.get("decision", ""),
                "decision_id": admission_result.get("decision_id", ""),
                "reason": admission_result.get("reason", ""),
                "admission_seq": admission_result.get("admission_seq", ""),
                "target_session_id": admission_result.get("target_session_id", ""),
                "target_tool_wait_step": admission_result.get("target_tool_wait_step", ""),
                "target_replay_due_offset_ms": round(target_due_offset_ms, 3)
                if target_due_offset_ms is not None
                else "",
                "decision_offset_ms": admission_result.get("decision_offset_ms", ""),
                "raw_estimated_runtime_ms": admission_result.get("raw_estimated_runtime_ms", ""),
                "estimated_runtime_ms": round(estimated_runtime_ms, 3)
                if estimated_runtime_ms is not None
                else "",
                "calibrated_runtime_ms": admission_result.get("calibrated_runtime_ms", ""),
                "runtime_class": admission_result.get("runtime_class", ""),
                "calibration_source": admission_result.get("calibration_source", ""),
                "calibration_sample_count": admission_result.get("calibration_sample_count", ""),
                "calibration_quantile": admission_result.get("calibration_quantile", ""),
                "calibration_floor_ms": admission_result.get("calibration_floor_ms", ""),
                "available_window_ms": admission_result.get("available_window_ms", ""),
                "time_until_next_replay_ms": admission_result.get("time_until_next_replay_ms", ""),
                "safety_margin_ms": admission_result.get("safety_margin_ms", ""),
                "expected_finish_offset_ms": admission_result.get("expected_finish_offset_ms", ""),
                "expected_overshoot_ms": admission_result.get("expected_overshoot_ms", ""),
                "actual_start_offset_ms": round(actual_start_offset_ms, 3),
                "actual_finish_offset_ms": round(actual_finish_offset_ms, 3),
                "actual_runtime_ms": round(actual_runtime_ms, 3),
                "estimation_error_ms": round(estimation_error_ms, 3)
                if estimation_error_ms is not None
                else "",
                "actual_overshoot_ms": round(actual_overshoot_ms, 3)
                if actual_overshoot_ms is not None
                else "",
                "verdict": (
                    "good_admit_finished_before_replay_due"
                    if actual_overshoot_ms is not None and actual_overshoot_ms <= 0
                    else (
                        "bad_admit_overshot_replay"
                        if actual_overshoot_ms is not None
                        else "unknown_admit_no_target_due"
                    )
                ),
                "offset_ms": round(offset_ms(), 3),
            },
        )

    async def wait_for_admission_if_needed(meta: dict[str, Any], *, request_id: str, stage: str) -> None:
        if not admission_gate_state or not admission_gate_state.get("active"):
            return
        release_event = admission_gate_state.get("release_event")
        if not isinstance(release_event, asyncio.Event) or release_event.is_set():
            return
        blocked_offset = offset_ms()
        gate_meta = admission_gate_state.get("meta")
        if not isinstance(gate_meta, dict):
            gate_meta = {}
        if safe_filler_scheduler is not None and mode in CONTROLLER_ORACLE_SAFE_SJF_MODES:
            return await safe_filler_scheduler.request(meta, request_id=request_id, stage=stage)
        if mode == CONTROLLER_ORACLE_TIMELINE_MODE:
            replay_due_offset_ms = float(gate_meta.get("replay_due_offset_ms") or 0)
            time_until_replay_ms = replay_due_offset_ms - blocked_offset
            estimated_runtime_ms = int(
                meta.get("estimated_runtime_ms")
                or estimate_request_runtime_ms(
                    int(meta.get("prompt_tokens") or 0),
                    int(meta.get("max_tokens") or 2),
                )
            )
            fits_before_replay = estimated_runtime_ms + oracle_safety_margin_ms < time_until_replay_ms
            decision = "admit" if fits_before_replay else "hold"
            reason = (
                "fits_before_next_target_replay"
                if fits_before_replay
                else "would_overlap_next_target_replay"
            )
            if trace is not None:
                write_trace(
                    trace,
                    {
                        "event": "m27.controller_oracle_timeline.admission_decision",
                        "session_id": meta.get("session_id", ""),
                        "mode": meta.get("mode", ""),
                        "harness": meta.get("harness", ""),
                        "pressure_level": meta.get("pressure_level", ""),
                        "phase": meta.get("phase", ""),
                        "request_id": request_id,
                        "label": request_id,
                        "task_index": meta.get("task_index", ""),
                        "tool_wait_step": meta.get("tool_wait_step", ""),
                        "task_replay_steps": meta.get("task_replay_steps", ""),
                        "stage": stage,
                        "decision": decision,
                        "reason": reason,
                        "estimated_runtime_ms": estimated_runtime_ms,
                        "time_until_next_replay_ms": round(time_until_replay_ms, 3),
                        "safety_margin_ms": oracle_safety_margin_ms,
                        "next_replay_due_offset_ms": round(replay_due_offset_ms, 3),
                        "gate_owner_session_id": gate_meta.get("session_id", ""),
                        "gate_tool_wait_step": gate_meta.get("tool_wait_step", ""),
                        "gate_open_offset_ms": gate_meta.get("open_offset_ms", ""),
                        "offset_ms": round(blocked_offset, 3),
                    },
                )
            if fits_before_replay:
                return {"decision": "admit", "reason": reason, "admitted_by_safe_sjf": False}
        if trace is not None:
            write_trace(
                trace,
                {
                    "event": "m27.controller_admission_gate.background_request_blocked",
                    "session_id": meta.get("session_id", ""),
                    "mode": meta.get("mode", ""),
                    "harness": meta.get("harness", ""),
                    "pressure_level": meta.get("pressure_level", ""),
                    "phase": meta.get("phase", ""),
                    "request_id": request_id,
                    "label": request_id,
                    "task_index": meta.get("task_index", ""),
                    "tool_wait_step": meta.get("tool_wait_step", ""),
                    "task_replay_steps": meta.get("task_replay_steps", ""),
                    "stage": stage,
                    "gate_owner_session_id": gate_meta.get("session_id", ""),
                    "gate_tool_wait_step": gate_meta.get("tool_wait_step", ""),
                    "gate_reason": gate_meta.get("reason", "target replay critical window"),
                    "gate_open_offset_ms": gate_meta.get("open_offset_ms", ""),
                    "offset_ms": round(blocked_offset, 3),
                },
            )
        await release_event.wait()
        if trace is not None:
            write_trace(
                trace,
                {
                    "event": "m27.controller_admission_gate.background_request_released",
                    "session_id": meta.get("session_id", ""),
                    "mode": meta.get("mode", ""),
                    "harness": meta.get("harness", ""),
                    "pressure_level": meta.get("pressure_level", ""),
                    "phase": meta.get("phase", ""),
                    "request_id": request_id,
                    "label": request_id,
                    "task_index": meta.get("task_index", ""),
                    "tool_wait_step": meta.get("tool_wait_step", ""),
                    "task_replay_steps": meta.get("task_replay_steps", ""),
                    "stage": stage,
                    "gate_owner_session_id": gate_meta.get("session_id", ""),
                    "gate_tool_wait_step": gate_meta.get("tool_wait_step", ""),
                    "blocked_for_ms": round(offset_ms() - blocked_offset, 3),
                    "offset_ms": round(offset_ms(), 3),
                },
            )
        return {"decision": "admit", "reason": "gate_released_after_target_replay", "admitted_by_safe_sjf": False}

    filler_session = f"{pair.session_id}_pressure_{idx:03d}"
    total_steps = max(1, len(wait_specs))
    initial_workload_meta: dict[str, Any] = {
        "agentic_workload_profile": agentic_workload_profile,
        "workload_request_kind": "synthetic_pressure_filler",
        "workload_phase_family": "pressure_filler",
        "workload_prompt_tokens_target": tokens,
        "workload_max_tokens": 2,
        "workload_description": "fixed synthetic background pressure request",
    }
    initial_max_tokens = 2
    if agentic_workload_profile == REALISTIC_AGENTIC_PROFILE:
        initial_shape = realistic_initial_shape(
            seed=tool_wait_seed,
            stream_key=filler_stream_key(filler_session, "initial"),
        )
        prompt = make_agentic_workload_prompt(
            session_id=filler_session,
            shape=initial_shape,
            role="filler",
            stage="initial",
            step_index=0,
            total_steps=total_steps,
        )
        initial_max_tokens = initial_shape.max_tokens
        initial_workload_meta = workload_meta(initial_shape, agentic_workload_profile)
    else:
        prompt = make_pressure_filler_prompt(filler_session, tokens)
        meta = {
            **current_meta_base(),
            "session_id": filler_session,
        "prefix_id": f"{filler_session}:prefix",
        "phase": "pressure_filler_initial" if filler_replay_deadlines else "pressure_filler",
        "label": f"{filler_session}_initial" if filler_replay_deadlines else f"{filler_session}_request",
        "task_index": pair.task_index,
        "prompt_hash": prompt_hash(prompt),
        "prompt_tokens": estimate_tokens(prompt),
        "priority_label": "low",
        "max_tokens": initial_max_tokens,
            "tool_wait_step": 0,
            "task_replay_steps": total_steps,
            **initial_workload_meta,
        }
        meta.update(value_aware_eviction_metadata(meta))
        meta["estimated_runtime_ms"] = estimate_request_runtime_ms(int(meta["prompt_tokens"]), int(meta["max_tokens"]))
    meta["oracle_runtime_key"] = oracle_runtime_key(meta)
    meta = attach_pre_harness_priority_intent(meta)
    write_background_reshape_signal(meta, request_id=str(meta["label"]), stage="initial")
    admission_result = await wait_for_admission_if_needed(meta, request_id=str(meta["label"]), stage="initial")
    request_start_offset_ms = offset_ms()
    try:
        if submit_request is not None:
            await submit_request(prompt, meta)
        else:
            await run_hatcher_request(gateway_base, model, prompt, meta)
    finally:
        request_finish_offset_ms = offset_ms()
        write_completion_linkage(
            meta,
            admission_result,
            request_id=str(meta["label"]),
            stage="initial",
            actual_start_offset_ms=request_start_offset_ms,
            actual_finish_offset_ms=request_finish_offset_ms,
        )
        if safe_filler_scheduler is not None and admission_result and admission_result.get("admitted_by_safe_sjf"):
            await safe_filler_scheduler.complete(str(meta["label"]))
    if not filler_replay_deadlines:
        return

    current_prompt = prompt
    for spec in wait_specs:
        wait_ms = int(spec.wait_ms)
        replay_label = replay_label_for(filler_session, spec.step_index, total_steps)
        tool_start_ms = offset_ms()
        replay_due_ms = tool_start_ms + wait_ms
        predictive_replay_fields: dict[str, Any] = {}
        prefetch_task: asyncio.Task[None] | None = None
        tool_wait_meta = {
            **current_meta_base(),
            "session_id": filler_session,
            "prefix_id": f"{filler_session}:prefix",
            "phase": "tool_wait",
            "label": f"{filler_session}_tool_wait_{spec.step_index:02d}",
            "task_index": pair.task_index,
            "tool_wait_ms": wait_ms,
            "tool_wait_step": spec.step_index,
            "task_replay_steps": total_steps,
            "tool_wait_class": spec.wait_class,
            "reuse_probability": 0.98,
            "recompute_cost_tokens": estimate_tokens(current_prompt),
        }
        if memory_admission_register is not None:
            memory_admission_register(
                tool_wait_meta,
                replay_due_ms,
                estimate_tokens(current_prompt),
                "filler",
                spec,
                replay_label,
            )
        if predictive_deadline_priority is not None:
            predictive_replay_fields = predictive_deadline_priority(
                {**tool_wait_meta, "label": replay_label},
                replay_due_ms,
                "filler",
                "tool_wait",
            )
        if targeted_kv_prefetch is not None:
            prefetch_task = targeted_kv_prefetch(
                tool_wait_meta,
                current_prompt,
                replay_label,
                tool_start_ms,
                replay_due_ms,
                spec,
                total_steps,
                "filler",
            )
        if trace is not None:
            write_trace(
                trace,
                {
                    "event": "m27.tool_wait.start",
                    "session_id": filler_session,
                    "mode": meta_base.get("mode", ""),
                    "harness": meta_base.get("harness", ""),
                    "phase": "pressure_filler",
                    "label": f"{filler_session}_tool_wait_{spec.step_index:02d}",
                    "task_index": pair.task_index,
                    "tool_wait_step": spec.step_index,
                    "task_replay_steps": total_steps,
                    "tool_wait_profile": meta_base.get("tool_wait_profile", ""),
                    "tool_wait_class": spec.wait_class,
                    "tool_wait_ms": wait_ms,
                    "tool_start_offset_ms": round(tool_start_ms, 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "deadline_after_completion_ms": wait_ms,
                    "request_group": "filler",
                    "offset_ms": round(offset_ms(), 3),
                    "expected_replay_request_id": replay_label,
                },
            )

        remaining_ms = replay_due_ms - offset_ms()
        if remaining_ms > 0:
            await asyncio.sleep(remaining_ms / 1000.0)
        if trace is not None:
            write_trace(
                trace,
                {
                    "event": "m27.replay.due",
                    "session_id": filler_session,
                    "mode": meta_base.get("mode", ""),
                    "harness": meta_base.get("harness", ""),
                    "phase": "pressure_filler",
                    "label": replay_label,
                    "request_id": replay_label,
                    "task_index": pair.task_index,
                    "tool_wait_step": spec.step_index,
                    "task_replay_steps": total_steps,
                    "tool_wait_profile": meta_base.get("tool_wait_profile", ""),
                    "tool_wait_class": spec.wait_class,
                    "tool_wait_ms": wait_ms,
                    "deadline_after_completion_ms": wait_ms,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "request_group": "filler",
                    "offset_ms": round(offset_ms(), 3),
                    "expected_replay_request_id": replay_label,
                },
            )

        replay_workload_meta: dict[str, Any] = {
            "agentic_workload_profile": agentic_workload_profile,
            "workload_request_kind": "synthetic_pressure_filler_replay",
            "workload_phase_family": "pressure_filler",
            "workload_prompt_tokens_target": estimate_tokens(current_prompt),
            "workload_max_tokens": 2,
            "workload_description": "fixed synthetic background replay request",
        }
        replay_max_tokens = 2
        if agentic_workload_profile == REALISTIC_AGENTIC_PROFILE:
            replay_shape = realistic_replay_shape(
                wait_class=spec.wait_class,
                seed=tool_wait_seed,
                stream_key=filler_stream_key(filler_session, "replay", spec.step_index),
            )
            replay_prompt = make_agentic_workload_prompt(
                session_id=filler_session,
                shape=replay_shape,
                role="filler",
                stage="replay",
                step_index=spec.step_index,
                total_steps=total_steps,
                previous_prompt=current_prompt,
            )
            replay_max_tokens = replay_shape.max_tokens
            replay_workload_meta = workload_meta(replay_shape, agentic_workload_profile)
        else:
            replay_prompt = filler_replay_prompt_for_step(current_prompt, filler_session, spec.step_index, total_steps)
        replay_meta = {
            **current_meta_base(),
            "session_id": filler_session,
            "prefix_id": f"{filler_session}:prefix",
            "session_generation": spec.step_index + 1,
            "phase": "pressure_filler",
            "label": replay_label,
            "task_index": pair.task_index,
            "prompt_hash": prompt_hash(replay_prompt),
            "prompt_tokens": estimate_tokens(replay_prompt),
            "priority_label": "low",
            "max_tokens": replay_max_tokens,
            "deadline_offset_ms": round(replay_due_ms, 3),
            "tool_wait_ms": wait_ms,
            "tool_wait_step": spec.step_index,
            "task_replay_steps": total_steps,
            "tool_wait_class": spec.wait_class,
            **replay_workload_meta,
        }
        replay_meta.update(value_aware_eviction_metadata(replay_meta))
        replay_meta["estimated_runtime_ms"] = estimate_request_runtime_ms(
            int(replay_meta["prompt_tokens"]),
            int(replay_meta["max_tokens"]),
        )
        replay_meta["oracle_runtime_key"] = oracle_runtime_key(replay_meta)
        if deadline_fair_priority is not None:
            replay_meta.update(deadline_fair_priority(replay_meta, replay_due_ms, "filler", "replay"))
        if predictive_replay_fields:
            replay_meta.update(predictive_replay_fields)
        replay_meta = attach_pre_harness_priority_intent(replay_meta)
        write_background_reshape_signal(replay_meta, request_id=replay_label, stage="replay")
        admission_result = await wait_for_admission_if_needed(replay_meta, request_id=replay_label, stage="replay")
        request_start_offset_ms = offset_ms()
        try:
            if submit_request is not None:
                await submit_request(replay_prompt, replay_meta)
            else:
                await run_hatcher_request(gateway_base, model, replay_prompt, replay_meta)
        finally:
            request_finish_offset_ms = offset_ms()
            write_completion_linkage(
                replay_meta,
                admission_result,
                request_id=replay_label,
                stage="replay",
                actual_start_offset_ms=request_start_offset_ms,
                actual_finish_offset_ms=request_finish_offset_ms,
            )
            if safe_filler_scheduler is not None and admission_result and admission_result.get("admitted_by_safe_sjf"):
                await safe_filler_scheduler.complete(replay_label)
        if prefetch_task is not None:
            await asyncio.gather(prefetch_task, return_exceptions=True)
        if trace is not None:
            write_trace(
                trace,
                {
                    "event": "m27.tool_wait.end",
                    "session_id": filler_session,
                    "mode": meta_base.get("mode", ""),
                    "harness": meta_base.get("harness", ""),
                    "phase": "pressure_filler",
                    "label": replay_label,
                    "request_id": replay_label,
                    "task_index": pair.task_index,
                    "tool_wait_step": spec.step_index,
                    "task_replay_steps": total_steps,
                    "tool_wait_profile": meta_base.get("tool_wait_profile", ""),
                    "tool_wait_class": spec.wait_class,
                    "tool_wait_ms": wait_ms,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "request_group": "filler",
                    "offset_ms": round(offset_ms(), 3),
                },
            )
        current_prompt = replay_prompt


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
    parser.add_argument(
        "--tool-wait-profile",
        default=os.environ.get("TOOL_WAIT_PROFILE", "fixed"),
        help="Tool-wait profile to sample per replay step. Use fixed, agentic_mixed, or realistic_agentic_mix.",
    )
    parser.add_argument(
        "--tool-wait-profile-spec",
        default=os.environ.get("TOOL_WAIT_PROFILE_SPEC", ""),
        help=(
            "Optional custom comma list: class:weight:wait_ms,class:weight:wait_ms. "
            "Ranges are supported, e.g. very_short:20:100-500,short:35:1000-5000."
        ),
    )
    parser.add_argument(
        "--tool-wait-seed",
        type=int,
        default=int(os.environ.get("TOOL_WAIT_SEED", "42") or "42"),
        help="Seed for reproducible sampled tool-wait profiles.",
    )
    parser.add_argument(
        "--task-replay-steps",
        type=int,
        default=int(os.environ.get("TASK_REPLAY_STEPS", "1") or "1"),
        help="Number of tool-wait/resume cycles per target task.",
    )
    parser.add_argument(
        "--agentic-workload-profile",
        default=os.environ.get("AGENTIC_WORKLOAD_PROFILE", SYNTHETIC_PRESSURE_PROFILE),
        help="Request-shape profile. Use synthetic_pressure or realistic_agentic_mix.",
    )
    parser.add_argument("--target-prompt-tokens", type=int, default=4096)
    parser.add_argument("--workload-jsonl", type=Path, default=os.environ.get("PROMPT_WORKLOAD_JSONL") or None)
    parser.add_argument("--filler-sessions", type=int, default=0)
    parser.add_argument("--filler-prompt-tokens", type=int, default=1536)
    parser.add_argument(
        "--filler-backlog-mode",
        choices=("once", "constant"),
        default=os.environ.get("FILLER_BACKLOG_MODE", "once"),
        help="once launches one fixed filler batch; constant replenishes fillers until target replays finish.",
    )
    parser.add_argument(
        "--filler-backlog-target",
        type=int,
        default=int(os.environ.get("FILLER_BACKLOG_TARGET", "0") or "0"),
        help="Desired active filler depth for --filler-backlog-mode=constant. Defaults to --filler-sessions.",
    )
    parser.add_argument(
        "--filler-backlog-total",
        type=int,
        default=int(os.environ.get("FILLER_BACKLOG_TOTAL", "0") or "0"),
        help="Maximum filler sessions to launch in constant backlog mode. Defaults to a bounded multiple of --filler-sessions.",
    )
    parser.add_argument(
        "--filler-replay-deadlines",
        action="store_true",
        default=env_flag("FILLER_REPLAY_DEADLINES", False),
        help="Make filler/background sessions run an initial call and a deadline-bearing replay call.",
    )
    parser.add_argument(
        "--filler-replay-deadline-ms",
        type=int,
        default=int(os.environ.get("FILLER_REPLAY_DEADLINE_MS", "0") or "0"),
        help="Milliseconds from filler tool-wait start to filler replay due. Defaults to --tool-wait-ms.",
    )
    parser.add_argument("--session-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--arrival-gap-ms", type=int, default=40)
    parser.add_argument("--nat-inferred-profile-out", type=Path)
    parser.add_argument(
        "--trace-profile",
        default=os.environ.get("TRACE_PROFILE", "full_debug"),
        help="Named instrumentation preset recorded with the run: minimal, deadline, controller_decision, idle_gap, cache_debug, or full_debug.",
    )
    parser.add_argument(
        "--trace-controller-decisions",
        default=os.environ.get("TRACE_CONTROLLER_DECISIONS", "1"),
        help="Set to 0 to suppress detailed SJF/controller decision events.",
    )
    parser.add_argument(
        "--trace-controller-completion-linkage",
        default=os.environ.get("TRACE_CONTROLLER_COMPLETION_LINKAGE", "0"),
        help="Set to 1 to emit compact completion rows linking SJF admit decisions to actual filler finish times.",
    )
    parser.add_argument(
        "--trace-replay-blockers",
        default=os.environ.get("TRACE_REPLAY_BLOCKERS", "0"),
        help="Set to 1 to capture bounded local pending/in-flight request-id snapshots for replay deep dives.",
    )
    parser.add_argument(
        "--trace-replay-blockers-max-ids",
        type=int,
        default=int(os.environ.get("TRACE_REPLAY_BLOCKERS_MAX_IDS", "16") or "16"),
        help="Maximum pending or in-flight request IDs to include in each replay blocker snapshot.",
    )
    parser.add_argument(
        "--trace-replay-blockers-events",
        default=os.environ.get("TRACE_REPLAY_BLOCKERS_EVENTS", "before_acquire,submit"),
        help="Comma-separated blocker snapshot points: before_acquire,submit.",
    )
    args = parser.parse_args()
    os.environ["TRACE_PROFILE"] = str(args.trace_profile)
    os.environ["TRACE_CONTROLLER_DECISIONS"] = str(args.trace_controller_decisions)
    os.environ["TRACE_CONTROLLER_COMPLETION_LINKAGE"] = str(args.trace_controller_completion_linkage)
    os.environ["TRACE_REPLAY_BLOCKERS"] = str(args.trace_replay_blockers)
    os.environ["TRACE_REPLAY_BLOCKERS_MAX_IDS"] = str(args.trace_replay_blockers_max_ids)
    os.environ["TRACE_REPLAY_BLOCKERS_EVENTS"] = str(args.trace_replay_blockers_events)
    if args.task_replay_steps < 1:
        raise SystemExit("--task-replay-steps must be at least 1")
    if args.filler_backlog_target < 0:
        raise SystemExit("--filler-backlog-target must be non-negative")
    if args.filler_backlog_total < 0:
        raise SystemExit("--filler-backlog-total must be non-negative")
    args.agentic_workload_profile = normalize_agentic_workload_profile(args.agentic_workload_profile)
    workload_mode_part = "" if env_truthy("WORKLOAD_SHAPE_MODE_INDEPENDENT") else f":{args.mode}"

    def workload_stream_key(pair: HarnessPair, role: str, stage: str, extra: str = "") -> str:
        suffix = f":{extra}" if extra else ""
        return (
            f"{args.harness}:{args.pressure_level}{workload_mode_part}:"
            f"{pair.session_id}:{role}:{stage}{suffix}"
        )

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
    if args.workload_jsonl:
        workload = [json.loads(line) for line in args.workload_jsonl.read_text().splitlines() if line.strip()]
        if not workload:
            raise ValueError("empty prompt encoding workload")
        for i, pair in enumerate(pairs):
            item = workload[i % len(workload)]
            pairs[i] = replace(pair, prompt=str(item["initial_prompt"]),
                               replay_prompt=str(item["replay_prompt"]), warmup_prompt=str(item["initial_prompt"]),
                               prompt_tokens=estimate_tokens(str(item["initial_prompt"])))
    target_wait_specs_by_session = {
        pair.session_id: sample_tool_wait_specs(
            profile=args.tool_wait_profile,
            base_wait_ms=args.tool_wait_ms,
            custom_spec=args.tool_wait_profile_spec,
            steps=args.task_replay_steps,
            seed=args.tool_wait_seed,
            stream_key=workload_stream_key(pair, "target", "waits"),
        )
        for pair in pairs
    }
    filler_base_wait_ms = args.filler_replay_deadline_ms if args.filler_replay_deadline_ms > 0 else args.tool_wait_ms
    def sample_filler_wait_specs(pair: HarnessPair, idx: int) -> list[ToolWaitSpec]:
        return sample_tool_wait_specs(
            profile=args.tool_wait_profile,
            base_wait_ms=filler_base_wait_ms,
            custom_spec=args.tool_wait_profile_spec,
            steps=args.task_replay_steps,
            seed=args.tool_wait_seed,
            stream_key=workload_stream_key(pair, "filler", "waits", str(idx)),
        )
    filler_backlog_target = (
        args.filler_backlog_target
        if args.filler_backlog_target > 0
        else args.filler_sessions
    )
    if args.filler_backlog_mode == "constant" and filler_backlog_target <= 0:
        raise SystemExit("--filler-backlog-mode=constant requires --filler-sessions or --filler-backlog-target > 0")
    if args.filler_backlog_mode == "constant" and args.filler_backlog_total > 0:
        filler_backlog_total = args.filler_backlog_total
    elif args.filler_backlog_mode == "constant":
        filler_backlog_total = max(filler_backlog_target, args.filler_sessions) * max(2, args.task_replay_steps * 3)
    else:
        filler_backlog_total = args.filler_sessions
    configured_background_requests = filler_backlog_target if args.filler_backlog_mode == "constant" else args.filler_sessions
    rows: list[dict[str, Any]] = []
    admission_lock = asyncio.Lock()
    admitted_warmups = 0
    admission_max_warmups = int(os.environ.get("CONTROLLER_ADMISSION_MAX_WARMUPS_PER_CASE", "1"))
    admission_min_tool_wait_ms = int(os.environ.get("CONTROLLER_ADMISSION_MIN_TOOL_WAIT_MS", "75"))
    admission_max_filler_sessions = int(os.environ.get("CONTROLLER_ADMISSION_MAX_FILLER_SESSIONS", "16"))
    admission_max_concurrency = int(os.environ.get("CONTROLLER_ADMISSION_MAX_CONCURRENCY", "8"))
    oracle_safety_margin_ms = int(os.environ.get("CONTROLLER_ORACLE_SAFETY_MARGIN_MS", "150") or "150")
    (
        safe_sjf_safety_margin_ms,
        safe_sjf_max_in_flight,
        safe_sjf_require_fit,
        safe_sjf_idle_override,
    ) = controller_safe_sjf_degree(args.mode)
    runtime_estimator_backend = os.environ.get("CONTROLLER_RUNTIME_ESTIMATOR", "").strip().lower()
    if args.mode == CONTROLLER_ORACLE_EXACT_RUNTIME_ADMISSION_MODE:
        runtime_calibrator = OracleExactRuntimeTable.from_env()
    elif runtime_estimator_backend in {"aiconfigurator", "ai_configurator", "aic"}:
        runtime_calibrator = AIConfiguratorRuntimeCalibrator.from_env(fallback=RuntimeCalibrator.from_env())
    elif args.mode == CONTROLLER_PRIORITY_DEMOTION_CALIBRATED_ADMISSION_MODE:
        runtime_calibrator = RuntimeCalibrator.from_env()
    else:
        runtime_calibrator = None
    workload_start = time.perf_counter()
    controller_demotion_state: dict[str, Any] = {"active": False, "meta": {}, "owners": set()}
    controller_enabled = controller_mode(args.mode)
    controller_active_deadline_fair = args.mode == CONTROLLER_DEADLINE_FAIR_MODE
    controller_active_predictive_deadline_queue = args.mode == CONTROLLER_PREDICTIVE_DEADLINE_QUEUE_MODE
    controller_active_priority = controller_scheduler_priority_mode(args.mode)
    controller_active_preload = controller_speculative_preload_mode(args.mode)
    controller_active_targeted_prefetch = controller_targeted_kv_prefetch_mode(args.mode)
    controller_active_proactive_kv_management = controller_proactive_kv_management_mode(args.mode)
    controller_active_demote_restore = controller_demote_restore_mode(args.mode)
    controller_active_priority_demotion_admission = controller_priority_demotion_admission_mode(args.mode)
    controller_active_admission = controller_admission_control_mode(args.mode)
    controller_active_memory_admission = controller_memory_admission_mode(args.mode)
    controller_active_full = controller_full_mode(args.mode)
    admission_aggressiveness = controller_admission_aggressiveness(args.mode)
    max_target_tool_wait_ms = max(
        [args.tool_wait_ms]
        + [spec.wait_ms for specs in target_wait_specs_by_session.values() for spec in specs]
    )
    controller_store = ControllerStateStore()
    controller_policy = ControllerPolicy(
        PolicyConfig(
            observe_only=not (
                controller_active_priority
                or controller_active_preload
                or controller_active_targeted_prefetch
                or controller_active_demote_restore
                or controller_active_priority_demotion_admission
                or controller_active_admission
                or controller_active_memory_admission
                or controller_active_full
                or controller_active_deadline_fair
            ),
            prepare_window_ms=0 if (controller_active_demote_restore or controller_active_full) else max(25, max_target_tool_wait_ms),
            min_demote_idle_ms=0 if (controller_active_demote_restore or controller_active_full) else 250,
            safety_margin_ms=0 if (controller_active_demote_restore or controller_active_full) else 6,
            signal_timing_enabled=controller_active_full,
            background_prefill_budget_tokens=min(1024, max(128, args.filler_prompt_tokens // 2)),
        )
    )
    controller_admission_gate_event = asyncio.Event()
    controller_admission_gate_event.set()
    controller_admission_gate_state: dict[str, Any] = {
        "active": False,
        "meta": {},
        "release_event": controller_admission_gate_event,
    }
    if controller_active_priority:
        controller_backend = GatewayPriorityBackendAdapter(
            BackendCapabilities(
                priority_queue=True,
                background_prefill_budget=False,
                kv_demote=False,
                kv_prefetch=False,
                kv_release=False,
                live_metrics=True,
                observe_only=False,
                backend_name=args.mode,
            )
        )
    elif controller_active_preload:
        controller_backend = GatewaySpeculativePreloadBackendAdapter(
            BackendCapabilities(
                priority_queue=False,
                background_prefill_budget=False,
                kv_demote=False,
                kv_prefetch=True,
                kv_release=False,
                live_metrics=True,
                observe_only=False,
                backend_name=args.mode,
            )
        )
    elif controller_active_targeted_prefetch:
        controller_backend = SGLangTargetedKVPrefetchBackendAdapter(
            BackendCapabilities(
                priority_queue=False,
                background_prefill_budget=False,
                kv_demote=False,
                kv_prefetch=True,
                kv_release=False,
                live_metrics=True,
                observe_only=False,
                backend_name=args.mode,
                backend_version=(
                    "direct_hook_available=1"
                    if os.environ.get("AGENTIC_KV_TARGETED_PREFETCH_HOOK", "").lower() in {"1", "true", "yes"}
                    else "direct_hook_available=0"
                ),
            )
        )
    elif controller_active_demote_restore:
        controller_backend = GatewayDemoteRestoreBackendAdapter(
            BackendCapabilities(
                priority_queue=True,
                background_prefill_budget=False,
                kv_demote=True,
                kv_prefetch=False,
                kv_release=True,
                live_metrics=True,
                observe_only=False,
                backend_name=args.mode,
            )
        )
    elif controller_active_memory_admission:
        controller_backend = GatewayAdmissionControlBackendAdapter(
            BackendCapabilities(
                priority_queue=False,
                background_prefill_budget=False,
                kv_demote=False,
                kv_prefetch=False,
                kv_release=False,
                live_metrics=True,
                observe_only=False,
                backend_name=args.mode,
            )
        )
    elif controller_active_admission:
        controller_backend = GatewayAdmissionControlBackendAdapter(
            BackendCapabilities(
                priority_queue=True,
                background_prefill_budget=True,
                kv_demote=False,
                kv_prefetch=True,
                kv_release=True,
                live_metrics=True,
                observe_only=False,
                backend_name=args.mode,
            )
        )
    elif controller_active_full:
        controller_backend = GatewayFullControllerBackendAdapter(
            BackendCapabilities(
                priority_queue=True,
                background_prefill_budget=True,
                kv_demote=True,
                kv_prefetch=False,
                kv_release=True,
                live_metrics=True,
                observe_only=False,
                backend_name=args.mode,
                backend_version=(
                    "v1:no_speculative_preload+chunked_prefill"
                    if args.mode == CONTROLLER_FULL_CHUNKED_PREFILL_MODE
                    else "v1:no_speculative_preload"
                ),
            )
        )
    else:
        controller_backend = ObserveOnlyBackendAdapter(
            BackendCapabilities(
                priority_queue=True,
                background_prefill_budget=True,
                kv_demote=True,
                kv_prefetch=True,
                kv_release=True,
                live_metrics=True,
                observe_only=True,
                backend_name="controller_observe_only",
            )
        )
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

    safe_filler_scheduler = (
        SafeFillerAdmissionScheduler(
            trace=args.trace,
            now_ms=offset_ms,
            gate_state=controller_admission_gate_state,
            safety_margin_ms=safe_sjf_safety_margin_ms,
            max_in_flight=safe_sjf_max_in_flight,
            require_fit_before_replay=safe_sjf_require_fit,
            idle_override=safe_sjf_idle_override,
            estimate_runtime_ms=estimate_request_runtime_ms,
            calibrate_runtime=runtime_calibrator.estimate if runtime_calibrator is not None else None,
        )
        if args.mode in CONTROLLER_ORACLE_SAFE_SJF_MODES
        else None
    )

    async def sleep_until(target_ms: float) -> None:
        delay = workload_start + target_ms / 1000.0 - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

    memory_admission_state: dict[str, Any] = {
        "predicted_replays": [],
        "decision_seq": 0,
        "delayed_candidates": 0,
        "admitted_candidates": 0,
    }

    def memory_admission_lookahead_ms() -> float:
        return float(os.environ.get("CONTROLLER_MEMORY_ADMISSION_LOOKAHEAD_MS", "15000") or "15000")

    def memory_admission_release_grace_ms() -> float:
        return float(os.environ.get("CONTROLLER_MEMORY_ADMISSION_RELEASE_GRACE_MS", "50") or "50")

    def memory_admission_min_candidate_tokens() -> int:
        return int(float(os.environ.get("CONTROLLER_MEMORY_ADMISSION_MIN_CANDIDATE_TOKENS", "1024") or "1024"))

    def memory_admission_max_delay_ms() -> float:
        return float(os.environ.get("CONTROLLER_MEMORY_ADMISSION_MAX_DELAY_MS", "30000") or "30000")

    def memory_admission_token_budget() -> int:
        configured = os.environ.get("CONTROLLER_MEMORY_ADMISSION_TOKEN_BUDGET", "").strip()
        if configured:
            return int(float(configured))
        return int(float(os.environ.get("MAX_TOTAL_TOKENS", "12288") or "12288"))

    def memory_admission_reserve_fraction() -> float:
        return max(0.0, float(os.environ.get("CONTROLLER_MEMORY_ADMISSION_RESERVE_FRACTION", "0.65") or "0.65"))

    def register_memory_admission_replay(
        meta: dict[str, Any],
        replay_due_ms: float,
        prompt_tokens: int,
        request_group: str,
        wait_spec: ToolWaitSpec,
        replay_label: str,
    ) -> None:
        if not controller_active_memory_admission:
            return
        entry = {
            "session_id": str(meta.get("session_id") or ""),
            "prefix_id": str(meta.get("prefix_id") or f"{meta.get('session_id', '')}:prefix"),
            "request_group": request_group,
            "expected_replay_request_id": replay_label,
            "replay_due_offset_ms": float(replay_due_ms),
            "expected_tool_return_ms": int(wait_spec.wait_ms),
            "next_ready_eta_ms": int(wait_spec.wait_ms),
            "eta_uncertainty_ms": max(1, int(wait_spec.wait_ms) // 4),
            "deadline_after_ready_ms": int(wait_spec.wait_ms),
            "reuse_probability": float(meta.get("reuse_probability") or 0.98),
            "recompute_cost_tokens": int(meta.get("recompute_cost_tokens") or prompt_tokens),
            "tool_wait_step": wait_spec.step_index,
            "tool_wait_class": wait_spec.wait_class,
        }
        memory_admission_state["predicted_replays"].append(entry)
        write_trace(
            args.trace,
            {
                "event": "m27.controller_memory_admission.replay_registered",
                "mode": args.mode,
                "harness": args.harness,
                "pressure_level": args.pressure_level,
                **entry,
                "offset_ms": round(offset_ms(), 3),
            },
        )

    def memory_admission_candidate(meta: dict[str, Any]) -> bool:
        if not controller_active_memory_admission:
            return False
        phase = str(meta.get("phase") or "")
        if phase in {"replay", "hint_prefetch", "speculative_prefill", "tool_wait"}:
            return False
        if phase.startswith("pressure_filler") and meta.get("deadline_offset_ms") not in (None, ""):
            return False
        return phase in {"initial_turn", "pressure_filler_initial", "pressure_filler", "background"}

    async def apply_memory_admission_if_needed(meta: dict[str, Any]) -> dict[str, Any]:
        if not memory_admission_candidate(meta):
            return meta
        now = offset_ms()
        lookahead_ms = memory_admission_lookahead_ms()
        release_grace_ms = memory_admission_release_grace_ms()
        token_budget = memory_admission_token_budget()
        reserve_fraction = memory_admission_reserve_fraction()
        min_candidate_tokens = memory_admission_min_candidate_tokens()
        max_delay_ms = memory_admission_max_delay_ms()
        active_replays = [
            entry
            for entry in memory_admission_state["predicted_replays"]
            if now <= float(entry["replay_due_offset_ms"]) <= now + lookahead_ms
        ]
        predicted_replay_tokens = int(
            round(
                sum(
                    float(entry.get("reuse_probability") or 0.0)
                    * max(1, int(entry.get("recompute_cost_tokens") or 0))
                    for entry in active_replays
                )
            )
        )
        candidate_tokens = int(meta.get("prompt_tokens") or 0)
        reserve_tokens = int(round(token_budget * reserve_fraction))
        earliest_due_ms = min(
            [float(entry["replay_due_offset_ms"]) for entry in active_replays],
            default=0.0,
        )
        projected_tokens = predicted_replay_tokens + candidate_tokens
        risky = bool(
            active_replays
            and candidate_tokens >= min_candidate_tokens
            and projected_tokens >= reserve_tokens
        )
        memory_admission_state["decision_seq"] += 1
        decision_seq = int(memory_admission_state["decision_seq"])
        delay_until_ms = min(earliest_due_ms + release_grace_ms, now + max_delay_ms) if risky else now
        delay_ms = max(0.0, delay_until_ms - now)
        decision = "delay" if delay_ms > 1 else "admit"
        reason = (
            "candidate_would_consume_reserved_near_future_replay_headroom"
            if decision == "delay"
            else "safe_current_capacity_after_near_future_replay_reserve"
        )
        base_event = {
            "event": "m27.controller_memory_admission.decision",
            "mode": args.mode,
            "harness": args.harness,
            "pressure_level": args.pressure_level,
            "decision_seq": decision_seq,
            "decision": decision,
            "reason": reason,
            "session_id": meta.get("session_id", ""),
            "prefix_id": meta.get("prefix_id", ""),
            "phase": meta.get("phase", ""),
            "request_id": meta.get("label", ""),
            "label": meta.get("label", ""),
            "candidate_tokens": candidate_tokens,
            "predicted_replay_count": len(active_replays),
            "predicted_replay_tokens": predicted_replay_tokens,
            "token_budget": token_budget,
            "reserve_fraction": reserve_fraction,
            "reserved_headroom_tokens": reserve_tokens,
            "projected_candidate_plus_replay_tokens": projected_tokens,
            "lookahead_ms": lookahead_ms,
            "earliest_replay_due_offset_ms": round(earliest_due_ms, 3) if earliest_due_ms else "",
            "planned_delay_ms": round(delay_ms, 3),
            "offset_ms": round(now, 3),
        }
        write_trace(args.trace, base_event)
        out = {
            **meta,
            "controller_memory_admission_decision": decision,
            "controller_memory_admission_reason": reason,
            "controller_memory_admission_predicted_replay_count": len(active_replays),
            "controller_memory_admission_predicted_replay_tokens": predicted_replay_tokens,
            "controller_memory_admission_reserved_headroom_tokens": reserve_tokens,
            "controller_memory_admission_candidate_tokens": candidate_tokens,
        }
        if decision == "delay":
            memory_admission_state["delayed_candidates"] += 1
            await sleep_until(delay_until_ms)
            released_at_ms = offset_ms()
            delayed_for_ms = released_at_ms - now
            out["controller_memory_admission_delayed_for_ms"] = round(delayed_for_ms, 3)
            write_trace(
                args.trace,
                {
                    **base_event,
                    "event": "m27.controller_memory_admission.released",
                    "actual_delay_ms": round(delayed_for_ms, 3),
                    "release_offset_ms": round(released_at_ms, 3),
                },
            )
        else:
            memory_admission_state["admitted_candidates"] += 1
        return out

    def record_controller_event(
        event_id: str,
        event_type: EventType,
        meta: dict[str, Any],
        *,
        monotonic_ms: int,
        expected_completion_ms: int | None = None,
        deadline_after_completion_ms: int | None = None,
        eta_uncertainty_ms: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if not controller_enabled:
            return None
        event = controller_event_from_meta(
            event_id,
            event_type,
            meta,
            monotonic_ms=monotonic_ms,
            expected_completion_ms=expected_completion_ms,
            deadline_after_completion_ms=deadline_after_completion_ms,
            eta_uncertainty_ms=eta_uncertainty_ms,
        )
        state, accepted = controller_store.apply_event(event)
        decision = controller_policy.plan(state, controller_backend.capabilities(), now_ms=monotonic_ms)
        results = [controller_backend.apply(command).to_dict() for command in decision.commands]
        decision_row = decision.to_dict()
        write_trace(
            args.trace,
            {
                "event": "m27.controller.decision",
                "controller_schema_version": decision.schema_version,
                "controller_event": event.event.value,
                "controller_event_id": event.event_id,
                "controller_event_accepted": accepted,
                "controller_phase": state.phase.value,
                "controller_decision": decision_row,
                "controller_backend_results": results,
                "session_id": event.session_id,
                "prefix_id": event.prefix_id,
                "harness_controller_signal": event.metadata.get("harness_controller_signal", {}),
                "mode": args.mode,
                "harness": args.harness,
                "pressure_level": args.pressure_level,
                "offset_ms": round(offset_ms(), 3),
            },
        )
        return decision_row, results

    def selected_controller_command(
        controller_result: tuple[dict[str, Any], list[dict[str, Any]]] | None,
        *,
        scheduler_action: str | None = None,
        kv_action: str | None = None,
        require_acted: bool = False,
    ) -> dict[str, Any] | None:
        if controller_result is None:
            return None
        decision_row, backend_results = controller_result
        result_by_command_id = {str(result.get("command_id")): result for result in backend_results}
        for command in decision_row.get("commands") or []:
            if not isinstance(command, dict):
                continue
            backend_result = result_by_command_id.get(str(command.get("command_id") or ""))
            if backend_result is None:
                continue
            if require_acted and not backend_result.get("acted"):
                continue
            if scheduler_action is not None and command.get("scheduler_action") != scheduler_action:
                continue
            if kv_action is not None and command.get("kv_action") != kv_action:
                continue
            return {
                **command,
                "backend_accepted": backend_result.get("accepted", ""),
                "backend_acted": backend_result.get("acted", ""),
                "backend_reason": backend_result.get("reason", ""),
                "backend_name": backend_result.get("backend_name", ""),
                "controller_decision_id": decision_row.get("decision_id", ""),
                "controller_decision_reason": decision_row.get("reason", ""),
            }
        return None

    def acted_controller_command(
        controller_result: tuple[dict[str, Any], list[dict[str, Any]]] | None,
        *,
        scheduler_action: str | None = None,
        kv_action: str | None = None,
    ) -> dict[str, Any] | None:
        return selected_controller_command(
            controller_result,
            scheduler_action=scheduler_action,
            kv_action=kv_action,
            require_acted=True,
        )

    def assign_due_time_priority(
        meta: dict[str, Any],
        replay_due_ms: float,
        request_group: str,
        stage: str,
        *,
        predictive: bool = False,
    ) -> dict[str, Any]:
        priority = deadline_fair_priority_for_due(replay_due_ms)
        policy_label = "predictive_deadline_queue" if predictive else "deadline_fair"
        event_name = (
            "m27.controller_predictive_deadline_queue.priority_assigned"
            if predictive
            else "m27.controller_deadline_fair.priority_assigned"
        )
        translation_source = (
            "controller_predictive_deadline_queue_tool_wait_eta"
            if predictive
            else "controller_deadline_fair_due_time"
        )
        reason = (
            "predictive_deadline_queue_orders_replay_requests_by_due_time_from_tool_wait_eta"
            if predictive
            else "deadline_fair_mode_orders_all_replays_by_due_time_without_semantic_high_priority"
        )
        fields = {
            "priority_label": policy_label,
            "controller_deadline_fair": not predictive,
            "controller_predictive_deadline_queue": predictive,
            "controller_sglang_priority": priority,
            "controller_deadline_due_offset_ms": round(replay_due_ms, 3),
            "controller_priority_translation": f"controller.{policy_label}.due_priority={priority}",
            "controller_priority_translation_source": translation_source,
        }
        write_trace(
            args.trace,
            {
                "event": event_name,
                "session_id": meta.get("session_id", ""),
                "mode": args.mode,
                "harness": args.harness,
                "pressure_level": args.pressure_level,
                "phase": meta.get("phase", ""),
                "label": meta.get("label", ""),
                "request_id": meta.get("label", ""),
                "request_group": request_group,
                "stage": stage,
                "task_index": meta.get("task_index", ""),
                "tool_wait_step": meta.get("tool_wait_step", ""),
                "task_replay_steps": meta.get("task_replay_steps", ""),
                "tool_wait_ms": meta.get("tool_wait_ms", ""),
                "tool_wait_class": meta.get("tool_wait_class", ""),
                "replay_due_offset_ms": round(replay_due_ms, 3),
                "assigned_sglang_priority": priority,
                "bucket_ms": int(os.environ.get("CONTROLLER_DEADLINE_FAIR_BUCKET_MS", "10") or "10"),
                "base_priority": int(os.environ.get("CONTROLLER_DEADLINE_FAIR_BASE_PRIORITY", "100000") or "100000"),
                "reason": reason,
                "offset_ms": round(offset_ms(), 3),
            },
        )
        return fields

    def assign_deadline_fair_priority(
        meta: dict[str, Any],
        replay_due_ms: float,
        request_group: str,
        stage: str,
    ) -> dict[str, Any]:
        return assign_due_time_priority(meta, replay_due_ms, request_group, stage, predictive=False)

    def assign_predictive_deadline_priority(
        meta: dict[str, Any],
        replay_due_ms: float,
        request_group: str,
        stage: str,
    ) -> dict[str, Any]:
        return assign_due_time_priority(meta, replay_due_ms, request_group, stage, predictive=True)

    async def controller_admission_decision(
        pair: HarnessPair,
        controller_result: tuple[dict[str, Any], list[dict[str, Any]]] | None,
        *,
        wait_spec: ToolWaitSpec,
        tool_start_ms: float,
        replay_due_ms: float,
    ) -> dict[str, Any]:
        nonlocal admitted_warmups
        prefetch_command = acted_controller_command(controller_result, kv_action="prefetch")
        budget_command = acted_controller_command(controller_result, scheduler_action="set_background_prefill_budget")
        if prefetch_command is None:
            skip_reason = (
                "controller_full_v1_skips_speculative_preload_by_policy"
                if controller_active_full
                else "controller_did_not_request_prefetch"
            )
            decision = {
                "admitted": False,
                "reason": skip_reason,
                "prefetch_command_id": "",
                "prefetch_decision_id": "",
                "budget_command_id": budget_command.get("command_id", "") if budget_command else "",
                "admitted_warmups_before": admitted_warmups,
                "admitted_warmups_after": admitted_warmups,
            }
        else:
            async with admission_lock:
                before = admitted_warmups
                skip_reasons: list[str] = []
                if wait_spec.wait_ms < admission_min_tool_wait_ms:
                    skip_reasons.append(
                        f"tool_wait_ms {wait_spec.wait_ms} below minimum {admission_min_tool_wait_ms}"
                    )
                if args.filler_sessions > admission_max_filler_sessions:
                    skip_reasons.append(
                        f"filler_sessions {args.filler_sessions} above limit {admission_max_filler_sessions}"
                    )
                if args.concurrency > admission_max_concurrency:
                    skip_reasons.append(f"concurrency {args.concurrency} above limit {admission_max_concurrency}")
                if admitted_warmups >= admission_max_warmups:
                    skip_reasons.append(
                        f"warmup budget exhausted {admitted_warmups}/{admission_max_warmups}"
                    )
                admitted = not skip_reasons
                if admitted:
                    admitted_warmups += 1
                decision = {
                    "admitted": admitted,
                    "reason": "admitted" if admitted else "; ".join(skip_reasons),
                    "prefetch_command_id": prefetch_command.get("command_id", ""),
                    "prefetch_decision_id": prefetch_command.get("controller_decision_id", ""),
                    "budget_command_id": budget_command.get("command_id", "") if budget_command else "",
                    "admitted_warmups_before": before,
                    "admitted_warmups_after": admitted_warmups,
                }
        write_trace(
            args.trace,
            {
                "event": "m27.controller_admission.decision",
                "session_id": pair.session_id,
                "mode": args.mode,
                "harness": args.harness,
                "pressure_level": args.pressure_level,
                "task_index": pair.task_index,
                "decision": "admit" if decision["admitted"] else "skip",
                "reason": decision["reason"],
                "prefetch_command_id": decision["prefetch_command_id"],
                "controller_decision_id": decision["prefetch_decision_id"],
                "budget_command_id": decision["budget_command_id"],
                "admitted_warmups_before": decision["admitted_warmups_before"],
                "admitted_warmups_after": decision["admitted_warmups_after"],
                "max_warmups_per_case": admission_max_warmups,
                "min_tool_wait_ms": admission_min_tool_wait_ms,
                "max_filler_sessions": admission_max_filler_sessions,
                "max_concurrency": admission_max_concurrency,
                "tool_wait_ms": wait_spec.wait_ms,
                "tool_wait_step": wait_spec.step_index,
                "task_replay_steps": args.task_replay_steps,
                "tool_wait_profile": args.tool_wait_profile,
                "tool_wait_class": wait_spec.wait_class,
                "filler_sessions": args.filler_sessions,
                "concurrency": args.concurrency,
                "tool_start_offset_ms": round(tool_start_ms, 3),
                "replay_due_offset_ms": round(replay_due_ms, 3),
                "offset_ms": round(offset_ms(), 3),
            },
        )
        return decision

    write_trace(
        args.trace,
        {
            "event": "m27.workload_start",
            "harness": args.harness,
            "mode": args.mode,
            "pressure_level": args.pressure_level,
            "model": args.model,
            "controller_runtime_estimator": runtime_estimator_backend,
            "pairs": len(pairs),
            "tool_wait_list_ms": [
                spec.wait_ms
                for pair in pairs
                for spec in target_wait_specs_by_session[pair.session_id]
            ],
            "tool_wait_profile": args.tool_wait_profile,
            "tool_wait_profile_spec": args.tool_wait_profile_spec,
            "tool_wait_seed": args.tool_wait_seed,
            "task_replay_steps": args.task_replay_steps,
            "agentic_workload_profile": args.agentic_workload_profile,
            "trace_profile": args.trace_profile,
            "trace_controller_decisions": args.trace_controller_decisions,
            "trace_controller_completion_linkage": args.trace_controller_completion_linkage,
            "tool_wait_distribution": [
                {
                    "class": name,
                    "weight": weight,
                    "wait_min_ms": wait_min_ms,
                    "wait_max_ms": wait_max_ms,
                    "wait_ms": wait_min_ms if wait_min_ms == wait_max_ms else "",
                }
                for name, weight, wait_min_ms, wait_max_ms in tool_wait_distribution(
                    args.tool_wait_profile,
                    args.tool_wait_ms,
                    args.tool_wait_profile_spec,
                )
            ],
            "filler_sessions": args.filler_sessions,
            "concurrency": args.concurrency,
            "filler_backlog_mode": args.filler_backlog_mode,
            "filler_backlog_target": filler_backlog_target if args.filler_backlog_mode == "constant" else "",
            "filler_backlog_total": filler_backlog_total if args.filler_backlog_mode == "constant" else "",
            "target_prompt_tokens": args.target_prompt_tokens,
            "filler_prompt_tokens": args.filler_prompt_tokens,
            "filler_replay_deadlines": args.filler_replay_deadlines,
            "filler_replay_deadline_ms": (
                args.filler_replay_deadline_ms if args.filler_replay_deadline_ms > 0 else args.tool_wait_ms
            ),
            "controller_admission_aggressiveness": admission_aggressiveness
            if controller_active_priority_demotion_admission
            else "",
            "controller_oracle_safety_margin_ms": oracle_safety_margin_ms
            if args.mode == CONTROLLER_ORACLE_TIMELINE_MODE or args.mode in CONTROLLER_ORACLE_SAFE_SJF_MODES
            else "",
            "controller_safe_sjf_safety_margin_ms": safe_sjf_safety_margin_ms
            if args.mode in CONTROLLER_ORACLE_SAFE_SJF_MODES
            else "",
            "controller_safe_sjf_max_in_flight": safe_sjf_max_in_flight
            if args.mode in CONTROLLER_ORACLE_SAFE_SJF_MODES
            else "",
            "controller_safe_sjf_require_fit_before_replay": safe_sjf_require_fit
            if args.mode in CONTROLLER_ORACLE_SAFE_SJF_MODES
            else "",
            "controller_safe_sjf_idle_override": safe_sjf_idle_override
            if args.mode in CONTROLLER_ORACLE_SAFE_SJF_MODES
            else "",
            "controller_memory_admission_lookahead_ms": memory_admission_lookahead_ms()
            if controller_active_memory_admission
            else "",
            "controller_memory_admission_release_grace_ms": memory_admission_release_grace_ms()
            if controller_active_memory_admission
            else "",
            "controller_memory_admission_token_budget": memory_admission_token_budget()
            if controller_active_memory_admission
            else "",
            "controller_memory_admission_reserve_fraction": memory_admission_reserve_fraction()
            if controller_active_memory_admission
            else "",
            "controller_memory_admission_min_candidate_tokens": memory_admission_min_candidate_tokens()
            if controller_active_memory_admission
            else "",
        },
    )

    class SubmissionPriorityGate:
        def __init__(
            self,
            capacity: int,
            *,
            predictive_enabled: bool,
            trace_blockers: bool,
            blocker_max_ids: int,
            blocker_events: set[str],
        ) -> None:
            self.capacity = max(1, capacity)
            self.predictive_enabled = predictive_enabled
            self.trace_blockers = trace_blockers
            self.blocker_max_ids = max(0, blocker_max_ids)
            self.blocker_events = blocker_events
            self.pending = 0
            self.in_flight = 0
            self.submit_seq = 0
            self.wait_seq = 0
            self.condition = asyncio.Condition()
            self.heap: list[list[Any]] = []
            self.in_flight_entries: dict[str, dict[str, Any]] = {}

        def priority_for(self, meta: dict[str, Any]) -> tuple[tuple[float, int], str, str]:
            if self.predictive_enabled and meta.get("controller_predictive_deadline_queue"):
                try:
                    priority = int(float(meta.get("controller_sglang_priority")))
                except (TypeError, ValueError):
                    priority = 0
                due_offset = optional_float(meta.get("deadline_offset_ms"))
                due_component = int(due_offset * 1000) if due_offset is not None else 0
                return (-float(priority), due_component), "predictive_deadline_queue", str(priority)
            return (0.0, 0), "fifo", ""

        def prune_cancelled(self) -> None:
            while self.heap and not self.heap[0][3]:
                heapq.heappop(self.heap)

        @staticmethod
        def is_replay_like(meta: dict[str, Any]) -> bool:
            phase = str(meta.get("phase") or "")
            label = str(meta.get("label") or "")
            return (
                phase in {"replay", "pressure_filler"}
                or "_replay_" in label
                or meta.get("deadline_offset_ms") not in (None, "")
            )

        @staticmethod
        def compact_entry(meta: dict[str, Any], *, queued_at_ms: float, sort_key: Any) -> dict[str, Any]:
            return {
                "request_id": meta.get("label", ""),
                "session_id": meta.get("session_id", ""),
                "phase": meta.get("phase", ""),
                "request_group": meta.get("request_group", ""),
                "tool_wait_step": meta.get("tool_wait_step", ""),
                "tool_wait_class": meta.get("tool_wait_class", ""),
                "tool_wait_ms": meta.get("tool_wait_ms", ""),
                "deadline_offset_ms": meta.get("deadline_offset_ms", ""),
                "controller_sglang_priority": meta.get("controller_sglang_priority", ""),
                "controller_replay_rank": meta.get("controller_replay_rank", ""),
                "queued_offset_ms": round(queued_at_ms, 3),
                "submission_sort_key": str(sort_key),
            }

        def blocker_snapshot(
            self,
            *,
            snapshot_event: str,
            entry: list[Any],
            meta: dict[str, Any],
            queued_at_ms: float,
            pending_before_acquire: int,
            in_flight_before_acquire: int,
        ) -> dict[str, Any] | None:
            if not self.trace_blockers or snapshot_event not in self.blocker_events or not self.is_replay_like(meta):
                return None
            started = time.perf_counter()
            request_id = str(meta.get("label") or "")
            active_pending = [item for item in self.heap if item[3]]
            active_pending.sort(key=lambda item: (item[0], item[1]))
            try:
                current_index = active_pending.index(entry)
            except ValueError:
                current_index = 0
            pending_ahead = active_pending[:current_index]
            pending_all_except_self = [item for item in active_pending if item is not entry]
            in_flight_entries = sorted(
                self.in_flight_entries.values(),
                key=lambda item: (optional_float(item.get("acquired_offset_ms")) or 0.0, str(item.get("request_id") or "")),
            )

            def ids_from_pending(items: list[list[Any]]) -> list[str]:
                return [str(item[4].get("request_id") or item[2] or "") for item in items[: self.blocker_max_ids]]

            def ids_from_inflight(items: list[dict[str, Any]]) -> list[str]:
                return [str(item.get("request_id") or "") for item in items[: self.blocker_max_ids]]

            pending_ahead_ids = ids_from_pending(pending_ahead)
            pending_ids = ids_from_pending(pending_all_except_self)
            in_flight_ids = ids_from_inflight(in_flight_entries)
            snapshot_build_ms = (time.perf_counter() - started) * 1000.0
            return {
                "event": "m27.replay_blocker_snapshot",
                "snapshot_event": snapshot_event,
                "snapshot_offset_ms": round(offset_ms(), 3),
                "snapshot_build_ms": round(snapshot_build_ms, 3),
                "trace_replay_blockers_max_ids": self.blocker_max_ids,
                "request_id": request_id,
                "session_id": meta.get("session_id", ""),
                "phase": meta.get("phase", ""),
                "request_group": meta.get("request_group", ""),
                "tool_wait_step": meta.get("tool_wait_step", ""),
                "tool_wait_class": meta.get("tool_wait_class", ""),
                "tool_wait_ms": meta.get("tool_wait_ms", ""),
                "deadline_offset_ms": meta.get("deadline_offset_ms", ""),
                "controller_sglang_priority": meta.get("controller_sglang_priority", ""),
                "controller_replay_rank": meta.get("controller_replay_rank", ""),
                "client_pending_before_acquire": pending_before_acquire,
                "client_inflight_before_acquire": in_flight_before_acquire,
                "pending_ahead_count": len(pending_ahead),
                "pending_snapshot_count": len(pending_all_except_self),
                "inflight_snapshot_count": len(in_flight_entries),
                "pending_ahead_ids": pending_ahead_ids,
                "pending_snapshot_ids": pending_ids,
                "inflight_snapshot_ids": in_flight_ids,
                "pending_ahead_truncated": len(pending_ahead) > len(pending_ahead_ids),
                "pending_snapshot_truncated": len(pending_all_except_self) > len(pending_ids),
                "inflight_snapshot_truncated": len(in_flight_entries) > len(in_flight_ids),
                "queued_offset_ms": round(queued_at_ms, 3),
            }

        async def acquire(self, meta: dict[str, Any]) -> dict[str, Any]:
            queued_at_ms = offset_ms()
            sort_key, policy, submission_priority = self.priority_for(meta)
            async with self.condition:
                self.pending += 1
                self.wait_seq += 1
                pending_before_acquire = self.pending
                in_flight_before_acquire = self.in_flight
                compact = self.compact_entry(meta, queued_at_ms=queued_at_ms, sort_key=sort_key)
                entry: list[Any] = [sort_key, self.wait_seq, meta.get("label", ""), True, compact]
                heapq.heappush(self.heap, entry)
                blocker_snapshots = [
                    snapshot
                    for snapshot in [
                        self.blocker_snapshot(
                            snapshot_event="before_acquire",
                            entry=entry,
                            meta=meta,
                            queued_at_ms=queued_at_ms,
                            pending_before_acquire=pending_before_acquire,
                            in_flight_before_acquire=in_flight_before_acquire,
                        )
                    ]
                    if snapshot is not None
                ]
                self.condition.notify_all()
                try:
                    while True:
                        self.prune_cancelled()
                        if self.in_flight < self.capacity and self.heap and self.heap[0] is entry:
                            heapq.heappop(self.heap)
                            entry[3] = False
                            self.pending = max(0, self.pending - 1)
                            self.in_flight += 1
                            self.submit_seq += 1
                            acquired_at_ms = offset_ms()
                            compact["acquired_offset_ms"] = round(acquired_at_ms, 3)
                            compact["client_submit_seq"] = self.submit_seq
                            self.in_flight_entries[str(compact.get("request_id") or "")] = compact
                            submit_snapshot = self.blocker_snapshot(
                                snapshot_event="submit",
                                entry=entry,
                                meta=meta,
                                queued_at_ms=queued_at_ms,
                                pending_before_acquire=pending_before_acquire,
                                in_flight_before_acquire=in_flight_before_acquire,
                            )
                            if submit_snapshot is not None:
                                blocker_snapshots.append(submit_snapshot)
                            info = {
                                "client_submit_seq": self.submit_seq,
                                "client_sem_capacity": self.capacity,
                                "client_pending_before_acquire": pending_before_acquire,
                                "client_inflight_before_acquire": in_flight_before_acquire,
                                "client_pending_at_submit": self.pending,
                                "client_inflight_at_submit": self.in_flight,
                                "client_queue_wait_ms": round(acquired_at_ms - queued_at_ms, 3),
                                "client_submission_policy": policy,
                                "client_submission_priority": submission_priority,
                                "client_blocker_snapshots": blocker_snapshots,
                            }
                            self.condition.notify_all()
                            return info
                        await self.condition.wait()
                except BaseException:
                    if entry[3]:
                        entry[3] = False
                        self.pending = max(0, self.pending - 1)
                        self.condition.notify_all()
                    raise

        async def release(self, request_id: str | None = None) -> None:
            async with self.condition:
                self.in_flight = max(0, self.in_flight - 1)
                if request_id:
                    self.in_flight_entries.pop(str(request_id), None)
                self.condition.notify_all()

    client_submission_gate = SubmissionPriorityGate(
        args.concurrency,
        predictive_enabled=controller_active_predictive_deadline_queue,
        trace_blockers=env_flag("TRACE_REPLAY_BLOCKERS", False),
        blocker_max_ids=max(0, int(args.trace_replay_blockers_max_ids)),
        blocker_events={
            item.strip()
            for item in str(args.trace_replay_blockers_events or "").split(",")
            if item.strip()
        },
    )
    async def bounded_request(prompt: str, meta: dict[str, Any]) -> None:
        meta = await apply_memory_admission_if_needed(meta)
        gate_info = await client_submission_gate.acquire(meta)
        blocker_snapshots = list(gate_info.pop("client_blocker_snapshots", []) or [])
        meta = {
            **meta,
            **gate_info,
            "harness_controller_signal": build_harness_controller_signal(meta),
        }
        try:
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
                    "tool_wait_step": meta.get("tool_wait_step", ""),
                    "task_replay_steps": meta.get("task_replay_steps", ""),
                    "tool_wait_profile": meta.get("tool_wait_profile", ""),
                    "tool_wait_class": meta.get("tool_wait_class", ""),
                    "harness_controller_signal": meta.get("harness_controller_signal", {}),
                    "offset_ms": round(offset_ms(), 3),
                    "priority_intent": meta.get("priority_intent", ""),
                    "workflow_node": meta.get("workflow_node", ""),
                    "workflow_node_goal": meta.get("workflow_node_goal", ""),
                    "inference_source": meta.get("inference_source", ""),
                    "expected_inferred_priority": meta.get("expected_inferred_priority", ""),
                    "harness_input_priority_signal": meta.get("harness_input_priority_signal", ""),
                    "harness_input_priority_signal_source": meta.get("harness_input_priority_signal_source", ""),
                    "client_submit_seq": meta.get("client_submit_seq", ""),
                    "client_sem_capacity": meta.get("client_sem_capacity", ""),
                    "client_pending_before_acquire": meta.get("client_pending_before_acquire", ""),
                    "client_inflight_before_acquire": meta.get("client_inflight_before_acquire", ""),
                    "client_pending_at_submit": meta.get("client_pending_at_submit", ""),
                    "client_inflight_at_submit": meta.get("client_inflight_at_submit", ""),
                    "client_queue_wait_ms": meta.get("client_queue_wait_ms", ""),
                    "client_submission_policy": meta.get("client_submission_policy", ""),
                    "client_submission_priority": meta.get("client_submission_priority", ""),
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
                    "tool_wait_step": meta.get("tool_wait_step", ""),
                    "task_replay_steps": meta.get("task_replay_steps", ""),
                    "tool_wait_profile": meta.get("tool_wait_profile", ""),
                    "tool_wait_class": meta.get("tool_wait_class", ""),
                    "harness_controller_signal": meta.get("harness_controller_signal", {}),
                    "client_submit_seq": meta.get("client_submit_seq", ""),
                    "client_queue_wait_ms": meta.get("client_queue_wait_ms", ""),
                    "client_submission_policy": meta.get("client_submission_policy", ""),
                    "client_submission_priority": meta.get("client_submission_priority", ""),
                    "offset_ms": round(offset_ms(), 3),
                },
            )
        finally:
            await client_submission_gate.release(str(meta.get("label") or ""))
            for snapshot in blocker_snapshots:
                write_trace(args.trace, snapshot)

    def prepared_prefix_control_url() -> str:
        return os.environ.get("AGENTIC_KV_PREPARE_CONTROL_URL", "http://127.0.0.1:31991/prepare_prefix_kv")

    async def prepare_prefix_kv_control_request(
        meta: dict[str, Any],
        prefetch_window: dict[str, Any],
        *,
        plan_only: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "session_id": meta.get("session_id", ""),
            "prefix_id": meta.get("prefix_id", ""),
            "prompt_hash": meta.get("prompt_hash", ""),
            "request_id": meta.get("parent_request_id", "") or meta.get("request_id", ""),
            "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
            "mem_quota": meta.get("direct_kv_mem_quota", ""),
            "plan_only": plan_only,
            "wait": not plan_only,
            "wait_timeout_ms": max(1, int(max(0.0, float(prefetch_window["latest_finish_ms"]) - offset_ms()))),
            "control_timeout_ms": max(1, int(max(0.0, float(prefetch_window["latest_finish_ms"]) - offset_ms())) + 1_000),
            "source": "controller_proactive_kv_management",
        }
        event_stem = (
            "m27.targeted_kv_prefetch.prepare_prefix_control_plan"
            if plan_only
            else "m27.targeted_kv_prefetch.prepare_prefix_control"
        )
        write_trace(
            args.trace,
            {
                "event": f"{event_stem}_request",
                "session_id": meta.get("session_id", ""),
                "prefix_id": meta.get("prefix_id", ""),
                "request_id": meta.get("parent_request_id", "") or meta.get("request_id", ""),
                "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
                "prompt_hash": meta.get("prompt_hash", ""),
                "prepare_control_url": prepared_prefix_control_url(),
                "plan_only": plan_only,
                "offset_ms": round(offset_ms(), 3),
            },
        )
        try:
            async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_keepalive_connections=0)) as client:
                response = await client.post(prepared_prefix_control_url(), json=payload)
                result = response.json()
                result.setdefault("http_status", response.status_code)
        except Exception as exc:  # noqa: BLE001
            result = {
                "ok": False,
                "status": "prepare_control_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        write_trace(
            args.trace,
            {
                "event": f"{event_stem}_result",
                "session_id": meta.get("session_id", ""),
                "prefix_id": meta.get("prefix_id", ""),
                "request_id": meta.get("parent_request_id", "") or meta.get("request_id", ""),
                "expected_replay_request_id": meta.get("expected_replay_request_id", ""),
                "prompt_hash": meta.get("prompt_hash", ""),
                "result": result,
                "plan_only": plan_only,
                "offset_ms": round(offset_ms(), 3),
            },
        )
        return result

    def direct_load_selective_admission_enabled() -> bool:
        return env_truthy("CONTROLLER_DIRECT_LOAD_SELECTIVE_ADMISSION", default=True)

    def direct_load_min_saved_tokens() -> int:
        try:
            return max(0, int(float(os.environ.get("CONTROLLER_DIRECT_LOAD_MIN_SAVED_TOKENS", "1024") or "1024")))
        except ValueError:
            return 1024

    def direct_load_admission_from_plan(plan_result: dict[str, Any]) -> dict[str, Any]:
        status = str(plan_result.get("status") or "")
        try:
            host_tokens = int(plan_result.get("host_tokens") or 0)
        except (TypeError, ValueError):
            host_tokens = 0
        min_tokens = direct_load_min_saved_tokens()
        if not direct_load_selective_admission_enabled():
            return {
                "admitted": True,
                "reason": "selective_admission_disabled",
                "plan_status": status,
                "host_tokens": host_tokens,
                "min_saved_tokens": min_tokens,
            }
        if status == "would_load_back":
            if host_tokens < min_tokens:
                return {
                    "admitted": False,
                    "reason": "skip_host_resident_prefix_too_small",
                    "plan_status": status,
                    "host_tokens": host_tokens,
                    "min_saved_tokens": min_tokens,
                }
            return {
                "admitted": True,
                "reason": "admit_host_resident_evicted_prefix",
                "plan_status": status,
                "host_tokens": host_tokens,
                "min_saved_tokens": min_tokens,
            }
        if status == "already_device_resident":
            return {
                "admitted": False,
                "reason": "skip_already_device_resident",
                "plan_status": status,
                "host_tokens": host_tokens,
                "min_saved_tokens": min_tokens,
            }
        return {
            "admitted": False,
            "reason": f"skip_prepare_plan_status_{status or 'unknown'}",
            "plan_status": status,
            "host_tokens": host_tokens,
            "min_saved_tokens": min_tokens,
        }

    def full_controller_priority_for_pair(index: int) -> tuple[int, int, int]:
        total = max(1, len(pairs))
        rank = min(total, max(1, index + 1))
        if total <= 1:
            return int(base_meta_priority()), rank, total
        # Earlier replay due times get a small ladder above the normal urgent
        # priority so P5 does not collapse into a flat all-urgent tie.
        return int(base_meta_priority() + (total - rank) * 10), rank, total

    def base_meta_priority() -> int:
        return 100

    def kv_storage_metadata() -> dict[str, Any]:
        if not storage_hicache_mode(args.mode):
            return {
                "kv_storage_enabled": False,
                "kv_storage_backend": "",
                "kv_storage_prefetch_policy": "",
                "kv_storage_path": "",
                "kv_storage_signal_source": "",
            }
        return {
            "kv_storage_enabled": True,
            "kv_storage_backend": os.environ.get("HICACHE_STORAGE_BACKEND", ""),
            "kv_storage_prefetch_policy": os.environ.get("HICACHE_STORAGE_PREFETCH_POLICY", ""),
            "kv_storage_path": os.environ.get("HICACHE_STORAGE_PATH", ""),
            "kv_storage_signal_source": "sglang_hicache_storage_backend",
        }

    async def run_pair(pair: HarnessPair, index: int) -> None:
        await sleep_until(index * args.arrival_gap_ms)
        target_wait_specs = target_wait_specs_by_session[pair.session_id]
        storage_meta = kv_storage_metadata()
        target_initial_prompt = pair.prompt
        target_initial_max_tokens = 8
        target_initial_workload_meta: dict[str, Any] = {
            "agentic_workload_profile": args.agentic_workload_profile,
            "workload_request_kind": "synthetic_target_initial",
            "workload_phase_family": "initial_turn",
            "workload_prompt_tokens_target": args.target_prompt_tokens,
            "workload_max_tokens": 8,
            "workload_description": "fixed synthetic target initial request",
        }
        if args.agentic_workload_profile == REALISTIC_AGENTIC_PROFILE and not args.workload_jsonl:
            target_initial_shape = realistic_initial_shape(
                seed=args.tool_wait_seed,
                stream_key=workload_stream_key(pair, "target", "initial"),
            )
            target_initial_prompt = make_agentic_workload_prompt(
                session_id=pair.session_id,
                shape=target_initial_shape,
                role="target",
                stage="initial",
                step_index=0,
                total_steps=len(target_wait_specs),
            )
            target_initial_max_tokens = target_initial_shape.max_tokens
            target_initial_workload_meta = workload_meta(target_initial_shape, args.agentic_workload_profile)
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
                "tool_wait_profile": args.tool_wait_profile,
                "tool_wait_seed": args.tool_wait_seed,
                "task_replay_steps": len(target_wait_specs),
                "sampled_tool_waits_ms": [spec.wait_ms for spec in target_wait_specs],
                "sampled_tool_wait_classes": [spec.wait_class for spec in target_wait_specs],
                "prompt_tokens": estimate_tokens(target_initial_prompt),
                "agentic_workload_profile": args.agentic_workload_profile,
                "workload_request_kind": target_initial_workload_meta.get("workload_request_kind", ""),
                "workload_phase_family": target_initial_workload_meta.get("workload_phase_family", ""),
                "workload_prompt_tokens_target": target_initial_workload_meta.get("workload_prompt_tokens_target", ""),
                "workload_max_tokens": target_initial_workload_meta.get("workload_max_tokens", ""),
                **storage_meta,
            },
        )
        base_meta = {
            "harness": args.harness,
            "mode": args.mode,
            "pressure_level": args.pressure_level,
            "model": args.model,
            "session_generation": 1,
            "prefix_id": f"{pair.session_id}:prefix",
            "high_priority": 100,
            "speculative_prefill_priority": 50,
            "low_priority": -100,
            "tool_wait_ms": args.tool_wait_ms,
            "tool_wait_profile": args.tool_wait_profile,
            "tool_wait_profile_spec": args.tool_wait_profile_spec,
            "tool_wait_seed": args.tool_wait_seed,
            "task_replay_steps": len(target_wait_specs),
            "active_background_requests": configured_background_requests,
            "demotable_background_requests": configured_background_requests,
            "background_safe_to_demote": configured_background_requests > 0,
            "background_can_delay_ms": max(500, args.tool_wait_ms * 2),
            "concurrency": args.concurrency,
            "filler_sessions": args.filler_sessions,
            "filler_backlog_mode": args.filler_backlog_mode,
            "filler_backlog_target": filler_backlog_target if args.filler_backlog_mode == "constant" else "",
            "filler_backlog_total": filler_backlog_total if args.filler_backlog_mode == "constant" else "",
            "cost_feedback_allow_background_demote": True,
            "reuse_probability": 0.98,
            "recompute_cost_tokens": args.target_prompt_tokens,
            "agentic_workload_profile": args.agentic_workload_profile,
            "controller_admission_aggressiveness": admission_aggressiveness
            if controller_active_priority_demotion_admission
            else "",
            "_trace_path": str(args.trace),
            "nat_inferred_prefix_total_requests": 10,
            "nat_inferred_prefix_osl": 512,
            "nat_inferred_prefix_iat": 50,
            "native_cache_profile": {
                "enabled": harness_native_cache_enabled({"mode": args.mode}),
                "policy": "harness_decides_gateway_translates_only",
                "cache_key_seed": f"{args.harness}:{args.pressure_level}",
            },
            **storage_meta,
        }
        initial_meta = {
            **base_meta,
            "session_id": pair.session_id,
            "phase": "initial_turn",
            "label": f"{pair.session_id}_initial",
            "task_index": pair.task_index,
            "prompt_hash": prompt_hash(target_initial_prompt),
            "prompt_tokens": estimate_tokens(target_initial_prompt),
            "priority_label": "normal"
            if (controller_active_deadline_fair or controller_active_predictive_deadline_queue)
            else "high",
            "max_tokens": target_initial_max_tokens,
            "tool_wait_step": 0,
            "speculative_prefill": args.mode == "e2e_priority_hints_speculative_prefill",
            **target_initial_workload_meta,
        }
        initial_meta.update(value_aware_eviction_metadata(initial_meta))
        initial_meta = attach_harness_priority_metadata(initial_meta)
        await bounded_request(target_initial_prompt, initial_meta)
        target_current_prompt = target_initial_prompt
        filler_base_meta = base_meta
        filler_tasks: list[asyncio.Task[None]] = []
        active_filler_tasks: set[asyncio.Task[None]] = set()
        filler_backlog_stop = asyncio.Event()
        filler_backlog_producer_task: asyncio.Task[None] | None = None
        next_filler_index = 0
        active_demote_command: dict[str, Any] | None = None
        demote_restore_active = False

        def schedule_all_request_targeted_kv_prefetch(
            tool_wait_meta: dict[str, Any],
            current_prompt: str,
            replay_label: str,
            tool_start_ms: float,
            replay_due_ms: float,
            wait_spec: ToolWaitSpec,
            total_steps: int,
            request_group: str,
        ) -> asyncio.Task[None] | None:
            session_id = str(tool_wait_meta.get("session_id") or "")
            controller_result = record_controller_event(
                f"{session_id}:tool_started:{wait_spec.step_index}",
                EventType.TOOL_STARTED,
                {**tool_wait_meta, "expected_replay_request_id": replay_label},
                monotonic_ms=int(tool_start_ms),
                expected_completion_ms=int(replay_due_ms),
                deadline_after_completion_ms=int(wait_spec.wait_ms),
                eta_uncertainty_ms=max(1, int(wait_spec.wait_ms) // 4),
            )
            targeted_prefetch_command = selected_controller_command(controller_result, kv_action="prefetch")
            if targeted_prefetch_command is None:
                return None

            direct_hook_available = str(targeted_prefetch_command.get("backend_acted")).lower() == "true"
            direct_load_prompt_tokens = estimate_tokens(current_prompt)
            prefetch_window = controller_direct_load_window(
                tool_start_ms=tool_start_ms,
                replay_due_ms=replay_due_ms,
                prompt_tokens=direct_load_prompt_tokens,
                wait_class=wait_spec.wait_class,
            )
            targeted_prefetch_lead_ms = int(max(0.0, replay_due_ms - float(prefetch_window["start_ms"])))
            targeted_prefetch_start_ms = float(prefetch_window["start_ms"])
            direct_kv_h2d_priority = deadline_fair_priority_for_due(replay_due_ms)
            direct_load_mechanism = controller_direct_load_mechanism()
            trigger_label = f"{session_id}_controller_proactive_direct_load_{wait_spec.step_index:02d}"
            targeted_event_base = {
                "session_id": session_id,
                "mode": args.mode,
                "harness": args.harness,
                "request_id": str(tool_wait_meta.get("label") or ""),
                "request_group": request_group,
                "prefetch_request_id": trigger_label,
                "expected_replay_request_id": replay_label,
                "prefix_id": tool_wait_meta.get("prefix_id", ""),
                "tool_wait_step": wait_spec.step_index,
                "task_replay_steps": total_steps,
                "tool_wait_profile": args.tool_wait_profile,
                "tool_wait_class": wait_spec.wait_class,
                "tool_wait_ms": int(wait_spec.wait_ms),
                "reuse_probability": tool_wait_meta.get("reuse_probability", ""),
                "recompute_cost_tokens": tool_wait_meta.get("recompute_cost_tokens", ""),
                "direct_load_prompt_tokens": direct_load_prompt_tokens,
                "direct_load_estimated_ms": prefetch_window["estimated_ms"],
                "direct_load_safety_margin_ms": prefetch_window["safety_margin_ms"],
                "direct_load_available_slack_ms": round(float(prefetch_window["available_slack_ms"]), 3),
                "direct_load_latest_finish_offset_ms": round(float(prefetch_window["latest_finish_ms"]), 3),
                "direct_load_admission_reason": prefetch_window["reason"],
                "direct_load_contract": prefetch_window["contract"],
                "direct_load_allowed_wait_classes": prefetch_window["allowed_wait_classes"],
                "direct_load_mechanism": direct_load_mechanism,
                "direct_load_execution_path": "sglang_prepare_prefix_control",
                "direct_kv_h2d_priority": direct_kv_h2d_priority,
                "direct_kv_h2d_priority_source": "harness_tool_wait_eta_replay_due",
                "controller_decision_id": targeted_prefetch_command.get("controller_decision_id", ""),
                "controller_command_id": targeted_prefetch_command.get("command_id", ""),
                "controller_command_reason": targeted_prefetch_command.get("reason", ""),
                "backend_name": targeted_prefetch_command.get("backend_name", ""),
                "backend_accepted": targeted_prefetch_command.get("backend_accepted", ""),
                "backend_acted": targeted_prefetch_command.get("backend_acted", ""),
                "backend_reason": targeted_prefetch_command.get("backend_reason", ""),
                "direct_hook_available": direct_hook_available,
                "tool_start_offset_ms": round(tool_start_ms, 3),
                "replay_due_offset_ms": round(replay_due_ms, 3),
                "targeted_prefetch_lead_ms": targeted_prefetch_lead_ms,
                "targeted_prefetch_start_offset_ms": round(targeted_prefetch_start_ms, 3),
                "offset_ms": round(offset_ms(), 3),
            }
            write_trace(args.trace, {**targeted_event_base, "event": "m27.targeted_kv_prefetch.requested"})
            write_trace(
                args.trace,
                {
                    **targeted_event_base,
                    "event": "m27.targeted_kv_prefetch.acted"
                    if direct_hook_available
                    else "m27.targeted_kv_prefetch.unavailable",
                },
            )
            if not prefetch_window["admitted"]:
                write_trace(
                    args.trace,
                    {
                        **targeted_event_base,
                        "event": "m27.targeted_kv_prefetch.skipped",
                        "status": "skipped",
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
                return None
            if not direct_hook_available:
                return None

            direct_load_meta = {
                **tool_wait_meta,
                "phase": "hint_prefetch",
                "label": trigger_label,
                "prompt_hash": prompt_hash(current_prompt),
                "priority_label": "background",
                "deadline_offset_ms": round(replay_due_ms, 3),
                "replay_due_offset_ms": round(replay_due_ms, 3),
                "expected_tool_return_ms": int(wait_spec.wait_ms),
                "next_ready_eta_ms": int(wait_spec.wait_ms),
                "direct_kv_h2d_priority": direct_kv_h2d_priority,
                "direct_kv_h2d_priority_source": "harness_tool_wait_eta_replay_due",
                "max_tokens": 1,
                "controller_targeted_kv_prefetch": True,
                "controller_proactive_kv_management": True,
                "prefetch_action": direct_load_mechanism,
                "parent_request_id": str(tool_wait_meta.get("label") or ""),
                "expected_replay_request_id": replay_label,
                "controller_decision_id": targeted_prefetch_command.get("controller_decision_id", ""),
                "controller_command_id": targeted_prefetch_command.get("command_id", ""),
                "controller_kv_translation": f"controller.proactive_kv_management={direct_load_mechanism}",
                "prepared_kv_contract": prefetch_window["contract"],
                "prepared_kv_expected_replay_id": replay_label,
            }
            direct_load_meta = attach_harness_priority_metadata(direct_load_meta)

            async def run_targeted_direct_load() -> None:
                await sleep_until(targeted_prefetch_start_ms)
                timeout_s = max(0.0, (float(prefetch_window["latest_finish_ms"]) - offset_ms()) / 1000.0)
                if timeout_s <= 0:
                    write_trace(
                        args.trace,
                        {
                            **targeted_event_base,
                            "event": "m27.targeted_kv_prefetch.direct_load_skipped_expired",
                            "status": "skipped",
                            "offset_ms": round(offset_ms(), 3),
                        },
                    )
                    return
                plan_result = await prepare_prefix_kv_control_request(
                    direct_load_meta,
                    prefetch_window,
                    plan_only=True,
                )
                selective_admission = direct_load_admission_from_plan(plan_result)
                if not selective_admission["admitted"]:
                    write_trace(
                        args.trace,
                        {
                            **targeted_event_base,
                            "event": "m27.targeted_kv_prefetch.skipped",
                            "status": "skipped",
                            "prepared_kv_status": "not_needed",
                            "selective_admission_reason": selective_admission["reason"],
                            "selective_admission_status": selective_admission["plan_status"],
                            "selective_admission_host_tokens": selective_admission["host_tokens"],
                            "selective_admission_min_saved_tokens": selective_admission["min_saved_tokens"],
                            "prepare_prefix_plan_result": plan_result,
                            "offset_ms": round(offset_ms(), 3),
                        },
                    )
                    return
                write_trace(
                    args.trace,
                    {
                        **targeted_event_base,
                        "event": "m27.targeted_kv_prefetch.direct_load_start",
                        "selective_admission_reason": selective_admission["reason"],
                        "selective_admission_status": selective_admission["plan_status"],
                        "selective_admission_host_tokens": selective_admission["host_tokens"],
                        "selective_admission_min_saved_tokens": selective_admission["min_saved_tokens"],
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
                try:
                    result = await prepare_prefix_kv_control_request(direct_load_meta, prefetch_window)
                    if not result.get("ok"):
                        write_trace(
                            args.trace,
                            {
                                **targeted_event_base,
                                "event": "m27.targeted_kv_prefetch.direct_load_error",
                                "status": str(result.get("status") or "error"),
                                "prepared_kv_status": "not_ready",
                                "prepare_prefix_result": result,
                                "offset_ms": round(offset_ms(), 3),
                            },
                        )
                        return
                    write_trace(
                        args.trace,
                        {
                            **targeted_event_base,
                            "event": "m27.targeted_kv_prefetch.direct_load_end",
                            "status": "ok",
                            "prepared_kv_status": "ready",
                            "direct_load_mechanism": direct_load_mechanism,
                            "offset_ms": round(offset_ms(), 3),
                        },
                    )
                except asyncio.TimeoutError:
                    write_trace(
                        args.trace,
                        {
                            **targeted_event_base,
                            "event": "m27.targeted_kv_prefetch.direct_load_timeout_before_replay",
                            "status": "timeout",
                            "prepared_kv_status": "not_ready",
                            "offset_ms": round(offset_ms(), 3),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    write_trace(
                        args.trace,
                        {
                            **targeted_event_base,
                            "event": "m27.targeted_kv_prefetch.direct_load_error",
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "offset_ms": round(offset_ms(), 3),
                        },
                    )

            return asyncio.create_task(run_targeted_direct_load())

        def launch_filler_task(idx: int, *, reason: str) -> asyncio.Task[None]:
            task = asyncio.create_task(
                run_filler(
                    args.gateway_base,
                    args.model,
                    pair,
                    idx,
                    filler_base_meta,
                    args.filler_prompt_tokens,
                    wait_specs=sample_filler_wait_specs(pair, idx),
                    trace=args.trace,
                    workload_start=workload_start,
                    filler_replay_deadlines=args.filler_replay_deadlines,
                    demotion_state=controller_demotion_state,
                    admission_gate_state=controller_admission_gate_state,
                    mode=args.mode,
                    oracle_safety_margin_ms=oracle_safety_margin_ms,
                    safe_filler_scheduler=safe_filler_scheduler,
                    agentic_workload_profile=args.agentic_workload_profile,
                    tool_wait_seed=args.tool_wait_seed,
                    trace_controller_completion_linkage=env_flag(
                        "TRACE_CONTROLLER_COMPLETION_LINKAGE",
                        False,
                    ),
                    submit_request=bounded_request,
                    deadline_fair_priority=assign_deadline_fair_priority
                    if controller_active_deadline_fair
                    else None,
                    predictive_deadline_priority=assign_predictive_deadline_priority
                    if controller_active_predictive_deadline_queue
                    else None,
                    targeted_kv_prefetch=schedule_all_request_targeted_kv_prefetch
                    if controller_active_proactive_kv_management
                    else None,
                    memory_admission_register=register_memory_admission_replay
                    if controller_active_memory_admission
                    else None,
                )
            )
            task.set_name(f"{pair.session_id}_pressure_{idx:03d}")
            filler_tasks.append(task)
            write_trace(
                args.trace,
                {
                    "event": "m27.filler_backlog.launch",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "pressure_level": args.pressure_level,
                    "task_index": pair.task_index,
                    "filler_index": idx,
                    "filler_task_name": task.get_name(),
                    "filler_backlog_mode": args.filler_backlog_mode,
                    "filler_backlog_target": filler_backlog_target if args.filler_backlog_mode == "constant" else "",
                    "filler_backlog_total": filler_backlog_total if args.filler_backlog_mode == "constant" else "",
                    "launch_reason": reason,
                    "active_filler_tasks_before_launch": len(active_filler_tasks),
                    "offset_ms": round(offset_ms(), 3),
                },
            )
            return task

        async def run_constant_filler_backlog() -> None:
            nonlocal next_filler_index
            write_trace(
                args.trace,
                {
                    "event": "m27.filler_backlog.start",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "pressure_level": args.pressure_level,
                    "task_index": pair.task_index,
                    "filler_backlog_mode": args.filler_backlog_mode,
                    "filler_backlog_target": filler_backlog_target,
                    "filler_backlog_total": filler_backlog_total,
                    "offset_ms": round(offset_ms(), 3),
                },
            )
            while len(active_filler_tasks) < filler_backlog_target and next_filler_index < filler_backlog_total:
                task = launch_filler_task(next_filler_index, reason="initial_backlog_fill")
                active_filler_tasks.add(task)
                next_filler_index += 1
            while active_filler_tasks:
                done, _pending = await asyncio.wait(
                    active_filler_tasks,
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    active_filler_tasks.discard(task)
                    if not filler_backlog_stop.is_set() and next_filler_index < filler_backlog_total:
                        replacement = launch_filler_task(next_filler_index, reason="replenish_after_completion")
                        active_filler_tasks.add(replacement)
                        write_trace(
                            args.trace,
                            {
                                "event": "m27.filler_backlog.replenish",
                                "session_id": pair.session_id,
                                "mode": args.mode,
                                "harness": args.harness,
                                "pressure_level": args.pressure_level,
                                "task_index": pair.task_index,
                                "completed_filler_task_name": task.get_name(),
                                "replacement_filler_index": next_filler_index,
                                "active_filler_tasks_after_replenish": len(active_filler_tasks),
                                "filler_backlog_target": filler_backlog_target,
                                "filler_backlog_total": filler_backlog_total,
                                "offset_ms": round(offset_ms(), 3),
                            },
                        )
                        next_filler_index += 1
                if not done and filler_backlog_stop.is_set():
                    continue
                if not active_filler_tasks:
                    break
            write_trace(
                args.trace,
                {
                    "event": "m27.filler_backlog.end",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "pressure_level": args.pressure_level,
                    "task_index": pair.task_index,
                    "filler_backlog_mode": args.filler_backlog_mode,
                    "filler_tasks_launched": len(filler_tasks),
                    "filler_backlog_target": filler_backlog_target,
                    "filler_backlog_total": filler_backlog_total,
                    "offset_ms": round(offset_ms(), 3),
                },
            )

        def activate_controller_demote(
            controller_demote_command: dict[str, Any],
            *,
            step_base_meta: dict[str, Any],
            wait_spec: ToolWaitSpec,
            tool_start_ms: float,
            replay_due_ms: float,
            trigger: str,
        ) -> None:
            nonlocal active_demote_command, demote_restore_active, filler_base_meta
            active_demote_command = controller_demote_command
            demote_restore_active = True
            demote_meta = {
                "controller_demote_restore_active": "yes",
                "controller_demote_priority": -100,
                "controller_demote_decision_id": controller_demote_command.get("controller_decision_id", ""),
                "controller_demote_command_id": controller_demote_command.get("command_id", ""),
                "controller_demote_translation": "controller.demote=pressure_filler.priority:-100",
                "controller_demote_trigger": trigger,
            }
            owner = f"{pair.session_id}:{wait_spec.step_index}"
            owners = controller_demotion_state.setdefault("owners", set())
            if isinstance(owners, set):
                owners.add(owner)
            controller_demotion_state["active"] = True
            controller_demotion_state["meta"] = demote_meta
            filler_base_meta = {**step_base_meta, **demote_meta}
            if controller_active_priority_demotion_admission:
                controller_admission_gate_event.clear()
                controller_admission_gate_state["active"] = True
                controller_admission_gate_state["meta"] = {
                    "session_id": pair.session_id,
                    "tool_wait_step": wait_spec.step_index,
                    "reason": "hold filler/background admission during target replay critical window",
                    "aggressiveness": admission_aggressiveness,
                    "open_offset_ms": round(offset_ms(), 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                }
                write_trace(
                    args.trace,
                    {
                        "event": "m27.controller_admission_gate.window_open",
                        "controller_policy": args.mode,
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "pressure_level": args.pressure_level,
                        "task_index": pair.task_index,
                        "tool_wait_step": wait_spec.step_index,
                        "task_replay_steps": len(target_wait_specs),
                        "tool_wait_profile": args.tool_wait_profile,
                        "tool_wait_class": wait_spec.wait_class,
                        "tool_wait_ms": wait_spec.wait_ms,
                        "demote_trigger": trigger,
                        "controller_admission_aggressiveness": admission_aggressiveness,
                        "controller_earlyprepare_lead_ms": controller_admission_lead_ms(args.mode, wait_spec.wait_ms)
                        if admission_aggressiveness == "earlyprepare"
                        else "",
                        "admission_action": "hold_background_until_target_replay_completes",
                        "demotable_background_requests": configured_background_requests,
                        "tool_start_offset_ms": round(tool_start_ms, 3),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "open_offset_ms": round(offset_ms(), 3),
                    },
                )
            write_trace(
                args.trace,
                {
                    "event": "m27.controller_demote_restore.demote_start",
                    "controller_policy": args.mode,
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "pressure_level": args.pressure_level,
                    "task_index": pair.task_index,
                    "tool_wait_step": wait_spec.step_index,
                    "task_replay_steps": len(target_wait_specs),
                    "tool_wait_profile": args.tool_wait_profile,
                    "tool_wait_class": wait_spec.wait_class,
                    "tool_wait_ms": wait_spec.wait_ms,
                    "controller_decision_id": controller_demote_command.get("controller_decision_id", ""),
                    "controller_command_id": controller_demote_command.get("command_id", ""),
                    "controller_command_reason": controller_demote_command.get("reason", ""),
                    "controller_decision_reason": controller_demote_command.get("controller_decision_reason", ""),
                    "backend_name": controller_demote_command.get("backend_name", ""),
                    "backend_accepted": controller_demote_command.get("backend_accepted", ""),
                    "backend_acted": controller_demote_command.get("backend_acted", ""),
                    "backend_reason": controller_demote_command.get("backend_reason", ""),
                    "demoted_phase": "pressure_filler",
                    "demoted_priority": -100,
                    "demote_trigger": trigger,
                    "controller_admission_aggressiveness": admission_aggressiveness
                    if controller_active_priority_demotion_admission
                    else "",
                    "controller_earlyprepare_lead_ms": controller_admission_lead_ms(args.mode, wait_spec.wait_ms)
                    if admission_aggressiveness == "earlyprepare"
                    else "",
                    "demotable_background_requests": configured_background_requests,
                    "background_safe_to_demote": configured_background_requests > 0,
                    "tool_start_offset_ms": round(tool_start_ms, 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "offset_ms": round(offset_ms(), 3),
                },
            )
            write_trace(
                args.trace,
                {
                    "event": "m27.controller_traffic_reshape.window_open",
                    "controller_policy": args.mode,
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "pressure_level": args.pressure_level,
                    "task_index": pair.task_index,
                    "tool_wait_step": wait_spec.step_index,
                    "task_replay_steps": len(target_wait_specs),
                    "tool_wait_profile": args.tool_wait_profile,
                    "tool_wait_class": wait_spec.wait_class,
                    "tool_wait_ms": wait_spec.wait_ms,
                    "demote_trigger": trigger,
                    "controller_admission_aggressiveness": admission_aggressiveness
                    if controller_active_priority_demotion_admission
                    else "",
                    "controller_earlyprepare_lead_ms": controller_admission_lead_ms(args.mode, wait_spec.wait_ms)
                    if admission_aggressiveness == "earlyprepare"
                    else "",
                    "demotable_background_requests": configured_background_requests,
                    "background_safe_to_demote": configured_background_requests > 0,
                    "controller_decision_id": controller_demote_command.get("controller_decision_id", ""),
                    "controller_command_id": controller_demote_command.get("command_id", ""),
                    "controller_decision_reason": controller_demote_command.get("controller_decision_reason", ""),
                    "effective_background_priority": -100,
                    "tool_start_offset_ms": round(tool_start_ms, 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "open_offset_ms": round(offset_ms(), 3),
                },
            )

        for wait_spec in target_wait_specs:
            wait_ms = wait_spec.wait_ms
            replay_label = replay_label_for(pair.session_id, wait_spec.step_index, len(target_wait_specs))
            warmup_label = warmup_label_for(pair.session_id, wait_spec.step_index, len(target_wait_specs))
            step_base_meta = {
                **base_meta,
                "session_generation": wait_spec.step_index + 1,
                "tool_wait_ms": wait_ms,
                "tool_wait_step": wait_spec.step_index,
                "tool_wait_class": wait_spec.wait_class,
            }
            tool_start_ms = offset_ms()
            replay_due_ms = tool_start_ms + wait_ms
            predictive_replay_fields: dict[str, Any] = {}
            register_memory_admission_replay(
                {
                    **step_base_meta,
                    "session_id": pair.session_id,
                    "prefix_id": base_meta.get("prefix_id", f"{pair.session_id}:prefix"),
                    "phase": "tool_wait",
                    "label": f"{pair.session_id}_tool_wait_{wait_spec.step_index:02d}",
                    "reuse_probability": 0.98,
                    "recompute_cost_tokens": estimate_tokens(target_current_prompt),
                },
                replay_due_ms=replay_due_ms,
                prompt_tokens=estimate_tokens(target_current_prompt),
                request_group="target",
                wait_spec=wait_spec,
                replay_label=replay_label,
            )
            if controller_active_predictive_deadline_queue:
                predictive_replay_fields = assign_predictive_deadline_priority(
                    {
                        **step_base_meta,
                        "session_id": pair.session_id,
                        "phase": "tool_wait",
                        "label": replay_label,
                        "task_index": pair.task_index,
                        "task_replay_steps": len(target_wait_specs),
                    },
                    replay_due_ms,
                    "target",
                    "tool_wait",
                )
            controller_tool_start_result = record_controller_event(
                f"{pair.session_id}:tool_started:{wait_spec.step_index}",
                EventType.TOOL_STARTED,
                {
                    **step_base_meta,
                    "session_id": pair.session_id,
                    "phase": "tool_wait",
                    "label": f"{pair.session_id}_tool_wait_{wait_spec.step_index:02d}",
                },
                monotonic_ms=int(tool_start_ms),
                expected_completion_ms=int(replay_due_ms),
                deadline_after_completion_ms=wait_ms,
                eta_uncertainty_ms=max(1, wait_ms // 4),
            )
            write_trace(
                args.trace,
                {
                    "event": "m27.tool_wait.start",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "task_index": pair.task_index,
                    "tool_wait_step": wait_spec.step_index,
                    "task_replay_steps": len(target_wait_specs),
                    "tool_wait_profile": args.tool_wait_profile,
                    "tool_wait_class": wait_spec.wait_class,
                    "tool_start_offset_ms": round(tool_start_ms, 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "tool_wait_ms": wait_ms,
                    "prompt_hash": prompt_hash(pair.prompt),
                    "expected_replay_request_id": replay_label,
                },
            )
            initial_gateway_row = (
                trace_event_row(args.trace, str(initial_meta["label"]), "m27.request.start")
                if args.mode == HARNESS_EMITTED_SIGNAL_MODE
                else {}
            )
            cache_signal_driven_preload = (
                args.mode == HARNESS_EMITTED_SIGNAL_MODE
                and str(initial_gateway_row.get("harness_native_cache_signal_seen") or "").lower() == "yes"
            )
            direct_gateway_speculative_prefill = args.mode == "e2e_priority_hints_speculative_prefill"
            controller_preload_command = acted_controller_command(controller_tool_start_result, kv_action="prefetch")
            targeted_prefetch_command = (
                selected_controller_command(controller_tool_start_result, kv_action="prefetch")
                if controller_active_targeted_prefetch
                else None
            )
            controller_demote_command = acted_controller_command(controller_tool_start_result, kv_action="demote")
            step_demote_restore_active = (
                (controller_active_demote_restore or controller_active_full)
                and controller_demote_command is not None
            )
            defer_admission_demote = (
                controller_active_priority_demotion_admission
                and controller_demote_command is not None
                and admission_aggressiveness in {"soft", "medium", "earlyprepare"}
            )
            if step_demote_restore_active and not defer_admission_demote:
                activate_controller_demote(
                    controller_demote_command,
                    step_base_meta=step_base_meta,
                    wait_spec=wait_spec,
                    tool_start_ms=tool_start_ms,
                    replay_due_ms=replay_due_ms,
                    trigger="tool_started",
                )
            controller_speculative_preload = controller_active_preload and controller_preload_command is not None
            controller_admission_row: dict[str, Any] | None = None
            if controller_active_admission or controller_active_full:
                controller_admission_row = await controller_admission_decision(
                    pair,
                    controller_tool_start_result,
                    wait_spec=wait_spec,
                    tool_start_ms=tool_start_ms,
                    replay_due_ms=replay_due_ms,
                )
            controller_admission_preload = bool(
                controller_active_admission
                and controller_admission_row
                and controller_admission_row.get("admitted")
            )
            warmup_task: asyncio.Task[None] | None = None
            if targeted_prefetch_command is not None:
                direct_hook_available = str(targeted_prefetch_command.get("backend_acted")).lower() == "true"
                direct_load_prompt_tokens = estimate_tokens(target_current_prompt)
                prefetch_window = controller_direct_load_window(
                    tool_start_ms=tool_start_ms,
                    replay_due_ms=replay_due_ms,
                    prompt_tokens=direct_load_prompt_tokens,
                    wait_class=wait_spec.wait_class,
                )
                targeted_prefetch_lead_ms = int(max(0.0, replay_due_ms - float(prefetch_window["start_ms"])))
                targeted_prefetch_start_ms = float(prefetch_window["start_ms"])
                direct_kv_h2d_priority = deadline_fair_priority_for_due(replay_due_ms)
                direct_load_mechanism = controller_direct_load_mechanism()
                trigger_label = f"{pair.session_id}_controller_targeted_direct_load_{wait_spec.step_index:02d}"
                targeted_event_base = {
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "request_id": str(initial_meta["label"]),
                    "prefetch_request_id": trigger_label,
                    "expected_replay_request_id": replay_label,
                    "prefix_id": step_base_meta["prefix_id"],
                    "tool_wait_step": wait_spec.step_index,
                    "task_replay_steps": len(target_wait_specs),
                    "tool_wait_profile": args.tool_wait_profile,
                    "tool_wait_class": wait_spec.wait_class,
                    "tool_wait_ms": wait_ms,
                    "reuse_probability": step_base_meta.get("reuse_probability", ""),
                    "recompute_cost_tokens": step_base_meta.get("recompute_cost_tokens", ""),
                    "direct_load_prompt_tokens": direct_load_prompt_tokens,
                    "direct_load_estimated_ms": prefetch_window["estimated_ms"],
                    "direct_load_safety_margin_ms": prefetch_window["safety_margin_ms"],
                    "direct_load_available_slack_ms": round(float(prefetch_window["available_slack_ms"]), 3),
                    "direct_load_latest_finish_offset_ms": round(float(prefetch_window["latest_finish_ms"]), 3),
                    "direct_load_admission_reason": prefetch_window["reason"],
                    "direct_load_contract": prefetch_window["contract"],
                    "direct_load_allowed_wait_classes": prefetch_window["allowed_wait_classes"],
                    "direct_load_mechanism": direct_load_mechanism,
                    "direct_load_execution_path": "sglang_prepare_prefix_control",
                    "direct_kv_h2d_priority": direct_kv_h2d_priority,
                    "direct_kv_h2d_priority_source": "harness_tool_wait_eta_replay_due",
                    "controller_decision_id": targeted_prefetch_command.get("controller_decision_id", ""),
                    "controller_command_id": targeted_prefetch_command.get("command_id", ""),
                    "controller_command_reason": targeted_prefetch_command.get("reason", ""),
                    "backend_name": targeted_prefetch_command.get("backend_name", ""),
                    "backend_accepted": targeted_prefetch_command.get("backend_accepted", ""),
                    "backend_acted": targeted_prefetch_command.get("backend_acted", ""),
                    "backend_reason": targeted_prefetch_command.get("backend_reason", ""),
                    "direct_hook_available": direct_hook_available,
                    "tool_start_offset_ms": round(tool_start_ms, 3),
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "targeted_prefetch_lead_ms": targeted_prefetch_lead_ms,
                    "targeted_prefetch_start_offset_ms": round(targeted_prefetch_start_ms, 3),
                    "offset_ms": round(offset_ms(), 3),
                }
                write_trace(args.trace, {**targeted_event_base, "event": "m27.targeted_kv_prefetch.requested"})
                write_trace(
                    args.trace,
                    {
                        **targeted_event_base,
                        "event": "m27.targeted_kv_prefetch.acted"
                        if direct_hook_available
                        else "m27.targeted_kv_prefetch.unavailable",
                    },
                )
                if not prefetch_window["admitted"]:
                    write_trace(
                        args.trace,
                        {
                            **targeted_event_base,
                            "event": "m27.targeted_kv_prefetch.skipped",
                            "status": "skipped",
                            "offset_ms": round(offset_ms(), 3),
                        },
                    )
                elif direct_hook_available:
                    direct_load_meta = {
                        **step_base_meta,
                        "session_id": pair.session_id,
                        "phase": "hint_prefetch",
                        "label": trigger_label,
                        "task_index": pair.task_index,
                        "prompt_hash": prompt_hash(target_current_prompt),
                        "priority_label": "background",
                        "deadline_offset_ms": round(replay_due_ms, 3),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "expected_tool_return_ms": int(wait_ms),
                        "next_ready_eta_ms": int(wait_ms),
                        "direct_kv_h2d_priority": direct_kv_h2d_priority,
                        "direct_kv_h2d_priority_source": "harness_tool_wait_eta_replay_due",
                        "max_tokens": 1,
                        "controller_targeted_kv_prefetch": True,
                        "prefetch_action": direct_load_mechanism,
                        "parent_request_id": str(initial_meta["label"]),
                        "expected_replay_request_id": replay_label,
                        "controller_decision_id": targeted_prefetch_command.get("controller_decision_id", ""),
                        "controller_command_id": targeted_prefetch_command.get("command_id", ""),
                        "controller_kv_translation": f"controller.targeted_prefetch={direct_load_mechanism}",
                        "prepared_kv_contract": prefetch_window["contract"],
                        "prepared_kv_expected_replay_id": replay_label,
                    }
                    direct_load_meta = attach_harness_priority_metadata(direct_load_meta)

                    async def run_targeted_direct_load() -> None:
                        await sleep_until(targeted_prefetch_start_ms)
                        timeout_s = max(0.0, (float(prefetch_window["latest_finish_ms"]) - offset_ms()) / 1000.0)
                        if timeout_s <= 0:
                            write_trace(
                                args.trace,
                                {
                                    **targeted_event_base,
                                    "event": "m27.targeted_kv_prefetch.direct_load_skipped_expired",
                                    "status": "skipped",
                                    "offset_ms": round(offset_ms(), 3),
                                },
                            )
                            return
                        plan_result = await prepare_prefix_kv_control_request(
                            direct_load_meta,
                            prefetch_window,
                            plan_only=True,
                        )
                        selective_admission = direct_load_admission_from_plan(plan_result)
                        if not selective_admission["admitted"]:
                            write_trace(
                                args.trace,
                                {
                                    **targeted_event_base,
                                    "event": "m27.targeted_kv_prefetch.skipped",
                                    "status": "skipped",
                                    "prepared_kv_status": "not_needed",
                                    "selective_admission_reason": selective_admission["reason"],
                                    "selective_admission_status": selective_admission["plan_status"],
                                    "selective_admission_host_tokens": selective_admission["host_tokens"],
                                    "selective_admission_min_saved_tokens": selective_admission["min_saved_tokens"],
                                    "prepare_prefix_plan_result": plan_result,
                                    "offset_ms": round(offset_ms(), 3),
                                },
                            )
                            return
                        write_trace(
                            args.trace,
                            {
                                **targeted_event_base,
                                "event": "m27.targeted_kv_prefetch.direct_load_start",
                                "selective_admission_reason": selective_admission["reason"],
                                "selective_admission_status": selective_admission["plan_status"],
                                "selective_admission_host_tokens": selective_admission["host_tokens"],
                                "selective_admission_min_saved_tokens": selective_admission["min_saved_tokens"],
                                "offset_ms": round(offset_ms(), 3),
                            },
                        )
                        try:
                            result = await prepare_prefix_kv_control_request(direct_load_meta, prefetch_window)
                            if not result.get("ok"):
                                write_trace(
                                    args.trace,
                                    {
                                        **targeted_event_base,
                                        "event": "m27.targeted_kv_prefetch.direct_load_error",
                                        "status": str(result.get("status") or "error"),
                                        "prepared_kv_status": "not_ready",
                                        "prepare_prefix_result": result,
                                        "offset_ms": round(offset_ms(), 3),
                                    },
                                )
                                return
                            write_trace(
                                args.trace,
                                {
                                    **targeted_event_base,
                                    "event": "m27.targeted_kv_prefetch.direct_load_end",
                                    "status": "ok",
                                    "prepared_kv_status": "ready",
                                    "direct_load_mechanism": direct_load_mechanism,
                                    "offset_ms": round(offset_ms(), 3),
                                },
                            )
                        except asyncio.TimeoutError:
                            write_trace(
                                args.trace,
                                {
                                    **targeted_event_base,
                                    "event": "m27.targeted_kv_prefetch.direct_load_timeout_before_replay",
                                    "status": "timeout",
                                    "prepared_kv_status": "not_ready",
                                    "offset_ms": round(offset_ms(), 3),
                                },
                            )
                        except Exception as exc:  # noqa: BLE001
                            write_trace(
                                args.trace,
                                {
                                    **targeted_event_base,
                                    "event": "m27.targeted_kv_prefetch.direct_load_error",
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "offset_ms": round(offset_ms(), 3),
                                },
                            )

                    warmup_task = asyncio.create_task(run_targeted_direct_load())
            if (
                direct_gateway_speculative_prefill
                or cache_signal_driven_preload
                or controller_speculative_preload
                or controller_admission_preload
            ):
                step_warmup_prompt = warmup_prompt_for_step(pair, wait_spec.step_index, len(target_wait_specs))
                warmup_role = (
                    "controller_admission_gated_speculative_kv_preload"
                    if controller_admission_preload
                    else
                    "controller_gateway_speculative_kv_preload"
                    if controller_speculative_preload
                    else
                    "gateway_speculative_kv_preload"
                    if cache_signal_driven_preload
                    else "dynamo_like_background_warmup"
                )
                warmup_strategy = (
                    "controller_admission_gated_gateway_speculative_kv_preload"
                    if controller_admission_preload
                    else
                    "controller_lifecycle_gateway_speculative_kv_preload"
                    if controller_speculative_preload
                    else
                    "harness_cache_signal_gateway_speculative_kv_preload"
                    if cache_signal_driven_preload
                    else "known_next_turn_prefix"
                )
                warmup_trigger = (
                    "controller_admission_decision"
                    if controller_admission_preload
                    else
                    "controller_prefetch_decision"
                    if controller_speculative_preload
                    else "harness_cache_signal"
                    if cache_signal_driven_preload
                    else "gateway_injected_speculative_prefill"
                )
                warmup_meta = {
                    **step_base_meta,
                    "session_id": pair.session_id,
                    "phase": "speculative_prefill",
                    "label": warmup_label,
                    "task_index": pair.task_index,
                    "prompt_hash": prompt_hash(step_warmup_prompt),
                    "priority_label": "background",
                    "deadline_offset_ms": round(replay_due_ms, 3),
                    "max_tokens": 1,
                    "speculative_prefill": True,
                    "speculative_prefill_role": warmup_role,
                    "speculative_prefill_strategy": warmup_strategy,
                    "parent_request_id": str(initial_meta["label"]),
                    "expected_replay_request_id": replay_label,
                    "warmup_prompt_tokens": estimate_tokens(step_warmup_prompt),
                    "harness_cache_signal_source": initial_gateway_row.get("harness_native_cache_signal_source", ""),
                    "harness_cache_signal": initial_gateway_row.get("harness_native_cache_signal", ""),
                    "controller_decision_id": (
                        controller_admission_row.get("prefetch_decision_id", "")
                        if controller_admission_preload and controller_admission_row
                        else controller_preload_command.get("controller_decision_id", "")
                        if controller_preload_command
                        else ""
                    ),
                    "controller_command_id": (
                        controller_admission_row.get("prefetch_command_id", "")
                        if controller_admission_preload and controller_admission_row
                        else controller_preload_command.get("command_id", "")
                        if controller_preload_command
                        else ""
                    ),
                    "controller_kv_translation": (
                        "controller.admission=admitted_gateway_speculative_kv_preload"
                        if controller_admission_preload
                        else "controller.prefetch=gateway_speculative_kv_preload"
                        if controller_preload_command
                        else ""
                    ),
                    "controller_admission_decision": "admit" if controller_admission_preload else "",
                    "controller_admission_reason": controller_admission_row.get("reason", "")
                    if controller_admission_row
                    else "",
                }
                warmup_meta = attach_harness_priority_metadata(warmup_meta)
                if args.mode == STORAGE_HICACHE_CONTROLLER_PREFETCH_MODE:
                    write_trace(
                        args.trace,
                        {
                            "event": "m27.storage.controller_prefetch.intent",
                            "session_id": pair.session_id,
                            "mode": args.mode,
                            "harness": args.harness,
                            "request_id": initial_meta["label"],
                            "warmup_request_id": warmup_label,
                            "expected_replay_request_id": replay_label,
                            **storage_meta,
                            "strategy": "controller_lifecycle_gateway_speculative_kv_preload_with_real_hicache_storage",
                            "tool_wait_step": wait_spec.step_index,
                            "tool_wait_ms": wait_ms,
                            "tool_start_offset_ms": round(tool_start_ms, 3),
                            "replay_due_offset_ms": round(replay_due_ms, 3),
                        },
                    )
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
                        "tool_wait_step": wait_spec.step_index,
                        "task_replay_steps": len(target_wait_specs),
                        "tool_wait_profile": args.tool_wait_profile,
                        "tool_wait_class": wait_spec.wait_class,
                        "tool_wait_ms": wait_ms,
                        "harness_cache_signal_source": initial_gateway_row.get("harness_native_cache_signal_source", ""),
                        "warmup_prompt_hash": warmup_meta["prompt_hash"],
                        "warmup_prompt_tokens": warmup_meta["warmup_prompt_tokens"],
                        "tool_start_offset_ms": round(tool_start_ms, 3),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "controller_decision_id": warmup_meta.get("controller_decision_id", ""),
                        "controller_command_id": warmup_meta.get("controller_command_id", ""),
                        "controller_kv_translation": warmup_meta.get("controller_kv_translation", ""),
                        "controller_admission_decision": warmup_meta.get("controller_admission_decision", ""),
                        "controller_admission_reason": warmup_meta.get("controller_admission_reason", ""),
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
                            "tool_wait_step": wait_spec.step_index,
                            "offset_ms": round(offset_ms(), 3),
                        },
                    )
                    try:
                        await run_gateway_background_warmup(args.gateway_base, args.model, step_warmup_prompt, warmup_meta)
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
                                "tool_wait_step": wait_spec.step_index,
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
                                "tool_wait_step": wait_spec.step_index,
                                "offset_ms": round(offset_ms(), 3),
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )

                warmup_task = asyncio.create_task(run_warmup())
            if warmup_task is not None:
                warmup_launch_grace_ms = max(0.0, float(os.environ.get("WARMUP_LAUNCH_GRACE_MS", "5")))
                await asyncio.sleep(warmup_launch_grace_ms / 1000.0)
            if not filler_tasks:
                if args.filler_backlog_mode == "constant":
                    filler_backlog_producer_task = asyncio.create_task(run_constant_filler_backlog())
                else:
                    for idx in range(args.filler_sessions):
                        launch_filler_task(idx, reason="one_shot_pressure_fill")
            if defer_admission_demote:
                admission_lead_ms = controller_admission_lead_ms(args.mode, wait_ms)
                admission_open_ms = max(tool_start_ms, replay_due_ms - admission_lead_ms)
                if admission_open_ms > offset_ms() + 1:
                    await sleep_until(admission_open_ms)
                activate_controller_demote(
                    controller_demote_command,
                    step_base_meta=step_base_meta,
                    wait_spec=wait_spec,
                    tool_start_ms=tool_start_ms,
                    replay_due_ms=replay_due_ms,
                    trigger=f"admission_{admission_aggressiveness}_window",
                )
                step_demote_restore_active = True
            prepare_lead_ms = controller_prepare_lead_ms(wait_ms) if controller_active_full else 0
            prepare_checkpoint_ms = max(tool_start_ms, replay_due_ms - prepare_lead_ms)
            prepare_controller_result: tuple[dict[str, Any], list[dict[str, Any]]] | None = None
            if (
                prepare_lead_ms > 0
                and prepare_checkpoint_ms > offset_ms() + 1
                and prepare_checkpoint_ms < replay_due_ms
            ):
                await sleep_until(prepare_checkpoint_ms)
                prepare_controller_result = record_controller_event(
                    f"{pair.session_id}:prepare_window:{wait_spec.step_index}",
                    EventType.TOOL_ETA_UPDATED,
                    {
                        **step_base_meta,
                        "session_id": pair.session_id,
                        "phase": "prepare",
                        "label": f"{pair.session_id}_prepare_{wait_spec.step_index:02d}",
                    },
                    monotonic_ms=int(prepare_checkpoint_ms),
                    expected_completion_ms=int(replay_due_ms),
                    deadline_after_completion_ms=wait_ms,
                    eta_uncertainty_ms=max(0, int(replay_due_ms - prepare_checkpoint_ms)),
                )
                prepare_demote_command = acted_controller_command(prepare_controller_result, kv_action="demote")
                if (
                    (controller_active_demote_restore or controller_active_full)
                    and prepare_demote_command is not None
                    and not step_demote_restore_active
                ):
                    activate_controller_demote(
                        prepare_demote_command,
                        step_base_meta=step_base_meta,
                        wait_spec=wait_spec,
                        tool_start_ms=tool_start_ms,
                        replay_due_ms=replay_due_ms,
                        trigger="prepare_window",
                    )
                    step_demote_restore_active = True
                write_trace(
                    args.trace,
                    {
                        "event": "m27.controller_full.prepare_window",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "pressure_level": args.pressure_level,
                        "task_index": pair.task_index,
                        "tool_wait_step": wait_spec.step_index,
                        "task_replay_steps": len(target_wait_specs),
                        "tool_wait_profile": args.tool_wait_profile,
                        "tool_wait_class": wait_spec.wait_class,
                        "tool_wait_ms": wait_ms,
                        "prepare_lead_ms": prepare_lead_ms,
                        "prepare_checkpoint_offset_ms": round(prepare_checkpoint_ms, 3),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "controller_decision_id": (
                            prepare_controller_result[0].get("decision_id", "")
                            if prepare_controller_result
                            else ""
                        ),
                        "controller_demote_activated": prepare_demote_command is not None,
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
            await sleep_until(replay_due_ms)
            due_prepare_result = record_controller_event(
                f"{pair.session_id}:prepare_checkpoint:{wait_spec.step_index}",
                EventType.TOOL_ETA_UPDATED,
                {
                    **step_base_meta,
                    "session_id": pair.session_id,
                    "phase": "prepare",
                    "label": f"{pair.session_id}_prepare_{wait_spec.step_index:02d}",
                },
                monotonic_ms=int(replay_due_ms),
                expected_completion_ms=int(replay_due_ms),
                deadline_after_completion_ms=wait_ms,
                eta_uncertainty_ms=0,
            )
            due_demote_command = acted_controller_command(due_prepare_result, kv_action="demote")
            if (
                (controller_active_demote_restore or controller_active_full)
                and due_demote_command is not None
                and not step_demote_restore_active
            ):
                activate_controller_demote(
                    due_demote_command,
                    step_base_meta=step_base_meta,
                    wait_spec=wait_spec,
                    tool_start_ms=tool_start_ms,
                    replay_due_ms=replay_due_ms,
                    trigger="replay_due_checkpoint",
                )
                step_demote_restore_active = True
            write_trace(
                args.trace,
                {
                    "event": "m27.pre_replay.checkpoint",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "tool_wait_step": wait_spec.step_index,
                    "task_replay_steps": len(target_wait_specs),
                    "tool_wait_profile": args.tool_wait_profile,
                    "tool_wait_class": wait_spec.wait_class,
                    "tool_wait_ms": wait_ms,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "expected_reuse": (
                        "controller_targeted_kv_prefetch"
                        if targeted_prefetch_command
                        else
                        "controller_gateway_speculative_kv_preload"
                        if controller_speculative_preload
                        else
                        "controller_admission_gated_speculative_kv_preload"
                        if controller_admission_preload
                        else
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
                    "label": replay_label,
                    "request_id": replay_label,
                    "tool_wait_step": wait_spec.step_index,
                    "task_replay_steps": len(target_wait_specs),
                    "tool_wait_profile": args.tool_wait_profile,
                    "tool_wait_class": wait_spec.wait_class,
                    "tool_wait_ms": wait_ms,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                    "expected_replay_request_id": replay_label,
                },
            )
            controller_ready_result = record_controller_event(
                f"{pair.session_id}:tool_completed:{wait_spec.step_index}",
                EventType.TOOL_COMPLETED,
                {**step_base_meta, "session_id": pair.session_id, "phase": "replay", "label": replay_label},
                monotonic_ms=int(replay_due_ms),
                expected_completion_ms=int(replay_due_ms),
                deadline_after_completion_ms=wait_ms,
                eta_uncertainty_ms=0,
            )
            controller_replay_priority: int | None = None
            controller_decision_id = ""
            controller_command_id = ""
            if controller_ready_result is not None:
                decision_row, backend_results = controller_ready_result
                controller_decision_id = str(decision_row.get("decision_id") or "")
                acted_command_ids = {
                    str(result.get("command_id"))
                    for result in backend_results
                    if result.get("acted")
                }
                for command in decision_row.get("commands") or []:
                    if not isinstance(command, dict):
                        continue
                    if command.get("scheduler_action") != "set_priority":
                        continue
                    if str(command.get("command_id") or "") not in acted_command_ids:
                        continue
                    try:
                        controller_replay_priority = int(command.get("priority"))
                        controller_command_id = str(command.get("command_id") or "")
                        break
                    except (TypeError, ValueError):
                        continue
            controller_replay_rank = ""
            controller_replay_count = ""
            controller_ladder_priority = ""
            if controller_active_full and controller_replay_priority is not None:
                ladder_priority, replay_rank, replay_count = full_controller_priority_for_pair(index)
                controller_replay_priority = max(controller_replay_priority, ladder_priority)
                controller_replay_rank = replay_rank
                controller_replay_count = replay_count
                controller_ladder_priority = controller_replay_priority
                write_trace(
                    args.trace,
                    {
                        "event": "m27.controller_full.priority_ladder",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "pressure_level": args.pressure_level,
                        "task_index": pair.task_index,
                        "tool_wait_step": wait_spec.step_index,
                        "task_replay_steps": len(target_wait_specs),
                        "tool_wait_profile": args.tool_wait_profile,
                        "tool_wait_class": wait_spec.wait_class,
                        "tool_wait_ms": wait_ms,
                        "controller_decision_id": controller_decision_id,
                        "controller_command_id": controller_command_id,
                        "replay_rank_by_due_time": replay_rank,
                        "urgent_replay_count": replay_count,
                        "assigned_sglang_priority": controller_replay_priority,
                        "ladder_step": 10 if replay_count > 1 else 0,
                        "reason": "deadline-aware priority ladder for tied urgent replay work",
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
            deadline_fair_replay_fields: dict[str, Any] = {}
            if controller_active_deadline_fair:
                deadline_fair_replay_fields = assign_deadline_fair_priority(
                    {
                        **step_base_meta,
                        "session_id": pair.session_id,
                        "phase": "replay",
                        "label": replay_label,
                        "task_index": pair.task_index,
                    },
                    replay_due_ms,
                    "target",
                    "replay",
                )
                controller_replay_priority = int(deadline_fair_replay_fields["controller_sglang_priority"])
                controller_replay_rank = "deadline_due_time"
                controller_replay_count = "all_due_replays"
                controller_ladder_priority = controller_replay_priority
            if predictive_replay_fields:
                controller_replay_priority = int(predictive_replay_fields["controller_sglang_priority"])
                controller_replay_rank = "predictive_deadline_due_time"
                controller_replay_count = "all_due_replays"
                controller_ladder_priority = controller_replay_priority
            replay_max_tokens = 8
            replay_workload_meta: dict[str, Any] = {
                "agentic_workload_profile": args.agentic_workload_profile,
                "workload_request_kind": "synthetic_target_replay",
                "workload_phase_family": "replay",
                "workload_prompt_tokens_target": args.target_prompt_tokens,
                "workload_max_tokens": 8,
                "workload_description": "fixed synthetic target replay request",
            }
            if args.agentic_workload_profile == REALISTIC_AGENTIC_PROFILE and not args.workload_jsonl:
                replay_shape = realistic_replay_shape(
                    wait_class=wait_spec.wait_class,
                    seed=args.tool_wait_seed,
                    stream_key=workload_stream_key(pair, "target", "replay", str(wait_spec.step_index)),
                )
                step_replay_prompt = make_agentic_workload_prompt(
                    session_id=pair.session_id,
                    shape=replay_shape,
                    role="target",
                    stage="replay",
                    step_index=wait_spec.step_index,
                    total_steps=len(target_wait_specs),
                    previous_prompt=target_current_prompt,
                )
                replay_max_tokens = replay_shape.max_tokens
                replay_workload_meta = workload_meta(replay_shape, args.agentic_workload_profile)
            else:
                step_replay_prompt = replay_prompt_for_step(pair, wait_spec.step_index, len(target_wait_specs))
            replay_meta = {
                **step_base_meta,
                "session_id": pair.session_id,
                "phase": "replay",
                "label": replay_label,
                "task_index": pair.task_index,
                "prompt_hash": prompt_hash(step_replay_prompt),
                "priority_label": "deadline_fair"
                if controller_active_deadline_fair
                else "predictive_deadline_queue"
                if controller_active_predictive_deadline_queue
                else "high",
                "deadline_offset_ms": round(replay_due_ms, 3),
                "max_tokens": replay_max_tokens,
                "prompt_tokens": estimate_tokens(step_replay_prompt),
                **replay_workload_meta,
                **deadline_fair_replay_fields,
                **predictive_replay_fields,
            }
            replay_meta.update(value_aware_eviction_metadata(replay_meta))
            if controller_replay_priority is not None:
                controller_fields = {
                    "controller_sglang_priority": controller_replay_priority,
                    "controller_decision_id": controller_decision_id,
                    "controller_command_id": controller_command_id,
                    "controller_replay_rank": controller_replay_rank,
                    "controller_urgent_replay_count": controller_replay_count,
                    "controller_priority_ladder": controller_ladder_priority,
                }
                if not (controller_active_deadline_fair or controller_active_predictive_deadline_queue):
                    controller_fields[
                        "controller_priority_translation"
                    ] = f"controller.set_priority={controller_replay_priority}"
                replay_meta.update(controller_fields)
            replay_meta = attach_harness_priority_metadata(replay_meta)
            if controller_demotion_state.get("active"):
                write_trace(
                    args.trace,
                    {
                        "event": "m27.controller_traffic_reshape.target_replay_entering",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "pressure_level": args.pressure_level,
                        "request_id": replay_label,
                        "label": replay_label,
                        "task_index": pair.task_index,
                        "tool_wait_step": wait_spec.step_index,
                        "task_replay_steps": len(target_wait_specs),
                        "controller_sglang_priority": controller_replay_priority or "",
                        "controller_decision_id": controller_decision_id,
                        "controller_command_id": controller_command_id,
                        "controller_replay_rank": controller_replay_rank,
                        "controller_urgent_replay_count": controller_replay_count,
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
            await bounded_request(step_replay_prompt, replay_meta)
            target_current_prompt = step_replay_prompt
            if controller_active_priority_demotion_admission and controller_admission_gate_state.get("active"):
                controller_admission_gate_state["active"] = False
                controller_admission_gate_event.set()
                gate_meta = controller_admission_gate_state.get("meta")
                if not isinstance(gate_meta, dict):
                    gate_meta = {}
                write_trace(
                    args.trace,
                    {
                        "event": "m27.controller_admission_gate.window_close",
                        "controller_policy": args.mode,
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "pressure_level": args.pressure_level,
                        "task_index": pair.task_index,
                        "tool_wait_step": wait_spec.step_index,
                        "task_replay_steps": len(target_wait_specs),
                        "admission_action": "release_background_after_target_replay",
                        "gate_open_offset_ms": gate_meta.get("open_offset_ms", ""),
                        "replay_due_offset_ms": round(replay_due_ms, 3),
                        "close_offset_ms": round(offset_ms(), 3),
                    },
                )
                controller_admission_gate_state["meta"] = {}
            write_trace(
                args.trace,
                {
                    "event": "m27.tool_wait.end",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "label": replay_label,
                    "request_id": replay_label,
                    "tool_wait_step": wait_spec.step_index,
                    "task_replay_steps": len(target_wait_specs),
                    "tool_wait_profile": args.tool_wait_profile,
                    "tool_wait_class": wait_spec.wait_class,
                    "tool_wait_ms": wait_ms,
                    "replay_due_offset_ms": round(replay_due_ms, 3),
                },
            )
            if warmup_task is not None:
                await asyncio.gather(warmup_task, return_exceptions=True)
            rows.append(
                {
                    "harness": args.harness,
                    "mode": args.mode,
                    "session_id": pair.session_id,
                    "tool_wait_ms": wait_ms,
                    "tool_wait_step": wait_spec.step_index,
                    "task_replay_steps": len(target_wait_specs),
                    "tool_wait_profile": args.tool_wait_profile,
                    "tool_wait_class": wait_spec.wait_class,
                    "filler_sessions": args.filler_sessions,
                    "target_prompt_tokens": args.target_prompt_tokens,
                    "pressure_level": args.pressure_level,
                }
            )
            if step_demote_restore_active:
                owners = controller_demotion_state.get("owners")
                if isinstance(owners, set):
                    owners.discard(f"{pair.session_id}:{wait_spec.step_index}")
                    controller_demotion_state["active"] = bool(owners)
                else:
                    controller_demotion_state["active"] = False
                if not controller_demotion_state["active"]:
                    controller_demotion_state["meta"] = {}
                filler_base_meta = base_meta
                write_trace(
                    args.trace,
                    {
                        "event": "m27.controller_demote_restore.step_restored",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "pressure_level": args.pressure_level,
                        "task_index": pair.task_index,
                        "tool_wait_step": wait_spec.step_index,
                        "task_replay_steps": len(target_wait_specs),
                        "restored_phase": "pressure_filler",
                        "restored_priority": 0,
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
                write_trace(
                    args.trace,
                    {
                        "event": "m27.controller_traffic_reshape.window_close",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "pressure_level": args.pressure_level,
                        "task_index": pair.task_index,
                        "tool_wait_step": wait_spec.step_index,
                        "task_replay_steps": len(target_wait_specs),
                        "restored_phase": "pressure_filler",
                        "restored_priority": 0,
                        "remaining_reshape_windows": len(owners) if isinstance(owners, set) else "",
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
        controller_finish_result = record_controller_event(
            f"{pair.session_id}:session_finished",
            EventType.SESSION_FINISHED,
            {
                **base_meta,
                "session_generation": len(target_wait_specs) + 1,
                "session_id": pair.session_id,
                "phase": "finished",
                "label": f"{pair.session_id}_finished",
            },
            monotonic_ms=int(offset_ms()),
        )
        controller_restore_command = acted_controller_command(controller_finish_result, kv_action="release")
        if demote_restore_active:
            owners = controller_demotion_state.get("owners")
            if isinstance(owners, set):
                for owner in list(owners):
                    if str(owner).startswith(f"{pair.session_id}:"):
                        owners.discard(owner)
                controller_demotion_state["active"] = bool(owners)
            else:
                controller_demotion_state["active"] = False
            if not controller_demotion_state["active"]:
                controller_demotion_state["meta"] = {}
            write_trace(
                args.trace,
                {
                    "event": "m27.controller_demote_restore.restored",
                    "session_id": pair.session_id,
                    "mode": args.mode,
                    "harness": args.harness,
                    "pressure_level": args.pressure_level,
                    "task_index": pair.task_index,
                    "controller_decision_id": (
                        controller_restore_command.get("controller_decision_id", "") if controller_restore_command else ""
                    ),
                    "controller_command_id": (
                        controller_restore_command.get("command_id", "")
                        if controller_restore_command
                        else active_demote_command.get("command_id", "")
                        if active_demote_command
                        else ""
                    ),
                    "controller_command_reason": controller_restore_command.get("reason", "") if controller_restore_command else "",
                    "backend_name": controller_restore_command.get("backend_name", "") if controller_restore_command else "",
                    "backend_accepted": controller_restore_command.get("backend_accepted", "") if controller_restore_command else "",
                    "backend_acted": controller_restore_command.get("backend_acted", "") if controller_restore_command else "",
                    "backend_reason": controller_restore_command.get("backend_reason", "") if controller_restore_command else "",
                    "restored_phase": "pressure_filler",
                    "restored_priority": 0,
                    "offset_ms": round(offset_ms(), 3),
                },
            )
        if filler_backlog_producer_task is not None:
            filler_backlog_stop.set()
            await asyncio.gather(filler_backlog_producer_task, return_exceptions=True)
        filler_results = await asyncio.gather(*filler_tasks, return_exceptions=True)
        filler_errors = [
            (task, result)
            for task, result in zip(filler_tasks, filler_results, strict=False)
            if isinstance(result, Exception)
        ]
        if filler_errors:
            for task, error in filler_errors:
                write_trace(
                    args.trace,
                    {
                        "event": "m27.filler_task.error",
                        "session_id": pair.session_id,
                        "mode": args.mode,
                        "harness": args.harness,
                        "pressure_level": args.pressure_level,
                        "task_index": pair.task_index,
                        "filler_task_name": task.get_name(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "offset_ms": round(offset_ms(), 3),
                    },
                )
            raise RuntimeError(f"{len(filler_errors)} filler task(s) failed; first error: {filler_errors[0][1]}")

    await asyncio.gather(*(run_pair(pair, idx) for idx, pair in enumerate(pairs)))
    write_trace(args.trace, {"event": "m27.workload_end", "harness": args.harness, "mode": args.mode, "row_count": len(rows)})
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
