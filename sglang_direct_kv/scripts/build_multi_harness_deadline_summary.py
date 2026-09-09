#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HARNESS_LABELS = {
    "hatcher": "DeepAgents",
    "codex": "Codex",
    "claude_code": "Claude Code",
    "opencode": "OpenCode",
    "qwen_code": "Qwen Code",
    "nemo_agent_toolkit": "NeMo Agent Toolkit / NAT",
    "nemo_agent_toolkit_service": "NeMo Agent Toolkit / NAT Service",
    "deepseek_harness": "DeepSeek Harness",
    "pi_agent_harness": "Pi Agent Harness",
    "openclaw": "OpenClaw",
    "hermes_agent": "Hermes Agent",
}


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))

HARNESS_SHORT_LABELS = {
    "hatcher": "DeepAgents",
    "codex": "Codex",
    "claude_code": "Claude",
    "opencode": "OpenCode",
    "qwen_code": "Qwen",
    "nemo_agent_toolkit": "NAT",
    "nemo_agent_toolkit_service": "NAT Service",
    "deepseek_harness": "DeepSeek",
    "pi_agent_harness": "Pi",
    "openclaw": "OpenClaw",
    "hermes_agent": "Hermes",
}

MODE_LABELS = {
    "no_prefetch": "NP = No prefetch",
    "e2e_priority_hints": "E2E = End-to-end priority hints",
    "pre_harness_priority_hints": "PH = Pre-harness priority hints",
    "nat_inferred_priority_hints": "NI = NAT inferred priority hints",
    "e2e_priority_hints_speculative_prefill": "SP = E2E priority + speculative prefill/preload",
    "no_cache_signal": "NC = No native cache signal",
    "harness_native_cache_lowered": "HC = Harness native cache lowered",
    "harness_emitted_signals": "HE = Harness emitted signals",
    "controller_observe_only": "CO = Controller observe-only",
    "controller_scheduler_priority": "CP = Controller scheduler priority",
    "controller_speculative_preload": "CL = Controller speculative KV preload",
    "controller_targeted_kv_prefetch": "CT = Controller targeted KV prefetch",
    "controller_demote_restore": "CD = Controller demote/restore",
    "controller_admission_control": "CA = Controller admission control",
    "controller_full": "CF = Full controller",
}

MODE_COLORS = {
    "no_prefetch": "#2563eb",
    "e2e_priority_hints": "#0f766e",
    "pre_harness_priority_hints": "#7c3aed",
    "nat_inferred_priority_hints": "#0891b2",
    "e2e_priority_hints_speculative_prefill": "#ea580c",
    "no_cache_signal": "#64748b",
    "harness_native_cache_lowered": "#f97316",
    "harness_emitted_signals": "#16a34a",
    "controller_observe_only": "#0f172a",
    "controller_scheduler_priority": "#be123c",
    "controller_speculative_preload": "#9333ea",
    "controller_targeted_kv_prefetch": "#f59e0b",
    "controller_demote_restore": "#0d9488",
    "controller_admission_control": "#2563eb",
    "controller_full": "#581c87",
}

MODE_ORDER = tuple(MODE_LABELS)

CHART_SIGNAL_BUCKETS = {
    "baseline": {
        "label": "Baseline",
        "description": "No signal was actually supplied or lowered for this replay",
        "color": "#475569",
        "modes": {"no_prefetch", "no_cache_signal"},
    },
    "harness_cache_emitted": {
        "label": "Harness Cache Emitted",
        "description": "Harness emitted cache/prompt-cache for this replay; gateway lowered it to speculative KV preload",
        "color": "#f97316",
        "modes": {"harness_native_cache_lowered"},
    },
    "harness_emitted": {
        "label": "Harness Emitted",
        "description": "Harness emitted cache and/or priority signal; gateway lowered whatever the harness produced",
        "color": "#16a34a",
        "modes": {"harness_emitted_signals"},
    },
    "harness_priority_emitted": {
        "label": "Harness Priority Emitted",
        "description": "Harness emitted priority/latency for this replay; gateway lowered it to SGLang priority",
        "color": "#0891b2",
        "modes": {"nat_inferred_priority_hints"},
    },
    "harness_cache_priority_emitted": {
        "label": "Harness Cache + Priority Emitted",
        "description": "Harness emitted both cache/preload and priority signals for this replay",
        "color": "#16a34a",
        "modes": set(),
    },
    "frontend_supplied": {
        "label": "Front-End Supplied",
        "description": "Front end supplied signal intent before the harness; gateway lowered what came through",
        "color": "#7c3aed",
        "modes": {"pre_harness_priority_hints"},
    },
    "gateway_priority_injected": {
        "label": "Gateway Priority Injected",
        "description": "Gateway added SGLang priority after the harness, before SGLang received the replay",
        "color": "#2563eb",
        "modes": {"e2e_priority_hints"},
    },
    "gateway_speculative_prefill": {
        "label": "Gateway Priority + Speculative Prefill",
        "description": "Gateway added priority and launched the Dynamo-like background prefill/preload warmup",
        "color": "#dc2626",
        "modes": {"e2e_priority_hints_speculative_prefill"},
    },
    "controller_observe": {
        "label": "Controller Observe-Only",
        "description": "Portable controller observed lifecycle state and recorded planned actions without mutating SGLang",
        "color": "#0f172a",
        "modes": {"controller_observe_only"},
    },
    "controller_scheduler": {
        "label": "Controller Scheduler Priority",
        "description": "Portable controller observed replay readiness and lowered its ready decision to SGLang priority",
        "color": "#be123c",
        "modes": {"controller_scheduler_priority"},
    },
    "controller_preload": {
        "label": "Controller Speculative KV Preload",
        "description": "Portable controller observed the tool-wait window and launched gateway speculative KV preload before replay",
        "color": "#9333ea",
        "modes": {"controller_speculative_preload"},
    },
    "controller_targeted_prefetch": {
        "label": "Controller Targeted KV Prefetch",
        "description": "Portable controller requested direct targeted host-to-device KV prefetch when the SGLang adapter exposes a stable hook",
        "color": "#f59e0b",
        "modes": {"controller_targeted_kv_prefetch"},
    },
    "controller_demote_restore": {
        "label": "Controller Demote/Restore",
        "description": "Portable controller lowers background/filler traffic during the replay-critical window, raises replay, then restores normal background behavior",
        "color": "#0d9488",
        "modes": {"controller_demote_restore"},
    },
    "controller_admission": {
        "label": "Controller Admission Control",
        "description": "Portable controller admits or skips speculative KV warmup based on pressure limits, with explicit skip reasons",
        "color": "#2563eb",
        "modes": {"controller_admission_control"},
    },
    "controller_full": {
        "label": "Full Controller",
        "description": "Portable controller combines filler demotion, replay priority, admission/budget policy, and P5 deadline-aware priority ranking",
        "color": "#581c87",
        "modes": {"controller_full"},
    },
}

CHART_SIGNAL_ORDER = (
    "baseline",
    "harness_cache_emitted",
    "harness_priority_emitted",
    "harness_cache_priority_emitted",
    "harness_emitted",
    "frontend_supplied",
    "gateway_priority_injected",
    "gateway_speculative_prefill",
    "controller_observe",
    "controller_scheduler",
    "controller_preload",
    "controller_targeted_prefetch",
    "controller_demote_restore",
    "controller_admission",
    "controller_full",
)

MANAGER_SIGNAL_BUCKETS = (
    "baseline",
    "harness_cache_emitted",
    "harness_cache_priority_emitted",
    "frontend_supplied",
    "controller_full",
)

COST_ACCOUNTING_SIGNAL_BUCKETS = (
    "baseline",
    "frontend_supplied",
)

COST_ACCOUNTING_COLORS = {
    "baseline": {
        "target": "#475569",
        "filler": "#cbd5e1",
    },
    "frontend_supplied": {
        "target": "#7c3aed",
        "filler": "#c4b5fd",
    },
}
COST_ACCOUNTING_DELTA_BETTER = "#16a34a"
COST_ACCOUNTING_DELTA_WORSE = "#dc2626"
COST_ACCOUNTING_DELTA_UNKNOWN = "#94a3b8"

CORE_PRESSURE_LEVELS = (
    "p0_control",
    "p3_high",
    "p5_boss_queue",
)

HARNESS_SYMBOLS = {
    "hatcher": "circle",
    "codex": "square",
    "claude_code": "triangle",
    "opencode": "diamond",
    "qwen_code": "cross",
    "nemo_agent_toolkit": "plus",
    "deepseek_harness": "star",
    "pi_agent_harness": "hexagon",
    "openclaw": "triangle-down",
    "hermes_agent": "ring",
}

PRESSURE_LABELS = {
    "p0_control": "P0 Control",
    "p1_mild": "P1 Short Wait",
    "p2_medium": "P2 Large KV",
    "p3_high": "P3 Queue Pressure",
    "p4_cliff": "P4 KV Pool Pressure",
    "p5_boss_queue": "P5 Boss Queue",
}

PRESSURE_DEFINITIONS = {
    "p0_control": {
        "goal": "Easy baseline. Confirms the replay path works when the system is not under pressure.",
        "knobs": "500 ms tool wait, 1024-token target prompt, no fillers, 1 urgent agent.",
    },
    "p1_mild": {
        "goal": "Short-wait pressure. Tests whether replay can resume after a small tool pause.",
        "knobs": "50 ms tool wait, 1024-token target prompt, no fillers, 1 urgent agent.",
    },
    "p2_medium": {
        "goal": "Large-KV pressure. Tests whether a larger target context makes replay readiness harder.",
        "knobs": "500 ms tool wait, 4096-token target prompt, no fillers, 1 urgent agent.",
    },
    "p3_high": {
        "goal": "Queue pressure. One urgent replay returns while older backend work is already queued.",
        "knobs": "50 ms tool wait, 4096-token target prompt, 32 filler sessions, 1 urgent agent.",
    },
    "p4_cliff": {
        "goal": "KV-pool pressure. More filler work pushes harder on cache capacity and backend pressure.",
        "knobs": "50 ms tool wait, 4096-token target prompt, 48 filler sessions, 1 urgent agent.",
    },
    "p5_boss_queue": {
        "goal": "Urgent burst pressure. Many urgent agents become ready together, so priority cannot make all of them first.",
        "knobs": "50 ms tool wait, 4096-token target prompt, 4 filler sessions per group, 8 urgent agents.",
    },
}

PRESSURE_ORDER = (
    "p0_control",
    "p1_mild",
    "p2_medium",
    "p3_high",
    "p4_cliff",
    "p5_boss_queue",
)

SIGNAL_FAMILY_DEFINITIONS = [
    {
        "family": "Baseline",
        "where_signal_is_added": "Nowhere",
        "what_it_means": "No priority or cache signal is supplied for the studied replay request.",
        "raw_modes": "no_prefetch",
    },
    {
        "family": "Harness-originated",
        "where_signal_is_added": "Inside or by the harness",
        "what_it_means": "The harness emits cache or priority intent; the gateway lowers cache signals to gateway speculative KV preload and priority signals to SGLang priority.",
        "raw_modes": "harness_emitted_signals",
    },
    {
        "family": "Front-end supplied",
        "where_signal_is_added": "Before the request enters the harness",
        "what_it_means": "The experiment/front end marks the request before the harness sees it; the gateway translates what survives.",
        "raw_modes": "pre_harness_priority_hints",
    },
    {
        "family": "Gateway-injected",
        "where_signal_is_added": "After the harness, before SGLang",
        "what_it_means": "The harness output is normal; the gateway attaches SGLang priority at the backend boundary.",
        "raw_modes": "e2e_priority_hints",
    },
    {
        "family": "Controller observe-only",
        "where_signal_is_added": "Portable controller sidecar",
        "what_it_means": "The controller consumes lifecycle state and records the actions it would take, but does not mutate SGLang.",
        "raw_modes": "controller_observe_only",
    },
    {
        "family": "Controller scheduler priority",
        "where_signal_is_added": "Portable controller sidecar, lowered by gateway",
        "what_it_means": "The controller consumes lifecycle state and, when replay is ready, authorizes SGLang scheduler priority. No speculative KV work is enabled in this phase.",
        "raw_modes": "controller_scheduler_priority",
    },
    {
        "family": "Controller speculative KV preload",
        "where_signal_is_added": "Portable controller sidecar, lowered by gateway",
        "what_it_means": "The controller consumes lifecycle state during tool wait and authorizes a gateway speculative KV preload before replay. No scheduler priority is enabled in this phase.",
        "raw_modes": "controller_speculative_preload",
    },
    {
        "family": "Controller targeted KV prefetch",
        "where_signal_is_added": "Portable controller sidecar, lowered by backend adapter when supported",
        "what_it_means": "The controller requests explicit host-to-device KV movement for the target prefix. If the active SGLang version has no stable direct hook, the report records that honestly instead of using a warmup fallback.",
        "raw_modes": "controller_targeted_kv_prefetch",
    },
    {
        "family": "Full controller",
        "where_signal_is_added": "Portable controller sidecar, lowered by gateway",
        "what_it_means": "The controller combines the useful EC2 pieces: demote background traffic, priority-raise replay, skip speculative preload by policy, restore after replay, and rank tied urgent replays by deadline.",
        "raw_modes": "controller_full",
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_run_config(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    config: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key] = value
    return config


def read_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv_table(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def case_key_from_name(name: str) -> tuple[str, str, str]:
    for harness in sorted(HARNESS_LABELS, key=len, reverse=True):
        prefix = f"{harness}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        for pressure in sorted(PRESSURE_LABELS, key=len, reverse=True):
            prefix_pressure = f"{pressure}_"
            if not rest.startswith(prefix_pressure):
                continue
            mode_part = rest[len(prefix_pressure) :]
            mode = next((candidate for candidate in sorted(MODE_ORDER, key=len, reverse=True) if mode_part.startswith(candidate)), "no_prefetch")
            return harness, pressure, mode
    return "", "", ""


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def is_truthy_text(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def has_value(value: Any) -> bool:
    return str(value or "").strip().lower() not in {"", "0", "none", "null", "false", "no"}


def row_agent_label(row: dict[str, Any]) -> str:
    for key in ("agent_label", "agent_request_id", "request_id", "label"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def timestamp_ns(value: Any) -> int:
    # Integer nanoseconds exceed float's exact range. Never round through float.
    try:
        return int(value) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def collect_rows(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        harness, pressure, mode = case_key_from_name(case_dir.name)
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        due_by_session: dict[str, dict[str, Any]] = {}
        request_starts: list[dict[str, Any]] = []
        request_ends: list[dict[str, Any]] = []
        sglang_receive_by_label: dict[str, dict[str, Any]] = {}
        first_decode_by_label: dict[str, dict[str, Any]] = {}
        for row in trace_rows:
            event = row.get("event")
            phase = row.get("phase")
            if event == "m27.replay.due":
                due_by_session[str(row.get("session_id") or "")] = row
            elif event == "m27.request.start" and phase in {"replay", "pressure_filler"}:
                request_starts.append(row)
            elif event == "m27.request.end" and phase in {"replay", "pressure_filler"}:
                request_ends.append(row)
            elif (
                event == "kv_telemetry.request_stage"
                and row.get("category") == "sglang_receive"
                and row_agent_label(row)
            ):
                label = row_agent_label(row)
                current = sglang_receive_by_label.get(label)
                if current is None or timestamp_ns(row.get("ts_ns")) < timestamp_ns(current.get("ts_ns")):
                    sglang_receive_by_label[label] = row
            elif (
                event == "kv_telemetry.request_stage"
                and row.get("category") == "scheduler_process_decode_result"
                and phase == "end"
                and row_agent_label(row)
            ):
                label = row_agent_label(row)
                current = first_decode_by_label.get(label)
                if current is None or timestamp_ns(row.get("ts_ns")) < timestamp_ns(current.get("ts_ns")):
                    first_decode_by_label[label] = row

        prefill_by_label = prefill_token_stats_by_label(trace_rows)

        def request_group(phase: str, session_id: str, label: str) -> str:
            if phase == "pressure_filler" or "_pressure_" in session_id or "_pressure_" in label:
                return "filler"
            return "target"

        def append_timing_row(
            *,
            label: str,
            session_id: str,
            source_row: dict[str, Any],
            start: dict[str, Any],
            due: dict[str, Any],
            first_token_ts_ns: int,
            request_end_ts_ns: int | str,
            first_token_source: str,
            status: Any,
            error: Any,
        ) -> None:
            start_ts_ns = timestamp_ns(start.get("ts_ns"))
            due_ts_ns = timestamp_ns(due.get("ts_ns"))
            ttft_ms = ((first_token_ts_ns - start_ts_ns) / 1_000_000.0) if first_token_ts_ns and start_ts_ns else float("nan")
            lateness_ms = ((first_token_ts_ns - due_ts_ns) / 1_000_000.0) if due_ts_ns and first_token_ts_ns else float("nan")
            replay_debt_ms = max(lateness_ms, 0.0) if math.isfinite(lateness_ms) else float("nan")
            sglang_receive = sglang_receive_by_label.get(label, {})
            sglang_receive_ts_ns = timestamp_ns(sglang_receive.get("ts_ns"))
            receive_source = "sglang_receive_hook" if sglang_receive_ts_ns else "gateway_request_start_fallback"
            backend_start_ts_ns = sglang_receive_ts_ns or start_ts_ns
            due_to_request_start_ms = ((start_ts_ns - due_ts_ns) / 1_000_000.0) if due_ts_ns and start_ts_ns else float("nan")
            due_to_sglang_receive_ms = ((backend_start_ts_ns - due_ts_ns) / 1_000_000.0) if due_ts_ns and backend_start_ts_ns else float("nan")
            sglang_receive_to_first_token_ms = (
                ((first_token_ts_ns - backend_start_ts_ns) / 1_000_000.0)
                if first_token_ts_ns and backend_start_ts_ns
                else float("nan")
            )
            row_harness = str(source_row.get("harness") or harness)
            row_mode = str(source_row.get("mode") or mode)
            row_phase = str(source_row.get("phase") or start.get("phase") or "")
            group = request_group(row_phase, session_id, label)
            out.append(
                {
                    **{key: value for key, value in source_row.items() if key.startswith("encoding_")},
                    **prefill_by_label.get(label, {}),
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                    "harness": row_harness,
                    "harness_label": HARNESS_LABELS.get(row_harness, row_harness),
                    "mode": row_mode,
                    "mode_label": MODE_LABELS.get(row_mode, row_mode),
                    "pressure_level": pressure,
                    "pressure_level_label": PRESSURE_LABELS.get(pressure, pressure),
                    "session_id": session_id,
                    "request_id": label,
                    "phase": row_phase,
                    "request_group": group,
                    "has_replay_deadline": "yes" if due_ts_ns else "no",
                    "first_token_lateness_ms": round(lateness_ms, 3) if math.isfinite(lateness_ms) else "",
                    "replay_debt_ms": round(replay_debt_ms, 3) if math.isfinite(replay_debt_ms) else "",
                    "due_to_request_start_ms": round(due_to_request_start_ms, 3) if math.isfinite(due_to_request_start_ms) else "",
                    "due_to_sglang_receive_ms": round(due_to_sglang_receive_ms, 3) if math.isfinite(due_to_sglang_receive_ms) else "",
                    "sglang_receive_to_first_token_ms": round(sglang_receive_to_first_token_ms, 3) if math.isfinite(sglang_receive_to_first_token_ms) else "",
                    "ttft_ms": round(ttft_ms, 3) if math.isfinite(ttft_ms) else "",
                    "request_start_ts_ns": start_ts_ns,
                    "sglang_receive_ts_ns": sglang_receive_ts_ns,
                    "first_token_ts_ns": first_token_ts_ns,
                    "replay_due_ts_ns": due_ts_ns,
                    "request_end_ts_ns": request_end_ts_ns,
                    "backend_receive_source": receive_source,
                    "first_token_source": first_token_source,
                    "sglang_priority": source_row.get("sglang_priority", ""),
                    "experiment_priority_intent": source_row.get("experiment_priority_intent", ""),
                    "harness_input_priority_signal": source_row.get("harness_input_priority_signal", ""),
                    "harness_input_priority_signal_source": source_row.get("harness_input_priority_signal_source", ""),
                    "harness_emit_priority_signal": source_row.get("harness_emit_priority_signal", ""),
                    "harness_emit_priority_signal_source": source_row.get("harness_emit_priority_signal_source", ""),
                    "gateway_priority_translation": source_row.get("gateway_priority_translation", ""),
                    "gateway_priority_translation_source": source_row.get("gateway_priority_translation_source", ""),
                    "harness_native_cache_signal_seen": source_row.get("harness_native_cache_signal_seen", ""),
                    "harness_native_cache_signal": source_row.get("harness_native_cache_signal", ""),
                    "harness_native_cache_signal_source": source_row.get("harness_native_cache_signal_source", ""),
                    "harness_native_cache_identity_signal": source_row.get("harness_native_cache_identity_signal", ""),
                    "harness_native_cache_identity_source": source_row.get("harness_native_cache_identity_source", ""),
                    "gateway_cache_translation": source_row.get("gateway_cache_translation", ""),
                    "gateway_cache_translation_source": source_row.get("gateway_cache_translation_source", ""),
                    "gateway_cache_lowered": source_row.get("gateway_cache_lowered", ""),
                    "gateway_cache_salt": source_row.get("gateway_cache_salt", ""),
                    "gateway_cache_invented_signal": source_row.get("gateway_cache_invented_signal", ""),
                    "controller_replay_rank": source_row.get("controller_replay_rank", ""),
                    "controller_urgent_replay_count": source_row.get("controller_urgent_replay_count", ""),
                    "controller_priority_ladder": source_row.get("controller_priority_ladder", ""),
                    "status": status,
                    "error": error,
                }
            )

        start_by_label = {str(row.get("label") or row.get("request_id") or ""): row for row in request_starts}
        ended_labels: set[str] = set()
        for end in request_ends:
            label = str(end.get("label") or end.get("request_id") or "")
            ended_labels.add(label)
            session_id = str(end.get("session_id") or "")
            start = start_by_label.get(label, {})
            due = due_by_session.get(session_id, {})
            start_ts_ns = timestamp_ns(start.get("ts_ns"))
            ttft_ms = float_value(end.get("ttft_ms"))
            if "first_content_ts_ns" in end:
                first_token_ts_ns = timestamp_ns(end.get("first_content_ts_ns"))
                first_token_source = "gateway_first_content_timestamp" if first_token_ts_ns else "not_observed"
            else:
                # Legacy runs only: their TTFT starts after gateway payload construction.
                first_token_ts_ns = start_ts_ns + int(round(ttft_ms * 1_000_000)) if not end.get("error") else 0
                first_token_source = "legacy_gateway_request_end_ttft_inferred"
            append_timing_row(
                label=label,
                session_id=session_id,
                source_row=end,
                start=start,
                due=due,
                first_token_ts_ns=first_token_ts_ns,
                request_end_ts_ns=timestamp_ns(end.get("ts_ns")),
                first_token_source=first_token_source,
                status=end.get("status", ""),
                error=end.get("error", ""),
            )
        for label, start in start_by_label.items():
            if label in ended_labels or label not in first_decode_by_label:
                continue
            session_id = str(start.get("session_id") or label.rsplit("_replay", 1)[0])
            due = due_by_session.get(session_id, {})
            append_timing_row(
                label=label,
                session_id=session_id,
                source_row=start,
                start=start,
                due=due,
                first_token_ts_ns=int(float_value(first_decode_by_label[label].get("ts_ns"))),
                request_end_ts_ns="",
                first_token_source="scheduler_process_decode_result",
                status="backend_first_token_only",
                error="client_stream_incomplete",
            )
    return out


def ns_to_ms_delta(start_ns: int, end_ns: int) -> float | None:
    if not start_ns or not end_ns:
        return None
    return (end_ns - start_ns) / 1_000_000.0


def prefill_token_stats_by_label(trace_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(dict)

    def update(label: str, request: dict[str, Any], source: str) -> None:
        if not label:
            return
        entry = stats[label]
        entry["source"] = source
        for key in (
            "prefill_full_input_tokens",
            "prefill_active_input_tokens",
            "prefill_cached_prefix_tokens",
            "prefill_uncached_token_count",
            "prefill_host_hit_tokens",
            "prefill_scheduler_trimmed_tokens",
        ):
            value = optional_float(request.get(key))
            if value is None:
                continue
            current = optional_float(entry.get(key))
            if current is None or value > current:
                entry[key] = int(value)

    for row in trace_rows:
        label = row_agent_label(row)
        if label:
            update(label, row, str(row.get("source_event") or row.get("event") or "trace"))
        attribution = row.get("batch_request_prefill_attribution")
        if not isinstance(attribution, list):
            continue
        for request in attribution:
            if not isinstance(request, dict):
                continue
            update(row_agent_label(request), request, str(row.get("source_event") or row.get("event") or "batch"))
    return stats


def collect_speculative_prefill_proof(root: Path, replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replay_by_session = {
        (str(row.get("case_dir") or ""), str(row.get("session_id") or "")): row
        for row in replay_rows
        if row.get("mode")
        in {"e2e_priority_hints_speculative_prefill", "harness_emitted_signals", "controller_speculative_preload"}
    }
    proof_rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        if not trace_rows:
            continue
        token_stats = prefill_token_stats_by_label(trace_rows)
        by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in trace_rows:
            by_event[str(row.get("event") or "")].append(row)
        warmup_starts = [row for row in by_event.get("m27.request.start", []) if row.get("phase") == "speculative_prefill"]
        for warmup_start in warmup_starts:
            session_id = str(warmup_start.get("session_id") or "")
            replay_row = replay_by_session.get((str(case_dir), session_id), {})
            warmup_label = str(warmup_start.get("label") or warmup_start.get("request_id") or "")
            expected_replay = str(warmup_start.get("expected_replay_request_id") or f"{session_id}_replay")
            warmup_end = next(
                (
                    row
                    for row in by_event.get("m27.request.end", [])
                    if row.get("phase") == "speculative_prefill"
                    and str(row.get("label") or row.get("request_id") or "") == warmup_label
                ),
                {},
            )
            hint_seen = next(
                (
                    row
                    for row in by_event.get("m27.speculative_prefill.hint_seen", [])
                    if str(row.get("warmup_request_id") or "") == warmup_label
                ),
                {},
            )
            warmup_start_ns = int(float_value(warmup_start.get("ts_ns")))
            warmup_end_ns = int(float_value(warmup_end.get("ts_ns")))
            replay_due_ns = int(float_value(replay_row.get("replay_due_ts_ns")))
            replay_start_ns = int(float_value(replay_row.get("request_start_ts_ns")))
            replay_receive_ns = int(float_value(replay_row.get("sglang_receive_ts_ns")))
            warmup_stats = token_stats.get(warmup_label, {})
            replay_stats = token_stats.get(expected_replay, {})
            warmup_before_due = bool(warmup_start_ns and replay_due_ns and warmup_start_ns <= replay_due_ns)
            completed_before_replay = bool(warmup_end_ns and replay_start_ns and warmup_end_ns <= replay_start_ns)
            completed_before_backend_receive = bool(
                warmup_end_ns
                and (replay_receive_ns or replay_start_ns)
                and warmup_end_ns <= (replay_receive_ns or replay_start_ns)
            )
            cached_tokens = replay_stats.get("prefill_cached_prefix_tokens", "")
            verdict = "warmup sent"
            if completed_before_backend_receive and cached_tokens not in ("", 0, "0"):
                verdict = "warmup completed and replay showed cached prefix"
            elif completed_before_backend_receive:
                verdict = "warmup completed before replay, cached-prefix evidence missing"
            elif warmup_end_ns:
                verdict = "warmup completed too late for replay"
            proof_rows.append(
                {
                    "harness": replay_row.get("harness", warmup_start.get("harness", "")),
                    "harness_label": replay_row.get(
                        "harness_label",
                        HARNESS_LABELS.get(str(warmup_start.get("harness") or ""), str(warmup_start.get("harness") or "")),
                    ),
                    "pressure_level": replay_row.get("pressure_level", ""),
                    "pressure_level_label": replay_row.get("pressure_level_label", ""),
                    "mode": warmup_start.get("mode", ""),
                    "session_id": session_id,
                    "warmup_request_id": warmup_label,
                    "expected_replay_request_id": expected_replay,
                    "hint_seen": "yes" if hint_seen else "no",
                    "strategy": warmup_start.get("speculative_prefill_strategy") or hint_seen.get("strategy", ""),
                    "warmup_started_before_replay_due": "yes" if warmup_before_due else "no",
                    "warmup_completed_before_replay": "yes" if completed_before_replay else "no",
                    "warmup_completed_before_sglang_receive": "yes" if completed_before_backend_receive else "no",
                    "warmup_total_latency_ms": warmup_end.get("total_latency_ms", ""),
                    "warmup_prompt_tokens": warmup_start.get("warmup_prompt_tokens", ""),
                    "warmup_full_input_tokens": warmup_stats.get("prefill_full_input_tokens", ""),
                    "warmup_uncached_tokens": warmup_stats.get("prefill_uncached_token_count", ""),
                    "replay_cached_prefix_tokens": replay_stats.get("prefill_cached_prefix_tokens", ""),
                    "replay_uncached_tokens": replay_stats.get("prefill_uncached_token_count", ""),
                    "replay_first_token_lateness_ms": replay_row.get("first_token_lateness_ms", ""),
                    "replay_backend_ms": replay_row.get("sglang_receive_to_first_token_ms", ""),
                    "warmup_start_to_replay_due_ms": (
                        round(ns_to_ms_delta(warmup_start_ns, replay_due_ns), 3)
                        if ns_to_ms_delta(warmup_start_ns, replay_due_ns) is not None
                        else ""
                    ),
                    "warmup_end_to_replay_start_ms": (
                        round(ns_to_ms_delta(warmup_end_ns, replay_start_ns), 3)
                        if ns_to_ms_delta(warmup_end_ns, replay_start_ns) is not None
                        else ""
                    ),
                    "verdict": verdict,
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                }
            )
    return proof_rows


def collect_targeted_kv_prefetch_proof(root: Path, replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replay_by_session = {
        (str(row.get("case_dir") or ""), str(row.get("session_id") or "")): row
        for row in replay_rows
        if row.get("mode") == "controller_targeted_kv_prefetch"
    }
    proof_rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        if not trace_rows:
            continue
        by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in trace_rows:
            by_event[str(row.get("event") or "")].append(row)

        for requested in by_event.get("m27.targeted_kv_prefetch.requested", []):
            session_id = str(requested.get("session_id") or "")
            replay_row = replay_by_session.get((str(case_dir), session_id), {})
            expected_replay = str(requested.get("expected_replay_request_id") or replay_row.get("request_id") or "")
            requested_ts_ns = int(float_value(requested.get("ts_ns")))
            replay_start_ts_ns = int(float_value(replay_row.get("request_start_ts_ns")))
            replay_receive_ts_ns = int(float_value(replay_row.get("sglang_receive_ts_ns")))
            replay_compute_ts_ns = int(float_value(replay_row.get("first_token_ts_ns")))
            outcome = next(
                (
                    row
                    for row in trace_rows
                    if str(row.get("event") or "").startswith("m27.targeted_kv_prefetch.")
                    and str(row.get("event") or "") != "m27.targeted_kv_prefetch.requested"
                    and str(row.get("controller_command_id") or "")
                    == str(requested.get("controller_command_id") or "")
                ),
                {},
            )

            load_back_events = 0
            h2d_copy_events = 0
            first_movement_ts_ns = 0
            for trace_row in trace_rows:
                event = str(trace_row.get("event") or "")
                category = str(trace_row.get("category") or "")
                method = str(trace_row.get("method") or "")
                source_event = str(trace_row.get("source_event") or "")
                action_text = " ".join((event, category, method, source_event)).lower()
                same_session = str(trace_row.get("session_id") or "") == session_id
                target_match = row_matches_request(trace_row, expected_replay)
                if not same_session and not target_match:
                    continue
                is_load_back = category in {"init_load_back", "load_back", "hicache_load"} or method in {
                    "init_load_back",
                    "load_back",
                    "hicache_load",
                }
                is_h2d = category == "host_to_device_copy" or method == "load_to_device_per_layer"
                if not (is_load_back or is_h2d or "load_back" in action_text):
                    continue
                ts_ns = int(float_value(trace_row.get("ts_ns")))
                if requested_ts_ns and ts_ns and ts_ns < requested_ts_ns:
                    continue
                if replay_compute_ts_ns and ts_ns and ts_ns > replay_compute_ts_ns:
                    continue
                if is_load_back or "load_back" in action_text:
                    load_back_events += 1
                if is_h2d:
                    h2d_copy_events += 1
                if ts_ns and (not first_movement_ts_ns or ts_ns < first_movement_ts_ns):
                    first_movement_ts_ns = ts_ns

            backend_acted = is_truthy_text(requested.get("backend_acted") or outcome.get("backend_acted"))
            backend_accepted = is_truthy_text(requested.get("backend_accepted") or outcome.get("backend_accepted"))
            direct_hook_available = is_truthy_text(requested.get("direct_hook_available") or outcome.get("direct_hook_available"))
            requested_before_due = bool(
                requested_ts_ns
                and replay_row.get("replay_due_ts_ns")
                and requested_ts_ns <= int(float_value(replay_row.get("replay_due_ts_ns")))
            )
            requested_before_replay = bool(requested_ts_ns and replay_start_ts_ns and requested_ts_ns <= replay_start_ts_ns)
            movement_before_receive = bool(
                first_movement_ts_ns and replay_receive_ts_ns and first_movement_ts_ns <= replay_receive_ts_ns
            )
            movement_before_compute = bool(
                first_movement_ts_ns and replay_compute_ts_ns and first_movement_ts_ns <= replay_compute_ts_ns
            )

            if backend_acted and movement_before_compute:
                verdict = "targeted prefetch acted and KV movement was observed before replay compute"
            elif backend_acted:
                verdict = "targeted prefetch hook acted, but KV movement telemetry was not observed before replay compute"
            elif backend_accepted and not direct_hook_available:
                verdict = "targeted prefetch requested, but this SGLang version exposed no stable direct hook"
            elif backend_accepted:
                verdict = "targeted prefetch accepted but did not act"
            else:
                verdict = "targeted prefetch was not accepted"

            proof_rows.append(
                {
                    "harness_label": replay_row.get(
                        "harness_label",
                        HARNESS_LABELS.get(str(requested.get("harness") or ""), str(requested.get("harness") or "")),
                    ),
                    "pressure_level_label": replay_row.get("pressure_level_label", ""),
                    "mode_label": replay_row.get("mode_label", MODE_LABELS.get(str(requested.get("mode") or ""), "")),
                    "session_id": session_id,
                    "request_id": requested.get("request_id", ""),
                    "expected_replay_request_id": expected_replay,
                    "controller_decision_id": requested.get("controller_decision_id", ""),
                    "controller_command_id": requested.get("controller_command_id", ""),
                    "backend_name": requested.get("backend_name", ""),
                    "backend_accepted": "yes" if backend_accepted else "no",
                    "backend_acted": "yes" if backend_acted else "no",
                    "direct_hook_available": "yes" if direct_hook_available else "no",
                    "requested_before_replay_due": "yes" if requested_before_due else "no",
                    "requested_before_replay_start": "yes" if requested_before_replay else "no",
                    "requested_to_replay_due_ms": (
                        round(ns_to_ms_delta(requested_ts_ns, int(float_value(replay_row.get("replay_due_ts_ns")))), 3)
                        if ns_to_ms_delta(requested_ts_ns, int(float_value(replay_row.get("replay_due_ts_ns")))) is not None
                        else ""
                    ),
                    "requested_to_replay_start_ms": (
                        round(ns_to_ms_delta(requested_ts_ns, replay_start_ts_ns), 3)
                        if ns_to_ms_delta(requested_ts_ns, replay_start_ts_ns) is not None
                        else ""
                    ),
                    "load_back_events_before_replay_compute": load_back_events,
                    "h2d_copy_events_before_replay_compute": h2d_copy_events,
                    "first_movement_before_sglang_receive": "yes" if movement_before_receive else "no",
                    "first_movement_before_replay_compute": "yes" if movement_before_compute else "no",
                    "backend_reason": requested.get("backend_reason", outcome.get("backend_reason", "")),
                    "verdict": verdict,
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                }
            )
    return proof_rows


def collect_controller_demote_restore_proof(root: Path, replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replay_by_session = {
        (str(row.get("case_dir") or ""), str(row.get("session_id") or "")): row
        for row in replay_rows
        if row.get("mode") in {"controller_demote_restore", "controller_full"}
    }
    proof_rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        if not trace_rows:
            continue
        demote_events = [
            row for row in trace_rows if str(row.get("event") or "") == "m27.controller_demote_restore.demote_start"
        ]
        if not demote_events:
            continue
        request_starts = [row for row in trace_rows if row.get("event") == "m27.request.start"]
        restore_events = [
            row for row in trace_rows if str(row.get("event") or "") == "m27.controller_demote_restore.restored"
        ]
        for demote in demote_events:
            session_id = str(demote.get("session_id") or "")
            replay_row = replay_by_session.get((str(case_dir), session_id), {})
            demote_ts_ns = int(float_value(demote.get("ts_ns")))
            replay_start_ts_ns = int(float_value(replay_row.get("request_start_ts_ns")))
            replay_due_ts_ns = int(float_value(replay_row.get("replay_due_ts_ns")))
            restore = next((row for row in restore_events if str(row.get("session_id") or "") == session_id), {})
            restore_ts_ns = int(float_value(restore.get("ts_ns")))
            filler_prefix = f"{session_id}_pressure_"
            filler_rows = [
                row
                for row in request_starts
                if str(row.get("phase") or "") == "pressure_filler"
                and str(row.get("session_id") or "").startswith(filler_prefix)
            ]
            filler_demoted = [
                row for row in filler_rows if optional_float(row.get("sglang_priority")) is not None and float_value(row.get("sglang_priority")) < 0
            ]
            filler_not_demoted = [
                row
                for row in filler_rows
                if not (optional_float(row.get("sglang_priority")) is not None and float_value(row.get("sglang_priority")) < 0)
            ]
            filler_before_replay = [
                row
                for row in filler_rows
                if demote_ts_ns
                and int(float_value(row.get("ts_ns")))
                and int(float_value(row.get("ts_ns"))) >= demote_ts_ns
                and (not replay_start_ts_ns or int(float_value(row.get("ts_ns"))) <= replay_start_ts_ns)
            ]
            replay_priority = optional_float(replay_row.get("sglang_priority"))
            demote_acted = is_truthy_text(demote.get("backend_acted"))
            restore_acted = is_truthy_text(restore.get("backend_acted"))
            replay_raised = replay_priority is not None and replay_priority >= 100
            if demote_acted and filler_rows and not filler_not_demoted and replay_raised and restore_acted:
                verdict = "filler traffic was demoted, replay was raised, and restore was recorded"
            elif demote_acted and not filler_rows and replay_raised and restore_acted:
                verdict = "demote/restore acted, but this pressure level had no filler requests"
            elif demote_acted and filler_demoted and replay_raised:
                verdict = "demotion and replay raise were seen; restore proof is incomplete"
            elif demote_acted:
                verdict = "demote command acted, but request-level proof is incomplete"
            else:
                verdict = "demote command did not act"
            proof_rows.append(
                {
                    "harness_label": replay_row.get(
                        "harness_label",
                        HARNESS_LABELS.get(str(demote.get("harness") or ""), str(demote.get("harness") or "")),
                    ),
                    "pressure_level_label": replay_row.get(
                        "pressure_level_label",
                        PRESSURE_LABELS.get(str(demote.get("pressure_level") or ""), str(demote.get("pressure_level") or "")),
                    ),
                    "mode_label": replay_row.get(
                        "mode_label",
                        MODE_LABELS.get(str(demote.get("mode") or ""), str(demote.get("mode") or "")),
                    ),
                    "session_id": session_id,
                    "demote_command_id": demote.get("controller_command_id", ""),
                    "demote_backend_acted": "yes" if demote_acted else "no",
                    "demoted_priority": demote.get("demoted_priority", ""),
                    "filler_requests_seen": len(filler_rows),
                    "filler_requests_between_demote_and_replay": len(filler_before_replay),
                    "filler_demoted_count": len(filler_demoted),
                    "filler_not_demoted_count": len(filler_not_demoted),
                    "replay_request_id": replay_row.get("request_id", ""),
                    "replay_sglang_priority": replay_row.get("sglang_priority", ""),
                    "replay_priority_raised": "yes" if replay_raised else "no",
                    "controller_replay_rank": replay_row.get("controller_replay_rank", ""),
                    "controller_urgent_replay_count": replay_row.get("controller_urgent_replay_count", ""),
                    "controller_priority_ladder": replay_row.get("controller_priority_ladder", ""),
                    "restore_command_id": restore.get("controller_command_id", ""),
                    "restore_backend_acted": "yes" if restore_acted else "no",
                    "demote_to_replay_due_ms": (
                        round(ns_to_ms_delta(demote_ts_ns, replay_due_ts_ns), 3)
                        if ns_to_ms_delta(demote_ts_ns, replay_due_ts_ns) is not None
                        else ""
                    ),
                    "demote_to_replay_start_ms": (
                        round(ns_to_ms_delta(demote_ts_ns, replay_start_ts_ns), 3)
                        if ns_to_ms_delta(demote_ts_ns, replay_start_ts_ns) is not None
                        else ""
                    ),
                    "replay_to_restore_ms": (
                        round(ns_to_ms_delta(replay_start_ts_ns, restore_ts_ns), 3)
                        if ns_to_ms_delta(replay_start_ts_ns, restore_ts_ns) is not None
                        else ""
                    ),
                    "first_token_lateness_ms": replay_row.get("first_token_lateness_ms", ""),
                    "verdict": verdict,
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                }
            )
    return proof_rows


def collect_controller_admission_proof(root: Path, replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replay_by_session = {
        (str(row.get("case_dir") or ""), str(row.get("session_id") or "")): row
        for row in replay_rows
        if row.get("mode") in {"controller_admission_control", "controller_full"}
    }
    proof_rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        if not trace_rows:
            continue
        admission_events = [
            row for row in trace_rows if str(row.get("event") or "") == "m27.controller_admission.decision"
        ]
        if not admission_events:
            continue
        warmup_start_by_session = {
            str(row.get("session_id") or ""): row
            for row in trace_rows
            if str(row.get("event") or "") == "m27.speculative_prefill.warmup_start"
            and str(row.get("trigger") or "") == "controller_admission_decision"
        }
        warmup_end_by_session = {
            str(row.get("session_id") or ""): row
            for row in trace_rows
            if str(row.get("event") or "") == "m27.speculative_prefill.warmup_end"
            and str(row.get("trigger") or "") == "controller_admission_decision"
        }
        warmup_error_by_session = {
            str(row.get("session_id") or ""): row
            for row in trace_rows
            if str(row.get("event") or "") == "m27.speculative_prefill.warmup_error"
            and str(row.get("trigger") or "") == "controller_admission_decision"
        }
        for admission in admission_events:
            session_id = str(admission.get("session_id") or "")
            replay_row = replay_by_session.get((str(case_dir), session_id), {})
            warmup_start = warmup_start_by_session.get(session_id, {})
            warmup_end = warmup_end_by_session.get(session_id, {})
            warmup_error = warmup_error_by_session.get(session_id, {})
            admitted = str(admission.get("decision") or "") == "admit"
            warmup_started = bool(warmup_start)
            warmup_completed = bool(warmup_end)
            warmup_failed = bool(warmup_error)
            replay_priority = optional_float(replay_row.get("sglang_priority"))
            replay_raised = replay_priority is not None and replay_priority >= 100
            if admitted and warmup_completed and replay_raised:
                verdict = "warmup admitted, completed, and replay was priority-raised"
            elif admitted and warmup_started and warmup_failed:
                verdict = "warmup admitted but failed"
            elif admitted and not warmup_started:
                verdict = "warmup admitted but no warmup request was observed"
            elif not admitted and replay_raised:
                verdict = "warmup skipped with explicit reason; replay was still priority-raised"
            else:
                verdict = "admission proof incomplete"
            proof_rows.append(
                {
                    "harness_label": replay_row.get(
                        "harness_label",
                        HARNESS_LABELS.get(str(admission.get("harness") or ""), str(admission.get("harness") or "")),
                    ),
                    "pressure_level_label": replay_row.get(
                        "pressure_level_label",
                        PRESSURE_LABELS.get(str(admission.get("pressure_level") or ""), str(admission.get("pressure_level") or "")),
                    ),
                    "mode_label": replay_row.get(
                        "mode_label",
                        MODE_LABELS.get(str(admission.get("mode") or ""), str(admission.get("mode") or "")),
                    ),
                    "session_id": session_id,
                    "admission_decision": admission.get("decision", ""),
                    "admission_reason": admission.get("reason", ""),
                    "prefetch_command_id": admission.get("prefetch_command_id", ""),
                    "budget_command_id": admission.get("budget_command_id", ""),
                    "admitted_warmups_before": admission.get("admitted_warmups_before", ""),
                    "admitted_warmups_after": admission.get("admitted_warmups_after", ""),
                    "max_warmups_per_case": admission.get("max_warmups_per_case", ""),
                    "tool_wait_ms": admission.get("tool_wait_ms", ""),
                    "filler_sessions": admission.get("filler_sessions", ""),
                    "concurrency": admission.get("concurrency", ""),
                    "min_tool_wait_ms": admission.get("min_tool_wait_ms", ""),
                    "max_filler_sessions": admission.get("max_filler_sessions", ""),
                    "max_concurrency": admission.get("max_concurrency", ""),
                    "warmup_started": "yes" if warmup_started else "no",
                    "warmup_completed": "yes" if warmup_completed else "no",
                    "warmup_failed": "yes" if warmup_failed else "no",
                    "replay_request_id": replay_row.get("request_id", ""),
                    "replay_sglang_priority": replay_row.get("sglang_priority", ""),
                    "replay_priority_raised": "yes" if replay_raised else "no",
                    "first_token_lateness_ms": replay_row.get("first_token_lateness_ms", ""),
                    "ttft_ms": replay_row.get("ttft_ms", ""),
                    "sglang_receive_to_first_token_ms": replay_row.get("sglang_receive_to_first_token_ms", ""),
                    "verdict": verdict,
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                }
            )
    return proof_rows


def priority_value_is_urgent(value: Any) -> bool:
    try:
        return int(float(value)) >= 100
    except (TypeError, ValueError):
        return False


def priority_class_from_intent(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("class") or "")
    if not isinstance(value, str) or not value:
        return ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return ""
    return str(parsed.get("class") or "") if isinstance(parsed, dict) else ""


def signal_field(signal: Any, name: str) -> str:
    if not isinstance(signal, str) or not signal:
        return ""
    match = re.search(rf"(?:^|;\s*){re.escape(name)}=([^;]+)", signal)
    return match.group(1).strip() if match else ""


def numeric_strings_match(left: Any, right: Any) -> bool:
    try:
        return int(float(left)) == int(float(right))
    except (TypeError, ValueError):
        return False


def collect_nat_service_priority_probe(root: Path) -> list[dict[str, Any]]:
    proof_rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        if not trace_rows:
            continue
        submits = [row for row in trace_rows if row.get("event") == "m27.nat_service_probe.client_submit"]
        if not submits:
            continue
        probe_start = next((row for row in trace_rows if row.get("event") == "m27.nat_service_probe.start"), {})
        gateway_starts = [
            row
            for row in trace_rows
            if row.get("event") == "m27.request.start"
            and str(row.get("harness") or "") == "nemo_agent_toolkit_service"
        ]
        gateway_ends = [
            row
            for row in trace_rows
            if row.get("event") == "m27.request.end"
            and str(row.get("harness") or "") == "nemo_agent_toolkit_service"
        ]
        client_done = [row for row in trace_rows if row.get("event") == "m27.nat_service_probe.client_done"]
        fake_receives = [row for row in trace_rows if row.get("event") == "m27.nat_service_probe.fake_sglang_receive"]
        starts_by_label = {str(row.get("label") or row.get("request_id") or ""): row for row in gateway_starts}
        ends_by_label = {str(row.get("label") or row.get("request_id") or ""): row for row in gateway_ends}
        done_by_label = {str(row.get("label") or row.get("request_id") or ""): row for row in client_done}
        fake_by_label = {str(row.get("label") or row.get("request_id") or ""): row for row in fake_receives}
        submit_order = {
            str(row.get("label") or row.get("request_id") or ""): index + 1
            for index, row in enumerate(sorted(submits, key=lambda item: int(float_value(item.get("ts_ns")))))
        }
        emit_order = {
            str(row.get("label") or row.get("request_id") or ""): index + 1
            for index, row in enumerate(sorted(gateway_starts, key=lambda item: int(float_value(item.get("ts_ns")))))
        }
        background_submits = [
            row
            for row in submits
            if priority_class_from_intent(row.get("priority_intent")) == "background"
        ]
        for submit in sorted(submits, key=lambda item: int(float_value(item.get("ts_ns")))):
            label = str(submit.get("label") or submit.get("request_id") or "")
            start = starts_by_label.get(label, {})
            end = ends_by_label.get(label, {})
            done = done_by_label.get(label, {})
            fake = fake_by_label.get(label, {})
            submit_ns = int(float_value(submit.get("ts_ns")))
            start_ns = int(float_value(start.get("ts_ns")))
            emit_rank = emit_order.get(label, "")
            priority_class = priority_class_from_intent(submit.get("priority_intent"))
            older_background_submitted = [
                row
                for row in background_submits
                if int(float_value(row.get("ts_ns"))) < submit_ns
            ]
            older_background_emitted_before = [
                row
                for row in older_background_submitted
                if emit_order.get(str(row.get("label") or row.get("request_id") or ""), 10**9)
                < (emit_rank if isinstance(emit_rank, int) else 10**9)
            ]
            native_signal = start.get("harness_emit_priority_signal", "")
            emitted_priority = signal_field(native_signal, "nvext.agent_hints.priority")
            emitted_latency = signal_field(native_signal, "nvext.agent_hints.latency_sensitivity")
            expected_inferred_priority = submit.get("expected_inferred_priority", "")
            frontend_priority_intent_present = "yes" if submit.get("priority_intent") else "no"
            if not start:
                verdict = "NAT did not emit request to gateway"
            elif probe_start.get("nat_provider") == "dynamo_inferred":
                if frontend_priority_intent_present == "yes":
                    verdict = "frontend priority intent present; inference proof is contaminated"
                elif not emitted_priority:
                    verdict = "NAT inferred-priority signal missing"
                elif expected_inferred_priority and numeric_strings_match(emitted_priority, expected_inferred_priority):
                    verdict = "NAT inferred priority from workflow profile and gateway translated it"
                else:
                    verdict = "NAT emitted inferred priority, but value did not match profile expectation"
            elif (
                priority_class == "urgent"
                and older_background_submitted
                and len(older_background_emitted_before) < len(older_background_submitted)
                and native_signal
            ):
                verdict = "priority-bearing urgent request overtook older background work before gateway"
            elif priority_class == "urgent" and older_background_submitted and len(older_background_emitted_before) < len(older_background_submitted):
                verdict = "urgent request overtook older background work; priority cause not proven"
            elif priority_class == "urgent" and older_background_submitted:
                verdict = "no NAT-side priority overtaking observed"
            elif priority_class == "urgent":
                verdict = "urgent request emitted; no older background work to overtake"
            else:
                verdict = "background request"
            proof_rows.append(
                {
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                    "nat_provider": probe_start.get("nat_provider", ""),
                    "nat_dynamo_enable_nvext_hints": probe_start.get("nat_dynamo_enable_nvext_hints", ""),
                    "request_id": label,
                    "priority_class": priority_class,
                    "workflow_node": submit.get("workflow_node", ""),
                    "workflow_node_goal": submit.get("workflow_node_goal", ""),
                    "inference_source": submit.get("inference_source", ""),
                    "expected_inferred_priority": expected_inferred_priority,
                    "frontend_priority_intent_present": frontend_priority_intent_present,
                    "submit_rank_into_nat": submit_order.get(label, ""),
                    "emit_rank_from_nat_to_gateway": emit_rank,
                    "older_background_submitted_before": len(older_background_submitted),
                    "older_background_emitted_before": len(older_background_emitted_before),
                    "submit_to_gateway_emit_ms": (
                        round(ns_to_ms_delta(submit_ns, start_ns), 3)
                        if ns_to_ms_delta(submit_ns, start_ns) is not None
                        else ""
                    ),
                    "gateway_to_fake_sglang_ms": (
                        round(ns_to_ms_delta(start_ns, int(float_value(fake.get("ts_ns")))), 3)
                        if ns_to_ms_delta(start_ns, int(float_value(fake.get("ts_ns")))) is not None
                        else ""
                    ),
                    "gateway_backend_ms": end.get("total_latency_ms", ""),
                    "client_latency_ms": done.get("client_latency_ms", ""),
                    "harness_input_signal": submit.get("harness_input_priority_signal", ""),
                    "harness_input_signal_source": submit.get("harness_input_priority_signal_source", ""),
                    "gateway_saw_marker_intent": "yes" if start.get("experiment_priority_intent") else "no",
                    "harness_output_signal": native_signal,
                    "harness_output_signal_source": start.get("harness_emit_priority_signal_source", ""),
                    "emitted_nvext_priority": emitted_priority,
                    "emitted_latency_sensitivity": emitted_latency,
                    "gateway_translated_priority": start.get("gateway_priority_translation", ""),
                    "gateway_translation_source": start.get("gateway_priority_translation_source", ""),
                    "sglang_priority_seen": start.get("sglang_priority", fake.get("sglang_priority", "")),
                    "client_status": done.get("status", ""),
                    "client_error": done.get("error", ""),
                    "verdict": verdict,
                }
            )
    return proof_rows


def collect_harness_priority_proof(root: Path, replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority_modes = {"pre_harness_priority_hints", "nat_inferred_priority_hints", "harness_emitted_signals"}
    replay_by_label = {
        (str(row.get("case_dir") or ""), str(row.get("request_id") or "")): row
        for row in replay_rows
        if row.get("mode") in priority_modes
    }
    proof_rows: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        if not trace_rows:
            continue
        by_label_event: dict[tuple[str, str], dict[str, Any]] = {}
        for row in trace_rows:
            label = str(row.get("label") or row.get("request_id") or "")
            event = str(row.get("event") or "")
            if label:
                by_label_event[(label, event)] = row
        for row in trace_rows:
            if row.get("event") != "m27.harness.request_input":
                continue
            mode = str(row.get("mode") or "")
            if row.get("phase") != "replay" or mode not in priority_modes:
                continue
            label = str(row.get("label") or row.get("request_id") or "")
            replay_row = replay_by_label.get((str(case_dir), label), {})
            gateway_start = by_label_event.get((label, "m27.request.start"), {})
            nat_config = by_label_event.get((label, "m27.nat_wrapper.config_written"), {})
            nat_process_start = by_label_event.get((label, "m27.nat_wrapper.process_start"), {})
            nat_gateway_emit = by_label_event.get((label, "m27.nat_wrapper.first_gateway_emit"), {})
            nat_process_exit = by_label_event.get((label, "m27.nat_wrapper.process_exit"), {})
            input_ns = int(float_value(row.get("ts_ns")))
            gateway_start_ns = int(float_value(gateway_start.get("ts_ns")))
            nat_config_ns = int(float_value(nat_config.get("ts_ns")))
            nat_process_start_ns = int(float_value(nat_process_start.get("ts_ns")))
            nat_gateway_emit_ns = int(float_value(nat_gateway_emit.get("ts_ns")))
            nat_process_exit_ns = int(float_value(nat_process_exit.get("ts_ns")))
            sglang_priority = replay_row.get("sglang_priority", gateway_start.get("sglang_priority", ""))
            translated = gateway_start.get("gateway_priority_translation", replay_row.get("gateway_priority_translation", ""))
            driver_intent_seen = "yes" if row.get("priority_intent") else "no"
            gateway_saw_intent = "yes" if gateway_start.get("experiment_priority_intent") else "no"
            native_signal = gateway_start.get("harness_emit_priority_signal", "")
            nat_wrapper_seen = "yes" if nat_config or nat_process_start or nat_process_exit else "no"
            if mode == "nat_inferred_priority_hints":
                if not gateway_start:
                    verdict = "harness did not emit inferred-priority request"
                elif not native_signal:
                    verdict = "harness inferred-priority signal missing"
                elif not priority_value_is_urgent(translated):
                    verdict = "gateway did not translate inferred urgent priority"
                elif not priority_value_is_urgent(sglang_priority):
                    verdict = "translated inferred priority missing from SGLang payload"
                else:
                    verdict = "harness inferred priority and gateway translated it"
            elif mode == "harness_emitted_signals":
                if not gateway_start:
                    verdict = "harness did not emit request"
                elif not native_signal:
                    verdict = "no harness priority signal observed for this replay"
                elif not priority_value_is_urgent(translated):
                    verdict = "harness priority signal observed but not translated as urgent"
                elif not priority_value_is_urgent(sglang_priority):
                    verdict = "translated harness priority missing from SGLang payload"
                else:
                    verdict = "harness emitted priority and gateway translated it"
            elif driver_intent_seen != "yes":
                verdict = "driver intent missing"
            elif not gateway_start:
                verdict = "harness did not emit marked request"
            elif not priority_value_is_urgent(translated):
                verdict = "gateway did not translate urgent priority"
            elif not priority_value_is_urgent(sglang_priority):
                verdict = "translated priority missing from SGLang payload"
            elif native_signal:
                verdict = "native emitted signal translated to SGLang priority"
            else:
                verdict = "adapter marker intent translated to SGLang priority"
            proof_rows.append(
                {
                    "harness": replay_row.get("harness", row.get("harness", "")),
                    "harness_label": replay_row.get(
                        "harness_label",
                        HARNESS_LABELS.get(str(row.get("harness") or ""), str(row.get("harness") or "")),
                    ),
                    "pressure_level": replay_row.get("pressure_level", ""),
                    "pressure_level_label": replay_row.get("pressure_level_label", ""),
                    "session_id": row.get("session_id", ""),
                    "request_id": label,
                    "driver_intent_seen": driver_intent_seen,
                    "gateway_saw_intent": gateway_saw_intent,
                    "harness_input_signal": row.get("harness_input_priority_signal", ""),
                    "harness_input_signal_source": row.get("harness_input_priority_signal_source", ""),
                    "harness_output_signal": native_signal,
                    "harness_output_signal_source": gateway_start.get("harness_emit_priority_signal_source", ""),
                    "gateway_translated_priority": translated,
                    "gateway_translation_source": gateway_start.get(
                        "gateway_priority_translation_source",
                        replay_row.get("gateway_priority_translation_source", ""),
                    ),
                    "sglang_priority_seen": sglang_priority,
                    "nat_wrapper_seen": nat_wrapper_seen,
                    "nat_priority_fields": nat_config.get("nat_priority_fields", ""),
                    "nat_priority_field_source": nat_config.get("nat_priority_field_source", ""),
                    "nat_config_delay_ms": (
                        round(ns_to_ms_delta(input_ns, nat_config_ns), 3)
                        if ns_to_ms_delta(input_ns, nat_config_ns) is not None
                        else ""
                    ),
                    "nat_process_start_delay_ms": (
                        round(ns_to_ms_delta(input_ns, nat_process_start_ns), 3)
                        if ns_to_ms_delta(input_ns, nat_process_start_ns) is not None
                        else ""
                    ),
                    "nat_first_gateway_emit_delay_ms": (
                        round(ns_to_ms_delta(input_ns, nat_gateway_emit_ns), 3)
                        if ns_to_ms_delta(input_ns, nat_gateway_emit_ns) is not None
                        else ""
                    ),
                    "nat_process_exit_delay_ms": (
                        round(ns_to_ms_delta(input_ns, nat_process_exit_ns), 3)
                        if ns_to_ms_delta(input_ns, nat_process_exit_ns) is not None
                        else ""
                    ),
                    "nat_gateway_emit_seen": nat_process_exit.get("gateway_emit_seen", ""),
                    "nat_wrapper_total_ms": nat_process_exit.get("wrapper_total_ms", ""),
                    "harness_emit_delay_ms": (
                        round(ns_to_ms_delta(input_ns, gateway_start_ns), 3)
                        if ns_to_ms_delta(input_ns, gateway_start_ns) is not None
                        else ""
                    ),
                    "backend_ms": replay_row.get("sglang_receive_to_first_token_ms", ""),
                    "first_token_lateness_ms": replay_row.get("first_token_lateness_ms", ""),
                    "verdict": verdict,
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                }
            )
    return proof_rows


def collect_harness_native_cache_signal_proof(replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proof_rows: list[dict[str, Any]] = []
    for row in replay_rows:
        mode = str(row.get("mode") or "")
        if mode not in {"no_cache_signal", "harness_native_cache_lowered", "harness_emitted_signals"}:
            continue
        signal_seen = str(row.get("harness_native_cache_signal_seen") or "no")
        lowered = str(row.get("gateway_cache_lowered") or "no")
        invented = str(row.get("gateway_cache_invented_signal") or "false")
        if mode == "harness_emitted_signals" and signal_seen == "yes" and lowered == "yes" and invented == "false":
            verdict = "harness emitted cache signal and gateway translated it to speculative KV preload"
        elif mode == "harness_emitted_signals" and signal_seen == "yes":
            verdict = "harness emitted cache signal but gateway did not lower it"
        elif mode == "harness_emitted_signals":
            verdict = "no native harness cache signal observed to lower"
        elif mode == "harness_native_cache_lowered" and signal_seen == "yes" and lowered == "yes" and invented == "false":
            verdict = "harness emitted cache signal and gateway translated it"
        elif mode == "harness_native_cache_lowered" and signal_seen == "yes":
            verdict = "harness emitted cache signal but gateway did not lower it"
        elif mode == "harness_native_cache_lowered":
            verdict = "no native harness cache signal observed to lower"
        elif signal_seen == "yes":
            verdict = "native cache signal observed in baseline; gateway did not lower it"
        else:
            verdict = "no native harness cache signal observed"
        proof_rows.append(
            {
                "harness_label": row.get("harness_label", ""),
                "pressure_level_label": row.get("pressure_level_label", ""),
                "mode_label": row.get("mode_label", ""),
                "session_id": row.get("session_id", ""),
                "request_id": row.get("request_id", ""),
                "native_cache_signal_seen": signal_seen,
                "native_cache_signal": row.get("harness_native_cache_signal", ""),
                "native_cache_signal_source": row.get("harness_native_cache_signal_source", ""),
                "native_cache_identity_signal": row.get("harness_native_cache_identity_signal", ""),
                "native_cache_identity_source": row.get("harness_native_cache_identity_source", ""),
                "gateway_cache_lowered": lowered,
                "gateway_cache_salt": row.get("gateway_cache_salt", ""),
                "gateway_cache_translation": row.get("gateway_cache_translation", ""),
                "gateway_cache_translation_source": row.get("gateway_cache_translation_source", ""),
                "gateway_invented_signal": invented,
                "first_token_lateness_ms": row.get("first_token_lateness_ms", ""),
                "sglang_receive_to_first_token_ms": row.get("sglang_receive_to_first_token_ms", ""),
                "verdict": verdict,
                "case_id": row.get("case_id", ""),
                "case_dir": row.get("case_dir", ""),
            }
        )
    return proof_rows


def row_matches_request(row: dict[str, Any], request_id: str) -> bool:
    if row_agent_label(row) == request_id:
        return True
    for key in ("request_prefill_attribution", "batch_request_prefill_attribution"):
        attribution = row.get(key)
        if not isinstance(attribution, list):
            continue
        for request in attribution:
            if isinstance(request, dict) and row_agent_label(request) == request_id:
                return True
    return False


def matching_request_attributions(row: dict[str, Any], request_id: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if row_agent_label(row) == request_id:
        matches.append(row)
    for key in ("request_prefill_attribution", "batch_request_prefill_attribution"):
        attribution = row.get(key)
        if not isinstance(attribution, list):
            continue
        for request in attribution:
            if isinstance(request, dict) and row_agent_label(request) == request_id:
                matches.append(request)
    return matches


def collect_cache_action_proof(root: Path, replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proof_rows: list[dict[str, Any]] = []
    cache_modes = {"no_cache_signal", "harness_native_cache_lowered", "harness_emitted_signals"}
    for replay_row in replay_rows:
        mode = str(replay_row.get("mode") or "")
        if mode not in cache_modes:
            continue
        case_dir = Path(str(replay_row.get("case_dir") or ""))
        if not case_dir.exists():
            case_dir = root / str(replay_row.get("case_id") or "")
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        request_id = str(replay_row.get("request_id") or "")
        if not trace_rows or not request_id:
            continue

        token_stats = prefill_token_stats_by_label(trace_rows).get(request_id, {})
        cache_match_events = 0
        load_back_events = 0
        h2d_copy_events = 0
        cache_finished_events = 0
        prefill_events = 0
        runtime_cache_namespace_seen = False
        effective_extra_key = ""
        radix_key_extra_key = ""
        first_cache_action_ts_ns = 0
        sglang_receive_ts_ns = int(float_value(replay_row.get("sglang_receive_ts_ns")))
        max_values: dict[str, float] = defaultdict(float)

        def update_max(output_key: str, value: Any) -> None:
            parsed = optional_float(value)
            if parsed is not None and parsed > max_values[output_key]:
                max_values[output_key] = parsed

        for key, output_key in (
            ("prefill_cached_prefix_tokens", "cached_prefix_tokens"),
            ("prefill_uncached_token_count", "uncached_tokens"),
            ("prefill_host_hit_tokens", "host_hit_tokens"),
        ):
            update_max(output_key, token_stats.get(key))

        for trace_row in trace_rows:
            if not row_matches_request(trace_row, request_id):
                continue
            event = str(trace_row.get("event") or "")
            category = str(trace_row.get("category") or "")
            method = str(trace_row.get("method") or "")
            source_event = str(trace_row.get("source_event") or "")
            action_text = " ".join((event, category, method, source_event)).lower()
            if "match_prefix" in action_text:
                cache_match_events += 1
            if category in {"init_load_back", "load_back", "hicache_load"} or method in {
                "init_load_back",
                "load_back",
                "hicache_load",
            }:
                load_back_events += 1
            if category == "host_to_device_copy" or method == "load_to_device_per_layer":
                h2d_copy_events += 1
            if category == "cache_finished_req" or "cache_finished_req" in action_text:
                cache_finished_events += 1
            if event.startswith("kv_telemetry.prefill") or any(
                key in trace_row for key in ("prefill_full_input_tokens", "batch_request_prefill_attribution")
            ):
                prefill_events += 1
            if has_value(trace_row.get("effective_extra_key")):
                effective_extra_key = effective_extra_key or str(trace_row.get("effective_extra_key"))
                runtime_cache_namespace_seen = True
            if has_value(trace_row.get("radix_key_extra_key")):
                radix_key_extra_key = radix_key_extra_key or str(trace_row.get("radix_key_extra_key"))
                runtime_cache_namespace_seen = True
            namespace = trace_row.get("cache_namespace")
            if isinstance(namespace, dict):
                if has_value(namespace.get("effective_extra_key")):
                    effective_extra_key = effective_extra_key or str(namespace.get("effective_extra_key"))
                    runtime_cache_namespace_seen = True
                radix_key = namespace.get("radix_key")
                if isinstance(radix_key, dict) and has_value(radix_key.get("extra_key")):
                    radix_key_extra_key = radix_key_extra_key or str(radix_key.get("extra_key"))
                    runtime_cache_namespace_seen = True

            if any(
                (
                    "match_prefix" in action_text,
                    category in {"init_load_back", "load_back", "hicache_load", "host_to_device_copy", "cache_finished_req"},
                    event.startswith("kv_telemetry.prefill"),
                )
            ):
                ts_ns = int(float_value(trace_row.get("ts_ns")))
                if ts_ns and sglang_receive_ts_ns and ts_ns >= sglang_receive_ts_ns:
                    if not first_cache_action_ts_ns or ts_ns < first_cache_action_ts_ns:
                        first_cache_action_ts_ns = ts_ns

            for attribution in matching_request_attributions(trace_row, request_id):
                update_max("cached_prefix_tokens", attribution.get("cached_prefix_tokens"))
                update_max("cached_prefix_tokens", attribution.get("prefill_cached_prefix_tokens"))
                update_max("cached_prefix_tokens", attribution.get("batch_request_cached_prefix_token_sum"))
                update_max("uncached_tokens", attribution.get("new_prefill_tokens_est"))
                update_max("uncached_tokens", attribution.get("prefill_uncached_token_count"))
                update_max("uncached_tokens", attribution.get("batch_request_uncached_token_sum"))
                update_max("host_hit_tokens", attribution.get("host_hit_tokens"))
                update_max("host_hit_tokens", attribution.get("prefill_host_hit_tokens"))
                update_max("host_load_tokens", attribution.get("host_load_tokens"))
                update_max("device_load_tokens", attribution.get("device_load_tokens"))
                update_max("cache_protected_tokens", attribution.get("cache_protected_tokens"))
                update_max("kv_committed_tokens", attribution.get("kv_committed_tokens"))

        native_cache_signal_seen = is_truthy_text(replay_row.get("harness_native_cache_signal_seen"))
        gateway_cache_lowered = is_truthy_text(replay_row.get("gateway_cache_lowered"))
        sglang_payload_cache_metadata_sent = gateway_cache_lowered and has_value(replay_row.get("gateway_cache_translation"))
        backend_cache_activity_seen = any(
            (
                cache_match_events,
                load_back_events,
                h2d_copy_events,
                cache_finished_events,
                prefill_events,
                max_values.get("cached_prefix_tokens", 0) > 0,
                max_values.get("host_hit_tokens", 0) > 0,
            )
        )
        if native_cache_signal_seen and sglang_payload_cache_metadata_sent and backend_cache_activity_seen:
            verdict = "cache metadata sent to SGLang and cache path acted"
            if runtime_cache_namespace_seen:
                causality_note = (
                    "Strong transport plus action proof for this target replay. Runtime telemetry also saw a cache namespace/extra_key "
                    "at the cache path, which is the strongest non-A/B proof that SGLang consumed a cache-routing field."
                )
            else:
                causality_note = (
                    "Strong transport plus action proof for this target replay. The current SGLang cache telemetry does not echo cache-control metadata, "
                    "so this still does not prove the signal caused the cache hit; normal prefix reuse can also use the same cache path."
                )
        elif native_cache_signal_seen and sglang_payload_cache_metadata_sent:
            verdict = "cache metadata sent to SGLang, but no cache action was observed"
            causality_note = "Gateway transport is proven for this target replay; backend cache work was not observed in the trace rows."
        elif native_cache_signal_seen:
            verdict = "harness cache signal seen, but gateway did not lower it"
            causality_note = "The harness emitted something cache-like, but it was not translated into backend cache metadata."
        elif backend_cache_activity_seen:
            verdict = "backend cache activity without harness cache signal"
            causality_note = "This is useful baseline evidence: SGLang can do normal prefix/cache work even without a harness cache signal."
        else:
            verdict = "no cache signal and no backend cache action observed"
            causality_note = "No target-scoped cache signal or cache-path work was visible in this trace."

        first_cache_action_after_receive_ms = (
            round(ns_to_ms_delta(sglang_receive_ts_ns, first_cache_action_ts_ns), 3)
            if ns_to_ms_delta(sglang_receive_ts_ns, first_cache_action_ts_ns) is not None
            else ""
        )
        proof_rows.append(
            {
                "harness_label": replay_row.get("harness_label", ""),
                "pressure_level_label": replay_row.get("pressure_level_label", ""),
                "mode_label": replay_row.get("mode_label", ""),
                "session_id": replay_row.get("session_id", ""),
                "request_id": request_id,
                "native_cache_signal_seen": "yes" if native_cache_signal_seen else "no",
                "gateway_cache_lowered": "yes" if gateway_cache_lowered else "no",
                "gateway_cache_salt": replay_row.get("gateway_cache_salt", ""),
                "sglang_payload_cache_metadata_sent": "yes" if sglang_payload_cache_metadata_sent else "no",
                "runtime_cache_namespace_seen": "yes" if runtime_cache_namespace_seen else "no",
                "effective_extra_key": effective_extra_key,
                "radix_key_extra_key": radix_key_extra_key,
                "cache_match_prefix_events": cache_match_events,
                "load_back_events": load_back_events,
                "h2d_copy_events": h2d_copy_events,
                "cache_finished_events": cache_finished_events,
                "prefill_attribution_events": prefill_events,
                "max_cached_prefix_tokens": int(max_values["cached_prefix_tokens"]) if max_values["cached_prefix_tokens"] else "",
                "max_uncached_tokens": int(max_values["uncached_tokens"]) if max_values["uncached_tokens"] else "",
                "max_host_hit_tokens": int(max_values["host_hit_tokens"]) if max_values["host_hit_tokens"] else "",
                "max_host_load_tokens": int(max_values["host_load_tokens"]) if max_values["host_load_tokens"] else "",
                "max_device_load_tokens": int(max_values["device_load_tokens"]) if max_values["device_load_tokens"] else "",
                "max_cache_protected_tokens": int(max_values["cache_protected_tokens"]) if max_values["cache_protected_tokens"] else "",
                "max_kv_committed_tokens": int(max_values["kv_committed_tokens"]) if max_values["kv_committed_tokens"] else "",
                "first_cache_action_after_sglang_receive_ms": first_cache_action_after_receive_ms,
                "first_token_lateness_ms": replay_row.get("first_token_lateness_ms", ""),
                "sglang_receive_to_first_token_ms": replay_row.get("sglang_receive_to_first_token_ms", ""),
                "verdict": verdict,
                "causality_note": causality_note,
                **{key: value for key, value in replay_row.items() if key.startswith("encoding_")},
                "case_id": replay_row.get("case_id", ""),
                "case_dir": replay_row.get("case_dir", ""),
            }
        )
    return proof_rows


ENCODING_COLUMNS = ["encoding_codec", "encoding_config_hash", "encoding_scope", "encoding_status",
                    "encoding_reason", "encoding_original_tokens", "encoding_encoded_tokens",
                    "encoding_candidate_tokens", "encoding_saved_tokens", "encoding_elapsed_ms",
                    "encoding_legend_tokens", "encoding_validation", "encoding_original_hash",
                    "encoding_encoded_hash", "encoding_tokenizer_id"]

RAW_COLUMNS = ENCODING_COLUMNS + ["prefill_full_input_tokens", "prefill_cached_prefix_tokens", "prefill_uncached_token_count"] + [
    "harness",
    "harness_label",
    "pressure_level",
    "pressure_level_label",
    "mode",
    "mode_label",
    "session_id",
    "request_id",
    "phase",
    "request_group",
    "has_replay_deadline",
    "first_token_lateness_ms",
    "replay_debt_ms",
    "due_to_request_start_ms",
    "due_to_sglang_receive_ms",
    "sglang_receive_to_first_token_ms",
    "ttft_ms",
    "sglang_priority",
    "experiment_priority_intent",
    "harness_input_priority_signal",
    "harness_input_priority_signal_source",
    "harness_emit_priority_signal",
    "harness_emit_priority_signal_source",
    "gateway_priority_translation",
    "gateway_priority_translation_source",
    "harness_native_cache_signal_seen",
    "harness_native_cache_signal",
    "harness_native_cache_signal_source",
    "harness_native_cache_identity_signal",
    "harness_native_cache_identity_source",
    "gateway_cache_translation",
    "gateway_cache_translation_source",
    "gateway_cache_lowered",
    "gateway_cache_salt",
    "gateway_cache_invented_signal",
    "controller_replay_rank",
    "controller_urgent_replay_count",
    "controller_priority_ladder",
    "first_token_source",
    "status",
    "error",
    "case_id",
    "case_dir",
    "replay_due_ts_ns",
    "request_start_ts_ns",
    "sglang_receive_ts_ns",
    "first_token_ts_ns",
    "request_end_ts_ns",
    "backend_receive_source",
]

SUMMARY_COLUMNS = ["encoding_codec", "encoding_config_hash", "encoding_scope", "requests", "failures"] + [
    "harness",
    "harness_label",
    "pressure_level",
    "pressure_level_label",
    "mode",
    "mode_label",
    "samples",
    "median_first_token_lateness_ms",
    "median_due_to_request_start_ms",
    "median_due_to_sglang_receive_ms",
    "median_ttft_ms",
    "median_sglang_receive_to_first_token_ms",
    "min_first_token_lateness_ms",
    "max_first_token_lateness_ms",
]

COST_ACCOUNTING_COLUMNS = [
    "encoding_codec",
    "encoding_config_hash",
    "encoding_scope",
    "harness",
    "harness_label",
    "pressure_level",
    "pressure_level_label",
    "mode",
    "mode_label",
    "signal_bucket",
    "signal_bucket_label",
    "target_request_count",
    "filler_request_count",
    "target_ttft_measured_requests",
    "filler_ttft_measured_requests",
    "sum_target_ttft_ms",
    "sum_filler_ttft_ms",
    "sum_total_ttft_ms",
    "target_replay_debt_measured_requests",
    "filler_replay_debt_measured_requests",
    "filler_replay_debt_unmeasured_requests",
    "sum_target_replay_debt_ms",
    "sum_filler_replay_debt_ms",
    "sum_total_replay_debt_ms",
]

SPECULATIVE_PREFILL_COLUMNS = [
    "harness_label",
    "pressure_level_label",
    "session_id",
    "warmup_request_id",
    "expected_replay_request_id",
    "hint_seen",
    "strategy",
    "warmup_started_before_replay_due",
    "warmup_completed_before_replay",
    "warmup_completed_before_sglang_receive",
    "warmup_prompt_tokens",
    "warmup_full_input_tokens",
    "warmup_uncached_tokens",
    "replay_cached_prefix_tokens",
    "replay_uncached_tokens",
    "replay_first_token_lateness_ms",
    "replay_backend_ms",
    "warmup_start_to_replay_due_ms",
    "warmup_end_to_replay_start_ms",
    "verdict",
    "case_id",
    "case_dir",
]

HARNESS_PRIORITY_COLUMNS = [
    "harness_label",
    "pressure_level_label",
    "session_id",
    "request_id",
    "driver_intent_seen",
    "gateway_saw_intent",
    "harness_input_signal",
    "harness_input_signal_source",
    "harness_output_signal",
    "harness_output_signal_source",
    "gateway_translated_priority",
    "gateway_translation_source",
    "sglang_priority_seen",
    "nat_wrapper_seen",
    "nat_priority_fields",
    "nat_priority_field_source",
    "nat_config_delay_ms",
    "nat_process_start_delay_ms",
    "nat_first_gateway_emit_delay_ms",
    "nat_process_exit_delay_ms",
    "nat_gateway_emit_seen",
    "nat_wrapper_total_ms",
    "harness_emit_delay_ms",
    "backend_ms",
    "first_token_lateness_ms",
    "verdict",
    "case_id",
    "case_dir",
]

NAT_SERVICE_PRIORITY_COLUMNS = [
    "nat_provider",
    "nat_dynamo_enable_nvext_hints",
    "request_id",
    "priority_class",
    "workflow_node",
    "workflow_node_goal",
    "inference_source",
    "expected_inferred_priority",
    "frontend_priority_intent_present",
    "submit_rank_into_nat",
    "emit_rank_from_nat_to_gateway",
    "older_background_submitted_before",
    "older_background_emitted_before",
    "submit_to_gateway_emit_ms",
    "gateway_to_fake_sglang_ms",
    "gateway_backend_ms",
    "client_latency_ms",
    "harness_input_signal",
    "harness_input_signal_source",
    "gateway_saw_marker_intent",
    "harness_output_signal",
    "harness_output_signal_source",
    "emitted_nvext_priority",
    "emitted_latency_sensitivity",
    "gateway_translated_priority",
    "gateway_translation_source",
    "sglang_priority_seen",
    "client_status",
    "client_error",
    "verdict",
    "case_id",
    "case_dir",
]

CACHE_SIGNAL_COLUMNS = [
    "harness_label",
    "pressure_level_label",
    "mode_label",
    "session_id",
    "request_id",
    "native_cache_signal_seen",
    "native_cache_signal",
    "native_cache_signal_source",
    "native_cache_identity_signal",
    "native_cache_identity_source",
    "gateway_cache_lowered",
    "gateway_cache_salt",
    "gateway_cache_translation",
    "gateway_cache_translation_source",
    "gateway_invented_signal",
    "first_token_lateness_ms",
    "sglang_receive_to_first_token_ms",
    "verdict",
    "case_id",
    "case_dir",
]

CACHE_ACTION_COLUMNS = ["encoding_codec", "encoding_config_hash", "encoding_scope"] + [
    "harness_label",
    "pressure_level_label",
    "mode_label",
    "session_id",
    "request_id",
    "native_cache_signal_seen",
    "gateway_cache_lowered",
    "gateway_cache_salt",
    "sglang_payload_cache_metadata_sent",
    "runtime_cache_namespace_seen",
    "effective_extra_key",
    "radix_key_extra_key",
    "cache_match_prefix_events",
    "load_back_events",
    "h2d_copy_events",
    "cache_finished_events",
    "prefill_attribution_events",
    "max_cached_prefix_tokens",
    "max_uncached_tokens",
    "max_host_hit_tokens",
    "max_host_load_tokens",
    "max_device_load_tokens",
    "max_cache_protected_tokens",
    "max_kv_committed_tokens",
    "first_cache_action_after_sglang_receive_ms",
    "first_token_lateness_ms",
    "sglang_receive_to_first_token_ms",
    "verdict",
    "causality_note",
    "case_id",
    "case_dir",
]

CACHE_BENEFIT_COLUMNS = ["encoding_codec", "encoding_config_hash", "encoding_scope"] + [
    "harness_label",
    "pressure_level_label",
    "samples_no_cache_signal",
    "samples_harness_native_cache_lowered",
    "native_cache_signal_seen",
    "gateway_cache_lowered",
    "gateway_cache_salt_seen",
    "runtime_cache_namespace_seen",
    "median_nc_backend_ttft_ms",
    "median_hc_backend_ttft_ms",
    "backend_ttft_delta_ms_hc_minus_nc",
    "backend_ttft_improvement_pct",
    "median_nc_ttft_ms",
    "median_hc_ttft_ms",
    "ttft_delta_ms_hc_minus_nc",
    "ttft_improvement_pct",
    "median_nc_first_token_lateness_ms",
    "median_hc_first_token_lateness_ms",
    "first_token_lateness_delta_ms_hc_minus_nc",
    "median_nc_cached_prefix_tokens",
    "median_hc_cached_prefix_tokens",
    "cached_prefix_delta_tokens",
    "median_nc_uncached_tokens",
    "median_hc_uncached_tokens",
    "uncached_delta_tokens_hc_minus_nc",
    "median_nc_cache_match_events",
    "median_hc_cache_match_events",
    "cache_match_delta_events",
    "median_nc_prefill_attribution_events",
    "median_hc_prefill_attribution_events",
    "prefill_attribution_delta_events",
    "verdict",
]


TARGETED_KV_PREFETCH_COLUMNS = [
    "harness_label",
    "pressure_level_label",
    "mode_label",
    "session_id",
    "request_id",
    "expected_replay_request_id",
    "controller_decision_id",
    "controller_command_id",
    "backend_name",
    "backend_accepted",
    "backend_acted",
    "direct_hook_available",
    "requested_before_replay_due",
    "requested_before_replay_start",
    "requested_to_replay_due_ms",
    "requested_to_replay_start_ms",
    "load_back_events_before_replay_compute",
    "h2d_copy_events_before_replay_compute",
    "first_movement_before_sglang_receive",
    "first_movement_before_replay_compute",
    "backend_reason",
    "verdict",
    "case_id",
    "case_dir",
]


CONTROLLER_DEMOTE_RESTORE_COLUMNS = [
    "harness_label",
    "pressure_level_label",
    "mode_label",
    "session_id",
    "demote_command_id",
    "demote_backend_acted",
    "demoted_priority",
    "filler_requests_seen",
    "filler_requests_between_demote_and_replay",
    "filler_demoted_count",
    "filler_not_demoted_count",
    "replay_request_id",
    "replay_sglang_priority",
    "replay_priority_raised",
    "controller_replay_rank",
    "controller_urgent_replay_count",
    "controller_priority_ladder",
    "restore_command_id",
    "restore_backend_acted",
    "demote_to_replay_due_ms",
    "demote_to_replay_start_ms",
    "replay_to_restore_ms",
    "first_token_lateness_ms",
    "verdict",
    "case_id",
    "case_dir",
]


CONTROLLER_ADMISSION_COLUMNS = [
    "harness_label",
    "pressure_level_label",
    "mode_label",
    "session_id",
    "admission_decision",
    "admission_reason",
    "prefetch_command_id",
    "budget_command_id",
    "admitted_warmups_before",
    "admitted_warmups_after",
    "max_warmups_per_case",
    "tool_wait_ms",
    "filler_sessions",
    "concurrency",
    "min_tool_wait_ms",
    "max_filler_sessions",
    "max_concurrency",
    "warmup_started",
    "warmup_completed",
    "warmup_failed",
    "replay_request_id",
    "replay_sglang_priority",
    "replay_priority_raised",
    "first_token_lateness_ms",
    "ttft_ms",
    "sglang_receive_to_first_token_ms",
    "verdict",
    "case_id",
    "case_dir",
]


SGLANG_CACHE_PATH_AUDIT_COLUMNS = [
    "signal",
    "expected_native_path",
    "parser_seen",
    "generate_req_input_seen",
    "req_field_seen",
    "radix_namespace_seen",
    "verdict",
    "source_evidence",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["harness"]), str(row["pressure_level"]), str(row["mode"]),
                 str(row.get("encoding_codec", "identity")), str(row.get("encoding_config_hash", "")),
                 str(row.get("encoding_scope", "")))].append(row)
    out: list[dict[str, Any]] = []
    for (harness, pressure, mode, codec, codec_hash, scope), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], PRESSURE_ORDER.index(item[0][1]) if item[0][1] in PRESSURE_ORDER else 999, item[0][2])):
        values = [float(row["first_token_lateness_ms"]) for row in group_rows if row.get("first_token_lateness_ms") != ""]
        due_to_request_start = [value for row in group_rows if (value := optional_float(row.get("due_to_request_start_ms"))) is not None]
        due_to_sglang_receive = [value for row in group_rows if (value := optional_float(row.get("due_to_sglang_receive_ms"))) is not None]
        ttft_values = [value for row in group_rows if (value := optional_float(row.get("ttft_ms"))) is not None]
        backend_values = [value for row in group_rows if (value := optional_float(row.get("sglang_receive_to_first_token_ms"))) is not None]
        out.append(
            {
                "harness": harness,
                "harness_label": HARNESS_LABELS.get(harness, harness),
                "pressure_level": pressure,
                "pressure_level_label": PRESSURE_LABELS.get(pressure, pressure),
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "encoding_codec": codec,
                "encoding_config_hash": codec_hash,
                "encoding_scope": scope,
                "requests": len(group_rows),
                "failures": sum(bool(row.get("error")) for row in group_rows),
                "samples": len(values),
                "median_first_token_lateness_ms": round(statistics.median(values), 3) if values else "",
                "median_due_to_request_start_ms": round(statistics.median(due_to_request_start), 3) if due_to_request_start else "",
                "median_due_to_sglang_receive_ms": round(statistics.median(due_to_sglang_receive), 3) if due_to_sglang_receive else "",
                "median_ttft_ms": round(statistics.median(ttft_values), 3) if ttft_values else "",
                "median_sglang_receive_to_first_token_ms": round(statistics.median(backend_values), 3) if backend_values else "",
                "min_first_token_lateness_ms": round(min(values), 3) if values else "",
                "max_first_token_lateness_ms": round(max(values), 3) if values else "",
            }
        )
    return out


def target_replay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("request_group") or "target") == "target"
        and str(row.get("phase") or "replay") == "replay"
    ]


def cost_signal_bucket(group_rows: list[dict[str, Any]], mode: str) -> str:
    target_rows = [row for row in group_rows if str(row.get("request_group") or "target") == "target"]
    candidate_rows = target_rows or group_rows
    for row in candidate_rows:
        bucket = chart_signal_bucket(row)
        if bucket != "baseline" or mode in {"no_prefetch", "no_cache_signal"}:
            return bucket
    return CHART_SIGNAL_BUCKETS.get(mode, {}).get("label", "") or chart_signal_bucket({"mode": mode})


def collect_cost_accounting_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("harness") or ""),
                str(row.get("pressure_level") or ""),
                str(row.get("mode") or ""),
                str(row.get("encoding_codec") or "identity"),
                str(row.get("encoding_config_hash") or ""),
                str(row.get("encoding_scope") or ""),
            )
        ].append(row)

    out: list[dict[str, Any]] = []
    for (harness, pressure, mode, codec, codec_hash, scope), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            HARNESS_LABELS.get(item[0][0], item[0][0]),
            PRESSURE_ORDER.index(item[0][1]) if item[0][1] in PRESSURE_ORDER else 999,
            item[0][2],
        ),
    ):
        target_rows = [row for row in group_rows if str(row.get("request_group") or "target") == "target"]
        filler_rows = [row for row in group_rows if str(row.get("request_group") or "target") == "filler"]
        target_ttft = [value for row in target_rows if (value := optional_float(row.get("ttft_ms"))) is not None]
        filler_ttft = [value for row in filler_rows if (value := optional_float(row.get("ttft_ms"))) is not None]
        def replay_debt_value(row: dict[str, Any]) -> float | None:
            value = optional_float(row.get("replay_debt_ms"))
            if value is not None:
                return value
            lateness = optional_float(row.get("first_token_lateness_ms"))
            if lateness is None:
                return None
            return max(lateness, 0.0)

        target_debt = [value for row in target_rows if (value := replay_debt_value(row)) is not None]
        filler_debt = [value for row in filler_rows if (value := replay_debt_value(row)) is not None]
        filler_debt_unmeasured = sum(
            1
            for row in filler_rows
            if optional_float(row.get("ttft_ms")) is not None and replay_debt_value(row) is None
        )
        bucket = cost_signal_bucket(group_rows, mode)
        sum_target_ttft = sum(target_ttft)
        sum_filler_ttft = sum(filler_ttft)
        sum_target_debt = sum(target_debt)
        sum_filler_debt = sum(filler_debt)
        out.append(
            {
                "encoding_codec": codec,
                "encoding_config_hash": codec_hash,
                "encoding_scope": scope,
                "harness": harness,
                "harness_label": HARNESS_LABELS.get(harness, harness),
                "pressure_level": pressure,
                "pressure_level_label": PRESSURE_LABELS.get(pressure, pressure),
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "signal_bucket": bucket,
                "signal_bucket_label": chart_signal_label(bucket),
                "target_request_count": len(target_rows),
                "filler_request_count": len(filler_rows),
                "target_ttft_measured_requests": len(target_ttft),
                "filler_ttft_measured_requests": len(filler_ttft),
                "sum_target_ttft_ms": round(sum_target_ttft, 3),
                "sum_filler_ttft_ms": round(sum_filler_ttft, 3),
                "sum_total_ttft_ms": round(sum_target_ttft + sum_filler_ttft, 3),
                "target_replay_debt_measured_requests": len(target_debt),
                "filler_replay_debt_measured_requests": len(filler_debt),
                "filler_replay_debt_unmeasured_requests": filler_debt_unmeasured,
                "sum_target_replay_debt_ms": round(sum_target_debt, 3),
                "sum_filler_replay_debt_ms": round(sum_filler_debt, 3),
                "sum_total_replay_debt_ms": round(sum_target_debt + sum_filler_debt, 3),
            }
        )
    return out


def median_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := optional_float(row.get(key))) is not None]
    if not values:
        return None
    return statistics.median(values)


def median_text(rows: list[dict[str, Any]], key: str) -> str:
    value = median_optional(rows, key)
    return f"{round(value, 3):g}" if value is not None else ""


def pct_improvement(baseline: float | None, candidate: float | None) -> str:
    if baseline is None or candidate is None or baseline <= 0:
        return ""
    return f"{round(((baseline - candidate) / baseline) * 100.0, 2):g}"


def delta_text(candidate: float | None, baseline: float | None) -> str:
    if candidate is None or baseline is None:
        return ""
    return f"{round(candidate - baseline, 3):g}"


def collect_cache_benefit_summary(
    summary_rows: list[dict[str, Any]],
    cache_action_rows: list[dict[str, Any]],
    _split_encoding: bool = True,
) -> list[dict[str, Any]]:
    if _split_encoding and any(row.get("encoding_config_hash") for row in summary_rows):
        fields = ("encoding_codec", "encoding_config_hash", "encoding_scope")
        def encoding_key(row):
            return tuple(str(row.get(field, "")) for field in fields)
        out = []
        for key in sorted({encoding_key(row) for row in summary_rows}):
            sub = collect_cache_benefit_summary(
                [row for row in summary_rows if encoding_key(row) == key],
                [row for row in cache_action_rows if encoding_key(row) == key], False)
            out.extend({**row, **dict(zip(fields, key))} for row in sub)
        return out
    summary_by_key = {
        (str(row.get("harness") or ""), str(row.get("pressure_level") or ""), str(row.get("mode") or "")): row
        for row in summary_rows
    }
    actions_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cache_action_rows:
        case_id = str(row.get("case_id") or "")
        harness, pressure, mode = case_key_from_name(case_id)
        if not harness or not pressure or not mode:
            continue
        actions_by_key[(harness, pressure, mode)].append(row)

    out: list[dict[str, Any]] = []
    pairs = sorted(
        {
            (harness, pressure)
            for harness, pressure, mode in summary_by_key
            if mode in {"no_cache_signal", "harness_native_cache_lowered"}
        },
        key=lambda item: (
            HARNESS_LABELS.get(item[0], item[0]),
            PRESSURE_ORDER.index(item[1]) if item[1] in PRESSURE_ORDER else 999,
        ),
    )
    for harness, pressure in pairs:
        nc_summary = summary_by_key.get((harness, pressure, "no_cache_signal"), {})
        hc_summary = summary_by_key.get((harness, pressure, "harness_native_cache_lowered"), {})
        if not nc_summary or not hc_summary:
            continue
        nc_actions = actions_by_key.get((harness, pressure, "no_cache_signal"), [])
        hc_actions = actions_by_key.get((harness, pressure, "harness_native_cache_lowered"), [])
        nc_backend = optional_float(nc_summary.get("median_sglang_receive_to_first_token_ms"))
        hc_backend = optional_float(hc_summary.get("median_sglang_receive_to_first_token_ms"))
        nc_ttft = optional_float(nc_summary.get("median_ttft_ms"))
        hc_ttft = optional_float(hc_summary.get("median_ttft_ms"))
        nc_lateness = optional_float(nc_summary.get("median_first_token_lateness_ms"))
        hc_lateness = optional_float(hc_summary.get("median_first_token_lateness_ms"))
        nc_cached = median_optional(nc_actions, "max_cached_prefix_tokens")
        hc_cached = median_optional(hc_actions, "max_cached_prefix_tokens")
        nc_uncached = median_optional(nc_actions, "max_uncached_tokens")
        hc_uncached = median_optional(hc_actions, "max_uncached_tokens")
        nc_match = median_optional(nc_actions, "cache_match_prefix_events")
        hc_match = median_optional(hc_actions, "cache_match_prefix_events")
        nc_prefill = median_optional(nc_actions, "prefill_attribution_events")
        hc_prefill = median_optional(hc_actions, "prefill_attribution_events")
        native_seen = any(is_truthy_text(row.get("native_cache_signal_seen")) for row in hc_actions)
        lowered = any(is_truthy_text(row.get("gateway_cache_lowered")) for row in hc_actions)
        salt_seen = any(has_value(row.get("gateway_cache_salt")) for row in hc_actions)
        namespace_seen = any(is_truthy_text(row.get("runtime_cache_namespace_seen")) for row in hc_actions)
        backend_delta = hc_backend - nc_backend if hc_backend is not None and nc_backend is not None else None
        cached_delta = hc_cached - nc_cached if hc_cached is not None and nc_cached is not None else None
        match_delta = hc_match - nc_match if hc_match is not None and nc_match is not None else None
        if backend_delta is not None and backend_delta < -50 and lowered:
            verdict = "HC improved backend TTFT; cache signal transport present"
            if namespace_seen:
                verdict += "; runtime namespace proof present"
            elif not salt_seen:
                verdict += "; no explicit cache_salt observed"
            else:
                verdict += "; direct namespace proof still missing"
        elif backend_delta is not None and backend_delta > 50 and lowered:
            verdict = "HC was slower on backend TTFT despite cache signal transport"
        elif lowered:
            verdict = "HC transport present, but backend TTFT change was small/noisy"
        elif native_seen:
            verdict = "Harness emitted cache signal, but gateway lowering was not proven"
        else:
            verdict = "No target native cache signal observed for HC"
        out.append(
            {
                "harness_label": HARNESS_LABELS.get(harness, harness),
                "pressure_level_label": PRESSURE_LABELS.get(pressure, pressure),
                "samples_no_cache_signal": nc_summary.get("samples", ""),
                "samples_harness_native_cache_lowered": hc_summary.get("samples", ""),
                "native_cache_signal_seen": "yes" if native_seen else "no",
                "gateway_cache_lowered": "yes" if lowered else "no",
                "gateway_cache_salt_seen": "yes" if salt_seen else "no",
                "runtime_cache_namespace_seen": "yes" if namespace_seen else "no",
                "median_nc_backend_ttft_ms": median_text([nc_summary], "median_sglang_receive_to_first_token_ms"),
                "median_hc_backend_ttft_ms": median_text([hc_summary], "median_sglang_receive_to_first_token_ms"),
                "backend_ttft_delta_ms_hc_minus_nc": delta_text(hc_backend, nc_backend),
                "backend_ttft_improvement_pct": pct_improvement(nc_backend, hc_backend),
                "median_nc_ttft_ms": median_text([nc_summary], "median_ttft_ms"),
                "median_hc_ttft_ms": median_text([hc_summary], "median_ttft_ms"),
                "ttft_delta_ms_hc_minus_nc": delta_text(hc_ttft, nc_ttft),
                "ttft_improvement_pct": pct_improvement(nc_ttft, hc_ttft),
                "median_nc_first_token_lateness_ms": median_text([nc_summary], "median_first_token_lateness_ms"),
                "median_hc_first_token_lateness_ms": median_text([hc_summary], "median_first_token_lateness_ms"),
                "first_token_lateness_delta_ms_hc_minus_nc": delta_text(hc_lateness, nc_lateness),
                "median_nc_cached_prefix_tokens": median_text(nc_actions, "max_cached_prefix_tokens"),
                "median_hc_cached_prefix_tokens": median_text(hc_actions, "max_cached_prefix_tokens"),
                "cached_prefix_delta_tokens": delta_text(hc_cached, nc_cached),
                "median_nc_uncached_tokens": median_text(nc_actions, "max_uncached_tokens"),
                "median_hc_uncached_tokens": median_text(hc_actions, "max_uncached_tokens"),
                "uncached_delta_tokens_hc_minus_nc": delta_text(hc_uncached, nc_uncached),
                "median_nc_cache_match_events": median_text(nc_actions, "cache_match_prefix_events"),
                "median_hc_cache_match_events": median_text(hc_actions, "cache_match_prefix_events"),
                "cache_match_delta_events": delta_text(hc_match, nc_match),
                "median_nc_prefill_attribution_events": median_text(nc_actions, "prefill_attribution_events"),
                "median_hc_prefill_attribution_events": median_text(hc_actions, "prefill_attribution_events"),
                "prefill_attribution_delta_events": delta_text(hc_prefill, nc_prefill),
                "verdict": verdict,
            }
        )
    return out


def collect_sglang_cache_path_audit(run_environment: dict[str, Any]) -> list[dict[str, Any]]:
    capabilities = run_environment.get("sglang_capabilities") if isinstance(run_environment, dict) else {}
    audit = capabilities.get("cache_signal_path_audit") if isinstance(capabilities, dict) else {}
    checks = audit.get("checks") if isinstance(audit, dict) else []
    if not isinstance(checks, list):
        return []
    return [row for row in checks if isinstance(row, dict)]


def symlog(value: float, linear_threshold: float = 50.0) -> float:
    sign = -1.0 if value < 0 else 1.0
    value = abs(value)
    if value <= linear_threshold:
        return sign * (value / linear_threshold)
    return sign * (1.0 + math.log10(value / linear_threshold))


def compact_ms(value: float) -> str:
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 1000:
        return f"{sign}{abs_value / 1000:.1f}s"
    return f"{sign}{abs_value:.0f}ms"


def svg_text_label(text: str, x: float, y: float, color: str, anchor: str = "middle") -> str:
    escaped = html.escape(text)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="9" '
        f'font-weight="700" fill="{color}" stroke="#ffffff" stroke-width="3" '
        f'paint-order="stroke" stroke-linejoin="round">{escaped}</text>'
    )


def signal_marker_style(bucket: str) -> str:
    return "solid"


def svg_symbol(kind: str, x: float, y: float, color: str, title: str, signal_style: str = "solid") -> str:
    escaped_title = html.escape(title)
    common = f'fill="{color}" stroke="{color}" stroke-width="2" opacity="0.9"'
    if kind == "square":
        shape = f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" rx="2" {common}/>'
    elif kind == "triangle":
        points = f"{x:.1f},{y-7:.1f} {x-6.5:.1f},{y+5.5:.1f} {x+6.5:.1f},{y+5.5:.1f}"
        shape = f'<polygon points="{points}" {common}/>'
    elif kind == "triangle-down":
        points = f"{x:.1f},{y+7:.1f} {x-6.5:.1f},{y-5.5:.1f} {x+6.5:.1f},{y-5.5:.1f}"
        shape = f'<polygon points="{points}" {common}/>'
    elif kind == "diamond":
        points = f"{x:.1f},{y-7:.1f} {x+7:.1f},{y:.1f} {x:.1f},{y+7:.1f} {x-7:.1f},{y:.1f}"
        shape = f'<polygon points="{points}" {common}/>'
    elif kind == "cross":
        shape = (
            f'<line x1="{x-6:.1f}" x2="{x+6:.1f}" y1="{y-6:.1f}" y2="{y+6:.1f}" {common}/>'
            f'<line x1="{x-6:.1f}" x2="{x+6:.1f}" y1="{y+6:.1f}" y2="{y-6:.1f}" {common}/>'
        )
    elif kind == "plus":
        shape = (
            f'<line x1="{x-7:.1f}" x2="{x+7:.1f}" y1="{y:.1f}" y2="{y:.1f}" {common}/>'
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y-7:.1f}" y2="{y+7:.1f}" {common}/>'
        )
    elif kind == "star":
        points = []
        for i in range(10):
            radius = 7 if i % 2 == 0 else 3.2
            angle = -math.pi / 2 + i * math.pi / 5
            points.append(f"{x + math.cos(angle) * radius:.1f},{y + math.sin(angle) * radius:.1f}")
        shape = f'<polygon points="{" ".join(points)}" {common}/>'
    elif kind == "hexagon":
        points = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            points.append(f"{x + math.cos(angle) * 7:.1f},{y + math.sin(angle) * 7:.1f}")
        shape = f'<polygon points="{" ".join(points)}" {common}/>'
    elif kind == "ring":
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.2" fill="#ffffff" stroke="{color}" stroke-width="2.4" opacity="0.95"/>'
    else:
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" {common}/>'
    return f'<g><title>{escaped_title}</title>{shape}</g>'


def inline_symbol(kind: str, color: str) -> str:
    return (
        '<svg class="legend-symbol" viewBox="0 0 24 24" aria-hidden="true">'
        f"{svg_symbol(kind, 12, 12, color, '')}"
        "</svg>"
    )


def inline_signal_symbol(bucket: str) -> str:
    color = chart_signal_color(bucket)
    return (
        '<svg class="legend-symbol" viewBox="0 0 24 24" aria-hidden="true">'
        f'{svg_symbol("circle", 12, 12, color, "", signal_marker_style(bucket))}'
        "</svg>"
    )


def chart_signal_bucket(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or "")
    if mode in {"no_prefetch", "no_cache_signal"}:
        return "baseline"
    if mode == "harness_native_cache_lowered":
        if (
            is_truthy_text(row.get("harness_native_cache_signal_seen"))
            and is_truthy_text(row.get("gateway_cache_lowered"))
            and str(row.get("gateway_cache_invented_signal") or "").strip().lower() != "true"
        ):
            return "harness_cache_emitted"
        return "baseline"
    if mode == "harness_emitted_signals":
        priority_lowered = has_value(row.get("harness_emit_priority_signal")) and has_value(row.get("sglang_priority"))
        cache_lowered = (
            is_truthy_text(row.get("harness_native_cache_signal_seen"))
            and is_truthy_text(row.get("gateway_cache_lowered"))
            and str(row.get("gateway_cache_invented_signal") or "").strip().lower() != "true"
        )
        if priority_lowered and cache_lowered:
            return "harness_cache_priority_emitted"
        if priority_lowered:
            return "harness_priority_emitted"
        if cache_lowered:
            return "harness_cache_emitted"
        return "baseline"
    if mode == "nat_inferred_priority_hints":
        if has_value(row.get("harness_emit_priority_signal")) and has_value(row.get("sglang_priority")):
            return "harness_priority_emitted"
        return "baseline"
    if mode == "pre_harness_priority_hints":
        if has_value(row.get("experiment_priority_intent")) or has_value(row.get("harness_input_priority_signal")):
            if has_value(row.get("gateway_priority_translation_source")) and has_value(row.get("sglang_priority")):
                return "frontend_supplied"
        return "baseline"
    if mode == "e2e_priority_hints":
        if has_value(row.get("sglang_priority")):
            return "gateway_priority_injected"
        return "baseline"
    if mode == "e2e_priority_hints_speculative_prefill":
        if has_value(row.get("sglang_priority")):
            return "gateway_speculative_prefill"
        return "baseline"
    if mode == "controller_observe_only":
        return "controller_observe"
    if mode == "controller_scheduler_priority":
        if has_value(row.get("sglang_priority")) and row.get("gateway_priority_translation_source") == "controller_ready_decision":
            return "controller_scheduler"
        return "baseline"
    if mode == "controller_speculative_preload":
        return "controller_preload"
    if mode == "controller_full":
        if has_value(row.get("sglang_priority")) and str(row.get("gateway_priority_translation_source") or "").startswith("controller_"):
            return "controller_full"
        return "baseline"
    for bucket, config in CHART_SIGNAL_BUCKETS.items():
        if mode in config["modes"]:
            return bucket
    return "baseline"


def chart_signal_label(bucket: str) -> str:
    return str(CHART_SIGNAL_BUCKETS.get(bucket, CHART_SIGNAL_BUCKETS["baseline"])["label"])


def chart_signal_color(bucket: str) -> str:
    return str(CHART_SIGNAL_BUCKETS.get(bucket, CHART_SIGNAL_BUCKETS["baseline"])["color"])


def linear_tick_values(values: list[float]) -> list[int]:
    if not values:
        return [0]
    v_min = min(values)
    v_max = max(values)
    top = max(v_max, 0.0)
    if top <= 1000:
        positive = [0, 50, 250, 500, 1000]
    elif top <= 10_000:
        positive = [0, 1000, 2500, 5000, 7500, 10_000]
    elif top <= 60_000:
        positive = [0, 10_000, 20_000, 40_000, 60_000]
    elif top <= 120_000:
        positive = [0, 20_000, 40_000, 60_000, 90_000, 120_000]
    else:
        positive = [0, 30_000, 60_000, 100_000, 150_000, 200_000]
    negative = [-1000, -500, -100] if v_min < 0 else []
    return [tick for tick in negative + positive if tick >= v_min * 1.05 and tick <= max(top * 1.12, 1000)]


def render_pressure_chart(rows: list[dict[str, Any]]) -> str:
    rows = target_replay_rows(rows)
    encoding_groups = defaultdict(list)
    for row in rows:
        encoding_groups[(row.get("encoding_codec", "identity"), row.get("encoding_config_hash", ""),
                         row.get("encoding_scope", ""))].append(row)
    if len(encoding_groups) > 1:
        return "".join(f"<h3>Encoding: {html.escape(str(key))}</h3>" + render_pressure_chart(group)
                       for key, group in sorted(encoding_groups.items()))
    pressures = [pressure for pressure in PRESSURE_ORDER if any(row["pressure_level"] == pressure for row in rows)]
    harnesses = [harness for harness in HARNESS_LABELS if any(row["harness"] == harness for row in rows)]
    signal_buckets = [
        bucket
        for bucket in CHART_SIGNAL_ORDER
        if any(chart_signal_bucket(row) == bucket for row in rows)
    ]
    if not pressures or not harnesses:
        return "<p>No replay rows found.</p>"

    pressure_w = max(760, len(harnesses) * 118 + 150)
    width = max(1400, pressure_w * len(pressures) + 220)
    left = 120
    right = 40
    panel_h = 310
    panel_gap = 250
    top_deadline = 82
    top_ttft = top_deadline + panel_h + panel_gap
    height = int(top_ttft + panel_h + 165)
    bottom_margin = 95
    plot_w = width - left - right
    pressure_group_w = plot_w / len(pressures)

    def signal_offset(bucket: str) -> float:
        if not signal_buckets:
            return 0.0
        try:
            index = signal_buckets.index(bucket)
        except ValueError:
            index = 0
        return (index - (len(signal_buckets) - 1) / 2) * 18.0

    def x_pos(pressure_index: int, harness_index: int, bucket: str) -> float:
        pressure_left = left + pressure_index * pressure_group_w
        harness_step = pressure_group_w / max(1, len(harnesses))
        base = pressure_left + harness_step * (harness_index + 0.5)
        return base + signal_offset(bucket)

    lines = [
        (
            f'<svg class="replay-pressure-chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'data-full-width="{width}" data-chart-height="{height}" data-left="{left}" data-right="{right}" '
            f'data-pressure-width="{pressure_group_w:.6f}" '
            f'data-pressure-order="{html.escape(json.dumps(pressures))}" '
            'role="img" aria-label="Replay Deadline Pressure Chart">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    rows_by_group_bucket: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = chart_signal_bucket(row)
        rows_by_group_bucket[(str(row["pressure_level"]), str(row["harness"]), bucket)].append(row)

    def append_panel_legend(panel_bottom: float) -> None:
        legend_y = panel_bottom + 88
        legend_x = left
        cursor = legend_x + 78
        lines.append(f'<text x="{legend_x:.1f}" y="{legend_y:.1f}" font-size="11" font-weight="800" fill="#111827">Legend</text>')
        for bucket in signal_buckets:
            color = chart_signal_color(bucket)
            label = chart_signal_label(bucket)
            marker_x = cursor + 8
            marker_y = legend_y - 4
            lines.append(f'<g class="signal-bucket legend-signal-item" data-signal-bucket="{html.escape(bucket)}">')
            lines.append(svg_symbol("circle", marker_x, marker_y, color, label, signal_marker_style(bucket)))
            text_x = cursor + 24
            lines.append(
                f'<text x="{text_x:.1f}" y="{legend_y:.1f}" font-size="10" font-weight="650" '
                f'fill="#334155">{html.escape(label)}</text>'
            )
            lines.append("</g>")
            cursor += max(104, len(label) * 6.1 + 42)
        style_y = legend_y + 24
        lines.append(
            f'<text x="{legend_x:.1f}" y="{style_y:.1f}" font-size="10" fill="#64748b">'
            'Each dot is the median replay for that harness and signal path. Labels show n= when multiple replay requests were summarized.'
            '</text>'
        )
        lines.append(f'<text x="{legend_x:.1f}" y="{style_y + 21:.1f}" font-size="10" fill="#64748b">Harness identity is the sub-window label on the x-axis.</text>')

    def draw_panel(
        panel_top: float,
        value_key: str,
        heading: str,
        note: str,
        y_axis_label: str,
        zero_label: str,
        unit_label: str,
        tick_values: list[int],
        scale: str = "symlog",
        group_class: str = "",
        data_axis: str = "",
    ) -> None:
        attributes = []
        if group_class:
            attributes.append(f'class="{html.escape(group_class)}"')
        if data_axis:
            attributes.append(f'data-axis="{html.escape(data_axis)}"')
        if attributes:
            lines.append(f'<g {" ".join(attributes)}>')
        panel_bottom = panel_top + panel_h
        panel_values = []
        for grouped_rows in rows_by_group_bucket.values():
            values = [value for row in grouped_rows if (value := optional_float(row.get(value_key))) is not None]
            if values:
                panel_values.append(statistics.median(values))
        if not panel_values:
            panel_values = [0.0]

        def transform(value: float) -> float:
            return symlog(value) if scale == "symlog" else value

        transformed = [transform(value) for value in panel_values + [float(tick) for tick in tick_values]]
        y_min = min(transformed)
        y_max = max(transformed)
        pad = max(0.2 if scale == "symlog" else 50.0, (y_max - y_min) * 0.08)
        y_min -= pad
        y_max += pad

        def y_pos_panel(value: float) -> float:
            mapped = transform(value)
            return panel_top + (y_max - mapped) / (y_max - y_min) * panel_h

        lines.append(f'<text x="{left}" y="{panel_top-46:.1f}" font-size="18" font-weight="800" fill="#111827">{html.escape(heading)}</text>')
        lines.append(f'<text x="{left}" y="{panel_top-24:.1f}" font-size="12" fill="#64748b">{html.escape(note)}</text>')
        for tick in tick_values:
            y = y_pos_panel(float(tick))
            stroke = "#111827" if tick == 0 else "#e5e7eb"
            width_attr = "1.5" if tick == 0 else "1"
            lines.append(f'<line class="chart-horizontal-span" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{width_attr}"/>')
            lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#374151">{tick} ms</text>')
        lines.append(f'<text class="chart-zero-label" x="{width-right-4}" y="{y_pos_panel(0)-8:.1f}" text-anchor="end" font-size="13" font-weight="700">{html.escape(zero_label)}</text>')

        for pressure_index, pressure in enumerate(pressures):
            x = left + pressure_index * pressure_group_w
            lines.append(
                f'<g class="pressure-level" data-pressure-level="{html.escape(pressure)}" '
                f'data-pressure-index="{pressure_index}">'
            )
            lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{panel_top}" y2="{panel_bottom+34:.1f}" stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="5 6"/>')
            if pressure_index % 2 == 1:
                lines.append(f'<rect x="{x:.1f}" y="{panel_top}" width="{pressure_group_w:.1f}" height="{panel_h}" fill="#f8fafc" opacity="0.62"/>')
            cx = x + pressure_group_w / 2
            lines.append(f'<text x="{cx:.1f}" y="{panel_bottom+56:.1f}" text-anchor="middle" font-size="16" font-weight="800" fill="#111827">{html.escape(PRESSURE_LABELS.get(pressure, pressure))}</text>')
            lines.append(f'<text x="{cx:.1f}" y="{panel_bottom+75:.1f}" text-anchor="middle" font-size="11" fill="#64748b">one sub-window per harness; color = signal path</text>')
            harness_step = pressure_group_w / max(1, len(harnesses))
            for harness_index, harness in enumerate(harnesses):
                harness_left = left + pressure_index * pressure_group_w + harness_step * harness_index
                harness_right = harness_left + harness_step
                harness_x = harness_left + harness_step / 2
                if harness_index % 2 == 1:
                    lines.append(
                        f'<rect x="{harness_left:.1f}" y="{panel_top}" width="{harness_step:.1f}" '
                        f'height="{panel_h + 34:.1f}" fill="#f8fafc" opacity="0.45"/>'
                    )
                lines.append(
                    f'<line x1="{harness_left:.1f}" x2="{harness_left:.1f}" y1="{panel_top}" y2="{panel_bottom+34:.1f}" '
                    f'stroke="#bfdbfe" stroke-width="1.6" stroke-dasharray="3 4"/>'
                )
                if harness_index == len(harnesses) - 1:
                    lines.append(
                        f'<line x1="{harness_right:.1f}" x2="{harness_right:.1f}" y1="{panel_top}" y2="{panel_bottom+34:.1f}" '
                        f'stroke="#bfdbfe" stroke-width="1.6" stroke-dasharray="3 4"/>'
                    )
                lines.append(
                    f'<text x="{harness_x:.1f}" y="{panel_bottom+24:.1f}" text-anchor="middle" '
                    f'font-size="10" font-weight="700" fill="#334155">'
                    f'{html.escape(HARNESS_SHORT_LABELS.get(harness, HARNESS_LABELS.get(harness, harness)))}</text>'
                )
                for bucket in signal_buckets:
                    sample_rows = rows_by_group_bucket.get((pressure, harness, bucket), [])
                    sample_rows = [row for row in sample_rows if optional_float(row.get(value_key)) is not None]
                    if not sample_rows:
                        continue
                    values = [float(row[value_key]) for row in sample_rows]
                    med = statistics.median(values)
                    color = chart_signal_color(bucket)
                    mx = x_pos(pressure_index, harness_index, bucket)
                    y = y_pos_panel(med)
                    label_y = y - 10 if bucket == "baseline" else y + 16
                    label_y = min(max(label_y, panel_top + 12), panel_bottom - 8)
                    label = compact_ms(med)
                    if len(sample_rows) > 1:
                        label = f"{label} n={len(sample_rows)}"
                    lines.append(f'<g class="signal-bucket" data-signal-bucket="{html.escape(bucket)}">')
                    lines.append(svg_text_label(label, mx, label_y, color))
                    raw_modes = sorted({str(row.get("mode") or "") for row in sample_rows})
                    title = (
                        f"{heading} | {PRESSURE_LABELS.get(pressure, pressure)} | "
                        f"{HARNESS_LABELS.get(harness, harness)} | "
                        f"{chart_signal_label(bucket)} | raw modes: {', '.join(MODE_LABELS.get(raw_mode, raw_mode) for raw_mode in raw_modes)} | "
                        f"median {med:.1f} {unit_label} | n={len(sample_rows)}"
                    )
                    lines.append(svg_symbol("circle", mx, y, color, title, "solid"))
                    lines.append("</g>")
            lines.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{panel_top}" y2="{panel_bottom+34:.1f}" '
                f'stroke="#94a3b8" stroke-width="2.1" stroke-dasharray="5 6"/>'
            )
            lines.append("</g>")
        lines.append(f'<line class="chart-right-boundary" x1="{width-right:.1f}" x2="{width-right:.1f}" y1="{panel_top}" y2="{panel_bottom+34:.1f}" stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="5 6"/>')
        lines.append(f'<text transform="translate(32 {panel_top + panel_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="14" font-weight="700">{html.escape(y_axis_label)}</text>')
        append_panel_legend(panel_bottom)
        if attributes:
            lines.append("</g>")

    draw_panel(
        top_deadline,
        "first_token_lateness_ms",
        "A. Replay Deadline Pressure (Symlog Axis)",
        "Delay view: every dot is one replay first token. Above zero missed the replay deadline; below zero was early.",
        "lateness vs replay deadline ms (symlog)",
        "0 ms deadline",
        "ms vs deadline",
        [-1000, -500, -100, 0, 50, 500, 1000, 5000, 10000, 60000],
        "symlog",
        "deadline-panel deadline-panel-symlog",
        "symlog",
    )
    draw_panel(
        top_deadline,
        "first_token_lateness_ms",
        "A. Replay Deadline Pressure (Linear Axis)",
        "Same replay-deadline data, but with a normal y-axis. This preserves true distance, though small values may bunch near the bottom.",
        "lateness vs replay deadline ms (linear)",
        "0 ms deadline",
        "ms vs deadline",
        linear_tick_values([value for row in rows if (value := optional_float(row.get("first_token_lateness_ms"))) is not None]),
        "linear",
        "deadline-panel deadline-panel-linear",
        "linear",
    )
    draw_panel(
        top_ttft,
        "ttft_ms",
        "B. Replay TTFT Impact",
        "Time from replay request start at the gateway/client boundary to first token. This is the clearest view for cache-control TTFT impact.",
        "replay TTFT ms (symlog)",
        "0 ms TTFT",
        "ms TTFT",
        [0, 50, 500, 1000, 5000, 10000, 60000],
        "symlog",
    )

    lines.append(f'<text class="chart-x-axis-label" x="{left + plot_w / 2:.1f}" y="{height-bottom_margin+34}" text-anchor="middle" font-size="14" font-weight="700">pressure level</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def render_cost_accounting_chart(
    cost_rows: list[dict[str, Any]],
    *,
    heading: str,
    note: str,
    target_key: str,
    filler_key: str,
    total_key: str,
    y_axis_label: str,
) -> str:
    if not cost_rows:
        return "<p>No cost accounting rows found.</p>"
    pressures = [pressure for pressure in PRESSURE_ORDER if any(row.get("pressure_level") == pressure for row in cost_rows)]
    harnesses = [harness for harness in HARNESS_LABELS if any(row.get("harness") == harness for row in cost_rows)]
    signal_buckets = [
        bucket
        for bucket in COST_ACCOUNTING_SIGNAL_BUCKETS
        if any(row.get("signal_bucket") == bucket for row in cost_rows)
    ]
    if not pressures or not harnesses or not signal_buckets:
        return "<p>No cost accounting rows found.</p>"

    pressure_w = max(820, len(harnesses) * 132 + 170)
    width = max(1400, pressure_w * len(pressures) + 220)
    left = 120
    right = 40
    top = 86
    chart_h = 330
    bottom = top + chart_h
    height = int(bottom + 145)
    plot_w = width - left - right
    pressure_group_w = plot_w / len(pressures)
    def measured_key(role: str) -> str:
        if "ttft" in target_key:
            return f"{role}_ttft_measured_requests"
        return f"{role}_replay_debt_measured_requests"

    def request_count_key(role: str) -> str:
        return f"{role}_request_count"

    def role_is_measured(row: dict[str, Any], role: str) -> bool:
        request_count = int(float(row.get(request_count_key(role)) or 0))
        measured = int(float(row.get(measured_key(role)) or 0))
        return request_count == 0 or measured > 0

    def row_total_is_measured(row: dict[str, Any]) -> bool:
        return role_is_measured(row, "target") and role_is_measured(row, "filler")

    def row_total_value(row: dict[str, Any]) -> float:
        return (optional_float(row.get(target_key)) or 0.0) + (optional_float(row.get(filler_key)) or 0.0)

    rows_by_key = {
        (str(row.get("pressure_level") or ""), str(row.get("harness") or ""), str(row.get("signal_bucket") or "")): row
        for row in cost_rows
    }
    delta_by_key: dict[tuple[str, str], float | None] = {}
    for pressure in pressures:
        for harness in harnesses:
            baseline_row = rows_by_key.get((pressure, harness, "baseline"))
            frontend_row = rows_by_key.get((pressure, harness, "frontend_supplied"))
            if baseline_row and frontend_row and row_total_is_measured(baseline_row) and row_total_is_measured(frontend_row):
                delta_by_key[(pressure, harness)] = row_total_value(baseline_row) - row_total_value(frontend_row)
            else:
                delta_by_key[(pressure, harness)] = None

    raw_values = []
    for row in cost_rows:
        for role, value_key in (("target", target_key), ("filler", filler_key)):
            if role_is_measured(row, role):
                raw_values.append(optional_float(row.get(value_key)) or 0.0)
    delta_values = [value for value in delta_by_key.values() if value is not None]
    max_positive = max([0.0, *raw_values, *[value for value in delta_values if value > 0]])
    min_negative = min([0.0, *[value for value in delta_values if value < 0]])
    if max_positive <= 0:
        max_positive = 1.0
    y_max = max_positive * 1.18
    y_min = min_negative * 1.18 if min_negative < 0 else 0.0
    if y_min == y_max:
        y_min = 0.0
        y_max = 1.0

    def y_pos(value: float) -> float:
        return bottom - ((value - y_min) / (y_max - y_min)) * chart_h

    def nice_ticks(bottom_value: float, top_value: float) -> list[float]:
        candidates = [0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
        positive_ticks = [tick for tick in candidates if tick <= top_value]
        if not positive_ticks or positive_ticks[-1] < top_value * 0.55:
            positive_ticks.append(top_value)
        negative_candidates = [-tick for tick in reversed(candidates[1:])]
        negative_ticks = [tick for tick in negative_candidates if bottom_value <= tick < 0]
        if bottom_value < 0 and (not negative_ticks or negative_ticks[0] > bottom_value * 0.55):
            negative_ticks.insert(0, bottom_value)
        return negative_ticks + positive_ticks

    def bucket_offset(bucket: str) -> float:
        index = signal_buckets.index(bucket)
        return (index - (len(signal_buckets) - 1) / 2) * 31.0

    def role_offset(role: str) -> float:
        return -6.5 if role == "target" else 6.5

    def bar_color(bucket: str, role: str) -> str:
        return COST_ACCOUNTING_COLORS.get(bucket, {}).get(role, chart_signal_color(bucket))

    lines = [
        (
            f'<svg class="replay-pressure-chart cost-accounting-chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'data-full-width="{width}" data-chart-height="{height}" data-left="{left}" data-right="{right}" '
            f'data-pressure-width="{pressure_group_w:.6f}" '
            f'data-pressure-order="{html.escape(json.dumps(pressures))}" '
            f'role="img" aria-label="{html.escape(heading)}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="{top-46:.1f}" font-size="18" font-weight="800" fill="#111827">{html.escape(heading)}</text>',
        f'<text x="{left}" y="{top-24:.1f}" font-size="12" fill="#64748b">{html.escape(note)}</text>',
    ]
    for tick in nice_ticks(y_min, y_max):
        y = y_pos(tick)
        lines.append(f'<line class="chart-horizontal-span" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#374151">{compact_ms(tick)}</text>')
    zero_y = y_pos(0.0)
    lines.append(f'<line class="chart-horizontal-span" x1="{left}" x2="{width-right}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="1.6"/>')

    for pressure_index, pressure in enumerate(pressures):
        x = left + pressure_index * pressure_group_w
        lines.append(
            f'<g class="pressure-level" data-pressure-level="{html.escape(pressure)}" '
            f'data-pressure-index="{pressure_index}">'
        )
        if pressure_index % 2 == 1:
            lines.append(f'<rect x="{x:.1f}" y="{top}" width="{pressure_group_w:.1f}" height="{chart_h}" fill="#f8fafc" opacity="0.62"/>')
        lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom+34:.1f}" stroke="#94a3b8" stroke-width="2.1" stroke-dasharray="5 6"/>')
        harness_step = pressure_group_w / max(1, len(harnesses))
        for harness_index, harness in enumerate(harnesses):
            harness_left = x + harness_step * harness_index
            harness_right = harness_left + harness_step
            harness_x = harness_left + harness_step / 2
            if harness_index % 2 == 1:
                lines.append(
                    f'<rect x="{harness_left:.1f}" y="{top}" width="{harness_step:.1f}" '
                    f'height="{chart_h + 34:.1f}" fill="#f8fafc" opacity="0.45"/>'
                )
            lines.append(
                f'<line x1="{harness_left:.1f}" x2="{harness_left:.1f}" y1="{top}" y2="{bottom+34:.1f}" '
                f'stroke="#bfdbfe" stroke-width="1.6" stroke-dasharray="3 4"/>'
            )
            if harness_index == len(harnesses) - 1:
                lines.append(
                    f'<line x1="{harness_right:.1f}" x2="{harness_right:.1f}" y1="{top}" y2="{bottom+34:.1f}" '
                    f'stroke="#bfdbfe" stroke-width="1.6" stroke-dasharray="3 4"/>'
                )
            lines.append(
                f'<text x="{harness_x:.1f}" y="{bottom+24:.1f}" text-anchor="middle" '
                f'font-size="10" font-weight="700" fill="#334155">'
                f'{html.escape(HARNESS_SHORT_LABELS.get(harness, HARNESS_LABELS.get(harness, harness)))}</text>'
            )
            for bucket in signal_buckets:
                row = rows_by_key.get((pressure, harness, bucket))
                if not row:
                    continue
                lines.append(f'<g class="signal-bucket" data-signal-bucket="{html.escape(bucket)}">')
                for role, value_key in (("target", target_key), ("filler", filler_key)):
                    value = optional_float(row.get(value_key)) or 0.0
                    measured = role_is_measured(row, role)
                    color = bar_color(bucket, role)
                    cx = harness_x + bucket_offset(bucket) + role_offset(role)
                    bar_w = 10.0
                    label_y = zero_y - 6 if zero_y > top + 28 else zero_y + 14
                    title_value = "n/a" if not measured else f"{value:.1f} ms"
                    title = (
                        f"{heading} | {PRESSURE_LABELS.get(pressure, pressure)} | "
                        f"{HARNESS_LABELS.get(harness, harness)} | {chart_signal_label(bucket)} | "
                        f"{role} {title_value}"
                    )
                    lines.append(f'<g><title>{html.escape(title)}</title>')
                    if not measured:
                        lines.append(
                            f'<line x1="{cx - bar_w / 2:.1f}" x2="{cx + bar_w / 2:.1f}" '
                            f'y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="{color}" stroke-width="2" opacity="0.75"/>'
                        )
                        lines.append(svg_text_label("n/a", cx, label_y, color))
                    else:
                        bar_top = y_pos(value)
                        lines.append(
                            f'<rect x="{cx - bar_w / 2:.1f}" y="{bar_top:.1f}" width="{bar_w:.1f}" '
                            f'height="{max(1.0, zero_y-bar_top):.1f}" fill="{color}" opacity="0.96" rx="2"/>'
                        )
                        lines.append(svg_text_label(compact_ms(value), cx, bar_top - 6, color))
                    lines.append("</g>")
                lines.append("</g>")
            delta = delta_by_key.get((pressure, harness))
            delta_cx = harness_x + 58.0
            delta_color = (
                COST_ACCOUNTING_DELTA_UNKNOWN
                if delta is None
                else COST_ACCOUNTING_DELTA_BETTER
                if delta >= 0
                else COST_ACCOUNTING_DELTA_WORSE
            )
            delta_title = (
                f"{heading} | {PRESSURE_LABELS.get(pressure, pressure)} | "
                f"{HARNESS_LABELS.get(harness, harness)} | net priority delta "
                f"(baseline total - front-end priority total) "
                f"{'n/a' if delta is None else f'{delta:.1f} ms'}"
            )
            lines.append(f'<g><title>{html.escape(delta_title)}</title>')
            if delta is None:
                delta_label_y = zero_y - 6 if zero_y > top + 28 else zero_y + 14
                lines.append(
                    f'<line x1="{delta_cx - 6:.1f}" x2="{delta_cx + 6:.1f}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" '
                    f'stroke="{delta_color}" stroke-width="2.2" opacity="0.8"/>'
                )
                lines.append(svg_text_label("net n/a", delta_cx, delta_label_y, delta_color))
            else:
                delta_y = y_pos(delta)
                bar_y = min(delta_y, zero_y)
                bar_h = max(1.0, abs(zero_y - delta_y))
                label = f"+{compact_ms(delta)} saved" if delta >= 0 else f"-{compact_ms(abs(delta))} worse"
                label_y_delta = delta_y - 6 if delta >= 0 else delta_y + 14
                lines.append(
                    f'<rect x="{delta_cx - 6:.1f}" y="{bar_y:.1f}" width="12" height="{bar_h:.1f}" '
                    f'fill="{delta_color}" opacity="0.96" rx="2"/>'
                )
                lines.append(svg_text_label(label, delta_cx, label_y_delta, delta_color))
            lines.append("</g>")
        cx = x + pressure_group_w / 2
        lines.append(f'<text x="{cx:.1f}" y="{bottom+56:.1f}" text-anchor="middle" font-size="16" font-weight="800" fill="#111827">{html.escape(PRESSURE_LABELS.get(pressure, pressure))}</text>')
        lines.append("</g>")
    lines.append(f'<line class="chart-right-boundary" x1="{width-right:.1f}" x2="{width-right:.1f}" y1="{top}" y2="{bottom+34:.1f}" stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="5 6"/>')
    lines.append(f'<text transform="translate(32 {top + chart_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="14" font-weight="700">{html.escape(y_axis_label)}</text>')
    legend_y = bottom + 91
    legend_x = left
    lines.append(f'<text x="{legend_x:.1f}" y="{legend_y:.1f}" font-size="11" font-weight="800" fill="#111827">Bars</text>')
    lines.append(f'<rect x="{legend_x+48:.1f}" y="{legend_y-10:.1f}" width="12" height="12" fill="{COST_ACCOUNTING_COLORS["baseline"]["target"]}" rx="2"/>')
    lines.append(f'<text x="{legend_x+66:.1f}" y="{legend_y:.1f}" font-size="11" fill="#334155">baseline target</text>')
    lines.append(f'<rect x="{legend_x+158:.1f}" y="{legend_y-10:.1f}" width="12" height="12" fill="{COST_ACCOUNTING_COLORS["baseline"]["filler"]}" rx="2"/>')
    lines.append(f'<text x="{legend_x+176:.1f}" y="{legend_y:.1f}" font-size="11" fill="#334155">baseline filler</text>')
    lines.append(f'<rect x="{legend_x+266:.1f}" y="{legend_y-10:.1f}" width="12" height="12" fill="{COST_ACCOUNTING_COLORS["frontend_supplied"]["target"]}" rx="2"/>')
    lines.append(f'<text x="{legend_x+284:.1f}" y="{legend_y:.1f}" font-size="11" fill="#334155">front-end priority target</text>')
    lines.append(f'<rect x="{legend_x+448:.1f}" y="{legend_y-10:.1f}" width="12" height="12" fill="{COST_ACCOUNTING_COLORS["frontend_supplied"]["filler"]}" rx="2"/>')
    lines.append(f'<text x="{legend_x+466:.1f}" y="{legend_y:.1f}" font-size="11" fill="#334155">front-end priority filler</text>')
    lines.append(f'<rect x="{legend_x+626:.1f}" y="{legend_y-10:.1f}" width="12" height="12" fill="{COST_ACCOUNTING_DELTA_BETTER}" rx="2"/>')
    lines.append(f'<rect x="{legend_x+642:.1f}" y="{legend_y-10:.1f}" width="12" height="12" fill="{COST_ACCOUNTING_DELTA_WORSE}" rx="2"/>')
    lines.append(f'<text x="{legend_x+662:.1f}" y="{legend_y:.1f}" font-size="11" fill="#334155">net delta: green saved, red worse</text>')
    lines.append(f'<text class="chart-x-axis-label" x="{left + plot_w / 2:.1f}" y="{height-10}" text-anchor="middle" font-size="14" font-weight="700">pressure level</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def render_cost_accounting_section(cost_rows: list[dict[str, Any]]) -> str:
    if not cost_rows:
        return '<h2>System Cost Accounting</h2><p>No target/filler cost accounting rows found.</p>'
    cost_table = render_table(
        cost_rows,
        [
            "harness_label",
            "pressure_level_label",
            "signal_bucket_label",
            "target_request_count",
            "filler_request_count",
            "sum_target_ttft_ms",
            "sum_filler_ttft_ms",
            "sum_total_ttft_ms",
            "sum_target_replay_debt_ms",
            "sum_filler_replay_debt_ms",
            "sum_total_replay_debt_ms",
            "filler_replay_debt_unmeasured_requests",
        ],
    )
    ttft_chart = render_cost_accounting_chart(
        cost_rows,
        heading="C. Total Replay TTFT Cost",
        note="Five bars per harness: baseline target/filler, front-end priority target/filler, then net delta = baseline total - priority total.",
        target_key="sum_target_ttft_ms",
        filler_key="sum_filler_ttft_ms",
        total_key="sum_total_ttft_ms",
        y_axis_label="total TTFT ms",
    )
    debt_chart = render_cost_accounting_chart(
        cost_rows,
        heading="D. Total Replay Deadline Debt",
        note="Same five-bar layout, but summing positive replay lateness. Upward net delta means priority reduced total replay debt.",
        target_key="sum_target_replay_debt_ms",
        filler_key="sum_filler_replay_debt_ms",
        total_key="sum_total_replay_debt_ms",
        y_axis_label="total replay debt ms",
    )
    return (
        "<h2>System Cost Accounting</h2>"
        "<p>This section asks whether priority/controller reduced target replay misses by moving delay onto filler/background work.</p>"
        "<p>The manager-facing cost charts intentionally compare only Baseline and Front-End Supplied. The fifth bar shows whether priority reduced or increased total system cost.</p>"
        f'<div class="card">{ttft_chart}</div>'
        f'<div class="card">{debt_chart}</div>'
        "<details><summary>Open cost accounting summary table</summary>"
        f'<div class="card">{cost_table}</div>'
        "</details>"
    )


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_lines = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body_lines.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_lines)}</tbody></table>"


def read_nat_inferred_priority_profile(report_dir: Path) -> dict[str, Any]:
    profile_path = report_dir / "nat_inferred_priority_profile.json"
    if not profile_path.exists():
        return {}
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def render_nat_inferred_priority_profile(profile: dict[str, Any]) -> str:
    nodes = profile.get("workflow_nodes")
    if not isinstance(nodes, list) or not nodes:
        return ""
    rows: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        rows.append(
            {
                "workflow_node": node.get("workflow_node", ""),
                "workflow_path": " / ".join(str(part) for part in node.get("workflow_path", []))
                if isinstance(node.get("workflow_path"), list)
                else node.get("workflow_path", ""),
                "workflow_node_goal": node.get("workflow_node_goal", ""),
                "latency_sensitivity": node.get("latency_sensitivity", ""),
                "expected_emitted_nvext_priority": node.get("expected_emitted_nvext_priority", ""),
            }
        )
    if not rows:
        return ""
    artifact_name = "nat_inferred_priority_profile.json"
    return (
        f"<p>Standalone profile artifact: <code>{artifact_name}</code>. "
        "These workflow-path entries are the source of the inferred priorities used by this NAT probe.</p>"
        + render_table(
            rows,
            [
                "workflow_node",
                "workflow_path",
                "workflow_node_goal",
                "latency_sensitivity",
                "expected_emitted_nvext_priority",
            ],
        )
    )


RUN_CONFIG_PRESSURE_KEYS = {
    "p0_control": "P0_CONTROL",
    "p1_mild": "P1_MILD",
    "p2_medium": "P2_MEDIUM",
    "p3_high": "P3_QUEUE_PRESSURE",
    "p4_cliff": "P4_CLIFF",
    "p5_boss_queue": "P5_BOSS_QUEUE",
}


def render_pressure_definition_table(rows: list[dict[str, Any]], run_config: dict[str, str]) -> str:
    present = {str(row["pressure_level"]) for row in rows}
    definition_rows: list[dict[str, Any]] = []
    for pressure in PRESSURE_ORDER:
        definition = PRESSURE_DEFINITIONS[pressure]
        knobs = run_config.get(RUN_CONFIG_PRESSURE_KEYS[pressure]) or definition["knobs"]
        definition_rows.append(
            {
                "level": PRESSURE_LABELS[pressure],
                "in_this_run": "Yes" if pressure in present else "No",
                "what_it_means": definition["goal"],
                "knobs": knobs,
            }
        )
    return render_table(definition_rows, ["level", "in_this_run", "what_it_means", "knobs"])


def render_signal_family_definition_table() -> str:
    return render_table(
        SIGNAL_FAMILY_DEFINITIONS,
        ["family", "where_signal_is_added", "what_it_means", "raw_modes"],
    )


def render_harness_cache_decision_table() -> str:
    rows = [
        {
            "Harness": "Claude Code",
            "Did target replay emit cache signal?": "Yes",
            "What signal appeared": "cache_control on system and messages blocks",
            "How the harness likely decided": "Structured prompt-builder rule",
            "Simple explanation": "Claude Code knows some blocks are stable, like system instructions, tool definitions, and earlier context. It marks those blocks cacheable.",
        },
        {
            "Harness": "Qwen Code",
            "Did target replay emit cache signal?": "Yes",
            "What signal appeared": "cache_control on system, messages, and tools blocks",
            "How the harness likely decided": "Cache-control setting plus structured prompt-builder rule",
            "Simple explanation": "Qwen Code was configured with cache control enabled. It then marked stable prompt sections such as system/tool/context blocks.",
        },
        {
            "Harness": "OpenCode",
            "Did target replay emit cache signal?": "Yes",
            "What signal appeared": "promptCacheKey",
            "How the harness likely decided": "Provider/session cache-key rule",
            "Simple explanation": "OpenCode was configured to always attach a cache key for that provider/session. It did not inspect the request deeply.",
        },
        {
            "Harness": "Codex",
            "Did target replay emit cache signal?": "Yes",
            "What signal appeared": "prompt_cache_key",
            "How the harness likely decided": "Session/provider cache-key rule",
            "Simple explanation": "Codex attached a stable cache key so related requests in the same task/session can be grouped for cache reuse.",
        },
        {
            "Harness": "Pi Agent Harness",
            "Did target replay emit cache signal?": "Yes",
            "What signal appeared": "cache_control, prompt_cache_key, prompt_cache_retention",
            "How the harness likely decided": "Session affinity plus cache compatibility rule",
            "Simple explanation": 'Pi carried both block-style cache markers and a session cache identity. It was mainly saying: "this session has reusable context."',
        },
        {
            "Harness": "OpenClaw",
            "Did target replay emit cache signal?": "Yes",
            "What signal appeared": "prompt_cache_key, prompt_cache_retention",
            "How the harness likely decided": "Session/provider cache-key rule",
            "Simple explanation": "OpenClaw attached a stable cache key and retention hint because the provider config said prompt cache keys and long retention are supported.",
        },
        {
            "Harness": "NeMo Agent Toolkit / NAT",
            "Did target replay emit cache signal?": "Yes",
            "What signal appeared": "nvext.cache_control",
            "How the harness likely decided": "Workflow/transport cache policy",
            "Simple explanation": "NAT's Dynamo-style transport was configured to always emit cache control for that workflow path.",
        },
        {
            "Harness": "DeepAgents / Hatcher",
            "Did target replay emit cache signal?": "No",
            "What signal appeared": "None seen",
            "How the harness likely decided": "No cache-emitting rule observed",
            "Simple explanation": "For the target replay rows, we did not see native cache metadata from this harness.",
        },
        {
            "Harness": "Hermes Agent",
            "Did target replay emit cache signal?": "No",
            "What signal appeared": "None seen",
            "How the harness likely decided": "No cache-emitting rule observed",
            "Simple explanation": "For the target replay rows, we did not see native cache metadata from this harness.",
        },
    ]
    grouping_rows = [
        {
            "Group": "Structured prompt block caching",
            "Harnesses": "Claude Code, Qwen Code, Pi",
            "Meaning": "The harness marks stable pieces of the prompt, like system instructions, tools, or context.",
        },
        {
            "Group": "Session cache key caching",
            "Harnesses": "Codex, OpenCode, OpenClaw, Pi",
            "Meaning": "The harness attaches a stable key so related requests can be associated with the same cache/session.",
        },
        {
            "Group": "Workflow transport cache policy",
            "Harnesses": "NAT",
            "Meaning": "The transport/workflow layer is configured to emit cache control.",
        },
        {
            "Group": "No cache signal observed",
            "Harnesses": "DeepAgents/Hatcher, Hermes",
            "Meaning": "No target replay cache signal was seen in this run.",
        },
    ]
    return "\n".join(
        [
            render_table(
                rows,
                [
                    "Harness",
                    "Did target replay emit cache signal?",
                    "What signal appeared",
                    "How the harness likely decided",
                    "Simple explanation",
                ],
            ),
            '<p class="muted">The simplest way to group them:</p>',
            render_table(grouping_rows, ["Group", "Harnesses", "Meaning"]),
            (
                "<p><strong>Main takeaway:</strong> These harnesses mostly emit cache signals because their prompt builder, "
                'provider config, session config, or workflow transport says "this request has reusable context." '
                "They are not generally deciding that the request is urgent.</p>"
            ),
        ]
    )


def render_chart_legend(rows: list[dict[str, Any]]) -> str:
    signal_items = []
    for bucket in CHART_SIGNAL_ORDER:
        config = CHART_SIGNAL_BUCKETS[bucket]
        if not any(chart_signal_bucket(row) == bucket for row in rows):
            continue
        signal_items.append(
            '<span class="legend-item">'
            f'{inline_signal_symbol(bucket)}'
            f'{html.escape(str(config["label"]))} <span class="muted">= {html.escape(str(config["description"]))}</span>'
            "</span>"
        )
    return (
        '<div class="legend-card">'
        '<div class="legend-row"><strong>Signal color</strong>'
        f'<div class="legend-items">{"".join(signal_items)}</div></div>'
        '<div class="legend-row"><strong>Dot meaning</strong>'
        '<div class="legend-items"><span class="legend-item">Each dot is the median replay for one harness and signal path; n= means multiple replay requests were summarized.</span></div></div>'
        '<div class="legend-row"><strong>Harness</strong>'
        '<div class="legend-items"><span class="legend-item">Harness identity is shown by the x-axis sub-window label; marker shape is no longer used for harness identity.</span></div></div>'
        "</div>"
    )


def present_signal_buckets(rows: list[dict[str, Any]]) -> list[str]:
    return [
        bucket
        for bucket in CHART_SIGNAL_ORDER
        if any(chart_signal_bucket(row) == bucket for row in rows)
    ]


def present_pressure_levels(rows: list[dict[str, Any]]) -> list[str]:
    return [
        pressure
        for pressure in PRESSURE_ORDER
        if any(str(row.get("pressure_level") or "") == pressure for row in rows)
    ]


def render_chart_controls(rows: list[dict[str, Any]]) -> str:
    signal_buckets = present_signal_buckets(rows)
    pressure_levels = present_pressure_levels(rows)
    default_visible = {bucket for bucket in signal_buckets if bucket in MANAGER_SIGNAL_BUCKETS}
    if not default_visible:
        default_visible = set(signal_buckets)
    signal_controls = []
    for bucket in signal_buckets:
        config = CHART_SIGNAL_BUCKETS[bucket]
        checked = " checked" if bucket in default_visible else ""
        signal_controls.append(
            '<label class="chart-chip" title="{description}">'
            '<input type="checkbox" data-signal-filter value="{bucket}"{checked}>'
            '<span class="chip-dot" style="background:{color}"></span>'
            '<span>{label}</span>'
            "</label>".format(
                bucket=html.escape(bucket),
                checked=checked,
                color=html.escape(str(config["color"])),
                label=html.escape(str(config["label"])),
                description=html.escape(str(config["description"])),
            )
        )
    pressure_controls = []
    for pressure in pressure_levels:
        pressure_controls.append(
            '<label class="chart-chip" title="{description}">'
            '<input type="checkbox" data-pressure-filter value="{pressure}" checked>'
            '<span>{label}</span>'
            "</label>".format(
                pressure=html.escape(pressure),
                label=html.escape(PRESSURE_LABELS.get(pressure, pressure)),
                description=html.escape(PRESSURE_DEFINITIONS.get(pressure, {}).get("goal", "")),
            )
        )
    return (
        '<div class="chart-controls" aria-label="Replay chart controls">'
        '<div class="control-row">'
        '<strong>View</strong>'
        '<button type="button" class="control-button" data-chart-preset="manager">Manager View</button>'
        '<button type="button" class="control-button" data-chart-preset="all">Show All</button>'
        '<span class="control-hint">Manager View keeps baseline, harness cache, harness cache + priority, front-end supplied, and full controller visible.</span>'
        "</div>"
        '<div class="control-row">'
        '<strong>Deadline axis</strong>'
        '<label class="chart-chip"><input type="radio" name="deadline-axis" value="linear" checked> Linear</label>'
        '<label class="chart-chip"><input type="radio" name="deadline-axis" value="symlog"> Symlog</label>'
        '<span class="control-hint">Linear shows true distance; symlog compresses very large misses so small misses stay visible.</span>'
        "</div>"
        '<div class="control-row">'
        '<strong>Signals</strong>'
        f'<div class="chip-list">{"".join(signal_controls)}</div>'
        "</div>"
        '<div class="control-row">'
        '<strong>Pressure levels</strong>'
        '<button type="button" class="control-button" data-pressure-preset="core">Core P0/P3/P5</button>'
        '<button type="button" class="control-button" data-pressure-preset="all">All Pressures</button>'
        f'<div class="chip-list">{"".join(pressure_controls)}</div>'
        "</div>"
        "</div>"
    )


def render_chart_interaction_script() -> str:
    manager_buckets = json.dumps(list(MANAGER_SIGNAL_BUCKETS))
    core_pressures = json.dumps(list(CORE_PRESSURE_LEVELS))
    return f"""
<script>
(function () {{
  const managerBuckets = new Set({manager_buckets});
  const corePressures = new Set({core_pressures});
  const filterInputs = Array.from(document.querySelectorAll('[data-signal-filter]'));
  const pressureInputs = Array.from(document.querySelectorAll('[data-pressure-filter]'));
  const axisInputs = Array.from(document.querySelectorAll('input[name="deadline-axis"]'));

  function selectedSignals() {{
    return new Set(filterInputs.filter((input) => input.checked).map((input) => input.value));
  }}

  function applySignalFilters() {{
    const selected = selectedSignals();
    document.querySelectorAll('[data-signal-bucket]').forEach((node) => {{
      const bucket = node.getAttribute('data-signal-bucket');
      node.style.display = selected.has(bucket) ? '' : 'none';
    }});
  }}

  function selectedPressures() {{
    return new Set(pressureInputs.filter((input) => input.checked).map((input) => input.value));
  }}

  function applyPressureFilters() {{
    let selected = selectedPressures();
    if (selected.size === 0) {{
      pressureInputs.forEach((input) => {{ input.checked = true; }});
      selected = selectedPressures();
    }}
    document.querySelectorAll('.replay-pressure-chart').forEach((svg) => {{
      const pressureOrder = JSON.parse(svg.getAttribute('data-pressure-order') || '[]');
      const visiblePressures = pressureOrder.filter((pressure) => selected.has(pressure));
      const pressureWidth = Number(svg.getAttribute('data-pressure-width') || 0);
      const left = Number(svg.getAttribute('data-left') || 0);
      const right = Number(svg.getAttribute('data-right') || 0);
      const height = Number(svg.getAttribute('data-chart-height') || svg.getAttribute('height') || 0);
      const activeCount = Math.max(1, visiblePressures.length);
      const activeRight = left + pressureWidth * activeCount;
      const activeWidth = activeRight + right;

      svg.setAttribute('viewBox', `0 0 ${{activeWidth}} ${{height}}`);
      svg.setAttribute('width', String(activeWidth));

      svg.querySelectorAll('[data-pressure-level]').forEach((node) => {{
        const pressure = node.getAttribute('data-pressure-level');
        const originalIndex = Number(node.getAttribute('data-pressure-index') || 0);
        const visibleIndex = visiblePressures.indexOf(pressure);
        if (visibleIndex === -1) {{
          node.style.display = 'none';
          node.removeAttribute('transform');
          return;
        }}
        node.style.display = 'inline';
        const dx = (visibleIndex - originalIndex) * pressureWidth;
        node.setAttribute('transform', `translate(${{dx}} 0)`);
      }});

      svg.querySelectorAll('.chart-horizontal-span').forEach((node) => {{
        node.setAttribute('x2', String(activeRight));
      }});
      svg.querySelectorAll('.chart-right-boundary').forEach((node) => {{
        node.setAttribute('x1', String(activeRight));
        node.setAttribute('x2', String(activeRight));
      }});
      svg.querySelectorAll('.chart-zero-label').forEach((node) => {{
        node.setAttribute('x', String(activeRight - 4));
      }});
      svg.querySelectorAll('.chart-x-axis-label').forEach((node) => {{
        node.setAttribute('x', String(left + (pressureWidth * activeCount) / 2));
      }});
    }});
  }}

  function applyAxis() {{
    const selected = (axisInputs.find((input) => input.checked) || {{ value: 'linear' }}).value;
    document.querySelectorAll('.deadline-panel').forEach((node) => {{
      node.style.display = node.getAttribute('data-axis') === selected ? 'inline' : 'none';
    }});
  }}

  function setPreset(name) {{
    const available = filterInputs.map((input) => input.value);
    let desired = name === 'manager'
      ? new Set(available.filter((bucket) => managerBuckets.has(bucket)))
      : new Set(available);
    if (desired.size === 0) desired = new Set(available);
    filterInputs.forEach((input) => {{ input.checked = desired.has(input.value); }});
    applySignalFilters();
  }}

  function setPressurePreset(name) {{
    const available = pressureInputs.map((input) => input.value);
    let desired = name === 'core'
      ? new Set(available.filter((pressure) => corePressures.has(pressure)))
      : new Set(available);
    if (desired.size === 0) desired = new Set(available);
    pressureInputs.forEach((input) => {{ input.checked = desired.has(input.value); }});
    applyPressureFilters();
  }}

  filterInputs.forEach((input) => input.addEventListener('change', applySignalFilters));
  pressureInputs.forEach((input) => input.addEventListener('change', applyPressureFilters));
  axisInputs.forEach((input) => input.addEventListener('change', applyAxis));
  document.querySelectorAll('[data-chart-preset]').forEach((button) => {{
    button.addEventListener('click', () => setPreset(button.getAttribute('data-chart-preset')));
  }});
  document.querySelectorAll('[data-pressure-preset]').forEach((button) => {{
    button.addEventListener('click', () => setPressurePreset(button.getAttribute('data-pressure-preset')));
  }});

  applySignalFilters();
  applyPressureFilters();
  applyAxis();
}}());
</script>
"""


def render_html(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    cost_accounting_rows: list[dict[str, Any]],
    speculative_prefill_rows: list[dict[str, Any]],
    targeted_kv_prefetch_rows: list[dict[str, Any]],
    controller_demote_restore_rows: list[dict[str, Any]],
    controller_admission_rows: list[dict[str, Any]],
    harness_priority_rows: list[dict[str, Any]],
    nat_service_priority_rows: list[dict[str, Any]],
    cache_signal_rows: list[dict[str, Any]],
    cache_action_rows: list[dict[str, Any]],
    cache_benefit_rows: list[dict[str, Any]],
    sglang_cache_path_audit_rows: list[dict[str, Any]],
    nat_inferred_priority_profile: dict[str, Any],
    report_label: str,
    run_config: dict[str, str],
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    hardware_profile = os.environ.get("HARDWARE_PROFILE") or run_config.get("HARDWARE_PROFILE") or "not recorded"
    hardware_profile_path = os.environ.get("HARDWARE_PROFILE_PATH") or run_config.get("HARDWARE_PROFILE_PATH") or "not recorded"
    chart = render_pressure_chart(rows)
    chart_controls = render_chart_controls(rows)
    chart_interaction_script = render_chart_interaction_script()
    cost_accounting_section = render_cost_accounting_section(cost_accounting_rows)
    signal_family_definition_table = render_signal_family_definition_table()
    harness_cache_decision_table = render_harness_cache_decision_table()
    pressure_definition_table = render_pressure_definition_table(rows, run_config)
    summary_table = render_table(
        summary,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "encoding_codec",
            "encoding_config_hash",
            "samples",
            "median_first_token_lateness_ms",
            "median_ttft_ms",
            "median_sglang_receive_to_first_token_ms",
            "median_due_to_sglang_receive_ms",
            "min_first_token_lateness_ms",
            "max_first_token_lateness_ms",
        ],
    )
    breakdown_table = render_table(
        summary,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "encoding_codec",
            "encoding_config_hash",
            "samples",
            "median_due_to_request_start_ms",
            "median_due_to_sglang_receive_ms",
            "median_ttft_ms",
            "median_sglang_receive_to_first_token_ms",
            "median_first_token_lateness_ms",
        ],
    )
    speculative_prefill_table = render_table(speculative_prefill_rows, SPECULATIVE_PREFILL_COLUMNS)
    targeted_kv_prefetch_table = render_table(targeted_kv_prefetch_rows, TARGETED_KV_PREFETCH_COLUMNS)
    harness_priority_table = render_table(harness_priority_rows, HARNESS_PRIORITY_COLUMNS)
    nat_service_priority_table = render_table(nat_service_priority_rows, NAT_SERVICE_PRIORITY_COLUMNS)
    cache_signal_table = render_table(cache_signal_rows, CACHE_SIGNAL_COLUMNS)
    cache_action_table = render_table(cache_action_rows, CACHE_ACTION_COLUMNS)
    cache_benefit_table = render_table(cache_benefit_rows, CACHE_BENEFIT_COLUMNS)
    sglang_cache_path_audit_table = render_table(sglang_cache_path_audit_rows, SGLANG_CACHE_PATH_AUDIT_COLUMNS)
    nat_inferred_priority_profile_table = render_nat_inferred_priority_profile(nat_inferred_priority_profile)
    raw_table = render_table(
        rows,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "encoding_codec",
            "encoding_config_hash",
            "session_id",
            "first_token_lateness_ms",
            "due_to_request_start_ms",
            "due_to_sglang_receive_ms",
            "sglang_receive_to_first_token_ms",
            "ttft_ms",
            "sglang_priority",
            "harness_input_priority_signal",
            "harness_emit_priority_signal",
            "gateway_priority_translation",
            "gateway_priority_translation_source",
            "harness_native_cache_signal_seen",
            "harness_native_cache_signal",
            "harness_native_cache_signal_source",
            "gateway_cache_lowered",
            "gateway_cache_translation",
            "gateway_cache_translation_source",
            "gateway_cache_invented_signal",
            "backend_receive_source",
            "first_token_source",
            "status",
            "error",
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Replay Deadline Pressure Chart</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #f8fafc; }}
main {{ max-width: 1600px; margin: 0 auto; padding: 32px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
h2 {{ margin-top: 32px; font-size: 22px; }}
p {{ line-height: 1.5; color: #334155; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-top: 18px; overflow-x: auto; }}
.legend-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin-top: 10px; }}
.legend-row {{ display: flex; gap: 18px; align-items: flex-start; margin: 8px 0; }}
.legend-row strong {{ flex: 0 0 130px; }}
.legend-items {{ display: flex; flex-wrap: wrap; gap: 12px 24px; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 8px; white-space: nowrap; }}
.muted {{ color: #64748b; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}
.legend-symbol {{ width: 18px; height: 18px; flex: 0 0 auto; overflow: visible; }}
.chart-controls {{ display: grid; gap: 10px; margin-bottom: 14px; padding: 14px 16px; border: 1px solid #dbeafe; border-radius: 8px; background: #f8fbff; }}
.control-row {{ display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; }}
.control-row strong {{ flex: 0 0 112px; }}
.control-button {{ appearance: none; border: 1px solid #cbd5e1; border-radius: 7px; background: #ffffff; color: #111827; font-weight: 700; padding: 6px 10px; cursor: pointer; }}
.control-button:hover {{ border-color: #64748b; background: #f8fafc; }}
.control-hint {{ color: #64748b; font-size: 12px; }}
.chip-list {{ display: flex; flex-wrap: wrap; gap: 8px 10px; }}
.chart-chip {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid #e2e8f0; border-radius: 999px; background: white; padding: 5px 9px; font-size: 12px; font-weight: 650; color: #334155; }}
.chip-dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
.deadline-panel-symlog {{ display: none; }}
.deadline-panel-linear {{ display: inline; }}
.note {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 16px; color: #1e3a8a; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
th {{ background: #f1f5f9; font-weight: 700; }}
code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
<h1>Replay Deadline Pressure Chart</h1>
<p>Report label: <code>{html.escape(report_label)}</code>. Generated {generated}.</p>
<p>Hardware profile: <code>{html.escape(hardware_profile)}</code>. Profile file: <code>{html.escape(hardware_profile_path)}</code>.</p>
    <p class="note">This lightweight all-harness report uses the completed workload traces directly. Each dot is the median replay for one harness and signal path; labels show n= when multiple replay requests were summarized. Use the controls to choose signal paths, switch the deadline-pressure axis between linear and symlog, and focus on selected pressure levels such as P0/P3/P5. The TTFT-impact view shows how long each replay request took to reach first token after it started. Lower is better. Exact lower-level modes remain in the evidence file.</p>
	<h2>Signal Family Definitions</h2>
	<p>This table explains who added the signal before it reached SGLang. The chart uses this family view first, while raw mode names remain in the evidence tables.</p>
	<div class="card">{signal_family_definition_table}</div>
	<h2>Harness Cache Signal Decision Map</h2>
	<p>This table explains how each harness produced cache signals in this report. It focuses on the target replay requests shown in the chart.</p>
	<div class="card">{harness_cache_decision_table}</div>
	<h2>Pressure Level Definitions</h2>
	<p>Each pressure level is a bundled stress setting, not a full Cartesian sweep. The chart below shows only the levels marked <strong>Yes</strong> for this run.</p>
	<div class="card">{pressure_definition_table}</div>
<div class="card">{chart_controls}{chart}</div>
{cost_accounting_section}
 	<h2>Evidence Tables</h2>
 	<p>The proof tables, raw replay rows, priority preservation audit, cache signal audit, cache action proof, and summary tables are now kept out of the main report.</p>
 	<p><a href="evidence_tables.html">Open the evidence tables / raw proof file</a>.</p>
 	</main>
{chart_interaction_script}
 	</body>
 	</html>
 	"""


def render_evidence_html(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    cost_accounting_rows: list[dict[str, Any]],
    speculative_prefill_rows: list[dict[str, Any]],
    targeted_kv_prefetch_rows: list[dict[str, Any]],
    controller_demote_restore_rows: list[dict[str, Any]],
    controller_admission_rows: list[dict[str, Any]],
    harness_priority_rows: list[dict[str, Any]],
    nat_service_priority_rows: list[dict[str, Any]],
    cache_signal_rows: list[dict[str, Any]],
    cache_action_rows: list[dict[str, Any]],
    cache_benefit_rows: list[dict[str, Any]],
    sglang_cache_path_audit_rows: list[dict[str, Any]],
    nat_inferred_priority_profile: dict[str, Any],
    report_label: str,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_table = render_table(
        summary,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "encoding_codec",
            "encoding_config_hash",
            "samples",
            "median_first_token_lateness_ms",
            "median_ttft_ms",
            "median_sglang_receive_to_first_token_ms",
            "median_due_to_sglang_receive_ms",
            "min_first_token_lateness_ms",
            "max_first_token_lateness_ms",
        ],
    )
    breakdown_table = render_table(
        summary,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "encoding_codec",
            "encoding_config_hash",
            "samples",
            "median_due_to_request_start_ms",
            "median_due_to_sglang_receive_ms",
            "median_ttft_ms",
            "median_sglang_receive_to_first_token_ms",
            "median_first_token_lateness_ms",
        ],
    )
    speculative_prefill_table = render_table(speculative_prefill_rows, SPECULATIVE_PREFILL_COLUMNS)
    targeted_kv_prefetch_table = render_table(targeted_kv_prefetch_rows, TARGETED_KV_PREFETCH_COLUMNS)
    controller_demote_restore_table = render_table(
        controller_demote_restore_rows,
        CONTROLLER_DEMOTE_RESTORE_COLUMNS,
    )
    controller_admission_table = render_table(controller_admission_rows, CONTROLLER_ADMISSION_COLUMNS)
    harness_priority_table = render_table(harness_priority_rows, HARNESS_PRIORITY_COLUMNS)
    nat_service_priority_table = render_table(nat_service_priority_rows, NAT_SERVICE_PRIORITY_COLUMNS)
    cache_signal_table = render_table(cache_signal_rows, CACHE_SIGNAL_COLUMNS)
    cache_action_table = render_table(cache_action_rows, CACHE_ACTION_COLUMNS)
    cache_benefit_table = render_table(cache_benefit_rows, CACHE_BENEFIT_COLUMNS)
    sglang_cache_path_audit_table = render_table(sglang_cache_path_audit_rows, SGLANG_CACHE_PATH_AUDIT_COLUMNS)
    nat_inferred_priority_profile_table = render_nat_inferred_priority_profile(nat_inferred_priority_profile)
    cost_accounting_table = render_table(cost_accounting_rows, COST_ACCOUNTING_COLUMNS)
    raw_table = render_table(
        rows,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "encoding_codec",
            "encoding_config_hash",
            "session_id",
            "request_id",
            "phase",
            "request_group",
            "has_replay_deadline",
            "first_token_lateness_ms",
            "replay_debt_ms",
            "due_to_request_start_ms",
            "due_to_sglang_receive_ms",
            "sglang_receive_to_first_token_ms",
            "ttft_ms",
            "sglang_priority",
            "harness_input_priority_signal",
            "harness_emit_priority_signal",
            "gateway_priority_translation",
            "gateway_priority_translation_source",
            "harness_native_cache_signal_seen",
            "harness_native_cache_signal",
            "harness_native_cache_signal_source",
            "gateway_cache_lowered",
            "gateway_cache_translation",
            "gateway_cache_translation_source",
            "gateway_cache_invented_signal",
            "backend_receive_source",
            "first_token_source",
            "status",
            "error",
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Replay Deadline Pressure Evidence Tables</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #f8fafc; }}
main {{ max-width: 1600px; margin: 0 auto; padding: 32px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
h2 {{ margin-top: 32px; font-size: 22px; }}
p {{ line-height: 1.5; color: #334155; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-top: 18px; overflow-x: auto; }}
.note {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 16px; color: #1e3a8a; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
th {{ background: #f1f5f9; font-weight: 700; }}
code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
a {{ color: #2563eb; }}
</style>
</head>
<body>
<main>
<h1>Replay Deadline Pressure Evidence Tables</h1>
<p>Report label: <code>{html.escape(report_label)}</code>. Generated {generated}.</p>
<p><a href="master_report.html">Back to chart-first master report</a>.</p>
<p class="note">These are the raw proof tables behind the chart-first report. Use this file when auditing exact request IDs, signal transport, gateway lowering, SGLang priority, cache/preload action, and timing values.</p>
<h2>Harness Priority Preservation Proof</h2>
<p>This table appears when the run includes <code>pre_harness_priority_hints</code>, <code>nat_inferred_priority_hints</code>, or <code>harness_emitted_signals</code>. It proves whether the harness carried or inferred priority, and whether the gateway translated that signal to SGLang priority.</p>
<div class="card">{harness_priority_table if harness_priority_rows else "<p>No harness priority proof rows found in this run.</p>"}</div>
<h2>NAT Inferred Priority Profile</h2>
<div class="card">{nat_inferred_priority_profile_table if nat_inferred_priority_profile_table else "<p>No standalone NAT inferred-priority profile found in this run.</p>"}</div>
<h2>NAT Shared-Service Priority Probe</h2>
<p>This table appears when NAT is run as a shared <code>nat serve</code> service. It compares the order requests entered NAT with the order NAT emitted model calls to the gateway. If urgent work jumps ahead of older background work here, that is NAT-side priority evidence.</p>
<div class="card">{nat_service_priority_table if nat_service_priority_rows else "<p>No NAT shared-service priority probe rows found in this run.</p>"}</div>
<h2>Harness Native Cache Signal Proof</h2>
<p>This table appears when the run includes <code>no_cache_signal</code>, <code>harness_native_cache_lowered</code>, or <code>harness_emitted_signals</code>. The gateway is always present, but it only translates cache fields that the harness emitted. In the unified harness-emitted mode, cache signals are lowered to gateway speculative KV preload. <code>gateway_invented_signal</code> should remain <code>false</code>.</p>
<div class="card">{cache_signal_table if cache_signal_rows else "<p>No harness native cache signal proof rows found in this run.</p>"}</div>
<h2>Cache Action Proof</h2>
<p>This target-scoped table checks whether the same replay request that carried a harness cache signal also caused a gateway cache/preload lowering action, then showed SGLang cache-path activity: prefix matching, load-back, host-to-device copy, prefill attribution, or cache-commit events.</p>
<div class="card">{cache_action_table if cache_action_rows else "<p>No cache action proof rows found in this run.</p>"}</div>
<h2>Cache Benefit Summary</h2>
<p>This table compares <code>harness_native_cache_lowered</code> against <code>no_cache_signal</code> for the same harness and pressure level. Negative TTFT delta means the cache-lowered run reached the first token faster after the replay request started.</p>
<div class="card">{cache_benefit_table if cache_benefit_rows else "<p>No paired cache-benefit rows found. Run both no_cache_signal and harness_native_cache_lowered for the same harness and pressure level.</p>"}</div>
<h2>SGLang Cache Signal Path Audit</h2>
<p>This static source audit is collected from the installed SGLang package on the experiment machine. Runtime proof still comes from the target-scoped trace rows above.</p>
<div class="card">{sglang_cache_path_audit_table if sglang_cache_path_audit_rows else "<p>No SGLang cache signal path audit rows found. Re-run with an environment collector on the experiment machine.</p>"}</div>
<h2>System Cost Accounting</h2>
<p>This table sums TTFT and positive replay debt for target versus filler/background requests. Filler debt is present only when the filler request had a replay due timestamp.</p>
<div class="card">{cost_accounting_table if cost_accounting_rows else "<p>No system cost accounting rows found.</p>"}</div>
<h2>Targeted KV Prefetch Proof</h2>
<p>This table appears when the run includes <code>controller_targeted_kv_prefetch</code>. It proves whether the controller requested explicit target-prefix KV movement, whether the active SGLang adapter exposed a direct hook, and whether load-back or host-to-device movement was observed before replay compute.</p>
<div class="card">{targeted_kv_prefetch_table if targeted_kv_prefetch_rows else "<p>No targeted KV prefetch rows found in this run.</p>"}</div>
<h2>Controller Demote/Restore Proof</h2>
<p>This table appears when the run includes <code>controller_demote_restore</code>. It proves whether background/filler traffic was lowered during the target replay window, whether the replay itself was raised to SGLang priority, and whether normal behavior was restored afterward.</p>
<div class="card">{controller_demote_restore_table if controller_demote_restore_rows else "<p>No controller demote/restore rows found in this run.</p>"}</div>
<h2>Controller Admission Proof</h2>
<p>This table appears when the run includes <code>controller_admission_control</code>. It proves whether controller speculative warmup was admitted or skipped, why it was skipped, and whether replay priority was still lowered into SGLang.</p>
<div class="card">{controller_admission_table if controller_admission_rows else "<p>No controller admission rows found in this run.</p>"}</div>
<h2>Speculative Prefill Proof</h2>
<p>This table appears when the run includes <code>e2e_priority_hints_speculative_prefill</code> or <code>harness_emitted_signals</code>. It proves whether a background <code>max_tokens=1</code> warmup was sent before replay, whether it was triggered by gateway speculative KV preload, and whether the replay showed cached-prefix reuse.</p>
<div class="card">{speculative_prefill_table if speculative_prefill_rows else "<p>No speculative prefill rows found in this run.</p>"}</div>
<h2>Summary</h2>
<div class="card">{summary_table}</div>
<h2>Delay Breakdown</h2>
<p>This table separates replay request TTFT from backend service time. <code>median_ttft_ms</code> is the number used in panel C of the main report; <code>median_sglang_receive_to_first_token_ms</code> remains the backend-only view.</p>
<div class="card">{breakdown_table}</div>
<h2>Raw Replay Proof</h2>
<div class="card">{raw_table}</div>
</main>
</body>
</html>
"""


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    cost_accounting_rows: list[dict[str, Any]],
    speculative_prefill_rows: list[dict[str, Any]],
    targeted_kv_prefetch_rows: list[dict[str, Any]],
    controller_demote_restore_rows: list[dict[str, Any]],
    controller_admission_rows: list[dict[str, Any]],
    harness_priority_rows: list[dict[str, Any]],
    nat_service_priority_rows: list[dict[str, Any]],
    cache_signal_rows: list[dict[str, Any]],
    cache_action_rows: list[dict[str, Any]],
    cache_benefit_rows: list[dict[str, Any]],
    sglang_cache_path_audit_rows: list[dict[str, Any]],
    nat_inferred_priority_profile: dict[str, Any],
    run_config: dict[str, str],
) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_kind": run_config.get("EXPERIMENT_KIND") or "multi_harness_deadline_pressure",
        "report_label": args.report_label,
        "script": "scripts/build_multi_harness_deadline_summary.py",
        "root": str(args.root),
        "report_dir": str(args.out_dir),
        "row_count": len(rows),
        "summary_row_count": len(summary),
        "cost_accounting_row_count": len(cost_accounting_rows),
        "speculative_prefill_row_count": len(speculative_prefill_rows),
        "targeted_kv_prefetch_row_count": len(targeted_kv_prefetch_rows),
        "controller_demote_restore_row_count": len(controller_demote_restore_rows),
        "controller_admission_row_count": len(controller_admission_rows),
        "harness_priority_row_count": len(harness_priority_rows),
        "nat_service_priority_row_count": len(nat_service_priority_rows),
        "cache_signal_row_count": len(cache_signal_rows),
        "cache_action_row_count": len(cache_action_rows),
        "cache_benefit_row_count": len(cache_benefit_rows),
        "sglang_cache_path_audit_row_count": len(sglang_cache_path_audit_rows),
        "nat_inferred_priority_profile": bool(nat_inferred_priority_profile),
        "nat_inferred_priority_profile_path": (
            str(args.out_dir / "nat_inferred_priority_profile.json") if nat_inferred_priority_profile else ""
        ),
        "hardware_profile": os.environ.get("HARDWARE_PROFILE") or run_config.get("HARDWARE_PROFILE", ""),
        "hardware_profile_path": os.environ.get("HARDWARE_PROFILE_PATH") or run_config.get("HARDWARE_PROFILE_PATH", ""),
        "harnesses": sorted({row["harness"] for row in rows}),
        "pressure_levels": sorted({row["pressure_level"] for row in rows}),
        "modes": sorted({row["mode"] for row in rows}),
        "prompt_encodings": sorted({str(row.get("encoding_codec", "identity")) + ":" + str(row.get("encoding_config_hash", "")) for row in rows}),
    }
    atomic_write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight all-harness replay deadline report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--report-label", default=os.environ.get("REPORT_LABEL") or f"multi_harness_deadline_summary_{int(time.time())}")
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--run-environment-json", type=Path)
    parser.add_argument("--rows-csv", type=Path, help="Reuse an existing global_kv_readiness_by_mode.csv instead of reading raw case traces.")
    parser.add_argument("--update-latest", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = read_run_config(args.run_config or args.out_dir / "run_config.env")
    if args.rows_csv:
        rows = read_csv_table(args.rows_csv)
        target_rows = target_replay_rows(rows)
        summary = read_csv_table(args.out_dir / "global_kv_readiness_by_mode_summary.csv") or summarize(target_rows)
        cost_accounting_rows = collect_cost_accounting_summary(rows)
        speculative_prefill_rows = read_csv_table(args.out_dir / "speculative_prefill_proof.csv")
        targeted_kv_prefetch_rows = read_csv_table(args.out_dir / "targeted_kv_prefetch_proof.csv")
        controller_demote_restore_rows = read_csv_table(args.out_dir / "controller_demote_restore_proof.csv")
        controller_admission_rows = read_csv_table(args.out_dir / "controller_admission_proof.csv")
        harness_priority_rows = read_csv_table(args.out_dir / "harness_priority_preservation_proof.csv")
        nat_service_priority_rows = read_csv_table(args.out_dir / "nat_service_priority_probe.csv")
        cache_signal_rows = read_csv_table(args.out_dir / "harness_native_cache_signal_proof.csv")
        cache_action_rows = read_csv_table(args.out_dir / "cache_action_proof.csv")
        cache_benefit_rows = read_csv_table(args.out_dir / "cache_benefit_summary.csv")
    else:
        rows = collect_rows(args.root)
        target_rows = target_replay_rows(rows)
        summary = summarize(target_rows)
        cost_accounting_rows = collect_cost_accounting_summary(rows)
        speculative_prefill_rows = collect_speculative_prefill_proof(args.root, target_rows)
        targeted_kv_prefetch_rows = collect_targeted_kv_prefetch_proof(args.root, target_rows)
        controller_demote_restore_rows = collect_controller_demote_restore_proof(args.root, target_rows)
        controller_admission_rows = collect_controller_admission_proof(args.root, target_rows)
        harness_priority_rows = collect_harness_priority_proof(args.root, target_rows)
        nat_service_priority_rows = collect_nat_service_priority_probe(args.root)
        cache_signal_rows = collect_harness_native_cache_signal_proof(target_rows)
        cache_action_rows = collect_cache_action_proof(args.root, target_rows)
        cache_benefit_rows = collect_cache_benefit_summary(summary, cache_action_rows)
    run_environment = read_json_file(args.run_environment_json or args.out_dir / "run_environment.json")
    sglang_cache_path_audit_rows = (
        read_csv_table(args.out_dir / "sglang_cache_signal_path_audit.csv")
        if args.rows_csv
        else collect_sglang_cache_path_audit(run_environment)
    )
    nat_inferred_priority_profile = read_nat_inferred_priority_profile(args.out_dir)
    write_csv(args.out_dir / "global_kv_readiness_by_mode.csv", rows, RAW_COLUMNS)
    write_csv(args.out_dir / "global_kv_readiness_by_mode_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(args.out_dir / "cost_accounting_summary.csv", cost_accounting_rows, COST_ACCOUNTING_COLUMNS)
    write_csv(args.out_dir / "speculative_prefill_proof.csv", speculative_prefill_rows, SPECULATIVE_PREFILL_COLUMNS)
    write_csv(args.out_dir / "targeted_kv_prefetch_proof.csv", targeted_kv_prefetch_rows, TARGETED_KV_PREFETCH_COLUMNS)
    write_csv(
        args.out_dir / "controller_demote_restore_proof.csv",
        controller_demote_restore_rows,
        CONTROLLER_DEMOTE_RESTORE_COLUMNS,
    )
    write_csv(args.out_dir / "controller_admission_proof.csv", controller_admission_rows, CONTROLLER_ADMISSION_COLUMNS)
    write_csv(args.out_dir / "harness_priority_preservation_proof.csv", harness_priority_rows, HARNESS_PRIORITY_COLUMNS)
    write_csv(args.out_dir / "nat_service_priority_probe.csv", nat_service_priority_rows, NAT_SERVICE_PRIORITY_COLUMNS)
    write_csv(args.out_dir / "harness_native_cache_signal_proof.csv", cache_signal_rows, CACHE_SIGNAL_COLUMNS)
    write_csv(args.out_dir / "cache_action_proof.csv", cache_action_rows, CACHE_ACTION_COLUMNS)
    write_csv(args.out_dir / "cache_benefit_summary.csv", cache_benefit_rows, CACHE_BENEFIT_COLUMNS)
    write_csv(args.out_dir / "sglang_cache_signal_path_audit.csv", sglang_cache_path_audit_rows, SGLANG_CACHE_PATH_AUDIT_COLUMNS)
    html_text = render_html(
        rows,
        summary,
        cost_accounting_rows,
        speculative_prefill_rows,
        targeted_kv_prefetch_rows,
        controller_demote_restore_rows,
        controller_admission_rows,
        harness_priority_rows,
        nat_service_priority_rows,
        cache_signal_rows,
        cache_action_rows,
        cache_benefit_rows,
        sglang_cache_path_audit_rows,
        nat_inferred_priority_profile,
        args.report_label,
        run_config,
    )
    evidence_html_text = render_evidence_html(
        rows,
        summary,
        cost_accounting_rows,
        speculative_prefill_rows,
        targeted_kv_prefetch_rows,
        controller_demote_restore_rows,
        controller_admission_rows,
        harness_priority_rows,
        nat_service_priority_rows,
        cache_signal_rows,
        cache_action_rows,
        cache_benefit_rows,
        sglang_cache_path_audit_rows,
        nat_inferred_priority_profile,
        args.report_label,
    )
    if any(row.get("encoding_config_hash") for row in rows):
        from agentic_prompt_codec.reporting import write_encoding_report
        encoding_html = write_encoding_report(args.root, args.out_dir, rows, reuse_existing=bool(args.rows_csv))
        html_text = html_text.replace("</body>", encoding_html + "</body>")
        evidence_html_text = evidence_html_text.replace("</body>", encoding_html + "</body>")
    report_path = args.out_dir / "master_report.html"
    evidence_path = args.out_dir / "evidence_tables.html"
    atomic_write_text(report_path, html_text)
    atomic_write_text(evidence_path, evidence_html_text)
    write_manifest(
        args.out_dir / "manifest.json",
        args,
        rows,
        summary,
        cost_accounting_rows,
        speculative_prefill_rows,
        targeted_kv_prefetch_rows,
        controller_demote_restore_rows,
        controller_admission_rows,
        harness_priority_rows,
        nat_service_priority_rows,
        cache_signal_rows,
        cache_action_rows,
        cache_benefit_rows,
        sglang_cache_path_audit_rows,
        nat_inferred_priority_profile,
        run_config,
    )
    if args.latest_root and args.update_latest:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.latest_root / "latest_master_report.html", html_text)
        atomic_write_text(args.latest_root / "evidence_tables.html", evidence_html_text)
        atomic_write_text(args.latest_root / "latest_evidence_tables.html", evidence_html_text)
        write_csv(args.latest_root / "latest_cost_accounting_summary.csv", cost_accounting_rows, COST_ACCOUNTING_COLUMNS)
        write_manifest(
            args.latest_root / "latest_manifest.json",
            args,
            rows,
            summary,
            cost_accounting_rows,
            speculative_prefill_rows,
            targeted_kv_prefetch_rows,
            controller_demote_restore_rows,
            controller_admission_rows,
            harness_priority_rows,
            nat_service_priority_rows,
            cache_signal_rows,
            cache_action_rows,
            cache_benefit_rows,
            sglang_cache_path_audit_rows,
            nat_inferred_priority_profile,
            run_config,
        )
    print(f"wrote {report_path}")
    print(f"wrote {evidence_path}")
    print(f"rows={len(rows)} summary_rows={len(summary)} cost_accounting_rows={len(cost_accounting_rows)}")


if __name__ == "__main__":
    main()
