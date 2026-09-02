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
    "hatcher": "Hatcher",
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

MODE_LABELS = {
    "no_prefetch": "NP = No prefetch",
    "e2e_priority_hints": "E2E = End-to-end priority hints",
    "pre_harness_priority_hints": "PH = Pre-harness priority hints",
    "nat_inferred_priority_hints": "NI = NAT inferred priority hints",
    "e2e_priority_hints_speculative_prefill": "SP = E2E priority + speculative prefill",
}

MODE_COLORS = {
    "no_prefetch": "#2563eb",
    "e2e_priority_hints": "#0f766e",
    "pre_harness_priority_hints": "#7c3aed",
    "nat_inferred_priority_hints": "#be185d",
    "e2e_priority_hints_speculative_prefill": "#ea580c",
}

MODE_ORDER = tuple(MODE_LABELS)

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


def row_agent_label(row: dict[str, Any]) -> str:
    for key in ("agent_label", "agent_request_id", "request_id", "label"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def collect_rows(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        harness, pressure, mode = case_key_from_name(case_dir.name)
        trace_rows = read_jsonl(case_dir / "m27_trace.jsonl")
        due_by_session: dict[str, dict[str, Any]] = {}
        replay_starts: list[dict[str, Any]] = []
        replay_ends: list[dict[str, Any]] = []
        sglang_receive_by_label: dict[str, dict[str, Any]] = {}
        first_decode_by_label: dict[str, dict[str, Any]] = {}
        for row in trace_rows:
            event = row.get("event")
            phase = row.get("phase")
            if event == "m27.replay.due":
                due_by_session[str(row.get("session_id") or "")] = row
            elif event == "m27.request.start" and phase == "replay":
                replay_starts.append(row)
            elif event == "m27.request.end" and phase == "replay":
                replay_ends.append(row)
            elif (
                event == "kv_telemetry.request_stage"
                and row.get("category") == "sglang_receive"
                and row_agent_label(row)
            ):
                label = row_agent_label(row)
                current = sglang_receive_by_label.get(label)
                if current is None or int(float_value(row.get("ts_ns"))) < int(float_value(current.get("ts_ns"))):
                    sglang_receive_by_label[label] = row
            elif (
                event == "kv_telemetry.request_stage"
                and row.get("category") == "scheduler_process_decode_result"
                and phase == "end"
                and row_agent_label(row)
            ):
                label = row_agent_label(row)
                current = first_decode_by_label.get(label)
                if current is None or int(float_value(row.get("ts_ns"))) < int(float_value(current.get("ts_ns"))):
                    first_decode_by_label[label] = row

        def append_replay_row(
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
            start_ts_ns = int(float_value(start.get("ts_ns")))
            due_ts_ns = int(float_value(due.get("ts_ns")))
            ttft_ms = ((first_token_ts_ns - start_ts_ns) / 1_000_000.0) if first_token_ts_ns and start_ts_ns else float("nan")
            lateness_ms = ((first_token_ts_ns - due_ts_ns) / 1_000_000.0) if due_ts_ns and first_token_ts_ns else float("nan")
            sglang_receive = sglang_receive_by_label.get(label, {})
            sglang_receive_ts_ns = int(float_value(sglang_receive.get("ts_ns")))
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
            out.append(
                {
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
                    "first_token_lateness_ms": round(lateness_ms, 3) if math.isfinite(lateness_ms) else "",
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
                    "status": status,
                    "error": error,
                }
            )

        start_by_label = {str(row.get("label") or row.get("request_id") or ""): row for row in replay_starts}
        ended_labels: set[str] = set()
        for end in replay_ends:
            label = str(end.get("label") or end.get("request_id") or "")
            ended_labels.add(label)
            session_id = str(end.get("session_id") or "")
            start = start_by_label.get(label, {})
            due = due_by_session.get(session_id, {})
            start_ts_ns = int(float_value(start.get("ts_ns")))
            ttft_ms = float_value(end.get("ttft_ms"))
            first_token_ts_ns = start_ts_ns + int(round(ttft_ms * 1_000_000))
            append_replay_row(
                label=label,
                session_id=session_id,
                source_row=end,
                start=start,
                due=due,
                first_token_ts_ns=first_token_ts_ns,
                request_end_ts_ns=int(float_value(end.get("ts_ns"))),
                first_token_source="gateway_request_end_ttft",
                status=end.get("status", ""),
                error=end.get("error", ""),
            )
        for label, start in start_by_label.items():
            if label in ended_labels or label not in first_decode_by_label:
                continue
            session_id = str(start.get("session_id") or label.rsplit("_replay", 1)[0])
            due = due_by_session.get(session_id, {})
            append_replay_row(
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
        if row.get("mode") == "e2e_priority_hints_speculative_prefill"
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
                    "mode": "e2e_priority_hints_speculative_prefill",
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
    replay_by_label = {
        (str(row.get("case_dir") or ""), str(row.get("request_id") or "")): row
        for row in replay_rows
        if row.get("mode") == "pre_harness_priority_hints"
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
            if row.get("phase") != "replay" or row.get("mode") != "pre_harness_priority_hints":
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
            if driver_intent_seen != "yes":
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


RAW_COLUMNS = [
    "harness",
    "harness_label",
    "pressure_level",
    "pressure_level_label",
    "mode",
    "mode_label",
    "session_id",
    "request_id",
    "first_token_lateness_ms",
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

SUMMARY_COLUMNS = [
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
    "median_sglang_receive_to_first_token_ms",
    "min_first_token_lateness_ms",
    "max_first_token_lateness_ms",
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
        if row.get("first_token_lateness_ms") == "":
            continue
        grouped[(str(row["harness"]), str(row["pressure_level"]), str(row["mode"]))].append(row)
    out: list[dict[str, Any]] = []
    for (harness, pressure, mode), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], PRESSURE_ORDER.index(item[0][1]) if item[0][1] in PRESSURE_ORDER else 999, item[0][2])):
        values = [float(row["first_token_lateness_ms"]) for row in group_rows if row.get("first_token_lateness_ms") != ""]
        due_to_request_start = [value for row in group_rows if (value := optional_float(row.get("due_to_request_start_ms"))) is not None]
        due_to_sglang_receive = [value for row in group_rows if (value := optional_float(row.get("due_to_sglang_receive_ms"))) is not None]
        backend_values = [value for row in group_rows if (value := optional_float(row.get("sglang_receive_to_first_token_ms"))) is not None]
        out.append(
            {
                "harness": harness,
                "harness_label": HARNESS_LABELS.get(harness, harness),
                "pressure_level": pressure,
                "pressure_level_label": PRESSURE_LABELS.get(pressure, pressure),
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "samples": len(values),
                "median_first_token_lateness_ms": round(statistics.median(values), 3),
                "median_due_to_request_start_ms": round(statistics.median(due_to_request_start), 3) if due_to_request_start else "",
                "median_due_to_sglang_receive_ms": round(statistics.median(due_to_sglang_receive), 3) if due_to_sglang_receive else "",
                "median_sglang_receive_to_first_token_ms": round(statistics.median(backend_values), 3) if backend_values else "",
                "min_first_token_lateness_ms": round(min(values), 3),
                "max_first_token_lateness_ms": round(max(values), 3),
            }
        )
    return out


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


def svg_symbol(kind: str, x: float, y: float, color: str, title: str) -> str:
    escaped_title = html.escape(title)
    common = f'fill="{color}" stroke="{color}" stroke-width="2" opacity="0.88"'
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


def render_pressure_chart(rows: list[dict[str, Any]]) -> str:
    pressures = [pressure for pressure in PRESSURE_ORDER if any(row["pressure_level"] == pressure for row in rows)]
    harnesses = [harness for harness in HARNESS_LABELS if any(row["harness"] == harness for row in rows)]
    modes = [mode for mode in MODE_ORDER if any(row["mode"] == mode for row in rows)]
    if not pressures or not harnesses:
        return "<p>No replay rows found.</p>"

    pressure_w = max(420, len(harnesses) * 44 + 110)
    width = max(1400, pressure_w * len(pressures) + 220)
    height = 980
    left = 120
    right = 40
    panel_h = 310
    panel_gap = 145
    top_a = 82
    top_b = top_a + panel_h + panel_gap
    bottom_margin = 95
    plot_w = width - left - right
    pressure_group_w = plot_w / len(pressures)

    def mode_offset(mode: str) -> float:
        if not modes:
            return 0.0
        try:
            index = modes.index(mode)
        except ValueError:
            index = 0
        return (index - (len(modes) - 1) / 2) * 13.0

    def x_pos(pressure_index: int, harness_index: int, mode: str, sample_index: int, sample_count: int) -> float:
        pressure_left = left + pressure_index * pressure_group_w
        harness_step = pressure_group_w / max(1, len(harnesses))
        base = pressure_left + harness_step * (harness_index + 0.5)
        jitter = 0.0 if sample_count <= 1 else (sample_index - (sample_count - 1) / 2) * 3.2
        return base + mode_offset(mode) + jitter

    lines = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Replay Deadline Pressure Chart">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    rows_by_group_mode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_group_mode[(str(row["pressure_level"]), str(row["harness"]), str(row["mode"]))].append(row)

    def draw_panel(
        panel_top: float,
        value_key: str,
        heading: str,
        note: str,
        y_axis_label: str,
        zero_label: str,
        unit_label: str,
        tick_values: list[int],
    ) -> None:
        panel_bottom = panel_top + panel_h
        panel_values = [value for row in rows if (value := optional_float(row.get(value_key))) is not None]
        transformed = [symlog(value) for value in panel_values + [float(tick) for tick in tick_values]]
        y_min = min(transformed)
        y_max = max(transformed)
        pad = max(0.2, (y_max - y_min) * 0.08)
        y_min -= pad
        y_max += pad

        def y_pos_panel(value: float) -> float:
            mapped = symlog(value)
            return panel_top + (y_max - mapped) / (y_max - y_min) * panel_h

        lines.append(f'<text x="{left}" y="{panel_top-46:.1f}" font-size="18" font-weight="800" fill="#111827">{html.escape(heading)}</text>')
        lines.append(f'<text x="{left}" y="{panel_top-24:.1f}" font-size="12" fill="#64748b">{html.escape(note)}</text>')
        for tick in tick_values:
            y = y_pos_panel(float(tick))
            stroke = "#111827" if tick == 0 else "#e5e7eb"
            width_attr = "1.5" if tick == 0 else "1"
            lines.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="{width_attr}"/>')
            lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#374151">{tick} ms</text>')
        lines.append(f'<text x="{width-right-4}" y="{y_pos_panel(0)-8:.1f}" text-anchor="end" font-size="13" font-weight="700">{html.escape(zero_label)}</text>')

        for pressure_index, pressure in enumerate(pressures):
            x = left + pressure_index * pressure_group_w
            lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{panel_top}" y2="{panel_bottom}" stroke="#cbd5e1" stroke-dasharray="5 6"/>')
            if pressure_index % 2 == 1:
                lines.append(f'<rect x="{x:.1f}" y="{panel_top}" width="{pressure_group_w:.1f}" height="{panel_h}" fill="#f8fafc" opacity="0.62"/>')
            cx = x + pressure_group_w / 2
            lines.append(f'<text x="{cx:.1f}" y="{panel_bottom+36:.1f}" text-anchor="middle" font-size="16" font-weight="800" fill="#111827">{html.escape(PRESSURE_LABELS.get(pressure, pressure))}</text>')
            lines.append(f'<text x="{cx:.1f}" y="{panel_bottom+56:.1f}" text-anchor="middle" font-size="11" fill="#64748b">all harnesses overlaid; color = mode, shape = harness</text>')
            for harness_index, harness in enumerate(harnesses):
                harness_x = left + pressure_index * pressure_group_w + (pressure_group_w / max(1, len(harnesses))) * (harness_index + 0.5)
                lines.append(f'<line x1="{harness_x:.1f}" x2="{harness_x:.1f}" y1="{panel_top}" y2="{panel_bottom}" stroke="#f1f5f9" stroke-width="1"/>')
                for mode in modes:
                    sample_rows = rows_by_group_mode.get((pressure, harness, mode), [])
                    sample_rows = [row for row in sample_rows if optional_float(row.get(value_key)) is not None]
                    if not sample_rows:
                        continue
                    med = statistics.median(float(row[value_key]) for row in sample_rows)
                    mx = x_pos(pressure_index, harness_index, mode, 0, 1)
                    y = y_pos_panel(med)
                    lines.append(f'<line x1="{mx-9:.1f}" x2="{mx+9:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{MODE_COLORS[mode]}" stroke-width="3" stroke-linecap="round"/>')
                    label_y = y - 10 if mode == "no_prefetch" else y + 16
                    label_y = min(max(label_y, panel_top + 12), panel_bottom - 8)
                    lines.append(svg_text_label(compact_ms(med), mx, label_y, MODE_COLORS[mode]))
                    for sample_index, row in enumerate(sample_rows):
                        value = float(row[value_key])
                        dot_x = x_pos(pressure_index, harness_index, mode, sample_index, len(sample_rows))
                        dot_y = y_pos_panel(value)
                        title = (
                            f"{heading} | {PRESSURE_LABELS.get(pressure, pressure)} | "
                            f"{HARNESS_LABELS.get(harness, harness)} | "
                            f"{MODE_LABELS[mode]} | {value:.1f} {unit_label}"
                        )
                        lines.append(svg_symbol(HARNESS_SYMBOLS.get(harness, "circle"), dot_x, dot_y, MODE_COLORS[mode], title))
        lines.append(f'<line x1="{width-right:.1f}" x2="{width-right:.1f}" y1="{panel_top}" y2="{panel_bottom}" stroke="#cbd5e1" stroke-dasharray="5 6"/>')
        lines.append(f'<text transform="translate(32 {panel_top + panel_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="14" font-weight="700">{html.escape(y_axis_label)}</text>')

    draw_panel(
        top_a,
        "first_token_lateness_ms",
        "A. Due -> First Token",
        "Full path: harness/client, gateway, SGLang queueing, KV movement, compute, and first token.",
        "lateness vs replay deadline ms (symlog)",
        "0 ms deadline",
        "ms vs deadline",
        [-1000, -500, -100, 0, 50, 500, 1000, 5000, 10000, 60000],
    )
    draw_panel(
        top_b,
        "sglang_receive_to_first_token_ms",
        "B. SGLang Receive -> First Token",
        "Backend-only path after SGLang sees the replay request. This removes most harness/client overhead.",
        "backend time after SGLang receive ms (symlog)",
        "0 ms receive",
        "ms after SGLang receive",
        [0, 50, 500, 1000, 5000, 10000, 60000],
    )

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height-bottom_margin+34}" text-anchor="middle" font-size="14" font-weight="700">pressure level</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_lines = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body_lines.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_lines)}</tbody></table>"


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


def render_chart_legend(rows: list[dict[str, Any]]) -> str:
    harnesses = [harness for harness in HARNESS_LABELS if any(row["harness"] == harness for row in rows)]
    mode_items = []
    for mode, label in MODE_LABELS.items():
        mode_items.append(
            '<span class="legend-item">'
            f'<span class="legend-dot" style="background:{MODE_COLORS[mode]}"></span>'
            f"{html.escape(label)}"
            "</span>"
        )
    harness_items = []
    for harness in harnesses:
        harness_items.append(
            '<span class="legend-item">'
            f'{inline_symbol(HARNESS_SYMBOLS.get(harness, "circle"), "#334155")}'
            f"{html.escape(HARNESS_LABELS.get(harness, harness))}"
            "</span>"
        )
    return (
        '<div class="legend-card">'
        '<div class="legend-row"><strong>Mode color</strong>'
        f'<div class="legend-items">{"".join(mode_items)}</div></div>'
        '<div class="legend-row"><strong>Harness symbol</strong>'
        f'<div class="legend-items">{"".join(harness_items)}</div></div>'
        "</div>"
    )


def render_html(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    speculative_prefill_rows: list[dict[str, Any]],
    harness_priority_rows: list[dict[str, Any]],
    nat_service_priority_rows: list[dict[str, Any]],
    report_label: str,
    run_config: dict[str, str],
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    hardware_profile = os.environ.get("HARDWARE_PROFILE") or run_config.get("HARDWARE_PROFILE") or "not recorded"
    hardware_profile_path = os.environ.get("HARDWARE_PROFILE_PATH") or run_config.get("HARDWARE_PROFILE_PATH") or "not recorded"
    chart = render_pressure_chart(rows)
    chart_legend = render_chart_legend(rows)
    pressure_definition_table = render_pressure_definition_table(rows, run_config)
    summary_table = render_table(
        summary,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
            "samples",
            "median_first_token_lateness_ms",
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
            "samples",
            "median_due_to_request_start_ms",
            "median_due_to_sglang_receive_ms",
            "median_sglang_receive_to_first_token_ms",
            "median_first_token_lateness_ms",
        ],
    )
    speculative_prefill_table = render_table(speculative_prefill_rows, SPECULATIVE_PREFILL_COLUMNS)
    harness_priority_table = render_table(harness_priority_rows, HARNESS_PRIORITY_COLUMNS)
    nat_service_priority_table = render_table(nat_service_priority_rows, NAT_SERVICE_PRIORITY_COLUMNS)
    raw_table = render_table(
        rows,
        [
            "harness_label",
            "pressure_level_label",
            "mode_label",
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
.legend-dot {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}
.legend-symbol {{ width: 18px; height: 18px; flex: 0 0 auto; overflow: visible; }}
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
<p class="note">This lightweight all-harness report uses the completed workload traces directly. Each symbol is one replay request. The first panel shows full replay-deadline lateness; the second panel starts the clock when SGLang receives the replay request. Pressure levels are grouped on the x-axis; harnesses are encoded by shape; mode is encoded by color. Lower is better.</p>
<h2>Pressure Level Definitions</h2>
<p>Each pressure level is a bundled stress setting, not a full Cartesian sweep. The chart below shows only the levels marked <strong>Yes</strong> for this run.</p>
<div class="card">{pressure_definition_table}</div>
<div class="card">{chart}</div>
{chart_legend}
<h2>Harness Priority Preservation Proof</h2>
<p>This table appears when the run includes <code>pre_harness_priority_hints</code>. It proves whether the driver supplied an urgent intent before the harness, whether the gateway saw that intent or a native emitted signal, and whether it translated to SGLang priority.</p>
<div class="card">{harness_priority_table if harness_priority_rows else "<p>No pre-harness priority proof rows found in this run.</p>"}</div>
<h2>NAT Shared-Service Priority Probe</h2>
<p>This table appears when NAT is run as a shared <code>nat serve</code> service. It compares the order requests entered NAT with the order NAT emitted model calls to the gateway. If urgent work jumps ahead of older background work here, that is NAT-side priority evidence.</p>
<div class="card">{nat_service_priority_table if nat_service_priority_rows else "<p>No NAT shared-service priority probe rows found in this run.</p>"}</div>
<h2>Speculative Prefill Proof</h2>
<p>This table appears when the run includes <code>e2e_priority_hints_speculative_prefill</code>. It proves whether the Dynamo-like background <code>max_tokens=1</code> warmup was sent before replay and whether the replay showed cached-prefix reuse.</p>
<div class="card">{speculative_prefill_table if speculative_prefill_rows else "<p>No speculative prefill rows found in this run.</p>"}</div>
<h2>Summary</h2>
<div class="card">{summary_table}</div>
<h2>Delay Breakdown</h2>
<p>This table separates client/gateway arrival time from backend service time. <code>median_sglang_receive_to_first_token_ms</code> is the backend-only number used in panel B.</p>
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
    speculative_prefill_rows: list[dict[str, Any]],
    harness_priority_rows: list[dict[str, Any]],
    nat_service_priority_rows: list[dict[str, Any]],
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
        "speculative_prefill_row_count": len(speculative_prefill_rows),
        "harness_priority_row_count": len(harness_priority_rows),
        "nat_service_priority_row_count": len(nat_service_priority_rows),
        "hardware_profile": os.environ.get("HARDWARE_PROFILE") or run_config.get("HARDWARE_PROFILE", ""),
        "hardware_profile_path": os.environ.get("HARDWARE_PROFILE_PATH") or run_config.get("HARDWARE_PROFILE_PATH", ""),
        "harnesses": sorted({row["harness"] for row in rows}),
        "pressure_levels": sorted({row["pressure_level"] for row in rows}),
        "modes": sorted({row["mode"] for row in rows}),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight all-harness replay deadline report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latest-root", type=Path)
    parser.add_argument("--report-label", default=os.environ.get("REPORT_LABEL") or f"multi_harness_deadline_summary_{int(time.time())}")
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--update-latest", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = read_run_config(args.run_config or args.out_dir / "run_config.env")
    rows = collect_rows(args.root)
    summary = summarize(rows)
    speculative_prefill_rows = collect_speculative_prefill_proof(args.root, rows)
    harness_priority_rows = collect_harness_priority_proof(args.root, rows)
    nat_service_priority_rows = collect_nat_service_priority_probe(args.root)
    write_csv(args.out_dir / "global_kv_readiness_by_mode.csv", rows, RAW_COLUMNS)
    write_csv(args.out_dir / "global_kv_readiness_by_mode_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(args.out_dir / "speculative_prefill_proof.csv", speculative_prefill_rows, SPECULATIVE_PREFILL_COLUMNS)
    write_csv(args.out_dir / "harness_priority_preservation_proof.csv", harness_priority_rows, HARNESS_PRIORITY_COLUMNS)
    write_csv(args.out_dir / "nat_service_priority_probe.csv", nat_service_priority_rows, NAT_SERVICE_PRIORITY_COLUMNS)
    html_text = render_html(rows, summary, speculative_prefill_rows, harness_priority_rows, nat_service_priority_rows, args.report_label, run_config)
    report_path = args.out_dir / "master_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    write_manifest(args.out_dir / "manifest.json", args, rows, summary, speculative_prefill_rows, harness_priority_rows, nat_service_priority_rows, run_config)
    if args.latest_root and args.update_latest:
        args.latest_root.mkdir(parents=True, exist_ok=True)
        (args.latest_root / "latest_master_report.html").write_text(html_text, encoding="utf-8")
        write_manifest(args.latest_root / "latest_manifest.json", args, rows, summary, speculative_prefill_rows, harness_priority_rows, nat_service_priority_rows, run_config)
    print(f"wrote {report_path}")
    print(f"rows={len(rows)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
