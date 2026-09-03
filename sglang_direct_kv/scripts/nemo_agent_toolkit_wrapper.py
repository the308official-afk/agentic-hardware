#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def write_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts_ns", time.time_ns())
    row.setdefault("pid", os.getpid())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def pre_harness_priority_enabled(meta: dict[str, Any]) -> bool:
    return str(meta.get("mode") or "") == "pre_harness_priority_hints"


def nat_inferred_priority_enabled(meta: dict[str, Any]) -> bool:
    return str(meta.get("mode") or "") in {"nat_inferred_priority_hints", "harness_emitted_signals"}


def nat_priority_config_fields(meta: dict[str, Any]) -> dict[str, Any]:
    if not pre_harness_priority_enabled(meta):
        return {}
    intent = as_dict(meta.get("priority_intent"))
    if str(intent.get("class") or "") != "urgent":
        return {}
    return {
        "service_tier": "priority",
        "extra_body": {
            "agentic_hints": {
                "priority_class": "urgent",
                "reason": str(intent.get("reason") or "tool_replay_deadline"),
                "deadline_ms": str(intent.get("deadline_ms") or ""),
            }
        },
    }


def compact(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def inferred_priority_profile(meta: dict[str, Any]) -> dict[str, Any]:
    profile = meta.get("nat_inferred_priority_profile")
    return profile if isinstance(profile, dict) else {}


def build_prediction_lookup(profile: dict[str, Any]) -> Any:
    from nat.profiler.prediction_trie.data_models import LLMCallPrediction
    from nat.profiler.prediction_trie.data_models import PredictionTrieNode
    from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup

    root = PredictionTrieNode(name="root")
    nodes = profile.get("workflow_nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            workflow_path = node.get("workflow_path")
            if isinstance(workflow_path, list) and workflow_path:
                workflow_node = str(workflow_path[-1])
            else:
                workflow_node = str(node.get("workflow_node") or "")
            if not workflow_node:
                continue
            root.children[workflow_node] = PredictionTrieNode(
                name=workflow_node,
                predictions_any_index=LLMCallPrediction(
                    latency_sensitivity=int(node.get("latency_sensitivity") or 2),
                ),
            )
    return PredictionTrieLookup(root)


async def run_nat_inferred_transport(request: dict[str, Any], meta: dict[str, Any], prompt: str) -> int:
    from nat.builder.context import Context
    from nat.llm.dynamo_llm import CacheControlMode
    from nat.llm.dynamo_llm import CachePinType
    from nat.llm.dynamo_llm import DynamoPrefixContext
    from nat.llm.dynamo_llm import _DynamoTransport

    profile = inferred_priority_profile(meta)
    transport = _DynamoTransport(
        transport=httpx.AsyncHTTPTransport(),
        total_requests=int(meta.get("nat_inferred_prefix_total_requests") or 10),
        osl=int(meta.get("nat_inferred_prefix_osl") or 512),
        iat=int(meta.get("nat_inferred_prefix_iat") or 50),
        prediction_lookup=build_prediction_lookup(profile),
        cache_pin_type=CachePinType.EPHEMERAL,
        cache_control_mode=CacheControlMode.ALWAYS,
        max_sensitivity=1000,
    )
    payload = {
        "model": str(request["model"]),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(meta.get("max_tokens") or 8),
        "temperature": 0,
        "stream": False,
    }
    workflow_node = str(meta.get("workflow_node") or "initial_turn")
    prefix_id = str(meta.get("session_id") or "nat-inferred-main")
    with DynamoPrefixContext.scope(prefix_id):
        with Context.scope(
            workflow_run_id=f"nat-inferred-{meta.get('label', '')}",
            function_path_stack=[workflow_node],
        ):
            async with httpx.AsyncClient(transport=transport, timeout=None) as client:
                response = await client.post(f"{str(request['gateway_base']).rstrip('/')}/v1/chat/completions", json=payload)
                response.raise_for_status()
                return response.status_code


def trace_has_gateway_emit(path: Path, label: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "m27.request.start" and str(row.get("label") or row.get("request_id") or "") == label:
                    return True
    except OSError:
        return False
    return False


def build_workflow_config(request: dict[str, Any]) -> dict[str, Any]:
    meta = as_dict(request.get("meta"))
    return {
        "llms": {
            "harness_llm": {
                "_type": "openai",
                "api_key": "dummy",
                "base_url": f"{str(request['gateway_base']).rstrip('/')}/v1",
                "model_name": str(request["model"]),
                "api_type": "chat_completion",
                "temperature": 0,
                "max_tokens": int(meta.get("max_tokens") or 8),
                "request_timeout": 900,
                "max_retries": 0,
                **nat_priority_config_fields(meta),
            }
        },
        "workflow": {
            "_type": "chat_completion",
            "llm_name": "harness_llm",
            "system_prompt": "You are a concise coding-agent wireability probe. Do not use tools.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable NeMo Agent Toolkit instrumentation wrapper.")
    parser.add_argument("--request-json", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    meta = as_dict(request.get("meta"))
    label = str(meta.get("label") or "")
    trace_path = Path(str(meta.get("_trace_path") or "")) if meta.get("_trace_path") else None
    config_path = Path(str(request["config_path"]))
    nat_log_path = Path(str(request["nat_log_path"]))
    nat_bin = str(request.get("nat_bin") or "nat")
    prompt = str(request["prompt"])

    common = {
        "harness": "nemo_agent_toolkit",
        "session_id": meta.get("session_id", ""),
        "phase": meta.get("phase", ""),
        "mode": meta.get("mode", ""),
        "label": label,
        "request_id": label,
        "priority_intent": meta.get("priority_intent", ""),
        "harness_input_priority_signal": meta.get("harness_input_priority_signal", ""),
        "harness_input_priority_signal_source": meta.get("harness_input_priority_signal_source", ""),
        "workflow_node": meta.get("workflow_node", ""),
        "workflow_node_goal": meta.get("workflow_node_goal", ""),
        "inference_source": meta.get("inference_source", ""),
        "expected_inferred_priority": meta.get("expected_inferred_priority", ""),
    }
    wrapper_start_ns = time.time_ns()
    write_jsonl(trace_path, {"event": "m27.nat_wrapper.request_received", **common})

    if nat_inferred_priority_enabled(meta):
        profile = inferred_priority_profile(meta)
        profile_path = config_path.parent / "nat_inferred_priority_profile.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_jsonl(
            trace_path,
            {
                "event": "m27.nat_wrapper.config_written",
                **common,
                "nat_config_path": str(profile_path),
                "nat_priority_fields": compact(profile),
                "nat_priority_field_source": "prediction_trie.workflow_path.latency_sensitivity",
            },
        )
        write_jsonl(
            trace_path,
            {
                "event": "m27.nat_wrapper.process_start",
                **common,
                "nat_cmd": "nat_dynamo_transport_direct_prediction_lookup",
            },
        )
        return_code = 0
        error = ""
        try:
            asyncio.run(run_nat_inferred_transport(request, meta, prompt))
        except Exception as exc:  # noqa: BLE001
            return_code = 1
            error = f"{type(exc).__name__}: {exc}"
            nat_log_path.parent.mkdir(parents=True, exist_ok=True)
            nat_log_path.write_text(error + "\n", encoding="utf-8")
        gateway_emit_seen = bool(trace_path and label and trace_has_gateway_emit(trace_path, label))
        if gateway_emit_seen:
            write_jsonl(trace_path, {"event": "m27.nat_wrapper.first_gateway_emit", **common})
        wrapper_end_ns = time.time_ns()
        write_jsonl(
            trace_path,
            {
                "event": "m27.nat_wrapper.process_exit",
                **common,
                "returncode": return_code,
                "nat_log_path": str(nat_log_path),
                "gateway_emit_seen": gateway_emit_seen,
                "wrapper_total_ms": round((wrapper_end_ns - wrapper_start_ns) / 1_000_000.0, 3),
                "error": error,
            },
        )
        if return_code:
            sys.exit(return_code)
        return

    config = build_workflow_config(request)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    priority_fields = nat_priority_config_fields(meta)
    write_jsonl(
        trace_path,
        {
            "event": "m27.nat_wrapper.config_written",
            **common,
            "nat_config_path": str(config_path),
            "nat_priority_fields": compact(priority_fields),
            "nat_priority_field_source": "openai_provider_extra_fields" if priority_fields else "",
        },
    )

    cmd = [
        nat_bin,
        "run",
        "--config_file",
        str(config_path),
        "--input",
        prompt,
        "--user_id",
        "harness_probe_user",
        "--conversation_id",
        label,
    ]
    env = os.environ.copy()
    env.update(as_dict(request.get("env")))
    env.setdefault("OPENAI_API_KEY", "dummy")

    nat_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(trace_path, {"event": "m27.nat_wrapper.process_start", **common, "nat_cmd": " ".join(cmd)})
    gateway_emit_seen = False
    with nat_log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(request.get("cwd") or "/tmp"),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while proc.poll() is None:
            if not gateway_emit_seen and trace_path and label and trace_has_gateway_emit(trace_path, label):
                gateway_emit_seen = True
                write_jsonl(trace_path, {"event": "m27.nat_wrapper.first_gateway_emit", **common})
            time.sleep(0.1)
        return_code = proc.returncode

    if not gateway_emit_seen and trace_path and label and trace_has_gateway_emit(trace_path, label):
        gateway_emit_seen = True
        write_jsonl(trace_path, {"event": "m27.nat_wrapper.first_gateway_emit", **common})
    wrapper_end_ns = time.time_ns()
    write_jsonl(
        trace_path,
        {
            "event": "m27.nat_wrapper.process_exit",
            **common,
            "returncode": return_code,
            "nat_log_path": str(nat_log_path),
            "gateway_emit_seen": gateway_emit_seen,
            "wrapper_total_ms": round((wrapper_end_ns - wrapper_start_ns) / 1_000_000.0, 3),
        },
    )
    if return_code:
        sys.exit(return_code)


if __name__ == "__main__":
    main()
