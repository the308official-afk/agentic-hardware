from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from agentic_kv.nvtx import range_scope
from agentic_kv.torch_cuda_profiler import maybe_start as maybe_start_torch_profiler
from agentic_kv.torch_cuda_profiler import record_event as record_torch_profiler_event


_INSTALLED = False
_CALL_SEQ = 0
_SESSION_RE = re.compile(r"coding agent session ([A-Za-z0-9_.:-]+)")
_ACTIVE_AGENT_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("agentic_kv_active_agent_context", default={})
_AGENT_BY_NODE_ID: dict[str, dict[str, Any]] = {}
_AGENT_BY_INDEX_SIG: dict[str, dict[str, Any]] = {}
_NODE_RESIDENCY_BY_ID: dict[str, dict[str, Any]] = {}
_REQUEST_INTAKE_BY_RID: dict[str, dict[str, Any]] = {}


def _trace_path() -> Path:
    return Path(os.environ.get("AGENTIC_KV_TRACE_PATH", "artifacts/kv_movement_trace.jsonl"))


def _copy_telemetry_path() -> Path | None:
    configured = os.environ.get("AGENTIC_KV_COPY_TELEMETRY_PATH")
    if configured:
        return Path(configured)
    if os.environ.get("AGENTIC_KV_COPY_TELEMETRY_ENABLE", "0") != "1":
        return None
    return _trace_path().with_name("kv_copy_telemetry.jsonl")


def _tensor_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            summary["shape"] = [int(dim) for dim in shape]
        except Exception:
            summary["shape"] = str(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        summary["device"] = str(device)
    try:
        summary["numel"] = int(value.numel())
    except Exception:
        pass
    try:
        is_integer_tensor = "int" in str(getattr(value, "dtype", "")).lower()
        if is_integer_tensor and "numel" in summary:
            max_exact = int(os.environ.get("AGENTIC_KV_TRACE_MAX_EXACT_INDICES", "256"))
            flat = value.detach().flatten().cpu()
            summary["index_count"] = int(flat.numel())
            if int(flat.numel()) > 0:
                summary["min"] = int(flat.min().item())
                summary["max"] = int(flat.max().item())
                values = [int(item) for item in flat.tolist()]
                digest = hashlib.sha1(",".join(str(item) for item in values).encode("utf-8")).hexdigest()[:16]
                summary["sha1_16"] = digest
                if int(flat.numel()) <= max_exact:
                    summary["values"] = values
                else:
                    sample_count = int(os.environ.get("AGENTIC_KV_TRACE_TENSOR_SAMPLE", "8"))
                    if sample_count > 0:
                        summary["head"] = values[:sample_count]
                        summary["tail"] = values[-sample_count:]
            return summary
    except Exception:
        pass
    try:
        if "numel" in summary and summary["numel"] <= 16:
            summary["values"] = value.detach().cpu().tolist()
        elif "numel" in summary and summary["numel"] > 16:
            sample_count = int(os.environ.get("AGENTIC_KV_TRACE_TENSOR_SAMPLE", "8"))
            if sample_count > 0:
                flat = value.detach().flatten()
                summary["head"] = flat[:sample_count].cpu().tolist()
                summary["tail"] = flat[-sample_count:].cpu().tolist()
    except Exception:
        pass
    return summary


def _arg_value(args: tuple[Any, ...], kwargs: dict[str, Any], position: int, name: str) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return None


def _index_context(value: Any) -> dict[str, Any] | None:
    if value is None or not hasattr(value, "shape") or not hasattr(value, "numel"):
        return None
    summary = _tensor_summary(value)
    return {
        key: value
        for key, value in summary.items()
        if key
        in {
            "type",
            "shape",
            "dtype",
            "device",
            "numel",
            "index_count",
            "min",
            "max",
            "sha1_16",
            "values",
            "head",
            "tail",
        }
    }


def _index_signature(index: dict[str, Any] | None) -> str:
    if not isinstance(index, dict):
        return ""
    count = index.get("index_count") or index.get("numel")
    digest = index.get("sha1_16")
    if count and digest:
        return f"{count}:{digest}"
    values = index.get("values")
    if isinstance(values, list):
        return f"{len(values)}:{','.join(str(item) for item in values)}"
    return ""


def _copy_agent_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: context.get(key)
        for key in (
            "agent_session_id",
            "agent_phase",
            "agent_label",
            "agent_mode",
            "agent_prompt_hash",
            "agent_priority",
        )
        if context.get(key) not in (None, "", [], {})
    }


def _agent_context_from_context(context: dict[str, Any]) -> dict[str, Any]:
    direct = _copy_agent_context(context)
    if direct:
        return direct
    req = context.get("request")
    if isinstance(req, dict):
        return _copy_agent_context(req)
    requests = context.get("requests")
    if isinstance(requests, list):
        for req in requests:
            if isinstance(req, dict):
                agent = _copy_agent_context(req)
                if agent:
                    return agent
    return {}


def _request_id_from_request(request: dict[str, Any]) -> str:
    value = request.get("rid") or request.get("request_id")
    return str(value) if value not in (None, "", [], {}) else ""


def _request_index_count(request: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = request.get(key)
        if isinstance(value, dict):
            for count_key in ("index_count", "numel", "count"):
                raw = value.get(count_key)
                try:
                    if raw not in (None, ""):
                        return int(raw)
                except (TypeError, ValueError):
                    pass
    return None


def _remember_request_intake(context: dict[str, Any]) -> None:
    candidates: list[dict[str, Any]] = []
    req = context.get("request")
    if isinstance(req, dict):
        candidates.append(req)
    requests = context.get("requests")
    if isinstance(requests, list):
        candidates.extend(item for item in requests if isinstance(item, dict))
    batch = context.get("batch")
    if isinstance(batch, dict):
        batch_requests = batch.get("requests")
        if isinstance(batch_requests, list):
            candidates.extend(item for item in batch_requests if isinstance(item, dict))

    for request in candidates:
        rid = _request_id_from_request(request)
        if not rid:
            continue
        full_count = _request_index_count(request, "input_ids", "origin_input_ids", "fill_ids")
        if full_count is None:
            continue
        existing = _REQUEST_INTAKE_BY_RID.get(rid, {})
        existing_count = _request_index_count(existing, "input_ids", "origin_input_ids", "fill_ids")
        if existing_count is not None and existing_count >= full_count:
            continue
        snapshot = dict(request)
        snapshot["full_input_tokens"] = full_count
        snapshot["intake_seen_ns"] = time.time_ns()
        _REQUEST_INTAKE_BY_RID[rid] = snapshot

    max_entries = int(os.environ.get("AGENTIC_KV_REQUEST_INTAKE_REGISTRY_MAX", "20000"))
    while len(_REQUEST_INTAKE_BY_RID) > max_entries:
        try:
            oldest = next(iter(_REQUEST_INTAKE_BY_RID))
        except StopIteration:
            break
        _REQUEST_INTAKE_BY_RID.pop(oldest, None)


def _apply_request_intake_context(context: dict[str, Any]) -> None:
    candidates: list[dict[str, Any]] = []
    req = context.get("request")
    if isinstance(req, dict):
        candidates.append(req)
    requests = context.get("requests")
    if isinstance(requests, list):
        candidates.extend(item for item in requests if isinstance(item, dict))
    batch = context.get("batch")
    if isinstance(batch, dict):
        batch_requests = batch.get("requests")
        if isinstance(batch_requests, list):
            candidates.extend(item for item in batch_requests if isinstance(item, dict))

    for request in candidates:
        rid = _request_id_from_request(request)
        intake = _REQUEST_INTAKE_BY_RID.get(rid, {}) if rid else {}
        if not intake:
            continue
        intake_tokens = _request_index_count(intake, "input_ids", "origin_input_ids", "fill_ids")
        active_tokens = _request_index_count(request, "origin_input_ids", "fill_ids", "input_ids")
        if intake_tokens is not None:
            request.setdefault("ingest_input_tokens", intake_tokens)
            if active_tokens is not None and active_tokens < intake_tokens:
                request.setdefault("active_input_tokens", active_tokens)
                request.setdefault("scheduler_trimmed_tokens", intake_tokens - active_tokens)
        for key in ("input_ids", "origin_input_ids", "fill_ids"):
            if isinstance(intake.get(key), dict) and f"ingest_{key}" not in request:
                request[f"ingest_{key}"] = intake[key]
        for key, value in _copy_agent_context(intake).items():
            request.setdefault(key, value)
            context.setdefault(key, value)


def _propagated_context_from_context(context: dict[str, Any]) -> dict[str, Any]:
    agent = _agent_context_from_context(context)
    if agent:
        return agent
    sessions = context.get("agent_sessions")
    if isinstance(sessions, list) and sessions:
        return {"agent_sessions": sessions}
    return {}


def _agent_for_node_or_indices(context: dict[str, Any]) -> dict[str, Any]:
    node_id = context.get("node_id")
    if node_id not in (None, "", [], {}):
        agent = _AGENT_BY_NODE_ID.get(str(node_id), {})
        if agent:
            return agent
    node_ids = context.get("node_ids")
    if isinstance(node_ids, list):
        for item in node_ids:
            agent = _AGENT_BY_NODE_ID.get(str(item), {})
            if agent:
                return agent
    for key in ("host_indices", "device_indices"):
        value = context.get(key)
        if isinstance(value, dict):
            sig = _index_signature(value)
            if sig and sig in _AGENT_BY_INDEX_SIG:
                return _AGENT_BY_INDEX_SIG[sig]
    return {}


def _with_queue_agent_context(context: dict[str, Any]) -> None:
    ops = context.get("queued_ops")
    requests = context.get("requests")
    candidates = ops if isinstance(ops, list) else requests if isinstance(requests, list) else None
    if not isinstance(candidates, list):
        return
    sessions: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for op in candidates:
        if not isinstance(op, dict):
            continue
        agent = _agent_for_node_or_indices(op)
        if not agent:
            agent = _copy_agent_context(op)
        if not agent:
            continue
        op.update({key: value for key, value in agent.items() if op.get(key) in (None, "", [], {})})
        sig = tuple(agent.get(key) for key in ("agent_session_id", "agent_phase", "agent_prompt_hash"))
        if sig in seen:
            continue
        seen.add(sig)
        sessions.append(agent)
    if sessions:
        context["agent_sessions"] = sessions
        if len(sessions) == 1:
            context.update({key: value for key, value in sessions[0].items() if context.get(key) in (None, "", [], {})})


def _remember_agent_context(context: dict[str, Any]) -> None:
    agent = _agent_context_from_context(context)
    if not agent:
        return

    node_ids: list[Any] = []
    node_id = context.get("node_id")
    if node_id not in (None, "", [], {}):
        node_ids.append(node_id)

    req = context.get("request")
    if isinstance(req, dict):
        for key in ("last_node_id", "last_host_node_id"):
            value = req.get(key)
            if isinstance(value, list):
                node_ids.extend(value)
            elif value not in (None, "", [], {}):
                node_ids.append(value)

    for node_id in node_ids:
        _AGENT_BY_NODE_ID[str(node_id)] = agent

    for key in ("host_indices", "device_indices", "prefix_indices"):
        value = context.get(key)
        if isinstance(value, dict):
            sig = _index_signature(value)
            if sig:
                _AGENT_BY_INDEX_SIG[sig] = agent

    if isinstance(req, dict):
        for key in ("prefix_indices",):
            value = req.get(key)
            if isinstance(value, dict):
                sig = _index_signature(value)
                if sig:
                    _AGENT_BY_INDEX_SIG[sig] = agent


def _apply_known_agent_context(context: dict[str, Any]) -> dict[str, Any]:
    _remember_request_intake(context)
    _apply_request_intake_context(context)
    _with_queue_agent_context(context)
    if _agent_context_from_context(context):
        _remember_agent_context(context)
        return context

    agent: dict[str, Any] = {}
    node_id = context.get("node_id")
    if node_id not in (None, "", [], {}):
        agent = _AGENT_BY_NODE_ID.get(str(node_id), {})

    if not agent:
        agent = _agent_for_node_or_indices(context)

    if not agent:
        active = _ACTIVE_AGENT_CONTEXT.get({})
        if isinstance(active.get("agent_sessions"), list):
            context["agent_sessions"] = active["agent_sessions"]
            if len(active["agent_sessions"]) == 1:
                agent = active["agent_sessions"][0]
        else:
            agent = active

    if agent:
        context.update({key: value for key, value in agent.items() if context.get(key) in (None, "", [], {})})
        _remember_agent_context(context)

    return context


def _cache_operation_summary(operation: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(operation).__name__, "object_id": hex(id(operation))}
    for attr in ("node_id", "node_ids", "priority", "request_id", "last_hash"):
        if hasattr(operation, attr):
            try:
                out[attr] = _safe_summary(getattr(operation, attr))
            except Exception:
                pass
    for attr in ("host_indices", "device_indices"):
        if hasattr(operation, attr):
            try:
                out[attr] = _index_context(getattr(operation, attr))
            except Exception:
                pass
    return out


def _request_like_context(req: Any) -> dict[str, Any]:
    context = _req_context(req)
    for attr in ("rid", "request_id", "session_id", "agent_session_id", "agent_phase", "agent_mode", "agent_label"):
        if hasattr(req, attr) and attr not in context:
            try:
                context[attr] = _safe_summary(getattr(req, attr))
            except Exception:
                pass
    for attr in ("input_ids", "origin_input_ids", "fill_ids", "output_ids", "prefix_indices"):
        if hasattr(req, attr) and attr not in context:
            try:
                value = getattr(req, attr)
                context[attr] = _index_context(value) if hasattr(value, "numel") else _int_list_signature(value)
            except Exception:
                pass
    return {key: value for key, value in context.items() if value not in (None, "", [], {})}


def _requests_from_value(value: Any, limit: int = 16) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_request_like_context(item) for item in list(value)[:limit]]
    for attr in ("reqs", "requests", "running_reqs", "waiting_queue", "chunked_req"):
        if not hasattr(value, attr):
            continue
        try:
            nested = getattr(value, attr)
        except Exception:
            continue
        if isinstance(nested, (list, tuple)):
            return [_request_like_context(item) for item in list(nested)[:limit]]
        if nested is not None and attr == "chunked_req":
            return [_request_like_context(nested)]
    return []


def _batch_context(batch: Any) -> dict[str, Any]:
    if batch is None:
        return {}
    out: dict[str, Any] = {"type": type(batch).__name__}
    for attr in ("batch_is_full", "forward_mode", "extend_num_tokens", "seq_lens_sum", "bs", "batch_size"):
        if hasattr(batch, attr):
            try:
                out[attr] = _safe_summary(getattr(batch, attr))
            except Exception:
                pass
    requests = _requests_from_value(batch)
    if requests:
        out["requests"] = requests
        out["request_count"] = len(requests)
    return {key: value for key, value in out.items() if value not in (None, "", [], {})}


def _residency_snapshot_for_context(context: dict[str, Any]) -> dict[str, Any]:
    node_ids: list[str] = []
    node_id = context.get("node_id")
    if node_id not in (None, "", [], {}):
        node_ids.append(str(node_id))
    req = context.get("request")
    if isinstance(req, dict):
        for key in ("last_node_id", "last_host_node_id"):
            value = req.get(key)
            if value not in (None, "", [], {}):
                node_ids.append(str(value))
    states = [_NODE_RESIDENCY_BY_ID[node_id] for node_id in node_ids if node_id in _NODE_RESIDENCY_BY_ID]
    if not states:
        return {}
    counts: dict[str, int] = {}
    for state in states:
        location = str(state.get("state") or "unknown")
        counts[location] = counts.get(location, 0) + int(state.get("token_count") or 0)
    return {
        "known_node_count": len(states),
        "gpu_resident_tokens": counts.get("gpu_resident", 0),
        "host_resident_tokens": counts.get("host_resident", 0),
        "evicted_tokens": counts.get("evicted_from_gpu", 0),
        "last_residency_event": states[-1].get("event", ""),
    }


def _update_residency_state(event_name: str, method_name: str, context: dict[str, Any]) -> None:
    node_id = context.get("node_id")
    if node_id in (None, "", [], {}):
        return
    token_count = _count_from_index_summary(context.get("device_indices")) or _count_from_index_summary(context.get("host_indices")) or 0
    if method_name in {"load", "load_back", "load_to_device_per_layer", "init_load_back"}:
        state = "gpu_resident"
    elif method_name in {"write", "backup_from_device_all_layer"}:
        state = "host_resident"
    elif method_name == "evict_device":
        state = "evicted_from_gpu"
    elif method_name == "evict_host":
        state = "missing"
    else:
        return
    _NODE_RESIDENCY_BY_ID[str(node_id)] = {
        "state": state,
        "token_count": token_count,
        "event": event_name,
        "updated_ns": time.time_ns(),
        **_copy_agent_context(context),
    }


def _req_from_params(params: Any) -> Any:
    if params is None:
        return None
    try:
        req = getattr(params, "req", None)
        if req is not None:
            return req
    except Exception:
        pass
    return None


def _int_list_signature(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        return {}
    out: dict[str, Any] = {"count": len(values)}
    if values:
        digest = hashlib.sha1(",".join(str(int(item)) for item in values).encode("utf-8")).hexdigest()[:16]
        out["sha1_16"] = digest
        out["head"] = [int(item) for item in values[:8]]
        out["tail"] = [int(item) for item in values[-8:]]
    return out


def _node_id(value: Any) -> Any:
    if value is None:
        return None
    try:
        return getattr(value, "id")
    except Exception:
        return None


def _req_context(req: Any) -> dict[str, Any]:
    if req is None:
        return {}
    context: dict[str, Any] = {"type": type(req).__name__}
    for attr in ("rid", "req_pool_idx", "priority", "host_hit_length", "cache_protected_len", "kv_committed_len"):
        if hasattr(req, attr):
            try:
                context[attr] = _safe_summary(getattr(req, attr))
            except Exception:
                pass
    try:
        sampling_params = getattr(req, "sampling_params", None)
        custom_params = getattr(sampling_params, "custom_params", None)
        if isinstance(custom_params, dict):
            agentic = custom_params.get("agentic_kv")
            if isinstance(agentic, dict):
                context["agent_session_id"] = _safe_summary(agentic.get("session_id"))
                context["agent_phase"] = _safe_summary(agentic.get("phase"))
                context["agent_label"] = _safe_summary(agentic.get("label"))
                context["agent_mode"] = _safe_summary(agentic.get("mode"))
                context["agent_prompt_hash"] = _safe_summary(agentic.get("prompt_hash"))
                context["agent_priority"] = _safe_summary(agentic.get("priority"))
    except Exception:
        pass
    try:
        text = getattr(req, "origin_input_text", "") or ""
        match = _SESSION_RE.search(text)
        if match and "agent_session_id" not in context:
            context["agent_session_id"] = match.group(1)
        context["origin_input_text_head"] = text[:160]
    except Exception:
        pass
    for attr in ("origin_input_ids", "fill_ids", "output_ids"):
        if hasattr(req, attr):
            try:
                context[attr] = _int_list_signature(getattr(req, attr))
            except Exception:
                pass
    for attr in ("prefix_indices",):
        if hasattr(req, attr):
            try:
                context[attr] = _index_context(getattr(req, attr))
            except Exception:
                pass
    for attr in ("last_node", "last_host_node"):
        if hasattr(req, attr):
            try:
                context[f"{attr}_id"] = _node_id(getattr(req, attr))
            except Exception:
                pass
    return {key: value for key, value in context.items() if value not in (None, "", [], {})}


def _queue_context(obj: Any, queue_name: str) -> list[dict[str, Any]]:
    try:
        queue = getattr(obj, queue_name)
    except Exception:
        return []
    try:
        return [_cache_operation_summary(operation) for operation in list(queue)]
    except Exception:
        return []


def _kv_context(event_name: str, method_name: str, self_obj: Any, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any = None) -> dict[str, Any]:
    context: dict[str, Any] = {}

    if method_name == "load":
        context["direction"] = "host_to_device"
        context["host_indices"] = _index_context(_arg_value(args, kwargs, 0, "host_indices"))
        context["device_indices"] = _index_context(result)
        context["priority"] = _safe_summary(_arg_value(args, kwargs, 1, "priority"))
        context["node_id"] = _safe_summary(_arg_value(args, kwargs, 2, "node_id"))
    elif method_name == "write":
        context["direction"] = "device_to_host"
        context["device_indices"] = _index_context(_arg_value(args, kwargs, 0, "device_indices"))
        context["host_indices"] = _index_context(result)
        context["priority"] = _safe_summary(_arg_value(args, kwargs, 1, "priority"))
        context["node_id"] = _safe_summary(_arg_value(args, kwargs, 2, "node_id"))
    elif method_name == "evict_device":
        context["direction"] = "device_evict"
        context["device_indices"] = _index_context(_arg_value(args, kwargs, 0, "device_indices"))
    elif method_name == "evict_host":
        context["direction"] = "host_evict"
        context["host_indices"] = _index_context(_arg_value(args, kwargs, 0, "host_indices"))
    elif method_name == "prefetch":
        context["direction"] = "storage_to_host"
        context["request_id"] = _safe_summary(_arg_value(args, kwargs, 0, "request_id"))
        context["host_indices"] = _index_context(_arg_value(args, kwargs, 1, "host_indices"))
        new_input_tokens = _arg_value(args, kwargs, 2, "new_input_tokens")
        context["new_input_token_count"] = len(new_input_tokens) if isinstance(new_input_tokens, list) else None
        context["last_hash"] = _safe_summary(_arg_value(args, kwargs, 3, "last_hash"))
    elif method_name == "start_loading":
        context["direction"] = "host_to_device"
        context["queued_ops"] = _queue_context(self_obj, "load_queue")
        context["queued_op_count"] = len(context["queued_ops"])
    elif method_name == "start_writing":
        context["direction"] = "device_to_host"
        context["queued_ops"] = _queue_context(self_obj, "write_queue")
        context["queued_op_count"] = len(context["queued_ops"])
    elif method_name == "load_to_device_per_layer":
        context["direction"] = "host_to_device"
        context["host_indices"] = _index_context(_arg_value(args, kwargs, 1, "host_indices"))
        context["device_indices"] = _index_context(_arg_value(args, kwargs, 2, "device_indices"))
        context["layer_id"] = _safe_summary(_arg_value(args, kwargs, 3, "layer_id"))
        context["io_backend"] = _safe_summary(_arg_value(args, kwargs, 4, "io_backend"))
    elif method_name == "backup_from_device_all_layer":
        context["direction"] = "device_to_host"
        context["host_indices"] = _index_context(_arg_value(args, kwargs, 1, "host_indices"))
        context["device_indices"] = _index_context(_arg_value(args, kwargs, 2, "device_indices"))
        context["io_backend"] = _safe_summary(_arg_value(args, kwargs, 3, "io_backend"))
    elif method_name in {"match_prefix", "init_load_back"}:
        params = _arg_value(args, kwargs, 0, "params")
        context["request"] = _req_context(_req_from_params(params))
        if method_name == "init_load_back":
            context["direction"] = "host_to_device"
            try:
                last_host_node = getattr(params, "last_host_node", None)
                context["node_id"] = _safe_summary(_node_id(last_host_node))
                context["host_indices"] = _index_context(getattr(last_host_node, "host_value", None))
            except Exception:
                pass
            try:
                context["host_hit_length"] = _safe_summary(getattr(params, "host_hit_length", None))
            except Exception:
                pass
    elif method_name == "load_back":
        node = _arg_value(args, kwargs, 0, "node")
        context["direction"] = "host_to_device"
        context["node_id"] = _safe_summary(_node_id(node))
        try:
            context["host_indices"] = _index_context(getattr(node, "host_value", None))
        except Exception:
            pass
    elif method_name in {"cache_finished_req", "cache_unfinished_req"}:
        context["direction"] = "cache_request"
        context["request"] = _req_context(_arg_value(args, kwargs, 0, "req"))
    elif method_name in {
        "handle_generate_request",
        "_add_request_to_queue",
        "_prefetch_kvcache",
    }:
        context["direction"] = "scheduler_request"
        context["request"] = _request_like_context(_arg_value(args, kwargs, 0, "req") or _arg_value(args, kwargs, 0, "recv_req"))
    elif method_name in {
        "process_input_requests",
    }:
        context["direction"] = "scheduler_input"
        context["requests"] = _requests_from_value(_arg_value(args, kwargs, 0, "recv_reqs"))
    elif method_name in {
        "get_new_batch_prefill",
        "get_next_batch_to_run",
        "run_batch",
        "process_batch_result_prefill",
        "process_batch_result_decode",
        "process_batch_result",
        "_run_batch_prebuilt",
    }:
        context["direction"] = "scheduler_batch"
        batch = result if method_name in {"get_new_batch_prefill", "get_next_batch_to_run"} else _arg_value(args, kwargs, 0, "batch")
        context["batch"] = _batch_context(batch)
        requests = context["batch"].get("requests") if isinstance(context.get("batch"), dict) else None
        if isinstance(requests, list):
            context["requests"] = requests
    elif method_name in {
        "forward_batch_generation",
        "forward_batch_split_prefill",
        "_forward_batch_generation_dllm",
        "forward_batch_embedding",
    }:
        context["direction"] = "model_forward"
        batch = _arg_value(args, kwargs, 0, "model_worker_batch") or _arg_value(args, kwargs, 0, "batch") or _arg_value(args, kwargs, 0, "forward_batch")
        context["batch"] = _batch_context(batch)
        requests = context["batch"].get("requests") if isinstance(context.get("batch"), dict) else None
        if isinstance(requests, list):
            context["requests"] = requests

    context = {key: value for key, value in context.items() if value is not None}
    context = _apply_known_agent_context(context)
    snapshot = _residency_snapshot_for_context(context)
    if snapshot:
        context["residency_snapshot"] = snapshot
    return context


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _safe_summary(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            return {"type": type(value).__name__, "len": len(value)}
        return [_safe_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _safe_summary(v) for k, v in list(value.items())[:16]}
    if hasattr(value, "shape") and hasattr(value, "numel"):
        return _tensor_summary(value)
    if _looks_like_tree_node(value):
        return _tree_node_summary(value)
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _looks_like_tree_node(value: Any) -> bool:
    return all(hasattr(value, attr) for attr in ("id", "value", "host_value", "children", "parent"))


def _tree_node_summary(node: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(node).__name__,
        "object_id": hex(id(node)),
    }
    for attr in ("id", "lock_ref", "host_ref_counter", "priority", "hit_count"):
        if hasattr(node, attr):
            try:
                summary[attr] = getattr(node, attr)
            except Exception:
                pass
    for prop in ("evicted", "backuped"):
        try:
            summary[prop] = bool(getattr(node, prop))
        except Exception:
            pass
    try:
        summary["parent_id"] = None if node.parent is None else node.parent.id
    except Exception:
        pass
    try:
        summary["children_count"] = len(node.children)
    except Exception:
        pass
    try:
        summary["key_len"] = len(node.key) if node.key is not None else 0
    except Exception:
        pass
    try:
        summary["value"] = _safe_summary(node.value)
    except Exception:
        pass
    try:
        summary["host_value"] = _safe_summary(node.host_value)
    except Exception:
        pass
    try:
        summary["hash_value_len"] = len(node.hash_value) if node.hash_value is not None else 0
    except Exception:
        pass
    return summary


def _interesting_attr_summary(obj: Any) -> dict[str, Any]:
    """Collect small, stable-looking cache metadata without walking large graphs."""

    attrs = (
        "size",
        "page_size",
        "dtype",
        "device",
        "mem_layout",
        "hicache_size",
        "write_policy",
        "disable",
        "enable_hierarchical_cache",
    )
    nested_attrs = (
        "device_pool",
        "host_pool",
        "token_to_kv_pool",
        "token_to_kv_pool_host",
        "req_to_token_pool",
        "tree_cache",
        "hicache_controller",
    )
    out: dict[str, Any] = {
        "object_id": hex(id(obj)),
        "object_type": type(obj).__name__,
    }
    for attr in attrs:
        if hasattr(obj, attr):
            try:
                out[attr] = _safe_summary(getattr(obj, attr))
            except Exception:
                pass
    for attr in nested_attrs:
        if not hasattr(obj, attr):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        nested: dict[str, Any] = {
            "type": type(value).__name__,
            "object_id": hex(id(value)),
        }
        value_len = _safe_len(value)
        if value_len is not None:
            nested["len"] = value_len
        for nested_attr in ("size", "page_size", "dtype", "device"):
            if hasattr(value, nested_attr):
                try:
                    nested[nested_attr] = _safe_summary(getattr(value, nested_attr))
                except Exception:
                    pass
        out[attr] = nested
    return out


def _result_metadata(result: Any) -> dict[str, Any]:
    if result is None:
        return {"result_type": "None"}
    metadata = {"result_type": type(result).__name__}
    result_len = _safe_len(result)
    if result_len is not None:
        metadata["result_len"] = result_len
    if isinstance(result, tuple):
        metadata["tuple_item_types"] = [type(item).__name__ for item in result[:8]]
    return metadata


def _write_event(event: dict[str, Any]) -> None:
    event.setdefault("ts_ns", time.time_ns())
    event.setdefault("pid", os.getpid())
    path = _trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _compact_index_for_telemetry(index: Any) -> dict[str, Any]:
    if not isinstance(index, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("index_count", "numel", "min", "max", "sha1_16"):
        if index.get(key) not in (None, "", [], {}):
            out[key] = index[key]
    return out


def _copy_telemetry_event(
    *,
    phase: str,
    call_id: str,
    event_name: str,
    method_name: str,
    class_name: str,
    context: dict[str, Any],
    duration_ms: float | None = None,
    error: str | None = None,
) -> dict[str, Any] | None:
    if method_name not in {
        "load_to_device_per_layer",
        "backup_from_device_all_layer",
        "evict_device",
        "evict_host",
    }:
        return None

    direction = context.get("direction", "")
    host = _compact_index_for_telemetry(context.get("host_indices"))
    device = _compact_index_for_telemetry(context.get("device_indices"))
    event: dict[str, Any] = {
        "event": f"kv_telemetry.copy.{phase}",
        "call_id": call_id,
        "source_event": event_name,
        "class": class_name,
        "method": method_name,
        "direction": direction,
        "layer_id": context.get("layer_id", ""),
        "io_backend": context.get("io_backend", ""),
        "host_index_count": host.get("index_count") or host.get("numel") or "",
        "host_index_min": host.get("min", ""),
        "host_index_max": host.get("max", ""),
        "host_index_sha1_16": host.get("sha1_16", ""),
        "device_index_count": device.get("index_count") or device.get("numel") or "",
        "device_index_min": device.get("min", ""),
        "device_index_max": device.get("max", ""),
        "device_index_sha1_16": device.get("sha1_16", ""),
    }
    event.update(_copy_agent_context(context))
    sessions = context.get("agent_sessions")
    if isinstance(sessions, list) and sessions:
        event["agent_sessions"] = [_copy_agent_context(item) for item in sessions if isinstance(item, dict)]
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    if error:
        event["error"] = error
    return {key: value for key, value in event.items() if value not in (None, "", [], {})}


def _count_from_index_summary(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    for key in ("index_count", "numel", "count"):
        raw = value.get(key)
        try:
            if raw not in (None, ""):
                return int(raw)
        except (TypeError, ValueError):
            pass
    return None


def _count_from_request_index(request: dict[str, Any], key: str) -> int | None:
    return _count_from_index_summary(request.get(key))


def _count_from_result_index(result: Any) -> int | None:
    try:
        return _count_from_index_summary(_index_context(result))
    except Exception:
        return None


def _match_prefix_result_counts(result: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        if len(result) > 0:
            prefix_count = _count_from_result_index(result[0])
            if prefix_count is not None:
                out["cached_prefix_tokens"] = prefix_count
        if len(result) > 3 and result[3] is not None:
            out["host_hit_tokens"] = int(result[3])
    except Exception:
        pass
    return out


def _cache_path_telemetry_event(
    *,
    call_id: str,
    event_name: str,
    method_name: str,
    class_name: str,
    context: dict[str, Any],
    result: Any = None,
    duration_ms: float | None = None,
) -> dict[str, Any] | None:
    category_by_method = {
        "match_prefix": "match_prefix",
        "ready_to_load_host_cache": "ready_to_load_host_cache",
        "init_load_back": "init_load_back",
        "load_back": "load_back",
        "load": "hicache_load",
        "cache_finished_req": "cache_finished_req",
        "cache_unfinished_req": "cache_unfinished_req",
    }
    category = category_by_method.get(method_name)
    if category is None:
        return None

    req = context.get("request") if isinstance(context.get("request"), dict) else {}
    active_input_tokens = (
        _count_from_request_index(req, "origin_input_ids")
        or _count_from_request_index(req, "fill_ids")
        or _count_from_request_index(req, "input_ids")
    )
    ingest_input_tokens = req.get("ingest_input_tokens")
    try:
        ingest_input_tokens = int(ingest_input_tokens) if ingest_input_tokens not in (None, "") else None
    except (TypeError, ValueError):
        ingest_input_tokens = None
    input_tokens = ingest_input_tokens or active_input_tokens
    cached_prefix_tokens = _count_from_request_index(req, "prefix_indices")
    host_hit_tokens = req.get("host_hit_length")
    if host_hit_tokens not in (None, ""):
        try:
            host_hit_tokens = int(host_hit_tokens)
        except (TypeError, ValueError):
            host_hit_tokens = None
    else:
        host_hit_tokens = None

    if method_name == "match_prefix":
        counts = _match_prefix_result_counts(result)
        cached_prefix_tokens = max(cached_prefix_tokens or 0, counts.get("cached_prefix_tokens") or 0)
        host_hit_tokens = max(host_hit_tokens or 0, counts.get("host_hit_tokens") or 0)

    scheduler_trimmed_tokens = None
    if input_tokens is not None and active_input_tokens is not None and input_tokens > active_input_tokens:
        scheduler_trimmed_tokens = input_tokens - active_input_tokens
        cached_prefix_tokens = max(cached_prefix_tokens or 0, scheduler_trimmed_tokens)

    host_load_tokens = _count_from_index_summary(context.get("host_indices"))
    device_load_tokens = _count_from_index_summary(context.get("device_indices"))
    result_device_tokens = _count_from_result_index(result)
    if result_device_tokens is not None:
        device_load_tokens = result_device_tokens

    new_prefill_tokens = None
    if input_tokens is not None and cached_prefix_tokens is not None:
        new_prefill_tokens = max(0, input_tokens - cached_prefix_tokens)

    event: dict[str, Any] = {
        "event": "kv_telemetry.cache.end",
        "call_id": call_id,
        "source_event": f"{event_name}.end",
        "class": class_name,
        "method": method_name,
        "category": category,
        "direction": context.get("direction", ""),
        "input_tokens": input_tokens,
        "active_input_tokens": active_input_tokens,
        "ingest_input_tokens": ingest_input_tokens,
        "scheduler_trimmed_tokens": scheduler_trimmed_tokens,
        "cached_prefix_tokens": cached_prefix_tokens,
        "new_prefill_tokens_est": new_prefill_tokens,
        "host_hit_tokens": host_hit_tokens,
        "host_load_tokens": host_load_tokens,
        "device_load_tokens": device_load_tokens,
        "cache_protected_tokens": req.get("cache_protected_len", ""),
        "kv_committed_tokens": req.get("kv_committed_len", ""),
        "request_id": req.get("rid", ""),
        "node_id": context.get("node_id", ""),
    }
    event.update(_copy_agent_context(context))
    sessions = context.get("agent_sessions")
    if isinstance(sessions, list) and sessions:
        event["agent_sessions"] = [_copy_agent_context(item) for item in sessions if isinstance(item, dict)]
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    return {key: value for key, value in event.items() if value not in (None, "", [], {})}


def _request_count(context: dict[str, Any]) -> int | str:
    requests = context.get("requests")
    if isinstance(requests, list):
        return len(requests)
    batch = context.get("batch")
    if isinstance(batch, dict):
        requests = batch.get("requests")
        if isinstance(requests, list):
            return len(requests)
    if isinstance(context.get("request"), dict):
        return 1
    return ""


def _first_request_id(context: dict[str, Any]) -> Any:
    req = context.get("request")
    if isinstance(req, dict):
        return req.get("rid") or req.get("request_id")
    requests = context.get("requests")
    if isinstance(requests, list):
        for req in requests:
            if isinstance(req, dict):
                value = req.get("rid") or req.get("request_id")
                if value not in (None, ""):
                    return value
    return ""


def _scheduler_telemetry_event(
    *,
    call_id: str,
    event_name: str,
    method_name: str,
    class_name: str,
    context: dict[str, Any],
    duration_ms: float | None = None,
) -> dict[str, Any] | None:
    category_by_method = {
        "handle_generate_request": "request_received",
        "process_input_requests": "input_batch_received",
        "_add_request_to_queue": "entered_scheduler_queue",
        "get_new_batch_prefill": "selected_for_prefill",
        "get_next_batch_to_run": "selected_to_run",
        "run_batch": "run_batch",
        "_run_batch_prebuilt": "run_prebuilt_batch",
        "process_batch_result_prefill": "prefill_result_processed",
        "process_batch_result_decode": "decode_result_processed",
        "process_batch_result": "batch_result_processed",
        "_prefetch_kvcache": "scheduler_prefetch_kvcache",
    }
    category = category_by_method.get(method_name)
    if category is None:
        return None
    event: dict[str, Any] = {
        "event": "kv_telemetry.scheduler.end",
        "call_id": call_id,
        "source_event": f"{event_name}.end",
        "class": class_name,
        "method": method_name,
        "category": category,
        "request_count": _request_count(context),
        "request_id": _first_request_id(context),
    }
    event.update(_copy_agent_context(context))
    sessions = context.get("agent_sessions")
    if isinstance(sessions, list) and sessions:
        event["agent_sessions"] = [_copy_agent_context(item) for item in sessions if isinstance(item, dict)]
    snapshot = context.get("residency_snapshot")
    if isinstance(snapshot, dict):
        event.update({f"residency_{key}": value for key, value in snapshot.items()})
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    return {key: value for key, value in event.items() if value not in (None, "", [], {})}


def _prefill_telemetry_event(
    *,
    call_id: str,
    event_name: str,
    method_name: str,
    class_name: str,
    context: dict[str, Any],
    duration_ms: float | None = None,
) -> dict[str, Any] | None:
    category_by_method = {
        "forward_batch_generation": "model_forward_generation",
        "forward_batch_split_prefill": "model_forward_split_prefill",
        "_forward_batch_generation_dllm": "model_forward_dllm",
        "forward_batch_embedding": "model_forward_embedding",
    }
    category = category_by_method.get(method_name)
    if category is None:
        return None
    batch = context.get("batch") if isinstance(context.get("batch"), dict) else {}
    event: dict[str, Any] = {
        "event": "kv_telemetry.prefill.end",
        "call_id": call_id,
        "source_event": f"{event_name}.end",
        "class": class_name,
        "method": method_name,
        "category": category,
        "request_count": _request_count(context),
        "request_id": _first_request_id(context),
        "forward_mode": batch.get("forward_mode", ""),
        "extend_num_tokens": batch.get("extend_num_tokens", ""),
        "seq_lens_sum": batch.get("seq_lens_sum", ""),
    }
    event.update(_copy_agent_context(context))
    sessions = context.get("agent_sessions")
    if isinstance(sessions, list) and sessions:
        event["agent_sessions"] = [_copy_agent_context(item) for item in sessions if isinstance(item, dict)]
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    return {key: value for key, value in event.items() if value not in (None, "", [], {})}


def _write_copy_telemetry(event: dict[str, Any] | None) -> None:
    if not event:
        return
    path = _copy_telemetry_path()
    if path is None:
        return
    event.setdefault("ts_ns", time.time_ns())
    event.setdefault("pid", os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _env_set(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {item.strip() for item in re.split(r"[,\s]+", raw) if item.strip()}


def _context_has_agent_phase(context: dict[str, Any], phase: str) -> bool:
    if not phase:
        return True
    if context.get("agent_phase") == phase:
        return True
    req = context.get("request")
    if isinstance(req, dict) and req.get("agent_phase") == phase:
        return True
    sessions = context.get("agent_sessions")
    if isinstance(sessions, list):
        return any(isinstance(item, dict) and item.get("agent_phase") == phase for item in sessions)
    return False


def _should_start_torch_profiler(event_name: str, context: dict[str, Any]) -> bool:
    start_events = _env_set("AGENTIC_KV_TORCH_PROFILER_START_EVENTS")
    if start_events and event_name not in start_events:
        return False
    start_phase = os.environ.get("AGENTIC_KV_TORCH_PROFILER_START_AGENT_PHASE", "").strip()
    if start_phase and not _context_has_agent_phase(context, start_phase):
        return False
    return True


def _wrap_method(cls: type, method_name: str, event_name: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_agentic_kv_wrapped", False):
        return

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _CALL_SEQ
        _CALL_SEQ += 1
        call_id = f"{os.getpid()}-{_CALL_SEQ}"
        start_ns = time.perf_counter_ns()
        nvtx_name = f"agentic_kv:{event_name}:{cls.__name__}.{method_name}"
        start_kv_context = _kv_context(event_name, method_name, self, args, kwargs)
        agent_context = _propagated_context_from_context(start_kv_context)
        context_token = _ACTIVE_AGENT_CONTEXT.set(agent_context) if agent_context else None
        start_event = {
            "event": f"{event_name}.start",
            "call_id": call_id,
            "class": cls.__name__,
            "method": method_name,
            "self": _interesting_attr_summary(self),
            "args": [_safe_summary(arg) for arg in args],
            "kwargs": {key: _safe_summary(value) for key, value in kwargs.items()},
            "kv_context": start_kv_context,
        }
        _write_event(start_event)
        _write_copy_telemetry(
            _copy_telemetry_event(
                phase="start",
                call_id=call_id,
                event_name=event_name,
                method_name=method_name,
                class_name=cls.__name__,
                context=start_kv_context,
            )
        )
        if _should_start_torch_profiler(event_name, start_kv_context):
            maybe_start_torch_profiler(nvtx_name)
        try:
            with range_scope(nvtx_name):
                result = original(self, *args, **kwargs)
        except Exception as exc:
            error_context = _kv_context(event_name, method_name, self, args, kwargs)
            _write_event(
                {
                    "event": f"{event_name}.error",
                    "call_id": call_id,
                    "class": cls.__name__,
                    "method": method_name,
                    "duration_ms": (time.perf_counter_ns() - start_ns) / 1_000_000,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _write_copy_telemetry(
                _copy_telemetry_event(
                    phase="error",
                    call_id=call_id,
                    event_name=event_name,
                    method_name=method_name,
                    class_name=cls.__name__,
                    context=error_context,
                    duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
                    error=str(exc),
                )
            )
            if context_token is not None:
                _ACTIVE_AGENT_CONTEXT.reset(context_token)
            raise

        try:
            end_context = _kv_context(event_name, method_name, self, args, kwargs, result)
            duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            _update_residency_state(event_name, method_name, end_context)
            _write_event(
                {
                    "event": f"{event_name}.end",
                    "call_id": call_id,
                    "class": cls.__name__,
                    "method": method_name,
                    "duration_ms": duration_ms,
                    "self": _interesting_attr_summary(self),
                    "result_metadata": _result_metadata(result),
                    "result": _safe_summary(result),
                    "kv_context": end_context,
                }
            )
            _write_copy_telemetry(
                _copy_telemetry_event(
                    phase="end",
                    call_id=call_id,
                    event_name=event_name,
                    method_name=method_name,
                    class_name=cls.__name__,
                    context=end_context,
                    duration_ms=duration_ms,
                )
            )
            cache_event = _cache_path_telemetry_event(
                call_id=call_id,
                event_name=event_name,
                method_name=method_name,
                class_name=cls.__name__,
                context=end_context,
                result=result,
                duration_ms=duration_ms,
            )
            if cache_event:
                _write_event(cache_event)
            scheduler_event = _scheduler_telemetry_event(
                call_id=call_id,
                event_name=event_name,
                method_name=method_name,
                class_name=cls.__name__,
                context=end_context,
                duration_ms=duration_ms,
            )
            if scheduler_event:
                _write_event(scheduler_event)
            prefill_event = _prefill_telemetry_event(
                call_id=call_id,
                event_name=event_name,
                method_name=method_name,
                class_name=cls.__name__,
                context=end_context,
                duration_ms=duration_ms,
            )
            if prefill_event:
                _write_event(prefill_event)
            record_torch_profiler_event(nvtx_name)
            return result
        finally:
            if context_token is not None:
                _ACTIVE_AGENT_CONTEXT.reset(context_token)

    wrapper._agentic_kv_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapper)


def _try_patch(importer: Callable[[], Any], class_name: str, methods: dict[str, str]) -> None:
    try:
        module = importer()
        cls = getattr(module, class_name)
    except Exception as exc:
        if os.environ.get("AGENTIC_KV_TRACE_DEBUG", "0") == "1":
            _write_event(
                {
                    "event": "trace.patch_skip",
                    "class": class_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        return

    for method_name, event_name in methods.items():
        _wrap_method(cls, method_name, event_name)


def install_sglang_kv_trace() -> None:
    """Install non-invasive SGLang KV movement trace hooks."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _write_event({"event": "trace.install.start"})

    _try_patch(
        lambda: __import__("sglang.srt.managers.cache_controller", fromlist=["HiCacheController"]),
        "HiCacheController",
        {
            "load": "hicache.load",
            "write": "hicache.write",
            "evict_device": "hicache.evict_device",
            "evict_host": "hicache.evict_host",
            "prefetch": "hicache.prefetch",
            "start_loading": "hicache.start_loading",
            "start_writing": "hicache.start_writing",
        },
    )
    _try_patch(
        lambda: __import__("sglang.srt.mem_cache.hiradix_cache", fromlist=["HiRadixCache"]),
        "HiRadixCache",
        {
            "match_prefix": "hiradix.match_prefix",
            "cache_finished_req": "hiradix.cache_finished_req",
            "cache_unfinished_req": "hiradix.cache_unfinished_req",
            "evict": "hiradix.evict",
            "load_back": "hiradix.load_back",
            "init_load_back": "hiradix.init_load_back",
            "ready_to_load_host_cache": "hiradix.ready_to_load_host_cache",
        },
    )
    _try_patch(
        lambda: __import__("sglang.srt.mem_cache.radix_cache", fromlist=["RadixCache"]),
        "RadixCache",
        {
            "match_prefix": "radix.match_prefix",
            "cache_finished_req": "radix.cache_finished_req",
            "cache_unfinished_req": "radix.cache_unfinished_req",
            "evict": "radix.evict",
        },
    )
    memory_pool_host = lambda: __import__("sglang.srt.mem_cache.memory_pool_host", fromlist=["HostPoolGroup"])
    for class_name in (
        "HostPoolGroup",
        "MHATokenToKVPoolHost",
        "MLATokenToKVPoolHost",
        "NSATokenToKVPoolHost",
        "MambaPoolHost",
    ):
        _try_patch(
            memory_pool_host,
            class_name,
            {
                "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
                "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
            },
        )

    if os.environ.get("AGENTIC_KV_TRACE_SCHEDULER", "0") == "1":
        _try_patch(
            lambda: __import__("sglang.srt.managers.scheduler", fromlist=["Scheduler"]),
            "Scheduler",
            {
                "handle_generate_request": "scheduler.handle_generate_request",
                "_add_request_to_queue": "scheduler.add_request_to_queue",
                "_prefetch_kvcache": "scheduler.prefetch_kvcache",
                "_run_batch_prebuilt": "scheduler.run_batch_prebuilt",
                "process_batch_result": "scheduler.process_batch_result",
                "process_batch_result_prefill": "scheduler.process_batch_result_prefill",
                "process_batch_result_decode": "scheduler.process_batch_result_decode",
                "run_batch": "scheduler.run_batch",
                "process_input_requests": "scheduler.process_input_requests",
                "get_next_batch_to_run": "scheduler.get_next_batch_to_run",
                "get_new_batch_prefill": "scheduler.get_new_batch_prefill",
                "event_loop_overlap": "scheduler.event_loop_overlap",
                "event_loop_normal": "scheduler.event_loop_normal",
            },
        )
        _try_patch(
            lambda: __import__("sglang.srt.managers.tp_worker", fromlist=["TpModelWorker"]),
            "TpModelWorker",
            {
                "forward_batch_generation": "worker.forward_batch_generation",
                "forward_batch_split_prefill": "worker.forward_batch_split_prefill",
                "_forward_batch_generation_dllm": "worker.forward_batch_generation_dllm",
                "forward_batch_embedding": "worker.forward_batch_embedding",
            },
        )

    _write_event({"event": "trace.install.end"})
