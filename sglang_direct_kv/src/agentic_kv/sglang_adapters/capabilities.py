from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from . import select_adapter_name


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""


def _server_arg_choices(name: str) -> list[str]:
    try:
        server_args = importlib.import_module("sglang.srt.server_args")
    except Exception:
        return []
    value = getattr(server_args, name, None)
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []


def _module_help(module_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "module": module_name,
        "available": False,
        "exit_code": None,
        "help": "",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result
    result["exit_code"] = proc.returncode
    result["help"] = proc.stdout or ""
    result["available"] = proc.returncode == 0
    return result


def _help_flag_present(help_text: str, flag: str) -> bool:
    return flag in help_text


def _help_choice_present(help_text: str, flag: str, choice: str) -> bool:
    if flag not in help_text:
        return False
    flag_pos = help_text.find(flag)
    window = help_text[flag_pos : flag_pos + 800]
    return bool(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(choice)}(?![A-Za-z0-9_-])", window))


def _entrypoint_launch_capabilities(module_name: str) -> dict[str, Any]:
    help_result = _module_help(module_name)
    help_text = str(help_result.get("help") or "")
    return {
        "module": module_name,
        "available": bool(help_result.get("available")),
        "exit_code": help_result.get("exit_code"),
        "error_type": help_result.get("error_type"),
        "error": help_result.get("error"),
        "enable_priority_scheduling_flag_supported": _help_flag_present(
            help_text, "--enable-priority-scheduling"
        ),
        "radix_eviction_policy_flag_supported": _help_flag_present(
            help_text, "--radix-eviction-policy"
        ),
        "radix_priority_eviction_supported": _help_choice_present(
            help_text, "--radix-eviction-policy", "priority"
        ),
        "priority_schedule_policy_supported": _help_choice_present(
            help_text, "--schedule-policy", "priority"
        ),
    }


def _module_class_methods(module_name: str, class_name: str, methods: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "module": module_name,
        "class": class_name,
        "module_found": False,
        "class_found": False,
        "missing_methods": methods,
    }
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result
    result["module_found"] = True
    cls = getattr(module, class_name, None)
    if cls is None:
        return result
    result["class_found"] = True
    result["missing_methods"] = [method for method in methods if not hasattr(cls, method)]
    return result


def _source_root(package: str) -> Path | None:
    spec = importlib.util.find_spec(package)
    if spec is None:
        return None
    locations = spec.submodule_search_locations
    if locations:
        return Path(list(locations)[0])
    if spec.origin:
        return Path(spec.origin).parent
    return None


def _read_relative_source(root: Path | None, relative: str) -> str:
    if root is None:
        return ""
    path = root / relative
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _object_source(module_name: str, object_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        obj = module
        for part in object_name.split("."):
            obj = getattr(obj, part)
        return inspect.getsource(obj)
    except Exception:
        return ""


def _contains_all(text: str, terms: tuple[str, ...]) -> bool:
    return all(term in text for term in terms)


def collect_cache_signal_path_audit() -> dict[str, Any]:
    """Inspect the installed SGLang source for cache-signal consumption paths.

    This is intentionally a static audit. It answers whether the installed
    SGLang code appears to have a real code path for a field, while runtime
    traces answer whether a particular request exercised that path.
    """

    root = _source_root("sglang")
    source_files = {
        "openai_protocol": _read_relative_source(root, "srt/entrypoints/openai/protocol.py"),
        "openai_adapter": _read_relative_source(root, "srt/entrypoints/openai/adapter.py"),
        "io_struct": _read_relative_source(root, "srt/managers/io_struct.py"),
        "schedule_batch": _read_relative_source(root, "srt/managers/schedule_batch.py"),
        "schedule_policy": _read_relative_source(root, "srt/managers/schedule_policy.py"),
        "radix_cache": _read_relative_source(root, "srt/mem_cache/radix_cache.py"),
        "hiradix_cache": _read_relative_source(root, "srt/mem_cache/hiradix_cache.py"),
    }
    combined = "\n".join(source_files.values())
    prompt_cache_key_seen = "prompt_cache_key" in combined
    cache_salt_seen = "cache_salt" in combined
    extra_key_seen = "extra_key" in combined
    cache_control_seen = "cache_control" in combined
    native_cache_bridge_seen = "native_cache_bridge" in combined

    checks = [
        {
            "signal": "prompt_cache_key",
            "expected_native_path": "OpenAI-compatible request field mapped to cache_salt or extra_key",
            "source_evidence": "prompt_cache_key found in installed SGLang source" if prompt_cache_key_seen else "prompt_cache_key not found in installed SGLang source",
            "parser_seen": prompt_cache_key_seen,
            "generate_req_input_seen": _contains_all(source_files["io_struct"], ("GenerateReqInput", "prompt_cache_key")),
            "req_field_seen": _contains_all(source_files["schedule_batch"], ("Req", "prompt_cache_key")),
            "radix_namespace_seen": _contains_all(source_files["radix_cache"], ("RadixKey", "extra_key")),
            "verdict": (
                "native prompt_cache_key path appears present"
                if prompt_cache_key_seen and ("cache_salt" in combined or "extra_key" in combined)
                else "native prompt_cache_key path not proven by source audit"
            ),
        },
        {
            "signal": "cache_salt",
            "expected_native_path": "request cache_salt becomes request extra_key / RadixKey extra_key",
            "source_evidence": "cache_salt found in installed SGLang source" if cache_salt_seen else "cache_salt not found in installed SGLang source",
            "parser_seen": cache_salt_seen,
            "generate_req_input_seen": _contains_all(source_files["io_struct"], ("GenerateReqInput", "cache_salt")),
            "req_field_seen": _contains_all(source_files["schedule_batch"], ("extra_key", "cache_salt")),
            "radix_namespace_seen": _contains_all(source_files["radix_cache"], ("RadixKey", "extra_key")),
            "verdict": (
                "native cache_salt namespace path appears present"
                if cache_salt_seen and extra_key_seen and _contains_all(source_files["radix_cache"], ("RadixKey", "extra_key"))
                else "native cache_salt namespace path not proven by source audit"
            ),
        },
        {
            "signal": "custom_params.nvext.cache_control",
            "expected_native_path": "custom cache-control metadata directly changes SGLang cache policy",
            "source_evidence": "cache_control found in installed SGLang source" if cache_control_seen else "cache_control not found in installed SGLang source",
            "parser_seen": cache_control_seen,
            "generate_req_input_seen": _contains_all(source_files["io_struct"], ("GenerateReqInput", "cache_control")),
            "req_field_seen": _contains_all(source_files["schedule_batch"], ("cache_control", "Req")),
            "radix_namespace_seen": _contains_all(source_files["radix_cache"], ("cache_control",)),
            "verdict": (
                "native cache_control action path appears present"
                if cache_control_seen and _contains_all(source_files["radix_cache"], ("cache_control",))
                else "native cache_control action path not proven by source audit"
            ),
        },
        {
            "signal": "native_cache_bridge",
            "expected_native_path": "testbed bridge metadata directly changes SGLang cache policy",
            "source_evidence": "native_cache_bridge found in installed SGLang source" if native_cache_bridge_seen else "native_cache_bridge not found in installed SGLang source",
            "parser_seen": native_cache_bridge_seen,
            "generate_req_input_seen": _contains_all(source_files["io_struct"], ("GenerateReqInput", "native_cache_bridge")),
            "req_field_seen": _contains_all(source_files["schedule_batch"], ("native_cache_bridge", "Req")),
            "radix_namespace_seen": _contains_all(source_files["radix_cache"], ("native_cache_bridge",)),
            "verdict": (
                "native bridge action path appears present"
                if native_cache_bridge_seen and _contains_all(source_files["radix_cache"], ("native_cache_bridge",))
                else "native bridge action path not proven by source audit"
            ),
        },
        {
            "signal": "priority",
            "expected_native_path": "request priority reaches scheduler and cache insertion/eviction priority",
            "source_evidence": "priority found in scheduler/cache source" if "priority" in combined else "priority not found in scheduler/cache source",
            "parser_seen": "priority" in combined,
            "generate_req_input_seen": _contains_all(source_files["io_struct"], ("GenerateReqInput", "priority")),
            "req_field_seen": _contains_all(source_files["schedule_batch"], ("priority", "Req")),
            "radix_namespace_seen": _contains_all(source_files["radix_cache"], ("priority", "InsertParams")),
            "verdict": (
                "native priority path appears present"
                if _contains_all(source_files["radix_cache"], ("priority", "InsertParams"))
                else "native priority path not proven by source audit"
            ),
        },
    ]
    module_presence = {
        name: bool(text)
        for name, text in source_files.items()
    }
    relevant_objects = {
        "RadixCache.match_prefix": bool(_object_source("sglang.srt.mem_cache.radix_cache", "RadixCache.match_prefix")),
        "RadixCache.cache_finished_req": bool(_object_source("sglang.srt.mem_cache.radix_cache", "RadixCache.cache_finished_req")),
        "HiRadixCache.match_prefix": bool(_object_source("sglang.srt.mem_cache.hiradix_cache", "HiRadixCache.match_prefix")),
    }
    return {
        "sglang_package_root": str(root or ""),
        "module_presence": module_presence,
        "relevant_objects": relevant_objects,
        "checks": checks,
        "summary": {
            "prompt_cache_key_native_path": checks[0]["verdict"],
            "cache_salt_native_path": checks[1]["verdict"],
            "cache_control_native_path": checks[2]["verdict"],
            "native_cache_bridge_path": checks[3]["verdict"],
            "priority_native_path": checks[4]["verdict"],
        },
    }


def collect_sglang_capabilities() -> dict[str, Any]:
    from . import get_hook_targets

    sglang_version = _package_version("sglang")
    radix_choices = _server_arg_choices("RADIX_EVICTION_POLICY_CHOICES")
    schedule_choices = _server_arg_choices("SCHEDULE_POLICY_CHOICES")
    entrypoints = {
        "sglang.launch_server": _entrypoint_launch_capabilities("sglang.launch_server"),
        "dynamo.sglang": _entrypoint_launch_capabilities("dynamo.sglang"),
    }
    help_has_priority_flag = any(
        row.get("enable_priority_scheduling_flag_supported") for row in entrypoints.values()
    )
    help_has_radix_policy_flag = any(
        row.get("radix_eviction_policy_flag_supported") for row in entrypoints.values()
    )
    help_has_radix_priority = any(
        row.get("radix_priority_eviction_supported") for row in entrypoints.values()
    )
    help_has_schedule_priority = any(
        row.get("priority_schedule_policy_supported") for row in entrypoints.values()
    )
    hook_targets = get_hook_targets(include_scheduler=True)
    hook_probe = [
        _module_class_methods(target.module, target.class_name, list(target.methods))
        for target in hook_targets
    ]
    return {
        "sglang_version": sglang_version,
        "selected_adapter": select_adapter_name(sglang_version),
        "packages": {
            "sglang": sglang_version,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "sgl-kernel": _package_version("sgl-kernel"),
        },
        "launch_capabilities": {
            "entrypoints": entrypoints,
            "radix_eviction_policy_choices": radix_choices,
            "radix_priority_eviction_supported": "priority" in radix_choices or help_has_radix_priority,
            "schedule_policy_choices": schedule_choices,
            "priority_schedule_policy_supported": "priority" in schedule_choices or help_has_schedule_priority,
            "enable_priority_scheduling_flag_supported": help_has_priority_flag,
            "radix_eviction_policy_flag_supported": help_has_radix_policy_flag,
        },
        "hook_probe": hook_probe,
        "cache_signal_path_audit": collect_cache_signal_path_audit(),
        "hook_probe_summary": {
            "targets": len(hook_probe),
            "classes_found": sum(1 for row in hook_probe if row.get("class_found")),
            "targets_with_missing_methods": sum(1 for row in hook_probe if row.get("missing_methods")),
        },
    }


def _markdown(data: dict[str, Any]) -> str:
    lines = [
        "# SGLang Capability Probe",
        "",
        f"- SGLang version: `{data.get('sglang_version') or 'not installed'}`",
        f"- Selected adapter: `{data.get('selected_adapter')}`",
        f"- Radix priority eviction supported: `{data['launch_capabilities'].get('radix_priority_eviction_supported')}`",
        f"- Priority schedule policy supported: `{data['launch_capabilities'].get('priority_schedule_policy_supported')}`",
        f"- `--enable-priority-scheduling` flag supported: `{data['launch_capabilities'].get('enable_priority_scheduling_flag_supported')}`",
        f"- `--radix-eviction-policy` flag supported: `{data['launch_capabilities'].get('radix_eviction_policy_flag_supported')}`",
        "",
        "## Launch Entrypoints",
        "",
        "| entrypoint | help ok | priority flag | radix policy flag | radix priority choice | schedule priority choice |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in (data.get("launch_capabilities", {}).get("entrypoints") or {}).items():
        lines.append(
            "| "
            f"`{name}` | "
            f"`{row.get('available')}` | "
            f"`{row.get('enable_priority_scheduling_flag_supported')}` | "
            f"`{row.get('radix_eviction_policy_flag_supported')}` | "
            f"`{row.get('radix_priority_eviction_supported')}` | "
            f"`{row.get('priority_schedule_policy_supported')}` |"
        )
    lines.extend(
        [
            "",
            "## Hook Probe",
            "",
            "| module | class | class_found | missing_methods |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in data.get("hook_probe", []):
        missing = ", ".join(row.get("missing_methods") or [])
        lines.append(
            f"| `{row.get('module')}` | `{row.get('class')}` | `{row.get('class_found')}` | `{missing}` |"
        )
    cache_audit = data.get("cache_signal_path_audit", {})
    if isinstance(cache_audit, dict):
        lines.extend(
            [
                "",
                "## Cache Signal Path Audit",
                "",
                f"- SGLang package root: `{cache_audit.get('sglang_package_root') or 'not found'}`",
                "",
                "| signal | expected native path | parser seen | GenerateReqInput seen | Req field seen | Radix/cache path seen | verdict |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in cache_audit.get("checks", []):
            lines.append(
                "| "
                f"`{row.get('signal')}` | "
                f"{row.get('expected_native_path')} | "
                f"`{row.get('parser_seen')}` | "
                f"`{row.get('generate_req_input_seen')}` | "
                f"`{row.get('req_field_seen')}` | "
                f"`{row.get('radix_namespace_seen')}` | "
                f"{row.get('verdict')} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe installed SGLang capabilities for adapter migration.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    data = collect_sglang_capabilities()
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote SGLang capability JSON to {args.out}")
    else:
        print(text)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_markdown(data), encoding="utf-8")
        print(f"Wrote SGLang capability markdown to {args.out_md}")


if __name__ == "__main__":
    main()
