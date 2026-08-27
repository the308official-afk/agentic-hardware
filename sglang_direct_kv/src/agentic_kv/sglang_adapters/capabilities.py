from __future__ import annotations

import argparse
import importlib
import json
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


def collect_sglang_capabilities() -> dict[str, Any]:
    from . import get_hook_targets

    sglang_version = _package_version("sglang")
    radix_choices = _server_arg_choices("RADIX_EVICTION_POLICY_CHOICES")
    schedule_choices = _server_arg_choices("SCHEDULE_POLICY_CHOICES")
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
            "radix_eviction_policy_choices": radix_choices,
            "radix_priority_eviction_supported": "priority" in radix_choices,
            "schedule_policy_choices": schedule_choices,
            "priority_schedule_policy_supported": "priority" in schedule_choices,
            "enable_priority_scheduling_flag_expected": True,
        },
        "hook_probe": hook_probe,
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
        "",
        "## Hook Probe",
        "",
        "| module | class | class_found | missing_methods |",
        "| --- | --- | --- | --- |",
    ]
    for row in data.get("hook_probe", []):
        missing = ", ".join(row.get("missing_methods") or [])
        lines.append(
            f"| `{row.get('module')}` | `{row.get('class')}` | `{row.get('class_found')}` | `{missing}` |"
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

