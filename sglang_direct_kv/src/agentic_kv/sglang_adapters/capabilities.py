from __future__ import annotations

import argparse
import importlib
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


def _launch_server_help() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "sglang.launch_server", "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except Exception:
        return ""
    return result.stdout or ""


def _help_flag_present(help_text: str, flag: str) -> bool:
    return flag in help_text


def _help_choice_present(help_text: str, flag: str, choice: str) -> bool:
    if flag not in help_text:
        return False
    flag_pos = help_text.find(flag)
    window = help_text[flag_pos : flag_pos + 800]
    return bool(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(choice)}(?![A-Za-z0-9_-])", window))


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
    help_text = _launch_server_help()
    help_has_priority_flag = _help_flag_present(help_text, "--enable-priority-scheduling")
    help_has_radix_policy_flag = _help_flag_present(help_text, "--radix-eviction-policy")
    help_has_radix_priority = _help_choice_present(help_text, "--radix-eviction-policy", "priority")
    help_has_schedule_priority = _help_choice_present(help_text, "--schedule-policy", "priority")
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
            "radix_priority_eviction_supported": "priority" in radix_choices or help_has_radix_priority,
            "schedule_policy_choices": schedule_choices,
            "priority_schedule_policy_supported": "priority" in schedule_choices or help_has_schedule_priority,
            "enable_priority_scheduling_flag_supported": help_has_priority_flag,
            "radix_eviction_policy_flag_supported": help_has_radix_policy_flag,
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
        f"- `--enable-priority-scheduling` flag supported: `{data['launch_capabilities'].get('enable_priority_scheduling_flag_supported')}`",
        f"- `--radix-eviction-policy` flag supported: `{data['launch_capabilities'].get('radix_eviction_policy_flag_supported')}`",
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
