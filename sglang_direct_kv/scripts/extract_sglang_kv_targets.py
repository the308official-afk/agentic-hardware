#!/usr/bin/env python3
"""Extract likely SGLang KV hook points using static source parsing.

This avoids importing heavy SGLang modules. Direct imports can trigger Triton
native compilation and CUDA setup before the EC2 environment is fully ready.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TARGETS: Dict[str, List[Tuple[str, str]]] = {
    "sglang/srt/managers/scheduler.py": [
        ("function", "_prefetch_kvcache"),
        ("function", "init_memory_pool_and_cache"),
        ("function", "check_memory"),
    ],
    "sglang/srt/managers/schedule_batch.py": [
        ("function", "offload_kv_cache"),
        ("function", "load_kv_cache"),
        ("function", "_evict_tree_cache_if_needed"),
        ("function", "alloc_paged_token_slots_extend"),
        ("function", "alloc_paged_token_slots_decode"),
    ],
    "sglang/srt/managers/cache_controller.py": [
        ("class", "HiCacheController"),
        ("function", "evict_device"),
        ("function", "evict_host"),
        ("function", "generic_page_transfer"),
        ("function", "generic_page_backup"),
    ],
    "sglang/srt/mem_cache/memory_pool.py": [
        ("class", "KVCache"),
        ("class", "MHATokenToKVPool"),
        ("function", "move_kv_cache"),
        ("function", "copy_all_layer_kv_cache"),
        ("function", "get_kv_buffer"),
        ("function", "set_kv_buffer"),
    ],
    "sglang/srt/mem_cache/memory_pool_host.py": [
        ("class", "HostKVCache"),
        ("class", "MHATokenToKVPoolHost"),
        ("function", "get_flat_data_page"),
        ("function", "set_from_flat_data_page"),
        ("function", "init_kv_buffer"),
    ],
    "sglang/srt/mem_cache/radix_cache.py": [
        ("class", "RadixCache"),
        ("function", "match_prefix"),
        ("function", "cache_finished_req"),
        ("function", "cache_unfinished_req"),
        ("function", "evict"),
    ],
    "sglang/srt/mem_cache/hiradix_cache.py": [
        ("class", "HiRadixCache"),
        ("function", "ready_to_load_host_cache"),
        ("function", "check_hicache_events"),
        ("function", "evict"),
        ("function", "evict_host"),
        ("function", "match_prefix"),
    ],
}


def package_root() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not locate installed sglang package.")
    return Path(spec.origin).resolve().parent


def node_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    return None


def node_kind(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "function"
    return None


def extract_target(path: Path, target_kind: str, target_name: str) -> Optional[dict[str, Any]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    tree = ast.parse(source)

    matches = []
    for node in ast.walk(tree):
        if node_kind(node) == target_kind and node_name(node) == target_name:
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            # Add a small context window.
            context_start = max(1, start - 5)
            context_end = min(len(lines), end + 5)
            snippet = "\n".join(
                f"{idx:5d}: {lines[idx - 1]}"
                for idx in range(context_start, context_end + 1)
            )
            matches.append(
                {
                    "kind": target_kind,
                    "name": target_name,
                    "start_line": start,
                    "end_line": end,
                    "context_start": context_start,
                    "context_end": context_end,
                    "snippet": snippet,
                }
            )
    if not matches:
        return None
    return {"path": str(path), "matches": matches}


def write_markdown(results: dict[str, Any], out_path: Path) -> None:
    parts = ["# SGLang KV Hook Target Extracts", ""]
    parts.append(f"SGLang package root: `{results['sglang_package_root']}`")
    parts.append("")
    for rel_path, file_result in results["files"].items():
        parts.append(f"## `{rel_path}`")
        parts.append("")
        if not file_result["targets"]:
            parts.append("No configured targets found.")
            parts.append("")
            continue
        for target in file_result["targets"]:
            parts.append(f"### `{target['kind']} {target['name']}`")
            parts.append("")
            for match in target["matches"]:
                parts.append(
                    f"Lines {match['start_line']}-{match['end_line']} "
                    f"(context {match['context_start']}-{match['context_end']})"
                )
                parts.append("")
                parts.append("```python")
                parts.append(match["snippet"])
                parts.append("```")
                parts.append("")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="artifacts/sglang_kv_targets.json")
    parser.add_argument("--out-md", default="artifacts/sglang_kv_targets.md")
    args = parser.parse_args()

    root = package_root()
    results: dict[str, Any] = {
        "sglang_package_root": str(root),
        "files": {},
    }

    for rel_path, targets in TARGETS.items():
        # rel_path starts with sglang/...; package root points at .../sglang.
        source_path = root.parent / rel_path
        file_result = {"path": str(source_path), "targets": []}
        if source_path.exists():
            for target_kind, target_name in targets:
                extracted = extract_target(source_path, target_kind, target_name)
                if extracted is not None:
                    file_result["targets"].append(
                        {
                            "kind": target_kind,
                            "name": target_name,
                            "matches": extracted["matches"],
                        }
                    )
        results["files"][rel_path] = file_result

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, out_md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
