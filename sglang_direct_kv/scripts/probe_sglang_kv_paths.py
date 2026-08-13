#!/usr/bin/env python3
"""Probe installed SGLang internals for KV/cache/offload-related symbols.

Run this on the EC2 GPU machine after installing SGLang. The output is a
starting map for direct KV instrumentation.

This script uses static source scanning by default. Importing every SGLang
submodule can trigger optional native builds or GPU initialization, which makes
the probe noisy and brittle.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Optional


KEYWORDS = (
    "kv",
    "cache",
    "radix",
    "prefix",
    "hicache",
    "offload",
    "evict",
    "memory",
    "page",
    "block",
)


def keyword_hit(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in KEYWORDS)


def safe_import(module_name: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def iter_source_files(root: Path) -> Iterable[Path]:
    skip_parts = {
        "__pycache__",
        "build",
        "dist",
        ".git",
        "node_modules",
    }
    for path in root.rglob("*.py"):
        if any(part in skip_parts for part in path.parts):
            continue
        yield path


def line_hits(lines: list[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if keyword_hit(line):
            stripped = line.strip()
            if stripped:
                hits.append({"line": line_no, "text": stripped[:240]})
    return hits


def ast_symbols(path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except Exception:
        return {"classes": [], "functions": []}

    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and keyword_hit(node.name):
            methods = [
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and keyword_hit(item.name)
            ]
            classes.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": methods,
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and keyword_hit(
            node.name
        ):
            functions.append({"name": node.name, "line": node.lineno})

    return {"classes": classes, "functions": functions}


def inspect_file(path: Path, package_root: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits = line_hits(lines)
    symbols = ast_symbols(path)
    rel_path = str(path.relative_to(package_root.parent))
    return {
        "path": str(path),
        "relative_path": rel_path,
        "line_hits": hits[:80],
        "line_hit_count": len(hits),
        "classes": symbols["classes"],
        "functions": symbols["functions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/sglang_probe.json")
    parser.add_argument("--max-files", type=int, default=1000)
    args = parser.parse_args()

    root = safe_import("sglang")
    if root is None:
        raise SystemExit("Could not import sglang. Install SGLang first.")

    root_file = getattr(root, "__file__", None)
    if root_file is None:
        raise SystemExit("Could not locate sglang package file.")
    package_root = Path(root_file).resolve().parent

    results: dict[str, Any] = {
        "sglang_file": root_file,
        "sglang_package_root": str(package_root),
        "keywords": KEYWORDS,
        "files": {},
        "summary": {
            "matched_files": 0,
            "total_line_hits": 0,
        },
    }

    for idx, path in enumerate(iter_source_files(package_root)):
        if idx >= args.max_files:
            break
        inspected = inspect_file(path, package_root)
        if (
            inspected["line_hit_count"] == 0
            and not inspected["classes"]
            and not inspected["functions"]
        ):
            continue
        results["files"][inspected["relative_path"]] = inspected
        results["summary"]["matched_files"] += 1
        results["summary"]["total_line_hits"] += inspected["line_hit_count"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote probe results to {out_path}")
    print(f"SGLang package root: {package_root}")
    print(f"Matched files: {results['summary']['matched_files']}")
    print(f"Total line hits: {results['summary']['total_line_hits']}")


if __name__ == "__main__":
    main()
