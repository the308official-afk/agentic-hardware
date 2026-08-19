#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def run_cmd(args: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return {"ok": False, "cmd": args, "error": "command not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "cmd": args, "error": "timeout"}
    return {
        "ok": proc.returncode == 0,
        "cmd": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def parse_key_value_file(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_free_bytes(text: str) -> dict[str, Any]:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    for parts in lines:
        if parts and parts[0].rstrip(":") == "Mem" and len(parts) >= 7:
            return {
                "total_bytes": int(parts[1]),
                "used_bytes": int(parts[2]),
                "free_bytes": int(parts[3]),
                "available_bytes": int(parts[6]),
            }
    return {}


def parse_lscpu(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"Architecture", "CPU(s)", "Model name", "Thread(s) per core", "Core(s) per socket", "Socket(s)"}:
            out[key] = value
    return out


def parse_nvidia_smi_query(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    fields = [
        "index",
        "name",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "driver_version",
        "compute_capability",
        "pci_bus_id",
    ]
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        row = dict(zip(fields, parts))
        for key in ("memory_total_mib", "memory_used_mib", "memory_free_mib"):
            try:
                row[key] = int(float(str(row.get(key, "")).replace("MiB", "").strip()))
            except ValueError:
                pass
        row["memory_type"] = infer_gpu_memory_type(str(row.get("name") or ""))
        gpus.append(row)
    return gpus


def infer_gpu_memory_type(name: str) -> str:
    normalized = name.lower()
    if "a10" in normalized:
        return "GDDR6 (inferred from GPU model)"
    if "a100" in normalized:
        return "HBM2e (inferred from GPU model)"
    if "h100" in normalized or "h200" in normalized or "gh200" in normalized:
        return "HBM-class GPU memory (inferred from GPU model)"
    if "v100" in normalized:
        return "HBM2 (inferred from GPU model)"
    if "l4" in normalized or "l40" in normalized:
        return "GDDR6-class GPU memory (inferred from GPU model)"
    return "unknown"


def nvidia_smi_summary() -> dict[str, Any]:
    query = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        timeout=10.0,
    )
    full = run_cmd(["nvidia-smi"], timeout=10.0)
    cuda_version = ""
    if full.get("stdout"):
        match = re.search(r"CUDA Version:\s*([0-9.]+)", str(full["stdout"]))
        if match:
            cuda_version = match.group(1)
    return {
        "query": query,
        "raw": full,
        "cuda_version_from_nvidia_smi": cuda_version,
        "gpus": parse_nvidia_smi_query(str(query.get("stdout") or "")) if query.get("ok") else [],
    }


def ec2_metadata(path: str) -> str:
    token_result = run_cmd(
        [
            "curl",
            "-fsS",
            "-m",
            "1",
            "-X",
            "PUT",
            "http://169.254.169.254/latest/api/token",
            "-H",
            "X-aws-ec2-metadata-token-ttl-seconds: 60",
        ],
        timeout=2.0,
    )
    headers: list[str] = []
    token = str(token_result.get("stdout") or "")
    if token:
        headers = ["-H", f"X-aws-ec2-metadata-token: {token}"]
    result = run_cmd(["curl", "-fsS", "-m", "1", *headers, f"http://169.254.169.254/latest/meta-data/{path}"], timeout=2.0)
    return str(result.get("stdout") or "") if result.get("ok") else ""


def python_packages() -> dict[str, str]:
    packages: dict[str, str] = {"python": sys.version.split()[0]}
    for package in ("sglang", "torch", "transformers", "triton", "flashinfer-python"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = ""
    return packages


def parse_server_args_line(text: str) -> dict[str, str]:
    useful = {
        "model_path",
        "dtype",
        "kv_cache_dtype",
        "context_length",
        "mem_fraction_static",
        "max_total_tokens",
        "enable_hierarchical_cache",
        "hicache_ratio",
        "hicache_size",
        "hicache_write_policy",
        "hicache_io_backend",
        "hicache_mem_layout",
        "radix_eviction_policy",
        "tp_size",
        "pp_size",
        "schedule_policy",
    }
    out: dict[str, str] = {}
    for key, raw in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^,)]*)", text):
        if key in useful:
            out[key] = raw.strip().strip("'\"")
    return out


def parse_runtime_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    patterns = {
        "context_len": r"context_len=([0-9]+)",
        "max_total_num_tokens": r"max_total_num_tokens=([0-9]+)",
        "available_gpu_mem": r"available_gpu_mem=([0-9.]+\s*GB)",
        "chunked_prefill_size": r"chunked_prefill_size=([0-9]+)",
        "max_prefill_tokens": r"max_prefill_tokens=([0-9]+)",
        "max_running_requests": r"max_running_requests=([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = match.group(1)
    return out


def collect_sglang_logs(root: Path | None) -> dict[str, Any]:
    if not root or not root.exists():
        return {}
    logs = sorted(root.glob("*/sglang_server.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return {}
    selected = logs[-1]
    text = selected.read_text(encoding="utf-8", errors="replace")
    server_line = ""
    for line in text.splitlines():
        if "server_args=ServerArgs" in line:
            server_line = line
            break
    return {
        "selected_log": str(selected),
        "server_args": parse_server_args_line(server_line),
        "runtime": parse_runtime_lines(text),
    }


def bytes_to_gib(value: Any) -> str:
    try:
        return f"{int(value) / (1024**3):.2f} GiB"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect machine, GPU, SGLang, and run configuration for a master report.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--run-config-env", type=Path)
    parser.add_argument("--controlled-root", type=Path)
    parser.add_argument("--live-root", type=Path)
    args = parser.parse_args()

    free = run_cmd(["free", "-b"])
    host_memory = parse_free_bytes(str(free.get("stdout") or "")) if free.get("ok") else {}
    lscpu = run_cmd(["lscpu"])
    nvidia = nvidia_smi_summary()
    run_config = parse_key_value_file(args.run_config_env)
    sglang_logs = collect_sglang_logs(args.controlled_root)

    instance_type = ec2_metadata("instance-type")
    availability_zone = ec2_metadata("placement/availability-zone")
    instance_id = ec2_metadata("instance-id")

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cloud": {
            "provider": "aws" if instance_type else "",
            "instance_type": instance_type,
            "availability_zone": availability_zone,
            "instance_id": instance_id,
        },
        "host_memory": {
            **host_memory,
            "total_gib": bytes_to_gib(host_memory.get("total_bytes")),
            "available_gib": bytes_to_gib(host_memory.get("available_bytes")),
        },
        "cpu": parse_lscpu(str(lscpu.get("stdout") or "")) if lscpu.get("ok") else {},
        "gpu": nvidia,
        "software": python_packages(),
        "model": args.model,
        "run_config": run_config,
        "sglang": sglang_logs,
        "paths": {
            "run_config_env": str(args.run_config_env or ""),
            "controlled_root": str(args.controlled_root or ""),
            "live_root": str(args.live_root or ""),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote run environment to {args.out}")


if __name__ == "__main__":
    main()
