#!/usr/bin/env python3
"""Sample coarse GPU utilization with nvidia-smi during an experiment run."""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STOP = False


def request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP
    STOP = True


def parse_float(value: str) -> float | str:
    value = value.strip()
    if not value or value.lower() in {"[not supported]", "n/a", "not supported"}:
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def sample_once() -> list[list[str]]:
    query = (
        "index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,power.draw,temperature.gpu"
    )
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")
    return list(csv.reader(proc.stdout.splitlines()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample nvidia-smi utilization to CSV.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ts_ns",
        "wall_time_s",
        "gpu_index",
        "gpu_name",
        "utilization_gpu_pct",
        "utilization_memory_pct",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
        "temperature_gpu_c",
        "sampler_error",
    ]
    interval_s = max(args.interval_ms, 10.0) / 1000.0
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        while not STOP:
            started = time.time()
            ts_ns = time.time_ns()
            try:
                rows = sample_once()
                if not rows:
                    writer.writerow({"ts_ns": ts_ns, "wall_time_s": started, "sampler_error": "no_gpu_rows"})
                for row in rows:
                    padded = [item.strip() for item in row] + [""] * 8
                    writer.writerow(
                        {
                            "ts_ns": ts_ns,
                            "wall_time_s": f"{started:.6f}",
                            "gpu_index": padded[0],
                            "gpu_name": padded[1],
                            "utilization_gpu_pct": parse_float(padded[2]),
                            "utilization_memory_pct": parse_float(padded[3]),
                            "memory_used_mib": parse_float(padded[4]),
                            "memory_total_mib": parse_float(padded[5]),
                            "power_draw_w": parse_float(padded[6]),
                            "temperature_gpu_c": parse_float(padded[7]),
                            "sampler_error": "",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - sampler must not crash the experiment runner.
                writer.writerow(
                    {
                        "ts_ns": ts_ns,
                        "wall_time_s": f"{started:.6f}",
                        "sampler_error": str(exc)[:240],
                    }
                )
            handle.flush()
            elapsed = time.time() - started
            time.sleep(max(0.0, interval_s - elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
