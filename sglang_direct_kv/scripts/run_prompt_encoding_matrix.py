#!/usr/bin/env python3
"""Independent encoding x scheduling sweep; does not update latest reports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--codecs", nargs="+", default=["identity", "dictionary_v1"])
    parser.add_argument("--modes", nargs="+", default=["no_prefetch", "controller_full"])
    parser.add_argument("--pressures", nargs="+", default=["p0_control", "p1_mild", "p3_high", "p4_cliff", "p5_boss_queue"])
    parser.add_argument("--harnesses", nargs="+", default=["hatcher"])
    parser.add_argument("--scope", choices=["target_requests", "replay", "all"], default="target_requests")
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", default=f"prompt_encoding_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.repeats < 1 or Path(args.label).name != args.label:
        parser.error("positive repeats and a simple report label are required")
    jobs = [(r, codec, mode) for r in range(args.repeats) for codec in args.codecs for mode in args.modes]
    random.Random(args.seed).shuffle(jobs)
    plan = []
    for index, (repeat, codec, mode) in enumerate(jobs):
        config = root / "configs" / "prompt_codecs" / f"{codec}.json"
        if codec not in {"identity", "dictionary_v1", "relations_v1"} or not config.is_file():
            parser.error(f"unknown configured codec: {codec}")
        plan.append({"order": index, "repeat": repeat, "codec": codec, "mode": mode,
                     "config": str(config), "scope": args.scope, "label": f"{args.label}_r{repeat}_{codec}_{mode}"})
    print(json.dumps({"seed": args.seed, "jobs": plan, "pressures": args.pressures,
                      "harnesses": args.harnesses, "model": args.model, "update_latest": False}, indent=2), flush=True)
    if args.dry_run:
        return
    report_root = root / "artifacts" / "results" / "reports" / args.label
    report_root.mkdir(parents=True, exist_ok=False)
    (report_root / "matrix_plan.json").write_text(json.dumps(vars(args) | {"workload": str(args.workload or ""), "jobs": plan}, indent=2))
    for job in plan:
        env = os.environ.copy()
        for key in ("RUN_ROOT", "REPORT_DIR", "PROMPT_WORKLOAD_JSONL"):
            env.pop(key, None)
        env.update(PROMPT_CODEC_CONFIG=job["config"], PROMPT_ENCODING_SCOPE=args.scope,
                   PROMPT_REPETITION=str(job["repeat"]), MODES=job["mode"],
                   HARNESSES=" ".join(args.harnesses), PRESSURE_LEVELS=" ".join(args.pressures),
                   UPDATE_LATEST="0", REPORT_BUILDER_MODE="lightweight", REPORT_LABEL=job["label"],
                   PYTHON_BIN=sys.executable, HARDWARE_PROFILE="ec2_a10g")
        if args.workload:
            env["PROMPT_WORKLOAD_JSONL"] = str(args.workload.resolve())
        subprocess.run(["bash", "scripts/run_harness_deadline_pressure.sh", args.model], cwd=root, env=env, check=True)
    (report_root / "complete.json").write_text(json.dumps({"completed_jobs": len(plan)}))


if __name__ == "__main__":
    main()
