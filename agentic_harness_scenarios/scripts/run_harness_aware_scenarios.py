#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from agentic_harness_scenarios import load_scenario_manifest, run_scenarios
from agentic_harness_scenarios.report import write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portable harness-aware synthetic scenarios.")
    parser.add_argument(
        "--manifest",
        default="configs/harness_aware_scenarios/minimal.json",
        help="Scenario manifest path, relative to sglang_direct_kv unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/results/harness_aware_scenarios",
        help="Output directory, relative to sglang_direct_kv unless absolute.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = _resolve(root, args.manifest)
    output_dir = _resolve(root, args.output_dir)
    scenarios = load_scenario_manifest(manifest)
    runs = run_scenarios(scenarios)
    write_outputs(runs, output_dir)
    print(f"Wrote {len(runs)} scenario runs to {output_dir}")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
