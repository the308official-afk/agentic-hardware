#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from agentic_kv.harness_scenarios.real_runner import (
    DEFAULT_MODEL,
    REAL_SCENARIOS,
    build_real_scenario_command,
    get_real_scenario_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a harness-aware claim through the existing instrumented SGLang pipeline."
    )
    parser.add_argument("scenario_id", choices=sorted(REAL_SCENARIOS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--report-label", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated command without running it.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    spec = get_real_scenario_spec(args.scenario_id)
    cmd = build_real_scenario_command(
        spec,
        repo_root=repo_root,
        model=args.model,
        report_label=args.report_label,
    )

    print(f"Scenario: {spec.scenario_id} - {spec.claim}")
    print("Delegated runner:")
    print(cmd[-1])
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
