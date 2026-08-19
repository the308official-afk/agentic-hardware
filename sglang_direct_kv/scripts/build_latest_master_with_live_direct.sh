#!/usr/bin/env bash
set -euo pipefail

CONTROLLED_ROOT="${CONTROLLED_ROOT:-}"
LIVE_DIRECT_ROOT="${LIVE_DIRECT_ROOT:-}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-18}"
RUN_ENV_JSON="${RUN_ENV_JSON:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ -z "${CONTROLLED_ROOT}" ]]; then
  CONTROLLED_ROOT="$(ls -td artifacts/results/milestone27_two_mode_* artifacts/results/milestone27_real_prompt_controlled_replay_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "${LIVE_DIRECT_ROOT}" ]]; then
  LIVE_DIRECT_ROOT="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import csv

base = Path("artifacts/results")
patterns = [
    "milestone26_live_direct_only_*",
    "milestone26_live_direct_kv_load_*",
    "milestone26_direct_kv_*",
    "milestone26_paired_direct_kv_*/live_direct_kv_load",
]

def count_gaps(root: Path) -> int:
    candidates = [
        root / "live_agentbench_prefetch_report" / "live_tool_gaps.csv",
        root / "live_tool_gaps.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    return max(sum(1 for _ in csv.DictReader(handle)), 0)
            except Exception:
                return 0
    return 0

roots = []
for pattern in patterns:
    roots.extend(path for path in base.glob(pattern) if path.is_dir())

if not roots:
    raise SystemExit(0)

# Prefer runs with more analyzed live gaps. This avoids accidentally picking
# tiny debug/report-only runs when several result folders have similar mtimes.
best = max(
    roots,
    key=lambda path: (
        count_gaps(path),
        path.stat().st_mtime,
        str(path),
    ),
)
print(best)
PY
)"
fi

if [[ -z "${CONTROLLED_ROOT}" || ! -d "${CONTROLLED_ROOT}" ]]; then
  echo "Could not find CONTROLLED_ROOT. Set CONTROLLED_ROOT=/path/to/milestone27_run." >&2
  exit 1
fi

if [[ -z "${LIVE_DIRECT_ROOT}" || ! -d "${LIVE_DIRECT_ROOT}" ]]; then
  echo "Could not find LIVE_DIRECT_ROOT. Set LIVE_DIRECT_ROOT=/path/to/live_direct_kv_run." >&2
  exit 1
fi

echo "Building latest master report with controlled replay plus live direct-prefetch evidence."
echo "CONTROLLED_ROOT=${CONTROLLED_ROOT}"
echo "LIVE_DIRECT_ROOT=${LIVE_DIRECT_ROOT}"
echo "LATEST_REPORT_ROOT=${LATEST_REPORT_ROOT}"

extra_args=()
if [[ -n "${RUN_ENV_JSON}" ]]; then
  extra_args+=(--run-environment-json "${RUN_ENV_JSON}")
fi

"${PYTHON_BIN}" scripts/build_milestone27_controlled_replay_report.py \
  --root "${CONTROLLED_ROOT}" \
  --out-dir "${CONTROLLED_ROOT}/controlled_replay_report" \
  --latest-root "${LATEST_REPORT_ROOT}" \
  --live-direct-root "${LIVE_DIRECT_ROOT}" \
  --max-timeline-gaps "${MAX_TIMELINE_GAPS}" \
  "${extra_args[@]}"

echo
echo "Latest master report: ${LATEST_REPORT_ROOT}/latest_master_report.html"
