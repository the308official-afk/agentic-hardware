#!/usr/bin/env bash
set -euo pipefail

CONTROLLED_ROOT="${CONTROLLED_ROOT:-}"
LIVE_DIRECT_ROOT="${LIVE_DIRECT_ROOT:-}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-18}"
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
  LIVE_DIRECT_ROOT="$(ls -td artifacts/results/milestone26_live_direct_only_* artifacts/results/milestone26_live_direct_kv_load_* artifacts/results/milestone26_direct_kv_* artifacts/results/milestone26_paired_direct_kv_*/live_direct_kv_load 2>/dev/null | head -1 || true)"
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

"${PYTHON_BIN}" scripts/build_milestone27_controlled_replay_report.py \
  --root "${CONTROLLED_ROOT}" \
  --out-dir "${CONTROLLED_ROOT}/controlled_replay_report" \
  --latest-root "${LATEST_REPORT_ROOT}" \
  --live-direct-root "${LIVE_DIRECT_ROOT}" \
  --max-timeline-gaps "${MAX_TIMELINE_GAPS}"

echo
echo "Latest master report: ${LATEST_REPORT_ROOT}/latest_master_report.html"
