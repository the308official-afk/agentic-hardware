#!/usr/bin/env bash
set -euo pipefail

REPORT_LABEL="${REPORT_LABEL:-live_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/labeled/live/${REPORT_LABEL}}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-16}"
NO_PREFETCH_ROOT="${NO_PREFETCH_ROOT:-}"
PREFETCH_ROOT="${PREFETCH_ROOT:-}"
UPDATE_LATEST="${UPDATE_LATEST:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ -z "${NO_PREFETCH_ROOT}" || -z "${PREFETCH_ROOT}" ]]; then
  echo "ERROR: set NO_PREFETCH_ROOT and PREFETCH_ROOT to existing live run folders." >&2
  exit 2
fi

if [[ ! -d "${NO_PREFETCH_ROOT}" ]]; then
  echo "ERROR: NO_PREFETCH_ROOT does not exist: ${NO_PREFETCH_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${PREFETCH_ROOT}" ]]; then
  echo "ERROR: PREFETCH_ROOT does not exist: ${PREFETCH_ROOT}" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"
RESULT_ROOT="$(cd "${RESULT_ROOT}" && pwd)"
REPORT_ROOT="${RESULT_ROOT}/report"
mkdir -p "${REPORT_ROOT}"

echo "Build labeled live master report"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "NO_PREFETCH_ROOT=${NO_PREFETCH_ROOT}"
echo "PREFETCH_ROOT=${PREFETCH_ROOT}"
echo "UPDATE_LATEST=${UPDATE_LATEST}"

args=(
  scripts/build_live_paired_agentbench_report.py
  --no-prefetch-root "${NO_PREFETCH_ROOT}"
  --prefetch-root "${PREFETCH_ROOT}"
  --out-dir "${REPORT_ROOT}"
  --max-timeline-gaps "${MAX_TIMELINE_GAPS}"
)

if [[ "${UPDATE_LATEST}" == "1" ]]; then
  mkdir -p "${LATEST_REPORT_ROOT}"
  args+=(--latest-root "${LATEST_REPORT_ROOT}")
fi

"${PYTHON_BIN}" "${args[@]}"

cp -f "${REPORT_ROOT}"/* "${RESULT_ROOT}/" 2>/dev/null || true
if [[ -f "${REPORT_ROOT}/live_paired_agentbench_report.html" ]]; then
  cp -f "${REPORT_ROOT}/live_paired_agentbench_report.html" "${RESULT_ROOT}/master_report.html"
fi
if [[ -f "${REPORT_ROOT}/live_paired_agentbench_report.json" ]]; then
  cp -f "${REPORT_ROOT}/live_paired_agentbench_report.json" "${RESULT_ROOT}/master_report.json"
fi
if [[ -f "${REPORT_ROOT}/live_paired_agentbench_report.md" ]]; then
  cp -f "${REPORT_ROOT}/live_paired_agentbench_report.md" "${RESULT_ROOT}/master_report.md"
fi

cat > "${RESULT_ROOT}/run_config.env" <<EOF
REPORT_LABEL=${REPORT_LABEL}
RESULT_ROOT=${RESULT_ROOT}
NO_PREFETCH_ROOT=${NO_PREFETCH_ROOT}
PREFETCH_ROOT=${PREFETCH_ROOT}
MAX_TIMELINE_GAPS=${MAX_TIMELINE_GAPS}
UPDATE_LATEST=${UPDATE_LATEST}
LATEST_REPORT_ROOT=${LATEST_REPORT_ROOT}
EOF

echo
echo "Labeled live report: ${RESULT_ROOT}/master_report.html"
if [[ "${UPDATE_LATEST}" == "1" ]]; then
  echo "Latest live master report updated: ${LATEST_REPORT_ROOT}/latest_master_report.html"
else
  echo "Latest live master report was not overwritten. Set UPDATE_LATEST=1 to refresh it."
fi
