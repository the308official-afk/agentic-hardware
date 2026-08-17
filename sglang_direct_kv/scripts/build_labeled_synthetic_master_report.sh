#!/usr/bin/env bash
set -euo pipefail

REPORT_LABEL="${REPORT_LABEL:-synthetic_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/labeled/synthetic/${REPORT_LABEL}}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
UPDATE_LATEST="${UPDATE_LATEST:-0}"
CLEAN_ROOT="${CLEAN_ROOT:-artifacts/results/milestone15_targeted_dma_validation/clean_performance}"
ATTRIBUTION_ROOT="${ATTRIBUTION_ROOT:-artifacts/results/milestone15_targeted_dma_validation/profiled_attribution}"
MODES="${MODES:-no_prefetch,oracle_direct_load}"
ATTRIBUTION_MODE="${ATTRIBUTION_MODE:-oracle_direct_load}"
MAX_TIMELINE_SESSIONS="${MAX_TIMELINE_SESSIONS:-6}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ ! -d "${CLEAN_ROOT}" ]]; then
  echo "ERROR: CLEAN_ROOT does not exist: ${CLEAN_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${ATTRIBUTION_ROOT}" ]]; then
  echo "ERROR: ATTRIBUTION_ROOT does not exist: ${ATTRIBUTION_ROOT}" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"
RESULT_ROOT="$(cd "${RESULT_ROOT}" && pwd)"
REPORT_ROOT="${RESULT_ROOT}/report"
mkdir -p "${REPORT_ROOT}"

echo "Build labeled synthetic master report"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "CLEAN_ROOT=${CLEAN_ROOT}"
echo "ATTRIBUTION_ROOT=${ATTRIBUTION_ROOT}"
echo "UPDATE_LATEST=${UPDATE_LATEST}"

args=(
  scripts/summarize_milestone12_paired_evidence.py
  --clean-root "${CLEAN_ROOT}"
  --attribution-root "${ATTRIBUTION_ROOT}"
  --out-root "${REPORT_ROOT}"
  --modes "${MODES}"
  --attribution-mode "${ATTRIBUTION_MODE}"
  --max-timeline-sessions "${MAX_TIMELINE_SESSIONS}"
)

if [[ "${UPDATE_LATEST}" == "1" ]]; then
  mkdir -p "${LATEST_REPORT_ROOT}"
  args+=(--latest-root "${LATEST_REPORT_ROOT}")
fi

"${PYTHON_BIN}" "${args[@]}"

cp -f "${REPORT_ROOT}"/* "${RESULT_ROOT}/" 2>/dev/null || true
if [[ -f "${REPORT_ROOT}/paired_report.html" ]]; then
  cp -f "${REPORT_ROOT}/paired_report.html" "${RESULT_ROOT}/master_report.html"
fi
if [[ -f "${REPORT_ROOT}/paired_report.json" ]]; then
  cp -f "${REPORT_ROOT}/paired_report.json" "${RESULT_ROOT}/master_report.json"
fi
if [[ -f "${REPORT_ROOT}/paired_report.md" ]]; then
  cp -f "${REPORT_ROOT}/paired_report.md" "${RESULT_ROOT}/master_report.md"
fi

cat > "${RESULT_ROOT}/run_config.env" <<EOF
REPORT_LABEL=${REPORT_LABEL}
RESULT_ROOT=${RESULT_ROOT}
CLEAN_ROOT=${CLEAN_ROOT}
ATTRIBUTION_ROOT=${ATTRIBUTION_ROOT}
MODES=${MODES}
ATTRIBUTION_MODE=${ATTRIBUTION_MODE}
MAX_TIMELINE_SESSIONS=${MAX_TIMELINE_SESSIONS}
UPDATE_LATEST=${UPDATE_LATEST}
LATEST_REPORT_ROOT=${LATEST_REPORT_ROOT}
EOF

echo
echo "Labeled synthetic report: ${RESULT_ROOT}/master_report.html"
if [[ "${UPDATE_LATEST}" == "1" ]]; then
  echo "Latest synthetic master report updated: ${LATEST_REPORT_ROOT}/latest_synthetic_master_report.html"
else
  echo "Latest synthetic master report was not overwritten. Set UPDATE_LATEST=1 to refresh it."
fi
