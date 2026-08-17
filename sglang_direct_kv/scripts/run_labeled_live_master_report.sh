#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}}"
REPORT_LABEL="${REPORT_LABEL:-live_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/labeled/live/${REPORT_LABEL}}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
UPDATE_LATEST="${UPDATE_LATEST:-0}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

mkdir -p "${RESULT_ROOT}"
RESULT_ROOT="$(cd "${RESULT_ROOT}" && pwd)"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/runs}"

if [[ "${UPDATE_LATEST}" == "1" ]]; then
  EFFECTIVE_LATEST_ROOT="${LATEST_REPORT_ROOT}"
else
  EFFECTIVE_LATEST_ROOT="${RESULT_ROOT}/_latest_scratch"
fi

export START_INDEX="${START_INDEX:-0}"
export END_INDEX="${END_INDEX:-15}"
export AGENTBENCH_EXECUTION_LOOP_MAX_STEPS="${AGENTBENCH_EXECUTION_LOOP_MAX_STEPS:-${MAX_STEPS:-10}}"
export MAX_TIMELINE_GAPS

echo "Run labeled live master report"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "MODEL=${MODEL}"
echo "TASK_RANGE=${START_INDEX}-${END_INDEX}"
echo "AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=${AGENTBENCH_EXECUTION_LOOP_MAX_STEPS}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "UPDATE_LATEST=${UPDATE_LATEST}"

RESULT_ROOT="${RUN_ROOT}" \
LATEST_REPORT_ROOT="${EFFECTIVE_LATEST_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_milestone24_live_paired_agentbench_report.sh "${MODEL}"

REPORT_ROOT="${RUN_ROOT}/live_paired_report"
mkdir -p "${RESULT_ROOT}"
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
MODEL=${MODEL}
RESULT_ROOT=${RESULT_ROOT}
RUN_ROOT=${RUN_ROOT}
START_INDEX=${START_INDEX}
END_INDEX=${END_INDEX}
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=${AGENTBENCH_EXECUTION_LOOP_MAX_STEPS}
MAX_TIMELINE_GAPS=${MAX_TIMELINE_GAPS}
UPDATE_LATEST=${UPDATE_LATEST}
LATEST_REPORT_ROOT=${LATEST_REPORT_ROOT}
EOF

if [[ "${UPDATE_LATEST}" == "1" ]]; then
  mkdir -p "${LATEST_REPORT_ROOT}"
  cp -f "${RESULT_ROOT}/master_report.html" "${LATEST_REPORT_ROOT}/latest_master_report.html"
fi

echo
echo "Labeled live master report: ${RESULT_ROOT}/master_report.html"
if [[ "${UPDATE_LATEST}" == "1" ]]; then
  echo "Latest live master report updated: ${LATEST_REPORT_ROOT}/latest_master_report.html"
else
  echo "Latest live master report was not overwritten. Set UPDATE_LATEST=1 to refresh it."
fi
