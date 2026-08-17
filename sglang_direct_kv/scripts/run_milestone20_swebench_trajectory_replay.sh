#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone20_swebench_trajectory_replay}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
MAX_SESSIONS="${MAX_SESSIONS:-24}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-250 500 900 1600 3000}"
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS:-120}"
RANDOM_SEED="${RANDOM_SEED:-20}"
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-0}"
WORKLOAD_JSONL="${WORKLOAD_JSONL:-${RESULT_ROOT}/swebench_trajectory_replay_workload.jsonl}"
WORKLOAD_CSV="${WORKLOAD_CSV:-${RESULT_ROOT}/swebench_trajectory_replay_workload.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CLEAN_MODES="${CLEAN_MODES:-no_prefetch oracle_direct_load}"
ATTRIBUTION_MODE="${ATTRIBUTION_MODE:-oracle_direct_load}"
ATTRIBUTION_TORCH_PROFILER_ENABLE="${ATTRIBUTION_TORCH_PROFILER_ENABLE:-0}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
HINT_DELAY_MS="${HINT_DELAY_MS:-120}"
ORACLE_LEAD_MS="${ORACLE_LEAD_MS:-500}"
TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY:-6}"
TIMELINE_MAX_SESSIONS="${TIMELINE_MAX_SESSIONS:-12}"
RUN_CLEAN="${RUN_CLEAN:-1}"
RUN_ATTRIBUTION="${RUN_ATTRIBUTION:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DIRECT_ROOT}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ -n "${CATALOG_CSV:-}" ]]; then
  CATALOG_CSV="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${CATALOG_CSV}")"
elif [[ -f "${PROJECT_ROOT}/../kv_cache_offloading/experiments/reports/latest_swebench_trajectory_prompt_catalog.csv" ]]; then
  CATALOG_CSV="$(cd "${PROJECT_ROOT}/../kv_cache_offloading/experiments/reports" && pwd)/latest_swebench_trajectory_prompt_catalog.csv"
elif [[ -f "${HOME}/kv_cache_offloading/experiments/reports/latest_swebench_trajectory_prompt_catalog.csv" ]]; then
  CATALOG_CSV="$(cd "${HOME}/kv_cache_offloading/experiments/reports" && pwd)/latest_swebench_trajectory_prompt_catalog.csv"
else
  echo "Could not find latest_swebench_trajectory_prompt_catalog.csv." >&2
  echo "Set CATALOG_CSV=/path/to/latest_swebench_trajectory_prompt_catalog.csv and rerun." >&2
  exit 1
fi

mkdir -p "${RESULT_ROOT}" "${LATEST_REPORT_ROOT}"

echo "Milestone 20: SWE-bench trajectory prompt replay"
echo "MODEL=${MODEL}"
echo "CATALOG_CSV=${CATALOG_CSV}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "MAX_SESSIONS=${MAX_SESSIONS}"
echo "TOOL_WAIT_LIST_MS=${TOOL_WAIT_LIST_MS}"
echo "CLEAN_MODES=${CLEAN_MODES}"
echo "ATTRIBUTION_MODE=${ATTRIBUTION_MODE}"
echo "ATTRIBUTION_TORCH_PROFILER_ENABLE=${ATTRIBUTION_TORCH_PROFILER_ENABLE}"
echo
echo "Goal:"
echo "  Replace synthetic prompts with real SWE-bench trajectory prompts."
echo "  Keep the rest of the paired-report experiment structure the same."
echo "  Tool waits are synthetic/speculated in this milestone."
echo

echo "Step 1/2: converting trajectory prompt catalog into replay workload"
"${PYTHON_BIN}" scripts/extract_swebench_trajectory_prompt_workload.py \
  --catalog-csv "${CATALOG_CSV}" \
  --out-jsonl "${WORKLOAD_JSONL}" \
  --out-csv "${WORKLOAD_CSV}" \
  --max-sessions "${MAX_SESSIONS}" \
  --tool-wait-list-ms "${TOOL_WAIT_LIST_MS}" \
  --arrival-gap-ms "${ARRIVAL_GAP_MS}" \
  --seed "${RANDOM_SEED}" \
  --max-prompt-chars "${MAX_PROMPT_CHARS}"

cp -f "${WORKLOAD_JSONL}" "${LATEST_REPORT_ROOT}/latest_swebench_trajectory_replay_workload.jsonl"
cp -f "${WORKLOAD_CSV}" "${LATEST_REPORT_ROOT}/latest_swebench_trajectory_replay_workload.csv"

echo
echo "Step 2/2: running paired clean + attribution evidence with real trajectory prompts"
RESULT_ROOT="${RESULT_ROOT}" \
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT}" \
CLEAN_MODES="${CLEAN_MODES}" \
ATTRIBUTION_MODE="${ATTRIBUTION_MODE}" \
RUN_CLEAN="${RUN_CLEAN}" \
RUN_ATTRIBUTION="${RUN_ATTRIBUTION}" \
ATTRIBUTION_TORCH_PROFILER_ENABLE="${ATTRIBUTION_TORCH_PROFILER_ENABLE}" \
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
WORKLOAD_JSONL="${WORKLOAD_JSONL}" \
SESSION_COUNT="${MAX_SESSIONS}" \
RANDOMIZE_TRAFFIC=0 \
RANDOM_SEED="${RANDOM_SEED}" \
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS}" \
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS}" \
HINT_DELAY_MS="${HINT_DELAY_MS}" \
ORACLE_LEAD_MS="${ORACLE_LEAD_MS}" \
TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY}" \
TIMELINE_MAX_SESSIONS="${TIMELINE_MAX_SESSIONS}" \
bash scripts/run_milestone12_paired_evidence.sh "${MODEL}"

if [[ -f "${LATEST_REPORT_ROOT}/latest_synthetic_master_report.html" ]]; then
  cp -f "${LATEST_REPORT_ROOT}/latest_synthetic_master_report.html" "${LATEST_REPORT_ROOT}/latest_swebench_trajectory_paired_report.html"
fi
if [[ -f "${LATEST_REPORT_ROOT}/latest_synthetic_master_report.md" ]]; then
  cp -f "${LATEST_REPORT_ROOT}/latest_synthetic_master_report.md" "${LATEST_REPORT_ROOT}/latest_swebench_trajectory_paired_report.md"
fi
if [[ -f "${LATEST_REPORT_ROOT}/latest_synthetic_master_report.json" ]]; then
  cp -f "${LATEST_REPORT_ROOT}/latest_synthetic_master_report.json" "${LATEST_REPORT_ROOT}/latest_swebench_trajectory_paired_report.json"
fi

echo
echo "Milestone 20 outputs written under ${RESULT_ROOT}"
echo "Workload JSONL: ${WORKLOAD_JSONL}"
echo "Latest workload JSONL: ${LATEST_REPORT_ROOT}/latest_swebench_trajectory_replay_workload.jsonl"
echo "Paired report HTML: ${RESULT_ROOT}/paired_report/paired_report.html"
echo "Latest synthetic master report HTML: ${LATEST_REPORT_ROOT}/latest_synthetic_master_report.html"
echo "Latest SWE-bench trajectory paired report HTML: ${LATEST_REPORT_ROOT}/latest_swebench_trajectory_paired_report.html"
