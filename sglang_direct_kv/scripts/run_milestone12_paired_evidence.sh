#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone12_paired_evidence}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
CLEAN_MODES="${CLEAN_MODES:-no_prefetch direct_load oracle_direct_load}"
ATTRIBUTION_MODE="${ATTRIBUTION_MODE:-oracle_direct_load}"
RUN_CLEAN="${RUN_CLEAN:-1}"
RUN_ATTRIBUTION="${RUN_ATTRIBUTION:-1}"
ATTRIBUTION_TORCH_PROFILER_ENABLE="${ATTRIBUTION_TORCH_PROFILER_ENABLE:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
SESSION_COUNT="${SESSION_COUNT:-12}"
RANDOMIZE_TRAFFIC="${RANDOMIZE_TRAFFIC:-1}"
RANDOM_SEED="${RANDOM_SEED:-7}"
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS:-120}"
ARRIVAL_GAP_RANGE_MS="${ARRIVAL_GAP_RANGE_MS:-60 220}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-250 500 900 1600}"
TOOL_WAIT_RANGE_MS="${TOOL_WAIT_RANGE_MS:-250 2200}"
PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST:-768 1024 1536}"
HINT_DELAY_MS="${HINT_DELAY_MS:-120}"
ORACLE_LEAD_MS="${ORACLE_LEAD_MS:-1000}"
TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY:-8}"
TIMELINE_MAX_SESSIONS="${TIMELINE_MAX_SESSIONS:-12}"
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS="${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:-300}"

CLEAN_ROOT="${RESULT_ROOT}/clean_performance"
ATTRIBUTION_ROOT="${RESULT_ROOT}/profiled_attribution"
REPORT_ROOT="${RESULT_ROOT}/paired_report"

mkdir -p "${RESULT_ROOT}" "${REPORT_ROOT}"

echo "Milestone 12 paired clean + attribution evidence"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "LATEST_REPORT_ROOT=${LATEST_REPORT_ROOT}"
echo "CLEAN_MODES=${CLEAN_MODES}"
echo "ATTRIBUTION_MODE=${ATTRIBUTION_MODE}"
echo "ATTRIBUTION_TORCH_PROFILER_ENABLE=${ATTRIBUTION_TORCH_PROFILER_ENABLE}"
echo "SESSION_COUNT=${SESSION_COUNT}"
echo "RANDOMIZE_TRAFFIC=${RANDOMIZE_TRAFFIC}"
echo "RANDOM_SEED=${RANDOM_SEED}"
echo "ARRIVAL_GAP_RANGE_MS=${ARRIVAL_GAP_RANGE_MS}"
echo "TOOL_WAIT_RANGE_MS=${TOOL_WAIT_RANGE_MS}"
echo "PROMPT_TOKEN_LIST=${PROMPT_TOKEN_LIST}"
echo "HINT_DELAY_MS=${HINT_DELAY_MS}"
echo "ORACLE_LEAD_MS=${ORACLE_LEAD_MS}"
echo
echo "Important:"
echo "  Clean run is for TTFT/performance claims."
echo "  Mechanism run is for KV movement attribution."
echo "  If torch profiler is enabled, do not use profiled TTFT values as performance numbers."

if [[ "${RUN_CLEAN}" == "1" ]]; then
  echo
  echo "Step 1/3: clean performance run with torch profiler OFF"
  RESULT_ROOT="${CLEAN_ROOT}" \
  MODES="${CLEAN_MODES}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
  SESSION_COUNT="${SESSION_COUNT}" \
  RANDOMIZE_TRAFFIC="${RANDOMIZE_TRAFFIC}" \
  RANDOM_SEED="${RANDOM_SEED}" \
  ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS}" \
  ARRIVAL_GAP_RANGE_MS="${ARRIVAL_GAP_RANGE_MS}" \
  TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS}" \
  TOOL_WAIT_RANGE_MS="${TOOL_WAIT_RANGE_MS}" \
  PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST}" \
  HINT_DELAY_MS="${HINT_DELAY_MS}" \
  ORACLE_LEAD_MS="${ORACLE_LEAD_MS}" \
  TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY}" \
  AGENTIC_KV_TORCH_PROFILER_ENABLE=0 \
  bash scripts/run_milestone9_agentic_traffic.sh "${MODEL}"
else
  echo
  echo "Step 1/3: skipping clean performance run because RUN_CLEAN=${RUN_CLEAN}"
fi

if [[ "${RUN_ATTRIBUTION}" == "1" ]]; then
  echo
  echo "Step 2/3: mechanism attribution run"
  RESULT_ROOT="${ATTRIBUTION_ROOT}" \
  MODE="${ATTRIBUTION_MODE}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
  SESSION_COUNT="${SESSION_COUNT}" \
  RANDOMIZE_TRAFFIC="${RANDOMIZE_TRAFFIC}" \
  RANDOM_SEED="${RANDOM_SEED}" \
  ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS}" \
  ARRIVAL_GAP_RANGE_MS="${ARRIVAL_GAP_RANGE_MS}" \
  TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS}" \
  TOOL_WAIT_RANGE_MS="${TOOL_WAIT_RANGE_MS}" \
  PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST}" \
  HINT_DELAY_MS="${HINT_DELAY_MS}" \
  ORACLE_LEAD_MS="${ORACLE_LEAD_MS}" \
  TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY}" \
  TIMELINE_MAX_SESSIONS="${TIMELINE_MAX_SESSIONS}" \
  AGENTIC_KV_TORCH_PROFILER_ENABLE="${ATTRIBUTION_TORCH_PROFILER_ENABLE}" \
  AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS="${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS}" \
  bash scripts/run_milestone11_agentic_timeline.sh "${MODEL}"
else
  echo
  echo "Step 2/3: skipping profiled attribution run because RUN_ATTRIBUTION=${RUN_ATTRIBUTION}"
fi

echo
echo "Step 3/3: building paired evidence report"
python scripts/summarize_milestone12_paired_evidence.py \
  --clean-root "${CLEAN_ROOT}" \
  --attribution-root "${ATTRIBUTION_ROOT}" \
  --out-root "${REPORT_ROOT}" \
  --modes "${CLEAN_MODES}" \
  --attribution-mode "${ATTRIBUTION_MODE}" \
  --max-timeline-sessions "${TIMELINE_MAX_SESSIONS}" \
  --latest-root "${LATEST_REPORT_ROOT}"

echo
echo "Milestone 12 outputs written under ${RESULT_ROOT}"
echo "Clean performance root: ${CLEAN_ROOT}"
echo "Profiled attribution root: ${ATTRIBUTION_ROOT}"
echo "Paired report HTML: ${REPORT_ROOT}/paired_report.html"
echo "Paired report Markdown: ${REPORT_ROOT}/paired_report.md"
echo "Latest paired report HTML: ${LATEST_REPORT_ROOT}/latest_paired_report.html"
echo "Latest paired report Markdown: ${LATEST_REPORT_ROOT}/latest_paired_report.md"
echo "Latest paired report JSON: ${LATEST_REPORT_ROOT}/latest_paired_report.json"
echo "Latest checkpoint CSV: ${LATEST_REPORT_ROOT}/latest_paired_checkpoint_results.csv"
echo "Latest key observations CSV: ${LATEST_REPORT_ROOT}/latest_paired_key_observations.csv"
