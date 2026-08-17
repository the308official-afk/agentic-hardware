#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
STRESS_PRESET="${STRESS_PRESET:-manager}"

case "${STRESS_PRESET}" in
  smoke)
    : "${RESULT_ROOT:=artifacts/results/milestone13_failure_stress_smoke}"
    : "${SESSION_COUNT:=8}"
    : "${ARRIVAL_GAP_RANGE_MS:=20 80}"
    : "${TOOL_WAIT_RANGE_MS:=150 700}"
    : "${PROMPT_TOKEN_LIST:=768 1024}"
    : "${HINT_DELAY_MS:=180}"
    : "${ORACLE_LEAD_MS:=100}"
    : "${TRAFFIC_CONCURRENCY:=8}"
    : "${TIMELINE_MAX_SESSIONS:=8}"
    : "${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:=350}"
    : "${MAX_TOTAL_TOKENS:=8192}"
    ;;
  manager)
    : "${RESULT_ROOT:=artifacts/results/milestone13_failure_stress}"
    : "${SESSION_COUNT:=32}"
    : "${ARRIVAL_GAP_RANGE_MS:=10 60}"
    : "${TOOL_WAIT_RANGE_MS:=100 700}"
    : "${PROMPT_TOKEN_LIST:=1536 2048}"
    : "${HINT_DELAY_MS:=200}"
    : "${ORACLE_LEAD_MS:=100}"
    : "${TRAFFIC_CONCURRENCY:=16}"
    : "${TIMELINE_MAX_SESSIONS:=16}"
    : "${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:=700}"
    : "${MAX_TOTAL_TOKENS:=8192}"
    ;;
  *)
    echo "Unknown STRESS_PRESET=${STRESS_PRESET}. Use smoke or manager." >&2
    exit 1
    ;;
esac

: "${LATEST_REPORT_ROOT:=artifacts/results}"
: "${CLEAN_MODES:=no_prefetch oracle_direct_load}"
: "${ATTRIBUTION_MODE:=oracle_direct_load}"
: "${RANDOMIZE_TRAFFIC:=1}"
: "${RANDOM_SEED:=13}"
: "${ARRIVAL_GAP_MS:=120}"
: "${TOOL_WAIT_LIST_MS:=100 200 400 700}"

export RESULT_ROOT
export LATEST_REPORT_ROOT
export CLEAN_MODES
export ATTRIBUTION_MODE
export MAX_TOTAL_TOKENS
export SESSION_COUNT
export RANDOMIZE_TRAFFIC
export RANDOM_SEED
export ARRIVAL_GAP_MS
export ARRIVAL_GAP_RANGE_MS
export TOOL_WAIT_LIST_MS
export TOOL_WAIT_RANGE_MS
export PROMPT_TOKEN_LIST
export HINT_DELAY_MS
export ORACLE_LEAD_MS
export TRAFFIC_CONCURRENCY
export TIMELINE_MAX_SESSIONS
export AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS

echo "Milestone 13 failure stress experiment"
echo "MODEL=${MODEL}"
echo "STRESS_PRESET=${STRESS_PRESET}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "LATEST_REPORT_ROOT=${LATEST_REPORT_ROOT}"
echo "SESSION_COUNT=${SESSION_COUNT}"
echo "ARRIVAL_GAP_RANGE_MS=${ARRIVAL_GAP_RANGE_MS}"
echo "TOOL_WAIT_RANGE_MS=${TOOL_WAIT_RANGE_MS}"
echo "PROMPT_TOKEN_LIST=${PROMPT_TOKEN_LIST}"
echo "HINT_DELAY_MS=${HINT_DELAY_MS}"
echo "ORACLE_LEAD_MS=${ORACLE_LEAD_MS}"
echo "TRAFFIC_CONCURRENCY=${TRAFFIC_CONCURRENCY}"
echo "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
echo
echo "Goal:"
echo "  Create realistic failure cases where software hints are late,"
echo "  too early/unprotected, or copied before replay but still reloaded."
echo

bash scripts/run_milestone12_paired_evidence.sh "${MODEL}"

echo
echo "Milestone 13 failure stress outputs written under ${RESULT_ROOT}"
echo "Open latest synthetic master report: ${LATEST_REPORT_ROOT}/latest_synthetic_master_report.html"
echo "Run-specific paired report: ${RESULT_ROOT}/paired_report/paired_report.html"
