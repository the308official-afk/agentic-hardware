#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
GREEN_BAR_PRESET="${GREEN_BAR_PRESET:-medium}"

case "${GREEN_BAR_PRESET}" in
  replay)
    : "${RESULT_ROOT:=artifacts/results/milestone13b_green_bar_replay}"
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
  small_full)
    : "${RESULT_ROOT:=artifacts/results/milestone13b_green_bar_small_full}"
    : "${SESSION_COUNT:=6}"
    : "${ARRIVAL_GAP_RANGE_MS:=20 90}"
    : "${TOOL_WAIT_RANGE_MS:=100 500}"
    : "${PROMPT_TOKEN_LIST:=1024 1536}"
    : "${HINT_DELAY_MS:=180}"
    : "${ORACLE_LEAD_MS:=100}"
    : "${TRAFFIC_CONCURRENCY:=6}"
    : "${TIMELINE_MAX_SESSIONS:=6}"
    : "${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:=350}"
    : "${MAX_TOTAL_TOKENS:=8192}"
    ;;
  medium)
    : "${RESULT_ROOT:=artifacts/results/milestone13b_green_bar_failure_stress}"
    : "${SESSION_COUNT:=12}"
    : "${ARRIVAL_GAP_RANGE_MS:=10 80}"
    : "${TOOL_WAIT_RANGE_MS:=100 700}"
    : "${PROMPT_TOKEN_LIST:=1024 1536}"
    : "${HINT_DELAY_MS:=200}"
    : "${ORACLE_LEAD_MS:=100}"
    : "${TRAFFIC_CONCURRENCY:=8}"
    : "${TIMELINE_MAX_SESSIONS:=12}"
    : "${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:=350}"
    : "${MAX_TOTAL_TOKENS:=8192}"
    ;;
  *)
    echo "Unknown GREEN_BAR_PRESET=${GREEN_BAR_PRESET}. Use medium, replay, or small_full." >&2
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

echo "Milestone 13B green-bar failure stress experiment"
echo "MODEL=${MODEL}"
echo "GREEN_BAR_PRESET=${GREEN_BAR_PRESET}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "LATEST_REPORT_ROOT=${LATEST_REPORT_ROOT}"
echo "SESSION_COUNT=${SESSION_COUNT}"
echo "ARRIVAL_GAP_RANGE_MS=${ARRIVAL_GAP_RANGE_MS}"
echo "TOOL_WAIT_RANGE_MS=${TOOL_WAIT_RANGE_MS}"
echo "PROMPT_TOKEN_LIST=${PROMPT_TOKEN_LIST}"
echo "HINT_DELAY_MS=${HINT_DELAY_MS}"
echo "ORACLE_LEAD_MS=${ORACLE_LEAD_MS}"
echo "TRAFFIC_CONCURRENCY=${TRAFFIC_CONCURRENCY}"
echo "AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS}"
echo
echo "Goal:"
echo "  Keep enough stress to create failures, but keep the run small enough"
echo "  that torch.profiler exports before shutdown and captures CUDA HtoD green bars."
echo

bash scripts/run_milestone12_paired_evidence.sh "${MODEL}"

echo
echo "Milestone 13B green-bar failure outputs written under ${RESULT_ROOT}"
echo "Open latest synthetic master report: ${LATEST_REPORT_ROOT}/latest_synthetic_master_report.html"
echo "Run-specific paired report: ${RESULT_ROOT}/paired_report/paired_report.html"
