#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"

: "${RESULT_ROOT:=artifacts/results/milestone14_lightweight_copy_telemetry}"
: "${LATEST_REPORT_ROOT:=artifacts/results}"
: "${CLEAN_MODES:=no_prefetch oracle_direct_load}"
: "${ATTRIBUTION_MODE:=oracle_direct_load}"
: "${ATTRIBUTION_TORCH_PROFILER_ENABLE:=0}"
: "${SESSION_COUNT:=32}"
: "${RANDOMIZE_TRAFFIC:=1}"
: "${RANDOM_SEED:=14}"
: "${ARRIVAL_GAP_RANGE_MS:=10 80}"
: "${TOOL_WAIT_RANGE_MS:=100 700}"
: "${PROMPT_TOKEN_LIST:=1024 1536}"
: "${HINT_DELAY_MS:=200}"
: "${ORACLE_LEAD_MS:=100}"
: "${TRAFFIC_CONCURRENCY:=8}"
: "${TIMELINE_MAX_SESSIONS:=32}"
: "${MAX_TOTAL_TOKENS:=8192}"

export RESULT_ROOT
export LATEST_REPORT_ROOT
export CLEAN_MODES
export ATTRIBUTION_MODE
export ATTRIBUTION_TORCH_PROFILER_ENABLE
export SESSION_COUNT
export RANDOMIZE_TRAFFIC
export RANDOM_SEED
export ARRIVAL_GAP_RANGE_MS
export TOOL_WAIT_RANGE_MS
export PROMPT_TOKEN_LIST
export HINT_DELAY_MS
export ORACLE_LEAD_MS
export TRAFFIC_CONCURRENCY
export TIMELINE_MAX_SESSIONS
export MAX_TOTAL_TOKENS

echo "Milestone 14 lightweight KV copy telemetry experiment"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "SESSION_COUNT=${SESSION_COUNT}"
echo "ATTRIBUTION_TORCH_PROFILER_ENABLE=${ATTRIBUTION_TORCH_PROFILER_ENABLE}"
echo "TIMELINE_MAX_SESSIONS=${TIMELINE_MAX_SESSIONS}"
echo
echo "Goal:"
echo "  Run larger agentic traffic without torch.profiler overhead."
echo "  Draw green copy bars from compact SGLang KV-copy telemetry."
echo "  Use small profiler runs separately to validate CUDA-level mapping."
echo

bash scripts/run_milestone12_paired_evidence.sh "${MODEL}"

echo
echo "Milestone 14 outputs written under ${RESULT_ROOT}"
echo "Open latest synthetic master report: ${LATEST_REPORT_ROOT}/latest_synthetic_master_report.html"
echo "Run-specific paired report: ${RESULT_ROOT}/paired_report/paired_report.html"
