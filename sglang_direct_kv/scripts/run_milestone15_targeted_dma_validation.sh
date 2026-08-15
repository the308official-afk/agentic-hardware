#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"

: "${RESULT_ROOT:=artifacts/results/milestone15_targeted_dma_validation}"
: "${LATEST_REPORT_ROOT:=artifacts/results}"
: "${CLEAN_MODES:=no_prefetch oracle_direct_load}"
: "${ATTRIBUTION_MODE:=oracle_direct_load}"
: "${ATTRIBUTION_TORCH_PROFILER_ENABLE:=1}"
: "${SESSION_COUNT:=6}"
: "${TIMELINE_MAX_SESSIONS:=6}"
: "${MAX_TOTAL_TOKENS:=8192}"
: "${RANDOMIZE_TRAFFIC:=1}"
: "${RANDOM_SEED:=15}"
: "${ARRIVAL_GAP_MS:=120}"
: "${ARRIVAL_GAP_RANGE_MS:=20 90}"
: "${TOOL_WAIT_LIST_MS:=100 200 400 700}"
: "${TOOL_WAIT_RANGE_MS:=150 700}"
: "${PROMPT_TOKEN_LIST:=768 1024}"
: "${HINT_DELAY_MS:=180}"
: "${ORACLE_LEAD_MS:=100}"
: "${TRAFFIC_CONCURRENCY:=6}"
: "${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:=220}"
: "${AGENTIC_KV_TORCH_PROFILER_START_EVENTS:=hostpool.load_to_device_per_layer}"
: "${AGENTIC_KV_TORCH_PROFILER_START_AGENT_PHASE:=hint_prefetch}"

export RESULT_ROOT
export LATEST_REPORT_ROOT
export CLEAN_MODES
export ATTRIBUTION_MODE
export ATTRIBUTION_TORCH_PROFILER_ENABLE
export SESSION_COUNT
export TIMELINE_MAX_SESSIONS
export MAX_TOTAL_TOKENS
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
export AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS
export AGENTIC_KV_TORCH_PROFILER_START_EVENTS
export AGENTIC_KV_TORCH_PROFILER_START_AGENT_PHASE

echo "Milestone 15 targeted DMA/HtoD validation"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "SESSION_COUNT=${SESSION_COUNT}"
echo "PROMPT_TOKEN_LIST=${PROMPT_TOKEN_LIST}"
echo "AGENTIC_KV_TORCH_PROFILER_ENABLE=${ATTRIBUTION_TORCH_PROFILER_ENABLE}"
echo "AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS}"
echo "AGENTIC_KV_TORCH_PROFILER_START_EVENTS=${AGENTIC_KV_TORCH_PROFILER_START_EVENTS}"
echo "AGENTIC_KV_TORCH_PROFILER_START_AGENT_PHASE=${AGENTIC_KV_TORCH_PROFILER_START_AGENT_PHASE}"
echo
echo "Goal:"
echo "  Keep the run small enough for torch.profiler, but start profiling"
echo "  near the hint-side KV host-to-device load path so dark-green CUDA"
echo "  HtoD bars appear in the timeline."
echo

bash scripts/run_milestone12_paired_evidence.sh "${MODEL}"

echo
echo "Milestone 15 outputs written under ${RESULT_ROOT}"
echo "Open latest paired report: ${LATEST_REPORT_ROOT}/latest_paired_report.html"
echo "Run-specific paired report: ${RESULT_ROOT}/paired_report/paired_report.html"
