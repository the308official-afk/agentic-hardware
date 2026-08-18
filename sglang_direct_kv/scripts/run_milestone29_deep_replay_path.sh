#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}}"

export EXPERIMENT_KIND="${EXPERIMENT_KIND:-controlled}"
export REPORT_LABEL="${REPORT_LABEL:-milestone29_deep_replay_path_$(date +%Y%m%d_%H%M%S)}"
export PRESSURE_PROFILE="${PRESSURE_PROFILE:-medium}"
export UPDATE_LATEST="${UPDATE_LATEST:-1}"
export MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-32}"
export AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-1}"
export MODES="${MODES:-no_prefetch direct_prefetch}"

echo "Running Milestone 29 deep replay-path instrumentation."
echo "MODEL=${MODEL}"
echo "EXPERIMENT_KIND=${EXPERIMENT_KIND}"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "PRESSURE_PROFILE=${PRESSURE_PROFILE}"
echo "AGENTIC_KV_TRACE_SCHEDULER=${AGENTIC_KV_TRACE_SCHEDULER}"

bash scripts/run_master_report.sh "${MODEL}"
