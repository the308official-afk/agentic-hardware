#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

export AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-1}"
export EXPERIMENT_KIND="${EXPERIMENT_KIND:-controlled}"
export REPORT_LABEL="${REPORT_LABEL:-forced_eviction_sanity_$(date +%Y%m%d_%H%M%S)}"
export PRESSURE_PROFILE="${PRESSURE_PROFILE:-eviction_sanity}"
export UPDATE_LATEST="${UPDATE_LATEST:-1}"
export MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-16}"

bash scripts/run_master_report.sh "${MODEL}"
