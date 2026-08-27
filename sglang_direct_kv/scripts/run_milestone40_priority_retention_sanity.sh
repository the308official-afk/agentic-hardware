#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone40_priority_retention_sanity_$(date +%Y%m%d_%H%M%S)}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

export WORKLOAD_SOURCE="${WORKLOAD_SOURCE:-synthetic}"
export MAX_PAIRS="${MAX_PAIRS:-1}"
export MODES="${MODES:-no_prefetch dynamo_priority_hints}"
export TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-500}"
export FILLER_LIST="${FILLER_LIST:-${DISTRACTOR_COUNTS:-32 64 128 256}}"
export FILLER_PROMPT_TOKENS="${FILLER_PROMPT_TOKENS:-${DISTRACTOR_PROMPT_TOKENS:-1024}}"
export SYNTHETIC_PROMPT_TOKENS="${SYNTHETIC_PROMPT_TOKENS:-${TARGET_PROMPT_TOKENS:-4096}}"
export TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-0}"
export SYNTHETIC_REPLAY_SUFFIX_TOKENS="${SYNTHETIC_REPLAY_SUFFIX_TOKENS:-64}"
export FILLER_DIVERGE_EARLY="${FILLER_DIVERGE_EARLY:-1}"
export PRIORITY_DIRECT_PREFETCH=0
export DYNAMO_HIGH_PRIORITY="${DYNAMO_HIGH_PRIORITY:-100}"
export DYNAMO_NORMAL_PRIORITY="${DYNAMO_NORMAL_PRIORITY:-0}"
export DYNAMO_LOW_PRIORITY="${DYNAMO_LOW_PRIORITY:--100}"
export DYNAMO_RADIX_EVICTION_POLICY="${DYNAMO_RADIX_EVICTION_POLICY:-priority}"
export AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-1}"
export AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL:-1}"
export REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-4}"
export MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-16}"
export MAX_TOKENS="${MAX_TOKENS:-8}"
export PREFETCH_MAX_TOKENS="${PREFETCH_MAX_TOKENS:-1}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-12288}"
export HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-16}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
export RESULT_ROOT
export LATEST_REPORT_ROOT

echo "Milestone 40: Dynamo Priority KV Retention Sanity"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "MODES=${MODES}"
echo "DISTRACTOR_COUNTS/FILLER_LIST=${FILLER_LIST}"
echo "MAX_PAIRS=${MAX_PAIRS}"
echo "DYNAMO_RADIX_EVICTION_POLICY=${DYNAMO_RADIX_EVICTION_POLICY}"
echo

bash scripts/run_milestone27_real_prompt_controlled_replay.sh "${MODEL}"

"${PYTHON_BIN}" scripts/summarize_priority_retention_sanity.py \
  --root "${RESULT_ROOT}" \
  --out-dir "${RESULT_ROOT}/priority_retention_report" \
  --latest-root "${LATEST_REPORT_ROOT}"

echo
echo "Milestone 40 finished."
echo "Report: ${RESULT_ROOT}/priority_retention_report/priority_retention_report.html"
echo "Latest report: ${LATEST_REPORT_ROOT}/latest_priority_retention_sanity_report.html"
