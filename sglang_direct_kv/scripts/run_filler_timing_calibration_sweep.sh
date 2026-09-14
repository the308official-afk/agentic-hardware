#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

REPORT_LABEL="${REPORT_LABEL:-filler_timing_calibration_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-artifacts/results}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/runs/controlled/${REPORT_LABEL}}"
REPORT_DIR="${REPORT_DIR:-${RESULTS_ROOT}/reports/${REPORT_LABEL}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

HARDWARE_PROFILE="${HARDWARE_PROFILE:-ec2_a10g}"
HARNESSES="${HARNESSES:-hatcher}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8}"
SAMPLES_PER_CONCURRENCY="${SAMPLES_PER_CONCURRENCY:-8}"
TOOL_WAIT_MS="${TOOL_WAIT_MS:-800}"
TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-1024}"
FILLER_PROMPT_TOKENS="${FILLER_PROMPT_TOKENS:-1536}"
SESSION_COUNT="${SESSION_COUNT:-1}"
TASK_REPLAY_STEPS="${TASK_REPLAY_STEPS:-2}"
TOOL_WAIT_PROFILE="${TOOL_WAIT_PROFILE:-realistic_agentic_mix}"
AGENTIC_WORKLOAD_PROFILE="${AGENTIC_WORKLOAD_PROFILE:-realistic_agentic_mix}"
REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"

mkdir -p "${RUN_ROOT}" "${REPORT_DIR}" "${RESULTS_ROOT}/run_logs"

echo "Filler Timing Calibration Sweep"
echo "MODEL=${MODEL}"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "HARDWARE_PROFILE=${HARDWARE_PROFILE}"
echo "HARNESSES=${HARNESSES}"
echo "CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS}"
echo "SAMPLES_PER_CONCURRENCY=${SAMPLES_PER_CONCURRENCY}"
echo "TASK_REPLAY_STEPS=${TASK_REPLAY_STEPS}"
echo "TOOL_WAIT_PROFILE=${TOOL_WAIT_PROFILE}"
echo "AGENTIC_WORKLOAD_PROFILE=${AGENTIC_WORKLOAD_PROFILE}"

for concurrency in ${CONCURRENCY_LEVELS}; do
  echo
  echo "==== Calibration concurrency=${concurrency} ===="
  P0_CONTROL_KNOBS="tool_wait_ms=${TOOL_WAIT_MS} target_prompt_tokens=${TARGET_PROMPT_TOKENS} filler_sessions=${SAMPLES_PER_CONCURRENCY} filler_prompt_tokens=${FILLER_PROMPT_TOKENS} session_count=${SESSION_COUNT} concurrency=${concurrency}" \
  HARDWARE_PROFILE="${HARDWARE_PROFILE}" \
  HARNESSES="${HARNESSES}" \
  MODES="no_prefetch" \
  PRESSURE_LEVELS="p0_control" \
  FILLER_REPLAY_DEADLINES=1 \
  FILLER_REPLAY_DEADLINE_MS="${TOOL_WAIT_MS}" \
  FILLER_BACKLOG_MODE=once \
  TOOL_WAIT_PROFILE="${TOOL_WAIT_PROFILE}" \
  TASK_REPLAY_STEPS="${TASK_REPLAY_STEPS}" \
  AGENTIC_WORKLOAD_PROFILE="${AGENTIC_WORKLOAD_PROFILE}" \
  TRACE_PROFILE=minimal \
  TRACE_CONTROLLER_DECISIONS=0 \
  TRACE_CONTROLLER_COMPLETION_LINKAGE=0 \
  TRACE_IDLE_GAP_AUDIT=0 \
  AGENTIC_KV_TRACE_SCHEDULER=0 \
  AGENTIC_KV_TRACE_KV_POOL=0 \
  AGENTIC_RUNTIME_TELEMETRY=0 \
  AGENTIC_KV_GPU_UTIL_SAMPLER=0 \
  AGENTIC_KV_COPY_TELEMETRY_ENABLE=0 \
  REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE}" \
  REPORT_LABEL="${REPORT_LABEL}" \
  RUN_ROOT="${RUN_ROOT}" \
  REPORT_DIR="${REPORT_DIR}" \
  UPDATE_LATEST=0 \
  SKIP_EXISTING_CASES=0 \
  ENCODING_CASE_KEY="calibc${concurrency}" \
  bash scripts/run_harness_deadline_pressure.sh "${MODEL}"
done

echo
echo "==== Build calibration dataset ===="
"${PYTHON_BIN}" scripts/build_filler_timing_calibration.py \
  --root "${RUN_ROOT}" \
  --out "${REPORT_DIR}/filler_timing_calibration.csv"

echo
echo "==== Fit calibration model ===="
"${PYTHON_BIN}" scripts/fit_filler_timing_model.py \
  --input "${REPORT_DIR}/filler_timing_calibration.csv" \
  --out-dir "${REPORT_DIR}"

echo
echo "Done."
echo "Calibration CSV: ${REPORT_DIR}/filler_timing_calibration.csv"
echo "Model JSON: ${REPORT_DIR}/filler_timing_model.json"
echo "Model summary: ${REPORT_DIR}/filler_timing_model_summary.csv"
