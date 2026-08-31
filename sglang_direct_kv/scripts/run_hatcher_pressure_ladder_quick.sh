#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"

# Hatcher pressure ladder quick run.
#
# This deliberately avoids a full Cartesian sweep. It runs three bundled
# pressure levels across two modes:
#   P0 Control      x no_prefetch/e2e_priority_hints
#   P3 High         x no_prefetch/e2e_priority_hints
#   P5 Burst Cliff  x no_prefetch/e2e_priority_hints
export REPORT_LABEL="${REPORT_LABEL:-hatcher_pressure_ladder_quick_$(date +%Y%m%d_%H%M%S)}"
export RESULTS_ROOT="${RESULTS_ROOT:-artifacts/results}"
export RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/runs/controlled/${REPORT_LABEL}}"
export REPORT_DIR="${REPORT_DIR:-${RESULTS_ROOT}/reports/${REPORT_LABEL}}"
export UPDATE_LATEST="${UPDATE_LATEST:-1}"
export MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-64}"
export MODES="${MODES:-no_prefetch e2e_priority_hints}"
export WORKLOAD_SOURCE="${WORKLOAD_SOURCE:-synthetic}"
export AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-1}"
export AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL:-1}"
export AGENTIC_RUNTIME_TELEMETRY="${AGENTIC_RUNTIME_TELEMETRY:-1}"
export AGENTIC_RUNTIME_TELEMETRY_BACKEND="${AGENTIC_RUNTIME_TELEMETRY_BACKEND:-sglang}"
export AGENTIC_KV_GPU_UTIL_SAMPLER="${AGENTIC_KV_GPU_UTIL_SAMPLER:-1}"
export GPU_UTIL_SAMPLE_INTERVAL_MS="${GPU_UTIL_SAMPLE_INTERVAL_MS:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

RESULTS_ROOT="$(mkdir -p "${RESULTS_ROOT}" && cd "${RESULTS_ROOT}" && pwd)"
RUN_ROOT="$(mkdir -p "${RUN_ROOT}" && cd "${RUN_ROOT}" && pwd)"
REPORT_DIR="$(mkdir -p "${REPORT_DIR}" && cd "${REPORT_DIR}" && pwd)"
RUN_CONFIG_ENV="${REPORT_DIR}/run_config.env"
RUN_ENV_JSON="${REPORT_DIR}/run_environment.json"
GPU_UTIL_CSV="${REPORT_DIR}/gpu_utilization_samples.csv"
GPU_UTIL_LOG="${REPORT_DIR}/gpu_utilization_sampler.log"
GPU_UTIL_SAMPLER_PID=""

start_gpu_util_sampler() {
  if [[ "${AGENTIC_KV_GPU_UTIL_SAMPLER}" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU util sampler disabled: nvidia-smi not found."
    return
  fi
  echo "Starting GPU util sampler: ${GPU_UTIL_CSV}"
  "${PYTHON_BIN}" scripts/sample_gpu_utilization.py \
    --out "${GPU_UTIL_CSV}" \
    --interval-ms "${GPU_UTIL_SAMPLE_INTERVAL_MS}" >"${GPU_UTIL_LOG}" 2>&1 &
  GPU_UTIL_SAMPLER_PID="$!"
}

stop_gpu_util_sampler() {
  if [[ -n "${GPU_UTIL_SAMPLER_PID}" ]]; then
    if kill -0 "${GPU_UTIL_SAMPLER_PID}" >/dev/null 2>&1; then
      kill "${GPU_UTIL_SAMPLER_PID}" >/dev/null 2>&1 || true
      wait "${GPU_UTIL_SAMPLER_PID}" >/dev/null 2>&1 || true
    fi
    GPU_UTIL_SAMPLER_PID=""
  fi
}
trap stop_gpu_util_sampler EXIT

write_run_config() {
  {
    echo "REPORT_LABEL=${REPORT_LABEL}"
    echo "MODEL=${MODEL}"
    echo "EXPERIMENT_KIND=pressure_ladder_quick"
    echo "WORKLOAD_SOURCE=${WORKLOAD_SOURCE}"
    echo "RESULTS_ROOT=${RESULTS_ROOT}"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "REPORT_DIR=${REPORT_DIR}"
    echo "UPDATE_LATEST=${UPDATE_LATEST}"
    echo "MAX_TIMELINE_GAPS=${MAX_TIMELINE_GAPS}"
    echo "MODES=${MODES}"
    echo "PRESSURE_LADDER_LEVELS=P0 Control; P3 High; P5 Burst Cliff"
    echo "P0_CONTROL=tool_wait_ms=500,target_prompt_tokens=1024,fillers=0,request_concurrency=1,urgent_agents=1,max_total_tokens=8192"
    echo "P3_HIGH=tool_wait_ms=50,target_prompt_tokens=4096,fillers=32,request_concurrency=8,urgent_agents=1"
    echo "P5_BURST_CLIFF=tool_wait_ms=50,target_prompt_tokens=4096,background_fillers_per_session=0,request_concurrency=8,urgent_agents=4,arrival_shape=burst,sync_replay_after_initials=1"
    echo "AGENTIC_KV_TRACE_SCHEDULER=${AGENTIC_KV_TRACE_SCHEDULER}"
    echo "AGENTIC_KV_TRACE_KV_POOL=${AGENTIC_KV_TRACE_KV_POOL}"
    echo "AGENTIC_RUNTIME_TELEMETRY=${AGENTIC_RUNTIME_TELEMETRY}"
    echo "AGENTIC_RUNTIME_TELEMETRY_BACKEND=${AGENTIC_RUNTIME_TELEMETRY_BACKEND}"
    echo "AGENTIC_KV_GPU_UTIL_SAMPLER=${AGENTIC_KV_GPU_UTIL_SAMPLER}"
    echo "GPU_UTIL_SAMPLE_INTERVAL_MS=${GPU_UTIL_SAMPLE_INTERVAL_MS}"
    echo "GPU_UTIL_CSV=${GPU_UTIL_CSV}"
  } >"${RUN_CONFIG_ENV}"
}

collect_run_environment() {
  if [[ -f scripts/collect_run_environment.py ]]; then
    "${PYTHON_BIN}" scripts/collect_run_environment.py \
      --out "${RUN_ENV_JSON}" \
      --model "${MODEL}" \
      --run-config-env "${RUN_CONFIG_ENV}" \
      --controlled-root "${RUN_ROOT}" || true
  fi
}

run_controlled_level() {
  local level="$1"
  local tool_wait_ms="$2"
  local target_prompt_tokens="$3"
  local fillers="$4"
  local request_concurrency="$5"
  local filler_prompt_tokens="$6"
  local max_total_tokens="$7"
  local mem_fraction_static="$8"

  echo
  echo "==== Pressure level ${level}: controlled replay ===="
  env \
    PRESSURE_LEVEL="${level}" \
    RESULT_ROOT="${RUN_ROOT}" \
    LATEST_REPORT_ROOT="${RUN_ROOT}/_latest_scratch" \
    WORKLOAD_SOURCE="${WORKLOAD_SOURCE}" \
    MODES="${MODES}" \
    MAX_PAIRS="1" \
    TOOL_WAIT_LIST_MS="${tool_wait_ms}" \
    FILLER_LIST="${fillers}" \
    FILLER_PROMPT_TOKENS="${filler_prompt_tokens}" \
    TARGET_PROMPT_TOKENS="${target_prompt_tokens}" \
    SYNTHETIC_PROMPT_TOKENS="${target_prompt_tokens}" \
    SYNTHETIC_REPLAY_SUFFIX_TOKENS="256" \
    FILLER_DIVERGE_EARLY="1" \
    REQUEST_CONCURRENCY="${request_concurrency}" \
    MAX_TOTAL_TOKENS="${max_total_tokens}" \
    HICACHE_SIZE_GB="8" \
    MEM_FRACTION_STATIC="${mem_fraction_static}" \
    MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS}" \
    AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER}" \
    AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL}" \
    AGENTIC_RUNTIME_TELEMETRY="${AGENTIC_RUNTIME_TELEMETRY}" \
    AGENTIC_RUNTIME_TELEMETRY_BACKEND="${AGENTIC_RUNTIME_TELEMETRY_BACKEND}" \
    AGENTIC_KV_GPU_UTIL_SAMPLER="${AGENTIC_KV_GPU_UTIL_SAMPLER}" \
    GPU_UTIL_SAMPLE_INTERVAL_MS="${GPU_UTIL_SAMPLE_INTERVAL_MS}" \
    bash scripts/run_milestone27_real_prompt_controlled_replay.sh "${MODEL}"
}

run_burst_level() {
  local level="$1"

  echo
  echo "==== Pressure level ${level}: urgent burst ===="
  env \
    PRESSURE_LEVEL="${level}" \
    RESULT_ROOT="${RUN_ROOT}" \
    LATEST_REPORT_ROOT="${RUN_ROOT}/_latest_scratch" \
    WORKLOAD_SOURCE="${WORKLOAD_SOURCE}" \
    MODES="${MODES}" \
    SESSION_COUNT="4" \
    SYNC_REPLAY_AFTER_INITIALS="1" \
    ARRIVAL_SHAPE="burst" \
    BURST_SIZE="4" \
    ARRIVAL_GAP_MS="0" \
    BURST_GAP_MS="0" \
    TOOL_WAIT_LIST_MS="50" \
    TOOL_WAIT_JITTER_MS="0" \
    BACKGROUND_FILLERS_PER_SESSION="0" \
    REQUEST_CONCURRENCY="8" \
    FILLER_PROMPT_TOKENS="1536" \
    TARGET_PROMPT_TOKENS="4096" \
    SYNTHETIC_PROMPT_TOKENS="4096" \
    SYNTHETIC_REPLAY_SUFFIX_TOKENS="256" \
    MAX_TOTAL_TOKENS="12288" \
    HICACHE_SIZE_GB="8" \
    MEM_FRACTION_STATIC="0.72" \
    MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS}" \
    AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER}" \
    AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL}" \
    AGENTIC_RUNTIME_TELEMETRY="${AGENTIC_RUNTIME_TELEMETRY}" \
    AGENTIC_RUNTIME_TELEMETRY_BACKEND="${AGENTIC_RUNTIME_TELEMETRY_BACKEND}" \
    AGENTIC_KV_GPU_UTIL_SAMPLER="${AGENTIC_KV_GPU_UTIL_SAMPLER}" \
    GPU_UTIL_SAMPLE_INTERVAL_MS="${GPU_UTIL_SAMPLE_INTERVAL_MS}" \
    bash scripts/run_milestone36_multi_session_agentic_replay.sh "${MODEL}"
}

build_final_report() {
  echo
  echo "==== Building combined pressure-ladder report ===="
  write_run_config
  collect_run_environment
  "${PYTHON_BIN}" scripts/build_milestone27_controlled_replay_report.py \
    --root "${RUN_ROOT}" \
    --out-dir "${REPORT_DIR}/report" \
    --latest-root "${RESULTS_ROOT}" \
    --max-timeline-gaps "${MAX_TIMELINE_GAPS}" \
    --run-environment-json "${RUN_ENV_JSON}" \
    --gpu-util-csv "${GPU_UTIL_CSV}"
  cp -f "${REPORT_DIR}/report/controlled_replay_report.html" "${REPORT_DIR}/master_report.html"
  for artifact in controlled_replay_report.json controlled_replay_gaps.csv global_kv_readiness_by_mode.csv global_kv_readiness_by_mode_summary.csv global_replay_start_by_mode_summary.csv replay_delay_gap_verdicts.csv replay_path_ledger.csv hardware_counterfactual.csv instrumentation_coverage.csv request_id_coverage_report.csv exact_kv_movement_attribution.csv exact_kv_movement_summary.csv kv_block_ledger.csv kv_block_ledger.json kv_block_lifecycle_summary.csv kv_block_gap_summary.csv gpu_utilization_samples.csv runtime_telemetry_events.csv request_state_snapshots.csv; do
    if [[ -f "${REPORT_DIR}/report/${artifact}" ]]; then
      cp -f "${REPORT_DIR}/report/${artifact}" "${REPORT_DIR}/${artifact}"
    fi
  done
  REPORT_LABEL="${REPORT_LABEL}" \
  MODEL="${MODEL}" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  REPORT_DIR="${REPORT_DIR}" \
  UPDATE_LATEST="${UPDATE_LATEST}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR"])
results_root = Path(os.environ["RESULTS_ROOT"])
manifest = {
    "report_label": os.environ["REPORT_LABEL"],
    "experiment_kind": "pressure_ladder_quick",
    "model": os.environ["MODEL"],
    "workload_source": "synthetic",
    "controlled_root": os.environ["RUN_ROOT"],
    "latest_report": str(results_root / "latest_master_report.html"),
    "archived_report": str(report_dir / "master_report.html"),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "script": "scripts/run_hatcher_pressure_ladder_quick.sh",
    "pressure_profile": "p0_p3_p5_quick",
    "pressure_knobs": {
        "modes": "no_prefetch e2e_priority_hints",
        "pressure_ladder_levels": "P0 Control; P3 High; P5 Burst Cliff",
        "p0_control": "tool_wait_ms=500,target_prompt_tokens=1024,fillers=0,request_concurrency=1,urgent_agents=1,max_total_tokens=8192",
        "p3_high": "tool_wait_ms=50,target_prompt_tokens=4096,fillers=32,request_concurrency=8,urgent_agents=1",
        "p5_burst_cliff": "tool_wait_ms=50,target_prompt_tokens=4096,background_fillers_per_session=0,request_concurrency=8,urgent_agents=4,arrival_shape=burst,sync_replay_after_initials=1",
    },
}
(report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if os.environ.get("UPDATE_LATEST") == "1":
    (results_root / "latest_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

echo "Hatcher Pressure Ladder Quick"
echo "MODEL=${MODEL}"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "REPORT_DIR=${REPORT_DIR}"
echo "MODES=${MODES}"
echo "Levels: P0 Control, P3 High, P5 Burst Cliff"

start_gpu_util_sampler
run_controlled_level "p0_control" "500" "1024" "0" "1" "768" "8192" "0.72"
run_controlled_level "p3_high" "50" "4096" "32" "8" "1536" "12288" "0.72"
run_burst_level "p5_burst_cliff"
stop_gpu_util_sampler
build_final_report

echo
echo "Done."
echo "Latest report: ${RESULTS_ROOT}/latest_master_report.html"
echo "Archived labeled report: ${REPORT_DIR}/master_report.html"
