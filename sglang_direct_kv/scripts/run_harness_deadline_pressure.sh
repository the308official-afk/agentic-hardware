#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:31080}"
REPORT_LABEL="${REPORT_LABEL:-harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-artifacts/results}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/runs/controlled/${REPORT_LABEL}}"
REPORT_DIR="${REPORT_DIR:-${RESULTS_ROOT}/reports/${REPORT_LABEL}}"
UPDATE_LATEST="${UPDATE_LATEST:-1}"
HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code nemo_agent_toolkit deepseek_harness}"
MODES="${MODES:-no_prefetch e2e_priority_hints}"
PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p3_high p5_boss_queue}"
MAX_TIMELINE_GAPS="${MAX_TIMELINE_GAPS:-96}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SERVER_READY_TIMEOUT_SECS="${SERVER_READY_TIMEOUT_SECS:-900}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-12288}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
BASE_EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule}"
AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-1}"
AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL:-1}"
AGENTIC_RUNTIME_TELEMETRY="${AGENTIC_RUNTIME_TELEMETRY:-1}"
AGENTIC_RUNTIME_TELEMETRY_BACKEND="${AGENTIC_RUNTIME_TELEMETRY_BACKEND:-sglang}"
AGENTIC_KV_GPU_UTIL_SAMPLER="${AGENTIC_KV_GPU_UTIL_SAMPLER:-1}"
GPU_UTIL_SAMPLE_INTERVAL_MS="${GPU_UTIL_SAMPLE_INTERVAL_MS:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
export PYTHON_BIN

RESULTS_ROOT="$(mkdir -p "${RESULTS_ROOT}" && cd "${RESULTS_ROOT}" && pwd)"
RUN_ROOT="$(mkdir -p "${RUN_ROOT}" && cd "${RUN_ROOT}" && pwd)"
REPORT_DIR="$(mkdir -p "${REPORT_DIR}" && cd "${REPORT_DIR}" && pwd)"
RUN_CONFIG_ENV="${REPORT_DIR}/run_config.env"
RUN_ENV_JSON="${REPORT_DIR}/run_environment.json"
GPU_UTIL_CSV="${REPORT_DIR}/gpu_utilization_samples.csv"
GPU_UTIL_LOG="${REPORT_DIR}/gpu_utilization_sampler.log"
GPU_UTIL_SAMPLER_PID=""
SERVER_PID=""
GATEWAY_PID=""

cleanup_case() {
  if [[ -n "${GATEWAY_PID}" ]]; then
    kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
    wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
    GATEWAY_PID=""
  fi
  if [[ -n "${SERVER_PID}" ]]; then
    kill "-${SERVER_PID}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "-${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}

stop_gpu_util_sampler() {
  if [[ -n "${GPU_UTIL_SAMPLER_PID}" ]]; then
    kill "${GPU_UTIL_SAMPLER_PID}" >/dev/null 2>&1 || true
    wait "${GPU_UTIL_SAMPLER_PID}" >/dev/null 2>&1 || true
    GPU_UTIL_SAMPLER_PID=""
  fi
}
trap 'cleanup_case; stop_gpu_util_sampler' EXIT

start_gpu_util_sampler() {
  if [[ "${AGENTIC_KV_GPU_UTIL_SAMPLER}" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU util sampler disabled: nvidia-smi not found."
    return
  fi
  "${PYTHON_BIN}" scripts/sample_gpu_utilization.py \
    --out "${GPU_UTIL_CSV}" \
    --interval-ms "${GPU_UTIL_SAMPLE_INTERVAL_MS}" >"${GPU_UTIL_LOG}" 2>&1 &
  GPU_UTIL_SAMPLER_PID="$!"
}

wait_for_server() {
  local log="$1"
  local ready=0
  for _ in $(seq 1 "${SERVER_READY_TIMEOUT_SECS}"); do
    if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "SGLang exited before becoming ready. Log tail:"
      tail -160 "${log}" || true
      exit 1
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "SGLang did not become ready. Log tail:"
    tail -160 "${log}" || true
    exit 1
  fi
}

wait_for_gateway() {
  for _ in $(seq 1 60); do
    if curl -fsS "${GATEWAY_URL}/health" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "Harness gateway did not become ready." >&2
  exit 1
}

level_knobs() {
  case "$1" in
    p0_control)
      echo "tool_wait_ms=500 target_prompt_tokens=1024 filler_sessions=0 filler_prompt_tokens=768 session_count=1 concurrency=1"
      ;;
    p3_high)
      echo "tool_wait_ms=50 target_prompt_tokens=4096 filler_sessions=32 filler_prompt_tokens=1536 session_count=1 concurrency=8"
      ;;
    p5_boss_queue)
      echo "tool_wait_ms=50 target_prompt_tokens=4096 filler_sessions=4 filler_prompt_tokens=2048 session_count=4 concurrency=12"
      ;;
    *)
      echo "Unknown pressure level: $1" >&2
      exit 2
      ;;
  esac
}

write_run_config() {
  {
    echo "REPORT_LABEL=${REPORT_LABEL}"
    echo "MODEL=${MODEL}"
    echo "EXPERIMENT_KIND=multi_harness_deadline_pressure"
    echo "RESULTS_ROOT=${RESULTS_ROOT}"
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "REPORT_DIR=${REPORT_DIR}"
    echo "HARNESSES=${HARNESSES}"
    echo "MODES=${MODES}"
    echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"
    echo "P0_CONTROL=tool_wait_ms=500,target_prompt_tokens=1024,fillers=0,urgent_agents=1"
    echo "P3_QUEUE_PRESSURE=tool_wait_ms=50,target_prompt_tokens=4096,fillers=32,urgent_agents=1"
    echo "P5_BOSS_QUEUE=tool_wait_ms=50,target_prompt_tokens=4096,fillers_per_session=4,urgent_agents=4"
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

run_case() {
  local harness="$1"
  local mode="$2"
  local level="$3"
  local knobs
  knobs="$(level_knobs "${level}")"
  local tool_wait_ms target_prompt_tokens filler_sessions filler_prompt_tokens session_count concurrency
  eval "${knobs}"
  local case_id="${harness}_${level}_${mode}_tw${tool_wait_ms}_f${filler_sessions}"
  local case_root="${RUN_ROOT}/${case_id}"
  local trace="${case_root}/m27_trace.jsonl"
  local telemetry="${case_root}/m27_copy_telemetry.jsonl"
  local runtime_telemetry="${case_root}/runtime_telemetry.jsonl"
  local metrics="${case_root}/m27_metrics.jsonl"
  local server_log="${case_root}/sglang_server.log"
  local gateway_log="${case_root}/harness_gateway.log"
  local gateway_events="${case_root}/harness_gateway_events.jsonl"

  mkdir -p "${case_root}"
  rm -f "${trace}" "${telemetry}" "${runtime_telemetry}" "${metrics}" "${server_log}" "${gateway_log}" "${gateway_events}"

  echo
  echo "==== Multi-harness case: harness=${harness} mode=${mode} level=${level} ===="
  export AGENTIC_KV_TRACE_ENABLE=1
  export AGENTIC_KV_TRACE_PATH="${trace}"
  export AGENTIC_KV_TRACE_SCHEDULER
  export AGENTIC_KV_TRACE_KV_POOL
  export AGENTIC_KV_COPY_TELEMETRY_ENABLE=1
  export AGENTIC_KV_COPY_TELEMETRY_PATH="${telemetry}"
  export AGENTIC_RUNTIME_TELEMETRY
  export AGENTIC_RUNTIME_TELEMETRY_BACKEND
  export AGENTIC_RUNTIME_TELEMETRY_PATH="${runtime_telemetry}"
  export HICACHE_SIZE_GB
  export MEM_FRACTION_STATIC
  export EXTRA_SERVER_ARGS="${BASE_EXTRA_SERVER_ARGS} --max-total-tokens ${MAX_TOTAL_TOKENS}"
  if [[ "${mode}" == "e2e_priority_hints" ]]; then
    export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS} --enable-cache-report --enable-priority-scheduling --default-priority-value 0 --schedule-policy fcfs"
  fi

  setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${server_log}" 2>&1 &
  SERVER_PID="$!"
  wait_for_server "${server_log}"

  "${PYTHON_BIN}" scripts/harness_sglang_gateway.py \
    --listen-host 127.0.0.1 \
    --listen-port 31080 \
    --target-base "${HOST_URL}" \
    --trace "${trace}" \
    --log "${gateway_events}" \
    --model "${MODEL}" >"${gateway_log}" 2>&1 &
  GATEWAY_PID="$!"
  wait_for_gateway

  "${PYTHON_BIN}" scripts/run_multi_harness_replay_driver.py \
    --harness "${harness}" \
    --mode "${mode}" \
    --pressure-level "${level}" \
    --gateway-base "${GATEWAY_URL}" \
    --model "${MODEL}" \
    --trace "${trace}" \
    --out "${metrics}" \
    --log-dir "${case_root}/harness_logs" \
    --tool-wait-ms "${tool_wait_ms}" \
    --target-prompt-tokens "${target_prompt_tokens}" \
    --filler-sessions "${filler_sessions}" \
    --filler-prompt-tokens "${filler_prompt_tokens}" \
    --session-count "${session_count}" \
    --concurrency "${concurrency}" | tee "${case_root}/driver.log"

  cleanup_case
  echo "==== Completed: ${case_id} ===="
}

build_final_report() {
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
  for artifact in controlled_replay_report.json controlled_replay_gaps.csv global_kv_readiness_by_mode.csv global_kv_readiness_by_mode_summary.csv global_replay_start_by_mode_summary.csv replay_delay_gap_verdicts.csv replay_path_ledger.csv instrumentation_coverage.csv request_id_coverage_report.csv exact_kv_movement_attribution.csv exact_kv_movement_summary.csv kv_block_ledger.csv kv_block_ledger.json kv_block_lifecycle_summary.csv kv_block_gap_summary.csv gpu_utilization_samples.csv runtime_telemetry_events.csv request_state_snapshots.csv; do
    if [[ -f "${REPORT_DIR}/report/${artifact}" ]]; then
      cp -f "${REPORT_DIR}/report/${artifact}" "${REPORT_DIR}/${artifact}"
    fi
  done
  REPORT_LABEL="${REPORT_LABEL}" MODEL="${MODEL}" RESULTS_ROOT="${RESULTS_ROOT}" RUN_ROOT="${RUN_ROOT}" REPORT_DIR="${REPORT_DIR}" UPDATE_LATEST="${UPDATE_LATEST}" HARNESSES="${HARNESSES}" MODES="${MODES}" PRESSURE_LEVELS="${PRESSURE_LEVELS}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR"])
results_root = Path(os.environ["RESULTS_ROOT"])
manifest = {
    "report_label": os.environ["REPORT_LABEL"],
    "experiment_kind": "multi_harness_deadline_pressure",
    "model": os.environ["MODEL"],
    "controlled_root": os.environ["RUN_ROOT"],
    "latest_report": str(results_root / "latest_master_report.html"),
    "archived_report": str(report_dir / "master_report.html"),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "script": "scripts/run_harness_deadline_pressure.sh",
    "harnesses": os.environ.get("HARNESSES", ""),
    "modes": os.environ.get("MODES", ""),
    "pressure_levels": os.environ.get("PRESSURE_LEVELS", ""),
    "chart_title": "Replay Deadline Pressure Chart",
}
(report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if os.environ.get("UPDATE_LATEST") == "1":
    (results_root / "latest_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

echo "Multi-Harness Replay Deadline Pressure"
echo "MODEL=${MODEL}"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "HARNESSES=${HARNESSES}"
echo "MODES=${MODES}"
echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"

start_gpu_util_sampler
for harness in ${HARNESSES}; do
  for level in ${PRESSURE_LEVELS}; do
    for mode in ${MODES}; do
      run_case "${harness}" "${mode}" "${level}"
    done
  done
done
stop_gpu_util_sampler
build_final_report

echo
echo "Done."
echo "Latest report: ${RESULTS_ROOT}/latest_master_report.html"
echo "Archived labeled report: ${REPORT_DIR}/master_report.html"
