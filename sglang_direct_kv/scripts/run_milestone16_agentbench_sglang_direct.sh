#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
FRONTEND_URL="${FRONTEND_URL:-${HOST_URL}/v1/chat/completions}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone16_agentbench_sglang_direct}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-0}"
DATASET="${DATASET:-ScaleAI/SWE-bench_Pro}"
SPLIT="${SPLIT:-test}"
APP_VARIANT="${APP_VARIANT:-upstream_deploy_coding_agent}"
HINT_PROFILE="${HINT_PROFILE:-high-reuse}"
HINT_PROVIDER="${HINT_PROVIDER:-agentbench}"
AGENTBENCH_EXECUTION_LOOP="${AGENTBENCH_EXECUTION_LOOP:-1}"
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS="${AGENTBENCH_EXECUTION_LOOP_MAX_STEPS:-3}"
AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST="${AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST:-0}"
AGENTBENCH_EXECUTION_GUARD="${AGENTBENCH_EXECUTION_GUARD:-0}"
AGENTBENCH_DEEPAGENTS_SOURCE="${AGENTBENCH_DEEPAGENTS_SOURCE:-upstream}"
AGENTBENCH_FORCE_TOOL_CHOICE="${AGENTBENCH_FORCE_TOOL_CHOICE:-auto}"
AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT="${AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT:-1}"
AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE="${AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE:-1}"
AGENTBENCH_SOFT_STOP_RECURSION="${AGENTBENCH_SOFT_STOP_RECURSION:-1}"
AGENTBENCH_AGENT_RECURSION_LIMIT="${AGENTBENCH_AGENT_RECURSION_LIMIT:-300}"
PROMPT_EVOLUTION_VALUE_CHAR_LIMIT="${PROMPT_EVOLUTION_VALUE_CHAR_LIMIT:-50000}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
AGENTBENCH_INSTALL_DEPS="${AGENTBENCH_INSTALL_DEPS:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-14}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DIRECT_ROOT}/.." && pwd)"
cd "${DIRECT_ROOT}"

if [[ -n "${AGENTBENCH_ROOT:-}" ]]; then
  AGENTBENCH_ROOT="$(cd "${AGENTBENCH_ROOT}" && pwd)"
elif [[ -d "${PROJECT_ROOT}/../kv_cache_offloading/agentbench" ]]; then
  AGENTBENCH_ROOT="$(cd "${PROJECT_ROOT}/../kv_cache_offloading" && pwd)"
elif [[ -d "${HOME}/kv_cache_offloading/agentbench" ]]; then
  AGENTBENCH_ROOT="$(cd "${HOME}/kv_cache_offloading" && pwd)"
else
  echo "Could not find kv_cache_offloading/agentbench." >&2
  echo "Set AGENTBENCH_ROOT=/path/to/kv_cache_offloading and rerun." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

mkdir -p "${RESULT_ROOT}" "${LATEST_REPORT_ROOT}"
TRACE="${RESULT_ROOT}/agentbench_sglang_trace.jsonl"
COPY_TELEMETRY="${RESULT_ROOT}/agentbench_sglang_kv_copy_telemetry.jsonl"
SERVER_LOG="${RESULT_ROOT}/sglang_server.log"
DRIVER_LOG="${RESULT_ROOT}/driver.log"
TASK_INDEX_CSV="${RESULT_ROOT}/agentbench_sglang_task_index.csv"
REPORT_ROOT="${RESULT_ROOT}/report"
WORKLOAD_JSONL="${RESULT_ROOT}/agentbench_replay_workload.jsonl"
WORKLOAD_CSV="${RESULT_ROOT}/agentbench_replay_workload.csv"

server_pid=""

cleanup_server() {
  if [[ -n "${server_pid}" ]]; then
    kill "-${server_pid}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "-${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

wait_for_server() {
  local ready=0
  for _ in $(seq 1 300); do
    if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "SGLang exited before becoming ready. Log tail:" | tee -a "${DRIVER_LOG}"
      tail -120 "${SERVER_LOG}" | tee -a "${DRIVER_LOG}" || true
      exit 1
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "SGLang did not become ready. Log tail:" | tee -a "${DRIVER_LOG}"
    tail -120 "${SERVER_LOG}" | tee -a "${DRIVER_LOG}" || true
    exit 1
  fi
}

latest_agentbench_result_dir() {
  ls -td "${AGENTBENCH_ROOT}/experiments/raw/agentbench/results/"* 2>/dev/null | head -1 || true
}

append_task_index_row() {
  local task_index="$1"
  local run_id="$2"
  local result_dir="$3"
  local task_log="$4"
  local status="$5"
  local report_dir="${AGENTBENCH_ROOT}/experiments/reports/runs/${run_id}"
  if [[ ! -f "${TASK_INDEX_CSV}" ]]; then
    printf 'task_index,run_id,status,result_dir,report_dir,task_log,repo,model\n' > "${TASK_INDEX_CSV}"
  fi
  local repo=""
  if [[ -f "${result_dir}/others/result.json" ]]; then
    repo="$("${PYTHON_BIN}" - "${result_dir}/others/result.json" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
print((data.get("task") or {}).get("repo", ""))
PY
)"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${task_index}" "${run_id}" "${status}" "${result_dir}" "${report_dir}" "${task_log}" "${repo}" "${MODEL}" \
    >> "${TASK_INDEX_CSV}"
}

echo "Milestone 16: AgentBench -> Deep Agents -> SGLang direct" | tee "${DRIVER_LOG}"
echo "MODEL=${MODEL}" | tee -a "${DRIVER_LOG}"
echo "AGENTBENCH_ROOT=${AGENTBENCH_ROOT}" | tee -a "${DRIVER_LOG}"
echo "FRONTEND_URL=${FRONTEND_URL}" | tee -a "${DRIVER_LOG}"
echo "RESULT_ROOT=${RESULT_ROOT}" | tee -a "${DRIVER_LOG}"
echo "TASK_RANGE=${START_INDEX}-${END_INDEX}" | tee -a "${DRIVER_LOG}"
echo "Dynamo is not used in this milestone." | tee -a "${DRIVER_LOG}"

if [[ "${AGENTBENCH_INSTALL_DEPS}" == "1" ]]; then
  echo "Installing/refreshing AgentBench Python dependencies..." | tee -a "${DRIVER_LOG}"
  (cd "${AGENTBENCH_ROOT}" && "${PYTHON_BIN}" -m pip install -r agentbench/requirements.txt) 2>&1 | tee -a "${DRIVER_LOG}"
fi

if [[ -x "${AGENTBENCH_ROOT}/agentbench/ensure_deepagents_ready.sh" ]]; then
  echo "Ensuring Deep Agents dependency is ready..." | tee -a "${DRIVER_LOG}"
  (cd "${AGENTBENCH_ROOT}" && AGENTBENCH_DEEPAGENTS_SOURCE="${AGENTBENCH_DEEPAGENTS_SOURCE}" ./agentbench/ensure_deepagents_ready.sh) 2>&1 | tee -a "${DRIVER_LOG}"
fi

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 16." >&2
  exit 1
fi

rm -f "${TRACE}" "${COPY_TELEMETRY}" "${SERVER_LOG}" "${TASK_INDEX_CSV}"

export AGENTIC_KV_TRACE_ENABLE=1
export AGENTIC_KV_TRACE_PATH="${TRACE}"
export AGENTIC_KV_COPY_TELEMETRY_ENABLE=1
export AGENTIC_KV_COPY_TELEMETRY_PATH="${COPY_TELEMETRY}"
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-} --max-total-tokens ${MAX_TOTAL_TOKENS}"
export HICACHE_SIZE_GB
export MEM_FRACTION_STATIC

echo "Starting SGLang direct server..." | tee -a "${DRIVER_LOG}"
setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${SERVER_LOG}" 2>&1 &
server_pid="$!"
echo "${server_pid}" > "${RESULT_ROOT}/server.pid"
wait_for_server
echo "SGLang ready." | tee -a "${DRIVER_LOG}"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  PREFLIGHT_DIR="${RESULT_ROOT}/tool_loop_preflight"
  rm -rf "${PREFLIGHT_DIR}"
  mkdir -p "${PREFLIGHT_DIR}"
  echo "Running direct-SGLang Deep Agents tool-loop preflight..." | tee -a "${DRIVER_LOG}"
  (
    cd "${AGENTBENCH_ROOT}"
    AGENTBENCH_DEEPAGENTS_SOURCE="${AGENTBENCH_DEEPAGENTS_SOURCE}" \
    AGENTBENCH_FORCE_TOOL_CHOICE="${AGENTBENCH_FORCE_TOOL_CHOICE}" \
    AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT="${AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT}" \
    AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE="${AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE}" \
    "${PYTHON_BIN}" "${DIRECT_ROOT}/scripts/run_agentbench_sglang_preflight.py" \
      --agentbench-root "${AGENTBENCH_ROOT}" \
      -- \
      --frontend-url "${FRONTEND_URL}" \
      --model "${MODEL}" \
      --case "${PROMPT_EVOLUTION_TOOL_LOOP_CASE:-edit-validate}" \
      --output-dir "${PREFLIGHT_DIR}"
  ) 2>&1 | tee "${RESULT_ROOT}/tool_loop_preflight.log"
fi

total_cases=$((END_INDEX - START_INDEX + 1))
case_num=0
for index in $(seq "${START_INDEX}" "${END_INDEX}"); do
  case_num=$((case_num + 1))
  TASK_LOG="${RESULT_ROOT}/task_${index}.log"
  BEFORE_RESULT="$(latest_agentbench_result_dir)"
  echo "===== AgentBench task [${case_num}/${total_cases}] index ${index} =====" | tee -a "${DRIVER_LOG}" "${TASK_LOG}"
  status=0
  set +e
  (
    cd "${AGENTBENCH_ROOT}"
    AGENTBENCH_DEEPAGENTS_SOURCE="${AGENTBENCH_DEEPAGENTS_SOURCE}" \
    AGENTBENCH_FORCE_TOOL_CHOICE="${AGENTBENCH_FORCE_TOOL_CHOICE}" \
    AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT="${AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT}" \
    AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE="${AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE}" \
    AGENTBENCH_EXECUTION_LOOP="${AGENTBENCH_EXECUTION_LOOP}" \
    AGENTBENCH_EXECUTION_LOOP_MAX_STEPS="${AGENTBENCH_EXECUTION_LOOP_MAX_STEPS}" \
    AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST="${AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST}" \
    AGENTBENCH_EXECUTION_GUARD="${AGENTBENCH_EXECUTION_GUARD}" \
    AGENTBENCH_SOFT_STOP_RECURSION="${AGENTBENCH_SOFT_STOP_RECURSION}" \
    AGENTBENCH_AGENT_RECURSION_LIMIT="${AGENTBENCH_AGENT_RECURSION_LIMIT}" \
    AGENTBENCH_SGLANG_PREFETCH_MODE="live_direct" \
    "${PYTHON_BIN}" "${DIRECT_ROOT}/scripts/run_agentbench_sglang_task.py" \
      --agentbench-root "${AGENTBENCH_ROOT}" \
      -- \
      --app-variant "${APP_VARIANT}" \
      --frontend-url "${FRONTEND_URL}" \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --split "${SPLIT}" \
      --index "${index}" \
      --hint-provider "${HINT_PROVIDER}" \
      --hint-profile "${HINT_PROFILE}" \
      --prompt-evolution-value-char-limit "${PROMPT_EVOLUTION_VALUE_CHAR_LIMIT}" \
      --quiet-checkpoints
  ) 2>&1 | tee -a "${DRIVER_LOG}" "${TASK_LOG}"
  status="${PIPESTATUS[0]}"
  set -e
  AFTER_RESULT="$(latest_agentbench_result_dir)"
  if [[ -n "${AFTER_RESULT}" && "${AFTER_RESULT}" != "${BEFORE_RESULT}" ]]; then
    run_id="$(basename "${AFTER_RESULT}")"
    append_task_index_row "${index}" "${run_id}" "${AFTER_RESULT}" "${TASK_LOG}" "${status}"
    echo "Task index ${index} produced run ${run_id} with status ${status}." | tee -a "${DRIVER_LOG}"
  else
    echo "Task index ${index} did not produce a new AgentBench result directory." | tee -a "${DRIVER_LOG}"
    if [[ "${status}" -ne 0 ]]; then
      exit "${status}"
    fi
  fi
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
done

echo "Building live AgentBench direct-SGLang report..." | tee -a "${DRIVER_LOG}"
"${PYTHON_BIN}" scripts/summarize_agentbench_sglang_direct.py \
  --index-csv "${TASK_INDEX_CSV}" \
  --trace "${TRACE}" \
  --copy-telemetry "${COPY_TELEMETRY}" \
  --out-root "${REPORT_ROOT}" \
  --latest-root "${LATEST_REPORT_ROOT}" \
  2>&1 | tee -a "${DRIVER_LOG}"

echo "Extracting real AgentBench prompts into replay workload..." | tee -a "${DRIVER_LOG}"
"${PYTHON_BIN}" scripts/extract_agentbench_trace_replay_workload.py \
  --index-csv "${TASK_INDEX_CSV}" \
  --out-jsonl "${WORKLOAD_JSONL}" \
  --out-csv "${WORKLOAD_CSV}" \
  --max-sessions "${AGENTBENCH_REPLAY_MAX_SESSIONS:-24}" \
  --min-gap-ms "${AGENTBENCH_REPLAY_MIN_GAP_MS:-0}" \
  2>&1 | tee -a "${DRIVER_LOG}"

mkdir -p "${LATEST_REPORT_ROOT}/latest_real"
cp -f "${WORKLOAD_JSONL}" "${LATEST_REPORT_ROOT}/latest_real/agentbench_replay_workload.jsonl"
cp -f "${WORKLOAD_CSV}" "${LATEST_REPORT_ROOT}/latest_real/agentbench_replay_workload.csv"

echo
echo "Milestone 16 outputs:"
echo "  Live report: ${REPORT_ROOT}/agentbench_sglang_direct_report.html"
echo "  Latest live report: ${LATEST_REPORT_ROOT}/latest_real/agentbench_sglang_direct_report.html"
echo "  Replay workload: ${WORKLOAD_JSONL}"
echo "  Latest replay workload: ${LATEST_REPORT_ROOT}/latest_real/agentbench_replay_workload.jsonl"
echo "  SGLang trace: ${TRACE}"
