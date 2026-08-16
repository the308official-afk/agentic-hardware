#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
FRONTEND_URL_EXPLICIT="${FRONTEND_URL+x}"
FRONTEND_URL="${FRONTEND_URL:-${HOST_URL}/v1/chat/completions}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone21_exp6_direct_sglang}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
PROMPT_EVOLUTION_BATCH_ID="${PROMPT_EVOLUTION_BATCH_ID:-exp6_direct_sglang_$(date +%Y%m%d_%H%M%S)}"
SWEBENCH_TRAJECTORY_CATALOG_ID="${SWEBENCH_TRAJECTORY_CATALOG_ID:-swebench_trajectory_prompts_${PROMPT_EVOLUTION_BATCH_ID}}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-15}"
DATASET="${DATASET:-ScaleAI/SWE-bench_Pro}"
SPLIT="${SPLIT:-test}"
APP_VARIANT="${APP_VARIANT:-upstream_deploy_coding_agent}"
HINT_PROFILE="${HINT_PROFILE:-high-reuse}"
HINT_PROVIDER="${HINT_PROVIDER:-agentbench}"
AGENTBENCH_EXECUTION_LOOP="${AGENTBENCH_EXECUTION_LOOP:-1}"
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS="${AGENTBENCH_EXECUTION_LOOP_MAX_STEPS:-10}"
AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST="${AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST:-0}"
AGENTBENCH_EXECUTION_GUARD="${AGENTBENCH_EXECUTION_GUARD:-0}"
AGENTBENCH_DEEPAGENTS_SOURCE="${AGENTBENCH_DEEPAGENTS_SOURCE:-upstream}"
AGENTBENCH_FORCE_TOOL_CHOICE="${AGENTBENCH_FORCE_TOOL_CHOICE:-auto}"
AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT="${AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT:-1}"
AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE="${AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE:-1}"
AGENTBENCH_SOFT_STOP_RECURSION="${AGENTBENCH_SOFT_STOP_RECURSION:-1}"
PROMPT_EVOLUTION_SKIP_RECURSION_FAILURES="${PROMPT_EVOLUTION_SKIP_RECURSION_FAILURES:-${AGENTBENCH_SOFT_STOP_RECURSION}}"
AGENTBENCH_AGENT_RECURSION_LIMIT="${AGENTBENCH_AGENT_RECURSION_LIMIT:-1000}"
AGENTBENCH_TRACE_AGENT_STREAM="${AGENTBENCH_TRACE_AGENT_STREAM:-0}"
AGENTBENCH_TRACE_AGENT_STREAM_MODE="${AGENTBENCH_TRACE_AGENT_STREAM_MODE:-values}"
AGENTBENCH_WORKFLOW_MODE="${AGENTBENCH_WORKFLOW_MODE:-phased}"
AGENTBENCH_DIRECT_SGLANG_TOOL_RICH="${AGENTBENCH_DIRECT_SGLANG_TOOL_RICH:-1}"
AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT="${AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT:-1}"
AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS="${AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS:-1}"
AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD="${AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD:-1}"
PROMPT_EVOLUTION_VALUE_CHAR_LIMIT="${PROMPT_EVOLUTION_VALUE_CHAR_LIMIT:-200000}"
PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP="${PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP:-1}"
PROMPT_EVOLUTION_TOOL_LOOP_CASE="${PROMPT_EVOLUTION_TOOL_LOOP_CASE:-edit-validate}"
PROMPT_EVOLUTION_REFRESH_TRAJECTORY_CATALOG_EACH_TASK="${PROMPT_EVOLUTION_REFRESH_TRAJECTORY_CATALOG_EACH_TASK:-1}"
PROMPT_EVOLUTION_REFRESH_PUBLIC_REPORTS_EACH_TASK="${PROMPT_EVOLUTION_REFRESH_PUBLIC_REPORTS_EACH_TASK:-1}"
PROMPT_EVOLUTION_BUILD_TRAJECTORY_CATALOG="${PROMPT_EVOLUTION_BUILD_TRAJECTORY_CATALOG:-1}"
PROMPT_EVOLUTION_RESUME="${PROMPT_EVOLUTION_RESUME:-auto}"
PROMPT_EVOLUTION_RERUN_FAILED="${PROMPT_EVOLUTION_RERUN_FAILED:-0}"
AGENTBENCH_BATCH_CONTINUE_ON_ERROR="${AGENTBENCH_BATCH_CONTINUE_ON_ERROR:-0}"
AGENTBENCH_INSTALL_DEPS="${AGENTBENCH_INSTALL_DEPS:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
REUSE_SERVER="${REUSE_SERVER:-0}"
SERVER_MODE="${SERVER_MODE:-simple}"
ENABLE_TOOL_NORMALIZER_PROXY="${ENABLE_TOOL_NORMALIZER_PROXY:-1}"
TOOL_NORMALIZER_HOST="${TOOL_NORMALIZER_HOST:-127.0.0.1}"
TOOL_NORMALIZER_PORT="${TOOL_NORMALIZER_PORT:-31003}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-auto}"
REASONING_PARSER="${REASONING_PARSER:-auto}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
SAMPLING_DEFAULTS="${SAMPLING_DEFAULTS:-openai}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-14}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
MODEL_SMOKE_RETRIES="${MODEL_SMOKE_RETRIES:-120}"
MODEL_SMOKE_DELAY_SECS="${MODEL_SMOKE_DELAY_SECS:-5}"
MODEL_COOLDOWN_SECS="${MODEL_COOLDOWN_SECS:-5}"
SERVER_READY_TIMEOUT_SECS="${SERVER_READY_TIMEOUT_SECS:-360}"
SHARED_CHART_DIR="${SHARED_CHART_DIR:-experiments/charts}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DIRECT_ROOT}/.." && pwd)"
cd "${DIRECT_ROOT}"

RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
LATEST_REPORT_ROOT="$(mkdir -p "${LATEST_REPORT_ROOT}" && cd "${LATEST_REPORT_ROOT}" && pwd)"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

MODEL_LC="$(printf '%s' "${MODEL}" | tr '[:upper:]' '[:lower:]')"
if [[ "${TOOL_CALL_PARSER}" = "auto" ]]; then
  if [[ "${MODEL_LC}" == *"qwen3-coder"* || "${MODEL_LC}" == *"qwen3_coder"* ]]; then
    TOOL_CALL_PARSER="qwen3_coder"
  elif [[ "${MODEL_LC}" == *"qwen2.5"* || "${MODEL_LC}" == *"qwen2_5"* || "${MODEL_LC}" == *"qwen25"* ]]; then
    TOOL_CALL_PARSER="qwen25"
  else
    TOOL_CALL_PARSER="hermes"
  fi
fi
if [[ "${REASONING_PARSER}" = "auto" ]]; then
  if [[ "${MODEL_LC}" == *"qwen3-coder"* || "${MODEL_LC}" == *"qwen3_coder"* ]]; then
    REASONING_PARSER="qwen3"
  else
    REASONING_PARSER=""
  fi
fi

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

BATCH_DIR="${AGENTBENCH_ROOT}/experiments/reports/batches/${PROMPT_EVOLUTION_BATCH_ID}"
TRACE_INDEX_CSV="${BATCH_DIR}/task_trace_index.csv"
TRACE_INDEX_MD="${BATCH_DIR}/task_trace_index.md"
LATEST_TRACE_INDEX_CSV="${AGENTBENCH_ROOT}/experiments/reports/latest_prompt_evolution_trace_index.csv"
LATEST_TRACE_INDEX_MD="${AGENTBENCH_ROOT}/experiments/reports/latest_prompt_evolution_trace_index.md"
PROGRESS_CSV="${BATCH_DIR}/progress_overview.csv"
SKIPPED_CSV="${BATCH_DIR}/skipped_tasks.csv"
DRIVER_LOG="${RESULT_ROOT}/driver.log"
SERVER_LOG="${RESULT_ROOT}/sglang_server.log"
SMOKE_LOG="${RESULT_ROOT}/model_smoke.log"
PREFLIGHT_DIR="${RESULT_ROOT}/tool_loop_preflight"
PREFLIGHT_LOG="${RESULT_ROOT}/tool_loop_preflight.log"
TASK_INDEX_CSV="${RESULT_ROOT}/exp6_direct_sglang_task_index.csv"
server_pid=""
normalizer_pid=""

mkdir -p "${RESULT_ROOT}" "${LATEST_REPORT_ROOT}" "${BATCH_DIR}" "${SHARED_CHART_DIR}"

cleanup_server() {
  if [[ -n "${normalizer_pid}" ]]; then
    kill "${normalizer_pid}" >/dev/null 2>&1 || true
    wait "${normalizer_pid}" >/dev/null 2>&1 || true
    normalizer_pid=""
  fi
  if [[ -n "${server_pid}" ]]; then
    kill "-${server_pid}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "-${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

latest_agentbench_result_dir() {
  ls -td "${AGENTBENCH_ROOT}/experiments/raw/agentbench/results/"* 2>/dev/null | head -1 || true
}

server_is_ready() {
  curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1
}

wait_for_server() {
  local ready=0
  for _ in $(seq 1 "${SERVER_READY_TIMEOUT_SECS}"); do
    if server_is_ready; then
      ready=1
      break
    fi
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "SGLang exited before becoming ready. Log tail:" | tee -a "${DRIVER_LOG}"
      tail -160 "${SERVER_LOG}" | tee -a "${DRIVER_LOG}" || true
      exit 1
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "SGLang did not become ready. Log tail:" | tee -a "${DRIVER_LOG}"
    tail -160 "${SERVER_LOG}" | tee -a "${DRIVER_LOG}" || true
    exit 1
  fi
}

start_tool_normalizer_proxy() {
  [[ "${ENABLE_TOOL_NORMALIZER_PROXY}" = "1" ]] || return 0

  if [[ -n "${FRONTEND_URL_EXPLICIT}" && "${FRONTEND_URL}" != "${HOST_URL}/v1/chat/completions" ]]; then
    echo "Using explicit FRONTEND_URL=${FRONTEND_URL}; not starting tool normalizer proxy." | tee -a "${DRIVER_LOG}"
    return 0
  fi

  local proxy_log="${RESULT_ROOT}/tool_normalizer_proxy.log"
  local proxy_jsonl="${RESULT_ROOT}/tool_normalizer_proxy.jsonl"
  local existing_pids
  existing_pids="$(pgrep -f "openai_proxy_logger.py --listen-port ${TOOL_NORMALIZER_PORT}" || true)"
  if [[ -n "${existing_pids}" ]]; then
    echo "Stopping existing tool normalizer proxy on port ${TOOL_NORMALIZER_PORT}: ${existing_pids}" | tee -a "${DRIVER_LOG}"
    kill ${existing_pids} >/dev/null 2>&1 || true
    sleep 1
  fi

  : > "${proxy_jsonl}"
  echo "Starting direct-SGLang tool normalizer proxy on ${TOOL_NORMALIZER_HOST}:${TOOL_NORMALIZER_PORT}." | tee -a "${DRIVER_LOG}"
  proxy_args=(
    scripts/openai_proxy_logger.py
    --listen-host "${TOOL_NORMALIZER_HOST}" \
    --listen-port "${TOOL_NORMALIZER_PORT}" \
    --target-base "${HOST_URL}" \
    --log "${proxy_jsonl}" \
    --normalize-tool-calls
  )
  if [[ -n "${LIVE_HINT_LOG:-}" ]]; then
    proxy_args+=(--hint-log "${LIVE_HINT_LOG}")
  fi
  if [[ -n "${LIVE_HINT_PAYLOAD_DIR:-}" ]]; then
    proxy_args+=(--hint-payload-dir "${LIVE_HINT_PAYLOAD_DIR}")
  fi
  "${PYTHON_BIN}" "${proxy_args[@]}" >"${proxy_log}" 2>&1 &
  normalizer_pid="$!"
  echo "${normalizer_pid}" > "${RESULT_ROOT}/tool_normalizer_proxy.pid"

  local proxy_model_info="http://${TOOL_NORMALIZER_HOST}:${TOOL_NORMALIZER_PORT}/model_info"
  for _ in $(seq 1 30); do
    if curl -fsS "${proxy_model_info}" >/dev/null 2>&1; then
      FRONTEND_URL="http://${TOOL_NORMALIZER_HOST}:${TOOL_NORMALIZER_PORT}/v1/chat/completions"
      echo "Tool normalizer proxy ready. FRONTEND_URL=${FRONTEND_URL}" | tee -a "${DRIVER_LOG}"
      return 0
    fi
    if ! kill -0 "${normalizer_pid}" >/dev/null 2>&1; then
      echo "Tool normalizer proxy exited early. Log tail:" | tee -a "${DRIVER_LOG}"
      tail -80 "${proxy_log}" | tee -a "${DRIVER_LOG}" || true
      return 1
    fi
    sleep 1
  done

  echo "Tool normalizer proxy did not become ready. Log tail:" | tee -a "${DRIVER_LOG}"
  tail -80 "${proxy_log}" | tee -a "${DRIVER_LOG}" || true
  return 1
}

smoke_test_model() {
  local payload
  payload="$("${PYTHON_BIN}" -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 10, "temperature": 0}))' "${MODEL}")"
  : > "${SMOKE_LOG}"
  for attempt in $(seq 1 "${MODEL_SMOKE_RETRIES}"); do
    echo "Smoke test ${attempt}/${MODEL_SMOKE_RETRIES} for ${MODEL}" | tee -a "${DRIVER_LOG}"
    if curl -fsS "${FRONTEND_URL}" -H "Content-Type: application/json" -d "${payload}" >> "${SMOKE_LOG}" 2>&1; then
      echo "Smoke test passed." | tee -a "${DRIVER_LOG}"
      return 0
    fi
    sleep "${MODEL_SMOKE_DELAY_SECS}"
  done
  echo "Smoke test failed. See ${SMOKE_LOG}" | tee -a "${DRIVER_LOG}" >&2
  return 1
}

task_already_recorded() {
  local task_index="$1"
  [[ "${PROMPT_EVOLUTION_RESUME}" = "1" ]] || return 1
  [[ "${PROMPT_EVOLUTION_RERUN_FAILED}" = "1" ]] && return 1
  [[ -f "${TRACE_INDEX_CSV}" ]] || return 1
  "${PYTHON_BIN}" - "${TRACE_INDEX_CSV}" "${task_index}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
task_index = str(sys.argv[2])
for row in csv.DictReader(path.open()):
    if str(row.get("task_index", "")) == task_index and row.get("run_id"):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

append_progress_row() {
  local run_id="$1"
  RUN_ID="${run_id}" PROGRESS_CSV="${PROGRESS_CSV}" AGENTBENCH_ROOT="${AGENTBENCH_ROOT}" "${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["AGENTBENCH_ROOT"])
run_id = os.environ["RUN_ID"]
progress_csv = Path(os.environ["PROGRESS_CSV"])
overview_csv = root / "experiments/reports/all_runs_overview.csv"
if not overview_csv.exists():
    raise SystemExit(0)
rows = list(csv.DictReader(overview_csv.open()))
row = next((item for item in rows if item.get("run_id") == run_id), None)
if row is None:
    raise SystemExit(0)
write_header = not progress_csv.exists()
progress_csv.parent.mkdir(parents=True, exist_ok=True)
with progress_csv.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row.keys()), lineterminator="\n")
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

append_trace_index_row() {
  local run_id="$1"
  local task_index="$2"
  RUN_ID="${run_id}" \
  TASK_INDEX="${task_index}" \
  AGENTBENCH_ROOT="${AGENTBENCH_ROOT}" \
  TRACE_INDEX_CSV="${TRACE_INDEX_CSV}" \
  TRACE_INDEX_MD="${TRACE_INDEX_MD}" \
  LATEST_TRACE_INDEX_CSV="${LATEST_TRACE_INDEX_CSV}" \
  LATEST_TRACE_INDEX_MD="${LATEST_TRACE_INDEX_MD}" \
  "${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["AGENTBENCH_ROOT"])
run_id = os.environ["RUN_ID"]
task_index = os.environ["TASK_INDEX"]
trace_index_csv = Path(os.environ["TRACE_INDEX_CSV"])
trace_index_md = Path(os.environ["TRACE_INDEX_MD"])
latest_trace_index_csv = Path(os.environ["LATEST_TRACE_INDEX_CSV"])
latest_trace_index_md = Path(os.environ["LATEST_TRACE_INDEX_MD"])

overview_csv = root / "experiments/reports/all_runs_overview.csv"
overview_row = {}
if overview_csv.exists():
    rows = list(csv.DictReader(overview_csv.open()))
    overview_row = next((row for row in rows if row.get("run_id") == run_id), {}) or {}

result_dir = root / "experiments/raw/agentbench/results" / run_id
report_dir = root / "experiments/reports/runs" / run_id
row = {
    "task_index": task_index,
    "run_id": run_id,
    "repo": overview_row.get("repo", ""),
    "model": overview_row.get("model", ""),
    "hint_profile": overview_row.get("hint_profile", ""),
    "total_tool_calls": overview_row.get("total_tool_calls", ""),
    "execution_phase_tools": overview_row.get("execution_phase_tools", ""),
    "patch_nonempty": overview_row.get("patch_nonempty", ""),
    "result_dir": str(result_dir),
    "report_dir": str(report_dir),
    "prompt_evolution_report_md": str(result_dir / "prompt_evolution_report.md"),
    "prompt_evolution_report_csv": str(result_dir / "prompt_evolution_report.csv"),
    "final_model_request_json": str(result_dir / "prompt_evolution_values/03_final_model_request.json"),
    "tool_runtime_context_json": str(result_dir / "prompt_evolution_values/05_tool_runtime_context.json"),
    "runtime_preprocessing_json": str(result_dir / "prompt_evolution_values/06_runtime_preprocessing.json"),
    "model_behavior_json": str(result_dir / "prompt_evolution_values/07_model_behavior.json"),
    "phase_summary_md": str(report_dir / "phase_summary.md"),
    "phase_summary_csv": str(report_dir / "phase_summary.csv"),
    "tool_call_details_md": str(report_dir / "tool_call_details.md"),
    "tool_call_details_csv": str(report_dir / "tool_call_details.csv"),
    "workspace_patch": str(result_dir / "workspace.patch"),
}
fieldnames = list(row.keys())
existing_rows = []
if trace_index_csv.exists():
    existing_rows = list(csv.DictReader(trace_index_csv.open()))
    existing_rows = [
        item for item in existing_rows
        if item.get("run_id") != run_id and str(item.get("task_index", "")) != str(task_index)
    ]
existing_rows.append(row)
existing_rows.sort(key=lambda item: int(item.get("task_index") or 0))

trace_index_csv.parent.mkdir(parents=True, exist_ok=True)
with trace_index_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(existing_rows)

latest_trace_index_csv.parent.mkdir(parents=True, exist_ok=True)
latest_trace_index_csv.write_text(trace_index_csv.read_text(encoding="utf-8"), encoding="utf-8")

lines = [
    "# Direct SGLang Prompt Evolution Task Trace Index",
    "",
    "Each row points to the prompt-evolution and tool-call artifacts for one SWE-bench task in this direct-SGLang batch.",
    "",
    "| Task | Repo | Run ID | Tools | Patch | Key files |",
    "| --- | --- | --- | --- | --- | --- |",
]
for item in existing_rows:
    key_files = "<br>".join(
        [
            f"`{item['prompt_evolution_report_md']}`",
            f"`{item['tool_call_details_md']}`",
            f"`{item['phase_summary_md']}`",
            f"`{item['model_behavior_json']}`",
        ]
    )
    lines.append(
        "| {task} | {repo} | `{run_id}` | {tools} | {patch} | {files} |".format(
            task=item.get("task_index", ""),
            repo=item.get("repo", ""),
            run_id=item.get("run_id", ""),
            tools=item.get("execution_phase_tools", "") or item.get("total_tool_calls", ""),
            patch=item.get("patch_nonempty", ""),
            files=key_files,
        )
    )

trace_index_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
latest_trace_index_md.write_text(trace_index_md.read_text(encoding="utf-8"), encoding="utf-8")
PY
}

append_task_index_row() {
  local task_index="$1"
  local run_id="$2"
  local status="$3"
  local result_dir="${AGENTBENCH_ROOT}/experiments/raw/agentbench/results/${run_id}"
  local report_dir="${AGENTBENCH_ROOT}/experiments/reports/runs/${run_id}"
  local task_log="${BATCH_DIR}/task_${task_index}.log"
  if [[ ! -f "${TASK_INDEX_CSV}" ]]; then
    printf 'task_index,run_id,status,result_dir,report_dir,task_log,model\n' > "${TASK_INDEX_CSV}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "${task_index}" "${run_id}" "${status}" "${result_dir}" "${report_dir}" "${task_log}" "${MODEL}" \
    >> "${TASK_INDEX_CSV}"
}

publish_prompt_evolution_reports() {
  mkdir -p "${RESULT_ROOT}" "${LATEST_REPORT_ROOT}" "${AGENTBENCH_ROOT}/${SHARED_CHART_DIR}"
  local pairs=(
    "experiments/reports/prompt_evolution_task_summary.csv:exp6_direct_prompt_evolution_task_summary.csv"
    "experiments/reports/prompt_evolution_run_overview.csv:exp6_direct_prompt_evolution_run_overview.csv"
    "experiments/reports/latest_prompt_evolution_trace_index.csv:exp6_direct_prompt_evolution_trace_index.csv"
    "experiments/reports/latest_prompt_evolution_trace_index.md:exp6_direct_prompt_evolution_trace_index.md"
    "experiments/reports/latest_swebench_trajectory_prompt_catalog.csv:exp6_direct_swebench_trajectory_prompt_catalog.csv"
    "experiments/reports/latest_swebench_trajectory_prompt_catalog.jsonl:exp6_direct_swebench_trajectory_prompt_catalog.jsonl"
    "experiments/reports/latest_swebench_trajectory_task_prompt_counts.csv:exp6_direct_swebench_trajectory_task_prompt_counts.csv"
  )
  local pair source_path target_name
  for pair in "${pairs[@]}"; do
    source_path="${AGENTBENCH_ROOT}/${pair%%:*}"
    target_name="${pair##*:}"
    if [[ -f "${source_path}" ]]; then
      cp -f "${source_path}" "${RESULT_ROOT}/${target_name}"
      cp -f "${source_path}" "${LATEST_REPORT_ROOT}/latest_${target_name}"
      cp -f "${source_path}" "${AGENTBENCH_ROOT}/${SHARED_CHART_DIR}/${target_name}"
    fi
  done
}

refresh_trajectory_catalog() {
  [[ "${PROMPT_EVOLUTION_BUILD_TRAJECTORY_CATALOG}" = "1" ]] || return 0
  (
    cd "${AGENTBENCH_ROOT}"
    PYTHON_BIN="${PYTHON_BIN}" \
    SWEBENCH_TRAJECTORY_CATALOG_ID="${SWEBENCH_TRAJECTORY_CATALOG_ID}" \
    SWEBENCH_TRAJECTORY_TRACE_INDEX="${LATEST_TRACE_INDEX_CSV}" \
    ./agentbench/prepare_swebench_trajectory_prompts.sh
  ) 2>&1 | tee -a "${DRIVER_LOG}"
  publish_prompt_evolution_reports
}

append_skipped_row() {
  local task_index="$1"
  local run_id="$2"
  local reason="$3"
  local status="$4"
  local task_log="$5"
  if [[ ! -f "${SKIPPED_CSV}" ]]; then
    printf 'task_index,run_id,reason,exit_status,task_log\n' > "${SKIPPED_CSV}"
  fi
  printf '%s,%s,%s,%s,%s\n' "${task_index}" "${run_id}" "${reason}" "${status}" "${task_log}" >> "${SKIPPED_CSV}"
}

if [[ "${PROMPT_EVOLUTION_RESUME}" = "auto" ]]; then
  if [[ -f "${TRACE_INDEX_CSV}" ]]; then
    PROMPT_EVOLUTION_RESUME=1
  else
    PROMPT_EVOLUTION_RESUME=0
  fi
fi

: > "${DRIVER_LOG}"
{
  echo "Milestone 21: Direct SGLang Experiment 6 Prompt Evolution"
  echo "MODEL=${MODEL}"
  echo "AGENTBENCH_ROOT=${AGENTBENCH_ROOT}"
  echo "FRONTEND_URL=${FRONTEND_URL}"
  echo "RESULT_ROOT=${RESULT_ROOT}"
  echo "PROMPT_EVOLUTION_BATCH_ID=${PROMPT_EVOLUTION_BATCH_ID}"
  echo "SWEBENCH_TRAJECTORY_CATALOG_ID=${SWEBENCH_TRAJECTORY_CATALOG_ID}"
  echo "TASK_RANGE=${START_INDEX}-${END_INDEX}"
  echo "SERVER_MODE=${SERVER_MODE}"
  echo "TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-<unset>}"
  echo "REASONING_PARSER=${REASONING_PARSER:-<unset>}"
  echo "SAMPLING_BACKEND=${SAMPLING_BACKEND:-<unset>}"
  echo "SAMPLING_DEFAULTS=${SAMPLING_DEFAULTS:-<unset>}"
  echo "ENABLE_TOOL_NORMALIZER_PROXY=${ENABLE_TOOL_NORMALIZER_PROXY}"
  echo "AGENTBENCH_DIRECT_SGLANG_TOOL_RICH=${AGENTBENCH_DIRECT_SGLANG_TOOL_RICH}"
  echo "AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT=${AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT}"
  echo "AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS=${AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS}"
  echo "AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD=${AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD}"
  echo "Dynamo is not used."
  echo
} | tee -a "${DRIVER_LOG}"

if [[ "${AGENTBENCH_INSTALL_DEPS}" = "1" ]]; then
  echo "Installing/refreshing AgentBench Python dependencies..." | tee -a "${DRIVER_LOG}"
  (cd "${AGENTBENCH_ROOT}" && "${PYTHON_BIN}" -m pip install -r agentbench/requirements.txt) 2>&1 | tee -a "${DRIVER_LOG}"
fi

if [[ -x "${AGENTBENCH_ROOT}/agentbench/ensure_deepagents_ready.sh" ]]; then
  echo "Ensuring Deep Agents dependency is ready..." | tee -a "${DRIVER_LOG}"
  (cd "${AGENTBENCH_ROOT}" && AGENTBENCH_DEEPAGENTS_SOURCE="${AGENTBENCH_DEEPAGENTS_SOURCE}" ./agentbench/ensure_deepagents_ready.sh) 2>&1 | tee -a "${DRIVER_LOG}"
fi

if server_is_ready; then
  if [[ "${REUSE_SERVER}" = "1" ]]; then
    echo "Reusing existing SGLang server at ${HOST_URL}." | tee -a "${DRIVER_LOG}"
  else
    echo "A server is already listening at ${HOST_URL}. Stop it or set REUSE_SERVER=1." | tee -a "${DRIVER_LOG}" >&2
    exit 1
  fi
else
  rm -f "${SERVER_LOG}"
  SERVER_EXTRA_ARGS="${EXTRA_SERVER_ARGS:-} --max-total-tokens ${MAX_TOTAL_TOKENS}"
  if [[ -n "${TOOL_CALL_PARSER}" && "${SERVER_EXTRA_ARGS}" != *"--tool-call-parser"* ]]; then
    SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS} --tool-call-parser ${TOOL_CALL_PARSER}"
  fi
  if [[ -n "${REASONING_PARSER}" && "${SERVER_EXTRA_ARGS}" != *"--reasoning-parser"* ]]; then
    SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS} --reasoning-parser ${REASONING_PARSER}"
  fi
  if [[ -n "${SAMPLING_BACKEND}" && "${SAMPLING_BACKEND}" != "auto" && "${SERVER_EXTRA_ARGS}" != *"--sampling-backend"* ]]; then
    SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS} --sampling-backend ${SAMPLING_BACKEND}"
  fi
  if [[ -n "${SAMPLING_DEFAULTS}" && "${SAMPLING_DEFAULTS}" != "auto" && "${SERVER_EXTRA_ARGS}" != *"--sampling-defaults"* ]]; then
    SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS} --sampling-defaults ${SAMPLING_DEFAULTS}"
  fi
  export EXTRA_SERVER_ARGS="${SERVER_EXTRA_ARGS}"
  echo "Starting SGLang direct server..." | tee -a "${DRIVER_LOG}"
  echo "EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS}" | tee -a "${DRIVER_LOG}"
  if [[ "${SERVER_MODE}" = "hicache" ]]; then
    export HICACHE_SIZE_GB MEM_FRACTION_STATIC
    setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${SERVER_LOG}" 2>&1 &
  else
    setsid bash scripts/run_sglang_server.sh "${MODEL}" >"${SERVER_LOG}" 2>&1 &
  fi
  server_pid="$!"
  echo "${server_pid}" > "${RESULT_ROOT}/server.pid"
  wait_for_server
  echo "SGLang ready." | tee -a "${DRIVER_LOG}"
fi

start_tool_normalizer_proxy
smoke_test_model
if [[ "${MODEL_COOLDOWN_SECS}" -gt 0 ]]; then
  echo "Cooldown: ${MODEL_COOLDOWN_SECS}s" | tee -a "${DRIVER_LOG}"
  sleep "${MODEL_COOLDOWN_SECS}"
fi

if [[ "${RUN_PREFLIGHT}" = "1" && "${PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP}" = "1" ]]; then
  rm -rf "${PREFLIGHT_DIR}"
  mkdir -p "${PREFLIGHT_DIR}"
  echo "Running direct-SGLang Deep Agents tool-loop preflight..." | tee -a "${DRIVER_LOG}"
  (
    cd "${AGENTBENCH_ROOT}"
    AGENTBENCH_DEEPAGENTS_SOURCE="${AGENTBENCH_DEEPAGENTS_SOURCE}" \
    AGENTBENCH_FORCE_TOOL_CHOICE="${AGENTBENCH_FORCE_TOOL_CHOICE}" \
    AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT="${AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT}" \
    AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE="${AGENTBENCH_REQUIRE_GENERAL_PURPOSE_SUBAGENT_DISABLE}" \
    AGENTBENCH_TOOL_LOOP_RECURSION_LIMIT="${AGENTBENCH_TOOL_LOOP_RECURSION_LIMIT:-30}" \
    AGENTBENCH_TOOL_LOOP_TIMEOUT_SECONDS="${AGENTBENCH_TOOL_LOOP_TIMEOUT_SECONDS:-180}" \
    "${PYTHON_BIN}" "${DIRECT_ROOT}/scripts/run_agentbench_sglang_preflight.py" \
      --agentbench-root "${AGENTBENCH_ROOT}" \
      -- \
      --frontend-url "${FRONTEND_URL}" \
      --model "${MODEL}" \
      --case "${PROMPT_EVOLUTION_TOOL_LOOP_CASE}" \
      --output-dir "${PREFLIGHT_DIR}"
  ) 2>&1 | tee "${PREFLIGHT_LOG}"
  "${PYTHON_BIN}" - "${PREFLIGHT_DIR}/summary.json" "${DRIVER_LOG}" "${PREFLIGHT_LOG}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
driver_log = Path(sys.argv[2])
preflight_log = Path(sys.argv[3])
if not summary_path.exists():
    message = f"CRITICAL FAIL: preflight summary missing. See {preflight_log}"
    with driver_log.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    raise SystemExit(message)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
tool_calls = int(summary.get("ai_tool_call_count") or 0)
tool_messages = int(summary.get("tool_message_count") or 0)
case_success = bool(summary.get("case_success"))
multi_tool_loop = bool(summary.get("multi_tool_loop_observed"))
lines = [
    "Deep Agents tool-loop preflight result:",
    f"  tool_calls={tool_calls}",
    f"  tool_messages={tool_messages}",
    f"  multi_tool_loop_observed={multi_tool_loop}",
    f"  case_success={case_success}",
]
print("\n".join(lines))
with driver_log.open("a", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
if not case_success:
    raise SystemExit(f"CRITICAL FAIL: tool-loop preflight failed. See {preflight_log}")
PY
fi

total_cases=$((END_INDEX - START_INDEX + 1))
case_num=0
for index in $(seq "${START_INDEX}" "${END_INDEX}"); do
  case_num=$((case_num + 1))
  if task_already_recorded "${index}"; then
    echo "===== Skipping SWE-bench index ${index} [${case_num}/${total_cases}] already recorded =====" | tee -a "${DRIVER_LOG}"
    continue
  fi

  task_log="${BATCH_DIR}/task_${index}.log"
  : > "${task_log}"
  echo "===== Running SWE-bench index ${index} [${case_num}/${total_cases}] =====" | tee -a "${DRIVER_LOG}" "${task_log}"
  before_result="$(latest_agentbench_result_dir)"
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
    AGENTBENCH_TRACE_AGENT_STREAM="${AGENTBENCH_TRACE_AGENT_STREAM}" \
    AGENTBENCH_TRACE_AGENT_STREAM_MODE="${AGENTBENCH_TRACE_AGENT_STREAM_MODE}" \
    AGENTBENCH_WORKFLOW_MODE="${AGENTBENCH_WORKFLOW_MODE}" \
    AGENTBENCH_DIRECT_SGLANG_TOOL_RICH="${AGENTBENCH_DIRECT_SGLANG_TOOL_RICH}" \
    AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT="${AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT}" \
    AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS="${AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS}" \
    AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD="${AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD}" \
    AGENTBENCH_SGLANG_PREFETCH_MODE="exp6_direct_sglang" \
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
  ) 2>&1 | tee -a "${DRIVER_LOG}" "${task_log}"
  status="${PIPESTATUS[0]}"
  set -e

  after_result="$(latest_agentbench_result_dir)"
  run_id=""
  if [[ -n "${after_result}" && "${after_result}" != "${before_result}" ]]; then
    run_id="$(basename "${after_result}")"
    (cd "${AGENTBENCH_ROOT}" && "${PYTHON_BIN}" experiments/scripts/agentbench_report/build_run_report.py \
      --agentbench-result-dir "${after_result}" \
      --transfer-log experiments/raw/sglang_transfer_logs/latest_sglang_transfer_events.jsonl) >/dev/null 2>&1 || true
    append_task_index_row "${index}" "${run_id}" "${status}"
    append_trace_index_row "${run_id}" "${index}" || true
    append_progress_row "${run_id}" || true
    publish_prompt_evolution_reports
    if [[ "${PROMPT_EVOLUTION_REFRESH_TRAJECTORY_CATALOG_EACH_TASK}" = "1" ]]; then
      refresh_trajectory_catalog || true
    fi
    echo "Task index ${index} produced run ${run_id} with status ${status}." | tee -a "${DRIVER_LOG}"
  else
    echo "Task index ${index} did not produce a new AgentBench result directory." | tee -a "${DRIVER_LOG}"
  fi

  if [[ "${status}" -ne 0 ]]; then
    append_skipped_row "${index}" "${run_id}" "task_error" "${status}" "${task_log}"
    if [[ "${AGENTBENCH_BATCH_CONTINUE_ON_ERROR}" = "1" ]]; then
      echo "Index ${index} failed; continuing because AGENTBENCH_BATCH_CONTINUE_ON_ERROR=1." | tee -a "${DRIVER_LOG}"
      continue
    fi
    echo "Index ${index} failed; stopping." | tee -a "${DRIVER_LOG}" >&2
    exit "${status}"
  fi
done

if [[ "${PROMPT_EVOLUTION_BUILD_TRAJECTORY_CATALOG}" = "1" ]]; then
  echo "Building final SWE-bench trajectory prompt catalog from direct-SGLang Exp6 traces..." | tee -a "${DRIVER_LOG}"
  refresh_trajectory_catalog
fi

publish_prompt_evolution_reports

{
  echo
  echo "Milestone 21 finished."
  echo "Batch dir: ${BATCH_DIR}"
  echo "Driver log: ${DRIVER_LOG}"
  echo "Trace index CSV: ${TRACE_INDEX_CSV}"
  echo "Latest trace index CSV: ${LATEST_TRACE_INDEX_CSV}"
  echo "Latest trajectory catalog: ${AGENTBENCH_ROOT}/experiments/reports/latest_swebench_trajectory_prompt_catalog.csv"
  echo "Local copied catalog: ${RESULT_ROOT}/exp6_direct_swebench_trajectory_prompt_catalog.csv"
  echo "Latest copied catalog: ${LATEST_REPORT_ROOT}/latest_exp6_direct_swebench_trajectory_prompt_catalog.csv"
  echo "Task index: ${TASK_INDEX_CSV}"
} | tee -a "${DRIVER_LOG}"
