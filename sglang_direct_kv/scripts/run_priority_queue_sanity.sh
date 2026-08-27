#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
RESULT_LABEL="${RESULT_LABEL:-priority_queue_sanity_v0511_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/runs/priority_queue_sanity/${RESULT_LABEL}}"
REPORT_ROOT="${REPORT_ROOT:-artifacts/results/reports/${RESULT_LABEL}/priority_queue_sanity}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-local/dynamo-sglang:runtime-json-logs-ec2}"
SGLANG_DOCKER_PULL="${SGLANG_DOCKER_PULL:-0}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOW_BEFORE_COUNT="${LOW_BEFORE_COUNT:-24}"
LOW_AFTER_COUNT="${LOW_AFTER_COUNT:-8}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-64}"
HIGH_SUBMIT_DELAY_MS="${HIGH_SUBMIT_DELAY_MS:-100}"
LOW_SUBMIT_STAGGER_MS="${LOW_SUBMIT_STAGGER_MS:-5}"
LOW_PROMPT_TOKENS="${LOW_PROMPT_TOKENS:-2048}"
HIGH_PROMPT_TOKENS="${HIGH_PROMPT_TOKENS:-512}"
LOW_MAX_TOKENS="${LOW_MAX_TOKENS:-96}"
HIGH_MAX_TOKENS="${HIGH_MAX_TOKENS:-24}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-12288}"
export HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-14}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
export DYNAMO_HIGH_PRIORITY="${DYNAMO_HIGH_PRIORITY:-100}"
export DYNAMO_NORMAL_PRIORITY="${DYNAMO_NORMAL_PRIORITY:-0}"
export DYNAMO_LOW_PRIORITY="${DYNAMO_LOW_PRIORITY:--100}"
export DYNAMO_SCHEDULE_POLICY="${DYNAMO_SCHEDULE_POLICY:-fcfs}"
export DYNAMO_RADIX_EVICTION_POLICY="${DYNAMO_RADIX_EVICTION_POLICY:-priority}"
export AGENTIC_KV_TRACE_SCHEDULER="${AGENTIC_KV_TRACE_SCHEDULER:-1}"
export AGENTIC_KV_TRACE_KV_POOL="${AGENTIC_KV_TRACE_KV_POOL:-0}"
export AGENTIC_KV_PRIORITY_QUEUE_SNAPSHOT_LIMIT="${AGENTIC_KV_PRIORITY_QUEUE_SNAPSHOT_LIMIT:-256}"
export AGENTIC_KV_PRIORITY_QUEUE_HEAD_SAMPLE="${AGENTIC_KV_PRIORITY_QUEUE_HEAD_SAMPLE:-32}"
export AGENTIC_KV_PRIORITY_QUEUE_MAX_SNAPSHOTS="${AGENTIC_KV_PRIORITY_QUEUE_MAX_SNAPSHOTS:-96}"

trace="${RESULT_ROOT}/priority_queue_sanity_trace.jsonl"
metrics="${RESULT_ROOT}/priority_queue_sanity_metrics.jsonl"
server_log="${RESULT_ROOT}/server.log"
mkdir -p "${RESULT_ROOT}" "${REPORT_ROOT}"
rm -f "${trace}" "${metrics}" "${server_log}"

cleanup_server() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_server EXIT

wait_for_server() {
  local deadline=$((SECONDS + 900))
  until curl -fsS "${HOST_URL}/v1/models" >/dev/null 2>&1 || curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; do
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "SGLang server exited early. Last server log lines:" >&2
      tail -120 "${server_log}" >&2 || true
      exit 1
    fi
    if (( SECONDS > deadline )); then
      echo "Timed out waiting for SGLang server. Last server log lines:" >&2
      tail -120 "${server_log}" >&2 || true
      exit 1
    fi
    sleep 2
  done
}

echo "Priority Queue Sanity"
echo "MODEL=${MODEL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "SGLANG_DOCKER_IMAGE=${SGLANG_DOCKER_IMAGE}"
echo "LOW_BEFORE_COUNT=${LOW_BEFORE_COUNT}"
echo "LOW_AFTER_COUNT=${LOW_AFTER_COUNT}"
echo "REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY}"
echo "DYNAMO_SCHEDULE_POLICY=${DYNAMO_SCHEDULE_POLICY}"
echo "DYNAMO_RADIX_EVICTION_POLICY=${DYNAMO_RADIX_EVICTION_POLICY}"

export SGLANG_DOCKER_IMAGE
export SGLANG_DOCKER_PULL
export AGENTIC_KV_TRACE_ENABLE=1
export AGENTIC_KV_TRACE_PATH="${trace}"
export HICACHE_SIZE_GB
export MEM_FRACTION_STATIC
export EXTRA_SERVER_ARGS="--max-total-tokens ${MAX_TOTAL_TOKENS} --enable-cache-report --enable-priority-scheduling --default-priority-value ${DYNAMO_NORMAL_PRIORITY}"
if [[ -n "${DYNAMO_SCHEDULE_POLICY}" ]]; then
  export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS} --schedule-policy ${DYNAMO_SCHEDULE_POLICY}"
fi
if [[ -n "${DYNAMO_RADIX_EVICTION_POLICY}" ]]; then
  export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS} --radix-eviction-policy ${DYNAMO_RADIX_EVICTION_POLICY}"
fi

setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${server_log}" 2>&1 &
server_pid="$!"
wait_for_server

"${PYTHON_BIN}" scripts/run_priority_queue_jump_workload.py \
  --base-url "${HOST_URL}/v1" \
  --model "${MODEL}" \
  --out "${metrics}" \
  --trace-out "${trace}" \
  --low-before-count "${LOW_BEFORE_COUNT}" \
  --low-after-count "${LOW_AFTER_COUNT}" \
  --request-concurrency "${REQUEST_CONCURRENCY}" \
  --high-submit-delay-ms "${HIGH_SUBMIT_DELAY_MS}" \
  --low-submit-stagger-ms "${LOW_SUBMIT_STAGGER_MS}" \
  --low-prompt-tokens "${LOW_PROMPT_TOKENS}" \
  --high-prompt-tokens "${HIGH_PROMPT_TOKENS}" \
  --low-max-tokens "${LOW_MAX_TOKENS}" \
  --high-max-tokens "${HIGH_MAX_TOKENS}" \
  --high-priority "${DYNAMO_HIGH_PRIORITY}" \
  --low-priority "${DYNAMO_LOW_PRIORITY}"

cleanup_server
server_pid=""

"${PYTHON_BIN}" scripts/summarize_priority_queue_sanity.py \
  --trace "${trace}" \
  --metrics "${metrics}" \
  --out-dir "${REPORT_ROOT}" \
  --high-priority "${DYNAMO_HIGH_PRIORITY}"

echo
echo "Priority queue audit:"
echo "  ${REPORT_ROOT}/priority_queue_sanity_summary.csv"
echo "  ${REPORT_ROOT}/priority_queue_sanity_report.md"
echo "Raw traces:"
echo "  ${trace}"
echo "  ${metrics}"
