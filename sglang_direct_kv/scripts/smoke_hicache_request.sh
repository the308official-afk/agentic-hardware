#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
LOG="${LOG:-artifacts/hicache_smoke.log}"
TRACE="${AGENTIC_KV_TRACE_PATH:-artifacts/kv_movement_trace.jsonl}"
PROMPT="${PROMPT:-Say OK only.}"
MAX_TOKENS="${MAX_TOKENS:-4}"
REQUEST_COUNT="${REQUEST_COUNT:-1}"

mkdir -p artifacts

started_server=0
server_pid=""

cleanup() {
  if [[ "${started_server}" == "1" && -n "${server_pid}" ]]; then
    kill "-${server_pid}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "-${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  rm -f "${LOG}"
  rm -f "${TRACE}"
  export AGENTIC_KV_TRACE_ENABLE="${AGENTIC_KV_TRACE_ENABLE:-1}"
  export AGENTIC_KV_TRACE_PATH="${TRACE}"
  setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${LOG}" 2>&1 &
  server_pid="$!"
  started_server=1

  ready=0
  for _ in $(seq 1 300); do
    if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "SGLang exited before becoming ready. Log tail:"
      tail -100 "${LOG}" || true
      exit 1
    fi
    sleep 1
  done

  if [[ "${ready}" != "1" ]]; then
    echo "SGLang did not become ready. Log tail:"
    tail -100 "${LOG}" || true
    exit 1
  fi
fi

echo "SGLang is ready at ${HOST_URL}"

for request_id in $(seq 1 "${REQUEST_COUNT}"); do
  echo
  echo "Request ${request_id}/${REQUEST_COUNT}"
  curl -fsS "${HOST_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
      \"max_tokens\": ${MAX_TOKENS},
      \"temperature\": 0
    }" | python -m json.tool
done

if [[ "${started_server}" == "1" ]]; then
  echo
  python scripts/summarize_kv_trace.py --trace "${TRACE}" || true

  echo
  echo "Log tail:"
  tail -60 "${LOG}" || true
else
  echo
  echo "Server was already running. Trace summary is only available if that server was started with AGENTIC_KV_TRACE_ENABLE=1."
fi
