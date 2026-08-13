#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
LOG="${LOG:-artifacts/hicache_smoke.log}"
PROMPT="${PROMPT:-Say OK only.}"
MAX_TOKENS="${MAX_TOKENS:-4}"

mkdir -p artifacts

started_server=0
server_pid=""

cleanup() {
  if [[ "${started_server}" == "1" && -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  rm -f "${LOG}"
  bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${LOG}" 2>&1 &
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

curl -fsS "${HOST_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
    \"max_tokens\": ${MAX_TOKENS},
    \"temperature\": 0
  }" | python -m json.tool

if [[ "${started_server}" == "1" ]]; then
  echo
  echo "Log tail:"
  tail -60 "${LOG}" || true
fi
