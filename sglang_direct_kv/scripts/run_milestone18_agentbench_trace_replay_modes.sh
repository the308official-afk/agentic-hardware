#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone18_agentbench_trace_replay_modes}"
LATEST_REPORT_ROOT="${LATEST_REPORT_ROOT:-artifacts/results}"
WORKLOAD_JSONL="${WORKLOAD_JSONL:-${LATEST_REPORT_ROOT}/latest_real/agentbench_replay_workload.jsonl}"
MODES="${MODES:-no_prefetch direct_load oracle_direct_load}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
HINT_DELAY_MS="${HINT_DELAY_MS:-120}"
ORACLE_LEAD_MS="${ORACLE_LEAD_MS:-500}"
TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-8}"
PREFETCH_MAX_TOKENS="${PREFETCH_MAX_TOKENS:-1}"
BASE_EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

mkdir -p "${RESULT_ROOT}" "${LATEST_REPORT_ROOT}" "${LATEST_REPORT_ROOT}/latest_real"

if [[ ! -f "${WORKLOAD_JSONL}" ]]; then
  echo "Replay workload not found: ${WORKLOAD_JSONL}" >&2
  echo "Run Milestone 16 first, or set WORKLOAD_JSONL=/path/to/agentbench_replay_workload.jsonl." >&2
  exit 1
fi

server_pid=""
case_idx=0

count_words() {
  local count=0
  local item
  for item in $1; do
    count=$((count + 1))
  done
  echo "${count}"
}

mode_count="$(count_words "${MODES}")"

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
  local log="$1"
  local ready=0
  for _ in $(seq 1 300); do
    if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "SGLang exited before becoming ready. Log tail:"
      tail -120 "${log}" || true
      exit 1
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "SGLang did not become ready. Log tail:"
    tail -120 "${log}" || true
    exit 1
  fi
}

run_mode() {
  local mode="$1"
  case_idx=$((case_idx + 1))
  local trace="${RESULT_ROOT}/${mode}_traffic_trace.jsonl"
  local copy_telemetry="${RESULT_ROOT}/${mode}_kv_copy_telemetry.jsonl"
  local metrics="${RESULT_ROOT}/${mode}_traffic_metrics.jsonl"
  local log="${RESULT_ROOT}/${mode}_server.log"
  local out_dir="${RESULT_ROOT}/${mode}_outcomes"

  echo
  echo "==== Milestone 18 AgentBench replay case [${case_idx}/${mode_count}]: ${mode} ===="
  echo "WORKLOAD_JSONL=${WORKLOAD_JSONL}"
  echo "HINT_DELAY_MS=${HINT_DELAY_MS}"
  echo "ORACLE_LEAD_MS=${ORACLE_LEAD_MS}"
  echo "TRAFFIC_CONCURRENCY=${TRAFFIC_CONCURRENCY}"

  rm -f "${trace}" "${copy_telemetry}" "${metrics}" "${log}"
  rm -rf "${out_dir}"

  export AGENTIC_KV_TRACE_ENABLE=1
  export AGENTIC_KV_TRACE_PATH="${trace}"
  export AGENTIC_KV_COPY_TELEMETRY_ENABLE=1
  export AGENTIC_KV_COPY_TELEMETRY_PATH="${copy_telemetry}"
  export EXTRA_SERVER_ARGS="${BASE_EXTRA_SERVER_ARGS} --max-total-tokens ${MAX_TOTAL_TOKENS}"

  setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${log}" 2>&1 &
  server_pid="$!"
  wait_for_server "${log}"

  python scripts/run_agentic_traffic_workload.py \
    --base-url "${HOST_URL}/v1" \
    --model "${MODEL}" \
    --mode "${mode}" \
    --workload-jsonl "${WORKLOAD_JSONL}" \
    --hint-delay-ms "${HINT_DELAY_MS}" \
    --oracle-lead-ms "${ORACLE_LEAD_MS}" \
    --concurrency "${TRAFFIC_CONCURRENCY}" \
    --max-tokens "${MAX_TOKENS}" \
    --prefetch-max-tokens "${PREFETCH_MAX_TOKENS}" \
    --out "${metrics}"

  python scripts/summarize_kv_trace.py --trace "${trace}" | head -45
  python scripts/analyze_hint_outcomes.py \
    --trace "${trace}" \
    --metrics "${metrics}" \
    --out-dir "${out_dir}"

  cleanup_server
  echo "==== Completed Milestone 18 AgentBench replay case [${case_idx}/${mode_count}]: ${mode} ===="
}

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 18." >&2
  exit 1
fi

echo "Milestone 18: real AgentBench prompt replay across prefetch modes"
echo "Total cases: ${mode_count}"
echo "Modes: ${MODES}"
echo "Each mode starts a fresh SGLang server."

for mode in ${MODES}; do
  run_mode "${mode}"
done

echo
python scripts/summarize_agentic_traffic_results.py \
  --root "${RESULT_ROOT}" \
  --modes "${MODES}"

cp -f "${WORKLOAD_JSONL}" "${RESULT_ROOT}/agentbench_replay_workload_used.jsonl"
if [[ -f "${RESULT_ROOT}/traffic_summary.csv" ]]; then
  cp -f "${RESULT_ROOT}/traffic_summary.csv" "${LATEST_REPORT_ROOT}/latest_real/agentbench_replay_mode_summary.csv"
fi
if [[ -f "${RESULT_ROOT}/traffic_summary.html" ]]; then
  cp -f "${RESULT_ROOT}/traffic_summary.html" "${LATEST_REPORT_ROOT}/latest_real/agentbench_replay_mode_summary.html"
fi
if [[ -f "${RESULT_ROOT}/traffic_summary.md" ]]; then
  cp -f "${RESULT_ROOT}/traffic_summary.md" "${LATEST_REPORT_ROOT}/latest_real/agentbench_replay_mode_summary.md"
fi

echo
echo "Milestone 18 outputs written under ${RESULT_ROOT}"
echo "Replay workload used: ${RESULT_ROOT}/agentbench_replay_workload_used.jsonl"
