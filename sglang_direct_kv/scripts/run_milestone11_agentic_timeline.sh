#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone11_agentic_timeline}"
MODE="${MODE:-oracle_direct_load}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
AGENTIC_KV_TRACE_MAX_EXACT_INDICES="${AGENTIC_KV_TRACE_MAX_EXACT_INDICES:-512}"
AGENTIC_KV_TORCH_PROFILER_ENABLE="${AGENTIC_KV_TORCH_PROFILER_ENABLE:-1}"
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS="${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:-220}"
SESSION_COUNT="${SESSION_COUNT:-16}"
RANDOMIZE_TRAFFIC="${RANDOMIZE_TRAFFIC:-1}"
RANDOM_SEED="${RANDOM_SEED:-7}"
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS:-120}"
ARRIVAL_GAP_RANGE_MS="${ARRIVAL_GAP_RANGE_MS:-60 220}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-250 500 900 1600}"
TOOL_WAIT_RANGE_MS="${TOOL_WAIT_RANGE_MS:-250 2200}"
PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST:-768 1024 1536}"
HINT_DELAY_MS="${HINT_DELAY_MS:-120}"
ORACLE_LEAD_MS="${ORACLE_LEAD_MS:-1000}"
TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY:-8}"
TIMELINE_MAX_SESSIONS="${TIMELINE_MAX_SESSIONS:-12}"

mkdir -p artifacts "${RESULT_ROOT}"

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 11." >&2
  exit 1
fi

trace="${RESULT_ROOT}/${MODE}_traffic_trace.jsonl"
metrics="${RESULT_ROOT}/${MODE}_traffic_metrics.jsonl"
log="${RESULT_ROOT}/${MODE}_server.log"
out_dir="${RESULT_ROOT}/${MODE}_outcomes"
torch_profile_dir="${RESULT_ROOT}/${MODE}_torch_cuda_profiles"
torch_summary_json="${RESULT_ROOT}/${MODE}_torch_cuda_profile_summary.json"
torch_summary_md="${RESULT_ROOT}/${MODE}_torch_cuda_profile_summary.md"
torch_correlation_json="${RESULT_ROOT}/${MODE}_torch_cuda_trace_correlation.json"
torch_correlation_md="${RESULT_ROOT}/${MODE}_torch_cuda_trace_correlation.md"
torch_copy_timeline_csv="${RESULT_ROOT}/${MODE}_torch_cuda_copy_timeline.csv"
agentic_timeline_csv="${RESULT_ROOT}/${MODE}_agentic_prefetch_timeline.csv"
agentic_timeline_json="${RESULT_ROOT}/${MODE}_agentic_prefetch_timeline.json"
agentic_timeline_html="${RESULT_ROOT}/${MODE}_agentic_prefetch_timeline.html"

rm -f \
  "${trace}" \
  "${metrics}" \
  "${log}" \
  "${torch_summary_json}" \
  "${torch_summary_md}" \
  "${torch_correlation_json}" \
  "${torch_correlation_md}" \
  "${torch_copy_timeline_csv}" \
  "${agentic_timeline_csv}" \
  "${agentic_timeline_json}" \
  "${agentic_timeline_html}"
rm -rf "${out_dir}" "${torch_profile_dir}"

server_pid=""

cleanup_server() {
  if [[ -n "${server_pid}" ]]; then
    echo "Stopping SGLang server..."
    kill -INT "-${server_pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    kill -TERM "-${server_pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 10); do
      if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    kill -KILL "-${server_pid}" >/dev/null 2>&1 || true
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

traffic_args=(
  --base-url "${HOST_URL}/v1"
  --model "${MODEL}"
  --mode "${MODE}"
  --session-count "${SESSION_COUNT}"
  --arrival-gap-ms "${ARRIVAL_GAP_MS}"
  --tool-wait-list-ms "${TOOL_WAIT_LIST_MS}"
  --prompt-token-list "${PROMPT_TOKEN_LIST}"
  --hint-delay-ms "${HINT_DELAY_MS}"
  --oracle-lead-ms "${ORACLE_LEAD_MS}"
  --concurrency "${TRAFFIC_CONCURRENCY}"
  --out "${metrics}"
)

if [[ "${RANDOMIZE_TRAFFIC}" == "1" ]]; then
  traffic_args+=(
    --randomize-traffic
    --seed "${RANDOM_SEED}"
    --arrival-gap-range-ms "${ARRIVAL_GAP_RANGE_MS}"
    --tool-wait-range-ms "${TOOL_WAIT_RANGE_MS}"
  )
fi

echo "Milestone 11 agentic prefetch timeline"
echo "MODEL=${MODEL}"
echo "MODE=${MODE}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "SESSION_COUNT=${SESSION_COUNT}"
echo "RANDOMIZE_TRAFFIC=${RANDOMIZE_TRAFFIC}"
echo "RANDOM_SEED=${RANDOM_SEED}"
echo "ARRIVAL_GAP_MS=${ARRIVAL_GAP_MS}"
echo "ARRIVAL_GAP_RANGE_MS=${ARRIVAL_GAP_RANGE_MS}"
echo "TOOL_WAIT_LIST_MS=${TOOL_WAIT_LIST_MS}"
echo "TOOL_WAIT_RANGE_MS=${TOOL_WAIT_RANGE_MS}"
echo "PROMPT_TOKEN_LIST=${PROMPT_TOKEN_LIST}"
echo "HINT_DELAY_MS=${HINT_DELAY_MS}"
echo "ORACLE_LEAD_MS=${ORACLE_LEAD_MS}"
echo "TRAFFIC_CONCURRENCY=${TRAFFIC_CONCURRENCY}"
echo "AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS}"

export AGENTIC_KV_TRACE_ENABLE=1
export AGENTIC_KV_NVTX_ENABLE=1
export AGENTIC_KV_TRACE_PATH="${trace}"
export AGENTIC_KV_TRACE_MAX_EXACT_INDICES
export AGENTIC_KV_TORCH_PROFILER_ENABLE
export AGENTIC_KV_TORCH_PROFILER_DIR="${torch_profile_dir}"
export AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS
export HICACHE_SIZE_GB
export EXTRA_SERVER_ARGS="--max-total-tokens ${MAX_TOTAL_TOKENS}"

echo
echo "Step 1/6: starting traced SGLang server"
setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${log}" 2>&1 &
server_pid="$!"
wait_for_server

echo "Step 2/6: running randomized agent traffic"
python scripts/run_agentic_traffic_workload.py "${traffic_args[@]}"

echo "Step 3/6: summarizing SGLang KV trace"
python scripts/summarize_kv_trace.py --trace "${trace}" | head -80

echo "Step 4/6: classifying hint outcomes"
python scripts/analyze_hint_outcomes.py \
  --trace "${trace}" \
  --metrics "${metrics}" \
  --out-dir "${out_dir}"

cleanup_server

echo "Step 5/6: summarizing and correlating torch CUDA copy activity"
python scripts/summarize_torch_cuda_profiles.py \
  --profile-dir "${torch_profile_dir}" \
  --out-json "${torch_summary_json}" \
  --out-md "${torch_summary_md}"
python scripts/correlate_torch_profile_with_agent_trace.py \
  --trace "${trace}" \
  --profile-dir "${torch_profile_dir}" \
  --out-json "${torch_correlation_json}" \
  --out-md "${torch_correlation_md}" \
  --out-copy-csv "${torch_copy_timeline_csv}"

echo "Step 6/6: building agentic prefetch timeline"
python scripts/build_agentic_prefetch_timeline.py \
  --trace "${trace}" \
  --copy-csv "${torch_copy_timeline_csv}" \
  --profile-dir "${torch_profile_dir}" \
  --out-csv "${agentic_timeline_csv}" \
  --out-json "${agentic_timeline_json}" \
  --out-html "${agentic_timeline_html}" \
  --max-sessions "${TIMELINE_MAX_SESSIONS}"

echo
echo "Milestone 11 outputs written under ${RESULT_ROOT}"
echo "Trace: ${trace}"
echo "Metrics: ${metrics}"
echo "Hint outcomes: ${out_dir}/hint_outcomes.html"
echo "Torch profile summary: ${torch_summary_md}"
echo "Torch trace correlation: ${torch_correlation_md}"
echo "CUDA copy timeline CSV: ${torch_copy_timeline_csv}"
echo "Agentic timeline CSV: ${agentic_timeline_csv}"
echo "Agentic timeline HTML: ${agentic_timeline_html}"
