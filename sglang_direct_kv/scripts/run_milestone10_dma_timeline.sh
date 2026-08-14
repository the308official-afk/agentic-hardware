#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST_URL="${HOST_URL:-http://127.0.0.1:30000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
RESULT_ROOT="${RESULT_ROOT:-artifacts/results/milestone10_dma_timeline}"
MODE="${MODE:-oracle_direct_load}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-8}"
ENABLE_NSYS="${ENABLE_NSYS:-1}"
AGENTIC_KV_TORCH_PROFILER_ENABLE="${AGENTIC_KV_TORCH_PROFILER_ENABLE:-0}"
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS="${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS:-0}"
AGENTIC_KV_TRACE_MAX_EXACT_INDICES="${AGENTIC_KV_TRACE_MAX_EXACT_INDICES:-512}"
SESSION_COUNT="${SESSION_COUNT:-12}"
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS:-120}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-250 500 900 1600}"
PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST:-768 1024 1536}"
HINT_DELAY_MS="${HINT_DELAY_MS:-120}"
ORACLE_LEAD_MS="${ORACLE_LEAD_MS:-1500}"
TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY:-8}"
NSYS_BIN="${NSYS_BIN:-nsys}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
NSYS_WAIT="${NSYS_WAIT:-all}"
NSYS_CUDA_MEMORY_USAGE="${NSYS_CUDA_MEMORY_USAGE:-true}"
NSYS_CUDA_TRACE_SCOPE="${NSYS_CUDA_TRACE_SCOPE:-system-wide}"
NSYS_MONITOR_DURATION_SEC="${NSYS_MONITOR_DURATION_SEC:-240}"
NSYS_MONITOR_WARMUP_SEC="${NSYS_MONITOR_WARMUP_SEC:-3}"
NSYS_PROFILE_SHAPE="${NSYS_PROFILE_SHAPE:-monitor}"
NSYS_USE_SUDO="${NSYS_USE_SUDO:-0}"
NSYS_RUN_AS_USER="${NSYS_RUN_AS_USER:-$(id -un)}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"

mkdir -p artifacts "${RESULT_ROOT}"

if [[ "${ENABLE_NSYS}" == "1" ]] && ! command -v "${NSYS_BIN}" >/dev/null 2>&1; then
  cat >&2 <<EOF
Nsight Systems CLI was not found: ${NSYS_BIN}

Install Nsight Systems or set NSYS_BIN to the full path.
On many CUDA images this is available as:
  /usr/local/cuda/bin/nsys
EOF
  exit 1
fi

if curl -fsS "${HOST_URL}/model_info" >/dev/null 2>&1; then
  echo "A server is already listening at ${HOST_URL}. Stop it before running Milestone 10." >&2
  exit 1
fi

trace="${RESULT_ROOT}/${MODE}_traffic_trace.jsonl"
metrics="${RESULT_ROOT}/${MODE}_traffic_metrics.jsonl"
log="${RESULT_ROOT}/${MODE}_server_nsys.log"
nsys_log="${RESULT_ROOT}/${MODE}_nsys_monitor.log"
out_dir="${RESULT_ROOT}/${MODE}_outcomes"
nsys_prefix="${RESULT_ROOT}/${MODE}_server"
sqlite="${RESULT_ROOT}/${MODE}_server.sqlite"
nsys_summary_json="${RESULT_ROOT}/${MODE}_dma_timeline_summary.json"
nsys_summary_md="${RESULT_ROOT}/${MODE}_dma_timeline_summary.md"
torch_profile_dir="${RESULT_ROOT}/${MODE}_torch_cuda_profiles"
torch_summary_json="${RESULT_ROOT}/${MODE}_torch_cuda_profile_summary.json"
torch_summary_md="${RESULT_ROOT}/${MODE}_torch_cuda_profile_summary.md"
torch_correlation_json="${RESULT_ROOT}/${MODE}_torch_cuda_trace_correlation.json"
torch_correlation_md="${RESULT_ROOT}/${MODE}_torch_cuda_trace_correlation.md"
torch_copy_timeline_csv="${RESULT_ROOT}/${MODE}_torch_cuda_copy_timeline.csv"

rm -f "${trace}" "${metrics}" "${log}" "${nsys_log}" "${sqlite}" "${nsys_summary_json}" "${nsys_summary_md}"
rm -f "${torch_summary_json}" "${torch_summary_md}" "${torch_correlation_json}" "${torch_correlation_md}" "${torch_copy_timeline_csv}"
rm -f "${nsys_prefix}.nsys-rep" "${nsys_prefix}.qdrep"
rm -rf "${out_dir}" "${torch_profile_dir}"

server_pid=""
nsys_pid=""

cleanup_server() {
  if [[ -n "${server_pid}" ]]; then
    echo "Stopping profiled SGLang server..."
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

cleanup_nsys() {
  if [[ -n "${nsys_pid}" ]]; then
    echo "Stopping Nsight monitor..."
    kill -INT "-${nsys_pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${nsys_pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    kill -TERM "-${nsys_pid}" >/dev/null 2>&1 || true
    wait "${nsys_pid}" >/dev/null 2>&1 || true
    nsys_pid=""
  fi
}
trap 'cleanup_server; cleanup_nsys' EXIT

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

echo "Milestone 10 DMA timeline profiling"
echo "MODE=${MODE}"
echo "HICACHE_SIZE_GB=${HICACHE_SIZE_GB}"
echo "ENABLE_NSYS=${ENABLE_NSYS}"
echo "AGENTIC_KV_TORCH_PROFILER_ENABLE=${AGENTIC_KV_TORCH_PROFILER_ENABLE}"
echo "AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS}"
echo "AGENTIC_KV_TRACE_MAX_EXACT_INDICES=${AGENTIC_KV_TRACE_MAX_EXACT_INDICES}"
echo "SESSION_COUNT=${SESSION_COUNT}"
echo "ARRIVAL_GAP_MS=${ARRIVAL_GAP_MS}"
echo "TOOL_WAIT_LIST_MS=${TOOL_WAIT_LIST_MS}"
echo "PROMPT_TOKEN_LIST=${PROMPT_TOKEN_LIST}"
echo "HINT_DELAY_MS=${HINT_DELAY_MS}"
echo "ORACLE_LEAD_MS=${ORACLE_LEAD_MS}"
echo "NSYS_TRACE=${NSYS_TRACE}"
echo "NSYS_WAIT=${NSYS_WAIT}"
echo "NSYS_CUDA_MEMORY_USAGE=${NSYS_CUDA_MEMORY_USAGE}"
echo "NSYS_CUDA_TRACE_SCOPE=${NSYS_CUDA_TRACE_SCOPE}"
echo "NSYS_MONITOR_DURATION_SEC=${NSYS_MONITOR_DURATION_SEC}"
echo "NSYS_PROFILE_SHAPE=${NSYS_PROFILE_SHAPE}"
echo "NSYS_USE_SUDO=${NSYS_USE_SUDO}"

export AGENTIC_KV_TRACE_ENABLE=1
export AGENTIC_KV_NVTX_ENABLE=1
export AGENTIC_KV_TRACE_PATH="${trace}"
export AGENTIC_KV_TRACE_MAX_EXACT_INDICES
export AGENTIC_KV_TORCH_PROFILER_ENABLE
export AGENTIC_KV_TORCH_PROFILER_DIR="${torch_profile_dir}"
export AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS
export HICACHE_SIZE_GB
export EXTRA_SERVER_ARGS="--max-total-tokens ${MAX_TOTAL_TOKENS}"

if [[ "${ENABLE_NSYS}" != "1" ]]; then
  setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${log}" 2>&1 &
  server_pid="$!"
else
  nsys_args=(
    profile
    --force-overwrite=true
    --output="${nsys_prefix}"
    --trace="${NSYS_TRACE}"
    --sample="${NSYS_SAMPLE}"
    --wait="${NSYS_WAIT}"
    --cuda-memory-usage="${NSYS_CUDA_MEMORY_USAGE}"
    --cuda-trace-scope="${NSYS_CUDA_TRACE_SCOPE}"
    --trace-fork-before-exec=true
  )
  if [[ "${NSYS_USE_SUDO}" == "1" ]]; then
    nsys_args=(
      env
      "PATH=${PATH}"
      "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
      "HOME=${HOME}"
      "PYTHONPATH=$(pwd)/src:${PYTHONPATH:-}"
      "AGENTIC_KV_TRACE_ENABLE=${AGENTIC_KV_TRACE_ENABLE}"
      "AGENTIC_KV_NVTX_ENABLE=${AGENTIC_KV_NVTX_ENABLE}"
      "AGENTIC_KV_TRACE_PATH=${AGENTIC_KV_TRACE_PATH}"
      "AGENTIC_KV_TRACE_MAX_EXACT_INDICES=${AGENTIC_KV_TRACE_MAX_EXACT_INDICES}"
      "AGENTIC_KV_TORCH_PROFILER_ENABLE=${AGENTIC_KV_TORCH_PROFILER_ENABLE}"
      "AGENTIC_KV_TORCH_PROFILER_DIR=${AGENTIC_KV_TORCH_PROFILER_DIR}"
      "AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=${AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS}"
      "HICACHE_SIZE_GB=${HICACHE_SIZE_GB}"
      "EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS}"
      "${NSYS_BIN}"
      "${nsys_args[@]}"
      --run-as
      "${NSYS_RUN_AS_USER}"
    )
    nsys_cmd=(sudo "${nsys_args[@]}")
  else
    nsys_cmd=("${NSYS_BIN}" "${nsys_args[@]}")
  fi

  if [[ "${NSYS_PROFILE_SHAPE}" == "monitor" ]]; then
    setsid "${nsys_cmd[@]}" \
      ${NSYS_EXTRA_ARGS} \
      sleep "${NSYS_MONITOR_DURATION_SEC}" >"${nsys_log}" 2>&1 &
    nsys_pid="$!"

    sleep "${NSYS_MONITOR_WARMUP_SEC}"

    setsid bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${log}" 2>&1 &
    server_pid="$!"
  elif [[ "${NSYS_PROFILE_SHAPE}" == "launch" ]]; then
    setsid "${nsys_cmd[@]}" \
      ${NSYS_EXTRA_ARGS} \
      bash scripts/run_sglang_hicache_server.sh "${MODEL}" >"${log}" 2>&1 &
    server_pid="$!"
  else
    echo "Unknown NSYS_PROFILE_SHAPE=${NSYS_PROFILE_SHAPE}; use monitor or launch." >&2
    exit 1
  fi
fi

wait_for_server

python scripts/run_agentic_traffic_workload.py \
  --base-url "${HOST_URL}/v1" \
  --model "${MODEL}" \
  --mode "${MODE}" \
  --session-count "${SESSION_COUNT}" \
  --arrival-gap-ms "${ARRIVAL_GAP_MS}" \
  --tool-wait-list-ms "${TOOL_WAIT_LIST_MS}" \
  --prompt-token-list "${PROMPT_TOKEN_LIST}" \
  --hint-delay-ms "${HINT_DELAY_MS}" \
  --oracle-lead-ms "${ORACLE_LEAD_MS}" \
  --concurrency "${TRAFFIC_CONCURRENCY}" \
  --out "${metrics}"

python scripts/summarize_kv_trace.py --trace "${trace}" | head -60
python scripts/analyze_hint_outcomes.py \
  --trace "${trace}" \
  --metrics "${metrics}" \
  --out-dir "${out_dir}"

cleanup_server
if [[ "${ENABLE_NSYS}" == "1" && "${NSYS_PROFILE_SHAPE}" == "monitor" ]]; then
  cleanup_nsys
fi

if [[ "${AGENTIC_KV_TORCH_PROFILER_ENABLE}" == "1" ]]; then
  sleep 2
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
fi

if [[ "${ENABLE_NSYS}" == "1" ]]; then
  report="${nsys_prefix}.nsys-rep"
  if [[ ! -f "${report}" && -f "${nsys_prefix}.qdrep" ]]; then
    report="${nsys_prefix}.qdrep"
  fi

  if [[ ! -f "${report}" ]]; then
    echo "Nsight report was not generated. Log tail:"
    tail -160 "${nsys_log}" || true
    exit 1
  fi

  "${NSYS_BIN}" export \
    --type sqlite \
    --force-overwrite=true \
    --output "${sqlite}" \
    "${report}"

  python scripts/summarize_nsys_dma_timeline.py \
    --sqlite "${sqlite}" \
    --trace "${trace}" \
    --out-json "${nsys_summary_json}" \
    --out-md "${nsys_summary_md}"
fi

echo
echo "Milestone 10 outputs written under ${RESULT_ROOT}"
if [[ "${ENABLE_NSYS}" == "1" ]]; then
  echo "Nsight report: ${report}"
  echo "SQLite export: ${sqlite}"
  echo "DMA summary: ${nsys_summary_md}"
  echo "Nsight monitor log: ${nsys_log}"
fi
if [[ "${AGENTIC_KV_TORCH_PROFILER_ENABLE}" == "1" ]]; then
  echo "Torch CUDA profiles: ${torch_profile_dir}"
  echo "Torch CUDA profile summary: ${torch_summary_md}"
  echo "Torch CUDA trace correlation: ${torch_correlation_md}"
  echo "Torch CUDA copy timeline CSV: ${torch_copy_timeline_csv}"
fi
