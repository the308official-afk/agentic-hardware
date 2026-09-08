#!/usr/bin/env bash
set -euo pipefail

# Run harnesses/gateway on the GH200 host while launching only the SGLang GPU
# backend in Docker for each case. This keeps NAT/Hermes on their Python 3.11
# host venvs and still uses the SGLang container for CUDA/GPU compatibility.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
cd "${DIRECT_ROOT}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-lmsysorg/sglang:latest}"
GH200_USER_NAME="${AGENTIC_GH200_USER:-$(id -un)}"
GH200_MODEL_CACHE="${AGENTIC_GH200_MODEL_CACHE:-${HOME}/dynamo_model_cache}"
GH200_TMP_HOME="${AGENTIC_GH200_TMP_HOME:-/tmp/gh200home}"
GH200_PASSWD_FILE="${AGENTIC_GH200_PASSWD_FILE:-/tmp/gh200_passwd}"

HARDWARE_PROFILE="${HARDWARE_PROFILE:-ec2_a10g}"
SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline harness_emitted frontend_supplied gateway_injected}"
HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent}"
PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p3_high p5_boss_queue}"
REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
REPORT_LABEL="${REPORT_LABEL:-gh200_host_signal_design_space_$(date +%Y%m%d_%H%M%S)}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-overlap-schedule}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" != "1" && ! -f ".venv/bin/activate" ]]; then
  echo "Missing ${DIRECT_ROOT}/.venv." >&2
  echo "Run on GH200 first: INSTALL_SYSTEM_DEPS=0 bash sglang_direct_kv/scripts/setup_gh200.sh" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" ]] && ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker on GH200 before GPU runs." >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" && ! -d "${GH200_MODEL_CACHE}" ]]; then
  echo "Model cache not found: ${GH200_MODEL_CACHE}" >&2
  echo "Set AGENTIC_GH200_MODEL_CACHE to the HuggingFace/model cache path." >&2
  exit 1
fi

mkdir -p \
  "${DIRECT_ROOT}/artifacts/results/run_logs" \
  "${DIRECT_ROOT}/src/agentic_kv.egg-info" \
  "${GH200_TMP_HOME}"

echo "${GH200_USER_NAME}:x:$(id -u):$(id -g)::${GH200_TMP_HOME}:/bin/bash" > "${GH200_PASSWD_FILE}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
export NVM_DIR="${NVM_DIR:-${HOME}/.nvm}"
if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
  # shellcheck source=/dev/null
  source "${NVM_DIR}/nvm.sh"
fi

export HARNESS_NAT_BIN="${HARNESS_NAT_BIN:-${REPO_ROOT}/.venvs/nat_py311/bin/nat}"
export HARNESS_HERMES_BIN="${HARNESS_HERMES_BIN:-${REPO_ROOT}/.venvs/hermes_agent_py311/bin/hermes}"
export SGLANG_DOCKER_IMAGE
export SGLANG_DOCKER_PULL="${SGLANG_DOCKER_PULL:-0}"
export SGLANG_DOCKER_GPU_ARGS="${SGLANG_DOCKER_GPU_ARGS:---gpus all}"
export EXTRA_SERVER_ARGS

docker_extra_args=(
  -u "$(id -u):$(id -g)"
  -v "${GH200_MODEL_CACHE}:/tmp/hfcache"
  -v "${GH200_PASSWD_FILE}:/etc/passwd:ro"
  -v "${GH200_TMP_HOME}:${GH200_TMP_HOME}"
  -e "HF_HOME=/tmp/hfcache"
  -e "HOME=${GH200_TMP_HOME}"
  -e "TORCHINDUCTOR_CACHE_DIR=${GH200_TMP_HOME}/torchinductor"
)
export SGLANG_DOCKER_EXTRA_ARGS="$(printf '%q ' "${docker_extra_args[@]}") ${SGLANG_DOCKER_EXTRA_ARGS:-}"

echo "GH200 Host-Harness Signal Design Space"
echo "MODEL=${MODEL}"
echo "SGLang Docker image=${SGLANG_DOCKER_IMAGE}"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "HARDWARE_PROFILE=${HARDWARE_PROFILE}"
echo "SIGNAL_FAMILIES=${SIGNAL_FAMILIES}"
echo "HARNESSES=${HARNESSES}"
echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"
echo "EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS}"
echo "HARNESS_NAT_BIN=${HARNESS_NAT_BIN}"
echo "HARNESS_HERMES_BIN=${HARNESS_HERMES_BIN}"

export HARDWARE_PROFILE
export SIGNAL_FAMILIES
export HARNESSES
export PRESSURE_LEVELS
export REPORT_BUILDER_MODE
export REPORT_LABEL
export DRY_RUN

bash scripts/run_harness_signal_design_space.sh "${MODEL}" \
  2>&1 | tee "artifacts/results/run_logs/${REPORT_LABEL}.log"
