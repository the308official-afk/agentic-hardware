#!/usr/bin/env bash
set -euo pipefail

# Run the consolidated harness signal design-space experiment on GH200 through
# the SGLang Docker image. Run this on the GH200 host from the repo checkout.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SGLANG_DOCKER_IMAGE="${SGLANG_DOCKER_IMAGE:-lmsysorg/sglang:latest}"
GH200_USER_NAME="${AGENTIC_GH200_USER:-$(id -un)}"
GH200_ABS_MOUNT="${AGENTIC_GH200_ABS_MOUNT:-/home/central/${GH200_USER_NAME}/agentic_hardware}"
GH200_MODEL_CACHE="${AGENTIC_GH200_MODEL_CACHE:-${HOME}/dynamo_model_cache}"
GH200_TMP_HOME="${AGENTIC_GH200_TMP_HOME:-/tmp/gh200home}"
GH200_PASSWD_FILE="${AGENTIC_GH200_PASSWD_FILE:-/tmp/gh200_passwd}"

HARDWARE_PROFILE="${HARDWARE_PROFILE:-ec2_a10g}"
SIGNAL_FAMILIES="${SIGNAL_FAMILIES:-baseline harness_emitted frontend_supplied gateway_injected}"
HARNESSES="${HARNESSES:-hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw}"
PRESSURE_LEVELS="${PRESSURE_LEVELS:-p0_control p3_high p5_boss_queue}"
REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE:-lightweight}"
REPORT_LABEL="${REPORT_LABEL:-gh200_signal_design_space_$(date +%Y%m%d_%H%M%S)}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph --disable-overlap-schedule}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker on GH200 before GPU runs." >&2
  exit 1
fi

mkdir -p \
  "${REPO_ROOT}/sglang_direct_kv/artifacts/results" \
  "${REPO_ROOT}/sglang_direct_kv/src/agentic_kv.egg-info" \
  "${GH200_TMP_HOME}"

echo "${GH200_USER_NAME}:x:$(id -u):$(id -g)::${GH200_TMP_HOME}:/bin/bash" > "${GH200_PASSWD_FILE}"

if [[ ! -d "${GH200_MODEL_CACHE}" ]]; then
  echo "Model cache not found: ${GH200_MODEL_CACHE}" >&2
  echo "Set AGENTIC_GH200_MODEL_CACHE to the HuggingFace/model cache path." >&2
  exit 1
fi

nvm_mount_args=()
if [[ -d "${HOME}/.nvm" ]]; then
  nvm_mount_args=(-v "${HOME}/.nvm:${GH200_TMP_HOME}/.nvm")
else
  echo "Warning: ${HOME}/.nvm not found. Node-based harnesses may fail until nvm/Node LTS is installed." >&2
fi

echo "GH200 Docker Signal Design Space"
echo "MODEL=${MODEL}"
echo "IMAGE=${SGLANG_DOCKER_IMAGE}"
echo "REPORT_LABEL=${REPORT_LABEL}"
echo "HARDWARE_PROFILE=${HARDWARE_PROFILE}"
echo "SIGNAL_FAMILIES=${SIGNAL_FAMILIES}"
echo "HARNESSES=${HARNESSES}"
echo "PRESSURE_LEVELS=${PRESSURE_LEVELS}"
echo "EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS}"

docker run --rm --gpus all \
  -u "$(id -u):$(id -g)" \
  -v "${REPO_ROOT}:/workspace/agentic_hardware" \
  -v "${REPO_ROOT}:${GH200_ABS_MOUNT}" \
  -v "${GH200_MODEL_CACHE}:/tmp/hfcache" \
  "${nvm_mount_args[@]}" \
  -v "${GH200_PASSWD_FILE}:/etc/passwd:ro" \
  -v "${GH200_TMP_HOME}:${GH200_TMP_HOME}" \
  -e HF_HOME=/tmp/hfcache \
  -e NVM_DIR="${GH200_TMP_HOME}/.nvm" \
  -e HOME="${GH200_TMP_HOME}" \
  -e TORCHINDUCTOR_CACHE_DIR="${GH200_TMP_HOME}/torchinductor" \
  -e EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS}" \
  -e HARDWARE_PROFILE="${HARDWARE_PROFILE}" \
  -e SIGNAL_FAMILIES="${SIGNAL_FAMILIES}" \
  -e HARNESSES="${HARNESSES}" \
  -e PRESSURE_LEVELS="${PRESSURE_LEVELS}" \
  -e REPORT_BUILDER_MODE="${REPORT_BUILDER_MODE}" \
  -e REPORT_LABEL="${REPORT_LABEL}" \
  -e HARNESS_NAT_BIN="/workspace/agentic_hardware/.venvs/nat_py311/bin/nat" \
  -e HARNESS_HERMES_BIN="/workspace/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes" \
  "${SGLANG_DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    source '${GH200_TMP_HOME}/.nvm/nvm.sh' 2>/dev/null || true
    cd /workspace/agentic_hardware/sglang_direct_kv
    mkdir -p artifacts/results/run_logs
    bash scripts/run_harness_signal_design_space.sh '${MODEL}' 2>&1 | tee \"artifacts/results/run_logs/${REPORT_LABEL}.log\"
  "
