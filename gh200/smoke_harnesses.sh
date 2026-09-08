#!/usr/bin/env bash
set -euo pipefail

# Host-side no-GPU smoke test. This is useful on GH200 before Docker GPU runs,
# especially for Python 3.11 harnesses that are not run inside the SGLang
# Python 3.12 Docker image.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIRECT_ROOT="${REPO_ROOT}/sglang_direct_kv"
cd "${DIRECT_ROOT}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing ${DIRECT_ROOT}/.venv. Run: INSTALL_SYSTEM_DEPS=0 bash scripts/setup_gh200.sh" >&2
  exit 1
fi

source .venv/bin/activate

export HARNESS_NAT_BIN="${HARNESS_NAT_BIN:-${REPO_ROOT}/.venvs/nat_py311/bin/nat}"
export HARNESS_HERMES_BIN="${HARNESS_HERMES_BIN:-${REPO_ROOT}/.venvs/hermes_agent_py311/bin/hermes}"

if [[ $# -gt 0 ]]; then
  harnesses=("$@")
else
  harnesses=(codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent)
fi

python scripts/smoke_multi_harness_wireability.py \
  --harnesses "${harnesses[@]}"
