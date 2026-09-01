#!/usr/bin/env bash
set -euo pipefail

# Rebuilds local dependencies directly on a GH200/ARM64 machine.
# Do not copy .venv directories from the EC2 x86_64 machine.

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
INSTALL_NODE_HINT="${INSTALL_NODE_HINT:-1}"
EXTRA_PYTHON_PACKAGES="${EXTRA_PYTHON_PACKAGES:-scikit-learn matplotlib seaborn}"
NAT_VENV="${NAT_VENV:-${HOME}/agentic_hardware/.venvs/nat_py311}"
HERMES_VENV="${HERMES_VENV:-${HOME}/agentic_hardware/.venvs/hermes_agent_py311}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DIRECT_ROOT}"

arch="$(uname -m)"
echo "Machine architecture: ${arch}"
if [[ "${arch}" != "aarch64" && "${arch}" != "arm64" ]]; then
  echo "Warning: GH200 is expected to report aarch64/arm64. Continuing anyway." >&2
fi

install_system_deps() {
  if [[ "${INSTALL_SYSTEM_DEPS}" != "1" ]]; then
    return
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo not found; skipping system dependency install."
    return
  fi
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y git curl gcc gcc-c++ make python3.11 python3.11-devel
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y git curl gcc gcc-c++ make python3.11 python3.11-devel
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git curl build-essential python3.11 python3.11-dev python3.11-venv
  else
    echo "No known package manager found; skipping system dependency install."
  fi
}

install_system_deps

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} not found. Set PYTHON_BIN to a Python >= 3.10 interpreter." >&2
  exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
  cuda_bin_dir="$(dirname "$(command -v nvcc)")"
  export CUDA_HOME="${CUDA_HOME:-$(cd "${cuda_bin_dir}/.." && pwd)}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "Using CUDA_HOME=${CUDA_HOME}"
else
  echo "nvcc not found. This is okay if the GH200 image already provides CUDA runtime libraries."
fi

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
if [[ -n "${EXTRA_PYTHON_PACKAGES}" ]]; then
  python -m pip install ${EXTRA_PYTHON_PACKAGES}
fi

mkdir -p "$(dirname "${NAT_VENV}")" "$(dirname "${HERMES_VENV}")"
"${PYTHON_BIN}" -m venv "${NAT_VENV}"
"${NAT_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${NAT_VENV}/bin/python" -m pip install "nvidia-nat[langchain]"

"${PYTHON_BIN}" -m venv "${HERMES_VENV}"
"${HERMES_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${HERMES_VENV}/bin/python" -m pip install hermes-agent

echo
echo "Python setup complete."
echo "Activate project venv: source ${DIRECT_ROOT}/.venv/bin/activate"
echo "NAT binary: ${NAT_VENV}/bin/nat"
echo "Hermes binary: ${HERMES_VENV}/bin/hermes"

if ! command -v node >/dev/null 2>&1 || ! command -v npx >/dev/null 2>&1; then
  echo
  echo "Node.js/npx not found. Install an ARM64 Node.js LTS build before running real CLI harnesses."
  if [[ "${INSTALL_NODE_HINT}" == "1" ]]; then
    cat <<'EOF'
Suggested nvm path:
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
  source "$HOME/.nvm/nvm.sh"
  nvm install --lts
  node -p "process.arch"
EOF
  fi
else
  echo "Node: $(node --version), arch=$(node -p 'process.arch')"
  echo "npx: $(npx --version)"
fi
