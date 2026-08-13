#!/usr/bin/env bash
set -euo pipefail

install_system_deps() {
  local cuda_packages="${CUDA_PACKAGES:-cuda-nvcc-12-8 cuda-cudart-devel-12-8 cuda-driver-devel-12-8 cuda-nvrtc-devel-12-8 libnvjitlink-devel-12-8}"

  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo not found; skipping system dependency install."
    return
  fi

  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y gcc python3.11 python3.11-devel
    if [[ "${INSTALL_CUDA_TOOLKIT:-1}" == "1" ]] && ! command -v nvcc >/dev/null 2>&1; then
      sudo dnf install -y ${cuda_packages}
    fi
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y gcc python3.11 python3.11-devel
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y gcc python3.11 python3.11-dev python3.11-venv
  else
    echo "No known package manager found; skipping system dependency install."
  fi
}

install_system_deps

if command -v nvcc >/dev/null 2>&1; then
  cuda_bin_dir="$(dirname "$(command -v nvcc)")"
  export CUDA_HOME="${CUDA_HOME:-$(cd "${cuda_bin_dir}/.." && pwd)}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "Using CUDA_HOME=${CUDA_HOME}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} not found. Set PYTHON_BIN to a Python >= 3.10 interpreter." >&2
  exit 1
fi

if [[ -d .venv ]]; then
  existing_version="$(
    .venv/bin/python - <<'PY' 2>/dev/null || true
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
  )"
  if [[ "${existing_version}" != "3.11" ]]; then
    echo "Removing incompatible .venv using Python ${existing_version:-unknown}."
    rm -rf .venv
  fi
fi

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

mkdir -p artifacts/results

echo "Setup complete. Activate with: source .venv/bin/activate"
