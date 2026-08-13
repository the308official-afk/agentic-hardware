#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.sh
source "${SCRIPT_DIR}/config.sh"

usage() {
  cat <<'EOF'
Usage:
  ./aws/check_ec2_ready.sh <idx>

Checks SSH, Python, disk, NVIDIA driver, and PyTorch CUDA visibility.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

INDEX="$1"
validate_server_index "${INDEX}"
chmod 400 "${PEM}"

ip="${SERVERS[$INDEX]}"
remote_host="${EC2_USER}@${ip}"

echo "==== Checking ${remote_host} ===="
ssh $(ssh_opts check) "${remote_host}" bash -s <<'REMOTE'
set -euo pipefail

echo "== Host =="
hostname
uname -a

echo "== Disk =="
df -h /

echo "== Python =="
python3 --version || true

echo "== NVIDIA =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo "== PyTorch CUDA =="
python3 - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", exc)
PY
REMOTE
