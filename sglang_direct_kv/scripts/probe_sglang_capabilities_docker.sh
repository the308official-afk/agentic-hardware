#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.11-cu129-runtime}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"
DOCKER_PULL="${DOCKER_PULL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${PROJECT_DIR}/artifacts"

IMAGE_SLUG="$(printf '%s' "${IMAGE}" | tr '/:@' '___' | tr -cd 'A-Za-z0-9_.-')"
OUT_JSON="${OUT_JSON:-artifacts/sglang_capabilities_${IMAGE_SLUG}.json}"
OUT_MD="${OUT_MD:-artifacts/sglang_capabilities_${IMAGE_SLUG}.md}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for this probe but was not found on PATH." >&2
  exit 1
fi

if [[ "${DOCKER_PULL}" == "1" ]]; then
  docker pull "${IMAGE}"
fi

docker run --rm \
  ${DOCKER_GPU_ARGS} \
  -e PYTHONPATH=/workspace/src \
  -v "${PROJECT_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  python scripts/probe_sglang_capabilities.py \
    --out "${OUT_JSON}" \
    --out-md "${OUT_MD}"

echo "Wrote Docker capability JSON to ${OUT_JSON}"
echo "Wrote Docker capability markdown to ${OUT_MD}"
