#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/artifacts/slides/build"

NODE_BIN="${NODE_BIN:-node}"

cd "${REPO_ROOT}"

if [[ -n "${RUNTIME_NODE_MODULES:-}" ]]; then
  ln -sfn "${RUNTIME_NODE_MODULES}" "${BUILD_DIR}/node_modules"
fi

"${NODE_BIN}" "${BUILD_DIR}/extract_report_charts.mjs"
"${NODE_BIN}" "${BUILD_DIR}/crop_chart_images.mjs"
"${NODE_BIN}" "${BUILD_DIR}/build_manager_deck_html.mjs"
