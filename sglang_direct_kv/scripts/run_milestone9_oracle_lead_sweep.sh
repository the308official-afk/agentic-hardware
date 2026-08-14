#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
RESULT_ROOT_BASE="${RESULT_ROOT_BASE:-artifacts/results/milestone9_oracle_lead_sweep}"
ORACLE_LEAD_LIST="${ORACLE_LEAD_LIST:-500 1000 1500}"
MODES="${MODES:-no_prefetch oracle_direct_load}"
SESSION_COUNT="${SESSION_COUNT:-12}"
ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS:-120}"
TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS:-250 500 900 1600}"
PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST:-768 1024 1536}"
HINT_DELAY_MS="${HINT_DELAY_MS:-120}"
TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY:-8}"

case_idx=0
lead_count=0
for _ in ${ORACLE_LEAD_LIST}; do
  lead_count=$((lead_count + 1))
done

echo "Milestone 9 oracle lead sweep"
echo "Oracle leads: ${ORACLE_LEAD_LIST}"
echo "Total lead cases: ${lead_count}"

for lead in ${ORACLE_LEAD_LIST}; do
  case_idx=$((case_idx + 1))
  result_root="${RESULT_ROOT_BASE}/lead_${lead}"
  echo
  echo "==== Oracle lead case [${case_idx}/${lead_count}]: ORACLE_LEAD_MS=${lead} ===="
  RESULT_ROOT="${result_root}" \
  MODES="${MODES}" \
  SESSION_COUNT="${SESSION_COUNT}" \
  ARRIVAL_GAP_MS="${ARRIVAL_GAP_MS}" \
  TOOL_WAIT_LIST_MS="${TOOL_WAIT_LIST_MS}" \
  PROMPT_TOKEN_LIST="${PROMPT_TOKEN_LIST}" \
  HINT_DELAY_MS="${HINT_DELAY_MS}" \
  ORACLE_LEAD_MS="${lead}" \
  TRAFFIC_CONCURRENCY="${TRAFFIC_CONCURRENCY}" \
  bash scripts/run_milestone9_agentic_traffic.sh "${MODEL}"
done

echo
echo "Oracle lead sweep outputs written under ${RESULT_ROOT_BASE}"
