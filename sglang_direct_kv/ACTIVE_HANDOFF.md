# Active Handoff

Updated: 2026-08-27

This file is the current source of truth for the `sglang_direct_kv` work. Prefer this file plus the repo over old chat history unless the user explicitly asks for older context.

## Current Goal

Run and analyze a controlled comparison experiment for:

- `no_prefetch`
- `dynamo_priority_hints`

The purpose is to check whether Dynamo-style priority hints actually help SGLang treat replay-critical agent requests as priority, and whether that improves replay/KV readiness timing.

## Main Experiment To Run Next

Run on EC2 from:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL=dynamo_priority_vs_no_prefetch_comparison_1 \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
MAX_TIMELINE_GAPS=32 \
MAX_PAIRS=2 \
MODES="no_prefetch dynamo_priority_hints" \
TOOL_WAIT_LIST_MS="500" \
FILLER_LIST="12 24 32" \
REQUEST_CONCURRENCY=4 \
FILLER_PROMPT_TOKENS=1024 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.72 \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

## Latest Report

Primary report path:

```text
sglang_direct_kv/artifacts/results/latest_master_report.html
```

The latest master report should come from one experiment only. Do not stitch old artifacts into the latest report.

## Current Report Preferences

- Keep the report uncluttered.
- Tables should mostly live in `Evidence Tables / Raw Proof`.
- The key visual sections are:
  - `Global KV Readiness by Mode`
  - `Comparison Timeline`
  - `Unified Forensic Stack Timeline`
- `Global KV Readiness by Mode` should focus on the clean dot chart only.
- `Comparison Timeline` should compare modes for the same task/gap/filler scenario.
- `Unified Forensic Stack Timeline` is the deep per-gap forensic view.

## Current Timeline Rule

The report builder was updated so that these lanes use one shared replay-relative time scale:

- overview
- KV zoom
- GPU pool zoom
- tool-wait GPU activity
- deadline zoom
- target KV residency

The replay execution zoom remains local because replay details are otherwise too compressed.

Latest pushed commit for this renderer change:

```text
e21117f Align comparison timeline scales
```

## Important Modes

- `no_prefetch`: baseline. No hint/direct prefetch.
- `dynamo_priority_hints`: sends Dynamo-style priority metadata only. It should not perform artificial direct KV prefetch.
- `direct_prefetch`: older direct hook prefetch mode; de-emphasized for now.
- `projected_hardware_bypass`: projected/not measured; leave out of the next experiment unless explicitly requested.

## What To Check After The Next Run

Answer these questions from the generated report and raw proof:

- Did `dynamo_priority_hints` reduce replay first-token lateness vs `no_prefetch`?
- Did `dynamo_priority_hints` reduce useful KV-ready lateness?
- Did SGLang receive the priority metadata?
- Did the scheduler/queue evidence prove that priority was actually honored?
- Was KV already resident, loaded from host, recomputed, or lost?
- Was the GPU KV pool near full during tool wait/replay?
- Did H2D/D2H/KV eviction traffic happen during the delay window?

## Key Concern

The user specifically wants proof that priority-labelled requests actually move ahead in SGLang's queue, not merely proof that the priority field was attached.

If the report cannot prove queue movement, state that clearly:

```text
Priority hints reached the request path, but queue reordering was not yet proven.
```

Then improve the priority queue effectiveness audit:

- request id + priority at client submit
- request id + priority when SGLang receives it
- queue insertion position
- number of lower-priority requests ahead
- scheduler admission order
- whether any lower-priority request was admitted before the priority request

## Instrumentation Status

The project has SGLang-level instrumentation for:

- KV lifecycle
- H2D/D2H movement
- GPU KV pool residency
- replay path timing
- scheduler/queue events where available
- priority hint request-path auditing

Be precise when explaining results:

- Directly instrumented = strong evidence.
- Inferred from timing/lifecycle reconstruction = useful but should be labelled as inferred.
- Projected hardware = projected, not measured.

## Workflow Rules

- Use simple words when explaining results.
- Do not overclaim.
- After code/report-generator edits, upload to EC2 with:

```bash
bash aws/upload.sh 0
```

- Push committed code changes to GitHub.
- Do not revert unrelated local changes.
- Known unrelated local files at the time this handoff was created:
  - modified `aws/config.sh`
  - untracked `SGLang_One_Worker_Agentic_Controller_Proposal.pdf`

