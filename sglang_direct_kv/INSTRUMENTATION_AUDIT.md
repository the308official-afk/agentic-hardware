# Instrumentation Evidence Audit

## Goal

The master report should make strong claims only when the underlying data is strong.

For every chart, table, and timeline bar, we want to know:

```text
What raw event produced this?
Which SGLang hook emitted it?
Which session/request/block does it belong to?
Is this direct evidence, a derived value, an inference, or not proven yet?
```

This audit is the contract for keeping the report honest.

## Evidence Levels

| Level | Meaning | Example |
| --- | --- | --- |
| `DIRECT` | A trace hook or driver event directly observed the thing being plotted. | SGLang emitted a host-to-device KV load event with start/end time. |
| `DERIVED` | The value was computed from direct events. | H2D lateness = H2D finish time - replay due time. |
| `INFERRED` | The value is likely, but not directly observed as a physical event. | Replay recomputed KV because prefix match was tiny and no H2D load occurred. |
| `NOT_YET_PROVEN` | The report does not have the required evidence yet. | Physical DMA engine saturation without CUPTI/Nsight/copy-engine counters. |

## Pipeline

```text
m27_trace.jsonl
  -> SGLang hook events
  -> normalized KV movement events
  -> exact_kv_movement_attribution.csv
  -> kv_block_ledger.csv
  -> replay_delay_stage_trace.csv
  -> report charts
  -> instrumentation evidence audit
```

The important rule:

```text
Charts should be built from structured artifacts.
Structured artifacts should point back to raw trace events.
Weak or inferred claims should be labeled as weak or inferred.
```

## What Is Strong Today

The current testbed has strong software-visible evidence for:

| Area | Why It Is Strong |
| --- | --- |
| H2D KV movement | Direct SGLang hooks record host-to-device KV copy windows. |
| D2H/write-host movement | Direct SGLang/HiCache hooks record KV being written to host. |
| GPU eviction | Direct hooks record device-side KV eviction/residency changes. |
| Host eviction | Direct hooks record host-side KV loss when visible in SGLang. |
| Request stages | SGLang request-stage hooks record receive, queue/admit, and cache/load-back stages when enabled. |
| Logical block lifecycle | The ledger assigns stable logical block IDs and tracks source events over time. |
| Client dispatch movement | The report filters direct movement events to the target dispatch window. |

## Hardening Added After The First Audit

The first audit showed one important bug and one important limitation:

```text
Bug:
  Some report rows used movement_kind, while the audit was counting kind.
  That made dispatch-window H2D/D2H/GPU-evict counts look like zero.

Limitation:
  Some SGLang movement rows had session identity but weak request/correlation
  identity, so they were harder to tie to one exact request.
```

The hardening pass adds:

| Improvement | What Changed |
| --- | --- |
| Shared movement vocabulary | `agentic_kv.evidence_schema` maps all rows to one movement vocabulary: H2D, D2H, GPU evict, host evict, cache match, recompute. |
| Stronger request identity | The workload driver sends `request_id`, `parent_run_id`, `correlation_id`, `case_id`, and `gap_id` in `custom_params`. |
| Trace-field propagation | `sglang_trace_patch.py` now preserves those request/correlation fields when SGLang exposes them. |
| Ledger identity fields | Normalized KV events and block records now carry request/correlation/case/gap fields. |
| Stronger audit coverage | The audit counts request/correlation identity coverage, not only the old `request_id` column. |
| Evidence-level labels | Exact movement rows expose `evidence_level` and `exact_correlation_source`. |

Important:

```text
Reports rebuilt from old traces will benefit from the movement-kind audit fix,
but they cannot magically contain request/correlation fields that were not
present in the old trace.

Rerun the experiment after this hardening pass to populate the new identity
fields in fresh traces.
```

## What Is Still Limited

The current report should not overclaim these:

| Area | Limitation |
| --- | --- |
| Hardware DMA saturation | SGLang-visible movement does not prove physical copy-engine occupancy. That needs CUPTI, Nsight, or vendor counters. |
| Recompute/rebuild bars | These are inferred from cache match, replay prefill, and missing H2D evidence. They are not yet physical per-token recompute hooks. |
| Hardware page identity | `block_id` is a stable logical report ID, not a hardware page-table ID. |
| All memory traffic | The report sees SGLang KV movement, not every possible CUDA allocation/copy from every library. |

## Audit Outputs

Each generated master report directory should contain:

```text
instrumentation_evidence_audit_summary.csv
instrumentation_evidence_audit_matrix.csv
instrumentation_chart_inventory.csv
instrumentation_artifact_inventory.csv
instrumentation_evidence_audit.md
```

The master HTML also includes an `Instrumentation Evidence Audit` section.

## Run The Audit

From EC2 or the local machine after downloading a report:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/audit_master_report_evidence.py \
  --report artifacts/results/reports/deep_delay_trace_h2d_sweet_spot_1/report
```

To write somewhere else:

```bash
python scripts/audit_master_report_evidence.py \
  --report artifacts/results/reports/deep_delay_trace_h2d_sweet_spot_1/report \
  --out-dir artifacts/results/reports/deep_delay_trace_h2d_sweet_spot_1/report/audit_check
```

## Pass/Fail Rules

Use these rules when deciding whether a chart is manager-grade:

| Claim | Required Evidence |
| --- | --- |
| "KV moved from host to GPU" | Direct H2D row with session identity, source hook, and copy start/end. |
| "KV moved during the hint path" | Direct H2D row with `phase=hint_prefetch` and matching session/gap. |
| "KV moved during replay" | Direct H2D row with `phase=replay` and matching session/gap. |
| "KV block was written to host" | Direct write-host/D2H row tied to a logical block. |
| "KV block was evicted from GPU" | Direct GPU eviction row tied to a logical block. |
| "KV block was evicted from host" | Direct host eviction row tied to a logical block. |
| "Replay recomputed missing KV" | Label as inferred unless a direct recompute hook exists. |
| "DMA engine was saturated" | Do not claim from this report alone; needs hardware counter evidence. |

## Simple Interpretation

If a timeline shows a cyan H2D bar, we can say:

```text
SGLang visibly loaded KV from host to GPU during replay.
```

If the report shows many orange D2H/evict events during client dispatch, we can say:

```text
The SGLang KV/cache path was busy moving or evicting KV while the target replay was waiting.
```

But we should not yet say:

```text
The physical DMA engine was saturated the whole time.
```

The stronger and safer statement is:

```text
The runtime-visible KV movement path was busy, and the target's useful KV movement happened late.
This motivates a hardware/runtime path with priority, deadlines, protection, and better telemetry.
```
