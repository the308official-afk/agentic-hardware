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

## SGLang Version Portability Audit

This is the pre-migration audit for moving from the current direct-SGLang setup
to `v0.5.11-cu129-runtime`.

The key conclusion:

```text
Most reporting, normalization, and ledger code is already modular.
The version-sensitive surface is concentrated in the SGLang hook installer and
the server launch flags.
```

### Portability Matrix

| Area | Main File(s) | Status | Why | Before Moving To `v0.5.11-cu129-runtime` |
| --- | --- | --- | --- | --- |
| Hook bootstrap | `src/sitecustomize.py` | Already modular | It is environment-gated by `AGENTIC_KV_TRACE_ENABLE=1` and imports our tracer without editing SGLang site-packages. | Keep this entry point. It should work across versions as long as `PYTHONPATH=src` is set. |
| SGLang hook installer | `src/agentic_kv/sglang_trace_patch.py` | Version-sensitive | It imports and wraps concrete SGLang classes such as `HiCacheController`, `HiRadixCache`, host KV pools, scheduler, and TP worker internals. Method names and signatures can change between SGLang versions. | Wrap this behind a version adapter before migration. Treat this as the main compatibility boundary. |
| Raw event vocabulary | `src/agentic_kv/block_ledger/normalizer.py` | Mostly modular, with a small version-sensitive map | The normalized schema is stable, but `EVENT_MAP` depends on raw event names like `hicache.load.end`, `hiradix.load_back.end`, and `hostpool.load_to_device_per_layer.end`. | Keep the normalized output stable. Move raw-event maps into versioned adapter tables if v0.5.11 renames hooks. |
| Stable KV event schema | `src/agentic_kv/block_ledger/events.py` | Already modular | It defines stable event types such as `WRITE_HOST`, `EVICT_GPU`, `LOAD_GPU`, and `MATCH_PREFIX`; it has no SGLang imports. | No migration change expected. |
| Stable block IDs | `src/agentic_kv/block_ledger/block_id.py` | Already modular | It creates logical IDs from stable fields such as session, token range, node id, and index signatures. | Keep unchanged. If v0.5.11 exposes better block/page IDs, add them as optional ID ingredients. |
| KV ledger state machine | `src/agentic_kv/block_ledger/ledger.py` | Already modular | It consumes normalized events and does not care which SGLang version emitted the raw trace. | Keep unchanged unless new v0.5.11 states need additional stable event types. |
| Evidence vocabulary | `src/agentic_kv/evidence_schema.py` | Already modular | It centralizes report-facing movement/evidence names like H2D, D2H, GPU evict, host evict, and recompute. | Keep unchanged. |
| Master report builder | `scripts/build_milestone27_controlled_replay_report.py` | Mostly modular | It consumes structured artifacts and normalized rows, but mode names and projected-hardware assumptions are report-level logic. | Keep report code stable. Only update if new v0.5.11 artifacts expose stronger evidence columns. |
| Environment capture | `scripts/collect_run_environment.py` | Mostly modular | It records run environment details, but should capture version/capability differences clearly. | Ensure the report records SGLang version, Docker image, priority scheduling support, and radix eviction policy choices. |
| HiCache server launcher | `scripts/run_sglang_hicache_server.sh` | Version/config-sensitive | Launch flags differ across SGLang builds. Current direct setup supports priority scheduling, but current `0.5.10.post1` does not support `--radix-eviction-policy priority`; v0.5.11 is expected to. | Add/keep capability-gated flags. Do not blindly pass `--radix-eviction-policy priority` unless the target version accepts it. |
| KV path probes | `scripts/probe_sglang_kv_paths.py`, `scripts/extract_sglang_kv_targets.py` | Migration helper | These scripts discover real SGLang KV/cache/offload symbols. | Rerun after installing v0.5.11 and compare target files/methods against the current map. |
| Run orchestration | `scripts/run_master_report.sh`, milestone run scripts | Mostly modular | These scripts orchestrate server launch, workloads, and report generation; they depend on the launcher and trace hooks. | Keep as-is where possible. Pass version-specific launch knobs through environment variables. |

### Recommended Adapter Boundary

Before the migration, isolate the version-sensitive SGLang code behind this
shape:

```text
src/agentic_kv/sglang_adapters/
  __init__.py
  base.py          stable adapter interface
  capabilities.py  detect installed SGLang version and supported flags
  v0510.py         current 0.5.10.post1 hook targets
  v0511.py         v0.5.11-cu129-runtime hook targets
```

The stable interface should answer:

```text
Which classes should be wrapped?
Which methods emit H2D, D2H, eviction, match-prefix, and request-stage events?
Which launch flags are supported?
Does this build support priority scheduling?
Does this build support radix eviction policy = priority?
```

### v0.5.11 Runtime Probe Result

We tested the plain public runtime image:

```text
lmsysorg/sglang:v0.5.11-cu129-runtime
```

Observed result:

```text
SGLang version: 0.5.11
Selected adapter: v0511
--enable-priority-scheduling: not exposed by plain sglang.launch_server
--radix-eviction-policy priority: not exposed by plain sglang.launch_server
python -m dynamo.sglang: unavailable in this image
```

Interpretation:

```text
The priority-retention behavior from the other project is likely not provided
by the plain SGLang runtime image alone. It likely depends on the Dynamo/SGLang
worker image or a patched wrapper such as local/dynamo-sglang:*.
```

Next migration gate:

```text
Probe the exact Dynamo/SGLang worker image used by the other project:
  local/dynamo-sglang:runtime-json-logs-gh200

If that image exposes the priority flags and still contains the hook classes,
then port the full report experiment runner to that image.
```

Probe command:

```bash
cd ~/agentic_hardware/sglang_direct_kv

IMAGE=local/dynamo-sglang:runtime-json-logs-gh200 \
DOCKER_PULL=0 \
OUT_JSON=artifacts/sglang_capabilities_dynamo_worker_gh200.json \
OUT_MD=artifacts/sglang_capabilities_dynamo_worker_gh200.md \
bash scripts/probe_sglang_capabilities_docker.sh
```

The probe now checks both `python -m sglang.launch_server --help` and
`python -m dynamo.sglang --help`, because the priority behavior may be exposed
by the Dynamo worker wrapper rather than plain upstream SGLang.

On the x86 EC2/G5 test machine, the runnable worker image should use the EC2
profile:

```bash
cd ~/kv_cache_offloading

DYNAMO_MACHINE_PROFILE=ec2 \
SKIP_FRONTEND=1 \
SKIP_WORKER=0 \
DYN_RUNTIME_JSON_LOGS=1 \
bash runtime_instrumentation/build_instrumented_dynamo_images.sh
```

The current EC2 host does not yet have `local/dynamo-sglang:runtime-json-logs-ec2`
or `local/dynamo-sglang:runtime-json-logs-gh200`. It also has about 52 GB free
under Docker root, while the Dynamo build guard asks for 80 GB.

Everything above the adapter should continue to consume stable normalized
events. That keeps the report, ledger, timelines, and audit reusable across
future SGLang versions.

### Migration Checklist

Use this checklist before replacing the current runtime:

1. Install or build `v0.5.11-cu129-runtime` in a separate environment.
2. Run the SGLang capability probe and record:

   ```bash
   cd ~/agentic_hardware/sglang_direct_kv
   source .venv/bin/activate

   PYTHONPATH=src python scripts/probe_sglang_capabilities.py \
     --out artifacts/sglang_capabilities.json \
     --out-md artifacts/sglang_capabilities.md
   ```

   - SGLang version
   - `--enable-priority-scheduling` support
   - `--radix-eviction-policy` choices
   - HiCache-related launch flags
3. Run `scripts/probe_sglang_kv_paths.py` against the new environment.
4. Compare the discovered classes/methods with the current hook matrix.
5. Add/update the v0.5.11 adapter only for changed hooks.
6. Run the forced-eviction sanity probe to verify H2D/D2H/eviction traces.
7. Run the Dynamo-priority retention sanity probe to check whether priority
   retention behavior improves under `--radix-eviction-policy priority`.
8. Rebuild the master report and run `scripts/audit_master_report_evidence.py`.

### Current Migration Risk

| Risk | Impact | Mitigation |
| --- | --- | --- |
| SGLang method names changed | Hooks silently miss events. | Capability/probe scripts must run before experiments. |
| Method signatures changed | Hook wrapper may capture weak context or fail. | Keep wrappers defensive and versioned. |
| Raw event names changed | Ledger/report may miss normalized events. | Version the raw-event mapping, not the stable event schema. |
| Priority eviction flag unsupported in current env | Priority-retention experiments may not reproduce Dynamo setup. | Move to v0.5.11 for this specific experiment. |
| Docker/runtime differences | CUDA, torch, and SGLang package paths may differ. | Capture environment in every report and keep launcher flags capability-gated. |

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
