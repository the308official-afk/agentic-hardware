# Replay Delay Deep Instrumentation

This note documents the deeper SGLang instrumentation added after the replay-delay
breakdown milestone.

## Goal

When replay-side KV H2D starts late, we want to know what happened before the
copy began.

The important question is:

```text
Was the delay caused by the client driver, SGLang request ingress, scheduler
queueing, cache lookup/load-back, other H2D movement, recompute/prefill, or
model execution?
```

## What Changed

The SGLang trace patch now emits exact request-stage events from wrapped SGLang
methods:

```text
kv_telemetry.request_stage
```

These events are emitted from inside SGLang method wrappers, not inferred from
server logs.

Tracked stages include:

```text
sglang_receive
scheduler_input_batch
scheduler_queue_enter
scheduler_prefetch_kvcache
scheduler_select_prefill
scheduler_select_run
scheduler_run_batch
cache_match_prefix
cache_host_ready_check
cache_load_back_plan
cache_load_back_node
host_to_device_copy
device_to_host_copy
model_forward_batch
model_forward_extend
model_forward_decode
```

## Why This Matters

Before this milestone, the report could say:

```text
H2D started 37 seconds after replay was due.
```

Now, after rerunning with the new instrumentation, the report can show a deeper
path:

```text
0 ms: replay was due
2 ms: client submitted replay request
15 ms: SGLang received the request
43 ms: request entered scheduler queue
940 ms: scheduler selected the request
960 ms: cache lookup/load-back began
37,300 ms: replay-side H2D began
37,540 ms: replay-side H2D finished
```

That gives a stronger argument than "the copy was late." It shows where the
software/runtime path spent time before KV movement happened.

## New Report Artifacts

Each controlled master report now writes:

```text
replay_delay_stage_trace.csv
replay_delay_h2d_activity.csv
replay_delay_gap_verdicts.csv
```

Simple meanings:

```text
replay_delay_stage_trace.csv
  exact SGLang method-stage events around each replay delay window

replay_delay_h2d_activity.csv
  exact H2D events visible between replay due and target KV readiness

replay_delay_gap_verdicts.csv
  compact per-gap verdicts explaining the most likely delay source
```

The same information appears in `latest_master_report.html` under:

```text
Replay Delay Breakdown
  Exact SGLang Request Stage Trace
  H2D Activity During The Delay Window
  Stage Duration Table
  What Was Running Instead
```

## Important Note

Existing reports generated before this milestone will not contain
`kv_telemetry.request_stage` rows. Rebuild-only runs can update the HTML layout,
but a new SGLang experiment must be run to populate the exact request-stage
trace.

