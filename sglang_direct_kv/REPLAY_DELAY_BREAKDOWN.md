# Replay Delay Breakdown Note

This note captures Milestone 33 for the SGLang direct-KV testbed.

## Goal

Explain why replay-side KV host-to-device movement starts late.

Simple question:

```text
Replay was due at time 0.
H2D started much later.
What happened in between?
```

## What The Report Adds

The master report now includes:

```text
Replay Delay Breakdown
  Delay Waterfall Timeline
  Main Verdicts
  Stage Duration Table
  What Was Running Instead
  Evidence Confidence
```

## Delay Stages

For each replay gap, the report tries to split the delay into:

```text
replay due
  -> client submitted replay
  -> client request call started
  -> SGLang received request
  -> request entered scheduler queue
  -> request was admitted
  -> first cache/prefix event
  -> replay-side H2D started
  -> replay-side H2D finished
  -> first token
```

This separates:

```text
client/workload dispatch delay
scheduler queue delay
cache/load-back delay
actual H2D copy time
post-H2D first-token work
```

## Manager-Friendly Verdicts

The report labels rows with verdicts such as:

```text
copy issued late, copy was fast
copy blocked behind other H2D
copy issued on time or near-time, copy was slow
no replay H2D; recompute/prefill path
no visible replay H2D
```

It also reports the dominant delay source, for example:

```text
client/workload dispatch dominated
scheduler queue dominated
cache/load-back path dominated
H2D copy dominated
post-H2D prefill/decode dominated
```

## What Was Running Instead

For each row, the report scans the same controlled case from replay due until
target H2D start and counts:

```text
scheduler events
model-forward events
prefill batch events
decode batch events
cache/HiCache events
raw hostpool H2D events
exact target H2D events
exact other H2D events
max scheduler queue length
max running batch size
```

This helps distinguish:

```text
The copy engine was busy with other H2D work.
The request was stuck before it reached the H2D path.
The replay rebuilt/prefilled instead of loading host KV.
```

## Evidence Confidence

The report keeps the evidence honest:

```text
measured
  client-side submit/request/first-token timing

exact
  SGLang receive, scheduler, and H2D hook timing when present

inferred
  first cache-event timing and recompute/prefill timing

missing
  stage was not observed in the trace
```

## Output Files

The report builder writes:

```text
artifacts/results/reports/<report_label>/report/replay_delay_breakdown.csv
artifacts/results/reports/<report_label>/report/replay_delay_verdicts.csv
artifacts/results/reports/<report_label>/report/replay_delay_running_context.csv
```

## Why This Matters

This lets us make a stronger hardware/runtime argument.

Instead of only saying:

```text
The H2D copy was late.
```

we can say:

```text
The replay missed its deadline because the KV movement request was issued late,
even though the actual H2D copy was short.
```

or:

```text
The replay missed because its H2D copy waited behind other visible H2D movement.
```

Those two cases motivate different hardware features:

```text
deadline-aware early KV movement
priority-aware DMA scheduling
prefetch protection / residency control
better telemetry
```
