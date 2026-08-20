# KV H2D Bandwidth Pressure Note

This note captures the next analysis layer for the SGLang direct-KV testbed.

## Goal

For each replay deadline, show how busy host-to-device KV movement was nearby.

Simple question:

```text
When this agent needed its KV, was the memory-movement path already busy?
```

This helps explain why replay-side KV movement can be late.

## Why This Matters

The current report already shows:

```text
replay due time
replay request start
H2D KV movement start
H2D KV movement finish
first token time
```

But that only explains one gap at a time.

The bandwidth-pressure view adds surrounding context:

```text
How many other KV H2D events were happening near this replay?
How many logical KV blocks were being touched?
How many token/index movements were involved?
How much H2D copy duration was visible?
Was this gap isolated, or did it happen during a busy movement window?
```

## Report Additions

The master report now includes a section:

```text
KV H2D Bandwidth Pressure
```

It contains:

```text
H2D Activity By Time Window
  Groups exact H2D copy events by time relative to replay due.

Per-Gap Nearby H2D Pressure
  For each timeline row, counts H2D work from:
    replay_due - 500 ms
    through
    max(replay_due + 1000 ms, observed H2D finish time)

Per-Gap H2D Contention Timeline
  Picks the latest replay-H2D rows and shows all exact H2D events in the same
  controlled case while that row was waiting for KV readiness.

Per-Gap Contention Verdicts
  Separates two causes of lateness:
    blocked behind other H2D
    H2D path quiet before target

Aligned H2D Event Samples
  Shows exact lower-level H2D events after aligning them to the replay-gap clock.
```

The readable timeline also has a small pressure strip per row:

```text
nearby H2D pressure
```

This strip makes the timeline easier to interpret:

```text
low pressure     little nearby H2D work
medium pressure  noticeable nearby H2D work
high pressure    many nearby H2D events or high overlap
```

The report also keeps the fixed near-deadline count:

```text
deadline_window_h2d_events = H2D events from replay_due - 500 ms to replay_due + 1000 ms
nearby_h2d_events          = H2D events from replay_due - 500 ms through KV readiness
```

This distinction matters when H2D movement starts very late. A fixed `+1000 ms`
window may correctly say there was no H2D near the deadline, while the wider
deadline-to-ready window shows the actual H2D work the replay eventually waited
for.

The contention timeline goes one level deeper. For a target replay row, it keeps
the target row fixed and asks:

```text
While this row was waiting for its KV H2D to finish, were other KV H2D copies
already running in the same controlled case?
```

Simple verdicts:

```text
blocked behind other H2D
  Other H2D events were visible after replay due and before the target H2D began.

H2D path quiet before target
  No other H2D events were visible before the target H2D began. The delay likely
  happened before SGLang reached the H2D copy path.

target H2D visible
  The target row's own H2D movement was visible in the window.

no visible contention
  The report did not see H2D movement in that contention window.
```

## Important Nuance

This is not raw Nsight/CUPTI hardware DMA-lane telemetry.

It is:

```text
SGLang-visible exact KV H2D movement timing
```

That means it is very useful for explaining SGLang KV behavior, but still one layer above the physical DMA engine.

The closest events are:

```text
hostpool.load_to_device_per_layer
```

The report prefers those rows when available so we do not double-count higher-level wrapper events around the same movement.

## Output Files

The report builder writes:

```text
artifacts/results/reports/<report_label>/report/h2d_activity_events.csv
artifacts/results/reports/<report_label>/report/h2d_pressure_by_gap.csv
artifacts/results/reports/<report_label>/report/h2d_activity_windows.csv
artifacts/results/reports/<report_label>/report/h2d_contention_by_gap.csv
artifacts/results/reports/<report_label>/report/h2d_contention_events.csv
```

## Manager-Facing Interpretation

If a replay is late and the pressure row says high H2D pressure, the story is:

```text
The replay needed KV at time T.
The KV movement path was already busy around time T.
The H2D load finished after the replay deadline.
This suggests the current movement path is not enforcing agent-aware urgency.
```

This strengthens the hardware argument:

```text
Software can decide which agent/session matters.
Hardware/runtime support can make that movement deadline-aware, priority-aware,
and more predictable under pressure.
```
