# Exact KV Movement Attribution

This milestone deepens the existing KV lifecycle ledger.

The older lifecycle layer answers:

```text
For this session/gap, did KV get written, evicted, loaded, or recomputed?
```

This exact attribution layer tries to answer:

```text
Which SGLang KV indices moved?
When did that movement start and finish?
Did it happen on the hint path or the replay path?
Was the same logical block later used, lost, or reloaded?
```

## What This Adds

### Exact Identity

Each normalized KV movement event now carries:

```text
session_id
phase
request_id
node_id
layer_id
host_index_signature
device_index_signature
host_index_start/end/count
device_index_start/end/count
copy_start_ms
copy_end_ms
duration_ms
```

Simple example:

```text
G04 replay:
host indices 1812..3859 moved into device indices 4200..6247
copy window: 1240.4 ms -> 1255.9 ms
source path: hostpool.load_to_device_per_layer
```

## Why This Is Better Than The Existing Lifecycle

The existing lifecycle was gap-level:

```text
G04 wrote KV to host.
G04 later loaded KV from host.
```

The deeper layer is block/index-level:

```text
G04 host_index_signature=2048:abcd...
was written to host,
evicted from GPU,
loaded back into device_index_signature=2048:ef12...
and the load happened during replay.
```

That is stronger because it follows the same logical KV block set across the
movement path.

## Evidence Levels

```text
host_and_device_indices
  strongest software-visible evidence

host_indices / device_indices
  strong, but only one side of the movement is visible

node_id
  useful SGLang cache-node identity

range_only
  weakest; based mostly on approximate token/index ranges
```

## Outputs

The master report builder now emits:

```text
exact_kv_movement_attribution.csv
exact_kv_movement_summary.csv
kv_block_ledger.csv
kv_block_ledger.json
kv_block_lifecycle_summary.csv
kv_block_gap_summary.csv
```

And the master HTML includes:

```text
Exact KV Movement Attribution
  how to read the evidence
  exact movement summary
  exact movement rows mapped back to G00/G01/... timeline rows
```

## Important Limitation

This is still software-visible SGLang evidence, not a physical DMA bus snooper.

The strongest path is:

```text
SGLang says host indices X moved to device indices Y
NVTX/profiler labels show this happened inside the corresponding CUDA window
the replay path later reports whether it reused, loaded, or recomputed
```

That is close to hardware evidence, but still not identical to reading the GPU
DMA engine's private queue.

