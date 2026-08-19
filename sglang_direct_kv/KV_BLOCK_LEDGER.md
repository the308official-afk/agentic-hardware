# Stable KV Block Ledger

This milestone turns SGLang KV cache traces into a stable block lifecycle
ledger.

## Goal

Track each logical KV block/page as a first-class object across its lifetime:

```text
created on GPU
written to host
evicted from GPU
evicted from host
loaded back to GPU
used by replay
recomputed/rebuilt
```

The goal is not to depend tightly on one SGLang version. The SGLang-specific
logic should live in one small normalizer. Everything else should consume stable
generic KV events.

## Architecture

```text
SGLang Hook Layer
  captures raw events from current SGLang internals

Normalization Layer
  converts version-specific SGLang events into stable generic events:
  KV_WRITE_HOST
  KV_EVICT_GPU
  KV_EVICT_HOST
  KV_LOAD_GPU
  KV_MATCH_PREFIX
  KV_RECOMPUTE

Stable Block-ID Layer
  assigns logical block IDs from:
  session_id
  token range
  node_id when available
  nearby range matching when node_id is missing

KV Ledger
  stores one row per logical KV block:
  first_seen
  last_seen
  current_state
  movement history
  associated session/gap
  evidence confidence

Report Layer
  shows:
  per-gap lifecycle summary
  per-block evidence table
  lost blocks
  reloaded blocks
  recomputed tokens
```

## Stable Block IDs

We should not rely only on SGLang's internal `node_id`, because node IDs can
change across versions or runs.

Preferred logical identity:

```text
agent_session_id + approximate token range + optional node_id
```

Example:

```text
kvblk_7f3a91:
  session = agent_42
  token_range = 2062..4109
  first written to host at 37139 ms
  evicted from GPU at 39740 ms
  evicted from host at 74302 ms
  replay did not load it back
```

Important nuance: some SGLang events report slightly shifted token index ranges
for the same logical chunk. The ledger therefore uses node IDs when available
and falls back to nearby-range matching.

## State Machine

Each logical block can move through these states:

```text
UNKNOWN
GPU_RESIDENT
HOST_RESIDENT
GPU_AND_HOST
MISSING
RELOADED_TO_GPU
RECOMPUTED
```

Example:

```text
hicache.write.end
  GPU_RESIDENT -> GPU_AND_HOST

hicache.evict_device.end
  GPU_AND_HOST -> HOST_RESIDENT

hicache.evict_host.end
  HOST_RESIDENT -> MISSING

replay with tiny prefix match and no H2D
  MISSING -> RECOMPUTED evidence at the gap level
```

## Outputs

Each controlled run should produce:

```text
kv_block_ledger.csv
kv_block_ledger.json
kv_block_lifecycle_summary.csv
kv_block_gap_summary.csv
```

The master HTML should include:

```text
KV Block Ledger Summary
  total blocks tracked
  blocks written to host
  blocks evicted from GPU
  blocks evicted from host
  blocks loaded back
  blocks lost before replay
  replay recomputed tokens

Per-Gap Lost Block Table
  row
  session
  lost_blocks
  lost_tokens
  replay_h2d_tokens
  replay_recomputed_tokens
  simple meaning
```

## Why This Is Strong

This lets us say something concrete:

```text
For G00:
- 4 logical KV blocks were written to host
- 3 blocks were evicted from GPU
- the same 3 blocks were evicted from host
- replay loaded 0 blocks back
- replay recomputed about 4452 tokens
```

That is stronger than only showing high TTFT or a missing green bar.

## Exact Attribution Extension

The next layer is documented in:

```text
KV_EXACT_MOVEMENT_ATTRIBUTION.md
```

Difference:

```text
This ledger:
  tracks logical KV blocks across lifecycle states.

Exact attribution:
  also records host/device index signatures, layer IDs, request IDs, and
  copy start/end windows for the SGLang movement functions.
```

In simple words, this ledger says:

```text
G04 had KV written, evicted, and loaded.
```

The exact attribution layer tries to say:

```text
G04 host indices 1812..3859 moved into GPU/device indices 4200..6247
between time A and time B.
```

## Modularity Rule

Only this file should be SGLang-version-sensitive:

```text
src/agentic_kv/block_ledger/normalizer.py
```

The state machine, block matching, CSV/JSON outputs, and report summaries should
consume stable normalized events. For a future SGLang version, update the
normalizer first before touching the rest of the infrastructure.
