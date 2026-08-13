# Hardware Emulation Environment

## Goal

Build one realistic software-emulated hardware environment to estimate the performance benefits of hint-guided KV cache prefetching.

The goal is to answer:

> If future hardware made agent-aware KV hints enforceable, how much could we reduce KV resume stalls, tail latency, and wasted prefetch work?

This does not require custom GPU hardware. The proposed hardware features are emulated in software while using real GPU memory movement where possible.

## Recommended Approach

Use one combined component:

```text
HintAwareKVHardwareEmulator
```

It emulates:

```text
hint-aware KV metadata
semantic KV prefetch requests
priority/deadline-aware migration
temporary residency protection
simple tier-aware placement
KV-aware telemetry
```

The actual GPU is unchanged. The emulator sits in the runtime path and controls KV placement, prefetch, protection, and measurement.

## High-Level Architecture

```text
Agent Workload Driver
        |
        v
Agent Runtime / Tool Loop
        |
        v
LLM Serving Runtime
        |
        v
HintAwareKVHardwareEmulator
        |
        v
GPU HBM / CPU DRAM / Emulated CXL Tier
```

Concrete version for the target stack:

```text
Deep Agents-style workload
        |
        v
Dynamo / scheduler layer
        |
        v
SGLang or vLLM serving
        |
        v
KV residency + prefetch emulator
        |
        v
GPU HBM / CPU DRAM
```

If CXL is unavailable, emulate it using CPU memory plus artificial bandwidth/latency limits.

## What Should Be Real vs Emulated

Real:

```text
GPU memory allocations
KV-sized tensors or real KV tensors
CPU <-> GPU copies
GPU memory pressure
tool-call timing
request scheduling
time-to-first-token measurement
```

Emulated:

```text
future hardware command interface
KV metadata tags
priority queues
protection bits
CXL-like memory tier
hardware telemetry counters
deadline-aware DMA behavior
```

This keeps the prototype realistic without requiring hardware changes.

## State Tracked By The Emulator

For each agent session:

```text
session_id
state: active_decode / tool_wait / ready_to_resume / done
priority
tool_start_time
expected_tool_return_time
reuse_confidence
kv_size
kv_location: HBM / CXL_EMULATED / CPU_DRAM
protected_until
```

For each KV object or KV block group:

```text
owner_session
size_bytes
current_tier
prefetch_status
protected: true/false
last_access_time
deadline
```

Start with session-level KV objects or KV block groups. Page-level accuracy can come later.

## Evaluation Modes

Run the same workload under three modes.

### Mode 1: Baseline / No Agent-Aware Prefetch

Behavior:

```text
Tool returns.
If KV is not in HBM, reload it then.
No proactive prefetch.
No residency protection.
```

Expected result:

```text
high tool-return-to-first-token latency
high P95/P99 resume latency under memory pressure
```

### Mode 2: Generic Software Prefetch

Behavior:

```text
When a session enters tool_wait, software may prefetch KV using simple rules.
Prefetch is generic and address/block based.
No strong priority enforcement.
No reliable protection.
No decode-aware throttling.
```

Example rule:

```text
prefetch after 200 ms of tool_wait
or prefetch in FIFO order
```

Expected result:

```text
helps some cases
fails when prefetch is late, evicted, or competes with decode
```

### Mode 3: Hint-Aware Hardware Emulation

Behavior:

```text
Runtime sends hints:
  session_id
  priority
  expected resume time
  reuse confidence
  protection window

Emulator enforces:
  urgent KV moves first
  prefetch throttles around decode
  prefetched KV is protected briefly
  telemetry records hit/late/wasted/evicted-before-use
```

Expected result:

```text
lower resume latency
better P95/P99 latency
fewer wasted prefetches
less decode interference
```

## How The Emulator Covers The Hardware Proposal

Hint-aware metadata/interface:

```python
PrefetchHint(
    session_id,
    kv_object_id,
    deadline_ms,
    priority,
    confidence,
    protect_ms,
)
```

Deadline/priority-aware migration:

```text
rank pending KV transfers by priority, deadline, and confidence
limit active copies
pause or delay low-priority copies
prefer urgent sessions
```

Eviction protection:

```text
do not evict protected KV unless forced
evict unprotected cold KV first
expire protection after timeout
```

Tier-aware memory:

```text
HBM = GPU tensors
CXL_EMULATED = pinned CPU memory with artificial latency/bandwidth model
CPU_DRAM = regular CPU tensors
```

Compression, optional later:

```text
simulate compressed KV by reducing bytes moved
or actually compress tensors with quantization
```

Telemetry:

```text
prefetch_submitted
prefetch_completed
prefetch_hit
prefetch_late
prefetch_wasted
evicted_before_use
stall_ms
decode_slowdown_ms
```

## Prototype Levels

### Level 1: Lightweight Real-Memory Emulator

Use real GPU tensors as KV-sized objects.

Example:

```text
Agent 42 KV = 2 GB torch tensor
tool_wait = 500 ms
HBM pressure = other tensors
prefetch = CPU -> GPU async copy
resume = wait until KV is in GPU
```

Benefits:

```text
fast to build
uses real GPU allocations
uses real CPU-GPU transfers
captures real bandwidth contention
does not require SGLang internals yet
```

This level proves the memory behavior.

### Level 2: LLM Runtime Integration

After Level 1 works, connect the emulator to SGLang, vLLM, or Dynamo.

Benefits:

```text
uses real LLM requests
captures runtime scheduling
captures real time-to-first-token
shows interaction with active decode
more convincing for managers
```

This level proves the idea inside a real serving path.

## Example Workload

Synthetic coding-agent traces:

```text
Agent 1:
  model turn = 300 ms
  tool wait = 500 ms
  KV size = 1.5 GB

Agent 2:
  model turn = 200 ms
  tool wait = 2 seconds
  KV size = 1 GB

Agent 3:
  model turn = 400 ms
  tool wait = 100 ms
  KV size = 2 GB
```

Scale concurrency:

```text
10 agents
50 agents
100 agents
500 agents
```

Sweep pressure:

```text
HBM pressure = low / medium / high
KV sizes = small / medium / large
tool gaps = predictable / bursty / noisy
```

## Best First Demo

Run:

```text
100 agents
one GPU
limited HBM budget
KV objects from 256 MB to 2 GB
tool gaps from 100 ms to 3 seconds
background decode-like memory load
bursty tool returns
```

Compare:

```text
Mode 1: no prefetch
Mode 2: generic software prefetch
Mode 3: hint-aware hardware emulation
```

Expected graph:

```text
x-axis: concurrent agents
y-axis: P95 tool-return-to-first-token latency
```

Expected shape:

```text
Mode 1: worst
Mode 2: better but unstable under pressure
Mode 3: best and more stable
```

## Metrics

Primary:

```text
tool-return-to-first-token latency
P50/P95/P99 resume latency
KV reload stall time
end-to-end agent completion time
```

Efficiency:

```text
prefetch hit rate
late prefetch rate
evicted-before-use rate
wasted prefetch bytes
decode slowdown from prefetch
GPU memory utilization
copy bandwidth utilization
```

Manager-friendly summary:

```text
P95 resume latency reduced by X%
KV reload stall time reduced by Y%
prefetch wasted bytes reduced by Z%
active decode slowdown kept below N%
```

## Recommendation

For the direct SGLang path, start with:

```text
sglang_direct_kv/
```

See: [SGLang Direct KV Instrumentation Testbed](sglang_direct_kv/README.md)

Earlier recommendation:

Start with the PyTorch-based real-memory emulator if we want the fastest memory-only prototype.

Reason:

```text
real GPU allocations
real CPU-GPU transfers
real bandwidth contention
easy to implement
not blocked on SGLang KV internals
```

Then integrate with SGLang/Dynamo after the memory-level result is clear.

The strongest story is:

```text
1. Controlled real-memory emulation proves the hardware mechanisms can help.
2. Runtime integration shows the same idea inside real LLM serving.
```
