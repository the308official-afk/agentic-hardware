# Hardware Proposal: Agent-Aware KV Cache Prefetch

## Purpose

This note lists the hardware/runtime additions needed to support hint-guided KV cache prefetching for agentic AI workloads.

The key distinction:

> Software can decide policy, but hardware can make enforcement cheaper, faster, more predictable, and scalable.

The goal is not to claim that software cannot prefetch KV cache. The goal is to identify where hardware support makes semantic prefetching more effective than generic memory movement.

## Topic Readmes

This project will track each hardware idea in its own README:

```text
1. Hint-Aware KV Metadata and Prefetch Interface
2. Deadline/Priority-Aware Migration Engine
3. Eviction Protection / Residency Hints
4. Tier-Aware KV Memory Manager
5. Optional KV Compression Path
6. KV-Aware Telemetry
```

Current detailed notes:

- [KV Page Tagging](KV_PAGE_TAGGING.md)
- [Deadline/Priority-Aware Migration Engine](DEADLINE_PRIORITY_AWARE_MIGRATION_ENGINE.md)

## Hardware-Oriented Features

### 1. Hint-Aware KV Metadata and Prefetch Interface (High Value Proposition)

Problem statement:

Today's GPUs mostly see addresses and memory ranges. They do not know that a memory block is KV cache for an agent session that is likely to resume soon.

Proposed feature:

```text
KV/session tags
+ semantic PREFETCH_KV command
+ priority, deadline, confidence, and protection fields
```

Without this feature:

```text
0 ms: Agent 42 starts run_tests()
20 ms: Agent 42 KV is offloaded
200 ms: software wants to prefetch Agent 42
210 ms: system issues generic prefetch for address range X
300 ms: memory system treats X like ordinary pages
350 ms: another copy request arrives first in the queue
500 ms: run_tests() returns
500-620 ms: Agent 42 waits for KV reload
620 ms: first token starts
```

With this feature:

```text
0 ms: Agent 42 starts run_tests()
20 ms: runtime tags Agent 42 KV as tool_wait, high priority
200 ms: runtime submits PREFETCH_KV for Agent 42
210 ms: memory system sees deadline = 500 ms, priority = high
300 ms: Agent 42 KV is prioritized for HBM
500 ms: run_tests() returns
505 ms: first token starts
```

Why it matters:

This lets the runtime request "make this agent's KV ready" instead of only "copy these bytes."

### 2. Deadline/Priority-Aware Migration Engine (High Value Proposition)

Problem statement:

Many KV prefetches may happen at the same time. A generic copy engine may move a low-priority KV block before an urgent one.

Proposed feature:

```text
priority-aware copy queues
+ deadline-aware scheduling
+ preemptible or throttleable migration
+ progress tracking
```

Without this feature:

```text
0 ms: Agent C starts long_build()
20 ms: Agent C low-priority KV copy begins
50 ms: Agent A starts run_tests()
80 ms: Agent A becomes likely to resume in 300 ms
90 ms: Agent A prefetch request is queued behind Agent C
300 ms: run_tests() returns
300-500 ms: Agent A waits for Agent C copy to finish and KV reload
500 ms: first token starts
```

With this feature:

```text
0 ms: Agent C starts long_build()
20 ms: Agent C low-priority KV copy begins
50 ms: Agent A starts run_tests()
80 ms: Agent A high-priority prefetch request arrives
90 ms: migration engine pauses or throttles Agent C copy
100 ms: migration engine starts Agent A KV copy
260 ms: Agent A KV is ready in HBM
300 ms: run_tests() returns
305 ms: first token starts
```

Why it matters:

This moves urgent KV first and prevents prefetch from blindly stealing bandwidth from active decode.

### 3. Eviction Protection / Residency Hints (High Value Proposition)

Problem statement:

Prefetch can be correct but still wasted if the KV is evicted before the agent resumes.

Proposed feature:

```text
prefetched/protected/evictable/cold states
+ temporary protection windows
+ priority-aware eviction choices
```

Without this feature:

```text
0 ms: Agent 42 starts run_tests()
20 ms: Agent 42 KV is offloaded
200 ms: prefetch starts
300 ms: Agent 42 KV arrives in HBM
350 ms: HBM pressure increases
360 ms: Agent 42 KV is evicted
500 ms: run_tests() returns
500-620 ms: KV must be reloaded again
620 ms: first token starts
```

With this feature:

```text
0 ms: Agent 42 starts run_tests()
20 ms: Agent 42 KV is offloaded
200 ms: prefetch starts with protect_after_prefetch = 400 ms
300 ms: Agent 42 KV arrives in HBM and is protected
350 ms: HBM pressure increases
360 ms: lower-priority unprotected KV is evicted instead
500 ms: run_tests() returns
505 ms: first token starts
```

Why it matters:

This prevents the failure mode: correct prefetch, immediate eviction, no latency benefit.

### 4. Tier-Aware KV Memory Manager

Problem statement:

Not all paused agents should be treated the same. Warm KV should stay closer to HBM than cold KV, especially when tool gaps differ.

Proposed feature:

```text
HBM/CXL/CPU/peer-GPU placement policy
+ hardware-supported migration across tiers
+ locality and transfer-cost visibility
```

Without this feature:

```text
0 ms: Agent 42 starts repo_search()
20 ms: HBM pressure increases
30 ms: Agent 42 KV is moved from HBM to CPU DRAM
180 ms: repo_search() returns
180-320 ms: KV reloads from CPU DRAM to HBM
320 ms: first token starts
```

With this feature:

```text
0 ms: Agent 42 starts repo_search()
20 ms: HBM pressure increases
30 ms: runtime marks Agent 42 KV as warm, expected reuse soon
40 ms: Agent 42 KV is moved to CXL or kept partially in HBM
180 ms: repo_search() returns
180-220 ms: small reload or promotion completes
220 ms: first token starts
```

Why it matters:

This keeps soon-to-be-used KV closer to the GPU while moving long-idle KV to cheaper tiers.

### 5. Optional KV Compression Path

Problem statement:

KV cache is large. Cold sessions can consume too much capacity and bandwidth even when they are unlikely to resume soon.

Proposed feature:

```text
compress cold/warm KV on offload
+ store compressed KV in CXL or CPU DRAM
+ decompress near HBM before reuse
```

Without this feature:

```text
0 ms: Agent 99 starts long_build()
20 ms: Agent 99 KV is offloaded to CPU DRAM
20-120 ms: full 4 GB KV is copied out
10,000 ms: long_build() returns
10,000-10,120 ms: full 4 GB KV is copied back
10,120 ms: first token starts
```

With this feature:

```text
0 ms: Agent 99 starts long_build()
20 ms: Agent 99 KV is classified cold
30-90 ms: KV is compressed from 4 GB to 2 GB and offloaded
10,000 ms: long_build() returns
10,000-10,070 ms: compressed KV is copied back and decompressed
10,070 ms: first token starts
```

Why it matters:

Compression reduces memory capacity pressure and transfer bandwidth for cold KV. This is useful for scaling, but not required for the first prototype.

### 6. KV-Aware Telemetry (High Value Proposition)

Problem statement:

Without KV-aware counters, the runtime cannot tell whether prefetch helped, arrived late, was evicted, or slowed active decode.

Proposed feature:

```text
prefetch hit/miss counters
+ late prefetch counters
+ evicted-before-use counters
+ bandwidth interference counters
+ stall-avoided estimates
```

Without this feature:

```text
0 ms: Agent 42 starts run_tests()
200 ms: prefetch starts
300 ms: KV arrives in HBM
500 ms: run_tests() returns
620 ms: first token starts
700 ms: runtime only sees end-to-end delay
```

With this feature:

```text
0 ms: Agent 42 starts run_tests()
200 ms: prefetch starts
300 ms: KV arrives in HBM
360 ms: telemetry records protected residency
500 ms: run_tests() returns
505 ms: first token starts
510 ms: telemetry reports prefetch_hit = true, stall_avoided = 115 ms
```

Why it matters:

Telemetry makes the system tunable and gives evidence that hint-guided prefetch improves agent resume latency.

## Minimal Required Set

The strongest minimum hardware-assisted design is:

```text
hint-aware KV metadata and prefetch interface
+ priority-aware migration
+ eviction protection
+ telemetry
```

Tiering, peer-GPU routing, and compression are scaling extensions.

## Expected Performance Benefits

These figures should be treated as target hypotheses for realistic, tool-heavy agentic workloads, not guaranteed results.

The main benefit is not faster raw token generation. The main benefit is faster and more predictable agent resumption after tool calls.

Expected ballpark gains:

```text
tool-return-to-first-token latency: 30-70% reduction
KV reload stall time: 40-80% reduction
P95/P99 resume latency: 2x-4x improvement under memory pressure
end-to-end agent task latency: 5-25% reduction
serving throughput under many paused agents: 10-30% improvement
wasted prefetch bandwidth: 20-50% reduction vs generic prefetch
evicted-before-use prefetches: 50-90% reduction
```

Concrete resume example:

```text
Baseline:
  tool returns
  KV reload stall = 120 ms
  first token starts at 130 ms

Generic software prefetch:
  some KV is ready
  KV reload stall = 40-70 ms

Hint-aware hardware-assisted prefetch:
  KV is prefetched, prioritized, and protected
  KV reload stall = 5-20 ms
```

Concrete task-level example:

```text
20 model resumes per agent task
baseline average KV stall = 100 ms
baseline total KV stall = 2.0 seconds

hint-aware average KV stall = 20 ms
hint-aware total KV stall = 0.4 seconds

estimated KV stall saved = 1.6 seconds
```

Manager-facing claim:

> For tool-heavy coding agents with long contexts and many concurrent sessions, hint-guided KV prefetch should reduce post-tool resume stalls by 40-80%, improve P95/P99 resume latency by 2-4x under memory pressure, and reduce end-to-end task latency by roughly 5-25%, depending on tool gap length and HBM pressure.

## Research Framing

This proposal does not require an entirely new GPU architecture.

It proposes targeted changes to the memory-management and data-movement path so software hints become enforceable at scale.

The runtime still decides:

```text
which task is important
which KV is likely to be reused
when the tool may return
how confident the prediction is
```

The hardware helps enforce:

```text
what to move first
when to throttle
what to protect
what to evict
what telemetry to report
```
