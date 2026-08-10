# Hardware Proposal: Agent-Aware KV Cache Prefetch

## Purpose

This note lists the hardware/runtime additions needed to support hint-guided KV cache prefetching for agentic AI workloads.

The key distinction:

> Software can decide policy, but hardware can make enforcement cheaper, faster, more predictable, and scalable.

The goal is not to claim that software cannot prefetch KV cache. The goal is to identify where hardware support makes semantic prefetching more effective than generic memory movement.

## Topic Readmes

This project will track each hardware idea in its own README:

```text
1. KV Page Tagging
2. Semantic Prefetch Command Interface
3. Deadline/Priority-Aware Migration Engine
4. Eviction Protection / Residency Hints
5. Tier-Aware KV Memory Manager
6. Optional KV Compression Path
7. KV-Aware Telemetry
```

Current detailed notes:

- [KV Page Tagging](KV_PAGE_TAGGING.md)

## Required Hardware-Oriented Features

### 1. KV Page Tagging

Add hardware-visible metadata tags to KV cache pages or page-table entries.

Example fields:

```text
session_id
priority
deadline
reuse_confidence
protection_window
tier
```

Why it matters:

Today, the GPU mostly sees memory addresses. With tags, the memory system can distinguish:

```text
old inactive KV
high-priority agent KV
KV likely needed after a tool returns
KV that should not be evicted yet
```

### 2. Semantic Prefetch Command Interface

Add a GPU command queue or API for semantic KV prefetch requests.

Example:

```text
PREFETCH_KV(session=42, pages=[...], deadline=500ms, priority=high)
```

Why it matters:

This is stronger than a generic address-range prefetch because the memory system can schedule by urgency, priority, and expected reuse.

### 3. Deadline/Priority-Aware Migration Engine

Add or extend a copy/DMA engine so it can move KV cache using deadlines and priorities.

It should support:

```text
prioritize urgent KV pages
throttle around active decode
pause and resume migrations
track partial completion
```

Why it matters:

KV prefetch should not blindly steal bandwidth from live decode. The migration engine should move useful KV early while limiting interference with active requests.

### 4. Eviction Protection / Residency Hints

Add page states or residency hints for prefetched KV.

Example states:

```text
prefetched
protected
evictable
cold
```

Why it matters:

This prevents a common failure mode:

```text
prefetch was correct
KV arrived in GPU memory
HBM pressure increased
KV was evicted before reuse
tool returned
agent stalled anyway
```

### 5. Tier-Aware KV Memory Manager

Support KV movement across memory tiers:

```text
HBM
peer GPU memory
CXL memory
CPU DRAM
```

The runtime decides which sessions matter. Hardware helps enforce migration, locality, and residency efficiently.

Example use case:

```text
Turn A ran on GPU 0.
The session's KV is still on GPU 0.
Turn B should preferably route to GPU 0, or prefetch KV to the target GPU before resume.
```

### 6. Optional KV Compression Path

Add compression/decompression support for cold or warm KV pages, especially when moving KV to CXL memory or CPU DRAM.

Why it matters:

KV cache is large. Compression can reduce capacity and bandwidth pressure, but it should be framed as an extension rather than a minimum requirement.

### 7. KV-Aware Telemetry

Expose counters that show whether prefetching helped or hurt.

Example counters:

```text
prefetch hit rate
late prefetches
wasted prefetches
evicted-before-use pages
decode bandwidth interference
time-to-first-token improvement
```

Why it matters:

Without telemetry, the runtime cannot tune prefetch policy or prove that hint-guided prefetch is improving agent performance.

## Minimal Required Set

The strongest minimum hardware-assisted design is:

```text
KV tags
+ semantic prefetch queue
+ priority-aware migration
+ eviction protection
+ telemetry
```

Tiering, peer-GPU routing, and compression are scaling extensions.

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
