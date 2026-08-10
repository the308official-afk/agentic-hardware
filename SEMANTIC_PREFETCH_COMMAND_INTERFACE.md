# Semantic Prefetch Command Interface

## Idea

A semantic prefetch command lets the runtime ask the GPU to prefetch KV cache using intent, not just addresses.

Today's generic prefetch looks like:

```text
prefetch address range X
copy memory block Y to GPU
```

A semantic KV prefetch command looks like:

```text
prefetch KV for Agent 42
deadline = 500 ms
priority = high
protect_after_prefetch = true
```

The difference is that the command carries meaning about the agent session and the expected next model turn.

## Example Command

```text
PREFETCH_KV(
  session_id = agent_42,
  kv_pages = [...],
  source_tier = CPU_DRAM,
  target_tier = HBM,
  deadline_ms = 500,
  priority = high,
  reuse_confidence = 0.85,
  protect_after_prefetch_ms = 400
)
```

This tells the memory system:

```text
which KV to move
where to move it
when it is likely needed
how important it is
how confident the runtime is
whether to protect it after arrival
```

## Concrete Coding-Agent Example

A coding agent is fixing a failing test.

```text
0 ms: Agent 42 finishes a model turn
5 ms: Agent 42 calls run_tests()
10 ms: runtime predicts tests may return around 500 ms
15 ms: runtime sends PREFETCH_KV for Agent 42
```

The command says:

```text
Agent 42's KV is likely needed soon.
Move it from CPU_DRAM to HBM before 500 ms.
Treat it as high priority.
Protect it briefly after it arrives.
```

Then, if the tests return at 500 ms, the next model turn can start quickly because the KV is already warm.

## Why Generic Prefetch Is Weaker

Generic prefetch can only say:

```text
move these bytes
```

It usually does not tell the hardware:

```text
these bytes are KV cache
this KV belongs to one agent session
this agent will likely resume soon
this session is high priority
this prefetch has a deadline
this prefetched KV should not be evicted immediately
```

So generic prefetch may be correct but still fail:

```text
prefetch starts too late
prefetch competes with active decode
prefetched KV is evicted before reuse
lower-priority KV is moved before urgent KV
wrong prefetch wastes bandwidth
```

## Mode 2 vs Mode 3

Mode 2, generic software prefetch:

```text
software calls prefetch(kv_blocks)
hardware sees addresses and bytes
runtime manually handles ordering and protection
```

Mode 3, semantic prefetch:

```text
software submits PREFETCH_KV(session, deadline, priority, confidence)
memory system can schedule using those hints
prefetched KV can be protected until reuse or timeout
telemetry reports whether the hint helped
```

Mode 2 asks:

```text
Can software move KV early using today's hardware?
```

Mode 3 asks:

```text
Would future hardware support make that movement more reliable and scalable?
```

## What Hardware Would Need

The GPU/runtime interface would need:

```text
a command format for KV prefetch hints
a queue for pending prefetch commands
priority and deadline fields
completion status
cancel/update support
telemetry counters
```

The command queue should support updates because agent predictions can change.

Example:

```text
Agent 42 was expected to resume in 500 ms.
Tests are still running after 2 seconds.
Lower the priority or cancel protection.
```

## How To Emulate This In The Prototype

Implement a software `HintPrefetchManager`.

The agent runtime submits hints:

```python
manager.submit_prefetch_hint(
    session_id="agent_42",
    kv_blocks=blocks,
    source_tier="CPU_DRAM",
    target_tier="HBM",
    deadline_ms=500,
    priority="high",
    reuse_confidence=0.85,
    protect_after_prefetch_ms=400,
)
```

The manager then:

```text
ranks pending hints by priority, deadline, and confidence
starts prefetches for the most urgent sessions
delays low-confidence or long-deadline hints
marks completed prefetches as protected
records hit, late, wasted, and evicted-before-use outcomes
```

This emulates the future hardware command interface while still running on today's GPUs.

## Simple Summary

A semantic prefetch command changes the request from:

```text
bring back this memory
```

to:

```text
make this agent's KV ready before its next model turn
```

Software still decides what should happen.

The proposed hardware interface gives the memory system enough information to enforce that decision efficiently.
