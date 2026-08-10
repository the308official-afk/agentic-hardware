# KV Page Tagging

## Idea

KV page tagging means attaching small labels to KV cache memory so the runtime and memory system know what each KV block represents.

Today, the GPU mostly sees memory like this:

```text
page A
page B
page C
```

With KV page tags, the system can see:

```text
page A: old inactive KV
page B: high-priority agent KV waiting on tests
page C: active decode KV
```

The hardware does not need to understand the agent's tool or plan. The runtime supplies the meaning. Hardware only sees compact labels.

## Example Tag Fields

```text
session_id
priority
deadline
reuse_confidence
protection_window
tier
```

Example:

```text
session_id = agent_42
priority = high
state = tool_wait
deadline = 500 ms
reuse_confidence = 0.85
protection_window = 400 ms
tier = CPU_DRAM
```

## Concrete Coding-Agent Example

A coding agent is working on a SWE-bench-style task.

```text
1. The model reads the issue and repository context.
2. The model builds KV cache for that session.
3. The agent calls run_tests().
4. While tests run, the model is paused.
5. The KV cache may be moved out of GPU HBM.
6. The runtime knows this session will likely resume when tests finish.
```

The runtime tags the KV:

```text
type = KV_CACHE
session_id = agent_42
state = tool_wait
priority = high
expected_reuse = soon
evictability = low
```

Now the memory system can make a better decision under pressure:

```text
Evict old inactive KV first.
Keep or prefetch agent_42 KV because it is likely needed soon.
```

## Why Software Alone Is Weaker Here

Software can track this metadata in normal data structures. The issue is scale and enforcement.

In a busy agent-serving system, there may be:

```text
thousands of active sessions
millions of KV pages
many tool calls returning at similar times
limited HBM capacity
constant decode traffic
```

If metadata only lives in software, the runtime has to micromanage which pages to copy, evict, protect, and prioritize.

With hardware-visible tags, the runtime can provide intent once:

```text
this KV belongs to agent_42
it is high priority
it is likely needed soon
avoid evicting it if possible
```

Then lower-level memory mechanisms can enforce that intent closer to the data movement path.

## What This Enables

KV page tagging enables later hardware features:

```text
semantic prefetch queues
priority-aware migration
eviction protection
tier-aware KV placement
KV-aware telemetry
```

Without tags, these features only see addresses.

With tags, they can make decisions based on agent/session importance.

## Simple Summary

KV page tagging turns anonymous memory pages into meaningful KV cache objects.

The runtime decides what the tags mean.

The hardware uses those tags to move, protect, evict, and report on KV cache more intelligently.
