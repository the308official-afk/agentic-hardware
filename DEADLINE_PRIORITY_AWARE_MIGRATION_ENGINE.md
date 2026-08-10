# Deadline/Priority-Aware Migration Engine

## Idea

A deadline/priority-aware migration engine is the part of the GPU memory system that moves KV cache between memory locations while respecting urgency, priority, and active decode pressure.

Example movements:

```text
CPU DRAM -> GPU HBM
CXL memory -> GPU HBM
GPU 0 -> GPU 1
GPU HBM -> CPU DRAM
```

Today, GPUs already have copy engines or DMA engines. They can move memory, but they mostly treat transfers as generic copies:

```text
copy buffer A to buffer B
```

The proposed change is to make the migration engine aware of:

```text
which KV is needed soonest
which session is highest priority
how much bandwidth active decode is using
whether a transfer should pause or resume
whether part of the KV should move first
```

## Concrete Coding-Agent Example

Agent 42 is waiting on a test run.

```text
Agent 42 state = tool_wait
expected resume = 500 ms
priority = high
KV size = 2 GB
current KV location = CPU DRAM
target location = GPU HBM
```

The runtime has already submitted a hint:

```text
Agent 42 will likely resume soon.
Make its KV ready before 500 ms.
```

The migration engine is responsible for actually moving the KV:

```text
CPU DRAM -> GPU HBM
```

## Why A Generic Copy Engine Is Weaker

A normal copy engine may execute transfers in simple submission order.

Suppose three agents are waiting:

```text
Agent A:
  expected resume = 300 ms
  priority = high
  KV size = 2 GB

Agent B:
  expected resume = 2 seconds
  priority = medium
  KV size = 1 GB

Agent C:
  expected resume = 20 seconds
  priority = low
  KV size = 4 GB
```

A generic copy engine may do:

```text
copy Agent C first because its request arrived first
then Agent B
then Agent A
```

That is bad because Agent A is likely to resume first.

A deadline/priority-aware migration engine should do:

```text
copy Agent A first
copy Agent B later if bandwidth and HBM allow
avoid copying Agent C too early
```

## Why Hardware Support Helps

Software can decide migration policy. It can rank sessions, predict tool returns, and issue prefetch requests.

The problem is enforcement.

The hardware copy/DMA engine is the part that actually moves bytes. It is closer to the memory system and can react faster to:

```text
memory bandwidth pressure
active decode traffic
partially completed transfers
urgent new transfers
priority changes
short tool-return windows
```

So the argument is not:

```text
software cannot schedule KV migration
```

The stronger argument is:

```text
software can decide what should move,
but hardware can enforce movement timing, priority, and bandwidth limits more predictably.
```

## Concrete Hardware Advantage

Suppose a low-priority transfer starts first:

```text
Agent C:
  KV size = 4 GB
  expected resume = 20 seconds
  priority = low

Agent A:
  KV size = 2 GB
  expected resume = 300 ms
  priority = high
```

Software-only behavior may look like:

```text
0 ms: Agent C copy starts
50 ms: Agent A becomes urgent
50 ms: software wants Agent A first
50-450 ms: Agent C copy keeps occupying bandwidth
300 ms: Agent A's tool returns
300-500 ms: Agent A stalls waiting for KV
```

With hardware support:

```text
0 ms: Agent C copy starts
50 ms: Agent A becomes urgent
55 ms: migration engine pauses or slows Agent C
60 ms: migration engine starts Agent A
260 ms: Agent A KV is ready
300 ms: Agent A's tool returns
305 ms: first token starts
```

The benefit comes from preemption and prioritization close to the copy engine.

## Decode Bandwidth Pressure

KV prefetch should not blindly steal bandwidth from live decode.

Bad behavior:

```text
active decode is running
large KV prefetch starts
prefetch consumes memory bandwidth
decode slows down
live requests suffer
```

Better behavior:

```text
active decode is using high bandwidth
migration engine slows or pauses prefetch
decode pressure drops
migration engine resumes prefetch
urgent KV still finishes before deadline if possible
```

So this feature is not only about moving the right KV. It is also about moving it at the right time.

This is hard to do purely in software because decode pressure changes quickly. Software may observe it too late or only through coarse signals.

Hardware can make finer-grained decisions:

```text
decode bandwidth is high -> throttle prefetch
decode pressure drops -> resume prefetch
urgent deadline is near -> allow higher prefetch bandwidth
low-priority copy is running -> pause or slow it
```

## Hardware Changes Needed

This does not require a completely new GPU architecture. It requires targeted additions to the memory/copy subsystem:

```text
priority fields on copy/prefetch commands
deadline fields on migration requests
preemptible DMA/copy transfers
bandwidth throttle controls
decode-pressure counters visible to the migration engine
completion/progress reporting per transfer
multiple priority queues for KV movement
```

These additions make software hints enforceable at the point where memory movement actually happens.

## Difference From Hint-Aware KV Metadata

The first hardware feature describes intent:

```text
This is Agent 42's KV.
It is high priority.
It is likely needed by 500 ms.
```

The migration engine enforces that intent:

```text
Move Agent 42's KV before lower-priority KV.
Throttle movement if active decode is busy.
Resume movement when bandwidth is available.
Track whether the transfer completed in time.
```

In short:

```text
Hint-aware metadata/interface = describe what matters
Migration engine = schedule the actual data movement
```

## Hardware Capabilities Needed

A deadline/priority-aware migration engine would need:

```text
priority queues for KV transfers
deadline-aware scheduling
pause/resume support
bandwidth throttling
partial transfer tracking
decode-pressure awareness
completion status
```

## How To Emulate This In The Prototype

Implement a software migration scheduler inside the `HintPrefetchManager`.

The scheduler should:

```text
rank pending KV transfers by priority, deadline, and confidence
start urgent transfers first
limit prefetch bandwidth while active decode is busy
pause low-priority prefetches under pressure
resume paused transfers when bandwidth frees up
track whether transfers completed before tool return
```

Example scoring logic:

```python
score = (
    priority_weight
    + reuse_confidence
    - time_until_deadline_ms * deadline_penalty
)
```

The exact scoring function can be simple at first. The important part is to compare it against a generic first-come, first-served prefetch baseline.

## Metrics

Useful metrics for this feature:

```text
KV transfer completion before deadline
tool-return-to-first-token latency
active decode slowdown
prefetch bandwidth used
paused/resumed transfer count
late prefetch count
wasted prefetch bytes
```

## Simple Summary

The migration engine is the traffic controller for KV movement.

It decides:

```text
which KV moves first
how fast it moves
when to pause
when to resume
how to avoid hurting active decode
```

The goal is to make urgent agent KV ready before the next model turn without slowing down live inference.
