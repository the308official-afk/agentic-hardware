# Hint-Guided KV Cache Prefetching for Agentic AI Workloads

## Goal

Build a realistic proof of concept showing that agent/runtime hints can reduce KV cache resume stalls in agentic LLM workloads............

The key question:

> When an agent is paused on a tool call, can the runtime use that pause to prefetch the agent's KV cache back into GPU memory before the next model turn arrives?

This prototype does not require new GPU hardware. Instead, it emulates hardware-assisted behavior in software using real LLM serving, real KV tensors, real tool gaps, and real GPU memory pressure.

## Workload Scenario

Use coding-agent-style workflows inspired by SWE-bench:

```text
LLM turn
-> tool call: search files / run tests / inspect error / edit code
-> tool wait
-> tool returns
-> next LLM turn
```

During the tool wait, the session's KV cache may be offloaded or evicted from fast GPU memory. When the tool returns, the next LLM turn needs that KV cache again.

If the KV cache is not resident in GPU memory, the agent stalls before first token.

## Three Evaluation Modes

### Mode 1: No Special Prefetch

The runtime does nothing during the tool gap.

Example:

```text
0 ms: Agent 42 starts run_tests()
20 ms: Agent 42 KV is offloaded from GPU memory
500 ms: run_tests() returns
500 ms: Agent 42 needs the model again
500-620 ms: KV is reloaded
620 ms: first token starts
```

This measures the baseline cost of cold KV cache resume.

### Mode 2: Generic Software Prefetch

The runtime uses ordinary software logic to prefetch KV during tool waits.

Example policy:

```python
if session.state == "tool_wait":
    prefetch(session.kv_blocks)
```

This mode is intentionally simple. It can copy KV blocks back to GPU memory, but it does not use rich session priority, deadlines, protection, or bandwidth-aware scheduling.

Example:

```text
0 ms: Agent 42 starts run_tests()
20 ms: Agent 42 KV is offloaded
200 ms: software sees Agent 42 is waiting
210 ms: software starts generic prefetch
350 ms: KV arrives in GPU memory
420 ms: KV is evicted again under HBM pressure
500 ms: run_tests() returns
500-620 ms: KV must be reloaded again
620 ms: first token starts
```

This answers:

> How much can ordinary software prefetch help?

### Mode 3: Hint-Guided KV Prefetch

The runtime emits structured hints, and a software prefetch manager emulates the proposed hardware behavior.

Example hint:

```python
hint = {
    "session_id": 42,
    "state": "tool_wait",
    "priority": "high",
    "expected_resume_ms": 500,
    "reuse_confidence": 0.85,
    "protect_after_prefetch_ms": 400,
    "throttle_if_decode_busy": True,
}
```

The hint-guided manager emulates hardware features:

- KV page/session tags
- priority-aware prefetch queue
- deadline-aware scheduling
- decode-aware bandwidth throttling
- temporary KV protection after prefetch
- telemetry for hits, misses, late prefetches, and wasted prefetches

Example:

```text
0 ms: Agent 42 starts run_tests()
10 ms: runtime submits hint: high priority, expected resume around 500 ms
50 ms: manager starts prefetch using spare bandwidth
180 ms: active decode gets busy, manager slows prefetch
260 ms: decode quiets down, manager resumes prefetch
330 ms: KV is back in GPU memory
330-500 ms: KV is protected from eviction
500 ms: run_tests() returns
505 ms: first token starts
```

This answers:

> Would semantic, hardware-style support make KV prefetch more reliable and efficient than generic software prefetch?

## Concrete Difference Between Mode 2 and Mode 3

Mode 2 says:

```text
Copy these KV blocks back to GPU.
```

Mode 3 says:

```text
This agent is likely to resume soon.
Its KV is high priority.
Prefetch it before the deadline.
Throttle around active decode.
Protect it after prefetch.
Track whether the hint helped.
```

Mode 2 is address/block based.

Mode 3 is intent based.

## Example Scheduling Case

Suppose three agents are waiting:

```text
Agent A:
  tool = run_tests
  expected return = 300 ms
  priority = high
  KV size = 2 GB

Agent B:
  tool = repo_search
  expected return = 2 seconds
  priority = medium
  KV size = 1 GB

Agent C:
  tool = long_build
  expected return = 20 seconds
  priority = low
  KV size = 4 GB
```

Generic software prefetch may prefetch in arrival order or after a fixed timeout.

Hint-guided prefetch should:

```text
1. Prefetch Agent A first.
2. Prefetch Agent B later if bandwidth and HBM allow.
3. Avoid prefetching Agent C too early.
```

## Metrics

Primary metrics:

- tool-return-to-first-token latency
- KV reload stall time
- P95 and P99 resume latency
- end-to-end agent task latency

Efficiency metrics:

- prefetch hit rate
- late prefetch rate
- wasted prefetch bandwidth
- prefetched-then-evicted rate
- active decode slowdown
- GPU memory pressure

## Expected Result

The expected result is not that hint-guided prefetch makes model decode faster.

The expected result is:

> Hint-guided KV prefetch makes agent resumption faster and more predictable under memory pressure.

The strongest gains should appear with:

- many concurrent agent sessions
- long contexts and large KV caches
- frequent tool calls
- bursty tool returns
- limited HBM capacity
- CPU/CXL/peer-GPU KV offload

## Research Claim

Software can decide prefetch policy, but hardware can make enforcement cheaper, faster, and more predictable.

This prototype emulates that future hardware behavior in software first. If the emulation shows meaningful improvements, it motivates hardware/runtime co-design for agent-aware KV cache prefetching.
