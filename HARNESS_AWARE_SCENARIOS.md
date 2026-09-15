# Harness-Aware Scenario Plan

This note is the source-of-truth checklist for the harness-aware inference
scenario workstream. The goal is to test information the harness/front end knows
but the backend does not reliably know from the current request alone.

## Core Principle

Separate harness-exposed knowledge from backend-measured state.

Harness-exposed signals answer:

```text
Who is this?
What workflow phase is it in?
When will it likely need the model again?
How urgent is it?
Is a user waiting?
How valuable is its KV/prefix state?
Can this work be delayed or cancelled?
```

Backend-measured state answers:

```text
How many prompt tokens arrived?
How much cache matched?
How full is the queue?
How busy is the GPU?
Which KV blocks are resident?
What is the current batch state?
```

The scenario suite should prove the value of the first category while holding
the second category constant.

## High-Value Harness Signal Core

Start with this contained list before expanding into more speculative signals.

1. `session_id`
   Which agent/session this belongs to.

2. `prefix_id`
   Which reusable workflow/context prefix this belongs to.

3. `phase`
   Whether the session is in `tool_wait`, `replay_ready`, `background`,
   `speculative`, `finished`, etc.

4. `expected_tool_return_ms` / `next_ready_eta_ms`
   The tool is expected to return in 2 minutes, 30 seconds, 50 ms, etc.

5. `eta_uncertainty_ms`
   How reliable that expected return time is.

6. `deadline_after_ready_ms`
   Once the tool returns, how quickly the next token matters.

7. `priority` / `urgency`
   Business, user, or task priority that is not visible from token shape.

8. `user_waiting` / `interactive`
   Whether a real user-facing interaction is waiting on this.

9. `reuse_probability`
   Whether this session's KV/prefix is likely to be reused.

10. `recompute_cost_tokens`
    How expensive it would be if the backend evicted/lost this reusable context.

11. `work_value` / `criticality`
    Whether the work is critical path, normal, background, or speculative.

12. `cancelable`
    Whether the backend/controller can safely cancel or drop the work under
    pressure.

## Package Location

The active implementation now lives inside the instrumented SGLang project:

```text
sglang_direct_kv/src/agentic_kv/harness_scenarios/
```

This keeps the scenario code modular while sharing the same SGLang install,
controller metadata, trace hooks, report builders, and artifact tree used by the
real backend experiments. The older root-level `agentic_harness_scenarios/`
directory is retained only as reference material from the first synthetic pass.

Important paths:

```text
sglang_direct_kv/configs/harness_scenarios/minimal.json
sglang_direct_kv/scripts/run_harness_aware_scenarios.py
sglang_direct_kv/scripts/run_harness_scenario_claim.py
sglang_direct_kv/src/agentic_kv/harness_scenarios/
sglang_direct_kv/tests/test_harness_scenarios.py
sglang_direct_kv/artifacts/results/harness_aware_scenarios/
```

Run the synthetic scenario suite with:

```bash
cd sglang_direct_kv
PYTHONPATH=src python3 scripts/run_harness_aware_scenarios.py
```

Run a real instrumented claim through the existing SGLang testbed with:

```bash
cd sglang_direct_kv
PYTHONPATH=src python3 scripts/run_harness_scenario_claim.py claim1_deadline_scheduling --dry-run
PYTHONPATH=src python3 scripts/run_harness_scenario_claim.py claim2_proactive_kv_hostmem --dry-run
```

## Minimal Scenario Plan

Each scenario should answer:

```text
Without harness signal X, backend made decision A.
With harness signal X, backend made decision B.
Decision B improved metric Y.
```

### 1. Better Deadline Scheduling

Harness signal:

```text
deadline_after_ready_ms
priority
user_waiting
phase = replay_ready
```

Backend state:

```text
Queue has one urgent replay and one long background prefill.
```

Baseline:

```text
FIFO/fair scheduling allows background work to delay urgent replay.
```

Harness-aware behavior:

```text
Deadline/priority-aware policy schedules urgent replay first.
```

Proof metric:

```text
Replay first-token lateness and scheduler wait.
```

### 2. Proactive KV-Cache Management

Harness signal:

```text
phase = tool_wait
expected_tool_return_ms
eta_uncertainty_ms
reuse_probability
recompute_cost_tokens
```

Backend state:

```text
HBM is pressured; inactive session KV can be retained, offloaded, or dropped.
```

Baseline:

```text
Inactive KV is treated as ordinary idle cache.
```

Harness-aware behavior:

```text
Backend keeps, offloads, or prefetches valuable soon-needed KV.
```

Proof metric:

```text
Less recompute and lower replay TTFT.
```

### 3. Better Cache Eviction Decisions

Harness signal:

```text
reuse_probability
recompute_cost_tokens
expected_tool_return_ms
```

Backend state:

```text
Two idle KV entries compete for limited memory.
```

Baseline:

```text
LRU evicts soon-needed high-value KV.
```

Harness-aware behavior:

```text
Value-aware policy evicts low-value KV.
```

Proof metric:

```text
Fewer evicted-before-use misses and fewer recomputed tokens.
```

### 4. Just-In-Time GPU Headroom

Harness signal:

```text
expected_tool_return_ms
eta_uncertainty_ms
deadline_after_ready_ms
user_waiting
```

Backend state:

```text
Background work can fill available capacity.
```

Baseline:

```text
Background work continues until urgent replay arrives.
```

Harness-aware behavior:

```text
Near ETA, backend reduces background admission and reserves small headroom.
```

Proof metric:

```text
Replay deadline misses and useful background work preserved before the prepare
window.
```

### 5. Smarter Prefill Chunking

Harness signal:

```text
expected_tool_return_ms
deadline_after_ready_ms
priority
```

Backend state:

```text
Long background prefill can run in large or small chunks.
```

Baseline:

```text
Static large chunks block urgent replay at a chunk boundary.
```

Harness-aware behavior:

```text
Chunk size shrinks near expected urgent return.
```

Proof metric:

```text
Replay scheduler wait and chunk-boundary delay.
```

### 6. More Accurate Memory Admission

Harness signal:

```text
expected_tool_return_ms
deadline_after_ready_ms
priority
user_waiting
```

Backend state:

```text
Several requests compete for KV memory and decode capacity.
```

Baseline:

```text
Admission ignores future urgent replay and over-admits background work.
```

Harness-aware behavior:

```text
Admission holds or delays background work when it threatens an upcoming replay.
```

Proof metric:

```text
Fewer replay misses and fewer bad admits.
```

### 7. Critical-Path Scheduling

Harness signal:

```text
work_value = critical_path
priority
```

Backend state:

```text
Two equal-size/equal-priority requests are ready, but one unlocks downstream
work.
```

Baseline:

```text
Both requests are treated equally.
```

Harness-aware behavior:

```text
Critical-path request runs first.
```

Proof metric:

```text
Workflow completion time and downstream unblock time.
```

### 8. Cancellation Of Low-Value Work

Harness signal:

```text
work_value = speculative
cancelable = true
```

Backend state:

```text
Speculative branches are consuming queue/KV capacity.
```

Baseline:

```text
Backend continues processing now-useless branches.
```

Harness-aware behavior:

```text
Backend cancels or drops speculative branches when pressure rises or another
branch wins.
```

Proof metric:

```text
Wasted work avoided and KV capacity freed.
```

### 9. Better Batching

Harness signal:

```text
priority
user_waiting
work_value
```

Backend state:

```text
Ready queue contains urgent short work and flexible background work.
```

Baseline:

```text
Incompatible work batches together.
```

Harness-aware behavior:

```text
Urgent interactive work is separated from flexible background batches.
```

Proof metric:

```text
Urgent p95 TTFT with controlled throughput impact.
```

### 10. Better Model / Execution-Path Selection

Harness signal:

```text
work_value
priority
user_waiting
```

Backend state:

```text
Simulator has two execution paths: cheap/fast and expensive/slow.
```

Baseline:

```text
All requests use the same path.
```

Harness-aware behavior:

```text
Low-value/background work uses a cheaper path; critical/user-facing work uses
the stronger path.
```

Proof metric:

```text
Lower cost proxy with critical quality preserved.
```

### 11. Predictive Rather Than Reactive Behavior

Harness signal:

```text
phase = tool_wait
expected_tool_return_ms
eta_uncertainty_ms
deadline_after_ready_ms
reuse_probability
recompute_cost_tokens
priority
user_waiting
```

Backend state:

```text
Background work, limited memory, and an upcoming replay.
```

Baseline:

```text
Backend acts only when replay arrives.
```

Harness-aware behavior:

```text
Backend prepares KV/headroom before replay arrives.
```

Proof metric:

```text
Tool-complete-to-first-token and whether preparation completed before replay.
```

## Charting Plan

Use three levels of evidence:

1. Executive scorecard:
   one row per benefit, showing baseline vs harness-aware improvement.

2. Scenario-specific chart:
   a simple bar, timeline, Gantt, or capacity chart tailored to the benefit.

3. Proof table:
   exact numeric rows for the underlying metrics.

The manager-facing story should be:

```text
Same backend state.
Extra harness signal.
Different backend decision.
Better outcome.
```
