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

Deep-dive analyzer:

```bash
cd sglang_direct_kv
TRACE_REPLAY_FRICTION_DEEP_DIVE=1 \
TRACE_REPLAY_BLOCKERS=1 \
TRACE_REPLAY_BLOCKERS_MAX_IDS=16 \
bash scripts/run_deadline_fair_realistic.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

This opt-in analyzer writes `replay_friction_deep_dive.html`,
`replay_friction_deep_dive.csv`, and `replay_friction_summary.csv` into the
run report directory. The run report version keeps the full detailed table.
It also refreshes a top-level `replay_friction_deep_dive.html` beside
`harness_aware_scenario_tracker.html`; that top-level version is an
automatically generated plain-English reader summary with representative
requests selected by fixed rules. It is off by default so other scenarios do
not pay extra reporting cost. The analyzer uses
`global_kv_readiness_by_mode.csv` for every run and upgrades to trace-rich
attribution when the run root still has `m27_trace.jsonl` and
`harness_gateway_events.jsonl`.

Optional blocker snapshots:

```text
TRACE_REPLAY_BLOCKERS=1
TRACE_REPLAY_BLOCKERS_MAX_IDS=16
TRACE_REPLAY_BLOCKERS_EVENTS=before_acquire,submit
```

These snapshots are bounded and replay-focused. They capture the first N local
pending-ahead and in-flight request IDs at key replay moments, then the offline
analyzer classifies each blocker as reasonable, questionable, avoidable, or
unknown. The trace also records snapshot build time and estimated snapshot bytes
so the deep-dive page can flag instrumentation overhead.

Current deep-dive artifact:

```text
replay_friction_deep_dive.html
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

Current all-request experiment shape:

```text
Run no_prefetch against controller_proactive_kv_management.
Treat every replay request as a real request, not as target versus filler.
Each replay session has its own session_id, prefix_id, expected return time,
reuse_probability, and recompute_cost_tokens. The controller may request
host-to-device KV movement for any replay session whose harness signals make
reuse likely and recompute costly.
```

Authorized KV load path:

```text
prepared_prefix_control only.
The controller must call SGLang/HiCache prepare-prefix control so SGLang itself
performs and records the host-to-device KV load. Do not use a separate request
as a cache-warming substitute.
```

Current controller admission rule:

```text
Before moving KV, the controller first sends a plan_only prepare-prefix request.
SGLang answers from its live prefix/cache state:
- already_device_resident: skip, because replay should already see it on GPU.
- host_backup_unavailable / load_back_not_admitted: skip, because there is no
  safe useful host-to-device load to perform.
- would_load_back: admit only if the host-resident prefix is large enough to
  save meaningful replay work.

This keeps proactive KV work from sitting in front of replay unless the backend
can prove there is useful GPU-evicted, host-backed KV to restore.
```

Repeatable wrapper:

```bash
cd sglang_direct_kv
bash scripts/run_proactive_kv_management_realistic.sh Qwen/Qwen2.5-Coder-7B-Instruct
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
Reusable KV from all replay-capable sessions competes under limited GPU memory.
The load generator may create pressure sessions, but they are treated as real
replay-capable workload requests, not as a special target/filler split.
```

Baseline:

```text
Ordinary eviction does not know which cached prefixes are likely to be needed soon or expensive to rebuild.
```

Harness-aware behavior:

```text
The controller translates reuse probability, recompute cost, and expected replay timing into per-request SGLang priorities. SGLang's own priority radix eviction then chooses lower-value KV as the better victim across the whole workload.
```

SGLang compatibility note:

```text
The EC2 SGLang version checked on 2026-09-17 is 0.5.10.post1. It already has
the internal PriorityStrategy and stores priority on radix cache nodes, but its
CLI choice list exposes only lru, lfu, and slru by default. Scenario 3 enables
an opt-in project shim, AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE=1, so
the existing internal priority strategy can be selected with
--radix-eviction-policy priority. This does not install or upgrade SGLang.
```

Proof metric:

```text
High-value prefixes receive higher priority, normal-value prefixes receive neutral priority, low-value prefixes receive lower priority, native HiCache eviction events are observed, and aggregate replay TTFT/lateness improves without relying on one special target request.
```

Run:

```bash
cd sglang_direct_kv
bash scripts/run_value_aware_eviction_realistic.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Preflight smoke:

```bash
cd sglang_direct_kv
AGENTIC_KV_ENABLE_PRIORITY_RADIX_EVICTION_CHOICE=1 \
python scripts/smoke_priority_radix_eviction.py \
  --out-json artifacts/results/priority_radix_eviction_smoke.json
```

The smoke does not launch a model server. It verifies that the opt-in shim
exposes `priority` as a valid radix eviction policy, that SGLang launch help
accepts it, and that a simulated SGLang `RadixCache` using
`PriorityStrategy` evicts low-priority KV before high-priority KV.

Latest result:

```text
Run value_aware_eviction_all_requests_fixed_20260917_163518 compared
no_prefetch with controller_value_aware_eviction across 32 replay requests.
Total TTFT improved from 482.3s to 233.5s, and total lateness improved from
1014.0s to 659.9s. The audit showed zero priority mismatches: high, normal,
and low value classes reached SGLang as priorities 100, 0, and -100. Native
SGLang/HiCache eviction events were observed. The trade-off is that total
workload window increased from 205.6s to 268.8s.
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
session_id
prefix_id
phase = tool_wait
expected_tool_return_ms
next_ready_eta_ms
eta_uncertainty_ms
deadline_after_ready_ms
reuse_probability
recompute_cost_tokens
work_value / criticality
cancelable
```

Backend state:

```text
Several requests compete for KV memory and decode capacity.
```

Baseline:

```text
Admission uses present-time capacity and can admit new work even when a near-future
wave of replay-capable sessions is expected to return soon.
```

Harness-aware behavior:

```text
The controller delays new candidate work before it enters SGLang when harness
timing predicts that immediate admission would consume headroom needed by the
near-future replay wave. Every replay-capable session remains part of the
measured workload; there is no special target request.
```

Proof metric:

```text
Admission decision counts, delayed-candidate time, replay TTFT/lateness,
workload window, native eviction/copy events, and predicted replay demand versus
actual replay arrivals.
```

Run:

```bash
cd sglang_direct_kv
bash scripts/run_memory_admission_realistic.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Latest result:

```text
Run memory_admission_all_requests_v4_20260917_182215 compared no_prefetch
with controller_memory_admission across 16 replay requests. The controller
registered 16 future replay predictions, made 8 admission decisions, and
delayed 4 candidate requests by about 5.0s each. Scenario isolation passed:
targeted KV prefetch, predictive deadline priority, value-aware eviction, and
SGLang priority metadata were not active.

This was not yet a clear win. Total lateness was essentially unchanged
(312.3s -> 312.1s), but total TTFT worsened (67.9s -> 73.6s) and workload
window increased (95.1s -> 99.8s). The admission plumbing is working, but the
policy needs queue-depth/runtime awareness so it delays only work that would
actually block near-future replay.
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
