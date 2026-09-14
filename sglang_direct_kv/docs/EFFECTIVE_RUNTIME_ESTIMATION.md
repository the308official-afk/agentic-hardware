# Effective Runtime Estimation For Controller Admission

This note captures the next direction for improving controller filler admission.
The key lesson from the calibrated-controller experiments is that token count
alone is not enough to decide whether a filler request can safely run before a
target replay deadline.

## Core Problem

The same filler request can be fast in an empty backend and very slow in a
crowded SGLang backend.

The controller currently estimates whether filler work can fit before replay.
But the observed filler time is not only the filler request's own compute cost.
It can include:

- gateway/client queue delay
- SGLang scheduler queue delay
- batching delay
- running batch interference
- KV/cache hit or miss behavior
- KV memory pressure, eviction, offload, or load-back delay
- output length variance

So a filler with a small prompt can still block replay if it enters at the wrong
time or gets trapped behind other backend work.

## Two Estimates We Need

The controller should track two separate estimates.

### 1. Pure Service-Time Estimate

This asks:

> If this filler started running immediately in a clean backend slot, how much
> backend work would it need?

Useful inputs:

- prompt tokens
- expected output tokens
- cached prefix tokens
- model
- GPU type
- batch size assumption
- request type or workload phase

First-order formula:

```text
service_time_ms =
  prefill_cost(uncached_prompt_tokens)
  + decode_cost(expected_output_tokens)
  + fixed_overhead_ms
```

The prefill and decode coefficients should come from calibration runs on the
actual hardware when possible. AIConfigurator-style estimates may help as a
planning signal, but should be validated against measured rows before admission
uses them.

### 2. Effective Occupancy Estimate

This asks:

> If I admit this filler now, how long until it is really out of the replay
> path?

Useful inputs:

- current SGLang waiting queue length
- current running batch request count
- current running batch token count
- estimated remaining work of the running batch
- gateway pending and in-flight requests
- number and type of requests ahead of this filler
- KV pool occupancy
- cache hit/miss status
- offload/load-back/H2D activity
- replay ETA

First-order formula:

```text
effective_occupancy_ms =
  queue_wait_estimate_ms
  + batching_wait_estimate_ms
  + service_time_ms
  + memory_pressure_penalty_ms
  + uncertainty_margin_ms
```

This is the number the controller actually needs for safe admission.

## Why The Tuned Idle Override Failed

In the `p3_calibrated_controller_tuned_20260914_164932` run, idle override
admitted two filler requests. Both were bad admits.

The controller estimated roughly:

```text
2.4s and 6.4s
```

The observed runtimes were roughly:

```text
18.7s and 124.8s
```

This means the request-shape estimate was not enough. The surrounding backend
state dominated the true elapsed time.

## Proposed Implementation Path

### Phase 1: Split The Ledger Fields

Add explicit decision-ledger fields:

- `service_time_estimate_ms`
- `queue_wait_estimate_ms`
- `batching_wait_estimate_ms`
- `memory_pressure_penalty_ms`
- `uncertainty_margin_ms`
- `effective_occupancy_estimate_ms`
- `effective_estimate_source`

Then every admit or hold can be audited component by component.

### Phase 2: Fit A Service-Time Model

Run isolated or low-contention calibration cases and fit:

```text
prefill_ms_per_1k_tokens
decode_ms_per_token
fixed_overhead_ms
```

Separate fits by hardware profile, model, prompt bucket, output bucket, and
cache state when possible.

### Phase 3: Add Queue-State Penalty

At admission time, record:

- running batch request count
- running batch extend token count
- waiting queue length
- gateway pending and in-flight counts
- client queue wait if available

Estimate:

```text
queue_wait_estimate_ms =
  remaining_running_batch_ms
  + sum(estimated work ahead of this filler)
```

### Phase 4: Add Memory/KV Penalty

Record and model:

- KV pool occupancy
- cache hit/miss
- H2D/load-back events
- eviction/offload events

If memory pressure is high, either add a penalty:

```text
effective_occupancy_ms += memory_penalty_ms
```

or use a multiplier:

```text
effective_occupancy_ms *= pressure_multiplier
```

### Phase 5: Add Class-Specific Uncertainty

Track prediction error per runtime class.

Examples:

- stable `file_inspect`: small margin
- noisy `test_build`: large margin
- `slow_external` or high-variance classes: do not admit near replay

The controller should decide using:

```text
safe_estimate_ms =
  effective_occupancy_estimate_ms
  + class_uncertainty_margin_ms
```

## Admission Rule To Target

The controller should stop asking only:

```text
Can this filler's own token runtime fit before replay?
```

It should ask:

```text
Given the current queue, batch, memory pressure, and uncertainty,
can this filler really finish before replay?
```

That distinction is the next major step toward a deterministic, defensible
controller.

## Near-Term Experiment Recommendation

Before adding more aggressive admission policies, build an estimator-debug run
that logs both service-time and effective-occupancy estimates but keeps
admission conservative. The goal is to compare:

```text
service estimate vs actual service/elapsed time
effective estimate vs actual elapsed time
```

Only after the effective estimate is trustworthy should idle override be used
for deadline-critical controller modes.
