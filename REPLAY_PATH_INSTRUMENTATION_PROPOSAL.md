# Replay Path Instrumentation Proposal

## Goal

For every timeline row, such as `G00` through `G31`, produce a direct evidence row that explains what happened when the replay request resumed.

The final answer for each replay should be simple:

```text
G04: scheduler wait dominated; KV was mostly already reusable; no host KV load observed.
G07: replay loaded KV from host to GPU; host load took X ms; TTFT was Y ms.
G12: prefix cache missed; replay recomputed N tokens.
```

The purpose is to replace vague labels like:

```text
scheduler/cache wait suspected
```

with stronger, instrumented labels:

```text
GPU-resident KV reuse
host-to-device KV load
recompute/prefill
scheduler wait
mixed path
```

## Why This Matters

The current report already shows useful evidence:

```text
tool wait deadline
prefetch attempt window
replay request window
TTFT
SGLang cache events
HiCache writes
device evictions
visible HtoD movement when present
```

But some rows still require inference. For example:

```text
long TTFT + no visible HtoD = scheduler/cache wait suspected
```

That is useful, but not strong enough for the final hardware argument.

The next step is to instrument SGLang deeply enough to say exactly which replay path happened.

## Main Question

For each replay request:

```text
Did the request reuse KV already resident in GPU memory?
Did it load KV from host memory back to GPU?
Did it recompute missing KV?
Did it wait in the scheduler before useful KV/cache work began?
Did more than one of these happen?
```

## Proposed Instrumentation

### 1. Request-ID Plumbing

Make sure the same request/session identity reaches every important SGLang path:

```text
HTTP/OpenAI request
SGLang scheduler
prefix/radix cache
HiCache
memory pool
load-back path
prefill/model-forward path
first-token path
```

Each event should carry:

```text
agent_session_id
request_id
phase: initial_turn / hint_prefetch / replay / pressure_filler
mode: no_prefetch / direct_prefetch
timeline_label: Gxx when available
case_id
```

Why this is needed:

```text
Without the same ID everywhere, we cannot confidently say:
"this host load belonged to G04 replay."
```

### 2. Scheduler Timing Ledger

Record timestamps for:

```text
request_received
entered_scheduler_queue
selected_for_prefill
prefill_started
decode_started
first_token_emitted
request_finished
```

Derived metrics:

```text
scheduler_wait_ms = prefill_started - request_received
prefill_to_first_token_ms = first_token_emitted - prefill_started
total_request_ms = request_finished - request_received
```

Why this is needed:

```text
A replay can be slow even if KV is already ready, because it waited behind other work.
This separates scheduler delay from KV delay.
```

### 3. Prefix/Radix Cache Counters

For every replay request, record:

```text
input_tokens
matched_prefix_tokens
unmatched_tokens
cache_hit_ratio
matched_node_ids
```

Example:

```text
G04 input_tokens=2409
matched_prefix_tokens=2409
unmatched_tokens=0
cache_hit_ratio=100%
```

Interpretation:

```text
Replay probably reused existing KV/prefix cache.
If TTFT is still high, the bottleneck is likely scheduler/waiting or decode-side delay.
```

### 4. HiCache Host Load-Back Counters

Inside SGLang's host-cache load path, record:

```text
host_cache_lookup_start
host_cache_lookup_end
host_hit_tokens
host_miss_tokens
load_back_start
load_back_end
loaded_token_count
loaded_block_count
loaded_bytes_estimate
```

Optional when available:

```text
loaded_block_ids
source_tier: host / CPU / CXL-like tier / peer GPU
destination_tier: GPU memory
```

Example:

```text
G07 host_hit_tokens=1536
load_back_duration_ms=14
loaded_block_count=12
```

Interpretation:

```text
Replay really loaded KV from host memory back to GPU.
```

### 5. Recompute / Prefill Counters

Record the amount of work SGLang had to compute because KV was not reusable:

```text
new_prefill_tokens
prefill_compute_start
prefill_compute_end
prefill_compute_ms
```

Example:

```text
G12 input_tokens=2400
matched_prefix_tokens=900
new_prefill_tokens=1500
prefill_compute_ms=1200
```

Interpretation:

```text
Replay recomputed missing KV for 1500 tokens.
```

### 6. Device Residency And Eviction State

Track the state of the matched KV blocks before replay:

```text
gpu_resident_tokens
host_resident_tokens
missing_tokens
evicted_from_gpu_time
loaded_to_gpu_time
protected_until
```

Possible states:

```text
gpu_resident
host_resident
missing
mixed
unknown
```

Why this is needed:

```text
Prefix cache hit does not always mean the KV is already in GPU memory.
It may mean the logical prefix exists, but physical pages may need load-back.
```

### 7. Per-Gap Replay Path Ledger

Build one machine-readable ledger row per timeline gap.

Example:

```json
{
  "gap": "G04",
  "mode": "direct_prefetch",
  "request_id": "m27_004_..._replay",
  "tool_wait_ms": 50,
  "prefetch_margin_ms": -244638,
  "scheduler_wait_ms": 930,
  "input_tokens": 2409,
  "matched_prefix_tokens": 2409,
  "gpu_resident_hit_tokens": 2409,
  "host_load_tokens": 0,
  "recomputed_tokens": 0,
  "ttft_ms": 4037,
  "final_path": "scheduler_wait_then_gpu_resident_reuse",
  "confidence": "medium"
}
```

This ledger should feed the HTML report directly.

## Confidence Labels

Every replay-path classification should include confidence:

```text
high confidence
  direct SGLang counters + matching CUDA/CUPTI HtoD event

medium confidence
  direct SGLang counters only

low confidence
  inferred mainly from TTFT/timeline shape
```

Why this helps:

```text
Managers can see which claims are proven directly and which are still inferred.
```

## Separate Scheduler Delay From KV Delay

Do not treat all TTFT as KV delay.

Break replay TTFT into:

```text
scheduler_wait_ms
kv_prepare_ms
prefill_compute_ms
host_load_ms
decode_to_first_token_ms
unknown_ms
```

This prevents the wrong conclusion.

Example:

```text
Bad interpretation:
G04 had high TTFT, so KV load was slow.

Better interpretation:
G04 had high TTFT, but host_load_tokens=0 and matched_prefix_tokens=2409.
The delay was likely scheduler/prefill/decode path delay, not HtoD KV load.
```

## Final Bottleneck Labels

Each `Gxx` row should get one slide-friendly label:

```text
scheduler dominated
host-load dominated
recompute dominated
GPU-resident cache hit but delayed
prefetch useful
prefetch late
prefetch wasted
mixed bottleneck
unknown / needs deeper trace
```

Example report table:

| Row | Final Path | Bottleneck | Confidence | Evidence |
| --- | --- | --- | --- | --- |
| G04 | GPU-resident reuse + scheduler wait | scheduler dominated | medium | matched_prefix=2409, host_load=0, TTFT=4037 ms |
| G07 | host KV load during replay | host-load dominated | high | host_load=1536 tokens + CUDA HtoD |
| G12 | partial prefix miss | recompute dominated | medium | new_prefill=1500 tokens |

## Counterfactual Hardware Estimate

For each late software prefetch, estimate what a deadline-aware hardware path might have done.

Use measured low-level movement time when available:

```text
observed software hint duration
measured host-load duration
tool wait deadline
available slack
```

Example:

```text
Observed:
tool_wait_ms = 100
software_prefetch_duration_ms = 800
actual_h2d_copy_ms = 12
prefetch finished late

Counterfactual:
if a deadline-aware DMA path had started the 12 ms H2D copy immediately,
it could have finished inside the 100 ms tool gap.
```

Why this is powerful:

```text
It separates policy from enforcement.
Software knew what to prefetch, but the normal software/SGLang path did not act predictably enough.
Hardware support could make the urgent movement cheaper and deadline-aware.
```

## Block-Level Tracking Strategy

Full block-level tracking can become noisy.

Start with compact block evidence:

```text
matched_node_id
block_count
token_count
gpu_resident_token_count
host_resident_token_count
missing_token_count
```

Only log exact block IDs for important cases:

```text
rows with host loads
rows with replay-side H2D
rows where prefetch was late
rows where replay had high TTFT
rows where classifier confidence is low
```

This keeps logs manageable while still giving deep proof where it matters.

## Cache State Before Replay

Right before replay becomes due, record:

```text
session_id
expected_reuse
gpu_resident_tokens
host_resident_tokens
missing_tokens
protected_tokens
last_eviction_time
last_load_time
```

Example:

```text
G04 before replay:
gpu_resident_tokens=2409
host_resident_tokens=0
missing_tokens=0
```

Interpretation:

```text
Replay was slow even though KV was resident.
That points away from host-device movement and toward scheduling or other runtime delay.
```

## CUDA/CUPTI Cross-Validation

Use SGLang counters as the main evidence source.

Use CUDA/CUPTI only for targeted validation:

```text
SGLang says host load happened.
CUDA profiler shows HtoD copy in the same window.
```

Why this is the recommended path:

```text
Full torch/CUDA profiling creates huge traces and distorts performance.
Targeted validation gives hardware-level credibility without drowning the experiment.
```

## Deterministic Validation Case

Before running big sweeps, create a small validation test that intentionally produces known outcomes.

Target cases:

```text
full GPU-resident reuse
host load-back
partial recompute
scheduler wait
prefetch late
prefetch useful
prefetch wasted
```

The classifier must label these correctly before we trust large experiments.

Example validation table:

| Case | Forced Condition | Expected Label |
| --- | --- | --- |
| A | KV stays in GPU | GPU-resident cache hit |
| B | KV offloaded to host | host-load dominated |
| C | prefix changed early | recompute dominated |
| D | many active requests ahead | scheduler dominated |
| E | hint starts after deadline | prefetch late |

## Recommended Milestones

### Milestone 29A: Request-ID Plumbing Audit

Confirm that replay request IDs flow through:

```text
scheduler
radix cache
HiCache
memory pool
load-back
prefill
first-token path
```

Output:

```text
request_id_coverage_report.json
```

### Milestone 29B: Scheduler Timing Ledger

Add exact scheduler timing events:

```text
request_received
queued
selected_for_prefill
prefill_started
decode_started
first_token_emitted
finished
```

Output:

```text
scheduler_wait_ms per Gxx
```

### Milestone 29C: Prefix/Radix Cache Token Counters

Add:

```text
input_tokens
matched_prefix_tokens
unmatched_tokens
cache_hit_ratio
matched_node_ids
```

Output:

```text
prefix_cache_evidence per Gxx
```

### Milestone 29D: HiCache Load-Back Counters

Add:

```text
host_hit_tokens
host_miss_tokens
loaded_token_count
loaded_block_count
load_back_ms
```

Output:

```text
host_load_evidence per Gxx
```

### Milestone 29E: Recompute / Prefill Counters

Add:

```text
new_prefill_tokens
prefill_compute_ms
```

Output:

```text
recompute_evidence per Gxx
```

### Milestone 29F: Cache State Before Replay

Add a checkpoint just before replay due:

```text
gpu_resident_tokens
host_resident_tokens
missing_tokens
protected_tokens
```

Output:

```text
pre_replay_cache_state per Gxx
```

### Milestone 29G: Confidence And Bottleneck Classifier

Replace vague labels with:

```text
final_path
bottleneck_label
confidence
evidence_summary
```

Output:

```text
replay_path_ledger.csv
```

### Milestone 29H: Counterfactual Hardware Estimate

For late prefetch rows, estimate:

```text
could deadline-aware hardware have completed the movement before replay due?
```

Output:

```text
hardware_counterfactual.csv
```

### Milestone 29I: Manager-Grade Report Update

Add these sections to `latest_master_report.html`:

```text
Replay Path Proof Table
Bottleneck Breakdown
Confidence Summary
Counterfactual Hardware Opportunity
Per-Gap Evidence Drilldown
```

## Final Report Shape

The final HTML should answer four questions clearly:

### 1. Did Prefetch Meet The Deadline?

Evidence:

```text
prefetch_margin_ms
deadline timeline
```

### 2. What Happened During Replay?

Evidence:

```text
scheduler_wait_ms
host_load_tokens
recomputed_tokens
gpu_resident_hit_tokens
TTFT
```

### 3. What Was The Bottleneck?

Evidence:

```text
bottleneck_label
confidence
evidence_summary
```

### 4. What Would Hardware Help With?

Evidence:

```text
software_prefetch_duration_ms
actual_copy_ms
deadline_miss_ms
counterfactual_success_possible
```

## Recommended Implementation Order

Do this in order:

```text
1. Request-ID plumbing audit
2. Scheduler timing ledger
3. Prefix/cache token counters
4. Host-load/recompute counters
5. Cache-state-before-replay checkpoint
6. Confidence + bottleneck classifier
7. Counterfactual hardware estimate
8. HTML report upgrade
9. Deterministic validation cases
10. Larger pressure sweeps
```

This gives the strongest evidence without jumping straight into noisy block-level tracing.

## Expected Outcome

After this instrumentation, every row should have an evidence-backed explanation:

```text
G00: direct prefetch was late; replay waited mostly in scheduler; no host KV load.
G01: no prefetch; replay loaded 1024 KV tokens from host.
G02: direct prefetch ran; no host KV existed to load; replay reused GPU-resident KV.
G03: prefix changed; replay recomputed 1536 tokens.
```

This is the strongest story for the hardware proposal:

```text
Software can decide which KV matters.
But without deadline-aware, priority-aware, residency-aware enforcement,
the normal runtime path can be late, unpredictable, or wasteful.
```
