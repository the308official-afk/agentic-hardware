# Hint Benchmarking Suite

This document is the source of truth for the hint benchmarking workstream.
The first implementation target is NeMo Agent Toolkit / NAT. Claude comes
later, after the benchmark shape is stable.

## Core Objective

Build a repeatable benchmark suite that shows which hints a harness can emit
and under what circumstances those hints appear.

For now, this suite is about harness hint emission. It is not trying to prove
SGLang performance improvement, and it is not primarily about converting hints
into SGLang fields. That lowering work can come later.

The benchmark should let us say, with evidence:

```text
When this NAT scenario runs, this NAT hint appears.
When this other NAT scenario runs, that hint does not appear.
Here is the raw emitted hint value captured at the harness boundary.
```

## Why This Matters

Today, it is easy to say that a harness "supports hints" without knowing what
that means in practice. This benchmark turns that vague statement into a
repeatable test.

The intended output is a clear report that product or infrastructure teams can
use to answer:

- what hints a harness can emit
- whether each hint is session-level, request-level, workflow-level, or
  provider-level
- whether the hint appears organically from workload behavior or only because
  a client/config supplied it
- how to reproduce each hint on demand
- which expected hints did not appear

## Initial Scope

Start with NAT only.

Do not start by benchmarking every harness. NAT is the best first target because
the signal table lists many explicit scheduling and cache-related hints for it.
Once NAT is solid, reuse the same suite structure for Claude, then the remaining
harnesses.

## Out Of Scope For The First Version

The first version should not focus on:

- SGLang performance benefit
- replay deadline improvement
- TTFT improvement
- full design-space experiments
- glue-script lowering into SGLang scheduler or cache fields
- proving whether SGLang acted on the hint

Those are later stages. The first stage only asks whether the harness can emit
the hint and what caused that emission.

## NAT Hint Inventory From The Deck

The starting NAT inventory comes from:

```text
presentation/Harness Signal Tables As-Is.pptx
```

The NAT-relevant rows are mainly on slide 5 and slide 10.

### Scheduling And Workload Hints

| Canonical area | NAT signal described in the deck | First benchmark expectation |
| --- | --- | --- |
| Soft priority | `priority` | Create high and low priority NAT scenarios and verify the emitted value. |
| Latency sensitivity | `latency_sensitivity` | Create a latency-sensitive scenario if NAT exposes this knob. |
| Provider speed/QoS | provider-dependent | Mark as provider/config supplied unless we confirm an organic NAT trigger. |
| Expected output length | `osl` | Create a scenario that supplies or estimates expected output length. |
| Expected interarrival time | `iat` | Create a scenario that supplies expected interarrival timing. |
| Predicted remaining calls | `total_requests` | Create a multi-step workflow scenario that exposes remaining call count. |
| Prefix/workflow reuse ID | `prefix_id` | Create repeated workflow/session scenarios that share a prefix ID. |

### KV And Cache Hints

| Canonical area | NAT signal described in the deck | First benchmark expectation |
| --- | --- | --- |
| Cache key/bucket | `prefix_id`, but not equivalent to a true cache key | Verify whether NAT emits prefix identity, while labeling it carefully. |
| Cache namespace/isolation | possible `nvext.cache_salt` | Create a tenant/session namespace scenario if NAT exposes this knob. |
| Cache TTL | `nvext.cache_control.ttl` | Create a cache-control scenario with TTL. |
| Cache entry type | experimental `type: "ephemeral"` | Create an ephemeral cache-control scenario if supported. |
| Eviction priority | priority-derived | Verify whether priority also appears as cache eviction priority metadata. |
| Cache-hit feedback | profiler/metrics | Capture metrics output when available, but keep this separate from request emission. |
| Cache pinning | experimental/version-bound | Treat as optional until a concrete NAT version path is verified. |

Important correction: the deck does not list `strict_priority` as a required NAT
hint. It appears for Dynamo. Do not add `strict_priority` to the required NAT
suite unless another NAT-specific source confirms it.

## Hint Source Classification

Every benchmark case must classify where the hint came from.

| Source class | Meaning | Example |
| --- | --- | --- |
| `organic` | The harness emitted the hint because the workload shape naturally triggered it. | A repeated workflow causes a reusable prefix marker. |
| `workflow_config` | The benchmark workflow explicitly configured the hint. | NAT workflow config marks a request as high priority. |
| `client_supplied` | The client/session supplied the hint before the harness processed the request. | A high-priority client sends priority intent. |
| `provider_config` | A provider setting caused or carried the hint. | Provider-dependent QoS setting. |
| `metrics_feedback` | The hint-like data appears after execution as metrics or usage feedback. | Cache-hit feedback from profiler metrics. |

This classification is required because not every hint is an independent
harness decision. Some hints are emitted because the benchmark configured the
workflow or client to carry that signal.

## Hint Injection Level

Every known or newly discovered hint must also record where it entered the
harness path. This is different from `scope`.

```text
scope = what the hint affects
injection_level = where the hint entered the system
```

For example, a priority hint may affect one request, so its scope is
`request`. But the value may have entered through workflow configuration, so
its injection level is `workflow_level`.

Use these injection levels:

| Injection level | Meaning |
| --- | --- |
| `provider_level` | Comes from provider or model configuration. |
| `client_level` | Comes from the client before the harness sees the task. |
| `session_level` | Attached to the whole session or conversation. |
| `task_level` | Attached to one agent task. |
| `workflow_level` | Attached to a workflow, workflow step, or workflow path. |
| `request_level` | Attached to one model request. |
| `prompt_block_level` | Attached to a specific prompt, message, system, or tool block. |
| `cache_entry_level` | Attached to a cache entry or cache retention policy. |
| `runtime_feedback_level` | Appears after execution as metrics, usage, or profiler feedback. |
| `unknown` | Observed by the benchmark, but not classified yet. |

This matters for future hints. If a future NAT or Claude version starts
emitting a new field, the benchmark should not ignore it. It should record the
raw field and mark it as needing classification:

```json
{
  "hint_id": "unknown",
  "raw_field": "new_hint_name",
  "source_class": "unknown",
  "injection_level": "unknown",
  "status": "needs_classification"
}
```

Then we inspect it and update the manifest with the correct source class and
injection level. This keeps the suite useful as harnesses evolve.

## Benchmark Evidence Model

Each benchmark scenario should capture a compact evidence record.

Minimum fields:

```json
{
  "scenario": "nat_priority_high",
  "harness": "nemo_agent_toolkit",
  "hint_id": "priority",
  "source_class": "workflow_config",
  "injection_level": "workflow_level",
  "scope": "request",
  "expected": true,
  "observed": true,
  "raw_emitted_value": {"priority": "high"},
  "notes": "Priority was supplied by the NAT workflow scenario."
}
```

The first version should stop at raw harness/boundary evidence. Later versions
can add normalized and SGLang-lowered fields.

## Initial NAT Scenarios

Start with a small set before expanding.

| Scenario | Purpose |
| --- | --- |
| `nat_no_hints_baseline` | Verify the default NAT path and capture which hints, if any, appear without special setup. |
| `nat_priority_high` | Verify that NAT can emit a high-priority request hint. |
| `nat_priority_low` | Verify that NAT can emit a background or low-priority request hint. |
| `nat_prefix_reuse_id` | Verify that NAT can attach a prefix/workflow reuse ID. |
| `nat_cache_control_ttl` | Verify that NAT can emit cache-control TTL metadata. |

After those pass, expand to:

| Scenario | Purpose |
| --- | --- |
| `nat_latency_sensitive` | Verify latency sensitivity metadata. |
| `nat_expected_output_length` | Verify `osl` or expected output length metadata. |
| `nat_expected_interarrival_time` | Verify `iat` metadata. |
| `nat_remaining_calls` | Verify `total_requests` or remaining-call metadata. |
| `nat_cache_namespace` | Verify cache namespace or `nvext.cache_salt` metadata. |
| `nat_cache_ephemeral` | Verify ephemeral cache entry metadata if this NAT version supports it. |
| `nat_cache_feedback_metrics` | Verify cache-hit feedback through metrics, not request metadata. |

## Report Shape

The benchmark report should be simple and readable.

Required summary table:

| Hint | Scenario | Source class | Scope | Expected? | Observed? | Example emitted value |
| --- | --- | --- | --- | --- | --- | --- |
| `priority` | `nat_priority_high` | `workflow_config` | request | yes | yes/no | raw value |
| `prefix_id` | `nat_prefix_reuse_id` | `workflow_config` or `organic` | workflow/session | yes | yes/no | raw value |
| `cache_ttl` | `nat_cache_control_ttl` | `workflow_config` | request/cache | yes | yes/no | raw value |

Optional report sections:

- raw evidence records
- failed expectations
- hints seen unexpectedly
- scenarios that require a provider-specific setup
- scenarios that are blocked by NAT version support

## Implementation Phases

### Phase 1: Manifest

Create a machine-readable NAT hint manifest from the deck.

Initial manifest path:

```text
sglang_direct_kv/configs/hint_benchmark/nat_hints.json
```

The manifest should list:

- hint ID
- display name
- canonical area
- raw NAT field
- scope
- expected source class
- default injection level
- required scenario name
- whether the first version treats it as required or optional

### Phase 2: Scenario Definitions

Create one synthetic scenario definition per NAT hint.

Initial scenario definition path:

```text
sglang_direct_kv/configs/hint_benchmark/nat_scenarios.json
```

Each scenario should describe:

- what workload or config it uses
- what hint should appear
- where the hint should appear
- whether the trigger is organic or configured
- where the hint is injected: session, task, workflow, request, provider, cache
  entry, prompt block, or runtime feedback

### Phase 3: Runner Skeleton

Add a runner that can execute NAT scenarios and write evidence records.

Initial runner path:

```text
sglang_direct_kv/scripts/run_hint_benchmark.py
```

The first runner supports:

```text
--harness nemo_agent_toolkit
--scenarios all
--dry-run
--out-dir <path>
```

`--dry-run` records the scenario recipes and expected evidence. It does not
claim that NAT actually emitted the hints. Real harness execution comes later.

Example:

```bash
cd sglang_direct_kv
python3 scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios smoke \
  --dry-run \
  --out-dir artifacts/results/hint_benchmark/nat_smoke
```

Expected dry-run outputs:

```text
run.json
scenario_records.jsonl
scenario_records.json
expected_hint_evidence.csv
```

The runner also supports a fixture smoke mode:

```bash
cd sglang_direct_kv
python3 scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios smoke \
  --fixture-observations \
  --out-dir artifacts/results/hint_benchmark/nat_fixture_smoke
```

Fixture observations are generated from scenario expectations. They prove the
benchmark selection, evidence writing, and validation pipeline. They do not
prove that real NAT emitted the hints.

### Phase 4: Evidence Validation

Add validators that compare expected and observed hints.

Initial validation outputs:

```text
hint_validation.csv
hint_validation.json
scenario_validation_summary.csv
unknown_hints.csv
```

The validator should produce:

- pass/fail per scenario
- missing expected hints
- unexpected hints
- raw examples

In dry-run mode, validation rows are marked `not_evaluated` because no real
harness evidence exists yet. Once a later runner supplies observed hint
evidence, the same validator can classify each expected hint as pass or fail.

Future unknown hints should be written to `unknown_hints.csv` with:

```text
status = needs_classification
source_class = unknown
injection_level = unknown
```

### Phase 5: NAT Smoke Run

Run the initial five NAT scenarios:

```text
nat_no_hints_baseline
nat_priority_high
nat_priority_low
nat_prefix_reuse_id
nat_cache_control_ttl
```

This smoke run should prove that the benchmark infrastructure works before we
add the rest of the NAT hints.

Run this on EC2, using `aws/config.sh` as the source of truth for the current
host:

```bash
cd /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware
source aws/config.sh
ssh $(ssh_opts hintbench) "$EC2_USER@${SERVERS[0]}"

cd ~/agentic_hardware
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios smoke \
  --nat-dynamo-transport-capture \
  --run-id nat_real_smoke_phase5_ec2 \
  --out-dir sglang_direct_kv/artifacts/results/hint_benchmark/nat_real_smoke_phase5_ec2
```

This capture mode uses NAT's `_DynamoTransport` and records the request body
after NAT injects its hints, but before the request would be sent to SGLang. It
does not require a live SGLang server, and it does not test SGLang lowering.

Current phase-5 EC2 result:

```text
run_id: nat_real_smoke_phase5_ec2
execution_mode: nat_dynamo_transport_capture
scenario_count: 5
validation_rows: 7
unknown_hint_rows: 0
result: all five smoke scenarios passed
```

Observed NAT smoke behavior:

- no-hints baseline emitted no benchmark hints
- high-priority workflow emitted `priority=100`
- low-priority/background workflow emitted `priority=2`
- repeated workflow emitted stable `prefix_id=nat_bench_shared_prefix_001`
- cache-control workflow emitted `nvext.cache_control.ttl="1s"`

NAT also emitted the standard `nvext.agent_hints` bundle when `_DynamoTransport`
was enabled: `latency_sensitivity`, `priority`, `osl`, `prefix_id`,
`total_requests`, and `iat`.

Important caveat: NAT warns that `nvext.cache_control` requires an SGLang build
newer than v0.5.9 with hierarchical cache enabled. This phase proves NAT sent
the field; it does not prove the backend acted on it.

If NAT is unavailable in a future environment, run fixture smoke first and mark
real NAT smoke as pending. Do not report fixture evidence as real NAT evidence.

### Phase 6: Full NAT Coverage

Add the remaining NAT scenarios from the deck and mark unsupported or
version-bound cases clearly.

Run this on EC2:

```bash
cd ~/agentic_hardware
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios full_nat_coverage \
  --nat-dynamo-transport-capture \
  --run-id nat_full_coverage_missing_paths_v2_ec2 \
  --out-dir sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2
```

Current phase-6 EC2 result:

```text
run_id: nat_full_coverage_missing_paths_v2_ec2
execution_mode: nat_dynamo_transport_capture
scenario_count: 16
validation_rows: 20
unknown_hint_rows: 0
local artifact copy: sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2
```

Observed in NAT `_DynamoTransport` capture:

| Hint | Scenario | Observed Value |
| --- | --- | --- |
| `priority` | `nat_priority_high` | `100` |
| `priority` | `nat_priority_low` | `2` |
| `prefix_id` | `nat_prefix_reuse_id` | `nat_bench_shared_prefix_001` |
| `cache_ttl` | `nat_cache_control_ttl` | `1s` |
| `latency_sensitivity` | `nat_latency_sensitive` | `100.0` |
| `expected_output_length` | `nat_expected_output_length` | `128` |
| `expected_interarrival_time` | `nat_expected_interarrival_time` | `750` |
| `predicted_remaining_calls` | `nat_remaining_calls` | `4` |
| `cache_namespace` | `nat_cache_namespace` | `nat_bench_tenant_a` |
| `cache_entry_type` | `nat_cache_ephemeral` | `ephemeral` |
| `priority` as eviction-priority intent | `nat_eviction_priority` | `100` |
| first-request cache-control type | `nat_cache_control_first_only` | request 1 emitted `ephemeral` |
| first-request cache-control TTL | `nat_cache_control_first_only` | request 1 emitted `1s`; request 2 omitted cache control |
| `provider_qos` | `nat_provider_qos` | `provider_specific_fast_or_priority` |

Optional-not-observed in this direct transport capture:

| Hint | Scenario | Why It Is Not A Failure |
| --- | --- | --- |
| `cache_hit_feedback` | `nat_cache_feedback_metrics` | This is runtime feedback after backend execution, not a request-boundary emission. |
| `cache_pinning` | `nat_cache_pinning` | This NAT version exposes ephemeral cache control, but no separate pinning field in the captured request. |

Important wording: `cache_namespace` and `provider_qos` are pass-through cases
in this benchmark. The client/provider metadata enters the request before NAT,
and the benchmark verifies that NAT preserves it at the outgoing boundary.
They are not organic NAT decisions.

Important wording for cache pinning: NAT 1.8.0 exposes
`CacheControlMode.FIRST_ONLY`, not a separate `cache_pinning=true` request
field. The suite therefore tests first-request cache-control behavior as the
nearest concrete NAT behavior, while leaving the separate `cache_pinning` field
as optional-not-observed.

Primary output files:

```text
hint_support_matrix.csv
scenario_validation_summary.csv
hint_validation.csv
observed_hint_evidence.jsonl
unknown_hints.csv
```

### Phase 7: Claude Extension

After NAT is stable, add Claude using the same manifest, scenario, runner, and
report structure.

Claude should not force us to rewrite the benchmark suite. It should only add a
new harness manifest and new scenario definitions.

## Success Criteria

The NAT-first benchmark succeeds when:

- every NAT hint from the deck is represented in the manifest
- every required NAT hint has at least one scenario
- the benchmark can reproduce the first required hints on demand
- the report clearly says which hints appeared, when they appeared, and why
- configured hints and organic hints are not confused
- unsupported or version-bound hints are labeled honestly

The benchmark should make future discussion more precise. Instead of saying
"NAT emits hints," we should be able to say:

```text
NAT emitted priority in the high-priority workflow scenario.
NAT emitted prefix_id in the repeated workflow scenario.
NAT did not emit cache_ttl unless cache-control TTL was configured.
```

The living findings table is:

```text
HINT_SIGNAL_FINDINGS.md
```

It should be updated after each meaningful benchmark run. It records the
signal, injection level, producing scenario, JSON shape, affected behavior,
evidence source, and benchmark knobs.

The executable NAT knob catalog is:

```text
sglang_direct_kv/configs/hint_benchmark/nat_knobs.json
```

The copy-paste runbook for operating the benchmark is:

```text
HINT_BENCHMARK_RUNBOOK.md
```

The runner supports named knob profiles with:

```bash
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture
```

## Later Work

After the benchmark suite is stable, future work can connect this to the rest
of the infrastructure:

- glue-script normalization
- SGLang-facing lowering
- replay deadline pressure experiments
- TTFT and cost-accounting reports
- production-like SWE-Bench Pro workloads
- multi-harness coverage

Keep those later layers separate so the hint benchmark remains easy to run and
easy to reason about.
