# Hint Signal Findings

This document is the living findings table for harness hint experiments. It
records what each benchmark run taught us about when hints appear, where they
enter the harness path, what JSON shape they use, and what benchmark knobs can
expose or hide them.

The current completed harness is NeMo Agent Toolkit / NAT. Claude and other
harnesses should add rows to this same structure instead of creating separate
one-off notes.

## Evidence

Current NAT evidence:

```text
run_id: nat_full_coverage_missing_paths_v2_ec2
harness: nemo_agent_toolkit
execution_mode: nat_dynamo_transport_capture
scenario_count: 16
validation_rows: 20
unknown_hint_rows: 0
artifact_dir: sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2
```

Primary evidence files:

```text
sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2/hint_support_matrix.csv
sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2/scenario_validation_summary.csv
sglang_direct_kv/artifacts/results/hint_benchmark/nat_full_coverage_missing_paths_v2_ec2/observed_hint_evidence.jsonl
```

Important boundary: this run captures NAT request bodies after NAT's transport
has injected or preserved hint metadata, but before the request would go to
SGLang. It proves request-boundary emission/preservation. It does not prove
SGLang acted on those hints.

Current Claude evidence:

```text
run_id: claude_all_request_boundary_ec2_20260914_195103
harness: claude_code
execution_mode: claude_synthetic_boundary_capture
scenario_count: 12
validation_rows: 18
unknown_hint_rows: 0
artifact_dir: sglang_direct_kv/artifacts/results/hint_benchmark/claude_all_request_boundary_ec2_20260914_195103
```

Important boundary: this run uses synthetic Claude-style request and response
payloads. It proves that the benchmark can generate and validate Claude-style
hint surfaces. It does not yet prove that the Claude Code CLI organically
emitted those fields.

## Signal Findings

| Harness | Signal | Observed? | Injection level | Scope / affects | What produces it | Scenario ID | Example JSON shape | Native vs pass-through | Evidence source | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAT | `priority` high | yes | workflow level | request scheduling | Workflow marks request as high priority / latency sensitive | `nat_priority_high` | `{"nvext":{"agent_hints":{"priority":100}}}` | NAT-native emitted by `_DynamoTransport` from workflow context | `nat_dynamo_transport_capture` | Higher number means higher priority in this NAT/Dynamo path. |
| NAT | `priority` low | yes | workflow level | request scheduling | Workflow marks request as low/background priority | `nat_priority_low` | `{"nvext":{"agent_hints":{"priority":2}}}` | NAT-native emitted by `_DynamoTransport` from workflow context | `nat_dynamo_transport_capture` | This is configured workflow behavior, not semantic urgency inference. |
| NAT | `latency_sensitivity` | yes | workflow level | request scheduling | Workflow sets latency sensitivity | `nat_latency_sensitive` | `{"nvext":{"agent_hints":{"latency_sensitivity":100.0}}}` | NAT-native emitted by `_DynamoTransport` from workflow context | `nat_dynamo_transport_capture` | `priority` is derived from the same sensitivity value. |
| NAT | `osl` / expected output length | yes | workflow level | request resource estimate | Workflow config supplies expected output length | `nat_expected_output_length` | `{"nvext":{"agent_hints":{"osl":128}}}` | NAT-native emitted by `_DynamoTransport` from configured value | `nat_dynamo_transport_capture` | Current scenario supplies the value; it does not prove NAT estimates it organically. |
| NAT | `iat` / expected interarrival time | yes | workflow level | workflow/request stream timing | Workflow config supplies request cadence | `nat_expected_interarrival_time` | `{"nvext":{"agent_hints":{"iat":750}}}` | NAT-native emitted by `_DynamoTransport` from configured value | `nat_dynamo_transport_capture` | Also influences NAT's derived cache TTL when cache control is enabled. |
| NAT | `total_requests` | yes | workflow level | workflow/request stream planning | Multi-step workflow config supplies planned request count | `nat_remaining_calls` | `{"nvext":{"agent_hints":{"total_requests":4}}}` | NAT-native emitted by `_DynamoTransport` from configured value | `nat_dynamo_transport_capture` | In this path it is total planned requests, not dynamically decremented remaining calls. |
| NAT | `prefix_id` | yes | workflow level | workflow/session reuse identity | Repeated workflow/session runs under a stable prefix ID | `nat_prefix_reuse_id` | `{"nvext":{"agent_hints":{"prefix_id":"nat_bench_shared_prefix_001"}}}` | NAT-native emitted by `_DynamoTransport` from prefix context | `nat_dynamo_transport_capture` | Useful as a reuse identity, but not identical to a backend cache key. |
| NAT | `nvext.cache_control.ttl` | yes | workflow level | cache entry/request cache lifetime | Cache-control scenario enables NAT cache control | `nat_cache_control_ttl` | `{"nvext":{"cache_control":{"ttl":"1s"}}}` | NAT-native emitted by `_DynamoTransport` | `nat_dynamo_transport_capture` | TTL is computed from `total_requests * iat`, rounded to seconds/minutes. |
| NAT | `nvext.cache_control.type` | yes | cache entry level | cache entry behavior | Cache-control scenario uses NAT `CachePinType.EPHEMERAL` | `nat_cache_ephemeral` | `{"nvext":{"cache_control":{"type":"ephemeral"}}}` | NAT-native emitted by `_DynamoTransport` | `nat_dynamo_transport_capture` | NAT 1.8.0 exposes `ephemeral`, not a larger enum of pinning policies. |
| NAT | first-request cache control | yes | cache entry level | cache entry behavior | `CacheControlMode.FIRST_ONLY` with repeated prefix | `nat_cache_control_first_only` | request 1: `{"nvext":{"cache_control":{"type":"ephemeral","ttl":"1s"}}}`; request 2 omits `cache_control` | NAT-native behavior | `nat_dynamo_transport_capture` | Closest concrete NAT behavior to pin-like first-prefix retention. |
| NAT | `nvext.cache_salt` / cache namespace | yes | session level | session/request cache isolation | Client/session attaches cache namespace before NAT | `nat_cache_namespace` | `{"nvext":{"cache_salt":"nat_bench_tenant_a"}}` | pass-through preserved by NAT | `nat_dynamo_transport_capture` | This benchmark does not show NAT inventing the namespace. |
| NAT | provider QoS | yes | provider level | provider/model behavior | Provider metadata is attached before NAT request capture | `nat_provider_qos` | `{"provider":{"qos_tier":"provider_specific_fast_or_priority"}}` | pass-through preserved by NAT | `nat_dynamo_transport_capture` | Needs a concrete provider integration before treating this as provider-native. |
| NAT | priority-derived eviction intent | yes | workflow level | scheduling/cache-retention intent | Cacheable high-priority workflow exposes priority beside cache control | `nat_eviction_priority` | `{"nvext":{"agent_hints":{"priority":100}}}` | NAT-native priority reused as eviction intent evidence | `nat_dynamo_transport_capture` | No separate eviction-priority field was observed. |
| NAT | cache-hit feedback | no | runtime feedback level | post-execution metrics | Requires real backend execution and cache metrics | `nat_cache_feedback_metrics` | expected future shape: metrics/profiler row | not request-boundary metadata | not observed in direct capture | Needs real SGLang runtime metrics. |
| NAT | separate `cache_pinning=true` | no | cache entry level | cache retention behavior | Would require a NAT version/path with explicit pinning field | `nat_cache_pinning` | expected future shape: `{"cache_pinning":true}` | not observed in NAT 1.8.0 | not observed in direct capture | NAT 1.8.0 exposes FIRST_ONLY/ephemeral cache control instead. |
| Claude Code | provider QoS / service tier | synthetic yes | session level | provider/model behavior | Client/session config sets Claude `service_tier` | `claude_service_tier_auto`, `claude_service_tier_standard_only` | `{"service_tier":"auto"}` or `{"service_tier":"standard_only"}` | synthetic Claude boundary shape | `claude_synthetic_boundary_capture` | Needs native Claude Code capture before claiming organic CLI emission. |
| Claude Code | exact-prefix cache key behavior | synthetic yes | cache entry level | prompt cache matching | Repeated identical cached prefix; provider derives exact-prefix cache key | `claude_exact_prefix_cache_policy` | `{"cache_key_policy":"provider_exact_prefix_hash"}` | documented provider behavior represented in synthetic evidence | `claude_synthetic_boundary_capture` | Claude does not expose a client-supplied cache key field for this. |
| Claude Code | top-level `cache_control` | synthetic yes | request level | prompt cache control | Prompt caching enabled at request level | `claude_top_level_cache_control_5m`, `claude_top_level_cache_control_1h` | `{"cache_control":{"type":"ephemeral"}}` | synthetic Claude boundary shape | `claude_synthetic_boundary_capture` | 5-minute retention is default when no `ttl` is supplied. |
| Claude Code | 1-hour cache TTL | synthetic yes | cache entry level | cache retention | Long-running session/cache scenario sets explicit `ttl` | `claude_top_level_cache_control_1h`, `claude_provider_retention_1h` | `{"cache_control":{"type":"ephemeral","ttl":"1h"}}` | synthetic Claude boundary shape | `claude_synthetic_boundary_capture` | This is TTL retention, not a literal pin flag. |
| Claude Code | tool block `cache_control` | synthetic yes | content block level | reusable tool definitions | Stable tool definition receives cache marker | `claude_tools_cache_control` | `{"tools":[{"cache_control":{"type":"ephemeral"}}]}` | synthetic Claude boundary shape | `claude_synthetic_boundary_capture` | Needs native prompt-builder capture to prove Claude Code places it organically. |
| Claude Code | system block `cache_control` | synthetic yes | content block level | reusable system prompt | Stable system content receives cache marker | `claude_system_cache_control` | `{"system":[{"cache_control":{"type":"ephemeral"}}]}` | synthetic Claude boundary shape | `claude_synthetic_boundary_capture` | Needs native prompt-builder capture to prove Claude Code places it organically. |
| Claude Code | message block `cache_control` | synthetic yes | content block level | reusable message prefix | Stable message content receives cache marker | `claude_messages_cache_control` | `{"messages":[{"content":[{"cache_control":{"type":"ephemeral"}}]}]}` | synthetic Claude boundary shape | `claude_synthetic_boundary_capture` | Needs native prompt-builder capture to prove Claude Code places it organically. |
| Claude Code | prompt cache prewarm | synthetic yes | request level | cache warmup before real request | Prewarm scenario sends `max_tokens=0` with cache control | `claude_prewarm_cache` | `{"max_tokens":0,"cache_control":{"type":"ephemeral"}}` | synthetic Claude boundary shape | `claude_synthetic_boundary_capture` | This is a Claude API pattern, not a measured SGLang preload. |
| Claude Code | cache-hit feedback | synthetic yes | runtime feedback level | post-execution usage metrics | Synthetic response includes Claude usage cache counters | `claude_cache_usage_feedback` | `{"usage":{"cache_creation_input_tokens":1200,"cache_read_input_tokens":800}}` | synthetic response shape | `claude_synthetic_boundary_capture` | Real proof needs a provider/backend response capture. |
| Claude Code | separate `cache_pinning=true` | no | cache entry level | cache retention behavior | Not exposed as a literal Claude field; represented through provider-managed TTL | `claude_provider_retention_1h` | no `cache_pinning` field; uses `ttl:"1h"` | negative evidence in synthetic scenario | `claude_synthetic_boundary_capture` | Do not describe Claude 1-hour retention as true pinning. |

## Benchmark Knobs

The executable knob catalog is:

```text
sglang_direct_kv/configs/hint_benchmark/nat_knobs.json
```

The operational runbook with copy-paste commands is:

```text
HINT_BENCHMARK_RUNBOOK.md
```

Use knob profiles when the benchmark user wants to stress one signal family
without manually listing scenario IDs.

| Knob | Values | Signals it can expose | What it changes | Applies to | Backend required? |
| --- | --- | --- | --- | --- | --- |
| `priority_level` | `unset`, `low`, `high` | `priority`, `latency_sensitivity` | Sets NAT workflow latency/priority context. | NAT | no |
| `expected_output_length` | `unset`, `configured` | `osl` | Sets expected output length sent through NAT's Dynamo transport. | NAT | no |
| `request_cadence` | `unset`, `configured` | `iat`, `total_requests`, derived TTL | Sets expected request cadence and planned request count. | NAT | no |
| `prefix_reuse` | `false`, `true` | `prefix_id` | Runs related requests under the same prefix/workflow identity. | NAT | no |
| `cache_control_mode` | `disabled`, `always`, `first_only` | `nvext.cache_control.ttl`, `nvext.cache_control.type` | Controls whether cache control appears never, on every request, or only on the first request. | NAT | no |
| `cache_namespace` | `disabled`, `client_supplied` | `nvext.cache_salt` | Adds client/session cache namespace and verifies preservation. | NAT | no |
| `provider_qos` | `disabled`, `provider_supplied` | `provider.qos_tier` | Adds provider-level QoS metadata and verifies preservation. | NAT | no |
| `runtime_metrics` | `disabled`, `real_backend_required` | `cache_hit_feedback` | Runs a real backend path and inspects post-execution metrics. | NAT future | yes |
| `claude_service_tier` | `auto`, `standard_only` | Claude `service_tier` | Selects provider service tier in synthetic Claude requests. | Claude synthetic | no |
| `claude_cache_control_location` | top-level, tools, system, messages | Claude `cache_control` | Places cache markers at different Claude prompt locations. | Claude synthetic | no |
| `claude_cache_ttl` | default 5m, explicit 1h | Claude `cache_control.ttl` | Exercises default and long-retention cache behavior. | Claude synthetic | no |
| `claude_cache_feedback` | synthetic response | `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens` | Exercises Claude-style cache usage feedback fields. | Claude synthetic | no |

## Knob Profiles

| Profile | Purpose | Scenario selection | Expected visibility |
| --- | --- | --- | --- |
| `baseline` | Hide intentional hints. | `nat_no_hints_baseline` | none |
| `scheduling_only` | Expose scheduling/workload hints. | priority, sensitivity, `osl`, `iat`, `total_requests` scenarios | `priority`, `latency_sensitivity`, `osl`, `iat`, `total_requests` |
| `cache_only` | Expose prefix and cache-control hints. | prefix, TTL, ephemeral, FIRST_ONLY, eviction-intent scenarios | `prefix_id`, `nvext.cache_control.ttl`, `nvext.cache_control.type`, `priority` |
| `passthrough_only` | Expose client/session and provider pass-through. | namespace and QoS scenarios | `nvext.cache_salt`, `provider.qos_tier` |
| `all_request_boundary` | Show everything visible without SGLang. | `full_nat_coverage` | all request-boundary observed signals |
| `runtime_feedback` | Reserved for cache-hit metrics. | `nat_cache_feedback_metrics` | `cache_hit_feedback` after backend run |
| Claude `baseline` | Hide intentional hints. | `claude_no_hints_baseline` | none |
| Claude `qos_only` | Expose service-tier variants. | service-tier scenarios | `service_tier` |
| Claude `cache_only` | Expose cache-control locations and TTL. | cache-control scenarios | `cache_control`, `cache_control.ttl` |
| Claude `prewarm_only` | Expose max-token-zero cache prewarm. | prewarm scenario | `max_tokens=0`, `cache_control` |
| Claude `feedback_only` | Expose cache usage feedback shape. | usage feedback scenario | `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens` |
| Claude `all_request_boundary` | Show every Claude synthetic boundary scenario. | `full_claude_coverage` | all synthetic Claude boundary signals |

Example:

```bash
cd ~/agentic_hardware
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture \
  --run-id nat_all_request_boundary \
  --out-dir sglang_direct_kv/artifacts/results/hint_benchmark/nat_all_request_boundary
```

## Naming Rules

Use these labels consistently:

- `NAT-native emitted`: NAT itself added the field during its transport path.
- `pass-through preserved`: the client/provider supplied the field before NAT,
  and NAT preserved it.
- `runtime feedback`: the value appears after real backend execution, not in
  the outgoing request body.
- `optional-not-observed`: the manifest tracks the hint, but this benchmark
  mode did not observe a concrete emitted field.

These distinctions prevent us from overclaiming as we extend the suite to
Claude and the other harnesses.

Claude source docs:

- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- https://docs.anthropic.com/en/api/messages
