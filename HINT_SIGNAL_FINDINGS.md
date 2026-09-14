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

Current Claude status:

```text
native capture adapter: implemented
harness: claude_code
execution_mode: claude_native_capture
native CLI/client requirement: Claude Code must be installed and configured on EC2
current evidence status: native full-coverage run completed
current native client version: Claude Code 2.1.270
current native run_id: claude_native_full_coverage_20260914_203812
rescored artifact_dir: sglang_direct_kv/artifacts/results/hint_benchmark/claude_native_full_coverage_20260914_203812_rescored_wildcards
```

Important boundary: Claude rows below are now native-client probes, not
synthetic payload evidence. A fixture run may test benchmark plumbing, but it
must not be counted as proof that Claude Code organically emitted a signal.

Important boundary: the earlier all-harness SGLang replay experiments are still
valuable adapter/glue evidence. They show that Claude-shaped request metadata
can travel through our gateway/backend path. They do not, by themselves, prove
that the official Claude Code CLI emitted those fields organically.

Benchmark outputs include an `evidence_tier` column:

| Evidence Tier | How To Interpret It |
| --- | --- |
| `native_client_or_transport_capture` | Claimable native client/transport evidence. |
| `external_observed_file` | Validate the observed file provenance before citing as native evidence. |
| `fixture_plumbing_only` | Parser/report smoke test only. |
| `recipe_only` | Scenario recipe only; no observed emission. |

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
| Claude Code | provider QoS / service tier | no | session level | provider/model behavior | Client/session config attempted to set Claude `service_tier` | `claude_service_tier_auto`, `claude_service_tier_standard_only` | expected: `{"service_tier":"auto"}` or `{"service_tier":"standard_only"}` | native Claude client probe | `claude_native_full_coverage_20260914_203812` | Claude Code 2.1.270 did not expose `service_tier` through the tested env-var path. |
| Claude direct API | fast speed / QoS | yes | request level | provider/model behavior | Direct Anthropic API request supplies fast speed and beta header | `claude_api_fast_mode` | `{"speed":"fast"}` plus `anthropic-beta: fast-mode-2026-02-01`; response `{"usage":{"speed":"fast"}}` | documented direct API payload | `anthropic_api_payload_capture` | This proves the benchmark can represent the API capability; it is not native Claude Code CLI emission. |
| Claude Bedrock config | service tier priority | yes | provider level | provider/model behavior | Bedrock provider config supplies service-tier header | `claude_bedrock_service_tier_priority` | `{"_capture":{"headers":{"x-amzn-bedrock-service-tier":"priority"}}}` | provider-config payload | `anthropic_api_payload_capture` | This is Bedrock/provider-config evidence, not normal Anthropic API or Claude Code CLI evidence. |
| Claude Code | exact-prefix cache key behavior | optional not observed | cache entry level | prompt cache matching | Repeated identical client context may cause provider-derived exact-prefix reuse | `claude_repeated_session_prefix` | optional expected marker: `{"cache_key_policy":"provider_exact_prefix_hash"}` | provider-derived behavior probe | `claude_native_full_coverage_20260914_203812` | No literal cache-key field was exposed in the captured request body. |
| Claude Code | prompt `cache_control` | yes | content block level | prompt cache control | Claude Code prompt builder marked reusable system/context blocks | `claude_long_running_cache_session` | `system.*.cache_control.type="ephemeral"` and `messages.*.content.*.cache_control.type="ephemeral"` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | Original exact-index validator missed this; wildcard re-score found it. |
| Claude Code | cache TTL | no | cache entry level | cache retention | Long-retention client/session config attempted to request any explicit cache TTL | `claude_provider_retention_1h` | expected any present TTL | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | No explicit TTL field was observed; default TTL may be implicit/provider-managed. |
| Claude direct API | explicit cache TTL | yes | cache entry level | cache retention | Direct Anthropic API request puts TTL on a cache-control block | `claude_api_cache_ttl_1h` | `{"system":[{"cache_control":{"type":"ephemeral","ttl":"1h"}}]}` | documented direct API payload | `anthropic_api_payload_capture` | Represents long-retention/pin-like behavior through TTL, not a separate `cache_pinning=true` field. |
| Claude Code | tool block `cache_control` | optional not observed | content block level | reusable tool definitions | Tool-heavy request checks whether stable tool definitions receive cache markers | `claude_tool_heavy_request` | expected optional `tools.*.cache_control.type="ephemeral"` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | The captured request had tools, but no tool-level `cache_control`. |
| Claude Code | system block `cache_control` | yes | content block level | reusable system prompt | Claude Code prompt builder marked stable system blocks | `claude_stable_system_context` | `system.*.cache_control.type="ephemeral"` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | Observed at nonzero system indexes. |
| Claude Code | message block `cache_control` | yes | content block level | reusable message prefix | Claude Code prompt builder marked stable environment/context message blocks | `claude_stable_message_context` | `messages.*.content.*.cache_control.type="ephemeral"` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | Observed at nonzero message/content indexes. |
| Claude Code | prompt cache prewarm | partial | request level | cache warmup before real request | Prewarm-like client probe checks whether the real client can emit `max_tokens=0` with cache control | `claude_prewarm_like_probe` | observed cache control, but `max_tokens=64000` | native Claude client probe | `claude_native_full_coverage_20260914_203812_rescored_wildcards` | This showed cache-control, but did not become a true `max_tokens=0` prewarm request. |
| Claude direct API | prompt cache prewarm | yes | request level | cache warmup before real request | Direct Anthropic API request uses `max_tokens=0` with cache-controlled prefix | `claude_api_prewarm_max_tokens_zero` | `{"max_tokens":0,"system":[{"cache_control":{"type":"ephemeral"}}]}` | documented direct API payload | `anthropic_api_payload_capture` | Direct API capability only; native Claude Code CLI did not emit this shape in the previous run. |
| Claude Code | cache-hit feedback | no | runtime feedback level | post-execution usage metrics | Real provider/backend response must include cache counters | `claude_provider_cache_feedback_probe` | expected: `{"usage":{"cache_creation_input_tokens":...,"cache_read_input_tokens":...}}` | real provider response required | `claude_native_full_coverage_20260914_203812` | Local capture endpoint is not a real provider cache, so it cannot prove cache-hit feedback. |
| Claude direct API | cache-hit feedback shape | yes | runtime feedback level | post-execution usage metrics | Direct API response fixture includes documented cache usage counters | `claude_api_cache_feedback_fixture` | `{"usage":{"cache_creation_input_tokens":248,"cache_read_input_tokens":1800}}` | documented response payload | `anthropic_api_payload_capture` | Validates reporting/validation path. Real cache proof still requires real provider execution. |
| Claude Code | separate `cache_pinning=true` | no | cache entry level | cache retention behavior | Negative probe checks whether the real client emits literal pinning | `claude_cache_pinning_negative_probe` | no `cache_pinning` field observed | native Claude client negative probe | `claude_native_full_coverage_20260914_203812` | Negative probe passed; Claude documents TTL/cache-control behavior, not a separate pin flag. |

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
| `claude_service_tier` | `auto`, `standard_only`, `fast`, Bedrock `priority` | Claude `service_tier`, `speed`, `usage.speed`, `anthropic-beta`, Bedrock service-tier header | Runs native Claude client probes plus documented API/provider-config capability probes. | Claude | no |
| `claude_cache_control_location` | tools, system, messages, long-running context | Claude `cache_control` | Runs native Claude client scenarios that may cause prompt-builder cache markers. | Claude | no |
| `claude_cache_ttl` | default/provider-managed, explicit `1h` | Claude `cache_control.ttl` | Runs native Claude client long-retention probes and direct API TTL probes. | Claude | no |
| `claude_cache_feedback` | real provider response | `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens` | Requires a real provider/backend response path to observe cache feedback. | Claude | yes |

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
| Claude `qos_only` | Probe Claude Code native service-tier and fast-mode variants. | native service-tier probes plus `fastMode=true` probe | `service_tier`, `anthropic-beta` |
| Claude `cache_only` | Probe Claude Code native cache-control locations and TTL. | native cache-control probes, `ENABLE_PROMPT_CACHING_1H`, and `FORCE_PROMPT_CACHING_5M` | `cache_control`, optional `cache_control.ttl`, optional cache beta/header behavior |
| Claude `prewarm_only` | Probe whether Claude Code CLI emits max-token-zero prewarm. | native prewarm-like probe | `max_tokens=0`, `cache_control` |
| Claude `feedback_only` | Probe whether Claude Code native path can expose cache feedback. | native real-provider probe | `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens` |
| Claude `real_provider_feedback` | Run real Claude Code provider-response probes. | real provider cache-feedback and TTL probes | cache creation/read usage counters and TTL usage buckets when exposed |
| Claude `native_client_boundary` | Run Claude Code CLI-only probes. | native Claude Code scenarios | native Claude Code probe targets only |
| Claude `direct_api_capabilities` | Run direct Anthropic API capability probes. | `direct_anthropic_api_coverage` | `speed`, explicit TTL, `max_tokens=0`, cache feedback |
| Claude `bedrock_provider_config` | Run Bedrock provider-config probes. | `bedrock_config_coverage` | Bedrock service-tier header |
| Claude `all_request_boundary` | Run every Claude Code native-client boundary probe. | `full_claude_coverage` | native Claude Code probe targets only |
| Claude `all_signal_recipes` | List every Claude signal recipe. | `all_claude_signal_recipes` | native CLI, direct API, and provider-config recipes |

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
