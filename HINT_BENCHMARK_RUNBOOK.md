# Hint Benchmark Runbook

Compact command table for the Agentic Hint Benchmark Suite.

Run these commands on the EC2 machine from the repo root:

```bash
cd ~/agentic_hardware
```

From the Mac, SSH with the configured EC2 host:

```bash
cd /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware
bash -lc 'source aws/config.sh && ssh $(ssh_opts hintbench) "$EC2_USER@${SERVERS[0]}"'
cd ~/agentic_hardware
```

## NeMo Agent Toolkit / NAT

<table>
<thead>
<tr>
<th>Run</th>
<th>Command</th>
<th>Signals Observed Today</th>
<th>Attachment Level</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>NAT baseline</td>
<td>

```bash
RUN_ID="nat_baseline_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile baseline \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>No intentional NAT hint fields expected</td>
<td>request level</td>
<td>Native NAT transport capture control case.</td>
</tr>
<tr>
<td>All NAT request-boundary signals</td>
<td>

```bash
RUN_ID="nat_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`priority`, `latency_sensitivity`, `osl`, `iat`, `total_requests`, `prefix_id`, `nvext.cache_control.ttl`, `nvext.cache_control.type`, first-only cache control, `nvext.cache_salt`, provider QoS pass-through</td>
<td>workflow, session, request/cache-entry, provider pass-through</td>
<td>Native NAT `_DynamoTransport` request-boundary capture.</td>
</tr>
<tr>
<td>Scheduling signals only</td>
<td>

```bash
RUN_ID="nat_scheduling_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile scheduling_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`priority`, `latency_sensitivity`, `osl`, `iat`, `total_requests`</td>
<td>workflow/request-stream level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache signals only</td>
<td>

```bash
RUN_ID="nat_cache_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile cache_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`prefix_id`, `nvext.cache_control.ttl`, `nvext.cache_control.type`, first-only cache control, priority-derived eviction intent</td>
<td>workflow and cache-entry level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Pass-through signals only</td>
<td>

```bash
RUN_ID="nat_passthrough_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile passthrough_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`nvext.cache_salt`, provider QoS metadata pass-through</td>
<td>session/provider level</td>
<td>Preserved through NAT transport; provider QoS is pass-through, not NAT-invented.</td>
</tr>
<tr>
<td>Priority high</td>
<td>

```bash
RUN_ID="nat_priority_high_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_high \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`priority=100`</td>
<td>workflow level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Priority low</td>
<td>

```bash
RUN_ID="nat_priority_low_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_low \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`priority=2`</td>
<td>workflow level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Latency sensitive</td>
<td>

```bash
RUN_ID="nat_latency_sensitive_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_latency_sensitive \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`latency_sensitivity`, derived `priority`</td>
<td>workflow level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Expected output length</td>
<td>

```bash
RUN_ID="nat_expected_output_length_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_output_length \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`osl`</td>
<td>workflow/request estimate level</td>
<td>Native NAT transport capture from configured workload metadata.</td>
</tr>
<tr>
<td>Expected interarrival time</td>
<td>

```bash
RUN_ID="nat_expected_interarrival_time_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_interarrival_time \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`iat`</td>
<td>workflow/request-stream level</td>
<td>Native NAT transport capture from configured workload metadata.</td>
</tr>
<tr>
<td>Planned request count</td>
<td>

```bash
RUN_ID="nat_remaining_calls_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_remaining_calls \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`total_requests`</td>
<td>workflow/request-stream level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Prefix reuse ID</td>
<td>

```bash
RUN_ID="nat_prefix_reuse_id_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_prefix_reuse_id \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`prefix_id`</td>
<td>workflow/session reuse level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache TTL</td>
<td>

```bash
RUN_ID="nat_cache_control_ttl_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_control_ttl \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`nvext.cache_control.ttl`</td>
<td>cache-entry/request level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Ephemeral cache entry</td>
<td>

```bash
RUN_ID="nat_cache_ephemeral_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_ephemeral \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`nvext.cache_control.type="ephemeral"`</td>
<td>cache-entry level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>First-only cache control</td>
<td>

```bash
RUN_ID="nat_cache_control_first_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_control_first_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>request 1 has `nvext.cache_control`; request 2 omits it</td>
<td>cache-entry/request sequence level</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache namespace</td>
<td>

```bash
RUN_ID="nat_cache_namespace_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_namespace \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`nvext.cache_salt`</td>
<td>session/request namespace level</td>
<td>Pass-through preserved by NAT transport.</td>
</tr>
<tr>
<td>Provider QoS pass-through</td>
<td>

```bash
RUN_ID="nat_provider_qos_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_provider_qos \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`provider.qos_tier`</td>
<td>provider level</td>
<td>Pass-through preserved by NAT transport; not NAT-invented.</td>
</tr>
</tbody>
</table>

## Claude Code

<table>
<thead>
<tr>
<th>Run</th>
<th>Command</th>
<th>Signals Observed Today</th>
<th>Attachment Level</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>Claude native baseline</td>
<td>

```bash
RUN_ID="claude_baseline_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile baseline \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>No intentional provider QoS or cache-control fields expected</td>
<td>request level</td>
<td>Native Claude Code request-boundary control case.</td>
</tr>
<tr>
<td>All Claude native request-boundary probes</td>
<td>

```bash
RUN_ID="claude_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile all_request_boundary \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`cache_control.type="ephemeral"` on system/message blocks, `cache_control.ttl="1h"` with 1h knob, fast-mode marker in `anthropic-beta`; not observed natively: `service_tier`, literal cache key, tool-level cache control, true `max_tokens=0` prewarm</td>
<td>session, request, content-block, cache-entry</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native QoS probes</td>
<td>

```bash
RUN_ID="claude_qos_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile qos_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>fast-mode marker in `anthropic-beta`; not observed natively: `service_tier=auto`, `service_tier=standard_only`</td>
<td>session/provider request header</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native cache probes</td>
<td>

```bash
RUN_ID="claude_cache_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile cache_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`cache_control.type="ephemeral"` on system/message blocks; `cache_control.ttl="1h"` with `ENABLE_PROMPT_CACHING_1H=1`; 5m knob shows cache control but no literal `ttl` field</td>
<td>content-block and cache-entry level</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native prewarm probe</td>
<td>

```bash
RUN_ID="claude_prewarm_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile prewarm_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`cache_control.type="ephemeral"` observed; true native `max_tokens=0` not observed</td>
<td>request and content-block level</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native fast mode only</td>
<td>

```bash
RUN_ID="claude_fast_mode_setting_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_fast_mode_setting \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`anthropic-beta` contains `fast-mode-2026-02-01`</td>
<td>session/provider request header</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native 1h cache TTL</td>
<td>

```bash
RUN_ID="claude_provider_retention_1h_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_provider_retention_1h \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`cache_control.type="ephemeral"`, `cache_control.ttl="1h"`</td>
<td>content-block/cache-entry level</td>
<td>Native Claude Code request-boundary capture using `ENABLE_PROMPT_CACHING_1H=1`.</td>
</tr>
<tr>
<td>Claude native 5m cache control</td>
<td>

```bash
RUN_ID="claude_provider_retention_5m_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_provider_retention_5m \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`cache_control.type="ephemeral"`; no literal `ttl` field observed</td>
<td>content-block/cache-entry level</td>
<td>Native Claude Code request-boundary capture using `FORCE_PROMPT_CACHING_5M=1`.</td>
</tr>
<tr>
<td>Claude real-provider feedback</td>
<td>

```bash
RUN_ID="claude_real_provider_feedback_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile real_provider_feedback \
  --claude-real-provider-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>blocked today on EC2: Claude CLI returned `Not logged in`; cache usage counters stayed zero</td>
<td>runtime feedback level</td>
<td>Implemented, but needs logged-in Claude provider execution before it can show cache read/write usage.</td>
</tr>
<tr>
<td>Direct Anthropic API capability coverage</td>
<td>

```bash
RUN_ID="claude_direct_api_capabilities_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile direct_api_capabilities \
  --anthropic-api-payload-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>direct API `speed="fast"`, explicit `cache_control.ttl="1h"`, `max_tokens=0`, cache feedback response shape</td>
<td>request, cache-entry, runtime feedback</td>
<td>Documented direct API payloads only; not proof that Claude Code CLI emitted these organically.</td>
</tr>
<tr>
<td>Bedrock provider-config service tier</td>
<td>

```bash
RUN_ID="claude_bedrock_service_tier_priority_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile bedrock_provider_config \
  --anthropic-api-payload-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

</td>
<td>`x-amzn-bedrock-service-tier="priority"`</td>
<td>provider header level</td>
<td>Provider-config payload evidence; not normal Claude Code CLI native emission.</td>
</tr>
</tbody>
</table>

## Missing Or Blocked Today

| Harness | Signal | Current State |
| --- | --- | --- |
| NAT | cache-hit runtime feedback | Not request-boundary metadata; needs backend/runtime cache metrics. |
| NAT | literal `cache_pinning=true` | Not observed in NAT 1.8.0; NAT exposes ephemeral/first-only cache control instead. |
| Claude Code | native `service_tier=auto` / `standard_only` body field | Not observed through tested Claude Code env-var path. |
| Claude Code | literal cache key | Not observed; Claude appears to use provider-derived exact-prefix matching. |
| Claude Code | tool-level `cache_control` | Not observed in the tested tool-heavy request. |
| Claude Code | native `max_tokens=0` prewarm | Not observed; direct API recipe supports it, Claude Code CLI did not emit it in our capture. |
| Claude Code | real cache-hit usage counters | Implemented runner path, but EC2 Claude CLI is not logged in today. |

## Inspect Results

```bash
export RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
column -s, -t "$RUN_DIR/scenario_validation_summary.csv" | less -S
column -s, -t "$RUN_DIR/hint_support_matrix.csv" | less -S
column -s, -t "$RUN_DIR/unknown_hints.csv" | less -S
```

Key files:

```text
run.json
knob_profile.json
scenario_records.jsonl
observed_hint_evidence.jsonl
expected_hint_evidence.csv
hint_validation.csv
scenario_validation_summary.csv
hint_support_matrix.csv
unknown_hints.csv
```
