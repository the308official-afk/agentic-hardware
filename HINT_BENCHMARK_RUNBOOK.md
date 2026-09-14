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
<td>
<ul>
<li>none</li>
</ul>
</td>
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
<td>
<ul>
<li><code>priority</code></li>
<li><code>latency_sensitivity</code></li>
<li><code>osl</code></li>
<li><code>iat</code></li>
<li><code>total_requests</code></li>
<li><code>prefix_id</code></li>
<li><code>nvext.cache_control.ttl</code></li>
<li><code>nvext.cache_control.type</code></li>
<li><code>nvext.cache_salt</code></li>
<li><code>provider.qos_tier</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>priority</code></li>
<li><code>latency_sensitivity</code></li>
<li><code>osl</code></li>
<li><code>iat</code></li>
<li><code>total_requests</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>prefix_id</code></li>
<li><code>nvext.cache_control.ttl</code></li>
<li><code>nvext.cache_control.type</code></li>
<li><code>priority</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>nvext.cache_salt</code></li>
<li><code>provider.qos_tier</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>priority=100</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>priority=2</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>latency_sensitivity</code></li>
<li><code>priority</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>osl</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>iat</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>total_requests</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>prefix_id</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>nvext.cache_control.ttl</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>nvext.cache_control.type="ephemeral"</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>nvext.cache_control</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>nvext.cache_salt</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>provider.qos_tier</code></li>
</ul>
</td>
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
<td>
<ul>
<li>none</li>
</ul>
</td>
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
<td>
<ul>
<li><code>system.*.cache_control.type="ephemeral"</code></li>
<li><code>messages.*.content.*.cache_control.type="ephemeral"</code></li>
<li><code>cache_control.ttl="1h"</code></li>
<li><code>anthropic-beta: fast-mode-2026-02-01</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>anthropic-beta: fast-mode-2026-02-01</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>system.*.cache_control.type="ephemeral"</code></li>
<li><code>messages.*.content.*.cache_control.type="ephemeral"</code></li>
<li><code>cache_control.ttl="1h"</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>system.*.cache_control.type="ephemeral"</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>anthropic-beta: fast-mode-2026-02-01</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>cache_control.type="ephemeral"</code></li>
<li><code>cache_control.ttl="1h"</code></li>
</ul>
</td>
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
<td>
<ul>
<li><code>cache_control.type="ephemeral"</code></li>
</ul>
</td>
<td>content-block/cache-entry level</td>
<td>Native Claude Code request-boundary capture using `FORCE_PROMPT_CACHING_5M=1`.</td>
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
| Claude Code | native `max_tokens=0` prewarm | Not observed from Claude Code CLI in our capture. |
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
