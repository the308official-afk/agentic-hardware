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

## Where Attached Legend

| Value | Meaning |
| --- | --- |
| Request | Attached to one model request. |
| Workflow | Comes from the harness workflow for this task or path. |
| Workflow stream | Describes a planned stream of requests, such as cadence or count. |
| Prompt block | Attached to one system, message, tool, or cacheable prompt block. |
| Header | Sent as a request header, outside the JSON body. |
| Session/request | Comes from session metadata or request metadata. |
| Provider metadata | Provider-specific metadata preserved at the boundary. |
| First request only | Appears only on the first request in a repeated sequence. |

## NeMo Agent Toolkit / NAT

<table>
<thead>
<tr>
<th>Run</th>
<th>Plain Purpose</th>
<th>Command</th>
<th>Signals Observed Today</th>
<th>Where Attached</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>NAT baseline</td>
<td>Confirm NAT emits no hints when knobs are off.</td>
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
<td>Request</td>
<td>Native NAT transport capture control case.</td>
</tr>
<tr>
<td>All NAT request-boundary signals</td>
<td>Produce every NAT signal we can observe today in one run.</td>
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
<td>Workflow / Session/request / Provider metadata</td>
<td>Native NAT `_DynamoTransport` request-boundary capture.</td>
</tr>
<tr>
<td>Scheduling signals only</td>
<td>Produce NAT priority and request-planning signals.</td>
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
<td>Workflow stream</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache signals only</td>
<td>Produce NAT cache reuse and cache-control signals.</td>
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
<td>Workflow / Request</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Pass-through signals only</td>
<td>Preserve provider and session metadata through NAT.</td>
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
<td>Session/request / Provider metadata</td>
<td>Preserved through NAT transport; provider QoS is pass-through, not NAT-invented.</td>
</tr>
<tr>
<td>Priority high</td>
<td>Produce a high-priority NAT request.</td>
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
<td>Workflow</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Priority low</td>
<td>Produce a low-priority NAT request.</td>
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
<td>Workflow</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Latency sensitive</td>
<td>Produce a latency-sensitive NAT request.</td>
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
<td>Workflow</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Expected output length</td>
<td>Produce the expected output length hint.</td>
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
<td>Workflow stream</td>
<td>Native NAT transport capture from configured workload metadata.</td>
</tr>
<tr>
<td>Expected interarrival time</td>
<td>Produce the expected request spacing hint.</td>
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
<td>Workflow stream</td>
<td>Native NAT transport capture from configured workload metadata.</td>
</tr>
<tr>
<td>Planned request count</td>
<td>Produce the planned request count hint.</td>
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
<td>Workflow stream</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Prefix reuse ID</td>
<td>Produce the reusable prefix/session ID hint.</td>
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
<td>Workflow / Session/request</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache TTL</td>
<td>Produce a cache lifetime hint.</td>
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
<td>Request</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Ephemeral cache entry</td>
<td>Produce an ephemeral cache-control hint.</td>
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
<td>Request</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>First-only cache control</td>
<td>Produce cache control only on the first repeated request.</td>
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
<td>First request only</td>
<td>Native NAT transport capture.</td>
</tr>
<tr>
<td>Cache namespace</td>
<td>Produce a cache namespace hint.</td>
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
<td>Session/request</td>
<td>Pass-through preserved by NAT transport.</td>
</tr>
<tr>
<td>Provider QoS pass-through</td>
<td>Preserve provider QoS metadata through NAT.</td>
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
<td>Provider metadata</td>
<td>Pass-through preserved by NAT transport; not NAT-invented.</td>
</tr>
</tbody>
</table>

## Claude Code

<table>
<thead>
<tr>
<th>Run</th>
<th>Plain Purpose</th>
<th>Command</th>
<th>Signals Observed Today</th>
<th>Where Attached</th>
<th>Evidence</th>
</tr>
</thead>
<tbody>
<tr>
<td>Claude native baseline</td>
<td>Confirm Claude emits no benchmark hints when knobs are off.</td>
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
<td>Request</td>
<td>Native Claude Code request-boundary control case.</td>
</tr>
<tr>
<td>All Claude native request-boundary probes</td>
<td>Produce all Claude Code signals observed today in one run.</td>
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
<td>Header / Prompt block</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native QoS probes</td>
<td>Produce the Claude fast-mode header.</td>
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
<td>Header</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native cache probes</td>
<td>Produce Claude prompt-cache markers and TTL.</td>
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
<td>Prompt block</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native prewarm probe</td>
<td>Send a request that warms a cacheable prompt block.</td>
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
<td>Request / Prompt block</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native fast mode only</td>
<td>Produce only the Claude fast-mode header.</td>
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
<td>Header</td>
<td>Native Claude Code request-boundary capture.</td>
</tr>
<tr>
<td>Claude native 1h cache TTL</td>
<td>Produce Claude cache control with a 1-hour TTL.</td>
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
<td>Prompt block</td>
<td>Native Claude Code request-boundary capture using `ENABLE_PROMPT_CACHING_1H=1`.</td>
</tr>
<tr>
<td>Claude native 5m cache control</td>
<td>Produce Claude cache control with default 5-minute retention.</td>
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
<td>Prompt block</td>
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
