# Hint Benchmark Runbook

Compact scenario runbook for the Agentic Hint Benchmark Suite.

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
| Request | Applies to one outgoing model request. |
| Task | Applies to one task or workflow run. |
| Session | Applies to a session, so many requests in that session may carry it. |
| Configuration | Comes from a client/profile setting, so any run using that setting may carry it. |
| Task / Request | A task setting causes the signal, and it appears on outgoing requests. |
| Session / Request | A session setting causes the signal, and it appears on outgoing requests. |
| Configuration / Request | A config setting causes the signal, and it appears on outgoing requests. |


## Claude Code

### Claude native baseline

**Plain Purpose:** Confirm Claude emits no benchmark hints when knobs are off.

**Signals Observed Today:**
- none

**Where Attached:** Request

**Evidence:** Native Claude Code request-boundary control case.

**Command:**

```bash
RUN_ID="claude_baseline_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile baseline \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### All Claude native request-boundary probes

**Plain Purpose:** Produce all Claude Code signals observed today in one run.

**Signals Observed Today:**
- `system.*.cache_control.type="ephemeral"`
- `messages.*.content.*.cache_control.type="ephemeral"`
- `cache_control.ttl="1h"`
- `anthropic-beta: fast-mode-2026-02-01`

**Where Attached:** Configuration / Request

**Evidence:** Native Claude Code request-boundary capture.

**Command:**

```bash
RUN_ID="claude_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile all_request_boundary \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Claude native QoS probes

**Plain Purpose:** Produce the Claude fast-mode header.

**Signals Observed Today:**
- `anthropic-beta: fast-mode-2026-02-01`

**Where Attached:** Configuration / Request

**Evidence:** Native Claude Code request-boundary capture.

**Command:**

```bash
RUN_ID="claude_qos_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile qos_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Claude native cache probes

**Plain Purpose:** Produce Claude prompt-cache markers and TTL.

**Signals Observed Today:**
- `system.*.cache_control.type="ephemeral"`
- `messages.*.content.*.cache_control.type="ephemeral"`
- `cache_control.ttl="1h"`

**Where Attached:** Configuration / Request

**Evidence:** Native Claude Code request-boundary capture.

**Command:**

```bash
RUN_ID="claude_cache_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile cache_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Claude native prewarm probe

**Plain Purpose:** Send a request that warms a cacheable prompt block.

**Signals Observed Today:**
- `system.*.cache_control.type="ephemeral"`

**Where Attached:** Request

**Evidence:** Native Claude Code request-boundary capture.

**Command:**

```bash
RUN_ID="claude_prewarm_only_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --knob-profile prewarm_only \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Claude native fast mode only

**Plain Purpose:** Produce only the Claude fast-mode header.

**Signals Observed Today:**
- `anthropic-beta: fast-mode-2026-02-01`

**Where Attached:** Configuration / Request

**Evidence:** Native Claude Code request-boundary capture.

**Command:**

```bash
RUN_ID="claude_fast_mode_setting_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_fast_mode_setting \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Claude native 1h cache TTL

**Plain Purpose:** Produce Claude cache control with a 1-hour TTL.

**Signals Observed Today:**
- `cache_control.type="ephemeral"`
- `cache_control.ttl="1h"`

**Where Attached:** Configuration / Request

**Evidence:** Native Claude Code request-boundary capture using `ENABLE_PROMPT_CACHING_1H=1`.

**Command:**

```bash
RUN_ID="claude_provider_retention_1h_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_provider_retention_1h \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Claude native 5m cache control

**Plain Purpose:** Produce Claude cache control with default 5-minute retention.

**Signals Observed Today:**
- `cache_control.type="ephemeral"`

**Where Attached:** Configuration / Request

**Evidence:** Native Claude Code request-boundary capture using `FORCE_PROMPT_CACHING_5M=1`.

**Command:**

```bash
RUN_ID="claude_provider_retention_5m_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness claude_code \
  --scenarios claude_provider_retention_5m \
  --claude-native-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

## NeMo Agent Toolkit / NAT

### NAT baseline

**Plain Purpose:** Confirm NAT emits no hints when knobs are off.

**Signals Observed Today:**
- none

**Where Attached:** Request

**Evidence:** Native NAT transport capture control case.

**Command:**

```bash
RUN_ID="nat_baseline_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile baseline \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### All NAT request-boundary signals

**Plain Purpose:** Produce every NAT signal we can observe today in one run.

**Signals Observed Today:**
- `priority`
- `latency_sensitivity`
- `osl`
- `iat`
- `total_requests`
- `prefix_id`
- `nvext.cache_control.ttl`
- `nvext.cache_control.type`
- `nvext.cache_salt`
- `provider.qos_tier`

**Where Attached:** Configuration / Request

**Evidence:** Native NAT `_DynamoTransport` request-boundary capture.

**Command:**

```bash
RUN_ID="nat_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Scheduling signals only

**Plain Purpose:** Produce NAT priority and request-planning signals.

**Signals Observed Today:**
- `priority`
- `latency_sensitivity`
- `osl`
- `iat`
- `total_requests`

**Where Attached:** Task / Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_scheduling_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile scheduling_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Cache signals only

**Plain Purpose:** Produce NAT cache reuse and cache-control signals.

**Signals Observed Today:**
- `prefix_id`
- `nvext.cache_control.ttl`
- `nvext.cache_control.type`
- `priority`

**Where Attached:** Configuration / Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_cache_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile cache_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Pass-through signals only

**Plain Purpose:** Preserve provider and session metadata through NAT.

**Signals Observed Today:**
- `nvext.cache_salt`
- `provider.qos_tier`

**Where Attached:** Session / Request

**Evidence:** Preserved through NAT transport; provider QoS is pass-through, not NAT-invented.

**Command:**

```bash
RUN_ID="nat_passthrough_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile passthrough_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Priority high

**Plain Purpose:** Produce a high-priority NAT request.

**Signals Observed Today:**
- `priority=100`

**Where Attached:** Task / Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_priority_high_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_high \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Priority low

**Plain Purpose:** Produce a low-priority NAT request.

**Signals Observed Today:**
- `priority=2`

**Where Attached:** Task / Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_priority_low_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_low \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Latency sensitive

**Plain Purpose:** Produce a latency-sensitive NAT request.

**Signals Observed Today:**
- `latency_sensitivity`
- `priority`

**Where Attached:** Task / Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_latency_sensitive_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_latency_sensitive \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Expected output length

**Plain Purpose:** Produce the expected output length hint.

**Signals Observed Today:**
- `osl`

**Where Attached:** Task / Request

**Evidence:** Native NAT transport capture from configured workload metadata.

**Command:**

```bash
RUN_ID="nat_expected_output_length_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_output_length \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Expected interarrival time

**Plain Purpose:** Produce the expected request spacing hint.

**Signals Observed Today:**
- `iat`

**Where Attached:** Task / Request

**Evidence:** Native NAT transport capture from configured workload metadata.

**Command:**

```bash
RUN_ID="nat_expected_interarrival_time_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_interarrival_time \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Planned request count

**Plain Purpose:** Produce the planned request count hint.

**Signals Observed Today:**
- `total_requests`

**Where Attached:** Task / Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_remaining_calls_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_remaining_calls \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Prefix reuse ID

**Plain Purpose:** Produce the reusable prefix/session ID hint.

**Signals Observed Today:**
- `prefix_id`

**Where Attached:** Session / Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_prefix_reuse_id_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_prefix_reuse_id \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Cache TTL

**Plain Purpose:** Produce a cache lifetime hint.

**Signals Observed Today:**
- `nvext.cache_control.ttl`

**Where Attached:** Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_cache_control_ttl_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_control_ttl \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Ephemeral cache entry

**Plain Purpose:** Produce an ephemeral cache-control hint.

**Signals Observed Today:**
- `nvext.cache_control.type="ephemeral"`

**Where Attached:** Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_cache_ephemeral_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_ephemeral \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### First-only cache control

**Plain Purpose:** Produce cache control only on the first repeated request.

**Signals Observed Today:**
- `nvext.cache_control`

**Where Attached:** Task / Request

**Evidence:** Native NAT transport capture.

**Command:**

```bash
RUN_ID="nat_cache_control_first_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_control_first_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Cache namespace

**Plain Purpose:** Produce a cache namespace hint.

**Signals Observed Today:**
- `nvext.cache_salt`

**Where Attached:** Session / Request

**Evidence:** Pass-through preserved by NAT transport.

**Command:**

```bash
RUN_ID="nat_cache_namespace_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_namespace \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Provider QoS pass-through

**Plain Purpose:** Preserve provider QoS metadata through NAT.

**Signals Observed Today:**
- `provider.qos_tier`

**Where Attached:** Configuration / Request

**Evidence:** Pass-through preserved by NAT transport; not NAT-invented.

**Command:**

```bash
RUN_ID="nat_provider_qos_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_provider_qos \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

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
