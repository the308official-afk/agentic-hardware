# Hint Benchmark Runbook

This runbook gives copy-pasteable commands for the Agentic Hint Benchmark
Suite. It is meant for handoff: another engineer should be able to choose the
behavior they want to stress, run one command, and inspect the benchmark output.

The current completed harness is NeMo Agent Toolkit / NAT. Claude Code and
other harnesses should follow the same runbook shape as adapters are added.

## What The Runner Does

The runner answers four practical questions:

1. Which hint appeared?
2. What benchmark scenario caused it to appear?
3. At what level did it appear: provider, session, workflow, request, cache
   entry, or runtime feedback?
4. Was it natively emitted by the harness, passed through from the client, or
   only simulated/injected for runner testing?

Important rule:

Injected or fixture-generated hints are useful for testing the benchmark
machinery, but they do not prove native harness support. Only real harness
capture mode counts as native emission evidence.

## Main Files

```text
HINT_BENCHMARKING_SUITE.md
HINT_SIGNAL_FINDINGS.md
HINT_BENCHMARK_RUNBOOK.md
sglang_direct_kv/configs/hint_benchmark/nat_hints.json
sglang_direct_kv/configs/hint_benchmark/nat_scenarios.json
sglang_direct_kv/configs/hint_benchmark/nat_knobs.json
sglang_direct_kv/scripts/run_hint_benchmark.py
```

## Output Files

Each run writes an artifact directory under:

```text
sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>
```

The most useful files are:

```text
run.json
knob_profile.json
scenario_records.jsonl
observed_hint_evidence.jsonl
expected_hint_evidence.jsonl
hint_validation.csv
scenario_validation_summary.csv
hint_support_matrix.csv
unknown_hints.csv
```

Use `hint_support_matrix.csv` for a quick coverage view. Use
`observed_hint_evidence.jsonl` when you need the raw JSON payload evidence.

## Local Command Shape

Run this from the top-level repository directory:

```bash
cd /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware

RUN_ID="nat_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

## EC2 Command Shape

Use `aws/config.sh` as the source of truth for the EC2 host.

```bash
cd /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware

bash -lc '
source aws/config.sh
ssh $(ssh_opts hintbench) "$EC2_USER@${SERVERS[0]}" "
  cd $REMOTE_PROJECT_DIR &&
  RUN_ID=\"nat_all_request_boundary_\$(date +%Y%m%d_%H%M%S)\" &&
  .venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
    --harness nemo_agent_toolkit \
    --knob-profile all_request_boundary \
    --nat-dynamo-transport-capture \
    --run-id \"\$RUN_ID\" \
    --out-dir \"sglang_direct_kv/artifacts/results/hint_benchmark/\$RUN_ID\"
"
'
```

Download a run from EC2:

```bash
cd /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware

bash -lc '
source aws/config.sh
RUN_ID="nat_all_request_boundary_YYYYMMDD_HHMMSS"
rsync -av \
  "$EC2_USER@${SERVERS[0]}:$REMOTE_PROJECT_DIR/sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID/" \
  "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID/" \
  -e "ssh $(ssh_opts hintbench)"
'
```

Replace `nat_all_request_boundary_YYYYMMDD_HHMMSS` with the actual run ID.

## Normal Profile Runs

These are the main commands most users should run. They use
`--nat-dynamo-transport-capture`, which captures actual NAT request payloads at
the NAT/Dynamo boundary without needing to send traffic to SGLang.

### Baseline: No Intentional Hints

Use this to confirm what NAT emits without special benchmark configuration.

```bash
cd ~/agentic_hardware

RUN_ID="nat_baseline_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile baseline \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected visibility: no intentional scheduling or cache hints.

### Scheduling Only

Use this to expose priority and workload-shape hints without cache-control
scenarios.

```bash
cd ~/agentic_hardware

RUN_ID="nat_scheduling_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile scheduling_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected visibility:

```text
priority
latency_sensitivity
osl
iat
total_requests
```

### Cache Only

Use this to expose prefix reuse and cache-control hints.

```bash
cd ~/agentic_hardware

RUN_ID="nat_cache_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile cache_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected visibility:

```text
prefix_id
nvext.cache_control.ttl
nvext.cache_control.type
priority as eviction-priority intent
FIRST_ONLY cache-control behavior
```

### Pass-Through Only

Use this to verify metadata supplied before NAT is preserved at the outgoing
request boundary.

```bash
cd ~/agentic_hardware

RUN_ID="nat_passthrough_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile passthrough_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected visibility:

```text
nvext.cache_salt
provider.qos_tier
```

These are pass-through checks, not proof that NAT invented those fields.

### All Request-Boundary Hints

Use this when you want the broadest NAT request-boundary coverage in one run.

```bash
cd ~/agentic_hardware

RUN_ID="nat_all_request_boundary_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected visibility: all currently observable NAT request-boundary signals.

### Runtime Feedback Placeholder

Use this profile only after a real backend runtime-feedback adapter is added.
It is reserved for cache-hit or post-execution metrics.

```bash
cd ~/agentic_hardware

RUN_ID="nat_runtime_feedback_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile runtime_feedback \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected current status: not enough evidence for true runtime feedback unless
the run is connected to a real backend metrics path.

## Single-Signal Scenario Runs

Use these when debugging one hint or explaining exactly what causes that hint
to appear.

### High Priority

```bash
cd ~/agentic_hardware

RUN_ID="nat_priority_high_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_high \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.agent_hints.priority=100`.

### Low Priority

```bash
cd ~/agentic_hardware

RUN_ID="nat_priority_low_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_low \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.agent_hints.priority=2`.

### Latency Sensitivity

```bash
cd ~/agentic_hardware

RUN_ID="nat_latency_sensitive_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_latency_sensitive \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.agent_hints.latency_sensitivity=100.0`.

### Expected Output Length

```bash
cd ~/agentic_hardware

RUN_ID="nat_expected_output_length_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_output_length \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.agent_hints.osl=128`.

### Expected Interarrival Time

```bash
cd ~/agentic_hardware

RUN_ID="nat_expected_interarrival_time_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_interarrival_time \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.agent_hints.iat=750`.

### Planned Request Count

```bash
cd ~/agentic_hardware

RUN_ID="nat_remaining_calls_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_remaining_calls \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.agent_hints.total_requests=4`.

### Prefix Reuse ID

```bash
cd ~/agentic_hardware

RUN_ID="nat_prefix_reuse_id_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_prefix_reuse_id \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.agent_hints.prefix_id=nat_bench_shared_prefix_001`.

### Cache TTL

```bash
cd ~/agentic_hardware

RUN_ID="nat_cache_control_ttl_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_control_ttl \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.cache_control.ttl=1s`.

### Ephemeral Cache Entry

```bash
cd ~/agentic_hardware

RUN_ID="nat_cache_ephemeral_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_ephemeral \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.cache_control.type=ephemeral`.

### Cache Namespace

```bash
cd ~/agentic_hardware

RUN_ID="nat_cache_namespace_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_namespace \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.cache_salt=nat_bench_tenant_a`.

This is a client/session pass-through signal in the current benchmark.

### Provider QoS

```bash
cd ~/agentic_hardware

RUN_ID="nat_provider_qos_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_provider_qos \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `provider.qos_tier=provider_specific_fast_or_priority`.

This is a provider/client pass-through signal in the current benchmark.

### Priority-Derived Eviction Intent

```bash
cd ~/agentic_hardware

RUN_ID="nat_eviction_priority_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_eviction_priority \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signal: `nvext.agent_hints.priority=100` beside cache-control
metadata. No separate eviction-priority field was observed in NAT 1.8.0.

### First-Only Cache Control

```bash
cd ~/agentic_hardware

RUN_ID="nat_cache_control_first_only_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_control_first_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected behavior:

```text
request 1: nvext.cache_control.type=ephemeral
request 1: nvext.cache_control.ttl=1s
request 2: no nvext.cache_control.ttl
```

This is the closest concrete NAT 1.8.0 behavior to a pin-like first-prefix
retention pattern.

### Cache-Hit Feedback Placeholder

```bash
cd ~/agentic_hardware

RUN_ID="nat_cache_feedback_metrics_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_feedback_metrics \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected current interpretation: this scenario documents a future runtime
feedback need. A request-boundary capture cannot prove cache-hit feedback,
because cache-hit feedback happens after backend execution.

## Scenario Group Runs

### Smoke Group

Use this for a quick sanity check across the first NAT scenarios.

```bash
cd ~/agentic_hardware

RUN_ID="nat_smoke_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios smoke \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected scenarios:

```text
nat_no_hints_baseline
nat_priority_high
nat_priority_low
nat_prefix_reuse_id
nat_cache_control_ttl
```

### Full NAT Coverage Group

This is equivalent to the `all_request_boundary` profile, but uses the scenario
group name directly.

```bash
cd ~/agentic_hardware

RUN_ID="nat_full_coverage_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios full_nat_coverage \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

## Runner Machinery Checks

These modes are useful when changing the benchmark runner itself.

### Dry Run

Use this to verify scenario selection and expected evidence without producing
observed harness evidence.

```bash
cd ~/agentic_hardware

RUN_ID="nat_dry_run_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile scheduling_only \
  --dry-run \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected interpretation: this checks benchmark setup only. It does not prove
NAT emitted anything.

### Fixture Observations

Use this to test validation/reporting with simulated observations.

```bash
cd ~/agentic_hardware

RUN_ID="nat_fixture_observations_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile cache_only \
  --fixture-observations \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected interpretation: useful for runner regression tests, but not native
harness evidence.

### Validate An Existing Observation File

Use this when another tool captured observations and you want the benchmark
runner to validate them.

```bash
cd ~/agentic_hardware

RUN_ID="nat_validate_existing_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios full_nat_coverage \
  --observed-jsonl path/to/observed_hint_evidence.jsonl \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

## Stress Patterns

These examples combine existing scenarios to stress specific behavior. They are
still request-boundary runs unless otherwise stated.

### Priority Spread

Use this to prove the benchmark can emit more than one priority value.

```bash
cd ~/agentic_hardware

RUN_ID="nat_priority_spread_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_priority_high nat_priority_low nat_latency_sensitive \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

### Workload Shape Metadata

Use this to expose the hints that describe expected request size and stream
shape.

```bash
cd ~/agentic_hardware

RUN_ID="nat_workload_shape_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_expected_output_length nat_expected_interarrival_time nat_remaining_calls \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signals: `osl`, `iat`, and `total_requests`.

### Reusable Prefix And Cache Control

Use this to show workflow/session reuse identity together with cache-control
metadata.

```bash
cd ~/agentic_hardware

RUN_ID="nat_reuse_and_cache_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_prefix_reuse_id nat_cache_control_ttl nat_cache_ephemeral nat_cache_control_first_only \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signals: `prefix_id`, cache TTL, cache entry type, and FIRST_ONLY
behavior.

### Boundary Pass-Through Stress

Use this when you specifically want glue scripts or later adapters to see
client/provider-level fields at the request boundary.

```bash
cd ~/agentic_hardware

RUN_ID="nat_boundary_passthrough_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_namespace nat_provider_qos \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

Expected signals: `nvext.cache_salt` and `provider.qos_tier`.

### Maximum Current NAT Signal Surface

Use this as the main handoff command when someone asks, "Show me every NAT hint
the current benchmark knows how to expose."

```bash
cd ~/agentic_hardware

RUN_ID="nat_max_signal_surface_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --knob-profile all_request_boundary \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

## Inspect Results

Show scenario pass/fail summary:

```bash
RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
column -s, -t "$RUN_DIR/scenario_validation_summary.csv" | less -S
```

Show signal support matrix:

```bash
RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
column -s, -t "$RUN_DIR/hint_support_matrix.csv" | less -S
```

Show unknown hints:

```bash
RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
column -s, -t "$RUN_DIR/unknown_hints.csv" | less -S
```

Inspect raw observed payload evidence:

```bash
export RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
python3 - <<'PY' | less
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_DIR"]) / "observed_hint_evidence.jsonl"
for line in path.read_text().splitlines():
    print(json.dumps(json.loads(line), indent=2, sort_keys=True))
PY
```

For JSONL, this can be easier:

```bash
export RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_DIR"]) / "observed_hint_evidence.jsonl"
for line in path.read_text().splitlines():
    row = json.loads(line)
    print(row.get("scenario_id"), row.get("hint_id"), row.get("observed_value"))
PY
```

Replace `<run_id>` before running the inspection commands.

## How To Read Validation Status

| Status | Meaning |
| --- | --- |
| `pass` | The expected hint was observed with the expected value or shape. |
| `fail` | A required expected hint was missing or had the wrong value. |
| `optional_missing` | The manifest tracks the hint, but this run mode did not have enough evidence to require it. |
| `unknown_hint` | The runner saw a raw field that is not yet cataloged in the manifest. |

## How To Choose The Right Run

| Goal | Run this |
| --- | --- |
| See whether the benchmark is wired correctly | `--dry-run` |
| Test reporting without NAT installed | `--fixture-observations` |
| Prove NAT emitted request-boundary fields | `--nat-dynamo-transport-capture` |
| Show no-hint control behavior | `--knob-profile baseline` |
| Show priority/workload hints | `--knob-profile scheduling_only` |
| Show cache hints | `--knob-profile cache_only` |
| Show provider/session pass-through | `--knob-profile passthrough_only` |
| Show all current request-boundary hints | `--knob-profile all_request_boundary` |
| Study cache hits after backend execution | future `runtime_feedback` backend mode |

## What To Update After A Meaningful Run

After a run changes our understanding, update:

```text
HINT_SIGNAL_FINDINGS.md
```

At minimum, record:

- run ID
- harness version
- execution mode
- signal observed or not observed
- injection level
- scenario that produced it
- example JSON shape
- whether it was native, pass-through, fixture, or injected
- caveats

Keep the benchmark honest: if a signal was not observed, mark it as not
observed or optional missing. Do not turn a fixture result into a native
harness claim.
