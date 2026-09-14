# Hint Benchmark Runbook

Minimal command list for running the Agentic Hint Benchmark Suite.

## Setup

Run from the top-level repo:

```bash
cd ~/agentic_hardware
```

If running from the Mac against EC2, SSH first:

```bash
cd /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware
bash -lc 'source aws/config.sh && ssh $(ssh_opts hintbench) "$EC2_USER@${SERVERS[0]}"'
cd ~/agentic_hardware
```

## Profile Runs

Baseline:

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

Scheduling only:

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

Cache only:

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

Pass-through only:

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

All request-boundary:

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

Runtime feedback placeholder:

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

## Scenario Group Runs

Smoke:

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

Full NAT coverage:

```bash
cd ~/agentic_hardware
RUN_ID="nat_full_nat_coverage_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios full_nat_coverage \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

## Individual Scenario Runs

No hints baseline:

```bash
cd ~/agentic_hardware
RUN_ID="nat_no_hints_baseline_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_no_hints_baseline \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

High priority:

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

Low priority:

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

Latency sensitivity:

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

Expected output length:

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

Expected interarrival time:

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

Planned request count:

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

Prefix reuse ID:

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

Cache-control TTL:

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

Ephemeral cache entry:

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

First-only cache control:

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

Cache namespace:

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

Provider QoS:

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

Priority-derived eviction intent:

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

Cache feedback metrics placeholder:

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

Cache pinning placeholder:

```bash
cd ~/agentic_hardware
RUN_ID="nat_cache_pinning_$(date +%Y%m%d_%H%M%S)"
.venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios nat_cache_pinning \
  --nat-dynamo-transport-capture \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

## Useful Combined Runs

Priority spread:

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

Workload shape:

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

Reuse and cache:

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

Boundary pass-through:

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

Maximum current NAT signal surface:

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

## Runner Checks

Dry run:

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

Fixture observations:

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

Validate existing observations:

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

## Optional Shortcuts

```bash
run_nat_profile() {
  PROFILE="$1"
  RUN_ID="nat_${PROFILE}_$(date +%Y%m%d_%H%M%S)"
  .venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
    --harness nemo_agent_toolkit \
    --knob-profile "$PROFILE" \
    --nat-dynamo-transport-capture \
    --run-id "$RUN_ID" \
    --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
}

run_nat_scenario() {
  SCENARIO="$1"
  RUN_ID="${SCENARIO}_$(date +%Y%m%d_%H%M%S)"
  .venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
    --harness nemo_agent_toolkit \
    --scenarios "$SCENARIO" \
    --nat-dynamo-transport-capture \
    --run-id "$RUN_ID" \
    --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
}
```

## Inspect Results

Set the run directory:

```bash
export RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
```

Scenario summary:

```bash
column -s, -t "$RUN_DIR/scenario_validation_summary.csv" | less -S
```

Signal support matrix:

```bash
column -s, -t "$RUN_DIR/hint_support_matrix.csv" | less -S
```

Unknown hints:

```bash
column -s, -t "$RUN_DIR/unknown_hints.csv" | less -S
```

Observed signal values:

```bash
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

## Key Output Files

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

## Notes

- `--nat-dynamo-transport-capture` is the real NAT request-boundary capture mode.
- `--dry-run` only checks benchmark setup.
- `--fixture-observations` only checks validation/reporting machinery.
- Injected or fixture hints do not prove native harness emission.
- Update `HINT_SIGNAL_FINDINGS.md` after any meaningful run.
