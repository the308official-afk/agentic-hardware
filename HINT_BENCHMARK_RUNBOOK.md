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

Helper functions:

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

run_nat_scenarios() {
  NAME="$1"
  shift
  RUN_ID="nat_${NAME}_$(date +%Y%m%d_%H%M%S)"
  .venvs/nat_py311/bin/python sglang_direct_kv/scripts/run_hint_benchmark.py \
    --harness nemo_agent_toolkit \
    --scenarios "$@" \
    --nat-dynamo-transport-capture \
    --run-id "$RUN_ID" \
    --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
}
```

## Profile Runs

```bash
run_nat_profile baseline
```

```bash
run_nat_profile scheduling_only
```

```bash
run_nat_profile cache_only
```

```bash
run_nat_profile passthrough_only
```

```bash
run_nat_profile all_request_boundary
```

```bash
run_nat_profile runtime_feedback
```

## Scenario Group Runs

```bash
run_nat_scenario smoke
```

```bash
run_nat_scenario full_nat_coverage
```

## Individual Scenario Runs

```bash
run_nat_scenario nat_no_hints_baseline
```

```bash
run_nat_scenario nat_priority_high
```

```bash
run_nat_scenario nat_priority_low
```

```bash
run_nat_scenario nat_latency_sensitive
```

```bash
run_nat_scenario nat_expected_output_length
```

```bash
run_nat_scenario nat_expected_interarrival_time
```

```bash
run_nat_scenario nat_remaining_calls
```

```bash
run_nat_scenario nat_prefix_reuse_id
```

```bash
run_nat_scenario nat_cache_control_ttl
```

```bash
run_nat_scenario nat_cache_ephemeral
```

```bash
run_nat_scenario nat_cache_control_first_only
```

```bash
run_nat_scenario nat_cache_namespace
```

```bash
run_nat_scenario nat_provider_qos
```

```bash
run_nat_scenario nat_eviction_priority
```

```bash
run_nat_scenario nat_cache_feedback_metrics
```

```bash
run_nat_scenario nat_cache_pinning
```

## Useful Combined Runs

```bash
run_nat_scenarios priority_spread \
  nat_priority_high \
  nat_priority_low \
  nat_latency_sensitive
```

```bash
run_nat_scenarios workload_shape \
  nat_expected_output_length \
  nat_expected_interarrival_time \
  nat_remaining_calls
```

```bash
run_nat_scenarios reuse_and_cache \
  nat_prefix_reuse_id \
  nat_cache_control_ttl \
  nat_cache_ephemeral \
  nat_cache_control_first_only
```

```bash
run_nat_scenarios boundary_passthrough \
  nat_cache_namespace \
  nat_provider_qos
```

```bash
run_nat_scenarios max_signal_surface full_nat_coverage
```

## Runner Checks

Dry run:

```bash
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
RUN_ID="nat_validate_existing_$(date +%Y%m%d_%H%M%S)"
python3 sglang_direct_kv/scripts/run_hint_benchmark.py \
  --harness nemo_agent_toolkit \
  --scenarios full_nat_coverage \
  --observed-jsonl path/to/observed_hint_evidence.jsonl \
  --run-id "$RUN_ID" \
  --out-dir "sglang_direct_kv/artifacts/results/hint_benchmark/$RUN_ID"
```

## Inspect Results

```bash
RUN_DIR="sglang_direct_kv/artifacts/results/hint_benchmark/<run_id>"
```

```bash
column -s, -t "$RUN_DIR/scenario_validation_summary.csv" | less -S
```

```bash
column -s, -t "$RUN_DIR/hint_support_matrix.csv" | less -S
```

```bash
column -s, -t "$RUN_DIR/unknown_hints.csv" | less -S
```

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
