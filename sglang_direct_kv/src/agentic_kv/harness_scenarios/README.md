# Harness Scenarios

This package is the modular scenario layer for harness-aware inference claims.
It lives inside `sglang_direct_kv` so real experiments reuse the existing
SGLang install, controller paths, trace instrumentation, and report builders.

The package has two roles:

1. Keep minimal synthetic scenarios in a small, deterministic format.
2. Map real claims onto the existing instrumented runner instead of creating a
   second backend path.

Useful entry points:

```bash
PYTHONPATH=src python3 scripts/run_harness_aware_scenarios.py
PYTHONPATH=src python3 scripts/run_harness_scenario_claim.py claim1_deadline_scheduling --dry-run
PYTHONPATH=src python3 scripts/run_harness_scenario_claim.py claim2_proactive_kv_hostmem --dry-run
```

The real claim runner delegates to:

```text
scripts/run_harness_deadline_pressure.sh
```

That delegation is deliberate. The scenario layer should describe and select
experiments; the backend project should remain the source of truth for how
SGLang is launched, instrumented, traced, and reported.
