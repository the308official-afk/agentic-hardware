# Agentic KV Controller Modules

This package keeps the controller infrastructure portable across SGLang versions.
Experiment scripts should import these modules instead of embedding controller
policy directly in a one-off driver.

## Module Boundaries

- `models.py`: backend-neutral controller events, decisions, commands, and enums.
- `state_store.py`: session lifecycle state observed from harness requests.
- `policy.py`: portable controller policy that turns harness state into commands.
- `backend.py`: gateway/SGLang-facing adapters that lower controller commands.
- `harness_signal.py`: normalized harness signal schema and metadata builder.
- `timing_estimator.py`: lightweight timing estimates from observed lifecycle events.
- `modes.py`: canonical experiment mode names and mode-family predicates.
- `workload.py`: reusable synthetic/realistic workload profiles and runtime estimates.
- `runtime_calibration.py`: conservative, backend-neutral runtime classes and
  calibrated estimates for filler admission.
- `aiconfigurator_estimator.py`: optional AIConfigurator CLI adapter that can
  replace raw filler-runtime estimates with model/hardware/cache-aware
  predictions while falling back safely when the external tool is unavailable.
- `sjf.py`: short-filler admission scheduler. The internal mode names still use
  `controller_oracle_safe_sjf*` for compatibility, but reports describe these
  modes by behavior:
  `Controller Priority + Demotion + Short-Filler Admission`,
  `Controller Priority + Demotion + Balanced Filler Admission`,
  `Controller Priority + Demotion + Aggressive Filler Admission`, and
  `Controller Priority + Demotion + Max Filler Admission`.

`controller_priority_demotion_calibrated_admission` is the strict calibrated
variant. Reports call it `Controller Priority + Demotion + Calibrated Filler
Admission`. It still gives target replay high priority and demotes background
work, but it admits filler only when a conservative calibrated runtime estimate
fits before the target replay due time.

## Decision Quality Ledger

Controller short-filler admission experiments should be explainable after the
run. The scheduler therefore emits a stable `decision_id` plus the fields needed
to audit each admit/hold choice: target replay due time, filler runtime
estimate, available window, safety margin, expected finish time, and expected
overshoot.

The report builder turns those trace rows into:

- `controller_decision_quality.csv`
- `controller_decision_quality.json`
- `controller_decision_quality_summary.csv`
- `controller_decision_quality_summary.json`

For lightweight controller diagnostics, enable
`TRACE_CONTROLLER_COMPLETION_LINKAGE=1`. This adds one compact
`m27.controller_completion_linkage` row when a controller-admitted filler
request finishes. The row links `decision_id` to actual start/finish time,
runtime, estimation error, and overshoot versus the target replay due time. It
is much cheaper than full scheduler tracing and gives the ledger enough
evidence to classify admitted filler as good or bad.

Calibrated admission adds these fields to the same decision rows:

- `runtime_class`: stable key built from harness, workload profile, phase,
  request kind, tool-wait class, prompt-token bucket, and output-token bucket.
- `raw_estimated_runtime_ms`: the older prompt/output-size estimate.
- `calibrated_runtime_ms`: the conservative estimate used for admission.
- `calibration_source`: `unknown_class_fallback_floor` or a history-based
  source when `CONTROLLER_CALIBRATION_HISTORY_CSV` is supplied.

The ledger classifies admitted filler as good when it finished before the target
replay due time and bad when it overshot that due time. Held filler is harder to
judge because it never ran, so the ledger only calls out possible over-holds when
the hold lines up with visible idle gaps. This keeps the proof deterministic
without pretending we know an impossible counterfactual.

## Trace Profiles

Run scripts accept portable instrumentation knobs:

- `TRACE_PROFILE=full_debug`: default; keep scheduler, KV, runtime, GPU, and
  controller decision evidence on.
- `TRACE_PROFILE=controller_decision`: focus on controller decisions, request
  timing, and GPU samples while keeping KV-pool trace lighter.
- `TRACE_PROFILE=idle_gap`: same practical footprint as controller-decision
  tracing, intended for utilization-gap studies.
- `TRACE_PROFILE=deadline`: keep deadline/request timing but skip controller
  decision and GPU-detail output by default.
- `TRACE_PROFILE=minimal`: shortest trace; useful for smoke runs.

Individual switches can override the preset:

- `TRACE_CONTROLLER_DECISIONS=0|1`
- `TRACE_CONTROLLER_COMPLETION_LINKAGE=0|1`
- `TRACE_IDLE_GAP_AUDIT=0|1`
- `AGENTIC_KV_TRACE_SCHEDULER=0|1`
- `AGENTIC_KV_TRACE_KV_POOL=0|1`
- `AGENTIC_RUNTIME_TELEMETRY=0|1`
- `AGENTIC_KV_GPU_UTIL_SAMPLER=0|1`

Calibrated admission knobs:

- `CONTROLLER_CALIBRATED_DEFAULT_FLOOR_MS`: optional minimum estimate for
  calibrated classes with history. Default: `0`, so measured history can lower
  the estimate.
- `CONTROLLER_CALIBRATED_UNKNOWN_FLOOR_MS`: minimum estimate for unknown classes.
  Default: `5500`.
- `CONTROLLER_CALIBRATED_MIN_SAMPLES`: minimum historical samples before using
  history. Default: `3`.
- `CONTROLLER_CALIBRATED_QUANTILE`: conservative history statistic, such as
  `p90`, `p95`, `p99`, or `max`. Default: `p95`.
- `CONTROLLER_CALIBRATION_HISTORY_CSV`: optional CSV of prior linkage rows with
  `runtime_class` and `actual_runtime_ms`.

AIConfigurator estimator knobs:

- `CONTROLLER_RUNTIME_ESTIMATOR=aiconfigurator`: make SJF admission use the
  AIConfigurator adapter for runtime estimates.
- `CONTROLLER_AICONFIGURATOR_MODEL_PATH`: model path or Hugging Face ID to pass
  to AIConfigurator. If unset, the experiment model metadata is used.
- `CONTROLLER_AICONFIGURATOR_SYSTEM`: AIConfigurator system name, such as
  `h200_sxm`. This is intentionally explicit because project hardware labels
  like `ec2_a10g` are not necessarily AIConfigurator system names.
- `CONTROLLER_AICONFIGURATOR_BACKEND`: backend name. Default: `sglang`.
- `CONTROLLER_AICONFIGURATOR_ESTIMATE_MODE`: estimate mode. Default:
  `static_ctx`.
- `CONTROLLER_AICONFIGURATOR_TP_SIZE`: tensor parallel size. Default: `1`.
- `CONTROLLER_AICONFIGURATOR_PREFIX_TOKENS`: default cached-prefix token count
  when a trace row does not already expose one. Default: `0`.
- `CONTROLLER_AICONFIGURATOR_SAFETY_MULTIPLIER` and
  `CONTROLLER_AICONFIGURATOR_SAFETY_MARGIN_MS`: optional conservative padding
  applied to the AIConfigurator prediction before admission uses it.
- `CONTROLLER_AICONFIGURATOR_COMMAND_TEMPLATE`: optional full command template
  for CLI/API compatibility experiments. Available placeholders include
  `{bin}`, `{model_path}`, `{system}`, `{backend}`, `{estimate_mode}`, `{isl}`,
  `{osl}`, `{prefix}`, `{batch_size}`, and `{tp_size}`.

Before trusting this estimator in a controller run, compare it against measured
filler calibration rows:

```bash
python scripts/evaluate_aiconfigurator_timing.py \
  --input artifacts/results/reports/<label>/filler_timing_calibration.csv \
  --out artifacts/results/reports/<label>/aiconfigurator_timing_predictions.csv \
  --summary-out artifacts/results/reports/<label>/aiconfigurator_timing_summary.json \
  --model-path Qwen/Qwen2.5-Coder-7B-Instruct \
  --system h200_sxm
```

This produces the underestimation rate and worst underestimate. Those two
numbers matter most for safe filler admission because underestimates create bad
admits that can push target replay past its deadline.

## Effective Runtime Estimation

The next controller direction is to separate a filler's own service-time
estimate from its effective occupancy time under live SGLang queue, batching,
and KV-memory pressure. The design note is in
`docs/EFFECTIVE_RUNTIME_ESTIMATION.md`.

## Portability Rule

Keep SGLang-specific behavior behind adapters or gateway translation code. The
controller should reason in terms of agent/session facts such as phase, tool-wait
ETA, deadline, background safety, prompt size, and estimated backend occupancy.

## Experiment Driver Rule

`scripts/run_multi_harness_replay_driver.py` should orchestrate experiments. It
should not own reusable controller policies. When a mode, workload profile, or
admission strategy needs to be reused or tuned across harnesses, put it here.
