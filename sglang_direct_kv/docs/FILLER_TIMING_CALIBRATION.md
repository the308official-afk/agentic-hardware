# Filler Timing Calibration

This calibration path measures how long filler/background requests take before
the controller tries to use those timings for admission decisions.

The goal is to avoid guessing that a filler request will finish before a target
replay deadline. We first collect timing rows, fit a simple model, then use the
model output to choose conservative safety margins for controller admission.

## Flow

1. Run a small filler timing sweep across concurrency levels.
2. Extract one CSV row per filler request.
3. Fit an interpretable timing model.
4. Inspect underestimation error, especially worst and p95 underestimation.
5. Use the recommended safety margin in a later controller experiment.

Optional: compare the measured rows against an external AIConfigurator estimate
before using that estimate inside the controller.

## Run

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARDWARE_PROFILE=ec2_a10g \
HARNESSES=hatcher \
CONCURRENCY_LEVELS="1 2 4 8" \
SAMPLES_PER_CONCURRENCY=8 \
TOOL_WAIT_MS=800 \
TASK_REPLAY_STEPS=2 \
TOOL_WAIT_PROFILE=realistic_agentic_mix \
AGENTIC_WORKLOAD_PROFILE=realistic_agentic_mix \
REPORT_LABEL="filler_timing_calibration_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_filler_timing_calibration_sweep.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

## Outputs

The run writes artifacts under:

```text
artifacts/results/reports/<REPORT_LABEL>/
```

Important files:

- `filler_timing_calibration.csv`: raw filler timing features and outcomes.
- `filler_timing_model.json`: fitted coefficients and validation summary.
- `filler_timing_model_summary.csv`: compact model quality summary.
- `filler_timing_model_by_concurrency.csv`: error broken down by concurrency.
- `filler_timing_model_predictions.csv`: validation predictions versus actuals.
- `aiconfigurator_timing_predictions.csv`: optional AIConfigurator predictions
  versus actuals.
- `aiconfigurator_timing_summary.json`: optional AIConfigurator error summary.

## AIConfigurator Probe

AIConfigurator is optional and intentionally stays behind an adapter. Use this
probe before trusting it in a controller run:

```bash
CONTROLLER_AICONFIGURATOR_SYSTEM=h200_sxm \
python scripts/evaluate_aiconfigurator_timing.py \
  --input artifacts/results/reports/<REPORT_LABEL>/filler_timing_calibration.csv \
  --out artifacts/results/reports/<REPORT_LABEL>/aiconfigurator_timing_predictions.csv \
  --summary-out artifacts/results/reports/<REPORT_LABEL>/aiconfigurator_timing_summary.json \
  --model-path Qwen/Qwen2.5-Coder-7B-Instruct
```

To use it in a later SJF controller run:

```bash
CONTROLLER_RUNTIME_ESTIMATOR=aiconfigurator \
CONTROLLER_AICONFIGURATOR_SYSTEM=h200_sxm \
CONTROLLER_AICONFIGURATOR_MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

If the CLI is missing, unsupported, or its output cannot be parsed, the adapter
falls back to the conservative calibrated estimator and records that fallback in
`calibration_source`.

## What To Look At

The controller should care most about underestimation, not average error.

If the model predicts `4s` but the filler really takes `12s`, the controller may
admit work that blocks target replay. The key fields are:

- `underestimate_count`
- `underestimate_rate`
- `worst_underestimate_ms`
- `recommended_p90_safety_margin_ms`
- `recommended_p95_safety_margin_ms`
- `recommended_max_safety_margin_ms`

Use p95 or max underestimation as the next safety margin when the goal is to
minimize bad admits.
