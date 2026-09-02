# Agentic Hardware: Replay Deadline Pressure For Agentic LLMs

This repository is a research testbed for measuring whether agentic LLM
systems can resume quickly after tool calls.

The current question is:

> When many agents return from tools and need their next LLM token, can priority
> signals help the replay request meet its deadline under GPU, KV-cache, queue,
> and burst pressure?

The project has moved beyond the older generic "software prefetch" framing. The
current experiments focus on replay-deadline readiness with real SGLang serving,
HiCache, live timestamped telemetry, SGLang priority scheduling, controlled
pressure levels, and multiple coding-agent harness shapes.

## Repository Map

| Path | Purpose |
| --- | --- |
| [sglang_direct_kv/](sglang_direct_kv/) | Main SGLang replay-deadline testbed. |
| [sglang_direct_kv/README.md](sglang_direct_kv/README.md) | Long-form milestone notebook with historical detail. |
| [sglang_direct_kv/scripts/](sglang_direct_kv/scripts/) | Experiment runners, workload drivers, report builders, and SGLang launch helpers. |
| [aws/README.md](aws/README.md) | EC2 sync and connection workflow. |
| [HARDWARE_EMULATION_ENVIRONMENT.md](HARDWARE_EMULATION_ENVIRONMENT.md) | Original hardware-emulation environment notes. |
| [REPLAY_PATH_INSTRUMENTATION_PROPOSAL.md](REPLAY_PATH_INSTRUMENTATION_PROPOSAL.md) | Replay-path instrumentation design notes. |

## Latest Report Outputs

Each run updates the latest report files:

```text
sglang_direct_kv/artifacts/results/latest_master_report.html
sglang_direct_kv/artifacts/results/latest_manifest.json
```

Archived reports are kept under:

```text
sglang_direct_kv/artifacts/results/reports/<REPORT_LABEL>/master_report.html
```

The latest master report is intentionally treated as a single-experiment report.
It should represent the most recent run, not a mixture of old and new
experiments.

## Environment

On the EC2 machine used for the current experiments:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate
```

From a local checkout, the common EC2 helper commands are:

```bash
./aws/upload.sh 0
./aws/ssh_to_ec2.sh 0
./aws/download.sh 0
```

## Core Modes

The current manager-facing comparisons use these modes:

| Mode | Meaning |
| --- | --- |
| `no_prefetch` | Baseline. The replay request receives no end-to-end priority treatment. |
| `e2e_priority_hints` | Current priority path. The driver marks replay urgency and the SGLang boundary carries that priority into scheduling. |
| `pre_harness_priority_hints` | Harness-preservation path. The driver marks replay urgency before the harness sees the request, then the gateway proves whether that intent or a native harness signal survived and was translated into SGLang priority. |
| `no_cache_signal` | Cache-signal baseline. The gateway is present and records native harness cache fields, but it does not lower them to SGLang. |
| `harness_native_cache_lowered` | Harness-native cache path. The gateway translates only cache fields emitted by the harness itself. |
| `e2e_priority_hints_speculative_prefill` | Dynamo-like proactive path. The replay still gets E2E priority, and the gateway sends a background `max_tokens=1` warmup for the known next-turn prefix during the tool wait. |

For NeMo Agent Toolkit / NAT, `pre_harness_priority_hints` uses NAT's OpenAI
provider pass-through path. The generated NAT workflow includes
`service_tier: priority` and `extra_body.agentic_hints.priority_class: urgent`;
LangChain emits that as `service_tier=priority` plus top-level
`agentic_hints.priority_class=urgent`. The gateway records those emitted fields
and translates them to SGLang `priority=100`.

NAT instrumentation is intentionally kept outside the installed NAT package.
The driver calls
[`sglang_direct_kv/scripts/nemo_agent_toolkit_wrapper.py`](sglang_direct_kv/scripts/nemo_agent_toolkit_wrapper.py),
which writes the NAT workflow config, launches `nat run`, records wrapper
lifecycle events, and lets the gateway prove the emitted priority-bearing HTTP
request. This keeps the test portable across NAT versions, EC2, and GH200.

For stronger NAT-side evidence, use the shared-service probe. It runs a real
`nat serve` process, sends older background requests into NAT first, then sends
an urgent request into the same NAT server. The report compares the order
requests entered NAT with the order NAT emitted model calls to the gateway.
This proves observable NAT service ordering without patching NAT internals.

This speculative prefill mode is not SGLang speculative decoding. It mimics
Dynamo's agent hint behavior: after the current turn/tool-call prefix is known,
send a small background prefill so the later replay can reuse warmed KV. Older
modes such as `dynamo_priority_hints`, `direct_prefetch`, and
`projected_hardware_bypass` are still useful for historical analysis, but they
are not the default comparison for the current pressure chart.

## Pressure Levels

Pressure levels bundle multiple knobs so experiments do not explode into a full
Cartesian sweep.

| Level | Name | Main Stressor | Typical Knobs |
| --- | --- | --- | --- |
| `p0_control` | Control | Easy baseline | `500 ms` tool wait, `1024` target prompt tokens, `0` fillers, `1` urgent agent |
| `p1_mild` | Mild pressure | Shorter tool wait, modest context | Small filler count, moderate prompt, one urgent replay |
| `p2_medium` | Medium pressure | More queue and KV pressure | Medium filler count, larger prompt, one urgent replay |
| `p3_high` | Queue pressure | One urgent replay behind older work | `50 ms` tool wait, `4096` target prompt tokens, `32` fillers, `1` urgent agent |
| `p4_cliff` | Deadline cliff | Short wait plus large KV and heavier backend pressure | Tight tool wait, large prompt, high filler pressure |
| `p5_boss_queue` | Boss queue | Many urgent replays compete at once | `50 ms` tool wait, `4096` target prompt tokens, fillers per session, multiple urgent agents |

Quick sentinel experiments often run only P0, P3, and P5 first. Full ladder
runs use P0 through P5.

## Hardware Profiles

Hardware profiles keep the experiment design the same while changing the amount
of pressure applied to the machine. The runner loads
`HARDWARE_PROFILE=ec2_a10g` by default.

| Profile | File | Use Case |
| --- | --- | --- |
| `ec2_a10g` | [sglang_direct_kv/configs/hardware/ec2_a10g.env](sglang_direct_kv/configs/hardware/ec2_a10g.env) | Current EC2/A10G-scale runs and apples-to-apples GH200 comparison. |
| `gh200` | [sglang_direct_kv/configs/hardware/gh200.env](sglang_direct_kv/configs/hardware/gh200.env) | GH200-scaled pressure run with larger token budget, larger HiCache, more fillers, and more urgent agents. |

Profile defaults can still be overridden inline:

```bash
HARDWARE_PROFILE=gh200 MAX_TOTAL_TOKENS=131072 \
bash scripts/run_native_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Recommended migration sequence for a new GH200 machine:

1. Run `HARDWARE_PROFILE=ec2_a10g` first. This gives an apples-to-apples
   comparison against the EC2 results.
2. Run `HARDWARE_PROFILE=gh200` next. This increases pressure so GH200 can find
   its own replay-deadline cliff.
3. Keep `REPORT_BUILDER_MODE=lightweight` for full multi-harness runs.

### GH200 Architecture Note

The current EC2 machine and a Grace Hopper 200 machine are not the same CPU
architecture.

| Machine | Expected CPU Architecture | What This Means |
| --- | --- | --- |
| Current EC2 GPU machine | `x86_64` | Uses normal x86 Linux packages and venv wheels. |
| GH200 | `aarch64` / `arm64` | Uses ARM64 packages and ARM64 Python/Node wheels. |

In simple terms: the repo code can move to GH200, but the installed
dependencies should not be copied from EC2. Rebuild the Python venvs, SGLang
environment, Node.js CLIs, NAT venv, and Hermes venv directly on GH200.

### GH200 Setup

After cloning or pulling this repository on the GH200 machine, rebuild the
Python environment directly on GH200:

```bash
cd ~/agentic_hardware/sglang_direct_kv

bash scripts/setup_gh200.sh
```

That script creates:

| Path | Purpose |
| --- | --- |
| `sglang_direct_kv/.venv` | Main project Python environment. Installs `requirements.txt`, editable `agentic-kv`, and analysis extras such as `scikit-learn`. |
| `$HOME/agentic_hardware/.venvs/nat_py311` | Isolated NeMo Agent Toolkit / NAT environment. |
| `$HOME/agentic_hardware/.venvs/hermes_agent_py311` | Isolated Hermes Agent environment. |

If you need a different Python binary:

```bash
PYTHON_BIN=python3.10 bash scripts/setup_gh200.sh
```

If the GH200 image already has system packages and CUDA configured:

```bash
INSTALL_SYSTEM_DEPS=0 bash scripts/setup_gh200.sh
```

If you want to add or change optional analysis packages:

```bash
EXTRA_PYTHON_PACKAGES="scikit-learn matplotlib seaborn pyarrow" \
bash scripts/setup_gh200.sh
```

The native CLI harnesses use Node.js through `npx`, so GH200 also needs an ARM64
Node.js install. If `setup_gh200.sh` says Node is missing, install Node LTS:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install --lts
node -p "process.arch"
```

The expected Node architecture on GH200 is:

```text
arm64
```

Before running experiments on GH200, check the machine:

```bash
uname -m
nvidia-smi
python3 --version
node -p "process.arch"
```

Expected GH200 architecture output:

```text
uname -m -> aarch64
node process.arch -> arm64
```

After installing dependencies on GH200, run a smoke test before the long
experiment:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/smoke_multi_harness_wireability.py \
  --harnesses codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent
```

## Most Recent Experiment: Real-Client Deadline Pressure

The newest experiment sends real coding-agent CLIs through the inspection
gateway, then forwards their requests to SGLang with normalized priority
metadata. This is stronger than the earlier adapter-only run because each
native CLI generates its own live request shape before our gateway normalizes
priority at the SGLang boundary.

Current archived native run:

```text
sglang_direct_kv/artifacts/results/reports/native_harness_deadline_pressure_backend_20260901_035640/master_report.html
```

Native client smoke status as of September 1, 2026:

```text
codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent
```

`nemo_agent_toolkit` and `hermes_agent` are installed in persistent isolated
Python 3.11 venvs on EC2 and should be passed into experiment runs with
`HARNESS_NAT_BIN` and `HARNESS_HERMES_BIN`. NAT needs the LangChain integration
extra, so install it as `nvidia-nat[langchain]`, not plain `nvidia-nat`.

Native-only run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="native_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_native_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Mixed native-plus-adapter run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent deepseek_harness" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="multi_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Pre-harness priority preservation smoke. This is the cleanest first run when
you want to see whether a harness preserves urgency before the request reaches
SGLang:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code" \
PRESSURE_LEVELS="p0_control p3_high" \
MODES="no_prefetch e2e_priority_hints pre_harness_priority_hints" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="pre_harness_priority_smoke_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

If a long EC2 run is interrupted after some cases finish, resume the same
report label without rerunning completed cases:

```bash
SKIP_EXISTING_CASES=1 \
HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent deepseek_harness" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="<existing_report_label>" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Headline result from the current native run. Values are median first-replay-token
lateness, so lower is better and anything above `0 ms` missed the replay
deadline:

| Harness | P0 no-prefetch | P0 E2E | P3 no-prefetch | P3 E2E | P5 no-prefetch | P5 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hatcher | `0.22 s` | `0.21 s` | `37.17 s` | `4.99 s` | `46.87 s` | `21.56 s` |
| Codex | `1.78 s` | `1.23 s` | `47.36 s` | `11.85 s` | `53.74 s` | `22.49 s` |
| Claude Code | `1.46 s` | `1.39 s` | `60.75 s` | `11.77 s` | `50.19 s` | `18.32 s` |
| OpenCode | `3.52 s` | `3.66 s` | `82.93 s` | `13.99 s` | `85.86 s` | `47.52 s` |
| Qwen Code | `5.86 s` | `5.69 s` | `76.81 s` | `19.90 s` | `95.11 s` | `67.11 s` |
| Pi Agent Harness | `1.42 s` | `1.30 s` | `57.52 s` | `11.77 s` | `52.60 s` | `18.72 s` |
| OpenClaw | `9.50 s` | `6.93 s` | `74.91 s` | `19.76 s` | `83.86 s` | `54.65 s` |
| NeMo Agent Toolkit / NAT | `3.62 s` | `3.62 s` | `67.66 s` | `10.57 s` | `50.88 s` | `19.73 s` |
| Hermes Agent | `12.67 s` | `12.44 s` | `72.94 s` | `19.63 s` | `67.45 s` | `40.29 s` |

Interpretation from this run: end-to-end priority hints helped every harness
under P3 queue pressure and helped every P5 boss-queue comparison, but the
priority path still missed the tight replay deadline under P3 and P5. Priority
can move a replay earlier in the queue; it cannot create extra GPU compute, KV
capacity, or host-to-device bandwidth. The P0 rows are also above zero because
real CLIs add startup/protocol overhead around the backend call. OpenCode's P3
and P5 no-prefetch rows use a backend decode-result fallback because the client
closed the stream before the gateway emitted `m27.request.end`; the raw proof
marks those rows with `first_token_source=scheduler_process_decode_result`.

## Multi-Harness Deadline Pressure

The broad experiment compares the same SGLang priority boundary across ten
non-Dynamo agent harness shapes. Eight entries now launch real native CLIs; the
DeepSeek Harness path remains adapter-backed until its native CLI exposes a
reliable headless command path.

| Harness | Current experiment path |
| --- | --- |
| `hatcher` | In-repo Hatcher / Deep Agents-style control harness. |
| `codex` | Real Codex CLI through the inspection gateway. |
| `claude_code` | Real Claude Code CLI through the inspection gateway. |
| `opencode` | Real OpenCode CLI through the inspection gateway. |
| `qwen_code` | Real Qwen Code CLI through the inspection gateway. |
| `pi_agent_harness` | Real Pi CLI with a generated OpenAI-compatible gateway provider extension. |
| `openclaw` | Real OpenClaw CLI with a generated OpenAI-compatible gateway provider config. |
| `nemo_agent_toolkit` | Real NAT CLI through a generated OpenAI-compatible workflow config. |
| `deepseek_harness` | Adapter-backed; native `dsh` CLI currently hangs during headless `--help` / `--version` probing. |
| `hermes_agent` | Real Hermes Agent CLI through a generated OpenAI-compatible provider config. |

It runs each harness in:

```text
no_prefetch
e2e_priority_hints
pre_harness_priority_hints
e2e_priority_hints_speculative_prefill
```

And uses the sentinel pressure levels:

```text
p0_control p3_high p5_boss_queue
```

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent deepseek_harness" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_LABEL="multi_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

GH200 apples-to-apples run. Use this first after migrating the checkout to
GH200:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="gh200_apples_to_apples_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_native_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

GH200-scaled pressure run. Use this after the apples-to-apples run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=gh200 \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent" \
PRESSURE_LEVELS="p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="gh200_scaled_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_native_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

The lightweight report shows the Replay Deadline Pressure Chart as a
pressure-first overlay with two panels. Panel A measures full replay-deadline
lateness from replay due time to first token. Panel B measures backend-only time
from SGLang receive to first token, which separates harness/client overhead from
SGLang queueing, KV movement, and compute. P0/P3/P5 are the main x-axis
sections, color separates `no_prefetch` from `e2e_priority_hints`, and symbol
shape separates harnesses.

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh) | Orchestrates the multi-harness pressure experiment and writes the latest report. |
| [sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh) | Runs only the native CLI harnesses plus the Hatcher control. |
| [sglang_direct_kv/scripts/run_multi_harness_replay_driver.py](sglang_direct_kv/scripts/run_multi_harness_replay_driver.py) | Generates target replay and filler traffic for the selected harnesses. |
| [sglang_direct_kv/scripts/nemo_agent_toolkit_wrapper.py](sglang_direct_kv/scripts/nemo_agent_toolkit_wrapper.py) | Portable NAT wrapper that records config, process, and gateway-emission lifecycle events without patching NAT itself. |
| [sglang_direct_kv/scripts/run_nemo_nat_service_priority_probe.py](sglang_direct_kv/scripts/run_nemo_nat_service_priority_probe.py) | Runs real `nat serve` as one shared service and can also inspect NAT Dynamo-provider `nvext.agent_hints` without a full Dynamo runtime. |
| [sglang_direct_kv/scripts/harness_sglang_gateway.py](sglang_direct_kv/scripts/harness_sglang_gateway.py) | Normalizes harness requests at the SGLang boundary and injects priority metadata. |
| [sglang_direct_kv/configs/hardware/](sglang_direct_kv/configs/hardware/) | EC2 and GH200 pressure profiles. |
| [sglang_direct_kv/scripts/run_real_client_wireability_probe.py](sglang_direct_kv/scripts/run_real_client_wireability_probe.py) | Launches real client CLIs against the inspection gateway and reports the live request shape observed at the boundary. |
| [sglang_direct_kv/scripts/smoke_multi_harness_wireability.py](sglang_direct_kv/scripts/smoke_multi_harness_wireability.py) | Fast local smoke test for CLI harness wireability through the gateway. |
| [sglang_direct_kv/scripts/run_sglang_hicache_server.sh](sglang_direct_kv/scripts/run_sglang_hicache_server.sh) | Launches SGLang with HiCache, priority scheduling, and runtime telemetry flags. |
| [sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) | Builds the master HTML report, evidence tables, and Replay Deadline Pressure Chart. |
| [sglang_direct_kv/scripts/build_multi_harness_deadline_summary.py](sglang_direct_kv/scripts/build_multi_harness_deadline_summary.py) | Lightweight all-harness report builder used when the rich timeline report would be too large. |

Smoke-test native CLI wireability without starting the real GPU server:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/smoke_multi_harness_wireability.py \
  --harnesses pi_agent_harness openclaw
```

Probe real client-generated traffic without running a full pressure sweep:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/run_real_client_wireability_probe.py \
  --clients codex claude_code opencode qwen_code pi_agent_harness openclaw \
  --out-dir artifacts/results/real_client_wireability/$(date +%Y%m%d_%H%M%S)
```

If `--target-base` is omitted, the probe uses a fake local SGLang-compatible
backend. Pass `--target-base http://127.0.0.1:30000` to observe real clients
through the gateway while forwarding to a running SGLang server.

For a stronger NAT-specific internal scheduling test, NAT also exposes
`nat serve`, which runs a workflow through the FastAPI front end. This command
uses that path and updates `artifacts/results/latest_master_report.html`:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
python scripts/run_nemo_nat_service_priority_probe.py \
  --report-label "nat_service_priority_probe_$(date +%Y%m%d_%H%M%S)" \
  --low-count 6 \
  --urgent-count 1 \
  --low-lead-ms 100 \
  --low-stagger-ms 5 \
  --fake-backend-delay-ms 1200 \
  --nat-workers 1 \
  --update-latest
```

Read `nat_service_priority_probe.csv` in the report folder. The key columns are
`submit_rank_into_nat`, `emit_rank_from_nat_to_gateway`,
`older_background_submitted_before`, `older_background_emitted_before`, and
`verdict`. If the urgent row has a lower emit rank than older background rows,
NAT emitted it earlier. If NAT also preserves structured priority fields, the
evidence is stronger; if priority is recovered only from the experiment marker,
the report says the priority cause is not proven.

To inspect NAT's Dynamo-provider hint format without installing the full Dynamo
runtime, run the same probe with NAT's Dynamo transport path:

```bash
cd ~/agentic_hardware/sglang_direct_kv

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
$HOME/agentic_hardware/.venvs/nat_py311/bin/python \
  scripts/run_nemo_nat_service_priority_probe.py \
  --report-label "nat_dynamo_direct_wireability_probe_$(date +%Y%m%d_%H%M%S)" \
  --nat-provider dynamo_direct \
  --low-count 2 \
  --urgent-count 1 \
  --low-lead-ms 100 \
  --low-stagger-ms 5 \
  --fake-backend-delay-ms 200 \
  --nat-workers 1 \
  --nvext-prefix-total-requests 10 \
  --nvext-prefix-osl 512 \
  --nvext-prefix-iat 50 \
  --update-latest
```

This is a lightweight wire-format probe. It imports NAT's Dynamo HTTP transport
and sends requests through our gateway/fake backend, so it does not require a
running Dynamo router. The expected proof is that background requests emit
`nvext.agent_hints.priority=2`, while urgent requests emit
`nvext.agent_hints.priority=100` plus prefix, OSL, IAT, total-request, and
cache-control hints. A real end-to-end Dynamo scheduling result still requires
the full NAT -> Dynamo -> SGLang stack, which is better suited for a larger
machine such as GH200.

To test NAT-only inferred priority, without putting `priority_intent` on the
frontend request and without installing Dynamo, run:

```bash
cd ~/agentic_hardware/sglang_direct_kv

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
$HOME/agentic_hardware/.venvs/nat_py311/bin/python \
  scripts/run_nemo_nat_service_priority_probe.py \
  --report-label "nat_inferred_priority_probe_$(date +%Y%m%d_%H%M%S)" \
  --nat-provider dynamo_inferred \
  --prompt-tokens 1024 \
  --fake-backend-delay-ms 100 \
  --nvext-prefix-total-requests 10 \
  --nvext-prefix-osl 512 \
  --nvext-prefix-iat 50 \
  --update-latest
```

This probe gives NAT a workflow prediction profile. The profile says some NAT
workflow nodes are low sensitivity and some are high sensitivity. The request
entering NAT does not say "urgent"; it only runs under a workflow path. NAT's
Dynamo transport looks up that workflow path, emits `nvext.agent_hints.priority`,
and our gateway translates that emitted value into the SGLang `priority` field.
The profile is saved beside the report as
`nat_inferred_priority_profile.json`, and it is also rendered as a table in the
HTML report. The proof lives in `nat_service_priority_probe.csv`; look for
`frontend_priority_intent_present=no`, `expected_inferred_priority`,
`emitted_nvext_priority`, `gateway_translated_priority`, and `verdict`.

To run the main replay-deadline pressure experiment for NAT only, compare the
baseline against NAT-inferred priority across all P0-P5 pressure levels:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="nemo_agent_toolkit" \
PRESSURE_LEVELS="p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue" \
MODES="no_prefetch nat_inferred_priority_hints" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="nat_inferred_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

In this run, the gateway does not attach priority from the experiment marker.
For `nat_inferred_priority_hints`, NAT emits `nvext.agent_hints.priority` from
the workflow profile, and the gateway only translates that emitted value into
SGLang's `priority` field. The chart to inspect is the Replay Deadline Pressure
Chart; the raw proof fields are `harness_emit_priority_signal`,
`gateway_priority_translation_source`, and `sglang_priority`.

## Harness-Native Cache Signal Experiment

The compact entry point for the current signal design space is:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
SIGNAL_FAMILIES="harness_emitted frontend_supplied" \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="signal_design_space_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_signal_design_space.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

The two public buckets are:

| `SIGNAL_FAMILIES` value | Meaning |
| --- | --- |
| `harness_emitted` | The harness creates the signal. The gateway only translates what the harness emitted for SGLang. This includes harness-native cache control, and NAT native priority inference when NAT is selected. |
| `frontend_supplied` | The experiment/front end gives signal intent to the harness first. The gateway translates whatever survives the harness path for SGLang. |
| `all` | Alias for `harness_emitted frontend_supplied`. |

The wrapper prints the detailed mode expansion before it starts. Today that
expands to the proven lower-level modes:

| Family piece | Lower-level modes |
| --- | --- |
| `harness_emitted/cache` | `no_cache_signal harness_native_cache_lowered` |
| `harness_emitted/priority` | `no_prefetch nat_inferred_priority_hints`, only for selected harnesses that support native priority inference today. |
| `frontend_supplied` | `no_prefetch pre_harness_priority_hints` |

Use `DRY_RUN=1` to preview the expansion without launching SGLang:

```bash
DRY_RUN=1 \
SIGNAL_FAMILIES="harness_emitted frontend_supplied" \
HARNESSES="qwen_code pi_agent_harness nemo_agent_toolkit" \
bash scripts/run_harness_signal_design_space.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

The detailed cache-signal experiment asks a narrower question:

> Can the harness decide what cache signal to emit, while the gateway only
> translates that harness-emitted signal for SGLang?

The rule is:

```text
harness decides -> gateway translates -> SGLang executes
```

The gateway stays in the path for both modes, but it is not allowed to invent a
cache hint from hidden experiment knowledge such as replay phase, pressure
level, or prompt size.

| Mode | Meaning |
| --- | --- |
| `no_cache_signal` | Baseline. The gateway observes traffic and records whether a harness emitted cache fields, but it does not lower cache signals to SGLang. |
| `harness_native_cache_lowered` | The gateway translates only cache fields that the harness emitted, such as `cache_control`, `prompt_cache_key`, `promptCacheKey`, `cacheRetention`, or `prompt_cache_retention`. |

Harness-specific notes:

| Harness | Native cache signal path used by this testbed |
| --- | --- |
| `qwen_code` | Uses the OpenAI-compatible wire path for `no_cache_signal`, because that path does not emit explicit cache-control fields. Uses the Anthropic-compatible wire path for `harness_native_cache_lowered`, where Qwen emits `cache_control` markers on stable prompt sections. |
| `pi_agent_harness` | Uses Pi session persistence plus `PI_CACHE_RETENTION=long`. In native-cache mode Pi emits Anthropic-style `cache_control`, OpenAI-style `prompt_cache_key` / `prompt_cache_retention`, and session-affinity headers. |
| `openclaw` | Uses OpenClaw cache-retention environment toggles and provider compat flags. |
| `opencode` | Uses OpenCode's own emitted prompt-cache key when available. |

Recommended first run on EC2:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARDWARE_PROFILE=ec2_a10g \
HARNESSES="opencode qwen_code pi_agent_harness openclaw" \
PRESSURE_LEVELS="p0_control p2_medium p3_high p5_boss_queue" \
MODES="no_cache_signal harness_native_cache_lowered" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="harness_native_cache_signals_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

For a faster smoke test:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARDWARE_PROFILE=ec2_a10g \
HARNESSES="qwen_code pi_agent_harness" \
PRESSURE_LEVELS="p0_control" \
MODES="no_cache_signal harness_native_cache_lowered" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="qwen_pi_native_cache_fix_smoke_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Read `harness_native_cache_signal_proof.csv` in the report directory. The key
columns are `native_cache_signal_seen`, `native_cache_signal_source`,
`gateway_cache_lowered`, `gateway_cache_translation_source`, and
`gateway_invented_signal`. A clean positive result has
`native_cache_signal_seen=yes`, `gateway_cache_lowered=yes`,
`gateway_cache_translation_source=harness_emitted_cache_signal`, and
`gateway_invented_signal=false`.

Latest EC2 wireability result:

```text
sglang_direct_kv/artifacts/results/real_client_wireability/real_client_probe_six_20260901_055102/real_client_wireability_report.html
```

That EC2 smoke, run on 2026-09-01, launched the real Codex, Claude Code,
OpenCode, Qwen Code, Pi, and OpenClaw CLIs against the inspection gateway. All
six reached the gateway, all six were tagged with `sglang_priority=100`, and the
gateway recorded their live request shape without storing prompt bodies:

| Client | API shape | Request body | Prompt chars |
| --- | --- | ---: | ---: |
| Codex | `/v1/responses` | `37.6 KB` | `4.3K` |
| Claude Code | `/v1/messages?beta=true` | `5.3 KB` | `1.6K` |
| OpenCode | `/v1/chat/completions` | `3.6 KB` | `3.2K` |
| Qwen Code | `/v1/chat/completions` | `97.8 KB` | `36.6K` |
| Pi Agent Harness | `/v1/chat/completions` | `1.5 KB` | `1.3K` |
| OpenClaw | `/v1/chat/completions` | `28.0 KB` | `19.9K` |

This is a wireability probe, not a pressure result. Its purpose is to prove
that real client-generated traffic can be inspected and priority-tagged at the
SGLang boundary before we run the heavier deadline-pressure ladder through real
client CLIs.

For large all-harness runs, [run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh)
automatically uses the lightweight summary report when the case count is large.
Set `REPORT_BUILDER_MODE=rich` to force the full timeline report for smaller
runs, or `REPORT_BUILDER_MODE=lightweight` to force the compact all-harness
report.

Previous archived adapter run:

```text
sglang_direct_kv/artifacts/results/reports/multi_harness_no_dynamo_20260831_232717/master_report.html
```

Headline result from that run. Values are median first-replay-token lateness,
so lower is better and anything above `0 ms` missed the replay deadline:

| Harness | P0 no-prefetch | P0 E2E | P3 no-prefetch | P3 E2E | P5 no-prefetch | P5 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hatcher | `221 ms` | `223 ms` | `33.0 s` | `9.2 s` | `50.4 s` | `26.5 s` |
| Codex | `1.4 s` | `1.3 s` | `39.9 s` | `10.5 s` | `15.7 s` | `7.7 s` |
| Claude Code | `1.3 s` | `1.3 s` | `68.9 s` | `11.9 s` | `52.2 s` | `27.9 s` |
| OpenCode | `3.6 s` | `3.4 s` | `80.6 s` | `15.7 s` | `81.3 s` | `41.3 s` |
| Qwen Code | `5.9 s` | `5.9 s` | `66.6 s` | `10.9 s` | `44.5 s` | `8.3 s` |
| NeMo Agent Toolkit / NAT | `8.7 s` | `9.0 s` | `74.2 s` | `12.9 s` | `53.8 s` | `30.6 s` |
| DeepSeek Harness | `8.8 s` | `9.3 s` | `74.0 s` | `12.9 s` | `53.2 s` | `28.2 s` |
| Pi Agent Harness | `9.0 s` | `8.9 s` | `74.6 s` | `12.8 s` | `52.9 s` | `29.4 s` |
| OpenClaw | `8.9 s` | `8.8 s` | `74.6 s` | `12.8 s` | `51.8 s` | `27.4 s` |
| Hermes Agent | `9.0 s` | `8.8 s` | `69.1 s` | `13.3 s` | `52.1 s` | `27.2 s` |

Interpretation: end-to-end priority hints help, especially under P3/P5
pressure, but they do not guarantee deadline readiness. Priority metadata can
move replay requests earlier in queues; it cannot by itself create more GPU
compute, KV capacity, or transfer bandwidth.

## Hatcher Pressure Ladder

This single-harness ladder focuses on the current Hatcher / Deep Agents-style
control harness and asks when replay deadlines start failing as pressure rises.

Full P0-P5 run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

PRESSURE_LEVELS="p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_LABEL="hatcher_pressure_ladder_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_hatcher_pressure_ladder.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Fast sentinel run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_LABEL="hatcher_pressure_ladder_quick_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_hatcher_pressure_ladder_quick.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_hatcher_pressure_ladder.sh](sglang_direct_kv/scripts/run_hatcher_pressure_ladder.sh) | Full P0-P5 Hatcher pressure ladder. |
| [sglang_direct_kv/scripts/run_hatcher_pressure_ladder_quick.sh](sglang_direct_kv/scripts/run_hatcher_pressure_ladder_quick.sh) | Reduced P0/P3/P5 Hatcher ladder. |
| [sglang_direct_kv/scripts/run_real_prompt_controlled_replay.py](sglang_direct_kv/scripts/run_real_prompt_controlled_replay.py) | Core controlled replay workload generator. |
| [sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) | Shared report builder. |

## Priority Queue Proof

This pre-flight test proves that SGLang can admit a high-priority request ahead
of older low-priority work when priority scheduling is enabled.

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_LABEL="priority_queue_jump_sanity_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_priority_queue_sanity.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_priority_queue_sanity.sh](sglang_direct_kv/scripts/run_priority_queue_sanity.sh) | Launches the priority proof experiment. |
| [sglang_direct_kv/scripts/run_priority_queue_jump_workload.py](sglang_direct_kv/scripts/run_priority_queue_jump_workload.py) | Creates older low-priority work followed by one high-priority replay request. |
| [sglang_direct_kv/scripts/summarize_priority_queue_sanity.py](sglang_direct_kv/scripts/summarize_priority_queue_sanity.py) | Summarizes whether the high-priority request jumped ahead. |

## Controlled Replay With Priority Modes

This controlled experiment family is useful when comparing individual scheduler
or replay modes without changing the harness.

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL="controlled_replay_priority_$(date +%Y%m%d_%H%M%S)" \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
MAX_TIMELINE_GAPS=96 \
MAX_PAIRS=2 \
MODES="no_prefetch e2e_priority_hints" \
TOOL_WAIT_LIST_MS="50" \
FILLER_LIST="32" \
REQUEST_CONCURRENCY=8 \
FILLER_PROMPT_TOKENS=1536 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.72 \
bash scripts/run_milestone27_real_prompt_controlled_replay.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_milestone27_real_prompt_controlled_replay.sh](sglang_direct_kv/scripts/run_milestone27_real_prompt_controlled_replay.sh) | Runs real-prompt controlled replay experiments. |
| [sglang_direct_kv/scripts/run_real_prompt_controlled_replay.py](sglang_direct_kv/scripts/run_real_prompt_controlled_replay.py) | Workload driver. |
| [sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) | Report builder. |

## Evidence And Reporting

The report contains presentation charts near the top and raw proof tables near
the bottom. Important evidence files include:

| Artifact | What It Shows |
| --- | --- |
| `global_kv_readiness_by_mode.csv` | Replay first-token lateness by mode, harness, and pressure level. |
| `harness_priority_preservation_proof.csv` | Pre-harness priority proof: driver intent, harness input signal, emitted harness signal, gateway translation, SGLang priority, and NAT wrapper lifecycle fields when NAT is used. |
| `nat_service_priority_probe.csv` | NAT shared-service and Dynamo-provider proof: request order into `nat serve` or NAT's Dynamo transport, emitted order to the gateway, older background work ahead, `nvext.agent_hints` when present, and whether urgent work overtook older requests. |
| `speculative_prefill_proof.csv` | Dynamo-like warmup proof: hint seen, background warmup timing, cached-prefix reuse, and replay timing. |
| `replay_queue_timing.csv` | Driver release, backend receive, scheduler admission, and first-token timing. |
| `dynamo_priority_queue_effectiveness.csv` | Priority queue behavior and older lower-priority work bypass evidence. |
| `request_state_snapshots.csv` | Timestamp-centered state for the target request and surrounding system pressure. |
| `runtime_telemetry_events.csv` | Live instrumentation events emitted while SGLang is running. |
| `kv_block_ledger.csv` | KV residency and movement evidence where available. |

The most important manager-facing chart is the **Replay Deadline Pressure
Chart**. Each dot is one replay request. Lower is better. Values above `0 ms`
missed the replay deadline; values below `0 ms` were early.

## Milestone Notebook

The full milestone history lives in
[sglang_direct_kv/README.md](sglang_direct_kv/README.md). Use it when you need
the detailed lab notebook, older commands, or historical context.

| Milestone Range | Focus |
| --- | --- |
| M0-M8 | Early SGLang setup, KV stress, and basic workload control. |
| M9-M15 | HiCache, real prompts, and first report automation. |
| M16-M27 | AgentBench/DeepAgents-style prompts and controlled replay experiments. |
| M28-M37 | Replay-path ledger, exact timing attribution, H2D evidence, live telemetry, and timestamp-centered reports. |
| M38-M40 | Priority queue proof, priority retention, and E2E priority sanity checks. |
| Current | Pressure ladder and multi-harness Replay Deadline Pressure Chart. |

## Harness Backlog

Native client status:

| Harness | Status | Next action |
| --- | --- | --- |
| Codex | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| Claude Code | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| OpenCode | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| Qwen Code | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| Pi Agent Harness | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| OpenClaw | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| NeMo Agent Toolkit / NAT | Native CLI wired and smoke-tested with `$HOME/agentic_hardware/.venvs/nat_py311/bin/nat`. | Include in real-client pressure runs with `HARNESS_NAT_BIN` set. |
| DeepSeek Harness | Adapter-backed. | Re-probe after the `dsh` CLI exposes a reliable headless command path. |
| Hermes Agent | Native CLI wired and smoke-tested with `$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes`. | Include in real-client pressure runs with `HARNESS_HERMES_BIN` set. |

The next recommended path is:

1. Run the P0/P3/P5 sentinel ladder across the eight real CLI harness paths plus the Hatcher control through [run_native_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh).
2. Keep DeepSeek Harness adapter-backed until `dsh` can run a non-interactive smoke request without hanging.
3. Leave Dynamo out of local EC2 runs unless the host has enough spare CPU, memory, and disk for its runtime stack.

## Current Research Claim

Priority hints are useful: they express urgency and can improve scheduler
admission. But software priority alone does not guarantee replay-deadline
readiness under heavy GPU compute pressure, KV-cache pressure, short tool waits,
or many urgent agents arriving together.

That gap is the hardware/runtime opportunity: make replay-critical KV residency,
movement, and admission cheaper, deadline-aware, and enforceable.
