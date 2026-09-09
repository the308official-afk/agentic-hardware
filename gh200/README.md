# GH200 Setup And Run Guide

This is the short operational guide for moving changes from your Mac to the
GH200 machine and running replay-deadline pressure experiments there.

The intended setup is:

```text
Mac
  edit code, commit, push, sync, download reports

GH200 host
  run harnesses, NAT, Hermes, gateway, experiment driver

GH200 Docker container
  run only SGLang + CUDA/GPU backend
```

This split lets NAT and Hermes use their Python 3.11 host virtual environments
while SGLang uses the Docker CUDA runtime required on the GH200 machine.

## 1. Configure Connection

The helpers default to:

| Variable | Default |
| --- | --- |
| `AGENTIC_GH200_USER` | `ojaiyeob` |
| `AGENTIC_GH200_HOST` | `gracehopper` |
| `AGENTIC_GH200_JUMP_HOST` | `falcon.7elements.com` |
| `AGENTIC_GH200_JUMP_PORT` | `1337` |
| `AGENTIC_GH200_REMOTE_DIR` | `/home/central/<user>/<repo-name>` (derived from local clone name) |
| `AGENTIC_GH200_MODEL_CACHE` | `$HOME/dynamo_model_cache` |

Override them on your Mac if needed:

```bash
export AGENTIC_GH200_USER="ojaiyeob"
export AGENTIC_GH200_HOST="gracehopper"
export AGENTIC_GH200_JUMP_HOST="falcon.7elements.com"
export AGENTIC_GH200_JUMP_PORT="1337"
```

> **Remote dir note**: `AGENTIC_GH200_REMOTE_DIR` defaults to
> `/home/central/<user>/<repo-name>` where `<repo-name>` is your local
> clone's directory name. If your local clone is named `agentic-hardware`
> but the GH200 working directory is `agentic_hardware`, set this explicitly:
>
> ```bash
> export AGENTIC_GH200_REMOTE_DIR="/home/central/ojaiyeob/agentic_hardware"
> ```
>
> Add this export to your shell profile or set it before every sync/download.

## 2. Sync Source From Mac To GH200

Run from the repo root on your Mac:

```bash
cd /path/to/agentic-hardware   # your local clone root

export AGENTIC_GH200_REMOTE_DIR="/home/central/ojaiyeob/agentic_hardware"

./gh200/sync_to_gh200.sh --dry
./gh200/sync_to_gh200.sh
```

The sync copies source code only. It excludes `.git/`, `.venv/`, `.venvs/`,
`node_modules/`, caches, logs, and `sglang_direct_kv/artifacts/`. Remote
experiment outputs are protected.

> **First-time bootstrap**: all `gh200/` helper scripts live in this repo and
> must be synced before they can be run on GH200. Always run
> `sync_to_gh200.sh` before following the steps below for the first time.

## 3. SSH Into GH200

From your Mac:

```bash
./gh200/ssh_to_gh200.sh
```

Quick machine checks:

```bash
uname -m       # expected: aarch64
nvidia-smi
```

## 4. Build GH200 Host Dependencies

Run on GH200:

```bash
cd ~/agentic_hardware/sglang_direct_kv

INSTALL_SYSTEM_DEPS=0 bash scripts/setup_gh200.sh
```

This creates:

| Path | Purpose |
| --- | --- |
| `~/agentic_hardware/sglang_direct_kv/.venv` | Main project venv |
| `~/agentic_hardware/.venvs/nat_py311` | NeMo Agent Toolkit / NAT venv |
| `~/agentic_hardware/.venvs/hermes_agent_py311` | Hermes Agent venv |

Use `INSTALL_SYSTEM_DEPS=0` on the current GH200 image to avoid the known DKMS
package conflict.

## 5. Install Node.js ARM64

Run on GH200 if `node` or `npx` is missing:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install --lts
node -p "process.arch"   # expected: arm64
```

For new login shells, add to `~/.bashrc`:

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
```

The GPU run scripts (`run_host_signal_design_space.sh` and anything that
calls it) source NVM automatically if `~/.nvm/nvm.sh` exists, so Node
does not need to be on `PATH` before invoking those scripts.

## 6. Smoke Test Harnesses

Run on GH200 before a long GPU experiment:

```bash
cd ~/agentic_hardware

./gh200/smoke_harnesses.sh
```

This is a no-GPU test. It verifies the harnesses can talk through the inspection
gateway path, including NAT and Hermes on the host. A successful run exits with
code 0 and prints a summary line per harness with no `ERROR` or `FAILED`
entries. Any non-zero exit or `FAILED` line means the host environment needs
attention before running GPU experiments.

## 7. Run GPU Experiments

GPU runs require a populated model cache. The default location is
`~/dynamo_model_cache`. If that directory is empty or missing, set:

```bash
export AGENTIC_GH200_MODEL_CACHE=/path/to/your/model_cache
```

The run scripts will abort early with a clear message if the cache is not
found.

Use `screen` so the job survives disconnects:

```bash
screen -S gh200_experiment
# detach:   Ctrl+A then D
# reattach: screen -r gh200_experiment
```

Then run from GH200:

```bash
cd ~/agentic_hardware

./gh200/run_sentinel.sh
```

`run_sentinel.sh` is a fast first check: Hatcher only, baseline vs
gateway-injected priority, three pressure levels. Watch for the
`REPORT_LABEL=gh200_sentinel_...` line in the output — that label is used
to download or tail logs later. A clean exit (exit code 0) means the
sentinel passed.

If that works, run the EC2-scale apples-to-apples experiment:

```bash
./gh200/run_apples_to_apples.sh
```

Then run the GH200-scaled pressure ladder:

```bash
./gh200/run_scaled_pressure.sh
```

### Phase 7 Controller Scale-Up

Use this after the sentinel passes when you want to test the portable
controller design on GH200 pressure. The default is intentionally focused:
`hatcher` is the internal DeepAgents-style control harness, and the runner
compares baseline/gateway priority against the controller modes from phases
2 through 6, plus the full-controller variants.

```bash
cd ~/agentic_hardware

./gh200/run_controller_scaleup.sh
```

Default Phase 7 controller families:

```text
baseline gateway_injected controller_scheduler controller_preload controller_targeted_prefetch controller_demote_restore controller_admission controller_full controller_full_chunked
```

Default Phase 7 pressure levels:

```text
p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue
```

To scale the same controller run across every currently wired harness:

```bash
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent" \
./gh200/run_controller_scaleup.sh
```

To run only the two strongest controller candidates:

```bash
SIGNAL_FAMILIES="baseline gateway_injected controller_demote_restore controller_admission" \
./gh200/run_controller_scaleup.sh
```

To compare the full controller against the chunked-prefill variant only:

```bash
SIGNAL_FAMILIES="baseline gateway_injected controller_full controller_full_chunked" \
CONTROLLER_CHUNKED_PREFILL_SIZE=2048 \
CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS=4096 \
./gh200/run_controller_scaleup.sh
```

### Full Signal Design Space Experiment

Use this when you want the full manager-facing comparison across all wired
harnesses, all pressure levels, and all consolidated signal families. It uses
the lightweight report builder so the final report stays compact.

```bash
cd ~/agentic_hardware

SIGNAL_FAMILIES="baseline harness_emitted frontend_supplied gateway_injected" \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent" \
PRESSURE_LEVELS="p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue" \
REPORT_BUILDER_MODE=lightweight \
./gh200/run_scaled_pressure.sh
```

Default GPU-run harnesses:

```text
hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent
```

Default signal families:

```text
baseline harness_emitted frontend_supplied gateway_injected
```

## 8. Watch Logs

Each run prints its `REPORT_LABEL`. Watch progress from another GH200 shell:

```bash
tail -f ~/agentic_hardware/sglang_direct_kv/artifacts/results/run_logs/<REPORT_LABEL>.log
```

The final report path will be:

```text
~/agentic_hardware/sglang_direct_kv/artifacts/results/latest_master_report.html
```

## 9. Download Reports Back To Mac

From your Mac:

```bash
cd /path/to/agentic-hardware   # your local clone root

export AGENTIC_GH200_REMOTE_DIR="/home/central/ojaiyeob/agentic_hardware"

./gh200/download.sh
```

This downloads compact latest report artifacts only.

To download one archived report folder:

```bash
./gh200/download.sh --label <REPORT_LABEL>
```

Avoid this unless you intentionally want all artifacts:

```bash
./gh200/download.sh --all
```

Raw traces can become very large.

## 10. Updating Code After A Local Change

Recommended loop:

```bash
# Mac (run from repo root)
git status
git add <changed files>
git commit -m "<message>"
git push origin main
export AGENTIC_GH200_REMOTE_DIR="/home/central/ojaiyeob/agentic_hardware"
./gh200/sync_to_gh200.sh

# GH200
cd ~/agentic_hardware
./gh200/run_sentinel.sh
```

If only Python scripts/configs changed, syncing is enough.

If `requirements.txt` changed, run on GH200:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate
pip install -r requirements.txt
```

If package structure changed, run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate
pip install -e .
```

## 11. Common Overrides

Run only one harness:

```bash
HARNESSES=hatcher ./gh200/run_apples_to_apples.sh
```

Run only one pressure level:

```bash
PRESSURE_LEVELS=p3_high ./gh200/run_apples_to_apples.sh
```

Run only baseline and gateway-injected priority:

```bash
SIGNAL_FAMILIES="baseline gateway_injected" ./gh200/run_apples_to_apples.sh
```

Use a different model:

```bash
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct ./gh200/run_scaled_pressure.sh
```

Use a different model cache:

```bash
AGENTIC_GH200_MODEL_CACHE=/path/to/model_cache ./gh200/run_scaled_pressure.sh
```

## 12. Known GH200 Notes

- GH200 is ARM64 (`aarch64`), so do not copy venvs from EC2 or Mac.
- SGLang runs in Docker because the GH200 CUDA/runtime setup is cleaner there.
- Harnesses run on the host so NAT and Hermes can use Python 3.11.
- The SGLang Docker image currently expects:

```bash
EXTRA_SERVER_ARGS="--disable-cuda-graph --disable-overlap-schedule"
```

The GH200 wrappers set that by default.

- The `gh200/run_*.sh` wrappers are thin entry points that delegate to
  `sglang_direct_kv/scripts/run_harness_signal_design_space.sh`. Both
  must be present on GH200 (via sync) before GPU runs work.
- Use `INSTALL_SYSTEM_DEPS=0` when re-running `setup_gh200.sh` on the current
  GH200 image to avoid a known DKMS package conflict with older NVIDIA kernel
  modules (`nvidia-dkms-550-open` residuals).
- The machine may already have long-running Docker containers (`dynamo-frontend`,
  `dynamo-nats`, `etcd`). These are unrelated to the SGLang GPU experiment
  containers and should be left running.
