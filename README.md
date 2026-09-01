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

The current manager-facing comparisons use two modes:

| Mode | Meaning |
| --- | --- |
| `no_prefetch` | Baseline. The replay request receives no end-to-end priority treatment. |
| `e2e_priority_hints` | Current priority path. The driver marks replay urgency and the SGLang boundary carries that priority into scheduling. |

Older modes such as `dynamo_priority_hints`, `direct_prefetch`, and
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

## Most Recent Experiment: Real-Client Deadline Pressure

The newest experiment sends real coding-agent CLIs through the inspection
gateway, then forwards their requests to SGLang with normalized priority
metadata. This is stronger than the earlier adapter-only run because Codex,
Claude Code, OpenCode, and Qwen Code each generate their own live request shape.

Current archived run:

```text
sglang_direct_kv/artifacts/results/reports/real_client_deadline_pressure_20260901_035223/master_report.html
```

Native-only run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_LABEL="native_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_native_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Mixed native-plus-adapter run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESSES="codex claude_code opencode qwen_code" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_BUILDER_MODE=lightweight \
MAX_TOTAL_TOKENS=24576 \
REPORT_LABEL="real_client_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

If a long EC2 run is interrupted after some cases finish, resume the same
report label without rerunning completed cases:

```bash
SKIP_EXISTING_CASES=1 \
HARNESSES="codex claude_code opencode qwen_code" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_BUILDER_MODE=lightweight \
MAX_TOTAL_TOKENS=24576 \
REPORT_LABEL="<existing_report_label>" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Headline result from the current run. Values are median first-replay-token
lateness, so lower is better and anything above `0 ms` missed the replay
deadline:

| Harness | P0 no-prefetch | P0 E2E | P3 no-prefetch | P3 E2E | P5 no-prefetch | P5 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Codex | `1.18 s` | `1.35 s` | `47.31 s` | `5.79 s` | `51.91 s` | `22.81 s` |
| Claude Code | `1.40 s` | `1.46 s` | `63.26 s` | `10.01 s` | `49.56 s` | `18.66 s` |
| OpenCode | `3.76 s` | `3.94 s` | `73.99 s` | `14.23 s` | `66.79 s` | `28.97 s` |
| Qwen Code | `5.78 s` | `5.86 s` | `73.24 s` | `20.77 s` | `92.95 s` | `69.11 s` |

Interpretation: end-to-end priority hints helped every stressed real-client
harness, especially at P3. But the priority path still missed the tight replay
deadline under P3 and P5 because priority can move a replay earlier in the
queue; it cannot create extra GPU compute, KV capacity, or host-to-device
bandwidth. The P0 rows are also above zero because real CLIs add their own
startup/protocol overhead around the backend call.

## Multi-Harness Deadline Pressure

The broad experiment compares the same SGLang priority boundary across ten
non-Dynamo agent harness shapes. Some entries now launch real native CLIs; the
remaining entries are still adapter-backed until their native client can be
installed and smoke-tested cleanly on the experiment host.

| Harness | Current experiment path |
| --- | --- |
| `hatcher` | In-repo Hatcher / Deep Agents-style control harness. |
| `codex` | Real Codex CLI through the inspection gateway. |
| `claude_code` | Real Claude Code CLI through the inspection gateway. |
| `opencode` | Real OpenCode CLI through the inspection gateway. |
| `qwen_code` | Real Qwen Code CLI through the inspection gateway. |
| `pi_agent_harness` | Real Pi CLI with a generated OpenAI-compatible gateway provider extension. |
| `openclaw` | Real OpenClaw CLI with a generated OpenAI-compatible gateway provider config. |
| `nemo_agent_toolkit` | Adapter-backed; native NAT package is blocked on the current EC2 Python/package constraints. |
| `deepseek_harness` | Adapter-backed; native `dsh` CLI currently hangs during headless `--help` / `--version` probing. |
| `hermes_agent` | Adapter-backed on the current EC2 image; Hermes requires Python 3.11+, while the EC2 host is Python 3.9. |

It runs each harness in:

```text
no_prefetch
e2e_priority_hints
```

And uses the sentinel pressure levels:

```text
p0_control p3_high p5_boss_queue
```

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESSES="hatcher codex claude_code opencode qwen_code nemo_agent_toolkit deepseek_harness pi_agent_harness openclaw hermes_agent" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_LABEL="multi_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

The lightweight report shows the Replay Deadline Pressure Chart as a
pressure-first overlay: P0/P3/P5 are the main x-axis sections, color separates
`no_prefetch` from `e2e_priority_hints`, and symbol shape separates harnesses.

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh) | Orchestrates the multi-harness pressure experiment and writes the latest report. |
| [sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh) | Runs only the native CLI harnesses plus the Hatcher control. |
| [sglang_direct_kv/scripts/run_multi_harness_replay_driver.py](sglang_direct_kv/scripts/run_multi_harness_replay_driver.py) | Generates target replay and filler traffic for the selected harnesses. |
| [sglang_direct_kv/scripts/harness_sglang_gateway.py](sglang_direct_kv/scripts/harness_sglang_gateway.py) | Normalizes harness requests at the SGLang boundary and injects priority metadata. |
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

Recent EC2 wireability result:

```text
sglang_direct_kv/artifacts/results/real_client_wireability/real_client_probe_20260901_031658/real_client_wireability_report.html
```

Latest local six-client wireability smoke, run on 2026-09-01, launched the real
Codex, Claude Code, OpenCode, Qwen Code, Pi, and OpenClaw CLIs against the
inspection gateway. All six reached the gateway, all six were tagged with
`sglang_priority=100`, and the gateway recorded their live request shape without
storing prompt bodies:

| Client | API shape | Request body | Prompt chars |
| --- | --- | ---: | ---: |
| Codex | `/v1/responses` | `164.9 KB` | `7.7K` |
| Claude Code | `/v1/messages?beta=true` | `5.2 KB` | `1.6K` |
| OpenCode | `/v1/chat/completions` | `3.6 KB` | `3.2K` |
| Qwen Code | `/v1/chat/completions` | `97.4 KB` | `36.3K` |
| Pi Agent Harness | `/v1/chat/completions` | `1.5 KB` | `1.2K` |
| OpenClaw | `/v1/chat/completions` | `29.0 KB` | `20.8K` |

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
| NeMo Agent Toolkit / NAT | Adapter-backed. | Re-probe on a Python 3.11+ EC2 image or install-compatible NAT environment. |
| DeepSeek Harness | Adapter-backed. | Re-probe after the `dsh` CLI exposes a reliable headless command path. |
| Hermes Agent | Adapter-backed on current EC2. | Re-probe after the EC2 image has Python 3.11+. |

The next recommended path is:

1. Run the P0/P3/P5 sentinel ladder across the six native harness paths through [run_native_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh).
2. Rebuild the EC2 Python side with Python 3.11+ before promoting NAT and Hermes from adapters to real clients.
3. Keep DeepSeek Harness adapter-backed until `dsh` can run a non-interactive smoke request without hanging.

## Current Research Claim

Priority hints are useful: they express urgency and can improve scheduler
admission. But software priority alone does not guarantee replay-deadline
readiness under heavy GPU compute pressure, KV-cache pressure, short tool waits,
or many urgent agents arriving together.

That gap is the hardware/runtime opportunity: make replay-critical KV residency,
movement, and admission cheaper, deadline-aware, and enforceable.
