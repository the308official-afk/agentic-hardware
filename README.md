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

## Most Recent Experiment: Multi-Harness Deadline Pressure

The newest experiment compares the same SGLang priority boundary across three
agent harness shapes:

| Harness | Meaning |
| --- | --- |
| `hatcher` | Current in-repo Hatcher / Deep Agents-style control harness. |
| `codex` | Codex-style coding-agent traffic shape. |
| `claude_code` | Claude Code-style coding-agent traffic shape. |
| `opencode` | OpenCode-style coding-agent traffic shape. |
| `qwen_code` | Qwen Code-style coding-agent traffic shape. |
| `nemo_agent_toolkit` | NeMo Agent Toolkit / NAT-style workflow adapter. |
| `deepseek_harness` | DeepSeek Harness-style provider adapter. |
| `pi_agent_harness` | Pi Agent Harness-style provider adapter. |
| `openclaw` | OpenClaw-style provider adapter. |
| `hermes_agent` | Hermes Agent-style provider adapter. |

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

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh) | Orchestrates the multi-harness pressure experiment and writes the latest report. |
| [sglang_direct_kv/scripts/run_multi_harness_replay_driver.py](sglang_direct_kv/scripts/run_multi_harness_replay_driver.py) | Generates target replay and filler traffic for the selected harnesses. |
| [sglang_direct_kv/scripts/harness_sglang_gateway.py](sglang_direct_kv/scripts/harness_sglang_gateway.py) | Normalizes harness requests at the SGLang boundary and injects priority metadata. |
| [sglang_direct_kv/scripts/smoke_multi_harness_wireability.py](sglang_direct_kv/scripts/smoke_multi_harness_wireability.py) | Fast local smoke test for CLI harness wireability through the gateway. |
| [sglang_direct_kv/scripts/run_sglang_hicache_server.sh](sglang_direct_kv/scripts/run_sglang_hicache_server.sh) | Launches SGLang with HiCache, priority scheduling, and runtime telemetry flags. |
| [sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) | Builds the master HTML report, evidence tables, and Replay Deadline Pressure Chart. |

Smoke-test only the newest harness adapters without starting the real GPU
server:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/smoke_multi_harness_wireability.py \
  --harnesses pi_agent_harness openclaw hermes_agent
```

Current archived run:

```text
sglang_direct_kv/artifacts/results/reports/multi_harness_deadline_pressure_20260831_214134/master_report.html
```

Headline result from that run:

| Harness | P0 no-prefetch | P0 E2E | P3 no-prefetch | P3 E2E | P5 no-prefetch | P5 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hatcher / Deep Agents-style control | `215 ms` | `237 ms` | `16.1 s` | `10.8 s` | `50.5 s` | `26.5 s` |
| Codex | `1.18 s` | `1.15 s` | `41.8 s` | `10.1 s` | `41.7 s` | `6.6 s` |
| Claude Code | `1.29 s` | `1.32 s` | `72.0 s` | `12.2 s` | `50.2 s` | `27.0 s` |

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

Recently added:

```text
OpenCode
Qwen Code
NeMo Agent Toolkit / NAT
DeepSeek Harness
Pi Agent Harness
OpenClaw
Hermes Agent
```

The NAT, DeepSeek Harness, Pi Agent Harness, OpenClaw, and Hermes Agent entries
are currently smoke-level wireability adapters. They prove the SGLang boundary
can receive their harness-shaped replay traffic and priority metadata. Full
native framework runs can be added once we want them in the pressure chart.

The next recommended path is:

1. Run the same P0/P3/P5 sentinel ladder across all ten non-Dynamo harness adapters through [run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh).
2. Replace the smoke-level framework adapters with full native profile invocations if those frameworks become part of the manager-facing pressure comparison.

## Current Research Claim

Priority hints are useful: they express urgency and can improve scheduler
admission. But software priority alone does not guarantee replay-deadline
readiness under heavy GPU compute pressure, KV-cache pressure, short tool waits,
or many urgent agents arriving together.

That gap is the hardware/runtime opportunity: make replay-critical KV residency,
movement, and admission cheaper, deadline-aware, and enforceable.
