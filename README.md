# Hint-Guided KV Cache Prefetching for Agentic AI Workloads

## Goal

Build a realistic proof of concept showing that agent/runtime hints can reduce KV cache resume stalls in agentic LLM workloads............

The key question:

> When an agent is paused on a tool call, can the runtime use that pause to prefetch the agent's KV cache back into GPU memory before the next model turn arrives?

This prototype does not require new GPU hardware. Instead, it emulates hardware-assisted behavior in software using real LLM serving, real KV tensors, real tool gaps, and real GPU memory pressure.

See also: [Hardware Emulation Environment](HARDWARE_EMULATION_ENVIRONMENT.md)

Direct SGLang testbed: [sglang_direct_kv](sglang_direct_kv/README.md)

Replay path instrumentation roadmap: [Replay Path Instrumentation Proposal](REPLAY_PATH_INSTRUMENTATION_PROPOSAL.md)

Realistic AgentBench/DeepAgents path: see Milestones 16-19 in
[sglang_direct_kv](sglang_direct_kv/README.md#milestone-16-agentbench--sglang-direct).

## Current Experiment Entry Points

The detailed milestone log lives in
[sglang_direct_kv/README.md](sglang_direct_kv/README.md). That file is the
long-form lab notebook. This top-level README is the quick GitHub landing page
for reproducing the newest results.

Latest generated report, after a run:

```text
sglang_direct_kv/artifacts/results/latest_master_report.html
```

Latest generated manifest:

```text
sglang_direct_kv/artifacts/results/latest_manifest.json
```

### Environment

On the EC2 machine used for the current experiments:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate
```

The helper scripts for syncing and connecting to EC2 are documented in
[aws/README.md](aws/README.md). From a local checkout, the common commands are:

```bash
./aws/upload.sh 0
./aws/ssh_to_ec2.sh 0
./aws/download.sh 0
```

### Most Recent Experiment: Harness Deadline Pressure

This is the newest manager-facing experiment. It compares the same SGLang
priority boundary across multiple agent harnesses:

```text
Hatcher / Deep Agents-style control
Codex CLI
Claude Code CLI
```

It runs each harness in two modes:

```text
no_prefetch
e2e_priority_hints
```

And it uses three sentinel pressure levels:

| Pressure Level | Meaning | Knobs |
| --- | --- | --- |
| `p0_control` | Easy baseline | `500 ms` tool wait, `1024` target prompt tokens, `0` fillers, `1` urgent agent |
| `p3_high` | Single urgent replay under queue pressure | `50 ms` tool wait, `4096` target prompt tokens, `32` fillers, `1` urgent agent |
| `p5_boss_queue` | Many urgent replays under pressure | `50 ms` tool wait, `4096` target prompt tokens, `4` fillers per session, `4` urgent agents |

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESSES="hatcher codex claude_code" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_LABEL="multi_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh) | Orchestrates the multi-harness P0/P3/P5 experiment and writes the latest report. |
| [sglang_direct_kv/scripts/run_multi_harness_replay_driver.py](sglang_direct_kv/scripts/run_multi_harness_replay_driver.py) | Generates the target replay/filler traffic for Hatcher, Codex, and Claude Code. |
| [sglang_direct_kv/scripts/harness_sglang_gateway.py](sglang_direct_kv/scripts/harness_sglang_gateway.py) | Normalizes harness requests at the SGLang boundary and injects priority metadata. |
| [sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) | Builds the master HTML report, evidence tables, and Replay Deadline Pressure Chart. |
| [sglang_direct_kv/scripts/run_sglang_hicache_server.sh](sglang_direct_kv/scripts/run_sglang_hicache_server.sh) | Launches SGLang with HiCache, priority scheduling, and runtime telemetry flags. |

Current run label:

```text
multi_harness_deadline_pressure_20260831_214134
```

Headline result from that run: end-to-end priority hints helped substantially
under pressure, especially for Codex and Claude Code, but P3/P5 still missed
deadlines by seconds. That suggests priority metadata helps queue admission, but
does not fully solve backend GPU/KV pressure.

### Hatcher Pressure Ladder

This is the compact single-harness pressure ladder used before the
multi-harness comparison. It focuses on the current Hatcher / Deep Agents-style
control harness and answers: "as pressure increases, when do replay deadlines
start failing?"

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

PRESSURE_LEVELS="p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_LABEL="hatcher_pressure_ladder_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_hatcher_pressure_ladder.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Faster sentinel version:

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

### Priority Queue Proof

This is the small pre-flight proof that SGLang can admit a high-priority request
ahead of older low-priority work when priority scheduling is enabled.

Main run command:

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

### Controlled Replay With Priority Modes

This is the older controlled experiment family used to compare:

```text
no_prefetch
dynamo_priority_hints
e2e_priority_hints
projected_hardware_bypass
```

The current manager-facing path usually excludes `dynamo_priority_hints` and
`projected_hardware_bypass` unless the question specifically needs them.

Main run command:

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

### Milestone Map

| Milestone | What It Proves | Primary Entry Point |
| --- | --- | --- |
| M16-M19 | AgentBench/DeepAgents prompts can drive SGLang directly and produce manager-facing reports. | [sglang_direct_kv/README.md#milestone-16-agentbench--sglang-direct](sglang_direct_kv/README.md#milestone-16-agentbench--sglang-direct) |
| M27 | Controlled replay over real/synthetic coding-agent prompts. | [run_milestone27_real_prompt_controlled_replay.sh](sglang_direct_kv/scripts/run_milestone27_real_prompt_controlled_replay.sh) |
| M29-M35 | Replay-path ledger, exact KV movement attribution, H2D pressure, delay breakdown, and evidence audit. | [build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) |
| M36-M37 | Multi-session replay forensics and GPU KV pool residency telemetry. | [run_milestone36_multi_session_agentic_replay.sh](sglang_direct_kv/scripts/run_milestone36_multi_session_agentic_replay.sh) |
| M38-M40 | Dynamo-style priority hints, priority queue proof, and priority retention sanity checks. | [run_priority_queue_sanity.sh](sglang_direct_kv/scripts/run_priority_queue_sanity.sh) |
| Current | Multi-harness Replay Deadline Pressure Chart for Hatcher, Codex, and Claude Code. | [run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh) |

### Next Harness Batch

The next planned harness batch is:

```text
OpenCode
Qwen Code
```

The recommended path is to first add wireability support in
[harness_sglang_gateway.py](sglang_direct_kv/scripts/harness_sglang_gateway.py)
and [run_multi_harness_replay_driver.py](sglang_direct_kv/scripts/run_multi_harness_replay_driver.py),
then run the same P0/P3/P5 sentinel ladder through
[run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh).
NeMo Agent Toolkit should get a P0 wireability probe after that because it is a
heavier workflow framework rather than a simple coding-agent CLI.

## Workload Scenario

Use coding-agent-style workflows inspired by SWE-bench:

```text
LLM turn
-> tool call: search files / run tests / inspect error / edit code
-> tool wait
-> tool returns
-> next LLM turn
```

During the tool wait, the session's KV cache may be offloaded or evicted from fast GPU memory. When the tool returns, the next LLM turn needs that KV cache again.

If the KV cache is not resident in GPU memory, the agent stalls before first token.

## Three Evaluation Modes

### Mode 1: No Special Prefetch

The runtime does nothing during the tool gap.

Example:

```text
0 ms: Agent 42 starts run_tests()
20 ms: Agent 42 KV is offloaded from GPU memory
500 ms: run_tests() returns
500 ms: Agent 42 needs the model again
500-620 ms: KV is reloaded
620 ms: first token starts
```

This measures the baseline cost of cold KV cache resume.

### Mode 2: Generic Software Prefetch

The runtime uses ordinary software logic to prefetch KV during tool waits.

Example policy:

```python
if session.state == "tool_wait":
    prefetch(session.kv_blocks)
```

This mode is intentionally simple. It can copy KV blocks back to GPU memory, but it does not use rich session priority, deadlines, protection, or bandwidth-aware scheduling.

Example:

```text
0 ms: Agent 42 starts run_tests()
20 ms: Agent 42 KV is offloaded
200 ms: software sees Agent 42 is waiting
210 ms: software starts generic prefetch
350 ms: KV arrives in GPU memory
420 ms: KV is evicted again under HBM pressure
500 ms: run_tests() returns
500-620 ms: KV must be reloaded again
620 ms: first token starts
```

This answers:

> How much can ordinary software prefetch help?

### Mode 3: Hint-Guided KV Prefetch

The runtime emits structured hints, and a software prefetch manager emulates the proposed hardware behavior.

Example hint:

```python
hint = {
    "session_id": 42,
    "state": "tool_wait",
    "priority": "high",
    "expected_resume_ms": 500,
    "reuse_confidence": 0.85,
    "protect_after_prefetch_ms": 400,
    "throttle_if_decode_busy": True,
}
```

The hint-guided manager emulates hardware features:

- KV page/session tags
- priority-aware prefetch queue
- deadline-aware scheduling
- decode-aware bandwidth throttling
- temporary KV protection after prefetch
- telemetry for hits, misses, late prefetches, and wasted prefetches

Example:

```text
0 ms: Agent 42 starts run_tests()
10 ms: runtime submits hint: high priority, expected resume around 500 ms
50 ms: manager starts prefetch using spare bandwidth
180 ms: active decode gets busy, manager slows prefetch
260 ms: decode quiets down, manager resumes prefetch
330 ms: KV is back in GPU memory
330-500 ms: KV is protected from eviction
500 ms: run_tests() returns
505 ms: first token starts
```

This answers:

> Would semantic, hardware-style support make KV prefetch more reliable and efficient than generic software prefetch?

## Concrete Difference Between Mode 2 and Mode 3

Mode 2 says:

```text
Copy these KV blocks back to GPU.
```

Mode 3 says:

```text
This agent is likely to resume soon.
Its KV is high priority.
Prefetch it before the deadline.
Throttle around active decode.
Protect it after prefetch.
Track whether the hint helped.
```

Mode 2 is address/block based.

Mode 3 is intent based.

## Example Scheduling Case

Suppose three agents are waiting:

```text
Agent A:
  tool = run_tests
  expected return = 300 ms
  priority = high
  KV size = 2 GB

Agent B:
  tool = repo_search
  expected return = 2 seconds
  priority = medium
  KV size = 1 GB

Agent C:
  tool = long_build
  expected return = 20 seconds
  priority = low
  KV size = 4 GB
```

Generic software prefetch may prefetch in arrival order or after a fixed timeout.

Hint-guided prefetch should:

```text
1. Prefetch Agent A first.
2. Prefetch Agent B later if bandwidth and HBM allow.
3. Avoid prefetching Agent C too early.
```

## Metrics

Primary metrics:

- tool-return-to-first-token latency
- KV reload stall time
- P95 and P99 resume latency
- end-to-end agent task latency

Efficiency metrics:

- prefetch hit rate
- late prefetch rate
- wasted prefetch bandwidth
- prefetched-then-evicted rate
- active decode slowdown
- GPU memory pressure

## Expected Result

The expected result is not that hint-guided prefetch makes model decode faster.

The expected result is:

> Hint-guided KV prefetch makes agent resumption faster and more predictable under memory pressure.

The strongest gains should appear with:

- many concurrent agent sessions
- long contexts and large KV caches
- frequent tool calls
- bursty tool returns
- limited HBM capacity
- CPU/CXL/peer-GPU KV offload

## Research Claim

Software can decide prefetch policy, but hardware can make enforcement cheaper, faster, and more predictable.

This prototype emulates that future hardware behavior in software first. If the emulation shows meaningful improvements, it motivates hardware/runtime co-design for agent-aware KV cache prefetching.
