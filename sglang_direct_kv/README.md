# SGLang Direct KV Instrumentation Testbed

## Goal

Build a direct SGLang-based testbed for hint-guided KV cache prefetching.

This project intentionally starts with SGLang rather than fake KV tensors. The goal is to find and instrument the real SGLang KV/cache/offload path, then emulate future hardware support in software.

## Table Of Contents

| Section | Status | Link |
| --- | --- | --- |
| Key findings so far | Active | [Key Findings So Far](#key-findings-so-far) |
| Manager-facing claims / research framing | Active | [Manager-Facing Claims / Research Framing](#manager-facing-claims--research-framing) |
| Milestone 0: Testbed Scaffold | Completed | [Milestone 0](#milestone-0-testbed-scaffold---completed) |
| Milestone 1: SGLang Internals Map | Completed | [Milestone 1](#milestone-1-sglang-internals-map---completed) |
| Milestone 2: Real SGLang + HiCache Smoke Test | Completed | [Milestone 2](#milestone-2-real-sglang--hicache-smoke-test---completed) |
| Milestone 3: Log Real KV Movement | Completed | [Milestone 3](#milestone-3-log-real-kv-movement---completed) |
| Milestone 4: Force Load/Evict And Add Direct Hint Hooks | Completed | [Milestone 4](#milestone-4-force-loadevict-and-add-direct-hint-hooks---completed) |
| Milestone 5: Compare Three Modes | Completed | [Milestone 5](#milestone-5-compare-three-modes---completed) |
| Milestone 6: Design-Space Sweep | Completed | [Milestone 6](#milestone-6-design-space-sweep) |
| Milestone 7: Direct KV Movement Hooks | Completed | [Milestone 7](#milestone-7-direct-kv-movement-hooks) |
| Milestone 7C: Session To Host Indices Mapping | Completed | [Milestone 7C](#milestone-7c-session-to-host-indices-mapping) |
| Milestone 7D: Direct Load-Back Trigger | Completed | [Milestone 7D](#milestone-7d-direct-load-back-trigger) |
| Milestone 8: Direct Load Design-Space Sweep | Completed | [Milestone 8](#milestone-8-direct-load-design-space-sweep) |
| Milestone 9: Multi-Session Agent Traffic | Completed | [Milestone 9](#milestone-9-multi-session-agent-traffic-with-hint-outcome-analysis) |
| Milestone 10: DMA Timeline Profiling | Completed | [Milestone 10](#milestone-10-dma-timeline-profiling) |
| Milestone 10B: CUDA Transfer To KV Block Attribution | Completed | [Milestone 10B](#milestone-10b-cuda-transfer-to-kv-block-attribution---completed) |
| Milestone 11: Agentic Prefetch Timeline Experiment | Completed | [Milestone 11](#milestone-11-agentic-prefetch-timeline-experiment) |
| Milestone 11B: Improved CUDA Copy Attribution | Completed | [Milestone 11B](#milestone-11b-improved-cuda-copy-attribution) |
| Milestone 11C: Profiler Coverage Diagnosis | Completed | [Milestone 11C](#milestone-11c-profiler-coverage-diagnosis) |
| Milestone 12: Paired Clean + Attribution Evidence | Ready | [Milestone 12](#milestone-12-paired-clean--attribution-evidence) |
| Milestone 13: Failure Stress Experiment | Completed | [Milestone 13](#milestone-13-failure-stress-experiment) |
| Milestone 13B: Green-Bar Failure Stress | Completed | [Milestone 13B](#milestone-13b-green-bar-failure-stress) |
| Milestone 14: Lightweight KV Copy Telemetry | Completed | [Milestone 14](#milestone-14-lightweight-kv-copy-telemetry) |
| Milestone 15: Targeted DMA/HtoD Validation | Completed | [Milestone 15](#milestone-15-targeted-dmahtod-validation) |
| Milestone 16: AgentBench -> SGLang Direct | Ready | [Milestone 16](#milestone-16-agentbench--sglang-direct) |
| Milestone 17: Real Trace Replay Workload | Ready | [Milestone 17](#milestone-17-real-trace-replay-workload) |
| Milestone 18: Real Prompt Prefetch Modes | Ready | [Milestone 18](#milestone-18-real-prompt-prefetch-modes) |
| Milestone 19: Realistic Manager Report | Ready | [Milestone 19](#milestone-19-realistic-manager-report) |
| Milestone 20: SWE-bench Trajectory Prompt Replay | Ready | [Milestone 20](#milestone-20-swe-bench-trajectory-prompt-replay) |
| Milestone 21: Direct SGLang Experiment 6 Prompt Evolution | Ready | [Milestone 21](#milestone-21-direct-sglang-experiment-6-prompt-evolution) |
| Milestone 22: Live AgentBench Tool-Gap Bridge | Ready | [Milestone 22](#milestone-22-live-agentbench-tool-gap-bridge) |
| Milestone 23: Live Prefetch Intervention | Ready | [Milestone 23](#milestone-23-live-prefetch-intervention) |
| Milestone 24: Live Paired AgentBench Report | Ready | [Milestone 24](#milestone-24-live-paired-agentbench-report) |
| Milestone 25: Labeled Reproducible Master Reports | Ready | [Milestone 25](#milestone-25-labeled-reproducible-master-reports) |
| Milestone 26: Live Direct KV Load Intervention | Ready | [Milestone 26](#milestone-26-live-direct-kv-load-intervention) |
| Milestone 27: Real-Prompt Controlled Replay | Ready | [Milestone 27](#milestone-27-real-prompt-controlled-replay) |
| Milestone 28: Hardened Master Report Workflow | Ready | [Milestone 28](#milestone-28-hardened-master-report-workflow) |
| Milestone 29: Replay Path Instrumentation Ledger | Ready | [Milestone 29](#milestone-29-replay-path-instrumentation-ledger) |
| Milestone 29B: Forced Eviction Sanity Probe | Ready | [Milestone 29B](#milestone-29b-forced-eviction-sanity-probe) |
| Milestone 30: Stable KV Block Ledger | Ready | [Milestone 30](#milestone-30-stable-kv-block-ledger) |
| Milestone 31: Exact KV Movement Attribution | Ready | [Milestone 31](#milestone-31-exact-kv-movement-attribution) |
| Milestone 32: KV H2D Bandwidth Pressure | Ready | [Milestone 32](#milestone-32-kv-h2d-bandwidth-pressure) |
| Milestone 33: Replay Delay Breakdown | Ready | [Milestone 33](#milestone-33-replay-delay-breakdown) |
| Milestone 34: Replay Delay Deep Instrumentation | Ready | [Milestone 34](#milestone-34-replay-delay-deep-instrumentation) |
| Milestone 35: Instrumentation Evidence Audit | Ready | [Milestone 35](#milestone-35-instrumentation-evidence-audit) |
| Milestone 36: Multi-Session Agentic Replay Forensics | Ready | [Milestone 36](#milestone-36-multi-session-agentic-replay-forensics) |
| Milestone 37: GPU KV Pool Residency Telemetry | Ready | [Milestone 37](#milestone-37-gpu-kv-pool-residency-telemetry) |
| Milestone 38: Dynamo Priority Hints And Projected Hardware | Ready | [Milestone 38](#milestone-38-dynamo-priority-hints-and-projected-hardware) |
| Milestone 38B: Dynamo Priority Hint Bridge | Ready | [Milestone 38B](#milestone-38b-dynamo-priority-hint-bridge) |
| Milestone 39: Projected Hardware Bypass Benefit | Ready | [Milestone 39](#milestone-39-projected-hardware-bypass-benefit) |

## Milestone Run Command Convention

Each new milestone section should include a `Main run command` block.
That block is the command a colleague should run first to reproduce the milestone.

If a milestone has extra smoke tests or alternate workflows, those can appear after
the main command, but the `Main run command` should remain the primary handoff path.

## What We Are Testing

Agentic workloads naturally create wait windows:

```text
model turn
-> tool call: run tests / search repo / build
-> wait
-> tool returns
-> next model turn
```

During the wait, the runtime often knows:

```text
this session may resume soon
this session has priority
this session's KV should be ready before the next turn
```

The testbed will compare:

```text
Mode 1: no prefetch
Mode 2: generic software prefetch
Mode 3: hint-aware direct KV prefetch/protection
```

## Key Findings So Far

The experiments so far show that agentic hints are useful, but software-issued hints are not enough by themselves.
Under overlapping agent traffic, hints can complete too late, or complete early and still fail because the KV is not protected from eviction.
The measured KV load calls are short compared with end-to-end hint completion, which suggests the issue is not only raw memory copy speed.
The stronger argument is that current runtime/GPU memory paths lack deadline-aware, priority-aware, and residency-aware enforcement for agentic KV prefetch.

### Manager-Facing Claims / Research Framing

These are the current core claims we can safely make from the testbed so far:

1. Our results suggest that agentic prefetch is not just a bandwidth problem.
   Under high KV-pool pressure, software hints can be issued but still fail to
   create useful residency before replay. This motivates hardware/runtime
   support that can make KV movement and residency enforceable: priority-aware
   migration, eviction choice, temporary protection, and telemetry showing
   whether the hinted KV actually became resident in time.

2. The core opportunity is not simply to prefetch earlier. It is to make sure
   the right KV is resident, protected, and reusable when the agent resumes.

3. Existing software can decide which sessions are likely to resume, but
   today's memory movement path does not cheaply enforce those decisions under
   contention.

4. A generic DMA engine can move memory, but it does not know that a transfer
   is urgent KV for a soon-resuming agent session.

5. High KV-pool pressure makes agentic prefetch a residency-management problem,
   not just a copy-bandwidth problem.

6. Current GPU DMA/copy engines can move memory efficiently, but they are mostly blind to agent context. They do not know that a transfer is urgent KV for a session expected to resume soon. A deadline-aware, agent-aware DMA path could prioritize the right KV transfers, throttle lower-priority movement, and expose whether the KV became resident before replay. This can reduce replay stalls and improve tail latency for tool-heavy agentic workloads.

7. We are not proposing generic prefetch. We are proposing an agent-aware KV movement path where the runtime provides deadline and priority hints, and the GPU memory/copy subsystem enforces them. The DMA engine becomes aware that some copies are urgent replay-critical KV, while others are background movement. This should reduce late KV loads, wasted prefetches, replay-side reloads, and TTFT tail latency in coding-agent/SWE-bench-style workloads.

8. Simply forcing priority in software can be expensive. The Milestone 38 deadline-priority emulation gave urgent KV work special treatment, but it still had to pass through SGLang's normal software/runtime path. That path added enough overhead that replay TTFT became worse in the latest stress run. This strengthens the hardware argument: we need a faster, lower-overhead, deadline-aware KV movement path, not just a software queueing trick.

9. The Dynamo priority-hint bridge lets us test a stronger software baseline:
   agent hints are emitted as `custom_params.nvext.agent_hints` and translated
   into SGLang's native `priority` field. If this still misses replay deadlines
   under KV-pool pressure, the claim becomes sharper: existing priority hints
   help express intent, but they still need a low-overhead KV movement and
   residency enforcement path closer to the memory system.

Latest harsher controlled run:

```text
forced_eviction_harsh_valid_target4096_f128_1_reclassified
```

This run used one real AgentBench prompt pair, 128 early-diverging filler requests,
100 ms tool wait, 2048-token filler prompts, and a 4096-token target prompt.
The replay was valid under SGLang's token budget and showed strong pressure:

```text
Replay TTFT:                      13551.504 ms
Replay input tokens:              13383
Initial replay cache match:       24 tokens
Final cached prefix after replay: 13383 tokens
Estimated replay prefill/recompute: 13359 tokens
Replay HtoD KV events observed:   0
```

Important interpretation: the full prefix existed after replay/cache work
progressed, but it was not a clean cache hit at replay start. This is why the
report now separates initial cache match from final post-replay cache state.

Latest synthetic large-prefix HiCache finding:

```text
synthetic_large_prefix_probe_1
```

HiCache was enabled and SGLang allocated an 8 GB host KV cache. The target
first turn wrote about 4213 KV tokens to host. Under 64/128 filler pressure,
the trace then showed target device-side eviction and target host-side eviction
before replay. At replay time, SGLang observed no target H2D load-back events,
matched only a tiny prefix, and rebuilt/prefilled the missing tokens.

Simple interpretation:

```text
HiCache existed.
The target KV was written to host.
But useful target host KV was evicted before replay.
So replay could not load it back and recomputed instead.
```

The master report now includes `KV Lifecycle Evidence` and `KV Block Ledger`
sections under the timeline. For each timeline row, it shows host writes, GPU
evictions, host evictions, replay H2D loads, replay prefix match, estimated
replay prefill, and logical KV blocks lost before replay.

Open this report first:

```text
artifacts/results/latest_master_report.html
```

Use this for controlled synthetic experiments:

```text
artifacts/results/latest_synthetic_master_report.html
```

Milestone-specific latest files now live inside detail folders:

```text
artifacts/results/latest_real/          real live experiment details
artifacts/results/latest_synthetic/     synthetic experiment details
```

The results-folder root is intentionally kept simple. The only top-level latest
HTML reports should be:

```text
latest_master_report.html              real SWE-bench / AgentBench live report
latest_synthetic_master_report.html    controlled synthetic report
```

### Real Live Experiment Setup

The latest real master report uses this setup:

```text
SWE-bench / AgentBench tasks
  -> DeepAgents harness
  -> tool-calling loop
     read_file, edit_file, ls, grep, execute, write_file
  -> SGLang OpenAI-compatible server
  -> Qwen Coder model + KV cache
  -> observed model turns, tool gaps, and resume requests
```

The live prefetch path runs beside the normal request path:

```text
tool-call response observed
  -> hint emitted: this session may resume soon
  -> live prefetch controller
  -> SGLang prefetch/direct-load style request
```

How the real experiment is conducted:

```text
1. Run real SWE-bench / AgentBench-style tasks through DeepAgents.
2. Let DeepAgents generate real structured tool calls.
3. Send model requests directly to SGLang, without Dynamo.
4. Compare no-prefetch against live software prefetch.
5. Pair tool gaps by SWE-bench task index and gap order.
6. Measure tool-gap duration, prefetch duration, prefetch margin, resume request latency, and late prefetch count.
```

Manager-facing interpretation:

```text
The traffic is real live agent traffic, not synthetic prompts.
Tool gaps can be very short.
The software prefetch path can take much longer than the available tool gap.
Most missed prefetches mean the runtime had the semantic hint, but the ordinary software/SGLang path did not finish in time.
This motivates hardware/runtime support for deadline-aware KV movement, residency protection, and telemetry.
```

The synthetic master report also includes a setup diagram and manager-facing setup section. Its setup is controlled rather than live:

```text
synthetic agentic driver
  -> initial request, controlled tool wait, replay request
  -> SGLang OpenAI-compatible server
  -> Qwen model + KV cache
  -> clean performance results and profiled KV/copy attribution
```

Use the two master reports this way:

```text
latest_master_report.html              real SWE-bench / DeepAgents traffic
latest_synthetic_master_report.html    controlled synthetic stress traffic
```

Most important timing insight so far:

```text
Avg hint total duration: ~1284-1345 ms
Total measured hicache.load time across all sessions: ~10-13 ms
```

This means the long delay is mostly not inside the raw `hicache.load` call.
It is likely dominated by scheduling, queueing, runtime bookkeeping, and contention with active inference work.

| ID | Finding | Evidence | Why It Matters | Hardware/Runtime Implication | Status |
| --- | --- | --- | --- | --- | --- |
| F1 | Software hints can reduce replay TTFT. | Milestone 9B: `direct_load` improved average replay TTFT by about `147 ms` vs `no_prefetch`. | The opportunity is real; agent wait windows can be used. | Keep the frontend/runtime hint path and make it more enforceable. | Observed |
| F2 | Hints can complete too late under load. | `oracle_direct_load` with `500 ms` and `1000 ms` lead still produced `late_prefetch: 12`. | Even good timing estimates do not guarantee KV readiness. | Need deadline-aware scheduling and priority-aware migration. | Strong |
| F3 | Starting earlier is not sufficient. | `1500 ms` oracle lead produced `late_prefetch: 10` and `too_early_or_unprotected: 2`. | More lead time can trade late prefetch for eviction-before-reuse. | Need residency protection, not just earlier prefetch. | Strong |
| F4 | Raw KV load time is not the main observed delay. | Oracle lead sweep: average hint duration was about `1284-1345 ms`, while total measured hint-side `hicache.load` time was only about `10-13 ms` across all sessions. | The long delay is mostly outside the raw KV load call. | Need a prioritized prefetch path that avoids normal request scheduling/queueing delays. | Strong |
| F5 | Correct prefetches can still be wasted. | `too_early_or_unprotected` sessions had both prefetch-side loads and replay-side loads, plus eviction pressure between hint and replay. | A correct hint can fail if the prefetched KV is not kept resident. | Need protected/pinned residency windows for prefetched KV. | Strong |
| F6 | Prompt-based warming is retired from the active testbed. | Earlier milestones used a normal warm request as a software-only baseline, but the current code path disables it and uses direct SGLang KV load-back hooks for prefetch experiments. | This keeps the evidence focused on real KV movement rather than prompt tricks. | Compare `direct_load` against `no_prefetch`; use old prompt-warming data only as historical background. | Updated |
| F7 | Oracle timing currently improves TTFT but still misses deadlines under pressure. | `ORACLE_LEAD_MS=1500` improved replay TTFT by about `219 ms`, but still produced `late_prefetch: 10`. | Better timing helps, but the prefetch path is not predictable enough. | Hardware/runtime support should expose deadline and progress telemetry. | Strong |
| F8 | Worker-local profiling can expose SGLang CUDA activity. | Milestone 10 torch-profiler smoke exported one worker trace with `14329` kernel-like events, `5569` memcpy-like events, `70` HtoD events, and `2229` DtoH events. | We now have a path to inspect GPU activity from inside the SGLang worker. | Use worker-local profiling for DMA/copy evidence when external Nsight misses worker GPU activity. | New |
| F9 | Agentic timelines can show where lateness happens. | Milestone 11 smoke produced SGLang KV windows, profiler CUDA copy rows, hint outcome labels, and a per-session HTML timeline. | We can now show whether the hint was late at the request level, the KV-load level, or the CUDA-copy level. | This is the clearest evidence path for motivating deadline-aware prefetch hardware. | New |
| F10 | CUDA copy attribution needs explicit agent context. | Milestone 11B changed `agent_001`, `agent_002`, and `agent_005` from SGLang-only rows into rows with profiler-visible HtoD copy windows. | Missing HtoD columns were mostly attribution weakness, not proof that no copy happened. | Carry agent/session context deeper into SGLang KV movement events and preserve batched session ownership. | New |
| F11 | Missing green bars can be profiler-coverage artifacts. | In Milestone 11B, `agent_003` had SGLang host-to-device load from `44435.564 -> 44496.473 ms`, but the torch profiler stopped at `44407.848 ms`. In Milestone 11C, a later profiler stop showed `agent_003` with `528` HtoD events. | "No green bar" does not always mean "no CUDA copy." It can mean the profiler was not recording when the copy happened. | Reports now include profiler window status and missing-HtoD reason columns. | New |
| F12 | Early CUDA copy is not enough for prefetch success. | In Milestone 11C300, CUDA copy was ready before replay for `6 / 6` sessions, but full hint completion succeeded for only `5 / 6`, replay reloaded KV for `6 / 6`, and clean success was `0 / 6`. | Even when CUDA copy happens before replay, the software-managed path can still fail because the full hint path is late, replay reloads KV again, or prefetched KV is not protected/reused predictably. | The need is not just "copy memory earlier"; the system must copy the right KV, finish predictably, protect residency, and make reuse visible/enforceable. | Strong |
| F13 | Failure-heavy stress shows software hints are not deadline-predictable. | Milestone 13 manager stress: clean `oracle_direct_load` improved avg replay TTFT by `629.573 ms` vs `no_prefetch`, but clean outcomes were `late_prefetch: 32 / 32`. Profiled attribution showed `cuda_ready=0 / 32`, `hint_done=0 / 32`, `replay_reloaded=32 / 32`, `clean_success=0 / 32`. | Hints can still help average latency while failing the actual deadline/residency requirement for every session under pressure. | This is the strongest case so far for deadline-aware migration, prefetch protection, and hardware-visible progress/telemetry. | Strong |
| F14 | Green-bar stress gives both CUDA-copy visibility and failure evidence. | Milestone 13B captured visible CUDA HtoD bars in `4 / 12` sessions while still showing `late_prefetch: 12 / 12`, `replay_reloaded=12 / 12`, and `clean_success=0 / 12`. | The report now shows the concrete host-to-device copy windows for some agents, while still proving that the software hint path missed every replay deadline under stress. | Use this report as the manager-facing visual bridge between SGLang-level KV movement and CUDA-level data movement. | Strong |
| F15 | Full torch-profiler traces do not scale to large timelines. | 32-session profiler traces can produce hundreds of thousands of unrelated CUDA/kernel events, while we only need per-agent KV movement windows. | Large profiler traces make reports slow and can still miss late sessions if the profiler window ends early. | Use lightweight SGLang KV-copy telemetry for large runs, and use short torch-profiler runs only to validate that this telemetry maps to real CUDA HtoD movement. | Strong |
| F16 | Lightweight KV-copy telemetry scales to all sessions in a 32-session stress run. | Milestone 14 captured lightweight KV-copy telemetry for `32 / 32` sessions with `0` torch-profiler files. The telemetry stream had `2848` compact rows. | We can now show green KV movement bars for larger traffic without collecting massive unrelated CUDA traces. | Use movement-only telemetry as the default evidence path, then use small torch-profiler runs as CUDA validation. | New |
| F17 | Targeted profiler validation can recover dark-green CUDA HtoD bars. | Milestone 15 captured `1` torch-profiler trace, `31941` CUDA events, and profiler-attributed HtoD rows for `3 / 6` sessions. For those sessions, the sanity table showed green copy bars were inside the purple hint window. | This validates that the lightweight SGLang KV-copy telemetry corresponds to lower-level CUDA HtoD activity in small runs. | Use Milestone 15 as the dark-green validation companion to Milestone 14's scalable light-green telemetry. | New |
| F18 | Direct software prefetch can be worse than no prefetch when it is late. | Latest clean paired report, `agent_000`: `no_prefetch` replay TTFT was `125.252 ms`, while `oracle_direct_load` replay TTFT was `234.449 ms`. Oracle direct load issued a hint, but `hint_completed_before_replay=0`, `hint_total_duration_ms=171.615`, and replay still loaded KV (`resume_load_count=10`, `resume_hicache_load_count=4`). | This shows that software prefetch is not automatically beneficial. A late hint can add work, compete with replay, and still fail to prevent replay-side KV loading. | Need deadline-aware migration, priority scheduling, residency protection, and telemetry so hints become enforceable rather than best-effort. | Strong |
| F19 | Real DeepAgents/SWE-bench tool gaps can be much shorter than the software prefetch path. | Milestone 22 bigger live run captured `12` real tool gaps across `4` SWE-bench tasks, with average gap about `11.5 ms` and max gap about `14.1 ms`. Milestone 23 live prefetch smoke matched `2` live prefetch attempts, but they took about `438 ms` and `498 ms`, with average prefetch margin about `-471 ms` and `0 / 2` finishing before resume. | This is a strong early live-traffic finding: the runtime can see useful tool-call hints, but the normal software/controller/SGLang request path is far too slow for very short resume windows. | Need a deadline-aware, hint-aware hardware/runtime path that can act on agent context quickly and predictably, instead of routing prefetch through ordinary best-effort serving work. | Strong |
| F20 | Low 7B tool-call counts were caused by harness/parser/tool-interface issues plus model weakness, not by lack of tool traffic in the workload. | Direct-SGLang debugging found three issues: Qwen2.5 needed `--tool-call-parser qwen25`, DeepAgents tools see the repo at `/` rather than the host checkout path, and the 7B model sometimes emits unsafe empty-string `edit_file` calls. After fixes, a 4-task run produced `93` real task model requests, `49` structured tool calls, `44` trajectory prompts, and `0` prose-only tool-intent misses. | The direct SGLang path is now useful for live tool-gap/KV experiments. The remaining gap versus prior 30B/40B runs is mostly model capability and batch size. | Keep the parser/root/safe-edit safeguards on for A10G 7B runs; use Qwen3-Coder 30B/40B-class models on compatible GPUs for manager-grade SWE-bench tool diversity. | Strong |
| F21 | Full live paired AgentBench traffic shows software prefetch usually misses the real tool-gap deadline. | Milestone 24 ran `START_INDEX=0`, `END_INDEX=15`, `AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10` twice: no-prefetch captured `267` analyzed live requests and `127` tool gaps; live-prefetch captured `254` analyzed live requests and `114` tool gaps. The live-prefetch run submitted `117` hints, matched `114` prefetch attempts, but only `2 / 114` finished before resume. Average tool gap was about `19.9 ms`, while average prefetch request duration was about `629.5 ms`; `112 / 114` attempts were late. | This is the strongest live-system evidence so far: the runtime can observe the agent/tool context, but the ordinary software/controller/SGLang request path is much slower than the real resume window. The paired run was slower on average by about `130 ms`, with `105` slower pairs and only `9` faster pairs. | Need a hint-aware, deadline-aware prefetch/migration path that does not compete as an ordinary best-effort request, plus residency protection and telemetry to make useful prefetch enforceable. | Strong |
| F22 | Dynamo priority hints and projected hardware test two different questions. | `dynamo_priority_hints` sends Dynamo-style priority metadata and an SGLang priority value, but does not issue our artificial direct KV prefetch hook. The new priority queue audit checks whether SGLang actually honors that priority in receive/queue/admission traces. `projected_hardware_bypass` estimates a lower-overhead memory path from measured H2D durations. | This keeps the current manager-facing comparison clean: priority hints today vs. a projected memory-system enforcement path. | If priority hints still miss deadlines but projected hardware could meet them, the issue is not just lack of hints; it is lack of a low-overhead memory-system enforcement path. | Strong |

Current strongest claim:

```text
In real tool-heavy DeepAgents/SWE-bench traffic, the runtime can observe useful
agentic hints, but a normal software prefetch path through SGLang is usually
too slow and unpredictable to meet post-tool resume deadlines.
```

Concrete evidence from Milestone 24:

```text
Average live tool gap: ~19.9 ms
Average software prefetch request duration: ~629.5 ms
Matched live prefetch attempts: 114
Prefetch attempts finished before resume: 2 / 114
Late prefetch attempts: 112 / 114
Prefetch slower than no-prefetch: 105 / 114 paired cases
Prefetch faster than no-prefetch: 9 / 114 paired cases
Average paired prefetch gain: -130 ms
```

What we can claim strongly:

```text
There is a real deadline mismatch.
The runtime sees useful agent/tool context.
The normal software/controller/SGLang request path is not predictable enough
for short post-tool resume windows.
This motivates deadline-aware, priority-aware KV movement below the ordinary
best-effort request path.
```

What we should not claim yet:

```text
Hardware will definitely improve performance by a specific percentage.
```

Manager-facing version:

```text
Our live AgentBench/SGLang experiment shows that real agent tool gaps can be
tens of milliseconds, while the software prefetch path takes hundreds of
milliseconds. That means the system often knows what KV may be needed next,
but cannot move or protect it fast enough through today's generic serving path.
This is concrete motivation for hint-aware GPU/runtime support.
```

Main deduction from the latest timeline:

```text
Even when CUDA copy happens before replay,
the software-managed path can still fail because:
- the full hint path is late,
- replay reloads KV again,
- prefetched KV is not protected/reused predictably.

So the need is not just:
  copy memory earlier.

The need is:
  copy the right KV,
  finish the hint path predictably,
  protect the prefetched KV,
  and make reuse visible/enforceable.
```

## Milestones

### Milestone 0: Testbed Scaffold - Completed

Status: completed locally and uploaded to EC2.

What it is:

```text
Create the project folder, scripts, Python package, configs, and EC2 sync workflow.
```

Why we need it:

```text
Before touching SGLang internals, we need a repeatable place to run experiments.
This prevents the project from becoming a set of one-off shell commands.
It also lets us upload the same code to EC2, run it, download artifacts, and debug cleanly.
```

What this proved:

```text
The project has a clean direct-SGLang testbed structure.
The EC2 upload/download/SSH scripts work.
The Python package installs in editable mode.
```

Important events to observe:

```text
setup_ec2 completes without Python/package errors
upload.sh syncs local files to EC2
download.sh pulls artifacts back from EC2
ssh_to_ec2.sh can run remote commands
```

Key files:

```text
scripts/setup_ec2.sh
scripts/run_sglang_server.sh
scripts/run_sglang_hicache_server.sh
scripts/probe_sglang_kv_paths.py
scripts/extract_sglang_kv_targets.py
scripts/run_workload.py
src/agentic_kv/instrumentation.py
```

### Milestone 1: SGLang Internals Map - Completed

Status: completed on EC2.

What it is:

```text
Scan the installed SGLang package and identify the real KV/cache/HiCache/radix/offload functions.
```

Why we need it:

```text
We do not want to guess where KV movement happens.
To instrument SGLang directly, we first need to know which classes allocate KV, cache prefixes, evict entries, and move pages between GPU and host memory.
This milestone gives us a map before we edit or wrap anything.
```

What we answered:

```text
Where are KV blocks allocated?
Where are prefix/radix cache entries stored?
Where does eviction happen?
Is HiCache or CPU offload available in this install?
Which functions look useful for direct KV instrumentation?
```

Commands used:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/probe_sglang_kv_paths.py --out artifacts/sglang_probe.json
python scripts/extract_sglang_kv_targets.py \
  --out-json artifacts/sglang_kv_targets.json \
  --out-md artifacts/sglang_kv_targets.md
```

Result:

```text
SGLang package scanned successfully.
Matched files: 633
Total line hits: 19707
Artifacts written under artifacts/.
```

Important SGLang targets found:

```text
HiCacheController.load()
HiCacheController.write()
HiCacheController.evict_device()
HiCacheController.evict_host()
HiRadixCache
RadixCache
KVCache
MHATokenToKVPool
MHATokenToKVPoolHost
scheduler._prefetch_kvcache()
```

Important events to observe:

```text
probe_sglang_kv_paths.py writes artifacts/sglang_probe.json
extract_sglang_kv_targets.py writes artifacts/sglang_kv_targets.json
extract_sglang_kv_targets.py writes artifacts/sglang_kv_targets.md
HiCacheController.* functions are found
HiRadixCache / RadixCache functions are found
```

### Milestone 2: Real SGLang + HiCache Smoke Test - Completed

Status: completed on EC2.

What it is:

```text
Run a real SGLang server on the EC2 GPU with hierarchical KV cache enabled, then send a real chat request.
```

Why we need it:

```text
The proposal is only convincing if the testbed uses a real model, real GPU memory, and SGLang's real HiCache path.
This milestone proves the environment works before we start measuring KV movement or hint policies.
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/smoke_hicache_request.sh Qwen/Qwen2.5-1.5B-Instruct
```

The script starts SGLang with HiCache if it is not already running, waits for `/model_info`, sends one real chat request, prints the JSON response, and stops the server it started.

To send multiple requests through the same traced server:

```bash
REQUEST_COUNT=3 bash scripts/smoke_hicache_request.sh Qwen/Qwen2.5-1.5B-Instruct
```

What we proved:

```text
SGLang runs on the EC2 GPU.
Qwen/Qwen2.5-1.5B-Instruct loads successfully.
Hierarchical KV cache can be enabled.
SGLang allocates real device KV cache.
SGLang allocates real host HiCache memory.
A real OpenAI-compatible chat request completes.
```

Successful smoke result:

```text
model: Qwen/Qwen2.5-1.5B-Instruct
response: OK
hierarchical cache: enabled
hicache_size: 14 GB
attention backend: triton
prefill attention backend: triton
decode attention backend: triton
```

Important events to observe:

```text
SGLang server reaches /model_info
KV Cache is allocated
host HiCache memory is allocated
chat request returns HTTP 200
model response contains expected text, for example "OK"
```

Important fixes made during this milestone:

```text
Use Python 3.11 instead of Python 3.9.
Install Python development headers.
Install minimal CUDA 12.8 JIT dependencies.
Auto-export CUDA_HOME when nvcc is available.
Disable CUDA graph and overlap scheduling for smoke tests.
Force Triton attention for prefill and decode.
Lower HiCache host pool from 16 GB to 14 GB on g5.2xlarge.
Prune old Docker state to free disk space.
```

### Milestone 3: Log Real KV Movement - Completed

Status: completed on EC2 for the initial HiCache write path.

What it is:

```text
Instrument SGLang's real HiCache path and log KV movement events.
```

Why we need it:

```text
The hardware proposal depends on knowing when KV is written, loaded, evicted, or reused.
Before we compare prefetch policies, we need evidence that our testbed can observe real SGLang KV/cache behavior.
This turns SGLang from a black box into something we can measure.
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/smoke_hicache_request.sh Qwen/Qwen2.5-1.5B-Instruct
```

For a slightly richer trace:

```bash
REQUEST_COUNT=3 bash scripts/smoke_hicache_request.sh Qwen/Qwen2.5-1.5B-Instruct
```

Trace output:

```text
artifacts/kv_movement_trace.jsonl
```

Summarize an existing trace:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/summarize_kv_trace.py --trace artifacts/kv_movement_trace.jsonl
```

Events we want to capture:

```text
request/session observed
KV allocated
KV written from GPU HBM to host memory
KV loaded from host memory back to GPU HBM
KV evicted from GPU
KV evicted from host
tokens/pages affected
time spent in each movement
```

Initial instrumentation points:

```text
HiCacheController.load()
HiCacheController.write()
HiCacheController.evict_device()
HiCacheController.evict_host()
HiCacheController.prefetch()
HiRadixCache.match_prefix()
HiRadixCache.cache_finished_req()
HiRadixCache.cache_unfinished_req()
HiRadixCache.evict()
HiRadixCache.ready_to_load_host_cache()
```

Important events to observe:

```text
trace.install.start / trace.install.end
hiradix.match_prefix.start / hiradix.match_prefix.end
hiradix.cache_unfinished_req.start / hiradix.cache_unfinished_req.end
hiradix.cache_finished_req.start / hiradix.cache_finished_req.end
hiradix.ready_to_load_host_cache.start / hiradix.ready_to_load_host_cache.end
hicache.write.start / hicache.write.end
```

Result from the first traced EC2 run:

```text
REQUEST_COUNT=3
Total trace events: 66
HiCache write calls: 4
HiRadixCache match_prefix calls: 8
HiRadixCache ready_to_load_host_cache calls: 4
HiRadixCache cache_unfinished_req calls: 4
HiRadixCache cache_finished_req calls: 4
```

Important interpretation:

```text
We have proven that our testbed can observe real SGLang HiCache activity.
The tiny smoke workload triggered GPU-to-host HiCache writes.
It did not yet trigger host-to-GPU loads or evictions because there was little memory pressure.
The next step is to create a pressure/resume workload that forces load and eviction events.
```

### Milestone 4: Force Load/Evict And Add Direct Hint Hooks - Completed

Status: completed on EC2 for the pressure/load/evict path and initial agent hint timeline.

What it is:

```text
Create a pressure/resume workload that triggers host-to-GPU loads and evictions, then connect agent/session hints to SGLang KV movement decisions.
```

Why we need it:

```text
The current smoke test proves we can observe HiCache writes, but it is too small to force host-to-GPU reloads or eviction pressure.
Agentic KV prefetch matters most when a session's KV has moved away from fast GPU memory and must be brought back before the agent resumes.
This milestone creates that realistic failure mode, then adds hints such as session_id, priority, deadline, and reuse confidence.
```

Run the pressure trace:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/run_milestone4_pressure_trace.sh Qwen/Qwen2.5-1.5B-Instruct
```

What this script does:

```text
Starts SGLang with HiCache enabled.
Shrinks the device KV pool with --max-total-tokens.
Warms a few target agent sessions.
Sends many unique filler sessions to create KV pressure.
Resumes the target sessions after the pressure phase.
Writes a KV movement trace and pressure workload metrics.
```

Useful knobs:

```bash
MAX_TOTAL_TOKENS=8192
TARGET_SESSIONS=2
FILLER_SESSIONS=18
PROMPT_TOKENS=1024
PRESSURE_CONCURRENCY=1
```

Trace output:

```text
artifacts/milestone4_kv_movement_trace.jsonl
artifacts/results/milestone4_pressure_resume_metrics.jsonl
```

Important events to observe:

```text
agent.session_warm
agent.hint_submitted
agent.pressure_start
agent.resume_start
agent.request.start / agent.request.end
hicache.write.start / hicache.write.end
hicache.load.start / hicache.load.end
hicache.evict_device.start / hicache.evict_device.end
hiradix.evict.start / hiradix.evict.end
```

Example hint:

```text
Agent 42 entered tool_wait.
Expected return: 500 ms.
Priority: high.
Reuse confidence: high.
```

Target behavior:

```text
Mark this session as likely to resume.
Prefer loading its KV before the resume request arrives.
Avoid evicting it immediately after loading.
Record whether the hint helped.
```

Success criteria:

```text
The workload produces real load and/or eviction trace events.
The trace can associate those events with a request/session or prefix-cache path.
The hint layer can mark a session as likely to resume soon.
```

Result from the first pressure EC2 run:

```text
MAX_TOTAL_TOKENS=8192
TARGET_SESSIONS=2
FILLER_SESSIONS=18
PROMPT_TOKENS=1024
Total trace events: 503
HiCache write calls: 46
HiCache load calls: 4
HiCache device eviction calls: 37
HiRadixCache eviction calls: 17
Agent request events: 44
Agent hint_submitted events: 2
Agent resume_start events: 2
```

Important interpretation:

```text
We have now forced the harder case that Milestone 3 did not trigger.
SGLang wrote KV to host HiCache, evicted device-side KV, and later loaded KV from host-side HiCache.
The same trace also contains agent-level hint events, so we can align "Agent 42 is likely to resume" with real SGLang KV movement.
This is still not a performance comparison. It is proof that the pressure/resume scenario and observability hooks work.
```

### Milestone 5: Compare Three Modes - Completed

Status: completed on EC2 for the first small three-mode comparison.

What it is:

```text
Run the same agentic workload under three modes.
```

Why we need it:

```text
A single optimized run does not prove the idea.
We need a fair comparison against no prefetch and generic software behavior.
This is where we show whether hint-aware KV movement helps beyond what SGLang already does by default.
```

Modes:

```text
Mode 1: no prefetch
Mode 2: generic software prefetch
Mode 3: hint-aware direct KV prefetch/protection
```

Mode setup:

| Mode | Extra action | When it happens | Simple meaning |
| --- | --- | --- | --- |
| `no_prefetch` | No extra target warming request | Never | Let SGLang handle KV only when the real resume request arrives. |
| `generic_prefetch` | Send extra target warming requests | Early, before pressure | Warm target KV/prefix early, but it may be evicted again before resume. |
| `hint_aware` | Send extra high-priority target warming requests | Late, after pressure and close to resume | Warm target KV/prefix closer to when the agent actually needs it. |

Mode timelines:

```text
Mode 1: no_prefetch
0 ms: target_0 and target_1 run
500 ms: target agents wait on tool
600 ms: filler requests create KV pressure
900 ms: target agents resume
900+ ms: measure resume TTFT
```

```text
Mode 2: generic_prefetch
0 ms: target_0 and target_1 run
500 ms: target agents wait on tool
520 ms: generic prefetch warms target KV/prefix early
600 ms: filler requests create KV pressure
900 ms: target agents resume
900+ ms: measure resume TTFT
```

```text
Mode 3: hint_aware
0 ms: target_0 and target_1 run
500 ms: target agents wait on tool
501 ms: hint says target agents are high priority and likely to resume soon
600 ms: filler requests create KV pressure
850 ms: hint-aware prefetch warms target KV/prefix close to resume
900 ms: target agents resume
900+ ms: measure resume TTFT
```

Important clarification:

```text
Milestone 5 does not yet call an internal SGLang API like "load exact KV block IDs into GPU memory."
It emulates prefetch by sending extra real SGLang requests that touch the target agent prefix.
This tests the policy timing question: is warming close to resume better than warming early or doing nothing?
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/run_milestone5_compare_modes.sh Qwen/Qwen2.5-1.5B-Instruct
```

Default pressure settings:

```text
MAX_TOTAL_TOKENS=4096
TARGET_SESSIONS=2
FILLER_SESSIONS=36
PROMPT_TOKENS=1536
PRESSURE_CONCURRENCY=1
```

Why this is harsher than the first run:

```text
MAX_TOTAL_TOKENS is smaller, so the GPU-side KV pool is tighter.
FILLER_SESSIONS is larger, so more unrelated sessions compete for KV space.
PROMPT_TOKENS is larger, so each filler request consumes more KV.
Together, this makes it more likely that target_0 and target_1 lose GPU-resident KV before resume.
```

If resume TTFT is still too low, make the run even harsher:

```bash
MAX_TOTAL_TOKENS=3072 \
FILLER_SESSIONS=48 \
PROMPT_TOKENS=2048 \
bash scripts/run_milestone5_compare_modes.sh Qwen/Qwen2.5-1.5B-Instruct
```

If SGLang runs out of memory or becomes unstable, back off:

```bash
MAX_TOTAL_TOKENS=6144 \
FILLER_SESSIONS=24 \
PROMPT_TOKENS=1024 \
bash scripts/run_milestone5_compare_modes.sh Qwen/Qwen2.5-1.5B-Instruct
```

What this script does:

```text
Runs the same pressure/resume workload three times.
Starts a fresh SGLang server for each mode.
Uses the same constrained device KV pool for each mode.
Writes one trace and one metrics file per mode.
Prints a final comparison table.
```

TTFT values reported:

```text
warm_ttft = TTFT for the first time target_0 and target_1 are sent.
resume_ttft = TTFT when target_0 and target_1 resume after tool wait and KV pressure.
```

Why both matter:

```text
warm_ttft shows the first-touch cost for the target agent sessions.
resume_ttft shows the cost after the agent pauses, other requests create KV pressure, and the target agents come back.
The main prefetch question is whether resume_ttft improves without making the rest of the run worse.
```

Important note:

```text
This is the first policy comparison harness.
The generic and hint-aware prefetch actions are still software-level emulations.
They use real SGLang requests to make the target prefix/KV path hot at different times.
Milestone 5 tells us whether the policy timing is promising before deeper direct control.
```

Output files:

```text
artifacts/results/milestone5/no_prefetch_trace.jsonl
artifacts/results/milestone5/generic_prefetch_trace.jsonl
artifacts/results/milestone5/hint_aware_trace.jsonl
artifacts/results/milestone5/no_prefetch_metrics.jsonl
artifacts/results/milestone5/generic_prefetch_metrics.jsonl
artifacts/results/milestone5/hint_aware_metrics.jsonl
artifacts/results/milestone5/summary.json
```

Main question:

```text
Does hint-aware KV movement reduce post-tool resume latency compared with no prefetch and generic prefetch?
```

Success criteria:

```text
Same model, same trace, same EC2 machine, same request mix.
Mode 3 improves tool-return-to-first-token latency or tail latency.
Mode 3 avoids obvious regressions such as too much wasted prefetch or decode slowdown.
```

Important events to observe:

```text
mode=no_prefetch run starts and completes
mode=generic_prefetch run starts and completes
mode=hint_aware run starts and completes
agent.hint_submitted appears in hint-aware mode
prefetch_attempted is recorded
prefetch_success or prefetch_miss is recorded
warm TTFT is recorded for each first target request
resume TTFT is recorded for each target resume
hicache.load / hicache.write / hicache.evict_device counts are summarized per mode
```

Result from the harsher comparison run:

```text
mode             warm_count  avg_warm_TTFT_ms  p95_warm_TTFT_ms  resume_count  avg_resume_TTFT_ms  p95_resume_TTFT_ms  hicache_load  hicache_evict_device
no_prefetch      2           250.727           373.147           2             79.499              80.405              4             87
generic_prefetch 2           255.053           381.734           2             80.661              82.055              4             89
hint_aware       2           252.274           373.880           2             72.189              81.950              6             92
```

Important interpretation:

```text
The comparison harness works.
All three modes ran on the same model, same EC2 instance, same constrained KV pool, and same pressure/resume workload.
The harsher run creates more KV pressure than the first run.
No-prefetch resume TTFT increased from about 48 ms to about 79 ms.
Generic pre-pressure prefetch did not help because later filler requests can evict or disturb the warmed target KV/prefix.
Hint-aware near-resume prefetch lowered average resume TTFT from 79.499 ms to 72.189 ms in this small run.
This is a promising signal, not yet a final conclusion, because there were only two target resume requests.
The next step is to run larger repetitions and package the results for the manager demo.
```

### Milestone 6: Design-Space Sweep

Status: implemented and smoke-tested on EC2.

What it is:

```text
Run a grid of experiments that varies:
1. when hint-aware prefetch happens
2. how much cache pressure exists before tool return
3. how large each request is
```

Why we need it:

```text
One comparison run is only one point.
The design-space sweep tells us when hint-aware prefetch helps, when it does not help, and when software emulation is not enough.
This helps estimate the minimum and maximum benefit we might get from real hardware/runtime support.
```

Three planes:

| Plane | Knob | Default Values | Simple Meaning |
| --- | --- | --- | --- |
| Prefetch timing | `TIMINGS` | `pre_pressure near_resume` | When target KV/prefix warming happens during the tool gap. |
| Cache pressure | `FILLER_LIST` | `12 24 96 192` | How many unrelated sessions compete for KV space. |
| Request size | `PROMPT_TOKEN_LIST` | `1024 1536` | How large each target/filler prompt is. |

Prefetch timing values:

```text
very_early_before_pressure = prefetch immediately after tool wait starts
pre_pressure = prefetch after the tool-wait delay, but before filler pressure
middle_during_pressure = prefetch halfway through filler pressure
near_resume = prefetch after filler pressure, right before resume
```

The default sweep uses only pre_pressure and near_resume timing.
The very-early and middle timing values remain available for optional follow-up sweeps.
The old names early_before_pressure and late_after_pressure are still accepted as aliases.

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/run_milestone6_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

Default sweep command:

```bash
RESULT_ROOT=artifacts/results/milestone6_design_space \
FILLER_LIST="12 24 96 192" \
PROMPT_TOKEN_LIST="1024 1536" \
TIMINGS="pre_pressure near_resume" \
bash scripts/run_milestone6_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

Default sweep result:

```text
The sweep runner writes metrics and traces for all 24 clean SGLang runs.
The summarizer writes summary.json and summary.csv.
The chart generator writes SVG charts and one combined HTML dashboard under charts/.
```

What the default sweep runs:

```text
2 request sizes x 4 pressure levels x 3 cases

For each request size and pressure level:
1 no_prefetch baseline
2 hint_aware timing choices
```

That is 24 total SGLang server runs by default.
Each design point starts a fresh SGLang server, runs one case, writes metrics/traces, then stops the server.
This avoids cache state leaking from one design point into the next.

Progress shown in the terminal:

```text
Total cases: 24
==== Milestone 6 case [1/24]: no_prefetch_near_resume_f12_p1024 ====
...
==== Completed Milestone 6 case [1/24]: no_prefetch_near_resume_f12_p1024 ====
```

Output files:

```text
artifacts/results/milestone6_design_space/*_metrics.jsonl
artifacts/results/milestone6_design_space/*_trace.jsonl
artifacts/results/milestone6_design_space/*_server.log
artifacts/results/milestone6_design_space/summary.json
artifacts/results/milestone6_design_space/summary.csv
artifacts/results/milestone6_design_space/charts/*.svg
artifacts/results/milestone6_design_space/charts/all_charts.html
```

Charts produced:

```text
benefit_vs_pressure_p1024.svg
benefit_vs_pressure_p1536.svg
resume_ttft_vs_pressure_p1024.svg
resume_ttft_vs_pressure_p1536.svg
prefetch_cost_vs_pressure_p1024.svg
prefetch_cost_vs_pressure_p1536.svg
all_charts.html
```

Chart meaning:

```text
x-axis = cache pressure, measured by filler sessions
y-axis = latency metric
lines = prefetch timing choices
separate charts = prompt size
all_charts.html shows each chart beside an exact-number table
dashboard tables include first-prompt TTFT and resume/prefetch/benefit values
```

Summary columns:

```text
warm_ttft_avg_ms = first target request TTFT
prefetch_ttft_avg_ms = cost of the hint-aware warm request
resume_ttft_avg_ms = target resume TTFT after tool wait and pressure
benefit_vs_no_prefetch_ms = no_prefetch resume TTFT - hint_aware resume TTFT
benefit_vs_no_prefetch_pct = percentage improvement over no_prefetch
hicache_load = host-to-GPU KV load events observed
hicache_evict_device = GPU-side KV eviction events observed
```

How to run a smaller sweep:

```bash
FILLER_LIST="12 24" \
PROMPT_TOKEN_LIST="1024" \
TIMINGS="pre_pressure near_resume" \
bash scripts/run_milestone6_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

How to run the wider timing sweep:

```bash
TIMINGS="very_early_before_pressure pre_pressure middle_during_pressure near_resume" \
bash scripts/run_milestone6_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

How to run a harsher sweep:

```bash
MAX_TOTAL_TOKENS=3072 \
FILLER_LIST="96 192 250" \
PROMPT_TOKEN_LIST="2048" \
bash scripts/run_milestone6_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

Important events to observe:

```text
agent.hint_submitted
agent.hint_prefetch_start
agent.hint_prefetch_end
agent.pressure_start
agent.resume_start
hicache.load
hicache.write
hicache.evict_device
hiradix.evict
```

Expected story:

```text
Low pressure: hint-aware prefetch may not help much because SGLang already keeps enough KV hot.
Medium/high pressure: timing starts to matter.
Early prefetch can be wasted because filler requests arrive after it.
Late prefetch should usually be stronger because it happens closer to target resume.
Very high pressure may expose the limit of software emulation because we still cannot protect exact KV blocks in hardware.
```

Success criteria:

```text
A table showing benefit across timing, pressure, and request-size settings.
At least one region where hint-aware timing clearly improves resume TTFT.
At least one region where benefit shrinks, showing the limit of software-only emulation.
Enough evidence to pick the best next hardware feature to emulate more directly.
```

### Milestone 7: Direct KV Movement Hooks

Status: implemented and smoke-tested on EC2 as a safe direct-hook probe.

What it is:

```text
Move beyond request-level warming and start wiring toward direct SGLang KV movement control.
This milestone does not yet force a real host-to-GPU KV load.
It creates the safe instrumentation and session mapping needed before calling internal SGLang movement functions.
```

Why we need it:

```text
Milestone 6 showed that near-resume policy helps.
But it still used request-level warming.
The hardware proposal is really about direct movement: "load this session's KV before resume."
Before we can call SGLang internals safely, we need to map agent sessions to prefix/cache evidence and see the relevant cache objects.
```

What changed:

```text
The SGLang trace hook now records compact object metadata for HiCache/Radix calls.
The workload now emits session_id, prompt_hash, request_role, and prompt_tokens metadata.
The workload supports --prefetch-action direct_probe.
direct_probe records where a direct host-to-GPU KV load should happen, but does not call the unsafe internal load yet.
The workload also supports --prefetch-action direct_load.
direct_load sends a marked trigger request during the tool gap and lets SGLang's own init_load_back/load_back path reload the full host-backed KV chain.
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/run_milestone7_direct_hooks.sh Qwen/Qwen2.5-1.5B-Instruct
```

Small smoke version:

```bash
RESULT_ROOT=artifacts/results/milestone7_smoke \
FILLER_SESSIONS=2 \
PROMPT_TOKENS=256 \
bash scripts/run_milestone7_direct_hooks.sh Qwen/Qwen2.5-1.5B-Instruct
```

Smoke result:

```text
Trace events: 142
agent.session_prefix_map events: 2
agent.direct_kv_prefetch_probe events: 2
cache event types observed: 10
session_cache_map.json and session_cache_map.md were generated
```

Pressure version for call-shape discovery:

```bash
RESULT_ROOT=artifacts/results/milestone7_pressure \
FILLER_SESSIONS=96 \
PROMPT_TOKENS=1024 \
bash scripts/run_milestone7_direct_hooks.sh Qwen/Qwen2.5-1.5B-Instruct
```

Pressure result:

```text
Trace events: 2266
agent.direct_kv_prefetch_probe events: 2
hicache.load.start / hicache.load.end: 4 / 4
hicache.write.start / hicache.write.end: 202 / 202
hicache.evict_device.start / hicache.evict_device.end: 207 / 207
hiradix.evict.start / hiradix.evict.end: 98 / 98
hicache_call_report.md was generated
```

Observed natural HiCache load shape:

```text
HiCacheController.load(host_indices=<cpu int64 tensor>, node_id=<int>)
Example host_indices sizes observed: 1335 and 1332 entries
Successful calls returned cuda int64 device_indices tensors with the same length
Some calls returned None when the device allocation was not available
```

Direct load-back trigger:

```bash
RESULT_ROOT=artifacts/results/milestone7_direct_load_probe \
PREFETCH_ACTION=direct_load \
FILLER_SESSIONS=24 \
PROMPT_TOKENS=1024 \
bash scripts/run_milestone7_direct_hooks.sh Qwen/Qwen2.5-1.5B-Instruct
```

This records `agent.direct_kv_load_attempt` and sends a marked trigger request.
If the target KV has been evicted to host memory, SGLang should naturally call:

```text
HiRadixCache.init_load_back(...)
HiRadixCache.load_back(...)
HiCacheController.load(...)
```

Output files:

```text
artifacts/results/milestone7/direct_hooks_trace.jsonl
artifacts/results/milestone7/direct_hooks_metrics.jsonl
artifacts/results/milestone7/direct_hooks_server.log
artifacts/results/milestone7/session_cache_map.json
artifacts/results/milestone7/session_cache_map.md
artifacts/results/milestone7/hicache_call_report.json
artifacts/results/milestone7/hicache_call_report.md
artifacts/results/milestone7/session_host_indices_map.json
artifacts/results/milestone7/session_host_indices_map.md
```

Smoke output files use:

```text
artifacts/results/milestone7_smoke/
```

Important events to observe:

```text
agent.session_prefix_map
agent.hint_submitted
agent.direct_kv_prefetch_probe
agent.direct_kv_load_attempt
agent.direct_kv_load_miss
agent.resume_start
hicache.load.start / hicache.load.end
hicache.write.start / hicache.write.end
hicache.evict_device.start / hicache.evict_device.end
hiradix.match_prefix.start / hiradix.match_prefix.end
```

Success criteria for this milestone:

```text
Trace includes target session_id and prompt_hash.
Trace includes direct KV prefetch probe events.
Trace includes richer SGLang cache object metadata.
session_cache_map.md summarizes target sessions, prompt hashes, direct probe points, cache event counts, and cache objects.
hicache_call_report.md summarizes natural HiCache load/prefetch/match argument and result shapes.
session_host_indices_map.md maps request windows to match_prefix and hicache.load evidence.
```

What this does not prove yet:

```text
direct_probe does not directly move exact KV blocks.
direct_load now exercises SGLang's real load_back path, but it is still triggered through a lightweight request.
It is not yet a clean out-of-band admin command like PREFETCH_KV(session=42).
```

Next substep:

```text
Compare direct_load against no_prefetch under the same pressure settings.
Then turn the trigger request into a cleaner in-server control hook.
```

Known direct-load function shape from the installed SGLang package:

```text
HiCacheController.load(host_indices, priority=None, node_id=-1) -> device_indices or None
```

Milestone 7B blocker:

```text
We still need to map target session/prefix -> host_indices.
Milestone 7C addresses this with a trace-based mapping report.
```

### Milestone 7C: Session To Host Indices Mapping

Status: implemented as a trace-based mapper.

What it is:

```text
Connect an agent request window to the SGLang host KV indices that are loaded during that request.
```

Why we need it:

```text
To prefetch Agent 42 directly, we need Agent 42's host_indices.
SGLang stores those host indices on radix TreeNode.host_value.
The mapper shows which target resume request caused which match_prefix result and which HiCache load.
```

Run it:

```bash
RESULT_ROOT=artifacts/results/milestone7_mapping \
FILLER_SESSIONS=96 \
PROMPT_TOKENS=1024 \
bash scripts/run_milestone7_direct_hooks.sh Qwen/Qwen2.5-1.5B-Instruct
```

Important events to observe:

```text
agent.request.start / agent.request.end
hiradix.match_prefix.end
hicache.load.start / hicache.load.end
```

Important output:

```text
artifacts/results/milestone7_mapping/session_host_indices_map.md
```

What this should show:

```text
target_0_resume -> match_prefix host_hit_length -> hicache.load host_indices length
target_1_resume -> match_prefix host_hit_length -> hicache.load host_indices length
```

Observed mapping result:

```text
request windows: 100
windows with HiCache load: 2
mapping evidence links: 4

target_0_resume:
  host_node_id: 212
  match_prefix host_hit_length: 1335
  hicache.load node_id: 212
  hicache.load host_indices length: 1335
  host_indices sample: head [43..50], tail [1370..1377]

target_1_resume:
  host_node_id: 215
  match_prefix host_hit_length: 1332
  hicache.load node_id: 215
  hicache.load host_indices length: 1332
  host_indices sample: head [1391..1398], tail [2715..2722]
```

Important nuance:

```text
The final host node's host_value may be only part of the full load.
For example, target_0 host node 212 had host_value length 1332,
but the actual hicache.load call used 1335 host indices.
That means SGLang may concatenate ancestor evicted nodes before calling load().
So the direct prefetch path should reproduce SGLang's load_back path, not blindly load only last_host_node.host_value.
```

In simple words:

```text
This proves which host KV pages SGLang naturally loads when a target agent resumes.
Once this mapping is stable, the next step is to issue that load before the resume request.
```

Follow-up after Milestone 7D:

```text
The current direct_load path uses a lightweight trigger request.
To make this closer to future hardware/runtime support, we should eventually replace the trigger request with an in-server control command that can:
1. match the target prefix,
2. identify evicted host-backed nodes,
3. call the same init_load_back / load_back path before the real request arrives.
```

### Milestone 7D: Direct Load-Back Trigger

Status: implemented and pressure-tested on EC2.

What it is:

```text
Replace the old placeholder direct_load path with a real SGLang load-back trigger.
During the tool gap, the workload sends a marked trigger request for the target session.
SGLang matches the target prefix and naturally calls init_load_back/load_back for evicted host-backed KV.
```

Why we need it:

```text
Milestone 7C showed that a full load can include ancestor KV pages.
So we should not manually load only last_host_node.host_value.
Milestone 7D uses SGLang's own load_back path, which collects the full missing KV chain and calls HiCacheController.load with the right host_indices.
```

Run it:

```bash
RESULT_ROOT=artifacts/results/milestone7d_direct_load_pressure \
PREFETCH_ACTION=direct_load \
FILLER_SESSIONS=96 \
PROMPT_TOKENS=1024 \
bash scripts/run_milestone7_direct_hooks.sh Qwen/Qwen2.5-1.5B-Instruct
```

Observed result:

```text
direct_load attempts: 2
direct_load misses: 0
init_load_back events: 11
hicache.load.start / hicache.load.end: 4 / 4

target_0_direct_load_back:
  phase: hint_prefetch
  TTFT: 74.774 ms
  host_node_id: 212
  hicache.load host_indices length: 1335

target_1_direct_load_back:
  phase: hint_prefetch
  TTFT: 62.191 ms
  host_node_id: 214
  hicache.load host_indices length: 1332

target_0_resume:
  TTFT: 49.196 ms
  hicache.load calls during resume: 0

target_1_resume:
  TTFT: 49.081 ms
  hicache.load calls during resume: 0
```

What this means:

```text
In the previous direct_probe pressure run, target resume TTFT was about 81 ms.
With direct_load, target resume TTFT was about 49 ms.
The expensive host-to-GPU KV reload moved from resume time into the tool-gap trigger request.
```

### Milestone 8: Direct Load Design-Space Sweep

Status: implemented as the next sweep.

What it is:

```text
Repeat the Milestone 6 design-space sweep, but use the direct SGLang load-back path.
This compares two active cases:
1. no_prefetch
2. direct_load
```

Why we need it:

```text
Milestone 7D showed that direct_load can move real SGLang load_back work into the tool gap.
Milestone 8 asks the bigger question:
Across cache pressure, prompt size, and timing, how much better is direct_load than no_prefetch?
```

Default design planes:

| Plane | Knob | Default Values | Simple Meaning |
| --- | --- | --- | --- |
| Prefetch timing | `TIMINGS` | `pre_pressure near_resume` | Whether the prefetch happens before pressure or close to resume. |
| Cache pressure | `FILLER_LIST` | `12 24 96 192` | How many unrelated sessions compete for KV space. |
| Request size | `PROMPT_TOKEN_LIST` | `1024 1536` | How large target and filler prompts are. |
| Prefetch action | `PREFETCH_ACTIONS` | `direct_load` | Use only the direct SGLang load-back trigger. |

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone8_direct_load_design_space \
FILLER_LIST="12 24 96 192" \
PROMPT_TOKEN_LIST="1024 1536" \
TIMINGS="pre_pressure near_resume" \
PREFETCH_ACTIONS="direct_load" \
bash scripts/run_milestone8_direct_load_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

What the default sweep runs:

```text
2 request sizes x 4 pressure levels x 3 cases

For each request size and pressure level:
1 no_prefetch baseline
2 direct_load timing choices
```

That is 24 total SGLang server runs by default.
Each design point starts a fresh SGLang server, runs one case, writes metrics/traces, then stops the server.
This avoids cache state leaking from one design point into the next.

Progress shown in the terminal:

```text
Total cases: 24
==== Milestone 8 case [1/24]: no_prefetch_direct_load_near_resume_f12_p1024 ====
...
==== Completed Milestone 8 case [1/24]: no_prefetch_direct_load_near_resume_f12_p1024 ====
```

Output files:

```text
artifacts/results/milestone8_direct_load_design_space/*_metrics.jsonl
artifacts/results/milestone8_direct_load_design_space/*_trace.jsonl
artifacts/results/milestone8_direct_load_design_space/*_server.log
artifacts/results/milestone8_direct_load_design_space/summary.json
artifacts/results/milestone8_direct_load_design_space/summary.csv
artifacts/results/milestone8_direct_load_design_space/charts/*.svg
artifacts/results/milestone8_direct_load_design_space/charts/all_charts.html
```

Charts produced:

```text
benefit_vs_pressure_p1024.svg
benefit_vs_pressure_p1536.svg
resume_ttft_vs_pressure_p1024.svg
resume_ttft_vs_pressure_p1536.svg
prefetch_cost_vs_pressure_p1024.svg
prefetch_cost_vs_pressure_p1536.svg
all_charts.html
```

Chart meaning:

```text
x-axis = cache pressure, measured by filler sessions
y-axis = latency metric
lines = direct_load timing choices
separate charts = prompt size
tables show first TTFT, no_prefetch resume TTFT, direct_load resume TTFT, and benefit values
```

Important events to observe:

```text
agent.direct_kv_load_attempt
agent.direct_kv_load_request.end
hiradix.init_load_back
hiradix.load_back
hicache.load
agent.resume_start
target resume TTFT
```

Expected story:

```text
no_prefetch pays the host-to-GPU KV load cost at resume time.
direct_load should move real SGLang load_back work into the tool gap.
If direct_load works well, resume TTFT should be lower and hicache.load calls should appear during the prefetch phase instead of the resume phase.
```

How to run a small smoke sweep:

```bash
RESULT_ROOT=artifacts/results/milestone8_smoke \
FILLER_LIST="12" \
PROMPT_TOKEN_LIST="1024" \
TIMINGS="near_resume" \
PREFETCH_ACTIONS="direct_load" \
bash scripts/run_milestone8_direct_load_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

Smoke result observed on EC2:

```text
filler_sessions: 12
prompt_tokens: 1024
timing: near_resume

no_prefetch resume TTFT: 52.389 ms
direct_load resume TTFT: 42.790 ms

direct_load benefit: 9.599 ms, 18.32%

direct_load attempts: 2
direct_load misses: 0
hiradix.init_load_back events in direct_load case: 3
hiradix.load_back events in direct_load case: 3
```

Smoke dashboard:

```text
artifacts/results/milestone8_smoke/charts/all_charts.html
```

### Milestone 9: Multi-Session Agent Traffic With Hint Outcome Analysis

Status: implemented as a realistic traffic milestone.

Milestone 9B fix:

```text
Hint execution is decoupled from replay arrival.
The frontend schedules the hint as a background task.
The replay request arrives at its tool-return deadline even if the hint is still running.
This means late_prefetch can now be observed instead of hidden by the workload driver.
```

Milestone 9B validation:

```text
EC2 late-prefetch smoke:
SESSION_COUNT=2
TOOL_WAIT_LIST_MS="250"
HINT_DELAY_MS=500
MODES="direct_load"

Observed outcome:
late_prefetch: 2

This confirms replay can now arrive before the delayed hint finishes.
```

What it is:

```text
Run many overlapping agent sessions instead of one target plus fillers.
Each session has:
1. an initial model request
2. a tool wait
3. an optional frontend hint/prefetch
4. a replay request when the tool returns
```

Why we need it:

```text
The earlier milestones isolated the mechanism.
This milestone creates a more realistic serving scenario:
many agent sessions arrive over time, each has its own replay deadline, and their KV cache states interfere with each other.
This is where too-early, too-late, and unprotected prefetch behavior becomes visible.
```

Mental model:

```text
0 ms:     Agent 000 initial request arrives
120 ms:   Agent 001 initial request arrives
240 ms:   Agent 002 initial request arrives
...

Agent 000 enters tool_wait and gets a prefetch hint.
Agent 001 enters tool_wait and gets a different replay deadline.
Agent 002 has a shorter tool call and may replay before Agent 000.

The frontend can send hints, but the GPU memory movement path still sees generic memory movement.
It does not know which KV belongs to which agent, which replay deadline matters, or which prefetched KV should be protected.
```

Modes:

| Mode | Meaning |
| --- | --- |
| `no_prefetch` | Initial requests and replays only. |
| `direct_load` | Frontend sends the direct SGLang load-back trigger during tool wait. |
| `oracle_direct_load` | Frontend sends direct load close to replay time. This is the timing upper bound. |

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone9_agentic_traffic \
MODES="no_prefetch direct_load oracle_direct_load" \
SESSION_COUNT=12 \
ARRIVAL_GAP_MS=120 \
TOOL_WAIT_LIST_MS="250 500 900 1600" \
PROMPT_TOKEN_LIST="768 1024 1536" \
HINT_DELAY_MS=120 \
ORACLE_LEAD_MS=120 \
bash scripts/run_milestone9_agentic_traffic.sh Qwen/Qwen2.5-1.5B-Instruct
```

What the default run does:

```text
Runs 3 clean SGLang server runs:
1. no_prefetch
2. direct_load
3. oracle_direct_load

Within each run, 12 agent sessions arrive 120 ms apart.
Each session has a tool wait selected from 250, 500, 900, and 1600 ms.
Each session has a prompt size selected from 768, 1024, and 1536 target tokens.
```

Progress shown in the terminal:

```text
Total cases: 4
==== Milestone 9 traffic case [1/4]: no_prefetch ====
...
==== Completed Milestone 9 traffic case [1/4]: no_prefetch ====
```

Output files:

```text
artifacts/results/milestone9_agentic_traffic/no_prefetch_traffic_trace.jsonl
artifacts/results/milestone9_agentic_traffic/no_prefetch_traffic_metrics.jsonl
artifacts/results/milestone9_agentic_traffic/no_prefetch_outcomes/hint_outcomes.csv
artifacts/results/milestone9_agentic_traffic/no_prefetch_outcomes/hint_outcomes.md
artifacts/results/milestone9_agentic_traffic/no_prefetch_outcomes/hint_outcomes.html

The same files are produced for direct_load and oracle_direct_load.

Combined summary:
artifacts/results/milestone9_agentic_traffic/traffic_summary.csv
artifacts/results/milestone9_agentic_traffic/traffic_summary.md
artifacts/results/milestone9_agentic_traffic/traffic_summary.html
```

The combined summary includes:

```text
Mode Summary
Tool Wait Breakdown
Prefetch Time Breakdown
Failure Mode Rows
Per-Session Delta vs Baseline
```

The Prefetch Time Breakdown is especially important.
It separates:

```text
hint_ttft_ms = end-to-end hint request time seen by the frontend
hint_start_to_first_hicache_load_ms = time before SGLang reaches KV load-back
hint_hicache_load_duration_total_ms = measured time spent inside hicache.load
hint_non_load_time_ms = hint time not explained by hicache.load duration
```

Why this matters:

```text
If hicache.load is tiny but hint_ttft_ms is large,
the bottleneck is not just raw KV copy time.
The hint is delayed by serving/runtime scheduling, queueing, cache bookkeeping, and contention with active work.
This supports the case for deadline-aware, priority-aware hardware/runtime prefetch support.
```

Outcome labels:

| Outcome | Meaning |
| --- | --- |
| `useful_prefetch` | Hint completed before replay, and replay did not need KV load-back. |
| `late_prefetch` | Replay started before the hint/prefetch completed. |
| `too_early_no_load_then_evicted` | Hint fired while KV was still resident, did not force useful movement/protection, eviction pressure happened, and replay later loaded KV. |
| `too_early_or_unprotected` | Prefetch loaded KV, eviction pressure happened later, and replay still loaded KV. |
| `resume_still_loaded_kv` | Hint happened, but replay still paid KV load-back. |
| `no_prefetch_needed` | KV was already resident; no load was needed. |
| `no_hint` | Baseline mode with no frontend hint. |

Important events to observe:

```text
agent.session_arrival
agent.tool_wait_start
agent.hint_submitted
agent.hint_task_scheduled
agent.hint_prefetch_start
agent.direct_kv_load_attempt
agent.direct_kv_load_request.end
agent.hint_prefetch_end
agent.replay_due
agent.resume_start
hicache.load
hicache.evict_device
hiradix.init_load_back
hiradix.load_back
hiradix.evict
```

Why this supports the hardware argument:

```text
If direct_load fires too early, KV can be evicted before replay.
If it fires too late, replay still stalls.
If many sessions overlap, one session's prefetch can compete with another session's active decode or useful KV.

Software can issue the hint.
The missing piece is enforcement:
deadline-aware scheduling, priority-aware migration, and temporary residency protection in the memory movement path.
```

How to run a small smoke test:

```bash
RESULT_ROOT=artifacts/results/milestone9_smoke \
MODES="no_prefetch direct_load" \
SESSION_COUNT=4 \
ARRIVAL_GAP_MS=100 \
TOOL_WAIT_LIST_MS="250 600" \
PROMPT_TOKEN_LIST="512 768" \
HINT_DELAY_MS=100 \
ORACLE_LEAD_MS=100 \
bash scripts/run_milestone9_agentic_traffic.sh Qwen/Qwen2.5-1.5B-Instruct
```

Smoke result observed on EC2:

```text
no_prefetch:
  sessions: 4
  outcomes:
    no_hint: 4

direct_load:
  sessions: 4
  direct_load attempts: 4
  outcomes:
    no_prefetch_needed: 2
    too_early_no_load_then_evicted: 2
```

What the direct_load smoke means:

```text
For the short 250 ms tool waits, the hint was enough because replay did not need KV load-back.
For the longer 600 ms tool waits, the hint fired about 500 ms before replay.
At hint time, KV was still resident, so no useful load-back happened.
Then eviction pressure happened before replay.
At replay time, SGLang still had to load KV.

This is the first realistic signal for the hardware argument:
software issued a hint, but without deadline-aware protection/residency enforcement, an early hint did not guarantee useful KV residency at replay time.
```

Smoke reports:

```text
artifacts/results/milestone9_smoke/no_prefetch_outcomes/hint_outcomes.html
artifacts/results/milestone9_smoke/direct_load_outcomes/hint_outcomes.html
```

How to generate a combined report from existing results:

```bash
python scripts/summarize_agentic_traffic_results.py \
  --root artifacts/results/milestone9_agentic_traffic \
  --modes "no_prefetch direct_load oracle_direct_load"
```

The fastest file to inspect is:

```text
artifacts/results/milestone9_agentic_traffic/traffic_summary.html
```

How to run an oracle lead sweep:

```bash
RESULT_ROOT_BASE=artifacts/results/milestone9_oracle_lead_sweep \
ORACLE_LEAD_LIST="500 1000 1500" \
MODES="no_prefetch oracle_direct_load" \
SESSION_COUNT=12 \
ARRIVAL_GAP_MS=120 \
TOOL_WAIT_LIST_MS="250 500 900 1600" \
PROMPT_TOKEN_LIST="768 1024 1536" \
HINT_DELAY_MS=120 \
bash scripts/run_milestone9_oracle_lead_sweep.sh Qwen/Qwen2.5-1.5B-Instruct
```

Expected interpretation:

```text
small oracle lead:
  more late_prefetch

larger oracle lead:
  fewer late_prefetch, but possibly more too_early_or_unprotected

That transition is the core hardware argument:
start too late and replay stalls;
start too early and KV may be evicted without residency protection.
```

Observed oracle lead sweep:

| Oracle Lead | Baseline Avg Replay TTFT | Oracle Avg Replay TTFT | Improvement | Outcomes |
| --- | ---: | ---: | ---: | --- |
| `500 ms` | `1010.465 ms` | `871.450 ms` | `139.016 ms` | `late_prefetch: 12` |
| `1000 ms` | `1013.611 ms` | `807.090 ms` | `206.520 ms` | `late_prefetch: 12` |
| `1500 ms` | `1026.244 ms` | `806.769 ms` | `219.475 ms` | `late_prefetch: 10`, `too_early_or_unprotected: 2` |

Timing breakdown from the sweep:

| Oracle Lead | Avg Hint TTFT | Avg Hint Total Duration | Avg Time To First `hicache.load` | Total Hint `hicache.load` Duration | Avg Non-Load Hint Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `500 ms` | `899.227 ms` | `1344.818 ms` | `711.877 ms` | `12.711 ms` | `1343.759 ms` |
| `1000 ms` | `874.432 ms` | `1284.391 ms` | `652.739 ms` | `10.056 ms` | `1283.553 ms` |
| `1500 ms` | `875.295 ms` | `1309.660 ms` | `759.399 ms` | `10.321 ms` | `1308.800 ms` |

Interpretation:

```text
The actual hicache.load time is tiny compared with the end-to-end hint duration.
The delay is mostly outside the raw KV load call.
That points to scheduling, queueing, runtime work, and contention with active inference.

This is stronger than saying "DMA is slow."
The better claim is:
software can issue hints, but the current runtime/memory path cannot make those hints deadline-aware, priority-aware, or residency-protected.
```

### Milestone 10: DMA Timeline Profiling

Status: completed for the worker-local torch.profiler path. External Nsight remains optional and limited on the current EC2 setup.

What it is:

```text
Run the multi-session agent traffic workload with deeper CUDA profiling.
There are now two profiling paths:
1. worker-local torch.profiler inside SGLang worker processes
2. external Nsight Systems profiling

The worker-local profiler is the recommended first path because external Nsight has not yet exposed SGLang's worker GPU activity on the current EC2 setup.
```

Why we need it:

```text
Milestone 9 showed that hint duration can be around 1300 ms while measured hicache.load time is only around 10-13 ms.
That suggests the long delay is not only the raw KV copy itself.
This milestone looks one layer deeper: when did CUDA copy activity actually happen, and what GPU work was it competing with?
```

Important limitation:

```text
Nsight Systems does not expose the GPU DMA engine's private internal scheduling queue.
When it works, it shows observable CUDA memcpy activity, CUDA kernels, streams, and NVTX ranges.
The worker-local torch.profiler path gives us another way to capture CUDA activity from inside the process that is actually running SGLang.
```

Important profiling detail:

```text
SGLang creates worker child processes.
The profiler runner supports two shapes:
1. monitor: start Nsight first as a short system-wide monitor, then start SGLang.
2. launch: let Nsight launch SGLang directly.

On the current EC2 setup, Nsight works on a simple PyTorch CUDA control test,
but does not yet expose SGLang worker memcpy/kernel tables.
So the current fix is to enable torch.profiler inside the SGLang worker and export Chrome traces from there.
```

Recommended run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone10_dma_timeline \
MODE=oracle_direct_load \
HICACHE_SIZE_GB=8 \
ENABLE_NSYS=0 \
AGENTIC_KV_TORCH_PROFILER_ENABLE=1 \
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=40 \
AGENTIC_KV_TRACE_MAX_EXACT_INDICES=512 \
SESSION_COUNT=12 \
ARRIVAL_GAP_MS=120 \
TOOL_WAIT_LIST_MS="250 500 900 1600" \
PROMPT_TOKEN_LIST="768 1024 1536" \
HINT_DELAY_MS=120 \
ORACLE_LEAD_MS=1500 \
bash scripts/run_milestone10_dma_timeline.sh Qwen/Qwen2.5-1.5B-Instruct
```

Smaller smoke run:

```bash
RESULT_ROOT=artifacts/results/milestone10_dma_timeline_smoke \
MODE=oracle_direct_load \
HICACHE_SIZE_GB=8 \
ENABLE_NSYS=0 \
AGENTIC_KV_TORCH_PROFILER_ENABLE=1 \
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=40 \
AGENTIC_KV_TRACE_MAX_EXACT_INDICES=512 \
SESSION_COUNT=4 \
ARRIVAL_GAP_MS=120 \
TOOL_WAIT_LIST_MS="500 900" \
PROMPT_TOKEN_LIST="768" \
ORACLE_LEAD_MS=1000 \
bash scripts/run_milestone10_dma_timeline.sh Qwen/Qwen2.5-1.5B-Instruct
```

Outputs:

```text
artifacts/results/milestone10_dma_timeline/oracle_direct_load_traffic_trace.jsonl
artifacts/results/milestone10_dma_timeline/oracle_direct_load_traffic_metrics.jsonl
artifacts/results/milestone10_dma_timeline/oracle_direct_load_outcomes/
artifacts/results/milestone10_dma_timeline/oracle_direct_load_torch_cuda_profiles/
artifacts/results/milestone10_dma_timeline/oracle_direct_load_torch_cuda_profile_summary.md
artifacts/results/milestone10_dma_timeline/oracle_direct_load_torch_cuda_profile_summary.json
artifacts/results/milestone10_dma_timeline/oracle_direct_load_torch_cuda_trace_correlation.md
artifacts/results/milestone10_dma_timeline/oracle_direct_load_torch_cuda_trace_correlation.json
artifacts/results/milestone10_dma_timeline/oracle_direct_load_torch_cuda_copy_timeline.csv
```

Optional external Nsight run:

```bash
RESULT_ROOT=artifacts/results/milestone10_nsys_smoke \
MODE=oracle_direct_load \
HICACHE_SIZE_GB=8 \
ENABLE_NSYS=1 \
NSYS_USE_SUDO=1 \
NSYS_PROFILE_SHAPE=launch \
SESSION_COUNT=4 \
ARRIVAL_GAP_MS=120 \
TOOL_WAIT_LIST_MS="500 900" \
PROMPT_TOKEN_LIST="768" \
ORACLE_LEAD_MS=1000 \
bash scripts/run_milestone10_dma_timeline.sh Qwen/Qwen2.5-1.5B-Instruct
```

Important events to observe:

```text
agent.hint_submitted
agent.hint_prefetch_start
agent.request.start phase=hint_prefetch
hicache.load.start
hicache.load.end
agent.replay_due
agent.resume_start
agent.request.start phase=replay
torch CUDA profile exported by a SGLang worker
kernel-like events in torch CUDA profile summary
memcpy-like events in torch CUDA profile summary
per-copy start/end rows in torch CUDA copy timeline CSV
overlap between copies and hicache/agent windows in torch CUDA trace correlation report
KV context fields on trace events:
  kv_context.direction
  kv_context.node_id
  kv_context.host_indices
  kv_context.device_indices
  kv_context.layer_id
optional Nsight NVTX/runtime tables
```

What we want to learn:

```text
hint_issue_to_first_cuda_copy_ms:
  Was the memory movement delayed after the hint?

cuda_copy_duration_ms:
  Was the actual copy small or large?

kernel_activity_during_hint:
  Was the GPU busy with active inference while the hint was waiting?

replay_before_hint_done:
  Did the tool return before the prefetch path completed?
```

Expected interpretation:

```text
If hicache.load is short but the CUDA profile shows GPU work happening elsewhere,
then the problem is likely scheduling/queueing/contention, not simply KV bytes being impossible to move.

That supports the hardware proposal:
software can issue hints,
but the memory movement path needs deadline, priority, and residency semantics to make the hints predictable.
```

Current EC2 validation:

```text
Completed a 4-session smoke run with Nsight Systems installed.
The run produced:
  - SGLang KV trace
  - hint outcome report
  - .nsys-rep report
  - SQLite export
  - DMA timeline summary

The profiler captured NVTX ranges and CUDA runtime initialization calls.
It did not yet expose CUPTI memcpy/kernel tables for the SGLang worker path.

A separate PyTorch CUDA control test on the same machine did expose CUPTI_ACTIVITY_KIND_MEMCPY and CUPTI_ACTIVITY_KIND_KERNEL.
So Nsight works on the machine, but our current SGLang launch/profile path still does not expose the worker GPU activity we want.
```

Current interpretation:

```text
Milestone 10 completed the worker-local profiler path.
We also have the external Nsight harness and can generate profiler artifacts.
But external Nsight did not expose SGLang memcpy/kernel tables on this EC2 setup.

The fix was to use a worker-local torch.profiler hook inside the SGLang worker process.
That showed actual worker CUDA activity from inside the process.
```

Worker-local profiler validation:

```text
Completed a 4-session torch-profiler smoke run.
The worker profiler exported one Chrome trace from the SGLang worker process.

Observed in the exported worker trace:
  kernel-like events: 14329
  memcpy-like events: 5569
  HtoD memcpy-like events: 70
  DtoH memcpy-like events: 2229
  DtoD memcpy-like events: 37
  CUDA-like total time: 756.028 ms
  kernel-like total time: 587.921 ms
  memcpy-like total time: 83.588 ms

This fixes the immediate visibility problem:
we can now see CUDA activity from inside the SGLang worker.
The next fix adds KV/block context:
we can now line up CUDA transfer events with SGLang host/device KV index movement windows.
```

### Milestone 10B: CUDA Transfer To KV Block Attribution - Completed

Status: completed on EC2.

What it is:

```text
The trace hooks record extra KV context on SGLang movement calls:
  - call_id
  - node_id
  - request/session tags passed through SGLang custom_params
  - direction: host_to_device / device_to_host / evict
  - host_indices
  - device_indices
  - layer_id for per-layer host-pool movement

This lets the CUDA copy timeline say:
  this CUDA copy overlapped this SGLang KV movement window,
  and that window was moving these host/device KV indices.
  when SGLang request tags are available, the window can also show agent_000 / hint_prefetch / replay.
```

Why we need it:

```text
Milestone 10 showed CUDA copies and SGLang KV windows, but not a clean link to the exact KV indices.
Milestone 10B connects the profiler-visible copy to the SGLang-level KV movement context.
This makes the evidence much more concrete: a copy can now be tied to host indices, device indices, layer id, and agent hint request.
```

Correlation outputs:

```text
oracle_direct_load_torch_cuda_trace_correlation.md:
  shows which CUDA kernels and actual GPU transfer events overlapped each agent request and each SGLang KV method window.
  "transfer events" means profiler rows such as Memcpy HtoD, Memcpy DtoH, Memcpy DtoD, and Memset.
  "copy-like events" means broader PyTorch copy operations, so they are kept separate.

oracle_direct_load_torch_cuda_copy_timeline.csv:
  one row per observed GPU transfer event.
  Each row includes:
    start_ms_from_trace_start
    end_ms_from_trace_start
    duration_ms
    direction: h2d / d2h / d2d / memset / unknown
    bytes
    overlapping agent or KV window when available
    enclosing agent session and phase when available
    overlapping KV direction, node id, layer id, host indices, and device indices when available
```

Simple example of the new evidence:

```text
hostpool.load_to_device_per_layer.start:
  layer_id = 12
  host_indices = [1040, 1041, 1042, ...]
  device_indices = [88, 89, 90, ...]

CUDA copy timeline row:
  Memcpy HtoD started at 25140.015 ms
  Memcpy HtoD ended at 25140.016 ms
  enclosing_agent_session_id = agent_000
  enclosing_agent_phase = hint_prefetch
  overlap_event = hostpool.load_to_device_per_layer
  overlap_kv_agent_session_id = agent_000
  overlap_kv_agent_phase = hint_prefetch
  overlap_layer_id = 12
  overlap_host_indices = [1040, 1041, 1042, ...]
  overlap_device_indices = [88, 89, 90, ...]

Simple meaning:
  this host-to-device copy happened during Agent 000's hint path,
  while SGLang was loading those KV indices for that layer.
```

Current Milestone 10B validation:

```text
Completed on EC2 with:
  RESULT_ROOT=artifacts/results/milestone10b_custom_tag_block_smoke
  MODE=oracle_direct_load
  SESSION_COUNT=4
  AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=180

The run produced:
  CUDA events: 35787
  reported windows with CUDA activity: 150

Example observed row:
  name = Memcpy HtoD (Pinned -> Device)
  direction = h2d
  bytes = 443392
  overlap_event = hostpool.load_to_device_per_layer
  overlap_kv_agent_session_id = agent_000
  overlap_kv_agent_phase = hint_prefetch
  overlap_kv_agent_label = agent_000_direct_load_hint
  overlap_kv_layer_id = 0
  overlap_host_index_count = 1055
  overlap_device_index_count = 1055

Simple meaning:
  we can now show a profiler-visible host-to-device CUDA copy
  and tie it to the SGLang KV indices being loaded for a specific agent hint request.
```

Important caution:

```text
torch.profiler adds high overhead.
Use it for short diagnostic runs, not final latency numbers.
For final performance numbers, use the normal Milestone 9/Milestone 9B runs without torch.profiler.
For DMA/copy evidence, use Milestone 10 torch-profiler traces.
```

### Milestone 11: Agentic Prefetch Timeline Experiment

Status: completed on EC2 for a 6-session smoke run.

What it is:

```text
Run a realistic multi-agent traffic trace and build a per-session timeline that shows:
1. when each agent starts,
2. when it enters tool wait,
3. when the prefetch hint is submitted,
4. when SGLang actually reaches the KV load path,
5. when profiler-visible host-to-device CUDA copies happen,
6. when the replay request becomes due,
7. whether prefetch finished before replay.
```

Why we need it:

```text
Milestone 10B showed that we can connect CUDA HtoD copies to SGLang KV movement windows.
Milestone 11 turns that into a clearer agentic timeline.

Instead of saying only "prefetch was late",
we can show exactly where it was late:
the hint was submitted,
the request waited in the normal serving path,
SGLang eventually reached the KV load path,
CUDA HtoD copies happened,
and the replay deadline may have arrived before the prefetch completed.
```

Simple mental model:

```text
Agent 000:
0 ms:    first request starts
420 ms:  first request ends and tool_wait starts
540 ms:  frontend submits a prefetch hint
1100 ms: SGLang starts loading host KV back to GPU
1115 ms: CUDA HtoD copies finish
1200 ms: tool returns and replay is due
1205 ms: replay starts

This is useful: prefetch finished before replay.
```

```text
Agent 001:
200 ms:  first request starts
610 ms:  first request ends and tool_wait starts
730 ms:  frontend submits a prefetch hint
1500 ms: replay is due
1570 ms: SGLang finally starts loading host KV back to GPU
1585 ms: CUDA HtoD copies finish
1590 ms: replay starts

This is late: the hint existed, but the movement path did not act soon enough.
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone11_agentic_timeline \
MODE=oracle_direct_load \
SESSION_COUNT=16 \
RANDOMIZE_TRAFFIC=1 \
RANDOM_SEED=7 \
ARRIVAL_GAP_RANGE_MS="60 220" \
TOOL_WAIT_RANGE_MS="250 2200" \
PROMPT_TOKEN_LIST="768 1024 1536" \
HINT_DELAY_MS=120 \
ORACLE_LEAD_MS=1000 \
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=220 \
bash scripts/run_milestone11_agentic_timeline.sh Qwen/Qwen2.5-1.5B-Instruct
```

Small smoke run:

```bash
RESULT_ROOT=artifacts/results/milestone11_timeline_smoke \
MODE=oracle_direct_load \
SESSION_COUNT=6 \
RANDOMIZE_TRAFFIC=1 \
RANDOM_SEED=7 \
ARRIVAL_GAP_RANGE_MS="60 220" \
TOOL_WAIT_RANGE_MS="250 1800" \
PROMPT_TOKEN_LIST="768 1024" \
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=220 \
bash scripts/run_milestone11_agentic_timeline.sh Qwen/Qwen2.5-1.5B-Instruct
```

What the script does:

```text
Step 1/6: start SGLang with HiCache, trace hooks, and worker-local torch.profiler
Step 2/6: run randomized multi-session agent traffic
Step 3/6: summarize the SGLang KV trace
Step 4/6: classify hint outcomes
Step 5/6: summarize and correlate torch CUDA copy activity
Step 6/6: build the per-session agentic prefetch timeline
```

Outputs:

```text
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_traffic_trace.jsonl
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_traffic_metrics.jsonl
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_outcomes/hint_outcomes.html
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_torch_cuda_profile_summary.md
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_torch_cuda_trace_correlation.md
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_torch_cuda_copy_timeline.csv
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_agentic_prefetch_timeline.csv
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_agentic_prefetch_timeline.json
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_agentic_prefetch_timeline.html
```

The fastest file to inspect is:

```text
artifacts/results/milestone11_agentic_timeline/oracle_direct_load_agentic_prefetch_timeline.html
```

Timeline columns:

```text
hint_submitted_ms:
  when the frontend/runtime issued the prefetch hint

sglang_copy_start_ms / sglang_copy_end_ms:
  when SGLang reached the KV load-back path for this hint

torch_copy_start_ms / torch_copy_end_ms:
  when profiler-visible host-to-device CUDA copies happened during that SGLang load window

replay_due_ms:
  when the tool result was ready and replay should start

prefetch_margin_ms:
  replay_due_ms - prefetch_done_ms
  positive means prefetch finished before replay
  negative means prefetch finished after replay was already due

late_prefetch:
  true when prefetch_margin_ms is negative
```

Important events to observe:

```text
traffic.workload_start with sampled random sessions
agent.session_arrival
agent.tool_wait_start
agent.hint_submitted
agent.hint_task_scheduled
agent.hint_prefetch_start
agent.request.start phase=hint_prefetch
agent.direct_kv_load_attempt
hostpool.load_to_device_per_layer.start
hostpool.load_to_device_per_layer.end
CUDA Memcpy HtoD rows overlapping hostpool.load_to_device_per_layer
agent.replay_due
agent.resume_start
agent.request.start phase=replay
agent.request.end phase=replay
```

Smoke result observed on EC2:

```text
RESULT_ROOT=artifacts/results/milestone11_timeline_smoke
MODE=oracle_direct_load
SESSION_COUNT=6
RANDOMIZE_TRAFFIC=1
RANDOM_SEED=7

SGLang trace events: 1050
hostpool.load_to_device_per_layer events: 168 start / 168 end
hicache.load events: 18 start / 18 end
hicache.evict_device events: 40 start / 40 end
agent.hint_submitted events: 6
agent.replay_due events: 6
agent.resume_start events: 6

torch profiles exported: 1
kernel-like events: 22467
memcpy-like events: 20653
CUDA events in correlation report: 53240
copy timeline rows: 9969
HtoD copy rows: 4593

hint_outcomes.html:
  late_prefetch: 4
  too_early_or_unprotected: 2

agentic_prefetch_timeline.html:
  sessions: 6
  copy/KV-level late_prefetch sessions: 1
```

The timeline HTML now contains:

```text
Summary:
  run-level counts, visible HtoD copy sessions, late prefetch sessions, average prefetch margin

Timeline:
  one row per agent session with initial request, tool wait, hint, SGLang KV load, CUDA HtoD copy, replay due, and replay.
  Each row also shows SUCCESS / SGLang OK / LATE status directly beside the agent id.
  Green dashed gaps mean prefetch finished before replay.
  Red dashed gaps mean replay was already due before prefetch finished.

Key Observations Per Session:
  a simple explanation of what happened for each agent and what deduction to make

Session Details:
  the exact timing numbers used to build the chart and observations
```

Smoke per-session timeline summary:

| Session | Tool Wait | SGLang KV Load Window | CUDA HtoD Copy Window | Replay Due | Margin | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `agent_000` | `913 ms` | `31644.234 -> 31767.257 ms` | `31661.501 -> 31766.530 ms` | `32093.616 ms` | `+327.086 ms` | Clean useful prefetch |
| `agent_001` | `1583 ms` | `31813.107 -> 31816.774 ms` | not confidently attributed | `32764.811 ms` | `+948.037 ms` | SGLang-level useful prefetch |
| `agent_002` | `1347 ms` | `31651.793 -> 31656.297 ms` | not confidently attributed | `32528.412 ms` | `+872.115 ms` | SGLang-level useful prefetch |
| `agent_003` | `1443 ms` | `44275.895 -> 44348.003 ms` | not confidently attributed | `33089.015 ms` | `-11258.988 ms` | Late prefetch |
| `agent_004` | `689 ms` | `31805.676 -> 31994.630 ms` | `31832.359 -> 31993.362 ms` | `32334.998 ms` | `+341.636 ms` | Clean useful prefetch |
| `agent_005` | `1138 ms` | `31820.643 -> 31825.232 ms` | not confidently attributed | `32784.271 ms` | `+959.039 ms` | SGLang-level useful prefetch |

Smoke per-session interpretation:

| Session | Key Observation | Deduction |
| --- | --- | --- |
| `agent_000` | SGLang KV load and profiler-visible CUDA HtoD copies both happened during the hint path, finishing about `327 ms` before replay was due. | This is a clean success case: the hint moved KV early enough. |
| `agent_001` | SGLang KV load finished about `948 ms` before replay, but profiler HtoD rows were not confidently attached to this session. | Useful SGLang-level evidence, but weaker CUDA-level evidence. |
| `agent_002` | SGLang KV load finished about `872 ms` before replay, again without confident profiler HtoD attribution. | Another useful SGLang-level success case. |
| `agent_003` | The hint was submitted before replay, but SGLang KV load did not occur until about `11259 ms` after replay was due. | This is the strongest failure case: software knew to prefetch, but the normal path acted too late. |
| `agent_004` | SGLang KV load and profiler-visible CUDA HtoD copies both happened during the hint path, finishing about `342 ms` before replay was due. | This is another clean success case. |
| `agent_005` | SGLang KV load finished about `959 ms` before replay, but profiler HtoD rows were not confidently attached to this session. | Useful SGLang-level success, but not a strong CUDA-attribution row. |

Important interpretation of the two late-prefetch counts:

```text
hint_outcomes late_prefetch:
  the whole hint request completed after replay was already due.

agentic timeline late_prefetch:
  the observed SGLang KV load / CUDA copy path completed after replay was already due.

These are related, but not identical.
The first is frontend-visible request timing.
The second is lower-level movement timing.
Both are useful because they show where the delay is happening.
```

Expected interpretation:

```text
If hint_submitted_ms is early but sglang_copy_start_ms is much later,
the prefetch waited inside the normal SGLang/request scheduling path.

If sglang_copy_start_ms is close to torch_copy_start_ms,
then once SGLang reaches the load path, CUDA movement begins quickly.

If torch_copy_end_ms is after replay_due_ms,
the prefetch was too late for that agent replay.

This supports the hardware/runtime argument:
software can create hints,
but the current memory movement path does not have deadline-aware, priority-aware, agent-aware enforcement.
```

Important caution:

```text
torch.profiler adds overhead.
Use Milestone 11 for timeline evidence and causality, not final latency numbers.
Use Milestone 9 style runs without torch.profiler for cleaner TTFT performance numbers.
```

### Milestone 11B: Improved CUDA Copy Attribution

Status: completed on EC2 for a 6-session attribution smoke run.

What it is:

```text
Improve the trace/correlation path so CUDA HtoD copies can be tied back to agent sessions more reliably.
```

Why we need it:

```text
In the first Milestone 11 smoke, some sessions showed SGLang KV load-back but no CUDA HtoD rows.
That was too easy to misread as "no GPU copy happened."
The better interpretation was:
the copy probably happened, but our profiler/correlator did not confidently attach it to that agent.
```

What changed:

```text
The SGLang trace hook now extracts agent context from match_prefix and init_load_back params.
That context is propagated through nested load_back, hicache.load, and hostpool calls.
Queued HiCache load operations can carry their mapped agent context.
The correlation CSV now has fields for shared/batched agent ownership.
The HTML wording now says "not attributed" instead of "not observed" for missing CUDA rows.
The timeline now records profiler coverage, so a missing green bar can be explained.
```

Run used:

```bash
RESULT_ROOT=artifacts/results/milestone11b_attribution_smoke \
MODE=oracle_direct_load \
SESSION_COUNT=6 \
RANDOMIZE_TRAFFIC=1 \
RANDOM_SEED=7 \
ARRIVAL_GAP_RANGE_MS="60 220" \
TOOL_WAIT_RANGE_MS="250 1800" \
PROMPT_TOKEN_LIST="768 1024" \
HINT_DELAY_MS=120 \
ORACLE_LEAD_MS=1000 \
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=220 \
bash scripts/run_milestone11_agentic_timeline.sh Qwen/Qwen2.5-1.5B-Instruct
```

Attribution smoke result:

| Session | SGLang KV Load Window | CUDA HtoD Copy Window | HtoD Events | Profiler Window | Profiler Status | Missing Reason | Replay Due | Margin | Result |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| `agent_000` | `31710.976 -> 31815.287 ms` | `31711.545 -> 31814.808 ms` | `1624` | `24409.362 -> 44407.848 ms` | `inside_profiler_window` | `none` | `32124.946 ms` | `+310.138 ms` | CUDA-attributed useful prefetch |
| `agent_001` | `31887.405 -> 32045.998 ms` | `31887.975 -> 32045.098 ms` | `2856` | `24409.362 -> 44407.848 ms` | `inside_profiler_window` | `none` | `32794.098 ms` | `+749.000 ms` | CUDA-attributed useful prefetch |
| `agent_002` | `31710.976 -> 31815.287 ms` | `31711.545 -> 31814.808 ms` | `1624` | `24409.362 -> 44407.848 ms` | `inside_profiler_window` | `none` | `32558.256 ms` | `+743.448 ms` | CUDA-attributed useful prefetch |
| `agent_003` | `44435.564 -> 44496.473 ms` | not attributed | `0` | `24409.362 -> 44407.848 ms` | `after_profiler_stopped` | `profiler_stopped_before_kv_load` | `33132.434 ms` | `-11364.039 ms` | Late prefetch |
| `agent_004` | `31887.405 -> 32045.998 ms` | `31887.975 -> 32045.098 ms` | `2856` | `24409.362 -> 44407.848 ms` | `inside_profiler_window` | `none` | `32378.433 ms` | `+333.335 ms` | CUDA-attributed useful prefetch |
| `agent_005` | `31887.405 -> 32045.998 ms` | `31887.975 -> 32045.098 ms` | `2856` | `24409.362 -> 44407.848 ms` | `inside_profiler_window` | `none` | `32827.943 ms` | `+782.845 ms` | CUDA-attributed useful prefetch |

Important interpretation:

```text
agent_002 used to look like:
  SGLang load observed, CUDA HtoD not attributed.

After Milestone 11B, agent_002 now shows:
  SGLang load observed,
  CUDA HtoD copy window observed,
  HtoD copy finished about 743 ms before replay.

This makes the evidence stronger:
the hint path did real host-to-device movement early enough for agent_002.
```

Important nuance:

```text
Some HtoD copy windows are shared/batched across multiple agent sessions.
For example, the same HtoD copy window may include agent_000 and agent_002.
That is not a bug.
It shows SGLang/runtime batching KV movement across sessions.
The hardware argument still applies: the movement path needs agent/session/deadline context even when copies are batched.
```

Agent 003 no-green diagnosis:

```text
In this run, agent_003 did have SGLang host-to-device KV load work:
  44435.564 ms -> 44496.473 ms

But torch.profiler stopped earlier:
  24409.362 ms -> 44407.848 ms

So the missing green bar means:
  the profiler was no longer recording when agent_003's KV load happened.

It does not mean:
  no CUDA host-to-device copy happened.
```

### Milestone 11C: Profiler Coverage Diagnosis

Status: completed on EC2 for a focused 6-session attribution-debug run.

What it is:

```text
Rerun the same 6-session traffic shape with a later torch-profiler event stop.
The goal is to test whether agent_003 gets CUDA HtoD attribution when the profiler stays alive long enough.
```

Why we need it:

```text
Milestone 11B showed agent_003 with SGLang KV movement but no green CUDA-copy bar.
Before claiming "the copy was missing," we need to prove whether the profiler actually covered that time window.
```

Run used:

```bash
RESULT_ROOT=artifacts/results/milestone11c_profiler_coverage_smoke_300 \
MODE=oracle_direct_load \
SESSION_COUNT=6 \
RANDOMIZE_TRAFFIC=1 \
RANDOM_SEED=7 \
ARRIVAL_GAP_RANGE_MS="60 220" \
TOOL_WAIT_RANGE_MS="250 1800" \
PROMPT_TOKEN_LIST="768 1024" \
HINT_DELAY_MS=120 \
ORACLE_LEAD_MS=1000 \
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=300 \
bash scripts/run_milestone11_agentic_timeline.sh Qwen/Qwen2.5-1.5B-Instruct
```

Important warning:

```text
This run is for attribution debugging only.
Do not use its TTFT numbers as performance evidence.
Stopping/exporting torch.profiler during live traffic can add large overhead to later requests.
Use Milestone 9-style runs without torch.profiler for performance numbers.
```

Agent 003 result:

| Session | SGLang KV Load Window | CUDA HtoD Copy Window | HtoD Events | HtoD Bytes | Profiler Window | Profiler Status | Missing Reason | Replay Due | Margin | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| `agent_003` | `33191.650 -> 47824.138 ms` | `33192.241 -> 33236.513 ms` | `528` | `23767040` | `24573.943 -> 47812.734 ms` | `partly_outside_profiler_window` | `none` | `34052.773 ms` | `+816.260 ms` | CUDA-attributed useful prefetch |

Conclusion:

```text
The old agent_003 no-green case was caused by profiler coverage.
When the profiler stayed alive past agent_003's HtoD copy window, agent_003 got green CUDA HtoD attribution.
So the attribution pipeline is now more precise:
  no green + after_profiler_stopped = profiler coverage issue
  no green + inside_profiler_window = real attribution gap to investigate
```

Three-checkpoint interpretation:

```text
The timeline now separates three questions:

1. Did the profiler-visible CUDA HtoD copy finish before replay?
2. Did the full software hint request finish before replay?
3. Did the replay still reload KV?

These are different.
A green bar only answers question 1.
It does not automatically mean the whole prefetch succeeded.
```

11C300 checkpoint result:

| Checkpoint | Count | Meaning |
| --- | ---: | --- |
| CUDA copy ready before replay | `6 / 6` | Every selected session had profiler-visible HtoD copy before replay due. |
| Full hint request done before replay | `5 / 6` | Agent 003 still had the full hint request path complete late. |
| Replay still reloaded KV | `6 / 6` | Every replay still triggered KV load-back work. |
| Clean success | `0 / 6` | No session satisfied all three success conditions at once. |

Simple conclusion:

```text
11C300 proves the profiler can attribute Agent 003 CUDA HtoD copies when coverage is long enough.
It does not prove the prefetch system had no failures.

For Agent 003:
  CUDA copy ready before replay: yes
  Full hint request done before replay: no
  Replay still reloaded KV: yes

So Agent 003 is not a clean success.
It is a useful case showing why raw copy visibility is only one part of the prefetch story.
```

### Milestone 12: Paired Clean + Attribution Evidence

Status: completed on EC2 with both smoke and manager stress presets.

What it is:

```text
Run two matching experiments with the same traffic shape.

Run A: clean performance run
  torch profiler off
  use this for TTFT and latency claims

Run B: profiled attribution run
  torch profiler on
  use this for CUDA HtoD/KV mechanism evidence
```

Why we need it:

```text
torch.profiler can distort TTFT heavily.
So we should not use profiled runs for performance claims.

Instead:
  clean run answers "how fast was it?"
  profiled run answers "what happened internally?"
```

What it produces:

```text
artifacts/results/milestone12_paired_evidence/clean_performance/
artifacts/results/milestone12_paired_evidence/profiled_attribution/
artifacts/results/milestone12_paired_evidence/paired_report/paired_report.html
artifacts/results/milestone12_paired_evidence/paired_report/paired_report.md
artifacts/results/milestone12_paired_evidence/paired_report/paired_session_evidence.csv

Stable latest report copies:
artifacts/results/latest_synthetic_master_report.html
artifacts/results/latest_synthetic/master_report.md
artifacts/results/latest_synthetic/master_report.json
artifacts/results/latest_synthetic/checkpoint_results.csv
artifacts/results/latest_synthetic/key_observations.csv
artifacts/results/latest_synthetic/session_details.csv
```

The run-specific report stays in the milestone folder.
The latest report files are replaced every time this milestone runs.

What the HTML report shows:

```text
Manager Summary:
  short explanation of the clean performance result and profiled mechanism result

How To Read This Report:
  explains which numbers are performance numbers and which numbers are attribution evidence

Key Deductions:
  highlights important lessons, especially cases where CUDA copy finished but replay still reloaded KV

Clean Performance Summary:
  profiler-off TTFT numbers

Profiled Attribution Summary:
  profiler-on CUDA HtoD, hint completion, and replay reload evidence

Timeline Summary:
  count of sessions, visible CUDA copies, late prefetches, reloads, and clean successes

Timeline:
  visual timeline from the profiled attribution run

Timeline Layers:
  explains the visual bars and markers

Prefetch Checkpoints:
  explains the three checkpoint questions: copy ready, hint done, replay reloaded KV

Checkpoint Results Per Session:
  shows pass/fail checkpoint results for each selected agent session

Key Observations Per Session:
  plain-English interpretation of what happened in each selected session

Session Details:
  raw timing fields used to defend the visual conclusions

Paired Session Evidence:
  one row per session joining the clean performance result with profiled mechanism evidence
```

Important:

```text
The timeline sections come from the profiled attribution run.
They are for mechanism evidence, not clean TTFT claims.

The clean performance table comes from the profiler-off run.
It is the right place to make latency claims.
```

Recommended run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone12_paired_evidence \
LATEST_REPORT_ROOT=artifacts/results \
CLEAN_MODES="no_prefetch direct_load oracle_direct_load" \
ATTRIBUTION_MODE=oracle_direct_load \
SESSION_COUNT=12 \
RANDOMIZE_TRAFFIC=1 \
RANDOM_SEED=7 \
ARRIVAL_GAP_RANGE_MS="60 220" \
TOOL_WAIT_RANGE_MS="250 2200" \
PROMPT_TOKEN_LIST="768 1024 1536" \
HINT_DELAY_MS=120 \
ORACLE_LEAD_MS=1000 \
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=300 \
bash scripts/run_milestone12_paired_evidence.sh Qwen/Qwen2.5-1.5B-Instruct
```

Tiny smoke run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone12_paired_smoke \
LATEST_REPORT_ROOT=artifacts/results \
CLEAN_MODES="no_prefetch oracle_direct_load" \
ATTRIBUTION_MODE=oracle_direct_load \
SESSION_COUNT=3 \
RANDOMIZE_TRAFFIC=1 \
RANDOM_SEED=7 \
ARRIVAL_GAP_RANGE_MS="80 160" \
TOOL_WAIT_RANGE_MS="500 1000" \
PROMPT_TOKEN_LIST="512" \
HINT_DELAY_MS=120 \
ORACLE_LEAD_MS=700 \
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=80 \
bash scripts/run_milestone12_paired_evidence.sh Qwen/Qwen2.5-1.5B-Instruct
```

How to read the report:

```text
Clean Performance Summary:
  use for replay TTFT and improvement numbers

Profiled Attribution Summary:
  use for CUDA HtoD copy readiness, hint completion, replay reloads, and clean-success counts

Paired Session Evidence:
  joins clean TTFT with profiled mechanism evidence by session_id
```

Open the most recent report here after any run:

```text
artifacts/results/latest_synthetic_master_report.html
```

Main rule:

```text
If a value comes from the clean run, it can support performance claims.
If a value comes from the profiled run, it supports mechanism/attribution claims.
```

### Milestone 13: Failure Stress Experiment

Status: completed on EC2.

What it is:

```text
Milestone 13 reuses the Milestone 12 paired report,
but runs a harsher traffic shape designed to create prefetch failures.

The goal is not to show the best-case speedup.
The goal is to show where software-managed hinting breaks down under pressure.
```

Why we need it:

```text
The richer Milestone 12 run already showed:
  CUDA copy ready before replay: 3 / 6
  full hint done before replay: 4 / 6
  replay reloaded KV: 6 / 6
  clean success: 0 / 6

Milestone 13 pushes harder so failures become more common and easier to explain.
```

What we observed in the manager stress run:

```text
Clean performance:
  no_prefetch avg replay TTFT:        4204.751 ms
  oracle_direct_load avg replay TTFT: 3575.177 ms
  average improvement:                629.573 ms / 14.97%

Clean hint outcomes:
  no_prefetch:         no_hint: 32
  oracle_direct_load:  late_prefetch: 32

Profiled attribution:
  sessions: 32
  CUDA copy ready before replay: 0 / 32
  full hint done before replay: 0 / 32
  replay reloaded KV: 32 / 32
  clean success: 0 / 32
  timeline late prefetch: 32 / 32
```

Simple deduction:

```text
Even under a best-effort oracle direct-load path,
every hinted session missed the prefetch deadline under stress.

Average TTFT still improved because some work was shifted or overlapped,
but the mechanism was not predictable:
  no hint finished on time,
  no CUDA copy was ready before replay,
  every replay still reloaded KV.

This is the core hardware argument:
software can issue hints,
but the memory movement path needs deadline-aware, priority-aware,
and residency-aware enforcement to make those hints reliable.
```

Important profiling note:

```text
The manager profiled run is for mechanism evidence only.
torch.profiler added large overhead.
Use the clean run for TTFT claims.

The profiled report still matters because it shows the internal outcome:
all selected sessions were late and all replays reloaded KV.
```

Failure conditions we intentionally create:

| Stress knob | Setting | Why it creates failures |
| --- | --- | --- |
| Short tool waits | `100-700 ms` | The prefetch window is tiny. |
| Bursty arrivals | `10-60 ms` between sessions | Many agents need service at nearly the same time. |
| More sessions | `32` sessions in manager preset | Many hints compete for the same serving path. |
| Larger prompts | `1536` and `2048` tokens | Larger KV footprint and more cache pressure. |
| Delayed hints | `HINT_DELAY_MS=200` | Emulates runtime/software overhead before prefetch starts. |
| Tight oracle lead | `ORACLE_LEAD_MS=100` | Even oracle-style direct load has little time to finish. |
| Higher concurrency | `TRAFFIC_CONCURRENCY=16` | Hints compete with active model work and replays. |

What success looks like:

```text
We expect more rows like:
  late_prefetch
  too_early_or_unprotected
  copy_ready_but_replay_reloaded
  hint_done_but_no_cuda_ready

These are useful failures.
They support the argument that generic DMA/copy paths do not understand
agent deadline, priority, residency, or reuse context.
```

Smoke run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

STRESS_PRESET=smoke \
bash scripts/run_milestone13_failure_stress.sh Qwen/Qwen2.5-1.5B-Instruct
```

Manager stress run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

STRESS_PRESET=manager \
bash scripts/run_milestone13_failure_stress.sh Qwen/Qwen2.5-1.5B-Instruct
```

Main output:

```text
artifacts/results/latest_synthetic_master_report.html
artifacts/results/milestone13_failure_stress/paired_report/paired_report.html
```

How to read the result:

```text
Clean Performance Summary:
  use this for TTFT and latency numbers

Timeline + Checkpoints:
  use this to show whether KV movement was ready before replay,
  whether the full hint request completed on time,
  and whether replay still reloaded KV

Key Observations Per Session:
  use this as the manager-facing explanation of each failure case
```

### Milestone 13B: Green-Bar Failure Stress

Status: completed on EC2.

What it is:

```text
Milestone 13B is a smaller stress run tuned for visualization.

Milestone 13:
  maximum failure pressure
  showed 32 / 32 late prefetch
  many green CUDA bars were missing because torch.profiler stopped before very late KV loads

Milestone 13B:
  medium pressure
  profiler export tuned for green-bar capture
  designed to show CUDA HtoD green bars while still showing failures
```

Why we need it:

```text
For manager-facing reports, the green CUDA HtoD bars make the timeline feel complete.
They show the actual host-to-device copy window, not only the SGLang-level KV-load window.
```

Recommended run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

GREEN_BAR_PRESET=medium \
bash scripts/run_milestone13b_green_bar_failure_stress.sh Qwen/Qwen2.5-1.5B-Instruct
```

Presets:

| Preset | Sessions | Profiler events | Use case |
| --- | ---: | ---: | --- |
| `medium` | `12` | `350` | Best balance: stress + green-bar visibility. |
| `replay` | `8` | `350` | Replays the known failure-heavy smoke shape with green-bar export enabled. |
| `small_full` | `6` | `350` | Highest chance of green bars, but weakest stress. |

Note: the profiler event number is an export threshold. If it is too high, the run can finish before a trace file is exported, which means the report may miss green CUDA bars even when SGLang-level KV movement happened.

Main output:

```text
artifacts/results/latest_synthetic_master_report.html
artifacts/results/milestone13b_green_bar_failure_stress/paired_report/paired_report.html
```

What we want to see:

```text
At least some sessions should show green CUDA HtoD bars.
Some sessions should still be late, reload KV, or fail clean success.

That gives us the complete visual story:
  software hint issued,
  SGLang KV movement happened,
  CUDA HtoD copy happened,
  replay timing still failed or replay still reloaded KV.
```

Observed 13B result:

```text
Clean performance run:
  no_prefetch avg replay TTFT: 1228.151 ms
  oracle_direct_load avg replay TTFT: 875.411 ms
  average improvement: 352.740 ms, or 28.72%
  oracle_direct_load outcome: late_prefetch 12 / 12

Profiled attribution run:
  sessions with visible CUDA HtoD green bars: 4 / 12
  CUDA copy ready before replay: 0 / 12
  full hint done before replay: 0 / 12
  replay reloaded KV: 12 / 12
  clean success: 0 / 12
```

Simple interpretation:

```text
This is the visual stress case we wanted.

For some agents, the report now shows green CUDA HtoD bars.
That proves the profiler can see real host-to-device data movement during the hint path.

But every agent still missed the useful prefetch checkpoint:
  the hint path was not complete before replay,
  replay still loaded KV again,
  and no session achieved clean success.

So the story is stronger:
  software can issue hints,
  CUDA copies can happen,
  but the current software-managed path is still not deadline-predictable
  and does not protect/reuse prefetched KV reliably under stress.
```

Important:

```text
Use the clean run for TTFT numbers.
Use the profiled run for green bars and CUDA/KV attribution.
Do not use profiled TTFT as performance evidence because torch.profiler can add large overhead.
```

### Milestone 14: Lightweight KV Copy Telemetry

Status: completed on EC2.

What it is:

```text
Milestone 14 is the scalable version of the timeline experiment.

Instead of using torch.profiler for every session, it records only compact
SGLang KV movement telemetry around the exact host-to-device KV load path.
```

Why we need it:

```text
Full torch-profiler traces are too heavy for large traffic runs.
They capture many unrelated kernels, memcpy events, memsets, runtime calls,
and scheduler activity.

For the manager-facing timeline, we mostly need:
  agent session id
  hint start/end
  KV host-to-device movement start/end
  replay due
  replay start
  whether replay reloaded KV
```

What changed:

```text
The SGLang trace hook now writes a compact movement-only JSONL stream:

AGENTIC_KV_COPY_TELEMETRY_PATH=<result_root>/<mode>_kv_copy_telemetry.jsonl

It records only targeted movement events such as:
  hostpool.load_to_device_per_layer
  hostpool.backup_from_device_all_layer
  evict_device / evict_host

Each telemetry row includes:
  session_id / phase / priority
  direction
  layer_id
  host/device index counts
  start/end timestamp
  duration
```

How to interpret the new green bars:

```text
Bright green KV bars:
  lightweight SGLang KV-copy telemetry
  scalable to many sessions
  best for large traffic timelines

Dark green HtoD bars:
  torch-profiler CUDA HtoD validation
  heavier
  best for small validation runs
```

Recommended run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/run_milestone14_lightweight_copy_telemetry.sh Qwen/Qwen2.5-1.5B-Instruct
```

Default shape:

```text
SESSION_COUNT=32
ATTRIBUTION_TORCH_PROFILER_ENABLE=0
TIMELINE_MAX_SESSIONS=32
```

Main output:

```text
artifacts/results/latest_synthetic_master_report.html
artifacts/results/milestone14_lightweight_copy_telemetry/paired_report/paired_report.html
```

Completed EC2 result:

```text
Clean performance:
  no_prefetch avg replay TTFT:        1477.667 ms
  oracle_direct_load avg replay TTFT: 1245.694 ms
  improvement:                         231.973 ms, or 15.7%

Attribution / timeline:
  sessions:                              32
  lightweight KV telemetry visible:       32 / 32
  torch-profiler CUDA HtoD rows:           0 / 32
  KV copy ready before replay:             0 / 32
  full hint done before replay:            0 / 32
  replay reloaded KV:                     32 / 32
  clean success:                           0 / 32
  compact telemetry rows:               2848
```

What this means:

```text
Milestone 14 proves we can get the important green KV-copy bars without
recording a huge torch-profiler trace.

It also shows the stress failure clearly:
  every session had visible KV movement,
  but every session missed the replay deadline,
  and every replay still reloaded KV.

So the failure is not "we cannot see the copy."
The failure is that the software-managed hint path did not make KV ready,
resident, and reusable by the replay deadline.
```

Simple interpretation:

```text
Use Milestone 14 to show large-scale timing behavior:
  which sessions had KV movement,
  when that movement started,
  when it finished,
  whether it beat the replay deadline,
  whether replay still reloaded KV.

Use Milestone 13B or smaller torch-profiler runs to validate that the
SGLang KV-copy telemetry corresponds to real CUDA HtoD movement.
```

### Milestone 15: Targeted DMA/HtoD Validation

Status: completed on EC2.

What it is:

```text
Milestone 15 is the small validation run for dark-green CUDA HtoD bars.

Milestone 14 scales to many sessions by using lightweight SGLang KV-copy
telemetry only.

Milestone 15 turns torch.profiler back on, but keeps the run small and
starts profiling near the hint-side KV host-to-device load path.
```

Why we need it:

```text
Light green bars show:
  SGLang executed the KV host-to-device load path.

Dark green bars show:
  torch.profiler observed CUDA host-to-device transfer events nearby.

The dark-green signal is closer to real GPU/DMA traffic, but it is too
expensive to use for large traffic runs.
```

What changed:

```text
The torch profiler can now be started with filters:

AGENTIC_KV_TORCH_PROFILER_START_EVENTS=hostpool.load_to_device_per_layer
AGENTIC_KV_TORCH_PROFILER_START_AGENT_PHASE=hint_prefetch

This means:
  do not start profiling at the first random SGLang event.
  start profiling when the hint-side KV load path begins.
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/run_milestone15_targeted_dma_validation.sh Qwen/Qwen2.5-1.5B-Instruct
```

Default shape:

```text
SESSION_COUNT=6
ATTRIBUTION_TORCH_PROFILER_ENABLE=1
AGENTIC_KV_TORCH_PROFILER_STOP_AFTER_EVENTS=220
AGENTIC_KV_TORCH_PROFILER_START_EVENTS=hostpool.load_to_device_per_layer
AGENTIC_KV_TORCH_PROFILER_START_AGENT_PHASE=hint_prefetch
```

Completed EC2 result:

```text
Profile files:                  1
CUDA events:                31941
Correlated windows:           180
Sessions with light-green KV telemetry: 3 / 6
Sessions with dark-green CUDA HtoD:     3 / 6
Green inside purple for visible rows:   yes
Hint/replay overlap observed: agent_000, 126.318 ms
Replay reloaded KV:                    6 / 6
Clean success:                         0 / 6
```

Important:

```text
Do not use Milestone 15 TTFT as performance evidence.
torch.profiler export can add large latency overhead.

Use Milestone 15 for mechanism evidence:
  one green copy-activity bar in the main timeline
  dark green when CUDA HtoD validation exists
  light green fallback when only SGLang KV telemetry exists
  whether green bars are inside purple hint windows
  whether purple hint overlaps red replay

Use Milestone 14 for larger clean timeline behavior.
```

Clean performance timeline:

```text
The paired report also includes Clean Performance Timelines.

These are built from runs with torch.profiler off, so they are better for
manager-facing TTFT and request-flow evidence.

They show:
  initial request
  tool wait
  hint request, when the mode sends one
  replay due
  replay request
  first token
  replay TTFT per session
  effective wait from replay due to first token

They intentionally do not show green CUDA/KV copy bars.
Use the profiled mechanism timeline for DMA/KV attribution.

Important:
  red replay start means the replay request was admitted.
  yellow first-token marker shows when the user actually sees output.

So no-prefetch may start replay exactly at replay due, but still pay TTFT
inside the replay request before first token appears.
```

Timeline clarity changes:

```text
Each agent row uses the compact overlapping style:
  gray: tool wait
  purple: software hint request
  green: one visible KV copy-activity bar
  black: replay due
  red: real replay request

Purple and red may overlap intentionally.
If they overlap, the software hint was still running when replay arrived.

The timeline is now focused around the prefetch/replay boundary.
This avoids spending most of the chart width on long replay generation.

Long red replay bars are clipped in the timeline and marked as continuing.
The exact replay duration still remains in the detailed tables.

The chart does not draw separate SGLang, telemetry, and torch copy bars.
It draws one green copy bar:
  prefer dark green CUDA HtoD when available
  otherwise use light green SGLang KV telemetry fallback

Green bars can be visually widened so they are easy to see.
Thin dark ticks on the green bar show the exact copy start and end.

Small labels mark only the hint boundaries:
  hint start / hint end

This keeps the chart readable when bars overlap.
Copy start/end still appear as thin ticks and tooltips.
Replay due/start timing remains visible through the black due line,
red replay edge, tooltips, and detailed tables.

If a session has no green bar, the status says:
  NO VISIBLE COPY

That does not automatically mean the experiment failed.
It means this report did not observe a host-to-device KV copy for that session.
The likely explanations are:
  the KV was already resident,
  the hint path did not trigger a load,
  or the copy was outside the captured telemetry/profiler window.

The detailed source timings remain in the tables:
  telemetry_copy_start_ms / telemetry_copy_end_ms
  torch_copy_start_ms / torch_copy_end_ms
  visible_copy_source

The report also keeps Timeline Sanity Checks:
  green_inside_purple
  light_green_inside_purple
  dark_green_inside_purple
  hint_overlaps_replay
  hint_replay_overlap_ms
  replay_reloaded_kv
```

### Milestone 16: AgentBench -> SGLang Direct

Status: validated on EC2/GPU with a single SWE-bench Pro task.

What it is:

```text
Run the realistic SWE-bench Pro -> AgentBench -> Deep Agents harness against
SGLang directly, with Dynamo removed.

Runtime path:
  SWE-bench Pro
  -> AgentBench
  -> Deep Agents
  -> SGLang OpenAI-compatible endpoint
  -> direct KV trace/reporting
```

Why we need it:

```text
The synthetic workload proves the mechanism.
This milestone proves the system can connect to real agent traffic:
  real SWE-bench tasks
  real repo checkout
  real Deep Agents planning/execution/review turns
  real tool-capable prompts
  real SGLang serving
  real KV trace hooks
```

Important design choice:

```text
Dynamo is not used.

The old AgentBench code normally sends Dynamo-style nvext fields.
For this direct-SGLang path, the wrapper patches only the ChatOpenAI client
construction point so requests carry:

  custom_params.agentic_kv

That gives SGLang trace hooks session/phase context without requiring Dynamo.
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=0 \
TOOL_CALL_PARSER=qwen25 \
bash scripts/run_milestone16_agentbench_sglang_direct.sh \
  Qwen/Qwen2.5-1.5B-Instruct
```

Recommended first run:

```text
START_INDEX=0
END_INDEX=0
RUN_PREFLIGHT=1
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=3
```

Latest validation:

```text
Run: agentbench-20260815_152259
Repo: NodeBB/NodeBB
Model turns captured: 6
Tool calls observed: 1
SGLang trace events: 396
KV copy telemetry events: 88
Replay sessions extracted: 5
```

Useful knobs:

```bash
AGENTBENCH_ROOT=~/kv_cache_offloading
START_INDEX=0
END_INDEX=2
RUN_PREFLIGHT=1
AGENTBENCH_INSTALL_DEPS=1
PROMPT_EVOLUTION_VALUE_CHAR_LIMIT=50000
MAX_TOTAL_TOKENS=16384
```

Important events to observe:

```text
SGLang /model_info becomes ready
Deep Agents tool-loop preflight passes
AgentBench task produces a run directory
SGLang trace file is written
AgentBench direct report is written
Replay workload is extracted
```

Outputs:

```text
artifacts/results/milestone16_agentbench_sglang_direct/report/agentbench_sglang_direct_report.html
artifacts/results/latest_real/agentbench_sglang_direct_report.html
artifacts/results/latest_real/agentbench_replay_workload.jsonl
```

What this milestone does not prove yet:

```text
It does not compare prefetch modes yet.
It proves the realistic live traffic path works without Dynamo.
```

### Milestone 17: Real Trace Replay Workload

Status: ready, with one current hardware compatibility blocker on `g5.2xlarge`.

What it is:

```text
Convert real AgentBench/Deep Agents model-turn traces into replayable sessions.
Each replay session contains:
  prompt from one real model turn
  replay_prompt from the next real model turn
  observed gap between those turns
  task id, repo, phase names, and priority metadata
```

Why we need it:

```text
Deep Agents owns the internal tool loop.
That makes live tool-gap prefetch hard to control directly on day one.

Trace replay gives us realistic prompts and realistic agent phase structure,
while keeping the timing/prefetch policy controlled enough for comparison.
```

Run manually if needed:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/extract_agentbench_trace_replay_workload.py \
  --index-csv artifacts/results/milestone16_agentbench_sglang_direct/agentbench_sglang_task_index.csv \
  --out-jsonl artifacts/results/milestone16_agentbench_sglang_direct/agentbench_replay_workload.jsonl \
  --out-csv artifacts/results/milestone16_agentbench_sglang_direct/agentbench_replay_workload.csv \
  --max-sessions 24
```

Important events to observe:

```text
The extractor finds AgentBench phase request/response pairs.
The JSONL workload has real prompts.
The workload has enough rows for a replay experiment.
```

### Milestone 18: Real Prompt Prefetch Modes

Status: validated on EC2/GPU with the latest AgentBench replay workload.

What it is:

```text
Use the real AgentBench replay workload from Milestone 16/17, then compare:

  no_prefetch
  direct_load
  oracle_direct_load

This is the controlled prefetch experiment using real SWE-bench/DeepAgents
prompts instead of synthetic filler prompts.
```

Why we need it:

```text
Milestone 16 gives live realism.
Milestone 18 gives clean A/B comparison.

Together:
  live run shows the harness is real
  replay run shows the prefetch policy impact
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

WORKLOAD_JSONL=artifacts/results/latest_real/agentbench_replay_workload.jsonl \
MODES="no_prefetch direct_load oracle_direct_load" \
ORACLE_LEAD_MS=500 \
TRAFFIC_CONCURRENCY=4 \
bash scripts/run_milestone18_agentbench_trace_replay_modes.sh \
  Qwen/Qwen2.5-1.5B-Instruct
```

Important events to observe:

```text
Each mode starts a fresh SGLang server.
The same real AgentBench prompts are replayed for every mode.
Hint outcomes are classified for each mode.
Traffic summary shows avg replay TTFT and outcomes.
```

Outputs:

```text
artifacts/results/milestone18_agentbench_trace_replay_modes/traffic_summary.html
artifacts/results/latest_real/agentbench_replay_mode_summary.html
artifacts/results/latest_real/agentbench_replay_mode_summary.csv
```

Latest validation:

```text
Replay sessions: 5
Modes compared: no_prefetch, direct_load, oracle_direct_load

Average replay TTFT:
  no_prefetch:        107.070 ms
  direct_load:        120.002 ms
  oracle_direct_load:  91.353 ms

Interpretation:
  On this small real-prompt replay, oracle_direct_load was best on average.
  The non-oracle hint modes were worse because the extracted waits were very short,
  so hints often arrived too late or competed with live replay work.
```

### Milestone 19: Realistic Manager Report

Status: ready as a report path.

What it is:

```text
A manager-facing evidence package combining:
  live AgentBench -> SGLang direct report
  real prompt replay workload
  prefetch-mode comparison summary
  SGLang KV movement trace summary
```

Why we need it:

```text
This lets us tell the full story:

1. The workload is realistic.
2. Agent/tool pauses naturally create prefetch windows.
3. Software-only prefetch can help, but can also be late or wasted.
4. Direct KV movement gives better mechanism visibility.
5. The missing piece is enforceable, deadline-aware, residency-aware hardware/runtime support.
```

Current report files:

```text
artifacts/results/latest_real/agentbench_sglang_direct_report.html
artifacts/results/latest_real/agentbench_replay_mode_summary.html
artifacts/results/latest_real/agentbench_replay_workload.csv
```

Simple interpretation:

```text
Milestone 16 answers:
  Can we run real AgentBench/DeepAgents traffic directly on SGLang?

Milestone 18 answers:
  On real AgentBench prompts, how do the prefetch modes compare?

The next deeper step after this is live tool-gap hooks inside Deep Agents,
so hints can be issued exactly when a real tool starts and evaluated exactly
when that tool returns.
```

### Milestone 20: SWE-bench Trajectory Prompt Replay

Status: ready.

What it is:

```text
Use real SWE-bench trajectory prompts as the request source for the existing
paired-report experiment.

This milestone changes only the prompt source:
  synthetic prompts
  -> real prompts from latest_swebench_trajectory_prompt_catalog.csv

The rest stays the same:
  clean performance run
  profiled/lightweight attribution run
  paired evidence report
  same timeline/checkpoint/session-detail format as latest_synthetic_master_report.html
```

Why we need it:

```text
This makes the manager-facing report look almost identical to the synthetic
paired report, while replacing toy prompts with real SWE-bench trajectory
prompts.

For now, tool waits are synthetic/speculated. Later, we can replace them with
actual tool wait timestamps from the trajectory source.
```

Expected input:

```text
experiments/reports/latest_swebench_trajectory_prompt_catalog.csv

The catalog must contain prompt_text_path or a similar prompt-path column.
Each row points to a real prompt text file.
```

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

CATALOG_CSV=~/kv_cache_offloading/experiments/reports/latest_swebench_trajectory_prompt_catalog.csv \
MAX_SESSIONS=6 \
CLEAN_MODES="no_prefetch oracle_direct_load" \
ATTRIBUTION_TORCH_PROFILER_ENABLE=0 \
TOOL_WAIT_LIST_MS="250 500 900 1600 3000" \
bash scripts/run_milestone20_swebench_trajectory_replay.sh \
  Qwen/Qwen2.5-1.5B-Instruct
```

Important events to observe:

```text
The converter writes a real-prompt workload JSONL.
Milestone 12 runs using WORKLOAD_JSONL instead of synthetic prompts.
latest_synthetic_master_report.html is regenerated with the same paired-report format.
latest_synthetic/swebench_trajectory_paired_report.html is also written as a stable detail copy.
```

Outputs:

```text
artifacts/results/milestone20_swebench_trajectory_replay/swebench_trajectory_replay_workload.jsonl
artifacts/results/latest_synthetic/swebench_trajectory_replay_workload.jsonl
artifacts/results/latest_synthetic_master_report.html
artifacts/results/latest_synthetic/swebench_trajectory_paired_report.html
```

Simple interpretation:

```text
Milestone 20 answers:
  What happens when we keep the paired evidence experiment the same,
  but replace synthetic prompts with real SWE-bench trajectory prompts?
```

### Milestone 21: Direct SGLang Experiment 6 Prompt Evolution

Status: ready.

What it is:

```text
Run the working Experiment 6 prompt-evolution batch shape from kv_cache_offloading,
but remove Dynamo and send Deep Agents directly to SGLang.

This milestone generates the real SWE-bench trajectory prompt catalog that
Milestone 20 consumes.
```

Why we need it:

```text
Milestone 20 needs real trajectory prompts.
Previously, those prompts came from kv_cache_offloading Experiment 6.
This milestone regenerates that source inside the direct-SGLang workflow:

SWE-bench task
  -> Deep Agents tool loop
  -> SGLang OpenAI-compatible endpoint
  -> prompt evolution reports
  -> trajectory prompt catalog
```

What changed from the original Experiment 6:

| Original Exp6 Piece | Direct-SGLang Version |
| --- | --- |
| Dynamo frontend on port 8000 | SGLang endpoint on port 30000 plus a lightweight tool-normalizer proxy |
| Dynamo start/stop wrapper | direct SGLang server launcher |
| GH200-oriented model path | direct SGLang path using `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` |
| same Deep Agents tool loop | kept |
| same SWE-bench task loop | kept |
| same prompt-evolution reports | kept |
| same trajectory catalog format | kept |

Tiny g5.2xlarge wiring validation run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_ID="exp6_direct_qwen7b_synced_smoke_$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="artifacts/results/${RUN_ID}" \
LATEST_REPORT_ROOT="artifacts/results" \
AGENTBENCH_ROOT=~/kv_cache_offloading \
PROMPT_EVOLUTION_BATCH_ID="${RUN_ID}" \
START_INDEX=0 \
END_INDEX=1 \
REUSE_SERVER=0 \
SERVER_MODE=simple \
MAX_TOTAL_TOKENS=32768 \
SERVER_READY_TIMEOUT_SECS=1800 \
TOOL_CALL_PARSER=auto \
SAMPLING_BACKEND=pytorch \
SAMPLING_DEFAULTS=openai \
ENABLE_TOOL_NORMALIZER_PROXY=1 \
EXTRA_SERVER_ARGS="--disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule" \
AGENTBENCH_DEEPAGENTS_SOURCE=upstream \
AGENTBENCH_EXECUTION_LOOP=1 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=3 \
AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST=0 \
AGENTBENCH_EXECUTION_GUARD=0 \
AGENTBENCH_FORCE_TOOL_CHOICE=auto \
AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT=1 \
AGENTBENCH_SOFT_STOP_RECURSION=1 \
AGENTBENCH_AGENT_RECURSION_LIMIT=1000 \
AGENTBENCH_TRACE_AGENT_STREAM=0 \
AGENTBENCH_DIRECT_SGLANG_TOOL_RICH=1 \
AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT=1 \
AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS=1 \
AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD=1 \
PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP=1 \
PROMPT_EVOLUTION_TOOL_LOOP_CASE=ls-read-execute \
bash scripts/run_milestone21_exp6_direct_sglang.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Standard g5.2xlarge tool-traffic run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_ID="qwen25_standard_16tasks_steps10_$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="artifacts/results/${RUN_ID}" \
LATEST_REPORT_ROOT="artifacts/results" \
AGENTBENCH_ROOT=~/kv_cache_offloading \
PROMPT_EVOLUTION_BATCH_ID="${RUN_ID}" \
START_INDEX=0 \
END_INDEX=15 \
REUSE_SERVER=0 \
SERVER_MODE=simple \
MAX_TOTAL_TOKENS=32768 \
SERVER_READY_TIMEOUT_SECS=1800 \
TOOL_CALL_PARSER=auto \
SAMPLING_BACKEND=pytorch \
SAMPLING_DEFAULTS=openai \
ENABLE_TOOL_NORMALIZER_PROXY=1 \
EXTRA_SERVER_ARGS="--disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule" \
AGENTBENCH_INSTALL_DEPS=0 \
RUN_PREFLIGHT=1 \
AGENTBENCH_DEEPAGENTS_SOURCE=upstream \
AGENTBENCH_EXECUTION_LOOP=1 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST=0 \
AGENTBENCH_EXECUTION_GUARD=0 \
AGENTBENCH_FORCE_TOOL_CHOICE=auto \
AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT=1 \
AGENTBENCH_SOFT_STOP_RECURSION=1 \
AGENTBENCH_AGENT_RECURSION_LIMIT=1000 \
AGENTBENCH_TRACE_AGENT_STREAM=0 \
PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP=1 \
PROMPT_EVOLUTION_TOOL_LOOP_CASE=ls-read-execute \
AGENTBENCH_DIRECT_SGLANG_TOOL_RICH=1 \
AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT=1 \
AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS=1 \
AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD=1 \
AGENTBENCH_BATCH_CONTINUE_ON_ERROR=1 \
bash scripts/run_milestone21_exp6_direct_sglang.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Use `AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=15` when you want a deeper, longer run.
The Milestone 21, 22, and 23 wrappers now default to this standard:
`START_INDEX=0`, `END_INDEX=15`, and `AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10`.

Expected good 7B preflight on g5.2xlarge:

```text
tool_calls=3
tool_messages=3
unique_tools=execute,ls,read_file
multi_tool_loop_observed=True
case_success=True
Deep Agents tool-loop preflight passed.
```

Previously validated 4-task tool-traffic run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_ID="qwen25_safeedit_4tasks_steps10_$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="artifacts/results/${RUN_ID}" \
LATEST_REPORT_ROOT="artifacts/results" \
AGENTBENCH_ROOT=~/kv_cache_offloading \
PROMPT_EVOLUTION_BATCH_ID="${RUN_ID}" \
START_INDEX=0 \
END_INDEX=3 \
REUSE_SERVER=0 \
SERVER_MODE=simple \
MAX_TOTAL_TOKENS=32768 \
SERVER_READY_TIMEOUT_SECS=1800 \
TOOL_CALL_PARSER=auto \
SAMPLING_BACKEND=pytorch \
SAMPLING_DEFAULTS=openai \
ENABLE_TOOL_NORMALIZER_PROXY=1 \
TOOL_NORMALIZER_PORT=31032 \
EXTRA_SERVER_ARGS="--disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule" \
AGENTBENCH_INSTALL_DEPS=0 \
RUN_PREFLIGHT=0 \
AGENTBENCH_DEEPAGENTS_SOURCE=upstream \
AGENTBENCH_EXECUTION_LOOP=1 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
AGENTBENCH_EXECUTION_LOOP_REQUIRE_TEST=0 \
AGENTBENCH_EXECUTION_GUARD=0 \
AGENTBENCH_FORCE_TOOL_CHOICE=auto \
AGENTBENCH_DISABLE_GENERAL_PURPOSE_SUBAGENT=1 \
AGENTBENCH_SOFT_STOP_RECURSION=1 \
AGENTBENCH_AGENT_RECURSION_LIMIT=1000 \
AGENTBENCH_TRACE_AGENT_STREAM=0 \
PROMPT_EVOLUTION_REQUIRE_TOOL_LOOP=0 \
AGENTBENCH_DIRECT_SGLANG_TOOL_RICH=1 \
AGENTBENCH_DIRECT_SGLANG_VIRTUAL_TOOL_ROOT=1 \
AGENTBENCH_DIRECT_SGLANG_EXCLUDE_WRITE_TODOS=1 \
AGENTBENCH_DIRECT_SGLANG_SAFE_EDIT_GUARD=1 \
AGENTBENCH_BATCH_CONTINUE_ON_ERROR=1 \
bash scripts/run_milestone21_exp6_direct_sglang.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Current `g5.2xlarge` note:

```text
The Qwen3-Coder-30B-FP8 model can be found and loaded on the current A10G
machine, but the first generation request fails inside the FP8 MoE Triton path:

ValueError("type fp8e4nv not supported in this architecture...")

That means HF access and parser wiring are not the main blocker anymore.
The current blocker is GPU/backend compatibility for this FP8 MoE model.
For the real Exp6-style run, use a GPU/backend with compatible FP8 MoE support,
for example a GH200/H100/H200-class setup. A smaller model can validate wiring,
but should not be treated as final realistic evidence.
```

Direct-SGLang tool-call health fixes:

```text
1. Parser:
   Qwen/Qwen2.5-Coder-7B-Instruct must use SGLang tool parser qwen25.
   The wrapper now maps TOOL_CALL_PARSER=auto -> qwen25 for Qwen2.5 models.

2. Tool filesystem root:
   DeepAgents tools see the SWE-bench repo mounted at `/`.
   The prompt patch now tells the model to use `/` or relative paths,
   not the EC2 host checkout path.

3. Tool discipline:
   The prompt/tool descriptions now say:
   - use ls for directories,
   - use read_file for file paths,
   - read before edit,
   - never call edit_file with empty old_string.

4. Safe edit guard:
   The harness now rejects empty edit_file old_string at the backend layer.
   This prevents one bad 7B edit call from creating a huge corrupt patch.
```

Observed 7B fallback result on `g5.2xlarge` after the fixes:

```text
Model: Qwen/Qwen2.5-Coder-7B-Instruct
Server result: SGLang loaded successfully, allocated KV cache, and passed smoke chat.
Parser result: TOOL_CALL_PARSER=auto selected qwen25.
Deep Agents result: natural auto mode executed real structured tools
through the normalizer proxy.

Validated preflight:
tool_calls=3
tool_messages=3
unique_tools=execute,ls,read_file
case_success=True

Validated single-task debug:
actual task model requests=38
actual structured tool calls=25
tool_intent_without_structured_call=0
workspace.patch size=0 bytes

Validated 4-task batch:
actual task model requests=93
structured tool-call rows=49
structured tool calls=49
tool mix: read_file=24, ls=12, edit_file=11, write_file=2
tool_intent_without_structured_call=0
trajectory prompt catalog rows=44
tasks completed=4/4

Interpretation:
The "less than a dozen tools" symptom is no longer a parser/harness blocker.
The direct-SGLang path can now produce real structured SWE-bench tool traffic.

However, the 7B model is still weaker than the 30B/40B coder setup:
it sometimes loops, uses shallow tool strategies, or attempts bad edits that
the safe edit guard must reject. Use the 7B path for A10G wiring and live
tool-gap/KV experiments. For manager-grade SWE-bench tool diversity and
solution quality, still prefer a larger Qwen3-Coder/40B-class model on a
compatible GPU/backend.
```

Why the tool-normalizer proxy exists:

```text
Without Dynamo, direct SGLang sometimes returns Qwen/Hermes tool calls as text,
for example JSON inside <tools> or <|im_start|> wrappers.

Deep Agents needs OpenAI-style structured tool_calls.
The proxy converts clear textual tool-call payloads into structured tool_calls,
which restores the frontend behavior needed for the Dynamo-free testbed.
```

Important events to observe:

```text
SGLang starts directly; Dynamo is not used.
TOOL_CALL_PARSER=auto resolves to qwen25 for Qwen2.5 models.
The smoke chat request passes.
The Deep Agents tool-loop preflight passes.
The proxy shows structured tool_calls, not only tool-looking text.
tool_intent_without_structured_call stays near zero.
Unsafe empty edit_file old_string calls are rejected by the safe edit guard.
Each SWE-bench task produces a new AgentBench result directory.
The task trace index is updated.
The trajectory prompt catalog is generated.
```

Key output files:

```text
artifacts/results/milestone21_exp6_direct_sglang/driver.log
artifacts/results/milestone21_exp6_direct_sglang/exp6_direct_sglang_task_index.csv
artifacts/results/milestone21_exp6_direct_sglang/exp6_direct_swebench_trajectory_prompt_catalog.csv
artifacts/results/latest_exp6_direct_swebench_trajectory_prompt_catalog.csv
~/kv_cache_offloading/experiments/reports/latest_swebench_trajectory_prompt_catalog.csv
```

Then run Milestone 20 using the generated catalog:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

CATALOG_CSV=~/kv_cache_offloading/experiments/reports/latest_swebench_trajectory_prompt_catalog.csv \
MAX_SESSIONS=6 \
CLEAN_MODES="no_prefetch oracle_direct_load" \
ATTRIBUTION_TORCH_PROFILER_ENABLE=0 \
TOOL_WAIT_LIST_MS="250 500 900 1600 3000" \
bash scripts/run_milestone20_swebench_trajectory_replay.sh \
  Qwen/Qwen2.5-1.5B-Instruct
```

Simple interpretation:

```text
Milestone 21 answers:
  Can we generate real SWE-bench/Deep Agents prompt trajectories without Dynamo?

Milestone 20 answers:
  What happens when those real trajectories replace the synthetic prompts
  in our paired prefetch evidence report?
```

### Milestone 22: Live AgentBench Tool-Gap Bridge

Status: ready.

What it is:

```text
Run real AgentBench / Deep Agents traffic directly through SGLang,
capture the live OpenAI-compatible requests at the proxy,
and build an analysis report from the actual tool-call gaps.
```

Why we need it:

```text
Milestone 20 used real prompts, but still replayed them in a controlled driver.
Milestone 22 starts moving closer to the real system:

SWE-bench task
  -> Deep Agents
  -> real tool calls
  -> tool execution gap
  -> next live model request
  -> direct SGLang

This lets us observe the real wait windows created by real tools.
Those windows are where a future hint-guided KV prefetch path would act.
```

Simple timeline:

```text
0 ms:    live model request finishes and emits tool_calls=[ls, read_file]
10 ms:   Deep Agents executes those tools
420 ms:  next live model request starts with the tool results

The 410 ms gap is the live prefetch opportunity window.
```

What this milestone does not do yet:

```text
It does not inject prefetch requests yet.
It is observe-only.

The goal is to prove that live Deep Agents traffic gives us real,
measurable tool-gap windows that can feed the same analysis infrastructure.

The Deep Agents preflight is excluded from the report by default.
Set INCLUDE_PREFLIGHT_IN_REPORT=1 only when debugging the preflight itself.
```

Standard `g5.2xlarge` command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_ID="milestone22_live_qwen7b_$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="artifacts/results/${RUN_ID}" \
LATEST_REPORT_ROOT="artifacts/results" \
AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=15 \
REUSE_SERVER=0 \
SERVER_MODE=simple \
MAX_TOTAL_TOKENS=32768 \
TOOL_CALL_PARSER=auto \
SAMPLING_BACKEND=pytorch \
SAMPLING_DEFAULTS=openai \
ENABLE_TOOL_NORMALIZER_PROXY=1 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
PROMPT_EVOLUTION_TOOL_LOOP_CASE=ls-read-execute \
bash scripts/run_milestone22_live_agentbench_bridge.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Deeper live-gap distribution run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_ID="milestone22_live_qwen7b_bigger_$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="artifacts/results/${RUN_ID}" \
LATEST_REPORT_ROOT="artifacts/results" \
AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=15 \
REUSE_SERVER=1 \
SERVER_MODE=simple \
MAX_TOTAL_TOKENS=32768 \
TOOL_CALL_PARSER=auto \
SAMPLING_BACKEND=pytorch \
SAMPLING_DEFAULTS=openai \
ENABLE_TOOL_NORMALIZER_PROXY=1 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=15 \
PROMPT_EVOLUTION_TOOL_LOOP_CASE=ls-read-execute \
bash scripts/run_milestone22_live_agentbench_bridge.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Build the Milestone 22 report from an existing Milestone 21 result:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_AGENTBENCH=0 \
EXISTING_RESULT_ROOT=artifacts/results/milestone21_qwen7b_synced_smoke_20260816_002909 \
LATEST_REPORT_ROOT=artifacts/results \
bash scripts/run_milestone22_live_agentbench_bridge.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Important events to observe:

```text
The proxy log captures live chat/completions requests.
Some requests return structured tool_calls.
The next request in the same live run starts after the tool executes.
The report converts that into a blue -> gray -> red timeline:
  blue = model turn that emitted tool calls
  gray = observed tool/harness wait gap
  red = next live model turn
```

Key output files:

```text
artifacts/results/<run>/tool_normalizer_proxy.jsonl
artifacts/results/<run>/live_agentbench_tool_gap_report/live_agentbench_tool_gap_report.html
artifacts/results/<run>/live_agentbench_tool_gap_report/live_tool_gaps.csv
artifacts/results/<run>/live_agentbench_tool_gap_report/live_requests.csv
artifacts/results/latest_real/m22_live_tool_gap_report.html
artifacts/results/latest_real/m22_live_tool_gaps.csv
artifacts/results/latest_real/m22_live_requests.csv
```

Simple interpretation:

```text
Milestone 22 answers:
  Can we feed real Deep Agents tool-call traffic into the analysis system live?

Next step after this:
  Use these live tool-gap windows to trigger direct SGLang KV loads,
  then compare whether the prefetch path finishes before the real resume turn.
```

### Milestone 23: Live Prefetch Intervention

Status: ready.

What it is:

```text
Turn the Milestone 22 live gap observer into a first live intervention path.

When the proxy sees a real model response with structured tool_calls,
it writes a live hint event.

A controller tails those hint events and immediately sends a tiny
prefetch/direct-load request directly to SGLang while Deep Agents is busy
executing the tool.
```

Why we need it:

```text
Milestone 22 proves the opportunity window exists.
Milestone 23 tests whether a software hint path can act inside that window.

This is the first live version of the hardware story:

agent emits tool call
  -> runtime knows this session may resume soon
  -> runtime submits a prefetch hint
  -> SGLang receives a marked prefetch/direct-load request
  -> real agent resume request arrives later
```

Simple timeline:

```text
0 ms:   live model request returns tool_call=execute
2 ms:   proxy writes live_hint.submitted
5 ms:   controller starts live prefetch request
40 ms:  controller prefetch request ends
120 ms: tool finishes and the real agent resume request starts

If the purple controller bar ends before the black resume boundary,
the software hint path was early enough for that turn.
If it ends after the black boundary, the software hint path was late.
```

Recommended `g5.2xlarge` smoke:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_ID="milestone23_live_prefetch_qwen7b_$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="artifacts/results/${RUN_ID}" \
LATEST_REPORT_ROOT="artifacts/results" \
AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=15 \
REUSE_SERVER=1 \
SERVER_MODE=simple \
MAX_TOTAL_TOKENS=32768 \
TOOL_CALL_PARSER=auto \
SAMPLING_BACKEND=pytorch \
SAMPLING_DEFAULTS=openai \
ENABLE_TOOL_NORMALIZER_PROXY=1 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
PROMPT_EVOLUTION_TOOL_LOOP_CASE=ls-read-execute \
bash scripts/run_milestone23_live_prefetch_intervention.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Important events to observe:

```text
live_hint.submitted appears after a real tool-call response.
live_prefetch.start appears when the controller begins acting on the hint.
live_prefetch.end appears when the controller request finishes.
The HTML report shows:
  blue   = real model turn that emitted tool calls
  gray   = real tool/harness wait gap
  purple = live controller prefetch/direct-load request
  black  = real resume boundary
  red    = real resume model turn
```

Key output files:

```text
artifacts/results/<run>/live_hint_events.jsonl
artifacts/results/<run>/live_hint_payloads/
artifacts/results/<run>/live_prefetch_controller.jsonl
artifacts/results/<run>/live_agentbench_prefetch_report/live_agentbench_tool_gap_report.html
artifacts/results/latest_real/m23_live_prefetch_report.html
artifacts/results/latest_real/m23_live_tool_gaps.csv
artifacts/results/latest_real/m23_live_requests.csv
```

Simple interpretation:

```text
Milestone 23 answers:
  Once real tool calls create a wait gap, can a software controller react
  quickly enough to submit and finish a prefetch-style request before resume?

This is still software emulation.
The hardware argument becomes stronger when purple bars are late,
overlap red bars, or add interference despite having the right semantic hint.
```

### Milestone 24: Live Paired AgentBench Report

Status: ready.

What it is:

```text
Run two live SWE-bench / Deep Agents / direct-SGLang experiments:

1. no_prefetch
2. live_prefetch_intervention

Then build one paired report that looks like the earlier paired evidence report,
but uses real live AgentBench traffic instead of synthetic requests.
```

Why we need it:

```text
Milestone 12 gave us a clear manager-style report, but the traffic was synthetic.
Milestone 22 and 23 gave us live traffic, but each report was single-run.

Milestone 24 combines both ideas:

real SWE-bench task
  -> real Deep Agents tool calls
  -> direct SGLang
  -> no-prefetch run
  -> live-prefetch run
  -> paired report
```

Simple timeline:

```text
No prefetch:
blue model turn -> gray tool wait -> black resume boundary -> red resume turn

Live prefetch:
blue model turn -> gray tool wait
                 -> purple prefetch request
                 -> black resume boundary -> red resume turn

If purple ends after black, the software prefetch path was late.
If purple ends before black, the software prefetch path met that resume window.
```

Standard `g5.2xlarge` command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_ID="milestone24_live_paired_qwen7b_$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="artifacts/results/${RUN_ID}" \
LATEST_REPORT_ROOT="artifacts/results" \
AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=15 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
TOOL_CALL_PARSER=auto \
SAMPLING_BACKEND=pytorch \
SAMPLING_DEFAULTS=openai \
ENABLE_TOOL_NORMALIZER_PROXY=1 \
AGENTBENCH_BATCH_CONTINUE_ON_ERROR=1 \
bash scripts/run_milestone24_live_paired_agentbench_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Useful faster debug command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RUN_ID="milestone24_debug_qwen7b_$(date +%Y%m%d_%H%M%S)"

RESULT_ROOT="artifacts/results/${RUN_ID}" \
LATEST_REPORT_ROOT="artifacts/results" \
AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=3 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
RUN_PREFLIGHT=0 \
bash scripts/run_milestone24_live_paired_agentbench_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Important events to observe:

```text
The no-prefetch run captures real structured tool calls.
The live-prefetch run emits live_hint.submitted events.
The live-prefetch controller emits live_prefetch.start and live_prefetch.end.
The paired report shows:
  Summary
  Manager Summary
  Key Deductions
  Clean Performance Summary
  Clean Performance Timelines
  Timeline Summary
  Timeline
  Timeline Layers
  Prefetch Checkpoints
  Checkpoint Results Per Session
  Key Observations Per Session
  Session Details
  Paired Session Evidence
```

Key output files:

```text
artifacts/results/<run>/no_prefetch_live/
artifacts/results/<run>/live_prefetch/
artifacts/results/<run>/live_paired_report/live_paired_agentbench_report.html
artifacts/results/<run>/live_paired_report/live_paired_session_evidence.csv
artifacts/results/latest_master_report.html
artifacts/results/latest_real/master_report.json
artifacts/results/latest_real/session_evidence.csv
artifacts/results/latest_real/m24_live_paired_report.html
artifacts/results/latest_real/m24_live_paired_report.json
artifacts/results/latest_real/m24_live_paired_report.md
```

Simple interpretation:

```text
Milestone 24 answers:
  In a real live AgentBench/SGLang workflow, does the software hint path
  finish before the real post-tool resume request, and does the paired run
  improve resume request latency?

Important metric note:
  The live proxy measures full resume request latency, not streaming TTFT.
  Use it as a clean end-to-end latency signal until we add streaming TTFT.

Important pairing note:
  The two live runs can diverge because the model may choose different tools.
  The report pairs rows by SWE-bench task index plus gap order inside the task.
  Aggregate trends are stronger than any single row.
```

### Milestone 25: Labeled Reproducible Master Reports

Status: ready.

Why this milestone is needed:

```text
The master reports are now the main artifacts we show and discuss.
But repeated runs should not accidentally overwrite the current latest report.

Milestone 25 adds labeled report scripts so each run can create an archived
manager-demo copy, while still allowing us to refresh the standard latest
report only when we explicitly ask for it.
```

What it is:

```text
Three small wrapper scripts:

1. run_labeled_live_master_report.sh
   Runs the full real SWE-bench / DeepAgents / SGLang paired experiment
   and writes a labeled live master report.

2. build_labeled_live_master_report.sh
   Rebuilds a labeled live master report from existing no-prefetch and
   live-prefetch run folders.

3. build_labeled_synthetic_master_report.sh
   Rebuilds a labeled synthetic master report from existing clean and
   profiled synthetic run folders.
```

Run a new labeled real live experiment:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=manager_demo_1 \
UPDATE_LATEST=0 \
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
START_INDEX=0 \
END_INDEX=15 \
MAX_STEPS=10 \
AGENTBENCH_ROOT=~/kv_cache_offloading \
bash scripts/run_labeled_live_master_report.sh
```

Output:

```text
artifacts/results/labeled/live/manager_demo_1/master_report.html
```

Rebuild a labeled live report from existing run folders:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=manager_demo_1_rebuild \
UPDATE_LATEST=0 \
NO_PREFETCH_ROOT=artifacts/results/<run>/no_prefetch_live \
PREFETCH_ROOT=artifacts/results/<run>/live_prefetch \
bash scripts/build_labeled_live_master_report.sh
```

Build a labeled synthetic report from existing synthetic runs:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=synthetic_manager_demo_1 \
UPDATE_LATEST=0 \
CLEAN_ROOT=artifacts/results/milestone15_targeted_dma_validation/clean_performance \
ATTRIBUTION_ROOT=artifacts/results/milestone15_targeted_dma_validation/profiled_attribution \
bash scripts/build_labeled_synthetic_master_report.sh
```

Output:

```text
artifacts/results/labeled/synthetic/synthetic_manager_demo_1/master_report.html
```

When to update the standard latest reports:

```text
Default:
  UPDATE_LATEST=0
  Create an archived labeled report.
  Do not overwrite latest_master_report.html or latest_synthetic_master_report.html.

Explicit refresh:
  UPDATE_LATEST=1
  Also update the standard latest report:
    artifacts/results/latest_master_report.html
    artifacts/results/latest_synthetic_master_report.html
```

Important events to observe:

```text
The script prints the labeled output path.
The labeled folder contains:
  master_report.html
  master_report.json, when available
  master_report.md, when available
  run_config.env

The generated HTML reports also include a "Reproduce This Report" section
with these same copy-paste commands.
```

The standard master report also includes an **Experiment Machine And Runtime
Configuration** subsection near the top. It records the EC2 instance, GPU name,
GPU memory size/type, host RAM, model, SGLang version, context length, and the
configured HiCache host KV shelf. This is important because physical host RAM
and the SGLang HiCache allocation are not the same thing.

### Milestone 26: Live Direct KV Load Intervention

Status: ready.

Why this milestone is needed:

```text
Milestone 24 used real DeepAgents/SWE-bench traffic, but the live prefetch
path was mostly a request-level prewarm signal.

Milestone 26 makes the live path more realistic:

real tool call
  -> live hint
  -> marked direct-load request
  -> SGLang's own HiRadix/HiCache load-back path
  -> trace evidence for init_load_back, load_back, hicache.load, or HtoD copy telemetry
```

Important nuance:

```text
The controller is outside the SGLang worker process.
So it cannot directly call a Python cache object inside SGLang.

Instead, it sends a marked direct-load request that exercises the same
SGLang load-back path we instrumented earlier.

The report then checks whether the real SGLang hooks observed:
  hiradix.init_load_back
  hiradix.load_back
  hicache.load
  hostpool.load_to_device_per_layer
  lightweight host-to-device copy telemetry
```

Run one live direct-KV intervention:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone26_direct_kv_qwen7b_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=15 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
AGENTBENCH_DIRECT_SGLANG_MAX_TOKENS=512 \
SERVER_MODE=hicache \
HICACHE_SIZE_GB=8 \
LIVE_PREFETCH_ACTION=direct_load \
bash scripts/run_milestone26_live_direct_kv_load_intervention.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Run a paired no-prefetch versus live direct-KV report:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone26_paired_direct_kv_qwen7b_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=15 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
SERVER_MODE=hicache \
HICACHE_SIZE_GB=8 \
bash scripts/run_milestone26_live_paired_direct_kv_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Important events to observe:

```text
live_hint.submitted
  The proxy saw a real model turn produce structured tool calls.

live_prefetch.start / live_prefetch.end
  The controller sent a direct-load hint request during the tool wait.

hiradix.init_load_back / hiradix.load_back
  SGLang's prefix/radix cache entered the load-back path.

hicache.load / hostpool.load_to_device_per_layer
  SGLang attempted host-to-device KV movement.

kv_telemetry.copy.*
  Lightweight copy telemetry saw host-to-device copy activity.
```

Key output files:

```text
artifacts/results/<run>/live_direct_kv_trace.jsonl
artifacts/results/<run>/live_direct_kv_copy_telemetry.jsonl
artifacts/results/<run>/live_direct_kv_load_report/live_direct_kv_load_report.html
artifacts/results/<run>/live_direct_kv_load_report/live_direct_kv_load_evidence.csv
artifacts/results/latest_real/m26_live_direct_kv_load_report.html
artifacts/results/latest_master_report.html
```

Simple interpretation:

```text
If the direct-KV report shows load-back/copy events:
  The live tool-call hint reached the real SGLang KV movement path.

If hints are still late:
  We have stronger evidence that the ordinary software/SGLang path can be
  too slow even when it knows the right session.

If there is no matching load-back/copy:
  Either the KV was still resident in GPU memory, the prefix was not in host
  cache, or attribution did not connect that hint to the internal copy event.
```

### Milestone 27: Real-Prompt Controlled Replay

Status: ready.

Why this milestone is needed:

```text
The fully live AgentBench runs are realistic, but hard to control:
  tool waits are whatever the harness produces
  cache pressure varies naturally
  prefetch timing is noisy

Milestone 27 keeps real AgentBench/DeepAgents prompt content, but controls:
  tool wait duration
  cache pressure
  direct KV hint timing
  replay timing

This gives a cleaner hardware-style experiment.
```

What it does:

```text
real prompt pair from AgentBench traces
  -> send Turn A to SGLang
  -> impose a controlled tool-wait window
  -> create cache pressure with filler requests
  -> optionally issue a direct KV load hint during the wait
  -> send Turn B exactly at the scheduled resume time
  -> measure whether the hint-side KV movement beat the replay path
```

Modes:

```text
no_prefetch:
  no hint is issued.
  replay arrives and SGLang handles KV normally.

direct_prefetch:
  a marked direct-load hint is issued during the controlled tool wait.
  this exercises SGLang's direct KV load-back path.

dynamo_priority_hints:
  emits a Dynamo-style hint payload under custom_params.nvext.agent_hints.
  the proxy driver translates that hint into SGLang's native priority field:
    - urgent replay requests get high priority
    - background filler pressure gets low priority
    - SGLang is launched with priority scheduling enabled for this mode
  this is the closest current-software baseline for "Dynamo hint -> SGLang
  priority scheduling" without using the full Dynamo stack.
  it does not call our direct KV prefetch hook.

oracle_prefetch:
  optional advanced mode for later sensitivity studies.
  excluded from the default run so the main report stays simple.
```

Run with an existing real prompt-pair workload:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone27_real_prompt_controlled_replay_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
WORKLOAD_JSONL=/path/to/real_prompt_pairs.jsonl \
MAX_PAIRS=12 \
MODES="no_prefetch dynamo_priority_hints" \
TOOL_WAIT_LIST_MS="100 250 500 1000" \
FILLER_LIST="16 64" \
PREFETCH_TIMING=near_resume \
PRIORITY_DIRECT_PREFETCH=0 \
DYNAMO_HIGH_PRIORITY=100 \
DYNAMO_NORMAL_PRIORITY=0 \
DYNAMO_LOW_PRIORITY=-100 \
PRIORITY_PREFETCH_HEAD_START_MS=50 \
PRIORITY_REPLAY_GUARD_MS=120 \
PRIORITY_REPLAY_RELEASE_MS=80 \
bash scripts/run_milestone27_real_prompt_controlled_replay.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Build the real prompt-pair workload from an AgentBench trace index:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone27_real_prompt_controlled_replay_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
MAX_PAIRS=12 \
MODES="no_prefetch dynamo_priority_hints" \
TOOL_WAIT_LIST_MS="100 250 500 1000" \
FILLER_LIST="16 64" \
PRIORITY_DIRECT_PREFETCH=0 \
DYNAMO_HIGH_PRIORITY=100 \
DYNAMO_NORMAL_PRIORITY=0 \
DYNAMO_LOW_PRIORITY=-100 \
PRIORITY_PREFETCH_HEAD_START_MS=50 \
PRIORITY_REPLAY_GUARD_MS=120 \
PRIORITY_REPLAY_RELEASE_MS=80 \
bash scripts/run_milestone27_real_prompt_controlled_replay.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Run with synthetic first/replay prompt pairs:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

WORKLOAD_SOURCE=synthetic \
RESULT_ROOT=artifacts/results/milestone27_synthetic_large_prefix_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
MAX_PAIRS=1 \
SYNTHETIC_PROMPT_TOKENS=4096 \
SYNTHETIC_REPLAY_SUFFIX_TOKENS=256 \
MODES="no_prefetch dynamo_priority_hints" \
TOOL_WAIT_LIST_MS="100" \
FILLER_LIST="0 64 128" \
FILLER_PROMPT_TOKENS=2048 \
FILLER_DIVERGE_EARLY=1 \
PRIORITY_DIRECT_PREFETCH=0 \
DYNAMO_HIGH_PRIORITY=100 \
DYNAMO_NORMAL_PRIORITY=0 \
DYNAMO_LOW_PRIORITY=-100 \
REQUEST_CONCURRENCY=8 \
MAX_TOTAL_TOKENS=24576 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.70 \
bash scripts/run_milestone27_real_prompt_controlled_replay.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Simple meaning:

```text
The first request and replay request contain the same large prefix.
The replay adds only a small synthetic tool-result suffix.

This lets us test:
  low/no pressure    -> does KV stay resident in GPU?
  medium pressure    -> does SGLang reload KV from host?
  high pressure      -> does SGLang recompute/prefill instead?
```

Important events to observe:

```text
m27.session.start
  A controlled replay gap starts.

m27.request.start / m27.request.end
  Turn A, prefetch hint, filler pressure, or Turn B was sent to SGLang.

m27.request.submitted
  For dynamo_priority_hints, this includes the Dynamo-style hint source,
  the translated SGLang priority value, and replay deadline metadata.

m27.tool_wait.start
  The controlled tool-wait window begins.

m27.hint.submitted
  The runtime knows this session may resume soon.

m27.prefetch.start / m27.prefetch.end
  The direct KV hint request actually runs.

kv_telemetry.copy.*
  Lightweight host-to-device copy telemetry saw KV movement.

hiradix.* / hicache.* / hostpool.*
  SGLang entered prefix-cache or HiCache load-back paths.
```

Key output files:

```text
artifacts/results/<run>/<case>/m27_trace.jsonl
artifacts/results/<run>/<case>/m27_copy_telemetry.jsonl
artifacts/results/<run>/<case>/m27_metrics.jsonl
artifacts/results/<run>/controlled_replay_report/controlled_replay_report.html
artifacts/results/<run>/controlled_replay_report/controlled_replay_gaps.csv
artifacts/results/<run>/controlled_replay_report/dynamo_priority_hint_translation.csv
artifacts/results/latest_controlled_replay_report.html
artifacts/results/latest_master_report.html
```

Dynamo priority hint translation proof:

```text
The report appendix includes "Dynamo Priority Hint Translation Rows".
This table shows, per gap:
  custom_params.nvext.agent_hints priority
  translated SGLang priority integer
  whether the hint/replay used high priority
  whether background filler traffic used low priority

This keeps the claim precise:
we are not running full Dynamo here; we are emulating the Dynamo hint contract
and mapping it onto SGLang's native priority field.
```

Investigate whether replay reused GPU KV, loaded KV from host, or recomputed:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/investigate_hicache_reuse.py \
  --run-root artifacts/results/runs/controlled/synthetic_large_prefix_probe_1 \
  --gaps-csv artifacts/results/reports/synthetic_large_prefix_probe_1/controlled_replay_gaps.csv \
  --out-json artifacts/results/reports/synthetic_large_prefix_probe_1/hicache_reuse_investigation.json \
  --out-md artifacts/results/reports/synthetic_large_prefix_probe_1/hicache_reuse_investigation.md
```

Attach a real live AgentBench direct-prefetch run to the same master report:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

# First run the live direct-prefetch experiment.
RESULT_ROOT=artifacts/results/milestone26_live_direct_kv_load_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
AGENTBENCH_ROOT=~/kv_cache_offloading \
START_INDEX=0 \
END_INDEX=15 \
AGENTBENCH_EXECUTION_LOOP_MAX_STEPS=10 \
SERVER_MODE=hicache \
HICACHE_SIZE_GB=8 \
LIVE_PREFETCH_ACTION=direct_load \
bash scripts/run_milestone26_live_direct_kv_load_intervention.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct

# Then rebuild the latest master report with:
#   1. the latest controlled replay run
#   2. the latest live direct-prefetch run
bash scripts/build_latest_master_with_live_direct.sh
```

Simple meaning:

```text
Controlled replay section:
  compares no_prefetch against direct_prefetch under controlled timing and pressure.

Live AgentBench section:
  shows one real DeepAgents/SWE-bench direct-prefetch run.
  This section is not trying to compare modes.
  It shows whether live tool gaps produce direct KV hints, late hints, and real KV HtoD evidence.

AGENTBENCH_DIRECT_SGLANG_MAX_TOKENS:
  optional safety cap for live report refreshes.
  It prevents one real agent turn from generating for many minutes.
  Leave it unset when you want the full natural model budget.
```

Preferred master-report workflow:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

EXPERIMENT_KIND=both \
REPORT_LABEL=manager_demo_1 \
UPDATE_LATEST=1 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Simple interpretation:

```text
If green hint-side HtoD appears before replay:
  the direct KV hint moved KV early enough.

If cyan replay-side HtoD appears before green:
  replay got there first and had to load KV itself.

If direct_prefetch is still late:
  the software hint existed, but actual KV movement did not happen early enough.
  This strengthens the case for deadline-aware, hint-aware KV movement support.
```

### Milestone 28: Hardened Master Report Workflow

Status: ready.

Why this milestone is needed:

```text
The project now has many useful experiment scripts and many generated files.
For manager-facing work, we need one clean entrypoint and one clean report.

Milestone 28 makes the workflow simple:
  run one script
  generate the full master report
  keep labeled evidence archived
  keep artifacts/results easy to scan
```

What it does:

```text
scripts/run_master_report.sh

This single script can run:
  controlled   controlled replay experiment only
  live         real AgentBench / DeepAgents live direct-prefetch experiment only
  both         controlled replay plus live direct-prefetch evidence
```

Run the full master report:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

EXPERIMENT_KIND=both \
REPORT_LABEL=manager_demo_1 \
PRESSURE_PROFILE=high \
UPDATE_LATEST=1 \
START_INDEX=0 \
END_INDEX=15 \
MAX_STEPS=10 \
MAX_TIMELINE_GAPS=32 \
AGENTBENCH_ROOT=~/kv_cache_offloading \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
MAX_PAIRS=8 \
MODES="no_prefetch dynamo_priority_hints" \
TOOL_WAIT_LIST_MS="100 250 500 1000" \
FILLER_LIST="16 32" \
REQUEST_CONCURRENCY=4 \
MAX_TOTAL_TOKENS=8192 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.72 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Build only from existing run folders:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

BUILD_ONLY=1 \
EXPERIMENT_KIND=both \
REPORT_LABEL=manager_demo_1_rebuild \
UPDATE_LATEST=0 \
CONTROLLED_ROOT=artifacts/results/runs/controlled/manager_demo_1 \
LIVE_DIRECT_ROOT=artifacts/results/runs/live/manager_demo_1 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Preview without running:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

DRY_RUN=1 \
EXPERIMENT_KIND=both \
REPORT_LABEL=manager_demo_1 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Important knobs:

```text
EXPERIMENT_KIND
  controlled, live, or both.

PRESSURE_PROFILE
  custom, low, medium, high, or extreme.
  This sets pressure defaults unless you override them manually.

REPORT_LABEL
  folder name for this run under artifacts/results/reports/.

UPDATE_LATEST
  1 replaces artifacts/results/latest_master_report.html.
  0 keeps the labeled report only.

BUILD_ONLY
  1 rebuilds HTML from existing run folders.

DRY_RUN
  1 prints what would run without launching SGLang.

CLEAN_TOPLEVEL
  1 archives loose top-level result files.

MAX_TIMELINE_GAPS
  number of rows shown in each timeline.

START_INDEX / END_INDEX
  AgentBench task range for live runs.

WORKLOAD_JSONL / TRACE_INDEX_CSV
  real prompt-pair source for controlled replay.

WORKLOAD_SOURCE
  real, synthetic, or fallback.
  synthetic creates controlled first/replay prompt pairs with a known shared prefix.

SYNTHETIC_PROMPT_TOKENS
  target size of the shared synthetic first-turn prompt.

SYNTHETIC_REPLAY_SUFFIX_TOKENS
  target size of the small replay-only suffix.
```

Pressure profiles:

| Profile | Controlled Replay Defaults | Live AgentBench Defaults | Use Case |
| --- | --- | --- | --- |
| `low` | `MAX_PAIRS=4`, `FILLER_LIST="8 16"`, `REQUEST_CONCURRENCY=2` | `START_INDEX=0`, `END_INDEX=3`, `MAX_STEPS=6` | Fast smoke run. |
| `medium` | `MAX_PAIRS=8`, `FILLER_LIST="16 32"`, `REQUEST_CONCURRENCY=4` | `START_INDEX=0`, `END_INDEX=15`, `MAX_STEPS=10` | Default manager-report run. |
| `high` | `MAX_PAIRS=16`, `FILLER_LIST="32 64 128"`, `REQUEST_CONCURRENCY=8` | `START_INDEX=0`, `END_INDEX=31`, `MAX_STEPS=10` | Stronger cache/request pressure. |
| `extreme` | `MAX_PAIRS=24`, `FILLER_LIST="64 128 192"`, `REQUEST_CONCURRENCY=12` | `START_INDEX=0`, `END_INDEX=63`, `MAX_STEPS=15` | Stress run; use only after high is stable. |
| `custom` | Only uses knobs you explicitly set. | Only uses knobs you explicitly set. | Manual tuning. |

Pressure knobs that matter most:

```text
FILLER_LIST
  Adds unrelated sessions that compete for KV space.

REQUEST_CONCURRENCY
  Sends more requests at once, increasing scheduling and memory pressure.

MAX_PAIRS
  More controlled replay prompt pairs.

START_INDEX / END_INDEX
  More live AgentBench tasks.

MAX_TOTAL_TOKENS
  Smaller values make the GPU-side KV pool tighter.

MEM_FRACTION_STATIC
  Higher values reserve more GPU memory for static/model memory,
  leaving less room for KV cache.

HICACHE_SIZE_GB
  Host-side HiCache capacity. Keep it large enough to hold useful offloaded KV.
```

### Milestone 29: Replay Path Instrumentation Ledger

Status: ready.

Why this milestone is needed:

```text
The timeline shows when requests, prefetch attempts, and KV movements happen.
But for each row, we also need a plain answer:
  did replay reuse KV?
  did replay load KV from host to GPU?
  did replay recompute/prefill missing tokens?
  did replay mostly wait in the scheduler/request path?

Milestone 29 turns each timeline row into an evidence-backed replay-path row.
```

What it does:

```text
For every controlled replay gap, the report now builds a replay-path ledger.

Each row includes:
  final_path
  bottleneck_label
  confidence
  prefetch_outcome
  input_tokens
  matched_prefix_tokens
  unmatched_tokens
  host_load_tokens
  recomputed_tokens_est
  scheduler_wait_ms
  kv_prepare_ms
  hardware counterfactual fields
```

Simple meaning:

```text
Instead of only saying:
  G04 had a long TTFT window.

The report can now say:
  G04 likely reused logical/GPU-resident KV, but waited in the scheduler path.

Or:
  G07 replay loaded KV from host to GPU.

Or:
  G12 had a prefix miss and likely recomputed/prefilled missing tokens.
```

Timeline outcome view:

```text
The mixed timeline now shows the replay path more directly:

cyan
  replay loaded KV from host to GPU

magenta
  replay recomputed/rebuilt missing prefix/KV work

gold
  remaining before-first-token work, such as queueing or normal prefill

red
  decode/generation after first token

Each timeline row also carries a compact verdict such as:
  PREFETCH HIT
  REPLAY HOST LOAD
  RECOMPUTE
  MIXED LOAD+RECOMPUTE
  FULL REUSE
  WASTED PREFETCH
  LATE PREFETCH
```

This means the chart itself can now answer:

```text
When the agent resumed, did it reuse KV, reload KV from host, or rebuild KV?
```

Readable phase timeline:

```text
The report also includes a Readable Phase Timeline.

This chart removes the global time axis.
Each row is still one tool gap, but each phase gets a fixed readable column:
  initial turn
  tool wait
  prefetch
  replay path

Every bar prints the true measured duration.
This makes long prefetch/replay requests easy to explain without compressing
small TTFT, HtoD, or recompute bars into the far right of the chart.
```

New report sections:

```text
Replay Path Proof Table
  one evidence-backed row per timeline gap

Bottleneck Breakdown
  groups rows by scheduler dominated, host-load dominated, recompute dominated, etc.

Confidence Summary
  high, medium, or low confidence for each classification

Counterfactual Hardware Opportunity
  estimates whether a deadline-aware hardware path might plausibly have met the tool-gap deadline

Instrumentation Coverage
  shows which evidence sources were present
```

New output files:

```text
artifacts/results/<report_label>/replay_path_ledger.csv
artifacts/results/<report_label>/hardware_counterfactual.csv
artifacts/results/<report_label>/instrumentation_coverage.csv
artifacts/results/<report_label>/request_id_coverage_report.csv

artifacts/results/<report_label>/report/replay_path_ledger.csv
artifacts/results/<report_label>/report/hardware_counterfactual.csv
artifacts/results/<report_label>/report/instrumentation_coverage.csv
artifacts/results/<report_label>/report/request_id_coverage_report.csv
```

Validate the classifier without running SGLang:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/validate_replay_path_classifier.py
```

Expected output:

```text
Validated 5 replay-path classifier cases.
Wrote artifacts/results/replay_path_classifier_validation.json
```

Rebuild the latest master report from an existing run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

BUILD_ONLY=1 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL=replay_path_ledger_rebuild \
UPDATE_LATEST=1 \
CONTROLLED_ROOT=artifacts/results/runs/controlled/strong_replay_attribution_1 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Run a fresh deep-instrumented controlled replay:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=milestone29_deep_run_1 \
PRESSURE_PROFILE=medium \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
bash scripts/run_milestone29_deep_replay_path.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Optional deep scheduler trace:

```bash
export AGENTIC_KV_TRACE_SCHEDULER=1
```

Simple meaning:

```text
This asks the SGLang monkeypatch tracer to also log selected scheduler methods.
Use it only for focused debug runs because scheduler events can be noisy.
```

Important events to observe:

```text
kv_telemetry.scheduler.end
  Scheduler/request-path events such as request received, queue insertion,
  selected for prefill, and batch run.

kv_telemetry.prefill.end
  Model-forward/prefill events, when the worker exposes a request/session
  mapping for the batch.

kv_telemetry.cache.end
  Prefix/radix/HiCache evidence: input tokens, cached prefix tokens,
  host-hit tokens, host-load tokens, and estimated new prefill tokens.

kv_telemetry.copy.*
  Host-to-device or device-to-host KV movement evidence.

m27.pre_replay.checkpoint
  A replay-deadline marker. Today it records the known deadline/session state;
  deeper block-level residency can attach to this checkpoint later.
```

Important interpretation:

```text
High confidence:
  direct SGLang counters plus HtoD movement evidence.

Medium confidence:
  direct SGLang prefix/cache counters, but no matching low-level HtoD event.

Low confidence:
  mostly inferred from TTFT and timeline shape.

The report should be read as an evidence ladder.
Strong rows support strong claims.
Low-confidence rows tell us where deeper SGLang block-level hooks are still needed.
```

### Milestone 29B: Forced Eviction Sanity Probe

Status: ready.

Why this milestone is needed:

```text
In earlier no-prefetch pressure runs, replay TTFT was high, but we saw very little
host-to-device KV movement.

That can mean one of several things:
  the target KV was still resident in GPU memory
  SGLang reused cached prefix blocks without needing a host load
  SGLang recomputed/prefilled missing tokens instead of loading from host
  our instrumentation missed the movement

This milestone creates a smaller but harsher sanity test so those cases are easier
to separate.
```

What it does:

```text
Runs one controlled replay pair.
Uses no_prefetch only.
Uses one short tool wait: 100 ms.
Pads the target first/replay prompts with a large shared prefix.
Creates many large filler requests that diverge early from the target prefix.
Shrinks the GPU KV budget so cache pressure is more likely.
Keeps HiCache enabled so host-side KV has somewhere to live.
```

Recommended run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

REPORT_LABEL=forced_eviction_sanity_1 \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
bash scripts/run_forced_eviction_sanity.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Equivalent master-runner form:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL=forced_eviction_sanity_1 \
PRESSURE_PROFILE=eviction_sanity \
UPDATE_LATEST=1 \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Default knobs used by `PRESSURE_PROFILE=eviction_sanity`:

```text
MAX_PAIRS=1
MODES=no_prefetch
TOOL_WAIT_LIST_MS=100
FILLER_LIST=256
REQUEST_CONCURRENCY=2
TARGET_PROMPT_TOKENS=6144
FILLER_PROMPT_TOKENS=4096
FILLER_DIVERGE_EARLY=1
MAX_TOTAL_TOKENS=8192
HICACHE_SIZE_GB=8
MEM_FRACTION_STATIC=0.70
```

Why the target prompt padding matters:

```text
The earlier real prompts had useful cached prefixes, but many were only a few
thousand tokens. That may not be large enough to force SGLang to prefer host-KV
loading over recompute or normal cache reuse.

This run adds a large shared prefix to the target first turn and replay turn.
That gives SGLang a bigger KV object to either keep in GPU memory, offload/load
through HiCache, or recompute.
```

Why the filler padding matters:

```text
The filler requests begin with unique text.
That makes them diverge early from the target prompt.

Simple meaning:
  target prompt creates useful KV for Agent A
  fillers create unrelated KV for other sessions
  those fillers should pressure the GPU KV pool instead of sharing Agent A's prefix
```

How to interpret the result:

| Observation | Meaning |
| --- | --- |
| Replay-side HtoD/cyan bar appears | The replay really loaded KV from host to GPU. The instrumentation saw host/device movement. |
| No HtoD, but prefix-cache counters show high GPU/cache hits | Pressure was still not enough to evict the target KV, or SGLang retained useful prefix blocks. |
| No HtoD, but replay TTFT/prefill counters are high | SGLang may have recomputed/prefilled instead of loading host KV. |
| No HtoD, no useful cache counters, and no recompute evidence | Treat as an instrumentation gap and inspect SGLang internals more deeply. |

Important events to observe:

```text
m27.session.start
  Confirms one target replay pair and its padded prompt size.

m27.request.start / m27.request.end
  Shows the first model turn, filler pressure requests, and replay request.

hiradix.match_prefix.*
  Shows whether SGLang found reusable prefix/cache blocks.

hicache.load.*
  Shows whether SGLang loaded host-side KV back toward the GPU path.

kv_telemetry.copy.*
  Shows observed KV copy activity when the lightweight telemetry hook sees it.

replay_path_ledger.csv
  Gives the final per-gap classification.
```

Report output must include:

```text
Summary
Experiment Setup
Global Replay H2D Readiness dot chart, for no-prefetch replay-H2D runs
Global Prefetch Margin dot chart, for direct-prefetch runs
How To Read Timelines
Controlled Replay Timeline, if controlled/both
Live AgentBench Direct Prefetch Timeline, if live/both
Key Observations
Mode Tables
Direct KV Evidence
Gap Details
Reproduce This Report
```

For no-prefetch-only experiments, the global chart answers:

```text
When replay was due, how late did replay-side KV host-to-device movement finish?
```

The report writes this data to:

```text
replay_h2d_readiness.csv
```

The important split is:

```text
replay due -> replay request start
  whether the replay request itself arrived on time or was delayed by the driver/load pattern

replay due -> H2D start
  how long the system waited before starting visible KV movement

replay request start -> H2D start
  after the replay request entered SGLang, how long it took before visible KV movement began

H2D start -> H2D end
  the visible host-to-device KV movement window

replay due -> H2D end
  total KV-readiness lateness
```

In the dot chart, values below zero mean the replay deadline arrived before KV
movement finished. Farther below zero means the replay-side H2D load was later.

The report also includes a companion chart:

```text
Replay Request vs H2D Start
```

This chart shows three markers per no-prefetch gap:

```text
replay request start
H2D start
H2D finish
```

This makes it clear whether the replay request itself arrived late, or whether
the request arrived and then waited before SGLang reached the host-to-device KV
load path.

Important output files:

```text
artifacts/results/latest_master_report.html
artifacts/results/latest_synthetic_master_report.html
artifacts/results/latest_manifest.json
artifacts/results/reports/<REPORT_LABEL>/master_report.html
artifacts/results/reports/<REPORT_LABEL>/manifest.json
artifacts/results/runs/controlled/<REPORT_LABEL>/
artifacts/results/runs/live/<REPORT_LABEL>/
```

Top-level cleanup:

```text
With CLEAN_TOPLEVEL=1, loose top-level result files are moved to:
  artifacts/results/archive/toplevel_cleanup_<timestamp>/

The top level should only show:
  latest_master_report.html
  latest_synthetic_master_report.html
  latest_manifest.json

plus result subfolders such as:
  reports/
  runs/
  latest_real/
  latest_synthetic/
  archive/
```

Simple interpretation:

```text
This milestone does not change the core experiment.
It makes the workflow reproducible and clean.

The timeline charts and global prefetch-margin dot charts remain the key
manager-facing evidence.
```

### Milestone 30: Stable KV Block Ledger

Status: ready.

Full proposal:

```text
KV_BLOCK_LEDGER.md
```

Why this milestone is needed:

```text
The lifecycle table can tell us that a session wrote KV to host, evicted KV from
GPU, evicted KV from host, and recomputed later.

The stronger version is:
  which logical KV blocks did this happen to?
  how many blocks were lost?
  how many tokens did those blocks represent?
  did any blocks load back?
```

What it does:

```text
Adds a modular Stable KV Block Ledger:

src/agentic_kv/block_ledger/events.py
  stable normalized KV event schema

src/agentic_kv/block_ledger/normalizer.py
  SGLang-version-specific trace event -> stable KV event

src/agentic_kv/block_ledger/block_id.py
  stable logical block identity and nearby-range matching

src/agentic_kv/block_ledger/ledger.py
  KV block lifecycle state machine

src/agentic_kv/block_ledger/report.py
  CSV/JSON/report summary helpers
```

Modularity rule:

```text
Only normalizer.py should be SGLang-version-sensitive.
The rest of the ledger consumes stable events, so future SGLang versions should
mostly require normalizer updates instead of report rewrites.
```

New output files:

```text
artifacts/results/reports/<report_label>/kv_block_ledger.csv
artifacts/results/reports/<report_label>/kv_block_ledger.json
artifacts/results/reports/<report_label>/kv_block_lifecycle_summary.csv
artifacts/results/reports/<report_label>/kv_block_gap_summary.csv
```

New master report section:

```text
KV Block Ledger
  Block Ledger Summary
  Per-Gap Block Summary
  Per-Block Ledger Rows
```

Validate the ledger without running SGLang:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/validate_kv_block_ledger.py
```

Expected output:

```text
KV block ledger validation passed.
```

Rebuild the latest report with block ledger outputs:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/build_milestone27_controlled_replay_report.py \
  --root artifacts/results/runs/controlled/quick_hicache_lifecycle_probe \
  --out-dir artifacts/results/reports/quick_hicache_lifecycle_probe \
  --latest-root artifacts/results \
  --max-timeline-gaps 16
```

Important interpretation:

```text
This is logical block tracking, not a physical GPU page snooper.

The ledger uses:
  node_id when SGLang exposes it
  token range when node_id is missing
  nearby-range matching when SGLang reports shifted ranges for the same block

That gives us a strong per-block lifecycle ledger while keeping the add-on
infrastructure reusable across future SGLang versions.
```

### Milestone 31: Exact KV Movement Attribution

Status: ready.

Full proposal:

```text
KV_EXACT_MOVEMENT_ATTRIBUTION.md
```

Why this milestone is needed:

```text
The existing KV lifecycle table is useful, but it is mostly gap-level evidence.

It can say:
  G04 wrote KV to host and later loaded KV from host.

Milestone 31 tries to say:
  G04 host indices 1812..3859 moved into GPU/device indices 4200..6247.
  The movement started at time A and ended at time B.
  It happened during replay, not during hint prefetch.
```

What it adds:

```text
host/device index signatures
host/device index ranges and counts
request_id, node_id, and layer_id where SGLang exposes them
copy_start_ms and copy_end_ms for SGLang movement functions
hint-loaded vs replay-loaded block counts
richer NVTX labels for profiler/Nsight validation
```

Why this is better:

```text
This follows the same logical KV block/index set across write, eviction, and
load-back events. That makes the evidence closer to actual memory movement.
```

New master report section:

```text
Exact KV Movement Attribution
  How To Read This Section
  Exact Movement Summary
  Exact Movement Rows For Timeline Sample
```

New output files:

```text
artifacts/results/reports/<report_label>/exact_kv_movement_attribution.csv
artifacts/results/reports/<report_label>/exact_kv_movement_summary.csv
```

Run it through the normal master report script:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL=exact_kv_attribution_demo \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
MAX_TIMELINE_GAPS=32 \
MAX_PAIRS=8 \
MODES="no_prefetch" \
TOOL_WAIT_LIST_MS="100" \
FILLER_LIST="32 64 128" \
REQUEST_CONCURRENCY=8 \
FILLER_PROMPT_TOKENS=1536 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.75 \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Important interpretation:

```text
This is still software-visible SGLang evidence, not a physical DMA bus snooper.

The strongest rows are rows with both:
  host_index_signature
  device_index_signature

Those rows can connect the host-side KV block set to the GPU-side destination
indices, and can be validated with torch.profiler/Nsight on smaller runs.
```

### Milestone 32: KV H2D Bandwidth Pressure

Status: ready.

Full note:

```text
KV_H2D_BANDWIDTH_PRESSURE.md
```

Why this milestone is needed:

```text
The lifecycle timeline can say what happened to one replay gap.

The bandwidth-pressure view adds surrounding context:
  how many KV H2D copies were happening near the replay deadline?
  how many blocks/tokens were being moved?
  was this replay late while the H2D path was already busy?
```

What it adds to `latest_master_report.html`:

```text
KV H2D Bandwidth Pressure
  H2D Activity By Time Window
  Per-Gap Deadline-To-Ready H2D Pressure
  Per-Gap H2D Contention Timeline
  Per-Gap Contention Verdicts
  Per-Gap Contention Event Rows
  Aligned H2D Event Samples

Readable KV Lifecycle Timeline
  nearby H2D pressure strip per row
```

New output files:

```text
artifacts/results/reports/<report_label>/report/h2d_activity_events.csv
artifacts/results/reports/<report_label>/report/h2d_pressure_by_gap.csv
artifacts/results/reports/<report_label>/report/h2d_activity_windows.csv
artifacts/results/reports/<report_label>/report/h2d_contention_by_gap.csv
artifacts/results/reports/<report_label>/report/h2d_contention_events.csv
```

Simple interpretation:

```text
If G00 is late and nearby H2D pressure is high, the replay did not miss its
deadline in isolation. It missed while the exact SGLang-visible H2D movement
path was already busy with KV movement.

If G00 is late but the contention timeline says the H2D path was quiet before
G00's own copy began, the delay likely happened before SGLang reached the
actual host-to-device copy path. That points to request scheduling / cache-path
latency, not raw copy bandwidth alone.
```

### Milestone 33: Replay Delay Breakdown

Status: ready.

Full note:

```text
REPLAY_DELAY_BREAKDOWN.md
```

Why this milestone is needed:

```text
The H2D pressure view can show that a replay-side KV copy started late.

The replay delay breakdown explains why it started late:
  was the replay submitted late?
  did it wait in the client/workload driver?
  did it wait inside SGLang's scheduler?
  did it reach cache/load-back late?
  was the actual H2D copy slow?
```

What it adds to `latest_master_report.html`:

```text
Replay Delay Breakdown
  Delay Waterfall Timeline
  Main Verdicts
  Stage Duration Table
  What Was Running Instead
  Evidence Confidence
```

New output files:

```text
artifacts/results/reports/<report_label>/report/replay_delay_breakdown.csv
artifacts/results/reports/<report_label>/report/replay_delay_verdicts.csv
artifacts/results/reports/<report_label>/report/replay_delay_running_context.csv
```

Simple interpretation:

```text
If G04 says "copy issued late, copy was fast", the real problem was not that
the H2D copy took 75 seconds. The problem was that the copy was not issued
until much later than the replay deadline.

If G04 says "copy blocked behind other H2D", then visible H2D work from other
rows was already happening before G04's own H2D started.
```

### Milestone 34: Replay Delay Deep Instrumentation

Full note:

```text
REPLAY_DELAY_DEEP_INSTRUMENTATION.md
```

Why this milestone is needed:

```text
Milestone 33 explains that replay-side KV H2D started late.

Milestone 34 goes deeper and instruments SGLang itself so we can see the exact
request-stage path before H2D begins:
  client/request arrival
  SGLang receive
  scheduler queue/admit
  cache lookup/load-back
  H2D copy
  model forward/recompute
```

What it adds:

```text
The SGLang trace patch now emits:

kv_telemetry.request_stage

These rows come from direct SGLang method hooks, not server-log parsing.
```

New output files:

```text
artifacts/results/reports/<report_label>/report/replay_delay_stage_trace.csv
artifacts/results/reports/<report_label>/report/replay_delay_h2d_activity.csv
artifacts/results/reports/<report_label>/report/replay_delay_gap_verdicts.csv
```

What to look for in `latest_master_report.html`:

```text
Replay Delay Breakdown
  Exact SGLang Request Stage Trace
  H2D Activity During The Delay Window
  Stage Duration Table
  What Was Running Instead
```

Simple interpretation:

```text
If H2D begins long after replay was due, this milestone helps show whether the
request waited in the client driver, waited in SGLang's scheduler, reached
cache/load-back late, or was blocked behind other visible H2D movement.
```

Important:

```text
Reports built from old traces will not contain the new request-stage rows.
Rerun the experiment after this milestone to populate the exact SGLang stage
trace.
```

### Milestone 35: Instrumentation Evidence Audit

Full note:

```text
INSTRUMENTATION_AUDIT.md
```

Why this milestone is needed:

```text
The report now has many charts:
  readable KV lifecycle timelines
  H2D readiness dots
  replay queue timelines
  client-dispatch KV movement timelines
  replay delay waterfalls
  detailed KV lifecycle tables

Milestone 35 audits the evidence behind each chart so we do not overclaim.
```

What it adds:

```text
Evidence levels:
  DIRECT          directly emitted by a driver/SGLang hook
  DERIVED         computed from direct events
  INFERRED        likely, but not directly observed as a physical event
  NOT_YET_PROVEN  outside the current evidence boundary

Hardening added after the first audit:
  one shared movement vocabulary for H2D/D2H/GPU-evict/host-evict
  request_id/correlation_id/case_id/gap_id propagation from the workload driver
  request/correlation fields in normalized KV events and block ledger records
  corrected dispatch-window movement-kind audit counts
  evidence_level and exact_correlation_source columns for exact movement rows
```

New master report section:

```text
Instrumentation Evidence Audit
```

New output files:

```text
artifacts/results/reports/<report_label>/report/instrumentation_evidence_audit_summary.csv
artifacts/results/reports/<report_label>/report/instrumentation_evidence_audit_matrix.csv
artifacts/results/reports/<report_label>/report/instrumentation_chart_inventory.csv
artifacts/results/reports/<report_label>/report/instrumentation_artifact_inventory.csv
artifacts/results/reports/<report_label>/report/instrumentation_evidence_audit.md
```

Run the standalone audit:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/audit_master_report_evidence.py \
  --report artifacts/results/reports/deep_delay_trace_h2d_sweet_spot_1/report
```

Important events/claims to observe:

```text
H2D bars should map to exact SGLang-visible host-to-device KV movement rows.
D2H/write-host rows should map to SGLang/HiCache host-write events.
GPU and host eviction rows should map to direct residency/eviction hooks.
Request delay rows should map to request-stage hooks where available.
Recompute bars should be labeled as inferred until a direct recompute hook exists.
Physical DMA saturation should be labeled not-yet-proven without CUPTI/Nsight/counter evidence.
```

Simple interpretation:

```text
This milestone is the report's honesty layer.
It tells us which visuals are backed by direct instrumentation and which are
still derived, inferred, or outside the current proof boundary.

Reports rebuilt from old traces can fix the movement-kind accounting, but
fresh experiments are needed to populate the newly added request/correlation
fields in raw SGLang trace events.
```

### Milestone 36: Multi-Session Agentic Replay Forensics

Why this milestone is needed:

```text
Earlier controlled experiments used one target request, then fillers, then
one replay. That is useful for isolating behavior, but real serving traffic
has many agent sessions overlapping.

Milestone 36 runs many agent-like sessions in the same SGLang server window.
Each session has:
  initial model turn
  tool-wait gap
  optional direct KV prefetch
  replay/resume request
```

What it tests:

```text
Can the direct KV prefetch path still finish before replay when many sessions
are active at once?

Do replay-side H2D loads arrive early or late across many sessions?

During a target session's delay window, were other sessions moving KV,
offloading KV, or evicting KV?
```

Important design choice:

```text
This milestone avoids old prompt-based request warming.

Main modes:
  no_prefetch
  dynamo_priority_hints
  projected_hardware_bypass

dynamo_priority_hints sends Dynamo-style priority metadata and an SGLang
priority value. It does not issue our artificial direct KV prefetch hook.

projected_hardware_bypass is not a measured SGLang mode. The report computes it
from measured H2D duration and asks: if a low-overhead hardware KV movement path
could start earlier during tool wait, would KV have been ready before replay?
```

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
AGENTIC_KV_TRACE_KV_POOL=1 \
AGENTIC_KV_GPU_UTIL_SAMPLER=1 \
GPU_UTIL_SAMPLE_INTERVAL_MS=100 \
EXPERIMENT_KIND=multi_session \
REPORT_LABEL=multi_session_agentic_replay_1 \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
WORKLOAD_SOURCE=real \
SESSION_COUNT=16 \
MODES="no_prefetch dynamo_priority_hints" \
ARRIVAL_SHAPE=staggered \
ARRIVAL_GAP_MS=120 \
TOOL_WAIT_LIST_MS="100 250 500 1000" \
TOOL_WAIT_JITTER_MS=50 \
PREFETCH_TIMING=early \
HINT_DELAY_MS=20 \
BACKGROUND_FILLERS_PER_SESSION=0 \
REQUEST_CONCURRENCY=8 \
TARGET_PROMPT_TOKENS=4096 \
MAX_TOTAL_TOKENS=16384 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.72 \
MAX_TIMELINE_GAPS=32 \
TRACE_INDEX_CSV=~/kv_cache_offloading/experiments/reports/latest_prompt_evolution_trace_index.csv \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

For a faster first check:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
AGENTIC_KV_TRACE_KV_POOL=1 \
AGENTIC_KV_GPU_UTIL_SAMPLER=1 \
GPU_UTIL_SAMPLE_INTERVAL_MS=100 \
EXPERIMENT_KIND=multi_session \
REPORT_LABEL=multi_session_smoke_1 \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
WORKLOAD_SOURCE=synthetic \
SESSION_COUNT=4 \
MODES="no_prefetch dynamo_priority_hints" \
ARRIVAL_SHAPE=burst \
ARRIVAL_GAP_MS=40 \
BURST_SIZE=4 \
TOOL_WAIT_LIST_MS="100 250" \
REQUEST_CONCURRENCY=4 \
TARGET_PROMPT_TOKENS=2048 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.72 \
MAX_TIMELINE_GAPS=8 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

What to look for in `latest_master_report.html`:

```text
Unified Forensic Stack Timeline:
  rows should include many sessions, not just one target replay case

Global Replay H2D Readiness:
  shows whether replay-side H2D movements finished late or early

Client Dispatch KV Movement:
  shows other sessions' H2D/D2H/evict activity during each target delay window

Detailed KV Block Lifecycle Table:
  shows the block-level evidence behind the visible rows
```

### Milestone 37: GPU KV Pool Residency Telemetry

Why this milestone is needed:

```text
When replay-side H2D starts late, we need to know whether the SGLang GPU KV
pool was already full or nearly full around that moment.

This should come from SGLang's KV memory-pool state, not from total GPU memory
reported by nvidia-smi.
```

What it adds:

```text
AGENTIC_KV_TRACE_KV_POOL=1 enables direct SGLang KV-pool sampling inside the
trace hooks.

The report now emits:
  kv_pool_samples.csv
  kv_pool_residency_by_gap.csv

The master report now includes:
  GPU KV Pool Residency
  per-row GPU KV pool summary inside the Unified Forensic Stack Timeline
```

Simple meaning:

```text
If the pool is very full around replay/H2D time, late KV movement may be
related to residency pressure: there may not be enough room to bring KV back
without evicting something else.

If the pool is not full, the lateness is more likely coming from another part
of the path, such as client dispatch, scheduler admission, cache lookup, or
other runtime work.
```

Important proof boundary:

```text
This is direct SGLang KV-pool telemetry. It is stronger than reading logs or
total GPU memory, but it is still not a physical DMA-engine saturation counter.
It tells us how SGLang's KV cache pool looked when replay/prefetch/H2D events
were happening.
```

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
AGENTIC_KV_TRACE_KV_POOL=1 \
EXPERIMENT_KIND=multi_session \
REPORT_LABEL=gpu_kv_pool_residency_1 \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
WORKLOAD_SOURCE=synthetic \
SESSION_COUNT=8 \
MODES="no_prefetch dynamo_priority_hints" \
ARRIVAL_SHAPE=burst \
ARRIVAL_GAP_MS=40 \
BURST_SIZE=4 \
TOOL_WAIT_LIST_MS="250 500" \
PREFETCH_TIMING=early \
HINT_DELAY_MS=20 \
BACKGROUND_FILLERS_PER_SESSION=2 \
REQUEST_CONCURRENCY=4 \
TARGET_PROMPT_TOKENS=4096 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.72 \
MAX_TIMELINE_GAPS=16 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

### Milestone 38: Dynamo Priority Hints And Projected Hardware

Why this milestone is needed:

```text
We want the main manager-facing comparison to stay simple:

  1. What happens with no prefetch?
  2. What happens when software sends Dynamo-style priority hints only?
  3. What might happen if a low-overhead hardware KV movement path could do the
     same useful copy earlier and more predictably?

This avoids mixing priority hints with our explicit direct KV prefetch hook.
The main measured software modes are now:

  no_prefetch = no hint help
  dynamo_priority_hints = priority metadata only, no direct KV hook
```

What it compares:

```text
no_prefetch:
  no hint is issued.
  replay pays whatever KV load/recompute cost exists.

dynamo_priority_hints:
  sends Dynamo-style agent priority metadata and an SGLang priority value.
  this does not issue our direct KV prefetch hook.

projected_hardware_bypass:
  not a measured SGLang request.
  the report estimates when a hardware bypass path could have completed by using
  the measured KV H2D duration plus a small fixed control overhead.
```

Simple meaning:

```text
dynamo_priority_hints says:
  "this request is high priority and replay-critical, but let today's SGLang
   priority scheduler decide how to act on that request metadata."

projected_hardware_bypass says:
  "if the hardware/runtime could move the same KV through a faster lower-overhead
   path, would the KV have been ready before replay?"
```

Timeline model:

```text
tool wait starts
-> Dynamo priority hints mode sends priority metadata but no direct KV hook
-> replay becomes due
-> replay either uses resident KV, loads KV from host, or recomputes missing KV
-> report adds a projected hardware row using measured H2D duration
```

What this does and does not prove:

```text
It proves what SGLang priority hints alone do without our direct KV hook.

It does not prove the projected hardware row as a measured hardware result.
That row is a clearly labeled what-if estimate.

The useful comparison is:
  measured no prefetch vs measured priority hints vs projected hardware bypass.
```

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone38_dynamo_hints_projection_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
WORKLOAD_SOURCE=synthetic \
MODES="no_prefetch dynamo_priority_hints" \
SESSION_COUNT=12 \
ARRIVAL_SHAPE=burst \
ARRIVAL_GAP_MS=40 \
BURST_SIZE=4 \
BURST_GAP_MS=400 \
TOOL_WAIT_LIST_MS="250 500" \
PREFETCH_TIMING=early \
HINT_DELAY_MS=10 \
BACKGROUND_FILLERS_PER_SESSION=4 \
REQUEST_CONCURRENCY=8 \
PRIORITY_PREFETCH_WINDOW_MS=750 \
PRIORITY_POST_PREFETCH_QUIET_MS=750 \
DEADLINE_RESERVE_WINDOW_MS=500 \
SYNTHETIC_PROMPT_TOKENS=4096 \
SYNTHETIC_REPLAY_SUFFIX_TOKENS=256 \
FILLER_PROMPT_TOKENS=1024 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.72 \
MAX_TIMELINE_GAPS=32 \
bash scripts/run_milestone38_dynamo_hints_projection.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Equivalent master-report workflow:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

EXPERIMENT_KIND=multi_session \
REPORT_LABEL=dynamo_hints_projection_demo_1 \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
WORKLOAD_SOURCE=synthetic \
MODES="no_prefetch dynamo_priority_hints" \
SESSION_COUNT=12 \
ARRIVAL_SHAPE=burst \
ARRIVAL_GAP_MS=40 \
BURST_SIZE=4 \
BURST_GAP_MS=400 \
TOOL_WAIT_LIST_MS="250 500" \
PREFETCH_TIMING=early \
HINT_DELAY_MS=10 \
BACKGROUND_FILLERS_PER_SESSION=4 \
REQUEST_CONCURRENCY=8 \
PRIORITY_PREFETCH_WINDOW_MS=750 \
PRIORITY_POST_PREFETCH_QUIET_MS=750 \
DEADLINE_RESERVE_WINDOW_MS=500 \
SYNTHETIC_PROMPT_TOKENS=4096 \
SYNTHETIC_REPLAY_SUFFIX_TOKENS=256 \
FILLER_PROMPT_TOKENS=1024 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.72 \
MAX_TIMELINE_GAPS=32 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Important events to observe:

```text
m27.prefetch.start / m27.prefetch.end
  the direct SGLang KV hook attempt itself.

hicache.load.start / hicache.load.end
  SGLang-visible host-to-device KV movement.

kv_pool.sample
  SGLang KV memory-pool residency around replay/prefetch events.
```

What to look for in `latest_master_report.html`:

```text
Global KV Readiness By Mode:
  compare replay-start timing and KV-ready timing across modes.

  circle = when the replay request started relative to replay due.
  square = when useful KV became ready relative to replay due.
  above zero = late, because the event happened after replay was due.
  below zero = early, because the event happened before replay was due.

  no_prefetch:
    KV-ready uses replay-side H2D finish, if replay H2D is observed.

  dynamo_priority_hints:
    no artificial/direct-prefetch H2D is credited.
    KV-ready uses replay-side H2D finish only, if replay H2D is observed.

  projected_hardware_bypass:
    KV-ready is projected, not measured.
    It estimates a low-overhead hardware path using measured H2D duration
    plus fixed control overhead.

Grouped Mode Comparison Timeline:
  compare the same task/gap scenario across:
    Cxx-NP = no prefetch
    Cxx-DH = Dynamo priority hints only, if MODES includes dynamo_priority_hints
  Projected hardware rows are intentionally hidden from this timeline for now,
  because they are estimates, not measured SGLang executions.

Unified Forensic Stack Timeline:
  detailed measured evidence for no-prefetch and Dynamo-priority-hints rows.

GPU KV Pool Residency:
  check whether the KV pool was full or near-full around replay/prefetch events.
```

### Milestone 38B: Dynamo Priority Hint Bridge

Why this milestone is needed:

```text
SGLang can accept request priority directly. Dynamo-style frontends can express
agent/session intent as hints. This milestone connects those ideas without
starting the full Dynamo stack:

  custom_params.nvext.agent_hints
    -> proxy-driver translation
    -> SGLang priority integer
    -> SGLang priority scheduler
```

What this mode means:

```text
dynamo_priority_hints:
  sends a Dynamo-style agent hint with phase, session, deadline, and priority.
  maps high-priority prefetch/replay work to SGLang priority=100.
  maps background filler pressure to SGLang priority=-100.
  launches SGLang with --enable-priority-scheduling for this mode.
```

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
AGENTIC_KV_TRACE_KV_POOL=1 \
AGENTIC_KV_GPU_UTIL_SAMPLER=1 \
GPU_UTIL_SAMPLE_INTERVAL_MS=100 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL=dynamo_priority_hints_compare_1 \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
WORKLOAD_SOURCE=synthetic \
MAX_TIMELINE_GAPS=32 \
MAX_PAIRS=2 \
MODES="no_prefetch dynamo_priority_hints projected_hardware" \
TOOL_WAIT_LIST_MS=500 \
FILLER_LIST="12 16 24" \
REQUEST_CONCURRENCY=4 \
FILLER_PROMPT_TOKENS=1024 \
SYNTHETIC_PROMPT_TOKENS=4096 \
SYNTHETIC_REPLAY_SUFFIX_TOKENS=256 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.72 \
PRIORITY_DIRECT_PREFETCH=0 \
DYNAMO_HIGH_PRIORITY=100 \
DYNAMO_NORMAL_PRIORITY=0 \
DYNAMO_LOW_PRIORITY=-100 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

GPU activity note:

```text
AGENTIC_KV_TRACE_KV_POOL=1 records direct SGLang KV-pool occupancy.
AGENTIC_KV_GPU_UTIL_SAMPLER=1 records lightweight nvidia-smi samples during
the run. The report uses those samples to show whole-GPU compute/memory
activity during tool waits.

Important: SGLang scheduler, KV pool, and KV movement rows are direct hooks.
GPU compute/memory utilization is sampled whole-GPU telemetry, not exact
per-request kernel attribution.
```

What to look for:

```text
Grouped Mode Comparison Timeline:
  Cxx-DH rows show the measured Dynamo-priority-hint path.

Global KV Readiness By Mode:
  DH dots show whether the Dynamo-priority-hint path finished useful KV
  readiness before replay.

Evidence Tables / Raw Proof:
  Dynamo Priority Hint Translation Rows show the exact hint-to-priority mapping.
```

Priority queue effectiveness audit:

```text
The hint translation table proves:
  our driver attached Dynamo-style hints
  our driver translated them into SGLang priority values

That alone does not prove:
  SGLang actually moved the request ahead inside its scheduler queue

So this milestone also records a priority queue audit:
  client submit request_id + priority
  SGLang receive request_id + priority
  scheduler queue snapshots from queue-like SGLang structures
  queue name, queue length, and queue position when visible
  queue head sample with request ids, priorities, and session/gap aliases
  how many lower-priority requests were ahead
  scheduler admission sequence/order when visible
  whether lower-priority requests were admitted before this request
  proof strength: receive-only, queue-snapshot observed, or admission-order observed

This lets us distinguish two cases:

  hint attached but not honored:
    the priority metadata reached the request, but scheduler behavior did not
    clearly move the request earlier.

  hint attached and honored:
    scheduler traces show the high-priority request being received/admitted
    ahead of lower-priority work.
```

How to read the audit:

```text
priority_seen_at_receive_only:
  SGLang received the priority metadata, but the trace did not prove queue
  movement.

priority_seen_in_scheduler_queue:
  The priority request appeared inside a captured scheduler queue snapshot.
  This is stronger than receive-only, but still may not prove admission order.

priority_honored_in_admission_order:
  The priority request appeared in the scheduler admission trace without
  lower-priority requests admitted before it.

lower_priority_admitted_before_target:
  The trace found lower-priority work admitted before the high-priority replay.
  This means the hint was not fully enforced by the software scheduler.
```

New output files:

```text
artifacts/results/<run>/dynamo_priority_hint_translation.csv
artifacts/results/<run>/dynamo_priority_queue_effectiveness.csv
```

Where to see it in the HTML report:

```text
Evidence Tables / Raw Proof
  -> Dynamo Priority Hint Translation
  -> Dynamo Priority Queue Effectiveness Audit
```

Important:

```text
Old reports cannot prove priority queue effectiveness because the scheduler
queue/admission hooks were not present when those traces were captured.
Rerun this milestone to populate the new queue-effectiveness evidence.

This audit is intentionally conservative. If it cannot see SGLang's exact queue
internals for a given version, it will say "receive only" or "queue snapshot
observed" instead of claiming that priority was fully honored.
```

### Milestone 39: Projected Hardware Bypass Benefit

Why this milestone is needed:

```text
The direct software prefetch mode is useful, but it is still expensive because
it goes through SGLang's normal software/runtime path.

This milestone asks:
  if a real hardware/runtime bypass path could start the same measured KV H2D
  copy earlier during tool wait, would it have met the replay deadline?
```

Important caveat:

```text
Projected hardware bypass is not a real hardware run.

It is a what-if estimate built from real measurements:
  real replay deadline
  real tool-wait window
  real observed KV H2D duration
  real observed replay TTFT

The projection does not assume copying is free. It assumes the same measured
copy can be started earlier through a lower-overhead deadline-aware path and
protected until replay.
```

Projection levels:

```text
best_case:
  projected time = measured KV H2D time

realistic:
  projected time = measured KV H2D time + 50 ms overhead

conservative:
  projected time = measured KV H2D time + 150 ms overhead
```

What it adds to `latest_master_report.html`:

```text
Projected Hardware Bypass Benefit:
  manager-facing chart showing observed software readiness margin versus
  best-case, realistic, and conservative hardware-bypass margins.

Evidence Tables / Raw Proof:
  projected_hardware_bypass_summary
  projected_hardware_bypass per-gap rows
```

Main rebuild command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

BUILD_ONLY=1 \
EXPERIMENT_KIND=multi_session \
REPORT_LABEL=multi_session_kv_pool_1 \
UPDATE_LATEST=1 \
MULTI_SESSION_ROOT=artifacts/results/runs/multi_session/multi_session_kv_pool_1 \
bash scripts/run_master_report.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Fresh run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone38_dynamo_hints_projection_$(date +%Y%m%d_%H%M%S) \
LATEST_REPORT_ROOT=artifacts/results \
WORKLOAD_SOURCE=synthetic \
MODES="no_prefetch dynamo_priority_hints" \
SESSION_COUNT=12 \
ARRIVAL_SHAPE=burst \
ARRIVAL_GAP_MS=40 \
BURST_SIZE=4 \
BURST_GAP_MS=400 \
TOOL_WAIT_LIST_MS="250 500" \
PREFETCH_TIMING=early \
HINT_DELAY_MS=10 \
BACKGROUND_FILLERS_PER_SESSION=4 \
REQUEST_CONCURRENCY=8 \
SYNTHETIC_PROMPT_TOKENS=4096 \
SYNTHETIC_REPLAY_SUFFIX_TOKENS=256 \
FILLER_PROMPT_TOKENS=1024 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=16 \
MEM_FRACTION_STATIC=0.72 \
MAX_TIMELINE_GAPS=32 \
bash scripts/run_milestone38_dynamo_hints_projection.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

What to look for:

```text
Projected Hardware Bypass Benefit:
  if best/realistic/conservative dots move above the 0 ms line, then measured
  KV copying was short enough that a low-overhead hardware path could likely
  have prepared KV before replay.

If observed software remains far below 0 ms while projected hardware is above
0 ms, the key blocker is not the raw copy duration alone. The blocker is the
software/runtime path before useful movement becomes deadline-ready.
```

## Directory Layout

```text
sglang_direct_kv/
  README.md
  KV_BLOCK_LEDGER.md
  KV_EXACT_MOVEMENT_ATTRIBUTION.md
  INSTRUMENTATION_AUDIT.md
  pyproject.toml
  requirements.txt

  configs/
    g5_2xlarge_smoke.yaml

  scripts/
    setup_ec2.sh
    run_sglang_server.sh
    run_sglang_hicache_server.sh
    smoke_hicache_request.sh
    run_milestone4_pressure_trace.sh
    run_milestone5_compare_modes.sh
    run_milestone6_design_space.sh
    run_milestone7_direct_hooks.sh
    run_milestone8_direct_load_design_space.sh
    run_milestone9_agentic_traffic.sh
    run_milestone9_oracle_lead_sweep.sh
    extract_swebench_trajectory_prompt_workload.py
    run_milestone20_swebench_trajectory_replay.sh
    run_milestone21_exp6_direct_sglang.sh
    run_milestone22_live_agentbench_bridge.sh
    run_milestone23_live_prefetch_intervention.sh
    run_milestone27_real_prompt_controlled_replay.sh
    run_real_prompt_controlled_replay.py
    build_milestone27_controlled_replay_report.py
    audit_master_report_evidence.py
    run_milestone24_live_paired_agentbench_report.sh
    live_prefetch_controller.py
    build_live_agentbench_tool_gap_report.py
    build_live_paired_agentbench_report.py
    run_milestone10_dma_timeline.sh
    run_milestone11_agentic_timeline.sh
    run_milestone12_paired_evidence.sh
    run_milestone13_failure_stress.sh
    run_milestone13b_green_bar_failure_stress.sh
    run_milestone14_lightweight_copy_telemetry.sh
    run_milestone15_targeted_dma_validation.sh
    run_milestone16_agentbench_sglang_direct.sh
    run_milestone18_agentbench_trace_replay_modes.sh
    run_agentbench_sglang_task.py
    run_agentbench_sglang_preflight.py
    extract_agentbench_trace_replay_workload.py
    summarize_agentbench_sglang_direct.py
    run_agentic_traffic_workload.py
    build_agentic_prefetch_timeline.py
    run_pressure_resume_workload.py
    analyze_hint_outcomes.py
    summarize_nsys_dma_timeline.py
    summarize_torch_cuda_profiles.py
    summarize_mode_comparison.py
    summarize_design_space.py
    summarize_agentic_traffic_results.py
    summarize_milestone12_paired_evidence.py
    plot_design_space.py
    build_session_cache_map.py
    validate_kv_block_ledger.py
    extract_hicache_call_report.py
    map_session_host_indices.py
    summarize_kv_trace.py
    probe_sglang_kv_paths.py
    extract_sglang_kv_targets.py
    run_workload.py

  src/
    agentic_kv/
      __init__.py
      agent_trace.py
      hints.py
      instrumentation.py
      metrics.py
      nvtx.py
      policies.py
      sglang_client.py
      sglang_trace_patch.py
      torch_cuda_profiler.py
      block_ledger/
        events.py
        normalizer.py
        block_id.py
        ledger.py
        report.py
```

## Recommended EC2 Machine

Start with:

```text
g5.2xlarge
1x NVIDIA A10G
24 GB GPU memory
8 vCPUs
32 GiB RAM
```

Use a small model first:

```text
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
Llama-3.2-1B-Instruct
Llama-3.2-3B-Instruct
```

Avoid 7B models at first because we need memory headroom for KV pressure.

## Setup On EC2

Recommended base image:

```text
AWS Deep Learning AMI GPU PyTorch
```

Then:

```bash
cd ~/agentic_hardware/sglang_direct_kv
bash scripts/setup_ec2.sh
source .venv/bin/activate
```

Verify GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Probe SGLang:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/probe_sglang_kv_paths.py --out artifacts/sglang_probe.json
python scripts/extract_sglang_kv_targets.py \
  --out-json artifacts/sglang_kv_targets.json \
  --out-md artifacts/sglang_kv_targets.md
```

Start SGLang:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/run_sglang_server.sh Qwen/Qwen2.5-1.5B-Instruct
```

Start SGLang with hierarchical KV cache enabled:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

bash scripts/run_sglang_hicache_server.sh Qwen/Qwen2.5-1.5B-Instruct
```

This enables SGLang's host/device KV cache path:

```text
--enable-hierarchical-cache
--hicache-size
--hicache-io-backend
--hicache-mem-layout
--mem-fraction-static
--attention-backend triton
--prefill-attention-backend triton
--decode-attention-backend triton
--disable-cuda-graph
--disable-overlap-schedule
```

The launcher uses the Triton attention backend for prefill and decode, and disables CUDA graph plus overlap scheduling by default. Some SGLang/FlashInfer/TVM paths still JIT compile kernels and require `CUDA_HOME`/`nvcc`, so `setup_ec2.sh` installs the minimal CUDA 12.8 JIT packages on Amazon Linux 2023 by default. Later performance runs can enable CUDA graphs and overlap scheduling again.

For `g5.2xlarge`, the launcher defaults to:

```text
HICACHE_SIZE_GB=14
MEM_FRACTION_STATIC=0.55
```

HiCache currently expects the host KV cache pool to be larger than the device KV cache pool. If this assertion fails, either increase `HICACHE_SIZE_GB` or lower `MEM_FRACTION_STATIC`.

Run workload driver:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/run_workload.py --config configs/g5_2xlarge_smoke.yaml --mode no_prefetch
python scripts/run_workload.py --config configs/g5_2xlarge_smoke.yaml --mode generic_prefetch
python scripts/run_workload.py --config configs/g5_2xlarge_smoke.yaml --mode hint_aware
```

## Direct Instrumentation Plan

The direct adapter lives in:

```text
src/agentic_kv/instrumentation.py
```

It exposes the operations we want future hardware to provide:

```text
tag_session_kv(session_id, priority, deadline, confidence)
prefetch_session_kv(session_id)
protect_session_kv(session_id, until_ms)
release_session_kv(session_id)
collect_kv_telemetry()
```

At first, the adapter is probe-only. After the SGLang KV/cache/offload paths are identified, we will replace the no-op methods with real calls into SGLang internals.

## Metrics

Primary:

```text
tool-return-to-first-token latency
time to first token
KV reload/prefetch wait time
P50/P95/P99 resume latency
```

Telemetry:

```text
prefetch submitted
prefetch completed
prefetch hit
prefetch late
prefetch wasted
evicted before use
protected until reuse
decode slowdown
```

## Important Note

This testbed is designed for direct KV instrumentation.

Prefix prewarming may still be useful as a fallback baseline, but it is not the main approach here. The main path is to find SGLang's real KV/cache/offload code and instrument that path directly.
