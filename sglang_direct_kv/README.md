# SGLang Direct KV Instrumentation Testbed

## Goal

Build a direct SGLang-based testbed for hint-guided KV cache prefetching.

This project intentionally starts with SGLang rather than fake KV tensors. The goal is to find and instrument the real SGLang KV/cache/offload path, then emulate future hardware support in software.

## Table Of Contents

| Section | Status | Link |
| --- | --- | --- |
| Key findings so far | Active | [Key Findings So Far](#key-findings-so-far) |
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
| F6 | `request_warm` remains an important software baseline. | Milestone 9B showed similar outcome patterns for `request_warm` and `direct_load`. | A manager can ask whether ordinary software warming is enough. | Compare hardware-assisted designs against software-only warming, not only against `no_prefetch`. | Baseline |
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
Compare direct_load against request_warm and no_prefetch under the same pressure settings.
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
Repeat the Milestone 6 design-space sweep, but add the direct SGLang load-back path.
This compares three cases:
1. no_prefetch
2. request_warm
3. direct_load
```

Why we need it:

```text
Milestone 6 showed that request-level warming can help.
Milestone 7D showed that direct_load can move real SGLang load_back work into the tool gap.
Milestone 8 asks the bigger question:
Across cache pressure, prompt size, and timing, how much better is direct_load than no_prefetch and request_warm?
```

Default design planes:

| Plane | Knob | Default Values | Simple Meaning |
| --- | --- | --- | --- |
| Prefetch timing | `TIMINGS` | `pre_pressure near_resume` | Whether the prefetch happens before pressure or close to resume. |
| Cache pressure | `FILLER_LIST` | `12 24 96 192` | How many unrelated sessions compete for KV space. |
| Request size | `PROMPT_TOKEN_LIST` | `1024 1536` | How large target and filler prompts are. |
| Prefetch action | `PREFETCH_ACTIONS` | `request_warm direct_load` | Whether we use normal request warming or the direct SGLang load-back trigger. |

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone8_direct_load_design_space \
FILLER_LIST="12 24 96 192" \
PROMPT_TOKEN_LIST="1024 1536" \
TIMINGS="pre_pressure near_resume" \
PREFETCH_ACTIONS="request_warm direct_load" \
bash scripts/run_milestone8_direct_load_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

What the default sweep runs:

```text
2 request sizes x 4 pressure levels x 5 cases

For each request size and pressure level:
1 no_prefetch baseline
2 request_warm timing choices
2 direct_load timing choices
```

That is 40 total SGLang server runs by default.
Each design point starts a fresh SGLang server, runs one case, writes metrics/traces, then stops the server.
This avoids cache state leaking from one design point into the next.

Progress shown in the terminal:

```text
Total cases: 40
==== Milestone 8 case [1/40]: no_prefetch_request_warm_near_resume_f12_p1024 ====
...
==== Completed Milestone 8 case [1/40]: no_prefetch_request_warm_near_resume_f12_p1024 ====
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
lines = request_warm/direct_load timing choices
separate charts = prompt size
tables show first TTFT, no_prefetch resume TTFT, request_warm resume TTFT, direct_load resume TTFT, and benefit values
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
request_warm may help, but it warms through a normal generation request.
direct_load should move real SGLang load_back work into the tool gap.
If direct_load works well, resume TTFT should be lower and hicache.load calls should appear during the prefetch phase instead of the resume phase.
```

How to run a small smoke sweep:

```bash
RESULT_ROOT=artifacts/results/milestone8_smoke \
FILLER_LIST="12" \
PROMPT_TOKEN_LIST="1024" \
TIMINGS="near_resume" \
PREFETCH_ACTIONS="request_warm direct_load" \
bash scripts/run_milestone8_direct_load_design_space.sh Qwen/Qwen2.5-1.5B-Instruct
```

Smoke result observed on EC2:

```text
filler_sessions: 12
prompt_tokens: 1024
timing: near_resume

no_prefetch resume TTFT: 52.389 ms
request_warm resume TTFT: 43.744 ms
direct_load resume TTFT: 42.790 ms

request_warm benefit: 8.645 ms, 16.50%
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
| `request_warm` | Dynamo-like frontend sends a normal SGLang warm request during tool wait. |
| `direct_load` | Frontend sends the direct SGLang load-back trigger during tool wait. |
| `oracle_direct_load` | Frontend sends direct load close to replay time. This is the timing upper bound. |

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_ROOT=artifacts/results/milestone9_agentic_traffic \
MODES="no_prefetch request_warm direct_load oracle_direct_load" \
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
Runs 4 clean SGLang server runs:
1. no_prefetch
2. request_warm
3. direct_load
4. oracle_direct_load

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

The same files are produced for request_warm, direct_load, and oracle_direct_load.

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
If request_warm or direct_load fires too early, KV can be evicted before replay.
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
  --modes "no_prefetch request_warm direct_load oracle_direct_load"
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
artifacts/results/latest_paired_report.html
artifacts/results/latest_paired_report.md
artifacts/results/latest_paired_report.json
artifacts/results/latest_paired_checkpoint_results.csv
artifacts/results/latest_paired_key_observations.csv
artifacts/results/latest_paired_session_details.csv
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
artifacts/results/latest_paired_report.html
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
artifacts/results/latest_paired_report.html
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
artifacts/results/latest_paired_report.html
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
artifacts/results/latest_paired_report.html
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

Small labels mark the important boundaries:
  hint start / hint end
  copy start / copy end
  replay due
  replay start

When replay starts almost exactly at the replay deadline, the chart may
merge those labels into:
  replay due/start

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

## Directory Layout

```text
sglang_direct_kv/
  README.md
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
    run_milestone10_dma_timeline.sh
    run_milestone11_agentic_timeline.sh
    run_milestone12_paired_evidence.sh
    run_milestone13_failure_stress.sh
    run_milestone13b_green_bar_failure_stress.sh
    run_milestone14_lightweight_copy_telemetry.sh
    run_milestone15_targeted_dma_validation.sh
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
