# SGLang Direct KV Instrumentation Testbed

## Goal

Build a direct SGLang-based testbed for hint-guided KV cache prefetching.

This project intentionally starts with SGLang rather than fake KV tensors. The goal is to find and instrument the real SGLang KV/cache/offload path, then emulate future hardware support in software.

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

### Milestone 8: Manager Demo Results

Status: planned.

What it is:

```text
Produce a small, credible result table and timeline traces.
```

Why we need it:

```text
The goal is not only to build a clever prototype.
We need evidence that a manager can read quickly and trust: real LLM, real KV behavior, clear baseline, clear benefit, clear limitations.
This milestone packages the experiment into a concise story for why hardware/runtime co-design may be worth exploring.
```

Outputs:

```text
TTFT after tool return
P50/P95/P99 resume latency
KV load/write/eviction counts
prefetch hit/late/wasted rate
bandwidth or decode interference signal if available
short timeline examples for Agent 42-style workflows
```

Important events to observe:

```text
tool_wait starts
agent.hint_submitted
hicache.write
hicache.evict_device
hicache.load
agent.resume_start
first token emitted
TTFT and total latency recorded
```

Success criteria:

```text
A short table comparing Mode 1, Mode 2, and Mode 3.
A timeline showing a coding-agent tool gap and KV movement around that gap.
A clear statement of what is proven, what is only emulated, and what hardware support would make cheaper or more predictable.
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
    run_pressure_resume_workload.py
    summarize_mode_comparison.py
    summarize_design_space.py
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
      policies.py
      sglang_client.py
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
