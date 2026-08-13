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

Run it:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

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
TTFT is recorded for each target resume
hicache.load / hicache.write / hicache.evict_device counts are summarized per mode
```

Result from the first comparison run:

```text
mode             resume_count  avg_resume_TTFT_ms  p95_resume_TTFT_ms  hicache_load  hicache_evict_device
no_prefetch      2             48.327              48.608              4             37
generic_prefetch 2             48.117              48.261              4             39
hint_aware       2             41.400              42.896              4             37
```

Important interpretation:

```text
The comparison harness works.
All three modes ran on the same model, same EC2 instance, same constrained KV pool, and same pressure/resume workload.
The hint-aware mode showed lower resume TTFT in this small run.
This is a promising signal, not yet a final conclusion, because there were only two target resume requests.
The next step is to run larger repetitions and package the results for the manager demo.
```

### Milestone 6: Manager Demo Results

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
    run_pressure_resume_workload.py
    summarize_mode_comparison.py
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
