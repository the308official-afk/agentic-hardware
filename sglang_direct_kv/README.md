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

## First Milestone

The first milestone is not performance yet. It is mapping SGLang internals on the GPU EC2 machine.

We need to answer:

```text
Where are KV blocks allocated?
Where are prefix/radix cache entries stored?
Where does eviction happen?
Is HiCache or CPU offload available in this install?
Can we associate KV blocks with a request/session?
Can we force, promote, prefetch, or protect selected KV blocks?
```

Run:

```bash
python scripts/probe_sglang_kv_paths.py --out artifacts/sglang_probe.json
python scripts/extract_sglang_kv_targets.py \
  --out-json artifacts/sglang_kv_targets.json \
  --out-md artifacts/sglang_kv_targets.md
```

The probe creates a JSON map of SGLang modules/classes/functions related to:

```text
kv
cache
radix
prefix
hicache
offload
evict
memory
page
block
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
cd agentic_hardware/sglang_direct_kv
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
python scripts/probe_sglang_kv_paths.py --out artifacts/sglang_probe.json
python scripts/extract_sglang_kv_targets.py \
  --out-json artifacts/sglang_kv_targets.json \
  --out-md artifacts/sglang_kv_targets.md
```

Start SGLang:

```bash
bash scripts/run_sglang_server.sh Qwen/Qwen2.5-1.5B-Instruct
```

Start SGLang with hierarchical KV cache enabled:

```bash
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

The launcher uses the Triton attention backend for prefill and decode, and disables CUDA graph plus overlap scheduling by default. Some SGLang/FlashInfer/TVM paths still JIT compile kernels and require `CUDA_HOME`/`nvcc`, so `setup_ec2.sh` installs CUDA Toolkit 12.8 on Amazon Linux 2023 by default. Later performance runs can enable CUDA graphs and overlap scheduling again.

For `g5.2xlarge`, the launcher defaults to:

```text
HICACHE_SIZE_GB=14
MEM_FRACTION_STATIC=0.55
```

HiCache currently expects the host KV cache pool to be larger than the device KV cache pool. If this assertion fails, either increase `HICACHE_SIZE_GB` or lower `MEM_FRACTION_STATIC`.

Run workload driver:

```bash
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
