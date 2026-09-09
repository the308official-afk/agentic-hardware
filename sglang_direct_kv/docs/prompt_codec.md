# Request-local prompt encoding

The encoder replaces repeated language with shorthand and a legend before the
model sees the request. It is disabled by default and does not change the
scheduling controller. Each request is independent: there is no conversation
summary, remembered dictionary, fine-tuning, or hidden inference call.

## What ships

- `agentic_prompt_codec`: an independent namespace with no imports from
  `agentic_kv`, SGLang, or any harness. The core uses only Python's standard library.
- `identity`: unchanged payload, no tokenizer initialization or calls.
- `dictionary_v1`: deterministic repeated-phrase substitution, local legends,
  collision-free aliases, and exact one-pass reconstruction validation.
- `relations_v1`: a narrow positive location grammar such as
  `the cat is on the table.` → `the cat @ the table.`. It does not infer arbitrary
  relations or rewrite uncertainty/negation. It passes through if the complete
  result, including its legend, is not smaller.
- A `SummaryCodec` callable extension for an explicitly supplied prose
  summarizer. No provider is selected automatically; this is an integration API,
  not a built-in trained summarization model or a validated semantic compressor.
- Adapters for plain text, OpenAI Chat, Anthropic Messages, and Responses.
  They replace only eligible user/tool text while retaining roles, tool IDs,
  schemas, cache fields, opaque blocks, and unknown fields.
- An opt-in hook in the experiment gateway, a separate streaming proxy,
  token/quality evaluator, pressure-matrix launcher, and encoding proof tables.

## Installation and portability

From `sglang_direct_kv`:

```bash
pip install -e '.[prompt-codec]'
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

The optional extra installs tokenizer/template support. Core library users can
copy or package only `src/agentic_prompt_codec` and provide a `TokenCounter`;
the controller and research dependencies are not required by that namespace.
The proxy/evaluator use `httpx`, already a dependency of this testbed.

Tokenizer files must be available before an active experiment. Fetch them once:

```bash
python -c 'from agentic_prompt_codec.tokenizers import HuggingFaceTokenCounter; HuggingFaceTokenCounter("Qwen/Qwen2.5-Coder-7B-Instruct", local_files_only=False)'
```

Serving configuration defaults to local-only tokenizer loading and refuses a
configured tokenizer model that differs from the runner's model. Pin tokenizer
revision in configuration when reproducibility across machines matters. The
vocabulary, chat template, special-token map, model, and declared revision are
fingerprinted in evidence. Reconcile local counts with backend input counts;
custom server templates need a matching `renderer` callback.

## Library API

```python
from agentic_prompt_codec import CodecConfig, PromptEncoder
from agentic_prompt_codec.tokenizers import HuggingFaceTokenCounter

encoder = PromptEncoder(
    CodecConfig(codec="dictionary_v1"),
    HuggingFaceTokenCounter("Qwen/Qwen2.5-Coder-7B-Instruct"),
)
result = encoder.encode(request, "openai_chat")
send_to_backend(result.payload)
record(result.evidence())
```

The caller retains the original payload. The result chooses the original on
no savings, missing tokenizer, unsupported accounting, validation failure, or
budget exhaustion. Configuration errors fail at startup; they do not silently
select a different codec. Use `PromptEncoder()` for identity or omit the hook.

`encode(..., budget_ms=remaining_budget)` permits optional deadline-aware
bypass without coupling the encoder to controller state. The budget is
cooperative, checked throughout built-in work. A tokenizer/plugin call cannot
be forcibly interrupted; external plugins must enforce their own timeouts.
All spent time, including rejected candidates, remains visible in evidence.

Implement `Codec.encode`, `TokenCounter.count_text/count_request`, or a custom
`RequestAdapter` to replace one component. Pass `adapter=` to define stricter
eligibility or exact-preservation boundaries. Built-in protection of code,
paths, URLs, and JSON-like lines is conservative, not a complete parser of all
languages. Exact quoting/output requirements should be excluded by an adapter.

Legends stay inside the originating user/tool text segment. Dictionary
definitions are data, and should not be promoted into system instructions.
Textual reversibility is a mechanical property; model understanding is a
separate empirical question.

For a prose comparison:

```python
from agentic_prompt_codec.codecs.summary import SummaryCodec

# summarize(text, timeout_ms) is explicitly provided by your application.
encoder = PromptEncoder(
    CodecConfig(codec="summary_v1"), counter, SummaryCodec(summarize)
)
```

This records `protected_content_only` validation, never `round_trip`. Measure
the summarizer's resource cost too. Sending the verbose input to the same LLM
backend first would not avoid processing it there.

## Experiment gateway

```bash
PROMPT_CODEC_CONFIG=configs/prompt_codecs/dictionary_v1.json \
PROMPT_ENCODING_SCOPE=target_requests \
HARNESSES=hatcher MODES='no_prefetch controller_full' \
PRESSURE_LEVELS='p0_control p1_mild' \
REPORT_BUILDER_MODE=lightweight UPDATE_LATEST=0 \
bash scripts/run_harness_deadline_pressure.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Omit `PROMPT_CODEC_CONFIG` to disable. Scopes:

| Scope | Eligible experiment phases |
|---|---|
| `target_requests` | Initial and replay requests; filler remains unchanged |
| `replay` | Replay only |
| `all` | All marked forwarded requests, including filler and warmup |

The hook acts on the normalized backend payload, after extracting scheduling
signals and removing experiment markers. Both comparison arms use the same
legacy normalization, which flattens messages into a user prompt. This gateway
remains an experiment adapter: it suppresses bookkeeping/duplicate marked
requests and buffers backend output before replying. Use the standalone proxy
for protocol-preserving traffic and actual streaming.

The optional `PROMPT_WORKLOAD_JSONL` selects externally supplied initial/replay
prompts rather than shared-context padding. Each JSONL item has `id`,
`initial_prompt`, `replay_prompt`, and optionally `expected_answer` for the
separate evaluator. Pressure runs select items in session order and wrap when
there are fewer items than sessions; they do not score answers. Expected answers
are never passed to the backend as metadata.

## Streaming proxy

```bash
python -m agentic_prompt_codec.proxy \
  --target-base http://127.0.0.1:30000 \
  --port 32080 --config configs/prompt_codecs/dictionary_v1.json \
  --events /tmp/prompt-proxy-events.jsonl
```

This proxy preserves the API endpoint and request structure; the upstream must
support that API. It never synthesizes answers or deduplicates requests. Client
credentials are forwarded only to the configured upstream. Streaming bytes are
forwarded as they arrive; cancellation closes the upstream stream. Encoding
contention bypasses instead of forming an unbounded encoding queue.

The built-in Hugging Face counter supports text and textual Chat requests.
Anthropic/Responses and multimodal accounting require an injected renderer or
counter. Without one, encoding bypasses while the original protocol passes
through. Their adapters are structurally supported, not a claim of verified
token savings on every native backend.

## Token and quality evaluation

```bash
python scripts/evaluate_prompt_codec.py \
  --workload tests/fixtures/prompt_codec/workload.jsonl \
  --config configs/prompt_codecs/dictionary_v1.json \
  --out-dir /tmp/codec-token-evaluation
```

Without `--base-url`, this counts tokens only. With an idle, explicitly chosen
backend, append `--base-url http://127.0.0.1:30000 --repeats 3` to run both arms
and score exact answers. Output directories must be new to prevent overwriting
results. The nine included cases are hand-authored fixtures covering negation,
uncertainty, chronology, exceptions, exact values, literal aliases, code, and
references. They are not real captured trajectories or a comprehensive benchmark.

The evaluator records all bypasses/errors, complete input token counts, output
token usage when supplied by the backend, encoding time, TTFT including encoding,
and task accuracy. Output auditing counts are outside the timed inference call;
the encoder's own counting/validation time is included. Both arms use the same
output allowance and counterbalanced order. `--cache-condition natural` uses the
backend default. `cold` assigns a fresh SGLang cache namespace per measured
request; `warm` first primes that same namespace with the exact input and a
one-token output allowance. Priming cost is recorded separately, outside measured
warm-request latency. These are requested conditions: confirm cache namespace
support and cached-token counts in backend evidence. No mode flushes shared cache.

## Pressure matrix

Inspect the exact randomized plan without launching anything:

```bash
python scripts/run_prompt_encoding_matrix.py --dry-run --repeats 3 \
  --workload tests/fixtures/prompt_codec/workload.jsonl
```

Remove `--dry-run` on an idle EC2 host to execute. Defaults are DeepAgents,
identity/dictionary encoding, baseline/full-controller scheduling, and
P0/P1/P3/P4/P5 pressure. The wrapper never updates `latest_*` reports, refuses
an existing matrix directory, and encoding cases refuse occupied serving ports.
Run it only after other experiments have released the GPU; port checks cannot
detect unrelated workloads using different ports.

Configuration, scope, repetition, source revision, and workload content hash
are included in encoding case identity. Run configuration is snapshotted before
launch. Child reports remain separate; `matrix_plan.json` lists their names and
order. Match workloads and offered filler traffic when comparing arms. Use a
separate `--scope all` experiment for whole-platform capacity effects.

Each pressure case starts a fresh SGLang process but performs an initial turn
before replay, so it is not a claim that every replay has a cold prefix cache.
For controlled cold/warm comparisons, run separate benchmark conditions on
an isolated server; never flush the cache of another task's experiment.

## Evidence and interpretation

`prompt_encoding_proof.csv` includes decisions, reasons, configuration identity,
source/encoded hashes, net savings including legends, validation type, encoding
duration, backend status, and replay timing. Backend prefill/cache counts are
included when existing telemetry observes them. Unknown values remain unknown.
`prompt_encoding_summary.csv` includes bypasses and failures in its denominator.

First-content timestamps are captured directly, with integer nanoseconds, at
backend stream observation. Reports use that timestamp rather than adding a
backend-only duration to an earlier request start. Old traces retain an explicitly
labeled inferred source. Empty/error responses do not fabricate first tokens.
The original replay-due event is unchanged. The experiment gateway reports
gateway-observed content; proxy first-client-write time is still not a measurement
of receipt at a remote client.

Encoding configurations are separate summary/plot groups. Compression ratios
do not establish latency or quality gains. Require net savings, measured
end-to-end benefit including encoding, and a predeclared acceptable quality
change before enabling a codec beyond experiments.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
python -m py_compile scripts/harness_sglang_gateway.py scripts/build_multi_harness_deadline_summary.py
bash -n scripts/run_harness_deadline_pressure.sh scripts/run_harness_signal_design_space.sh
DRY_RUN=1 bash scripts/run_ec2_controller_repeatability.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Tests cover reconstruction, alias collisions, protected data, roles/tool IDs,
token-budget rejection, disabled behavior, request-local concurrency, fake-backend
forwarding, first-content accounting, streaming delivery before completion, and
report grouping. GPU latency and model task quality require separate live runs.
