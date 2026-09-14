# Relational Prompt Shorthand

This document defines the prompt-shorthand layer used by the replay-deadline
testbed.

The package lives at:

```text
src/agentic_prompt_codec
```

It is intentionally portable. It must not import SGLang, Dynamo, or harness
implementation code.

## What Shorthand Means In This Project

Shorthand means **representing repeated relationship types with compact symbols
or forms defined in a legend**.

Original text:

```text
The cat is on the table. The book is on the shelf. The cup is next to the plate.
```

Encoded text:

```text
Legend:
[x1] = the subject is on the object
[x2] = the subject is next to the object

cat [x1] table.
book [x1] shelf.
cup [x2] plate.
```

The receiving model gets both the legend and the encoded text. The goal is to
preserve every stated fact and qualification while reducing the prompt size.

Shorthand is not summarization. It should not omit facts.

Shorthand is also not repeated-phrase aliasing. The older exact phrase alias
prototype has been removed from this project to avoid confusing future work.

## Active Codec

The active codec is:

```text
agent_trace_relations_v1
```

The default config is:

```text
configs/prompt_codecs/agent_trace_relations_v1.json
```

This codec targets coding-agent trajectory structure. It only encodes narrow,
rule-based relations that can be mechanically expanded back to the original
text. Unsupported text remains unchanged.

Current examples:

```text
WR1=/home/ec2-user/agentbench/repos/example_repo
F1=`path/to/file.py`
FN1=normalize_url
VC1=python -m pytest tests/unit/test_example.py
P(review)
PH(planning)
SH(1; Inspect F1)
SB(1; Identify Relevant Files)
R(F1; understand current behavior)
TC(read_file tests/unit/test_example.py)
X(`execute("python -m pytest")`; ensure tests pass)
M(host blocking logic; consider subdomains)
E(whitelist rules take precedence)
G(no_tool_calls)
T(none)
```

Each request carries its own legend. There is no hidden memory, no cross-request
state, no fine-tuning, and no extra model call.

## Modular Rule Layout

The codec is deliberately built as a small rule pipeline rather than one large
rewriter:

```text
src/agentic_prompt_codec/codecs/relations.py
src/agentic_prompt_codec/rules/base.py
src/agentic_prompt_codec/rules/entities.py
src/agentic_prompt_codec/rules/commands.py
src/agentic_prompt_codec/rules/workflow.py
src/agentic_prompt_codec/rules/phase.py
src/agentic_prompt_codec/rules/action.py
src/agentic_prompt_codec/rules/runtime.py
src/agentic_prompt_codec/rules/registry.py
```

`rules/registry.py` is the public switchboard. It owns the stable rule names,
the default order, and the legend text. The config can enable a subset:

```json
{
  "codec": {
    "codec": "agent_trace_relations_v1",
    "enabled_rules": ["file_path_alias", "function_alias", "phase", "read_for_purpose"]
  }
}
```

Each applied request records `encoding_rule_counts` in the gateway evidence.
That field shows which shorthand rules actually fired for the request.

The alias rules run first. They define request-local symbols such as `WR1`,
`F1`, `FN1`, and `VC1` when the same workspace root, file path, function name,
or validation command appears repeatedly inside one prompt. Later relation
rules can then use those aliases, for example `R(F1; understand behavior)`.

After aliases, each line gets at most one relation rewrite. This intentionally
prevents nested shorthand forms, which keeps one-pass decoding safe and easier
to audit.

## Safety Rules

Only add a shorthand rule when the encoded form preserves:

- negation
- uncertainty
- time and ordering
- quantities
- entity identity
- file paths
- commands
- exception text
- exact-output requirements

If a relation is ambiguous, leave the text unchanged.

The codec is allowed to pass through unchanged for any reason: no matching
relation, no net token savings after the legend, budget exhaustion, unsupported
payload shape, or validation failure.

## Library API

```python
from agentic_prompt_codec import CodecConfig, PromptEncoder
from agentic_prompt_codec.tokenizers import HuggingFaceTokenCounter

encoder = PromptEncoder(
    CodecConfig(codec="agent_trace_relations_v1"),
    HuggingFaceTokenCounter("Qwen/Qwen2.5-Coder-7B-Instruct"),
)
result = encoder.encode(request, "openai_chat")
send_to_backend(result.payload)
record(result.evidence())
```

Use `PromptEncoder()` with no config for identity/no-op behavior.

## Experiment Gateway

The controller shorthand mode uses this config by default:

```bash
CONTROLLER_SHORTHAND_CODEC_CONFIG="configs/prompt_codecs/agent_trace_relations_v1.json"
```

Example:

```bash
cd sglang_direct_kv
HARDWARE_PROFILE=ec2_a10g \
HARNESSES=hatcher \
PRESSURE_LEVELS="p3_high" \
SIGNAL_FAMILIES="baseline frontend_supplied controller_priority_demotion_admission controller_shorthand" \
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES="controller_priority_demotion_admission" \
CONTROLLER_SHORTHAND_MODES="controller_priority_demotion_admission_shorthand" \
CONTROLLER_SHORTHAND_CODEC_CONFIG="configs/prompt_codecs/agent_trace_relations_v1.json" \
CONTROLLER_SHORTHAND_ENCODING_SCOPE="target_requests" \
FILLER_REPLAY_DEADLINES=1 \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="shorthand_controller_p3_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

## Evidence

Prompt-shorthand evidence is written to:

```text
prompt_encoding_proof.csv
prompt_encoding_summary.csv
```

Important fields:

- `encoding_codec`
- `encoding_status`
- `encoding_reason`
- `encoding_original_tokens`
- `encoding_encoded_tokens`
- `encoding_saved_tokens`
- `encoding_elapsed_ms`
- `encoding_validation`
- `encoding_rule_counts`

Interpret latency only after checking that the encoded request actually saved
tokens after including the legend. Token savings alone do not prove task quality;
quality must be evaluated separately.

## Trajectory Scan

Before adding a new shorthand idea to a controller experiment, scan real agent
trajectories to see whether the rule fires often enough to matter:

```bash
cd sglang_direct_kv
python3 scripts/scan_prompt_shorthand_trajectories.py \
  --root /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/.codex_external/agentbench_results \
  --approx-token-counter \
  --max-records 500
```

The scanner writes:

```text
artifacts/results/prompt_shorthand_trajectory_scan.csv
```

Use the approximate counter for quick local scans. Use the real tokenizer on
EC2/GH200 when judging final token savings.

## Adding A New Rule

1. Add a narrow rule under `src/agentic_prompt_codec/rules/`.
2. Register the rule and its legend in `rules/registry.py`.
3. Enable it in a separate config JSON.
4. Add a round-trip test proving that decoding restores the original text.
5. Run the trajectory scanner and inspect `rule_counts` before using it in a
   full experiment.

## Validation

From `sglang_direct_kv`:

```bash
PYTHONPATH=src python -m unittest tests/test_prompt_codec.py -v
python -m py_compile scripts/harness_sglang_gateway.py scripts/build_multi_harness_deadline_summary.py
bash -n scripts/run_harness_deadline_pressure.sh scripts/run_harness_signal_design_space.sh
```
