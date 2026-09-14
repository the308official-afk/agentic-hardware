# Agentic Prompt Codec

This package is the portable prompt-shorthand layer for the replay-deadline
testbed. It must stay independent of SGLang, Dynamo, and any specific harness.

## What Shorthand Means Here

In this project, **shorthand means a reusable symbol for a relationship**. The
relationship is defined in a legend, then reused with different subjects and
objects.

```text
Original:
The cat is on the table. The book is on the shelf. The cup is next to the plate.

Legend:
[x1] = the subject is on the object
[x2] = the subject is next to the object

Encoded:
cat [x1] table.
book [x1] shelf.
cup [x2] plate.
```

The receiving model sees both the legend and the encoded text. The goal is to
preserve every stated fact while using fewer tokens.

## What It Does Not Mean

Do not use repeated-phrase aliasing as the shorthand system for this project.
That older prototype has been removed to avoid confusion.

Do not treat shorthand as summarization. Summaries may omit, merge, or rephrase
facts. Shorthand should preserve the original facts, including negation,
uncertainty, time, quantities, entity identity, paths, commands, exceptions, and
exact-output requirements.

## Active Codec

The active shorthand codec is:

```text
agent_trace_relations_v1
```

Its default config is:

```text
configs/prompt_codecs/agent_trace_relations_v1.json
```

It is rule-based and request-local. It only rewrites narrow coding-agent
trajectory relationships that can be mechanically expanded back to the original
text. Unsupported text passes through unchanged.

The implementation is split into small pieces:

```text
codecs/relations.py     composes enabled rules into one request-local codec
rules/base.py           shared reversible rule helpers
rules/entities.py       request-local workspace/file/function aliases
rules/commands.py       validation-command and tool-command shorthand
rules/workflow.py       markdown phase/step heading shorthand
rules/phase.py          phase shorthand rules
rules/action.py         read/run/modify/ensure/do-not shorthand rules
rules/runtime.py        guard/tool-call shorthand rules
rules/registry.py       stable rule names, default order, and legend text
```

The enabled rule list lives in:

```text
configs/prompt_codecs/agent_trace_relations_v1.json
```

Each applied request records `encoding_rule_counts`, so reports can show which
rules fired. This is the proof path for future shorthand experiments.

Examples of allowed forms:

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

When extending this package, add new relation rules only when their meaning is
explicit, scoped by the legend, and mechanically reversible or otherwise covered
by a strict semantic-preservation test.

Extension checklist:

1. Add the rule to a focused file under `rules/`, or create a new file there.
2. Register the rule name and legend in `rules/registry.py`.
3. Enable it in a config JSON only after tests pass.
4. Add round-trip tests that prove `decode(encode(text)) == text`.
5. Run the trajectory scanner before using it in controller experiments.

The highest-value active rules are request-local aliases and repeated workflow
structure. If one prompt mentions the same workspace root, path, function, or
validation command many times, the codec can define `WR1`, `F1`, `FN1`, or `VC1`
once in the legend and reuse that symbol throughout the prompt. The workflow
forms, such as `PH(...)`, `SH(...)`, and `SB(...)`, compact repeated phase and
step headings without changing their meaning.

To keep decoding simple and robust, aliases are applied before relation rules,
and then each line gets at most one relation rewrite. This avoids nested
shorthand forms that would be hard to expand safely in one pass.
