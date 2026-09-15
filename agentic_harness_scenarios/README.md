# Archived Scenario Prototype

This directory is the first standalone prototype of the harness-aware scenario
suite. The active implementation has moved into:

```text
../sglang_direct_kv/src/agentic_kv/harness_scenarios/
```

The move lets real experiments share the existing SGLang installation,
controller metadata, trace hooks, report builders, and artifact layout under
`sglang_direct_kv`.

Keep this directory only as reference material while the migration settles. New
scenario work should happen in `sglang_direct_kv`.
