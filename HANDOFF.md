# Codex Handoff

Updated: 2026-09-09

This is the current handoff for another Codex task working on the agentic
hardware replay-deadline infrastructure. Treat this file plus the top-level
`README.md` as the active source of truth. The older long-form notebook at
`sglang_direct_kv/README.md` is useful history, but it includes outdated
milestones and should not drive current implementation choices by itself.

## Project Overview

This project studies a specific bottleneck in agentic LLM systems: after an
agent calls a tool, it has to return to the model and generate the next token
quickly. That return-to-model request is called the replay request.

The user cares about whether that replay request meets a deadline. A replay is
good if the first replay token arrives on time; it is bad if queueing, GPU
pressure, KV-cache pressure, or other agents make it late.

The testbed creates controlled pressure around that replay request:

- background/filler model requests
- queue pressure inside SGLang
- KV-cache pressure
- short tool-wait windows
- multiple agents becoming replay-ready at the same time

Then it compares whether different signal/controller paths make the replay
request faster or more deadline-aware.

The main measured path is:

```text
experiment driver -> harness -> gateway -> SGLang -> first replay token
```

The harness represents the agent framework or coding-agent client shape. The
gateway is the portability boundary. It observes and translates request signals
into SGLang-compatible fields without requiring each harness or SGLang version
to be patched directly.

The current controller work adds a portable policy layer around this path:

```text
agent lifecycle events -> controller policy -> gateway/SGLang actions
```

The controller watches when an agent enters tool wait, when replay becomes
likely, when replay is ready, and when replay is finished. It can then decide to
raise replay priority, temporarily demote background work, skip speculative work
under overload, or restore normal traffic afterward.

The main report output is the Replay Deadline Pressure Chart. It shows how late
or early replay first tokens were under different pressure levels and modes.
Exact proof lives in the evidence tables and CSV artifacts.

Simple framing for the core idea:

```text
Priority says: this request matters.
Controller says: this request matters, and I will temporarily make room for it.
```

## Current State

The repository is focused on measuring whether agentic LLM replay requests can
meet deadlines under SGLang queue, GPU, KV-cache, and multi-agent pressure.

The current active design is the portable agent-aware controller:

- lifecycle state is tracked outside SGLang
- SGLang-specific behavior lives behind thin gateway/adapter boundaries
- priority, demote/restore, admission, speculative preload, and targeted KV
  prefetch are tested as explicit modes
- the lightweight report builder is the default report path

Latest pushed commit at handoff time:

```text
23b6d85 Document full controller optimization phase
```

Always push source changes to GitHub `main` after committing.

## Active Machine

Work from this computer should target the EC2 machine, not GH200.

The user currently does not have GH200 access from this computer. GH200 remains
a later migration/scale-up target only.

Local development checkout:

```bash
cd /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware
```

EC2 helper workflow:

```bash
./aws/check_ec2_ready.sh 0
./aws/upload.sh 0
./aws/ssh_to_ec2.sh 0
```

On EC2:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate
```

Do not copy local `.venv/`, `.venvs/`, `node_modules/`, or large artifact trees
between machines.

## Latest Important Report

Most recent downloaded report:

```text
sglang_direct_kv/artifacts/results/latest_master_report.html
sglang_direct_kv/artifacts/results/latest_evidence_tables.html
sglang_direct_kv/artifacts/results/latest_manifest.json
```

Most recent EC2 controller repeatability archive:

```text
sglang_direct_kv/artifacts/results/reports/ec2_controller_repeatability_20260909_034231/
```

That run used:

```text
harness: hatcher / DeepAgents
pressure levels: p1_mild p3_high p4_cliff p5_boss_queue
modes: no_prefetch e2e_priority_hints controller_scheduler_priority controller_demote_restore controller_admission_control
```

Observed median first-token lateness:

| Pressure | Baseline | Gateway Priority | Best Controller Result |
| --- | ---: | ---: | ---: |
| `p1_mild` | 8.77 s | 2.10 s | 2.10 s |
| `p3_high` | 12.16 s | 3.99 s | 3.88 s |
| `p4_cliff` | 62.81 s | 13.63 s | 12.53 s |
| `p5_boss_queue` | 48.43 s | 19.37 s | 6.48 s |

Interpretation: controller scheduler, demote/restore, and admission-control
paths are the strongest current EC2 evidence. They help because they shape the
traffic around the replay deadline, not just the replay request itself.

## Current Next Task

Implement and validate `controller_full` for one harness before expanding to
more harnesses.

`controller_full` should combine only the controller behaviors that helped on
EC2:

1. Track session lifecycle state: tool wait, replay ready, replay submitted,
   replay finished.
2. Demote matching filler/background work during the replay-critical window.
3. Raise replay scheduler priority when the replay becomes ready.
4. For `p5_boss_queue`, assign deadline-aware priority ranks among urgent
   replays instead of giving every urgent replay the same priority.
5. Use admission control to skip speculative preload/prefetch under overload.
6. Restore filler/background priority after the replay-critical window closes.

Do not include speculative preload or targeted KV prefetch in the first
`controller_full` default. Those are mechanically wired, but prior EC2 timing
showed weak or negative benefit.

## Expected Validation Command

After implementing `controller_full`, run the first validation only on
DeepAgents/Hatcher:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESSES=hatcher \
PRESSURE_LEVELS="p1_mild p3_high p4_cliff p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints controller_scheduler_priority controller_demote_restore controller_admission_control controller_full" \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_deadline_pressure.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

If using the outer signal-family script, add a `controller_full` family only
after the lower-level runner supports `controller_full`.

## Success Criteria

`controller_full` should match or beat the best individual controller mode at
each pressure level, especially `p5_boss_queue`.

The report/evidence tables must prove:

- replay priority was assigned
- filler/background priority was demoted
- filler/background priority was restored
- admission decision was recorded
- urgent replay rank/order was recorded for multi-replay cases
- first-token lateness improved or tied the best individual controller mode

Do not claim a controller win from chart position alone; tie the claim to the
CSV proof fields and request IDs.

## Code Areas To Inspect First

Use this map to decide where to make changes.

| Need | File or directory | Notes |
| --- | --- | --- |
| Understand the controller API | `sglang_direct_kv/src/agentic_kv/controller/__init__.py` | Public exports for controller models, policies, state, timing, and backend adapters. |
| Add or modify controller decisions | `sglang_direct_kv/src/agentic_kv/controller/policy.py` | Main place for `controller_full`. Keep policy logic backend-neutral. |
| Add new controller event/command fields | `sglang_direct_kv/src/agentic_kv/controller/models.py` | Add portable fields here first, then lower them at the gateway/backend boundary. |
| Track per-session lifecycle state | `sglang_direct_kv/src/agentic_kv/controller/state_store.py` | Stores tool-wait, ready, generation, and session state. |
| Change deadline or timing estimates | `sglang_direct_kv/src/agentic_kv/controller/timing_estimator.py` | Keep timing logic separate from SGLang-specific code. |
| Lower controller commands to backend actions | `sglang_direct_kv/src/agentic_kv/controller/backend.py` | Adapter/enforcer layer. Do not put high-level policy here. |
| Add a new experiment mode | `sglang_direct_kv/scripts/run_multi_harness_replay_driver.py` | Driver creates replay/filler traffic and emits lifecycle events. Add `controller_full` behavior here after policy support exists. |
| Ensure SGLang launch flags match the mode | `sglang_direct_kv/scripts/run_harness_deadline_pressure.sh` | Add `controller_full` to the priority-enabled mode list so SGLang gets priority scheduling flags. |
| Add a signal-family wrapper | `sglang_direct_kv/scripts/run_harness_signal_design_space.sh` | Only needed if `controller_full` should be selectable as a signal family. Lower-level mode support should come first. |
| Translate harness/controller metadata at the SGLang boundary | `sglang_direct_kv/scripts/harness_sglang_gateway.py` | Gateway should translate or enforce signals, not invent hidden experiment behavior unless the mode explicitly asks for it. |
| Update report chart/proof tables | `sglang_direct_kv/scripts/build_multi_harness_deadline_summary.py` | Add summary/proof columns for `controller_full`, urgent replay rank, demote/restore, and admission evidence. |
| Collect environment/capability proof | `sglang_direct_kv/scripts/collect_run_environment.py` | Use this when a report needs machine, SGLang, or capability metadata. |
| Smoke-test controller logic without GPU | `sglang_direct_kv/scripts/smoke_agentic_controller.py` | Fast sanity check for policy behavior. |
| Unit-test controller behavior | `sglang_direct_kv/tests/test_agentic_controller.py` | Add tests before EC2 runs. At minimum test full-controller commands and proof fields. |
| Run the focused EC2 repeatability ladder | `sglang_direct_kv/scripts/run_ec2_controller_repeatability.sh` | Existing script for the current one-harness EC2 controller comparison. Update later if `controller_full` becomes the default. |
| EC2 connection and sync | `aws/check_ec2_ready.sh`, `aws/upload.sh`, `aws/ssh_to_ec2.sh` | Use these from the local repo root. |
| GH200 docs and runners | `gh200/README.md`, `gh200/run_controller_scaleup.sh` | Migration target only from this computer. Do not assume GH200 access here. |

Suggested first implementation path for `controller_full`:

1. Add a backend-neutral policy path in `controller/policy.py`.
2. Add any missing command/proof fields in `controller/models.py`.
3. Add unit coverage in `tests/test_agentic_controller.py`.
4. Wire the mode into `run_multi_harness_replay_driver.py`.
5. Add launch/report support in `run_harness_deadline_pressure.sh` and
   `build_multi_harness_deadline_summary.py`.
6. Run local validation.
7. Upload to EC2 and run the DeepAgents/Hatcher validation.

## Validation Before Push

Run these locally before pushing source changes:

```bash
cd sglang_direct_kv
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m py_compile scripts/build_multi_harness_deadline_summary.py
bash -n scripts/run_harness_deadline_pressure.sh scripts/run_harness_signal_design_space.sh
```

For runner-only changes, also use dry-run expansion when available:

```bash
cd sglang_direct_kv
DRY_RUN=1 bash scripts/run_ec2_controller_repeatability.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

## Reporting Preferences

The latest master report should be chart-first and uncluttered:

- keep the Replay Deadline Pressure chart prominent
- keep exact raw proof in `latest_evidence_tables.html`
- do not mix old experiments into the latest report
- avoid overclaiming when evidence is inferred rather than directly traced

The user wants simple explanations of results. Preferred framing:

```text
Priority says: this request matters.
Controller says: this request matters, and I will temporarily make room for it.
```

## Cautions

- Do not treat GH200 docs as the active runtime for this computer.
- Do not patch installed SGLang or harness packages directly unless the user
  explicitly requests it. Prefer wrappers, gateways, adapters, and portable
  proof tables.
- Do not resurrect old `dynamo_priority_hints` framing as the main path. It is
  historical context now.
- Keep `hatcher` user-facing labels as `DeepAgents` where possible.
- Preserve unrelated local or generated changes unless the user asks otherwise.
