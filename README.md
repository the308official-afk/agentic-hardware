# Agentic Hardware: Replay Deadline Pressure For Agentic LLMs

This repository is a research testbed for measuring whether agentic LLM
systems can resume quickly after tool calls.

The current question is:

> When many agents return from tools and need their next LLM token, can priority
> signals help the replay request meet its deadline under GPU, KV-cache, queue,
> and burst pressure?

The project has moved beyond the older generic "software prefetch" framing. The
current experiments focus on replay-deadline readiness with real SGLang serving,
HiCache, live timestamped telemetry, SGLang priority scheduling, controlled
pressure levels, and multiple coding-agent harness shapes.

## What We Mean By Shorthand

In this project, **shorthand means representing a relationship with a reusable
symbol defined in a legend**. Each expression has the form
`subject [symbol] object`. The same symbol keeps the same meaning when the
subject and object change.

For example, the original text is:

```text
The cat is on the table. The book is on the shelf. The cup is next to the plate.
```

Its symbolic representation is:

```text
Legend:
[x1] = the subject is on the object
[x2] = the subject is next to the object

cat [x1] table.
book [x1] shelf.
cup [x2] plate.
```

The receiving LLM gets both the legend and the encoded text. The goal is to
preserve every stated fact and qualification while expressing relationships
compactly. Summarization and repeated-phrase substitution are separate
techniques; neither defines what we mean by shorthand here.

This example defines the intended notation, not the current codec's literal
output or a measured token saving. The portable `agentic_prompt_codec` module
currently includes a narrow `is on` relation prototype using `@`, alongside a
separate dictionary codec. See the [shorthand definition and implementation
boundaries](sglang_direct_kv/docs/prompt_codec.md#what-shorthand-means-in-this-project)
and the [module README](sglang_direct_kv/src/agentic_prompt_codec/README.md).

## Repository Map

| Path | Purpose |
| --- | --- |
| [sglang_direct_kv/](sglang_direct_kv/) | Main SGLang replay-deadline testbed. |
| [sglang_direct_kv/README.md](sglang_direct_kv/README.md) | Long-form milestone notebook with historical detail. |
| [sglang_direct_kv/scripts/](sglang_direct_kv/scripts/) | Experiment runners, workload drivers, report builders, and SGLang launch helpers. |
| [HANDOFF.md](HANDOFF.md) | Current handoff for another Codex task working on this infrastructure. |
| [aws/README.md](aws/README.md) | EC2 sync and connection workflow. |
| [gh200/README.md](gh200/README.md) | Short GH200 setup and experiment run guide. |
| [HARDWARE_EMULATION_ENVIRONMENT.md](HARDWARE_EMULATION_ENVIRONMENT.md) | Original hardware-emulation environment notes. |
| [REPLAY_PATH_INSTRUMENTATION_PROPOSAL.md](REPLAY_PATH_INSTRUMENTATION_PROPOSAL.md) | Replay-path instrumentation design notes. |

## Latest Report Outputs

Each run updates the latest report files:

```text
sglang_direct_kv/artifacts/results/latest_master_report.html
sglang_direct_kv/artifacts/results/latest_evidence_tables.html
sglang_direct_kv/artifacts/results/latest_manifest.json
```

Archived reports are kept under:

```text
sglang_direct_kv/artifacts/results/reports/<REPORT_LABEL>/master_report.html
```

The latest master report is intentionally treated as a single-experiment report.
It should represent the most recent run, not a mixture of old and new
experiments.

## Environment

On the EC2 machine used for the current experiments:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate
```

From a local checkout, the common EC2 helper commands are:

```bash
./aws/upload.sh 0
./aws/ssh_to_ec2.sh 0
./aws/download.sh 0
```

## GH200 Transfer And Run Workflow

Use the Mac as the development machine, EC2 for quick GPU validation, and GH200
for the serious scaled experiments. Move source code only; do not copy `.venv/`,
`.venvs/`, `node_modules/`, or `artifacts/` between architectures.

The GH200 helpers default to:

| Variable | Default |
| --- | --- |
| `AGENTIC_GH200_USER` | `ojaiyeob` |
| `AGENTIC_GH200_HOST` | `gracehopper` |
| `AGENTIC_GH200_JUMP_HOST` | `falcon.7elements.com` |
| `AGENTIC_GH200_JUMP_PORT` | `1337` |
| `AGENTIC_GH200_REMOTE_DIR` | `/home/central/ojaiyeob/agentic_hardware` |

Override any of these before running the helper scripts if the GH200 login
changes.

### 1. Sync Source To GH200

Run from this local checkout:

```bash
./gh200/sync_to_gh200.sh --dry
./gh200/sync_to_gh200.sh
```

The sync excludes generated outputs, virtual environments, `node_modules`, git
metadata, and caches. It also protects the remote `sglang_direct_kv/artifacts/`
directory so GH200 experiment results are not overwritten by a source sync.

### 2. Enter GH200 And Build Host Dependencies

```bash
./gh200/ssh_to_gh200.sh

cd ~/agentic_hardware/sglang_direct_kv
INSTALL_SYSTEM_DEPS=0 bash scripts/setup_gh200.sh
```

Use `INSTALL_SYSTEM_DEPS=0` on the current GH200 image to avoid the known DKMS
package conflict. This builds fresh ARM64 Python environments directly on GH200.

Install ARM64 Node.js with `nvm` if needed:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install --lts
node -p "process.arch"   # expected: arm64
```

### 3. Smoke Test Harness Wiring

Run this on GH200 before long GPU jobs:

```bash
cd ~/agentic_hardware
./gh200/smoke_harnesses.sh
```

This no-GPU smoke test verifies the native client harnesses can reach the
inspection gateway path. It includes NAT and Hermes on the GH200 host.

### 4. Run GH200 GPU Experiments

Run GPU experiments inside `screen` so the job survives disconnects:

```bash
screen -S gh200_experiment
# detach:   Ctrl+A then D
# reattach: screen -r gh200_experiment
```

First run the sentinel:

```bash
cd ~/agentic_hardware
./gh200/run_sentinel.sh
```

Then run the EC2-scale apples-to-apples comparison on GH200:

```bash
./gh200/run_apples_to_apples.sh
```

Then run the GH200-scaled pressure ladder:

```bash
./gh200/run_scaled_pressure.sh
```

While a run is active, watch progress from another GH200 shell with:

```bash
tail -f ~/agentic_hardware/sglang_direct_kv/artifacts/results/run_logs/<REPORT_LABEL>.log
```

All three wrappers call
[`gh200/run_host_signal_design_space.sh`](gh200/run_host_signal_design_space.sh).
That script runs the experiment driver, gateway, and harness clients on the
GH200 host, but launches the SGLang GPU backend inside
`lmsysorg/sglang:latest`. Because the repo is mounted into the SGLang backend
container, source changes synced to GH200 are immediately visible inside
Docker; no Docker rebuild is needed.

The host-harness GH200 GPU runs default to:

```text
hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent
```

This split is what allows NAT and Hermes to participate in GH200 GPU runs:
their Python 3.11 venvs stay on the host, while only SGLang uses the Docker
CUDA/runtime environment.

The older
[`gh200/run_signal_design_space_docker.sh`](gh200/run_signal_design_space_docker.sh)
helper still exists for Docker-only debugging with Docker-compatible harnesses,
but the recommended GH200 path is the host-harness split above.

### 5. Download GH200 Reports

From the Mac:

```bash
./gh200/download.sh
```

This downloads only compact latest report artifacts by default. To download one
archived report folder:

```bash
./gh200/download.sh --label <REPORT_LABEL>
```

Avoid `--all` unless you intentionally want the full remote artifact tree,
because raw traces can become very large.

## Optional Prompt Encoding

A removable, request-local shorthand encoder is available as the independent
`agentic_prompt_codec` package. It is disabled by default. Dictionary shorthand
and a small relation grammar are token-counted with their legends included;
failed or unhelpful transformations pass through unchanged. Scheduling modes
remain independent of encoding.

The controller can also turn shorthand into an explicit experiment knob with
`controller_priority_demotion_admission_shorthand`. That mode keeps the current
priority + demotion + admission behavior, then asks the gateway to apply
dictionary shorthand only when the codec proves net token savings for the
configured scope. The encoder remains request-local and portable: it does not
require SGLang changes, harness changes, fine-tuning, or remembered dictionaries
across requests.

See [the prompt codec guide](sglang_direct_kv/docs/prompt_codec.md) for library
usage, the streaming proxy, token/quality evaluation, and an isolated EC2
pressure matrix. Do not launch the matrix while another GPU experiment is active.

## Core Modes

The current manager-facing comparisons use these modes:

| Mode | Meaning |
| --- | --- |
| `no_prefetch` | Baseline. The replay request receives no end-to-end priority treatment. |
| `e2e_priority_hints` | Current priority path. The driver marks replay urgency and the SGLang boundary carries that priority into scheduling. |
| `pre_harness_priority_hints` | Harness-preservation path. The driver marks replay urgency before the harness sees the request, then the gateway proves whether that intent or a native harness signal survived and was translated into SGLang priority. |
| `harness_emitted_signals` | Unified harness-originated path. If the harness emits priority, the gateway lowers it to SGLang priority. If the harness emits cache intent, the gateway lowers it to gateway speculative KV preload. |
| `no_cache_signal` | Cache-signal baseline. The gateway is present and records native harness cache fields, but it does not lower them to SGLang. |
| `harness_native_cache_lowered` | Harness-native cache path. The gateway translates only cache fields emitted by the harness itself. |
| `e2e_priority_hints_speculative_prefill` | Older direct backend probe for Dynamo-like proactive warmup. This is not part of the default consolidated gateway-injected family; keep it for targeted speculative KV preload tests. |
| `controller_full` | Combined controller path. The portable controller demotes filler/background work, raises replay priority, records admission decisions, restores background behavior, and ranks tied urgent replays by deadline. It intentionally excludes speculative preload in v1. |
| `controller_full_chunked_prefill` | Same full-controller decisions, but SGLang is launched with smaller prefill chunks so urgent replays get more scheduler boundaries between background chunks. Override with `CONTROLLER_CHUNKED_PREFILL_SIZE` and `CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS`. |
| `controller_observe_only` | Portable controller phase 1. The controller consumes lifecycle state and records the actions it would take, but does not mutate SGLang. |
| `controller_scheduler_priority` | Portable controller phase 2. The controller observes replay readiness and lowers only its ready-phase priority decision to SGLang scheduler priority. |
| `controller_speculative_preload` | Portable controller phase 3. The controller observes the tool-wait window and lowers an accepted KV prefetch decision to gateway speculative KV preload. |
| `controller_targeted_kv_prefetch` | Portable controller phase 4. The controller requests explicit target-prefix KV movement through a capability-gated SGLang adapter. If the active SGLang version exposes no stable direct hook, the report records that instead of using a warmup fallback. |
| `controller_demote_restore` | Portable controller phase 5. The controller lowers matching background/filler traffic during the replay-critical window, raises replay priority, and records restore/release afterward. |
| `controller_priority_demote` | Minimal controller probe. The controller does only two active things: lower filler/background requests to priority `-100` during the tool-wait window, and raise the target replay to priority `100`. |
| `controller_priority_demotion_admission` | Minimal three-action controller probe. The controller raises target replay priority, lowers filler/background priority, and temporarily holds filler/background admission during the replay-critical window. |
| `controller_priority_demotion_admission_earlyprepare` | Same priority + demotion + admission path, but it opens the background hold/demotion window before the replay returns. Configure the lead time with `CONTROLLER_EARLYPREPARE_LEAD_MS`; default is `500`. |
| `controller_priority_demotion_admission_shorthand` | Same priority + demotion + admission path, plus request-local dictionary shorthand. Configure with `CONTROLLER_SHORTHAND_CODEC_CONFIG`; default is `configs/prompt_codecs/dictionary_v1.json`. |
| `controller_admission_control` | Portable controller phase 6. The controller admits or skips speculative warmup based on pressure limits, so overload cases get explicit skip reasons instead of unbounded background work. |

The lightweight master report also includes a **System Cost Accounting** section.
It sums TTFT and positive replay-deadline debt separately for target replay
requests and filler/background requests. This is the tradeoff view: it shows
whether priority or controller modes reduced target replay misses by increasing
background/filler cost. When available, the chart compares Baseline,
Front-End Supplied, and Full Controller side by side, with net bars showing
whether each non-baseline mode saved or added total system cost. The CSV
artifact is:

```text
sglang_direct_kv/artifacts/results/latest_cost_accounting_summary.csv
```

When filler/background requests do not have a replay due timestamp, their TTFT
is still counted, but their replay-debt field is marked as unmeasured rather
than treated as a real missed deadline.

For production-like cost accounting, enable filler replay deadlines. In this
mode every filler/background session makes two calls: an initial pressure call,
then a resume/replay call after its own tool-wait deadline. The report can then
measure filler replay debt the same way it measures target replay debt.

```bash
cd sglang_direct_kv

FILLER_REPLAY_DEADLINES=1 \
bash scripts/run_harness_signal_design_space.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

If `FILLER_REPLAY_DEADLINE_MS` is omitted, the filler replay deadline uses the
same `tool_wait_ms` as the active pressure level.

### Production-like tool waits

By default, each task still uses the pressure level's fixed tool-wait gap. To
make the workload look more like real coding-agent traffic, enable sampled
tool waits and multiple resume cycles per task:

```bash
cd sglang_direct_kv

TOOL_WAIT_PROFILE=agentic_mixed \
TASK_REPLAY_STEPS=2 \
TOOL_WAIT_SEED=42 \
FILLER_REPLAY_DEADLINES=1 \
bash scripts/run_harness_signal_design_space.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

`TOOL_WAIT_PROFILE=agentic_mixed` samples each tool wait from this reproducible
mix: 70% quick waits at 200 ms, 25% moderate waits at 2 seconds, and 5% slow
waits at 20 seconds. `TASK_REPLAY_STEPS=2` means each target task can go through
two tool-wait/resume cycles instead of only one. When
`FILLER_REPLAY_DEADLINES=1` is also set, filler/background sessions use the same
kind of repeated resume structure, so the cost-accounting charts can measure
whether target replay gains are shifting delay onto background work.

For a custom distribution, set:

```bash
TOOL_WAIT_PROFILE_SPEC="quick:70:200,moderate:25:2000,slow:5:20000"
```

The sampler is deterministic for the same `TOOL_WAIT_SEED`, harness, pressure
level, mode, and session id. This keeps runs reproducible while still giving the
experiment richer timing structure.

## Portable Agent-Aware Controller Foundation

The controller prototype is intentionally backend-neutral. It lives in
`sglang_direct_kv/src/agentic_kv/controller/` and owns versioned lifecycle
events, per-session state, timing estimates, policy decisions, and backend
capability checks. SGLang-specific code should stay in a thin adapter/enforcer
layer so the controller can move across EC2, GH200, and newer SGLang releases.

Harnesses expose their controller-visible facts through the portable
`harness_controller_signal.v1` envelope. The envelope is carried in driver
metadata, controller-event metadata, gateway trace rows, and the forwarded
`nvext.agent_hints` payload. It separates harness facts from controller
decisions: the harness reports what it knows, the controller decides what to do,
and the gateway lowers accepted decisions to SGLang.

Current normalized signal buckets:

| Bucket | Examples |
| --- | --- |
| Task identity | `session_id`, `request_id`, `prefix_id`, `session_generation`, `task_index` |
| Agent phase | `phase.name`, `work_class`, `user_waiting`, `manager_visible`, `interactive` |
| Tool timing | `tool.name`, `tool.type`, `estimated_duration_ms`, `expected_done_at_ms`, `eta_uncertainty_ms` |
| Tool-wait profile | `profile`, `wait_class`, `duration_ms`, `step_index`, `total_steps` |
| Replay expectation | `likely`, `expected_count`, `deadline_after_tool_ms`, `expected_request_id` |
| Cache context | `stable_prefix`, `cache_key`, `reuse_scope`, `conversation_prefix_hash`, native cache signal source |
| Scheduling context | `urgency`, `priority`, `latency_sensitivity`, `safe_to_demote`, `preemptible`, `can_delay_ms` |
| Competition context | `active_background_requests`, `demotable_background_requests`, `background_safe_to_demote`, `background_can_delay_ms`, `concurrency` |
| Cost feedback | `recent_target_replay_debt_ms`, `recent_background_ttft_ms`, `recent_background_slowdown_ratio`, `allow_background_demote` |
| Cost/resource context | `expected_output_tokens`, `max_tokens`, `prompt_tokens`, cached/uncached token estimates |

The controller policy now uses this envelope for timing-aware full-controller
decisions. Short waits avoid early demotion and rely mostly on replay priority.
Medium and long waits enter a prepare window shortly before replay due time.
During that window, `controller_full` can demote safe background/filler work,
lower background prefill budget, and later restore normal background behavior.
Those decisions stay backend-neutral; the gateway lowers accepted commands into
the SGLang request fields.

Current foundation smoke test:

```bash
cd sglang_direct_kv
PYTHONPATH=src python3 scripts/smoke_agentic_controller.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The first implementation slice is observe-only by default. It records what the
controller would do when a session enters tool wait, becomes ready after tool
completion, or finishes, without mutating SGLang. Later phases should connect
the same command envelope to scheduler priority, background prefill budget,
gateway speculative KV preload, targeted KV prefetch, and eventually
demote/restore actions.

Small controller-family run:

```bash
cd sglang_direct_kv
SIGNAL_FAMILIES=controller_observe \
HARNESSES=hatcher \
PRESSURE_LEVELS=p0_control \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Small scheduler-priority controller run:

```bash
cd sglang_direct_kv
SIGNAL_FAMILIES=controller_scheduler \
HARNESSES=hatcher \
PRESSURE_LEVELS="p0_control p3_high" \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Small controller speculative-preload run:

```bash
cd sglang_direct_kv
SIGNAL_FAMILIES=controller_preload \
HARNESSES=hatcher \
PRESSURE_LEVELS="p1_mild p2_medium" \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Small targeted-prefetch controller run:

```bash
cd sglang_direct_kv
SIGNAL_FAMILIES=controller_targeted_prefetch \
HARNESSES=hatcher \
PRESSURE_LEVELS="p1_mild p2_medium" \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Small demote/restore controller run:

```bash
cd sglang_direct_kv
SIGNAL_FAMILIES=controller_demote_restore \
HARNESSES=hatcher \
PRESSURE_LEVELS="p1_mild p3_high" \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Small admission-control controller run:

```bash
cd sglang_direct_kv
SIGNAL_FAMILIES=controller_admission \
HARNESSES=hatcher \
PRESSURE_LEVELS="p1_mild p4_cliff" \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

EC2 controller repeatability run:

```bash
cd sglang_direct_kv
bash scripts/run_ec2_controller_repeatability.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

This repeatability run intentionally stays on the EC2 profile and uses the
DeepAgents/Hatcher harness only. It compares the useful controller paths against
baseline and direct gateway priority across `p1_mild`, `p3_high`, `p4_cliff`,
and `p5_boss_queue`. The purpose is to check whether the earlier controller
ordering result repeats before spending time on broader harness or GH200 runs.
It skips interim per-family report builds by default and writes the combined
lightweight report once at the end.

Current EC2 controller observation:

| Path | Current interpretation |
| --- | --- |
| Baseline | Slow under queue pressure because the replay enters as ordinary work. |
| Gateway-injected priority | Helps by raising replay scheduler priority at the SGLang boundary. |
| Controller scheduler priority | Helps when controller lifecycle state marks the replay-ready point and lowers that decision to SGLang priority. |
| Controller demote/restore | Helps by temporarily pushing matching filler/background work down while replay is critical, then restoring normal behavior. |
| Controller admission control | Helps by skipping speculative work when the system is already overloaded while still raising replay priority. |
| Full controller + chunked prefill | Keeps the same full-controller policy, but starts SGLang with chunked prefill so long background prefills are broken into smaller scheduling units. This should help most when filler work is still entering SGLang during the replay-critical window. |
| Controller speculative preload / targeted prefetch | Mechanically validated, but not in this repeatability run because prior EC2 timing showed little benefit and sometimes extra load. |

Current controller optimization target:

Use the combined `controller_full` mode before expanding the controller to more
harnesses. The purpose is to test the strongest controller policy on one known
harness first, instead of carrying weak or noisy modes into the broader harness
comparison.

`controller_full` should combine only the controller actions that have helped on
EC2 so far:

1. Track lifecycle state so the controller knows when a session is in tool wait,
   replay-ready, replay-submitted, and replay-finished.
2. Demote matching filler/background work shortly before or during the
   replay-critical window.
3. Raise replay scheduler priority when the replay becomes ready.
4. In multi-replay pressure such as `p5_boss_queue`, break urgent-request ties
   with a deadline-aware priority ladder instead of assigning every replay the
   same priority.
5. Use admission control to avoid speculative preload/prefetch when the system
   is already overloaded.
6. Restore background/filler priority after the replay-critical window closes.

Do not include speculative preload or targeted KV prefetch in the first
`controller_full` default. Those paths are mechanically useful but have not yet
shown consistent EC2 timing benefit; keeping them out prevents the full
controller from adding avoidable load.

Focused validation with explicit modes:

```bash
cd sglang_direct_kv
HARNESSES=hatcher \
PRESSURE_LEVELS="p1_mild p3_high p4_cliff p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints controller_scheduler_priority controller_demote_restore controller_admission_control controller_full" \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_deadline_pressure.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Minimal priority-plus-demotion controller probe:

```bash
cd sglang_direct_kv
HARDWARE_PROFILE=ec2_a10g \
HARNESSES=hatcher \
PRESSURE_LEVELS="p3_high" \
SIGNAL_FAMILIES="baseline frontend_supplied controller_priority_demote" \
P3_HIGH_KNOBS="tool_wait_ms=1000 target_prompt_tokens=4096 filler_sessions=32 filler_prompt_tokens=1536 session_count=1 concurrency=8" \
FILLER_REPLAY_DEADLINES=1 \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="priority_demote_p3_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Minimal priority-plus-demotion-plus-admission controller probe:

```bash
cd sglang_direct_kv
HARDWARE_PROFILE=ec2_a10g \
HARNESSES=hatcher \
PRESSURE_LEVELS="p3_high" \
SIGNAL_FAMILIES="baseline frontend_supplied controller_priority_demote controller_priority_demotion_admission" \
P3_HIGH_KNOBS="tool_wait_ms=1000 target_prompt_tokens=4096 filler_sessions=32 filler_prompt_tokens=1536 session_count=1 concurrency=8" \
FILLER_REPLAY_DEADLINES=1 \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="priority_demotion_admission_p3_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

EarlyPrepare controller probe:

```bash
cd sglang_direct_kv
HARDWARE_PROFILE=ec2_a10g \
HARNESSES=hatcher \
PRESSURE_LEVELS="p3_high" \
SIGNAL_FAMILIES="baseline frontend_supplied controller_priority_demotion_admission" \
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES="controller_priority_demotion_admission controller_priority_demotion_admission_earlyprepare" \
CONTROLLER_EARLYPREPARE_LEAD_MS=500 \
P3_HIGH_KNOBS="tool_wait_ms=1000 target_prompt_tokens=4096 filler_sessions=32 filler_prompt_tokens=1536 session_count=1 concurrency=8" \
FILLER_REPLAY_DEADLINES=1 \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="earlyprepare_p3_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Shorthand controller probe:

```bash
cd sglang_direct_kv
HARDWARE_PROFILE=ec2_a10g \
HARNESSES=hatcher \
PRESSURE_LEVELS="p3_high" \
SIGNAL_FAMILIES="baseline frontend_supplied controller_priority_demotion_admission controller_shorthand" \
CONTROLLER_PRIORITY_DEMOTION_ADMISSION_MODES="controller_priority_demotion_admission" \
CONTROLLER_SHORTHAND_MODES="controller_priority_demotion_admission_shorthand" \
CONTROLLER_SHORTHAND_CODEC_CONFIG="configs/prompt_codecs/dictionary_v1.json" \
CONTROLLER_SHORTHAND_ENCODING_SCOPE="target_requests" \
P3_HIGH_KNOBS="tool_wait_ms=1000 target_prompt_tokens=4096 filler_sessions=32 filler_prompt_tokens=1536 session_count=1 concurrency=8" \
FILLER_REPLAY_DEADLINES=1 \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="shorthand_controller_p3_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Success condition: `controller_priority_demotion_admission_shorthand` should
show the same controller proof as `controller_priority_demotion_admission`, plus
`gateway.prompt_encoding` events and `prompt_encoding_proof.csv` rows showing
whether shorthand was applied, skipped, or failed. Interpret latency only after
checking that net prompt tokens fell after including the shorthand legend.

Focused validation of the chunked-prefill controller variant:

```bash
cd sglang_direct_kv
HARNESSES=hatcher \
PRESSURE_LEVELS="p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints controller_full controller_full_chunked_prefill" \
CONTROLLER_CHUNKED_PREFILL_SIZE=512 \
CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS=4096 \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_deadline_pressure.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Equivalent consolidated selector:

```bash
cd sglang_direct_kv
HARNESSES=hatcher \
PRESSURE_LEVELS="p3_high p5_boss_queue" \
SIGNAL_FAMILIES="baseline gateway_injected controller_full controller_full_chunked" \
CONTROLLER_CHUNKED_PREFILL_SIZE=512 \
CONTROLLER_CHUNKED_MAX_PREFILL_TOKENS=4096 \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The same validation can also be run through the consolidated family selector:

```bash
cd sglang_direct_kv
HARNESSES=hatcher \
PRESSURE_LEVELS="p1_mild p3_high p4_cliff p5_boss_queue" \
SIGNAL_FAMILIES="baseline gateway_injected controller_scheduler controller_demote_restore controller_admission controller_full" \
REPORT_BUILDER_MODE=lightweight \
bash scripts/run_harness_signal_design_space.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

Success condition: `controller_full` should match or beat the best individual
controller mode at each pressure level, especially `p5_boss_queue`. The report
must also prove the mechanism: replay priority assigned, filler priority
demoted, filler priority restored, admission decision recorded, urgent replay
rank/order recorded, and first-token lateness improved or tied.

## Agent-Aware Controller Roadmap

Use this as the phase checklist for integrating the one-worker agentic
controller from the proposal. Each phase should land as a portable wrapper,
adapter, or policy layer first; SGLang-specific code should stay in the
smallest possible boundary adapter.

| Phase | Status | Goal | Proof Before Moving On |
| --- | --- | --- | --- |
| Phase 0: Capability and integration map | Started | Detect which SGLang/harness/backend features are available on the current machine and version. | Capability report records priority, cache, prefill, KV movement, and live trace support for EC2 and GH200. |
| Phase 1: Passive lifecycle controller | Validated on EC2 | Observe agent lifecycle events without changing scheduling or KV behavior. | `controller_observe_only` emitted tool-start, prepare-checkpoint, tool-complete, and session-finish decisions for Hatcher/DeepAgents and NAT at P0/P3; all backend results were observe-only. |
| Phase 2: Scheduler-only controller | Mechanically validated on EC2; outcome mixed | Convert controller decisions into scheduler priority only, with no speculative KV work yet. | Proof shows `tool_completed` produces `set_priority=100`, gateway source is `controller_ready_decision`, and SGLang scheduler receives the replay with priority `100`. A Hatcher/DeepAgents P3/P5 comparison showed P3 worse in one run and P5 median slightly better, so repeated samples are needed before claiming a performance win. |
| Phase 3: Gateway speculative KV preload | Mechanically validated on EC2; timing mixed | When a likely replay becomes predictable, send background preload/prefill work before the real replay arrives. | `controller_speculative_preload` accepts a controller `kv_action=prefetch` decision and lowers it to a gateway background warmup request. A Hatcher/DeepAgents P1/P2 validation showed controller warmup launch from the driver, warmup completion before SGLang received replay, and cached-prefix evidence on replay. P2 still missed the stricter warmup-before-deadline proof, so the next phase needs earlier prediction or admission control. |
| Phase 4: Targeted KV prefetch hook | Implemented as portable capability/proof scaffold | Add the thinnest possible backend hook for explicit host-to-device KV movement when SGLang exposes a stable path. | `controller_targeted_kv_prefetch` records controller prefetch request, backend acceptance, direct-hook availability, and any matching SGLang load-back or host-to-device copy before replay compute. If no stable direct hook exists, the evidence table says so explicitly. |
| Phase 5: Demote and restore | Validated on EC2 | Temporarily lower background/filler priority while preserving correctness and restoring normal priority afterward. | `controller_demote_restore` records a controller demote command during tool wait, lowers matching filler requests to background priority at the gateway boundary, raises the replay request, then records restore/release after replay. The EC2 P1 validation demoted 8/8 matching filler requests to priority `-100`, raised replay to priority `100`, and wrote `controller_demote_restore_proof.csv`. |
| Phase 6: Admission and overload control | Validated on EC2 | Decide when the system is too busy to accept more speculative work or urgent bursts. | `controller_admission_control` admits warmup only when the tool-wait window, filler count, concurrency, and per-case warmup budget stay under configured limits. The EC2 validation admitted P1 warmup and skipped P4 with explicit reasons: `tool_wait_ms 25 below minimum 75`, `filler_sessions 48 above limit 16`, and `concurrency 10 above limit 8`. Replay priority was still lowered to SGLang priority `100` in both cases. |
| Phase 6.5: Single-harness full-controller optimization | Implemented; EC2 performance rerun pending | Combine the EC2-winning pieces into `controller_full` and tune them on DeepAgents/Hatcher before expanding to other harnesses. | Unit tests prove timed prepare-window transition, short-wait no-demote behavior, metadata preservation, background demotion guardrails, replay priority, budget command, and release. Next EC2 run should check whether `controller_full` matches or beats the best individual controller mode across `p1_mild`, `p3_high`, `p4_cliff`, and `p5_boss_queue`. |
| Phase 6.6: Chunked-prefill full-controller scheduling | Implemented; EC2 performance run pending | Keep `controller_full` policy unchanged, but launch SGLang with smaller prefill chunks so the controller has more safe scheduling boundaries to insert urgent replay work. | Run `controller_full_chunked_prefill` against `controller_full` and `e2e_priority_hints` on DeepAgents/Hatcher at `p3_high` and `p5_boss_queue`; proof should show the mode, chunked-prefill launch knobs, controller traffic-reshape events, and replay lateness/TTFT deltas. |
| Phase 6.7: Minimal priority-plus-demotion probe | Implemented; EC2 run pending | Strip the controller back to the two highest-value actions: raise target replay priority and aggressively demote filler/background work. | Run `controller_priority_demote` against baseline and front-end priority on DeepAgents/Hatcher at `p3_high` with a 1000 ms fixed tool wait. Success requires proof that filler requests entering during the replay window were lowered to priority `-100`, the target replay reached SGLang with priority `100`, and the target replay improved without hiding filler cost. |
| Phase 6.8: Minimal priority-plus-demotion-plus-admission probe | Implemented; EC2 run pending | Add direct gateway admission control to the minimal controller: hold filler/background requests at the gateway boundary during the replay-critical window. | Run `controller_priority_demotion_admission` against baseline, front-end priority, and `controller_priority_demote` on DeepAgents/Hatcher at `p3_high` with a 1000 ms fixed tool wait. Success requires proof that filler/background requests were blocked before entering SGLang, released after target replay, and cost accounting shows where the delay moved. |
| Phase 6.9: EarlyPrepare admission window | Implemented; EC2 run pending | Start demotion/admission before the target replay returns, using the expected tool-wait completion time exposed by the harness. | Run `controller_priority_demotion_admission_earlyprepare` against baseline, front-end priority, and the prior admission modes on DeepAgents/Hatcher at `p3_high`. Success requires proof that the hold/demotion window opened before `m27.replay.due`, replay priority still reached SGLang, and target replay lateness moves closer to zero without excessive total TTFT cost. |
| Phase 7: GH200 profile and scale-up | Prepared; GH200 run pending | Re-run the same controller design on GH200 with larger pressure profiles and host-harness/Docker-SGLang split. | [gh200/run_controller_scaleup.sh](gh200/run_controller_scaleup.sh) runs the EC2-validated controller modes with `HARDWARE_PROFILE=gh200`, host-side harnesses, Dockerized SGLang, and the lightweight report builder. GH200 report should use the same scripts and modes as EC2, with only hardware profile and host/container setup differences. |

For Phase 6.5, the demote/restore proof is window-aware. Earlier filler
requests that entered before controller demotion are not counted as a demotion
failure. The proof table separately records `demote_trigger`,
`filler_requests_between_demote_and_replay`, `window_filler_demoted_count`, and
`window_filler_not_demoted_count` so the report can show whether the controller
actually shaped traffic during the replay-critical window.

The driver also emits explicit traffic-reshape trace signals during processing:
`m27.controller_traffic_reshape.window_open`,
`m27.controller_traffic_reshape.background_request_lowered`,
`m27.controller_traffic_reshape.target_replay_entering`, and
`m27.controller_traffic_reshape.window_close`. These are proof breadcrumbs, not
extra scheduling actions. They make it clear that the controller opened a
replay-critical window before the target replay arrived, lowered background
requests during that window, submitted the target replay with controller
priority, and then closed/restored the window.

Phase gate for each implementation slice:

1. Add or update a portable policy/wrapper first.
2. Add a smoke test or unit test that runs without a GPU.
3. Run a small P0/P3 validation before a full ladder.
4. Update the lightweight master report so the result is visible.
5. Update this README and push to `main`.

For NeMo Agent Toolkit / NAT, `pre_harness_priority_hints` uses NAT's OpenAI
provider pass-through path. The generated NAT workflow includes
`service_tier: priority` and `extra_body.agentic_hints.priority_class: urgent`;
LangChain emits that as `service_tier=priority` plus top-level
`agentic_hints.priority_class=urgent`. The gateway records those emitted fields
and translates them to SGLang `priority=100`.

NAT instrumentation is intentionally kept outside the installed NAT package.
The driver calls
[`sglang_direct_kv/scripts/nemo_agent_toolkit_wrapper.py`](sglang_direct_kv/scripts/nemo_agent_toolkit_wrapper.py),
which writes the NAT workflow config, launches `nat run`, records wrapper
lifecycle events, and lets the gateway prove the emitted priority-bearing HTTP
request. This keeps the test portable across NAT versions, EC2, and GH200.

For stronger NAT-side evidence, use the shared-service probe. It runs a real
`nat serve` process, sends older background requests into NAT first, then sends
an urgent request into the same NAT server. The report compares the order
requests entered NAT with the order NAT emitted model calls to the gateway.
This proves observable NAT service ordering without patching NAT internals.

This speculative prefill mode is not SGLang speculative decoding. It mimics
Dynamo's agent hint behavior: after the current turn/tool-call prefix is known,
send a small background prefill so the later replay can reuse warmed KV. Older
modes such as `dynamo_priority_hints`, `direct_prefetch`, and
`projected_hardware_bypass` are still useful for historical analysis, but they
are not the default comparison for the current pressure chart.

## Pressure Levels

Pressure levels bundle multiple knobs so experiments do not explode into a full
Cartesian sweep.

| Level | Name | Main Stressor | Typical Knobs |
| --- | --- | --- | --- |
| `p0_control` | Control | Easy baseline | `500 ms` tool wait, `1024` target prompt tokens, `0` fillers, `1` urgent agent |
| `p1_mild` | Mild pressure | Shorter tool wait, modest context | Small filler count, moderate prompt, one urgent replay |
| `p2_medium` | Medium pressure | More queue and KV pressure | Medium filler count, larger prompt, one urgent replay |
| `p3_high` | Queue pressure | One urgent replay behind older work | `50 ms` tool wait, `4096` target prompt tokens, `32` fillers, `1` urgent agent |
| `p4_cliff` | Deadline cliff | Short wait plus large KV and heavier backend pressure | Tight tool wait, large prompt, high filler pressure |
| `p5_boss_queue` | Boss queue | Many urgent replays compete at once | `50 ms` tool wait, `4096` target prompt tokens, fillers per session, multiple urgent agents |

Quick sentinel experiments often run only P0, P3, and P5 first. Full ladder
runs use P0 through P5.

## Hardware Profiles

Hardware profiles keep the experiment design the same while changing the amount
of pressure applied to the machine. The runner loads
`HARDWARE_PROFILE=ec2_a10g` by default.

| Profile | File | Use Case |
| --- | --- | --- |
| `ec2_a10g` | [sglang_direct_kv/configs/hardware/ec2_a10g.env](sglang_direct_kv/configs/hardware/ec2_a10g.env) | Current EC2/A10G-scale runs and apples-to-apples GH200 comparison. |
| `gh200` | [sglang_direct_kv/configs/hardware/gh200.env](sglang_direct_kv/configs/hardware/gh200.env) | GH200-scaled pressure run with larger token budget, larger HiCache, more fillers, and more urgent agents. |

Profile defaults can still be overridden inline:

```bash
HARDWARE_PROFILE=gh200 MAX_TOTAL_TOKENS=131072 \
bash scripts/run_native_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Recommended migration sequence for a new GH200 machine:

1. Run `HARDWARE_PROFILE=ec2_a10g` first. This gives an apples-to-apples
   comparison against the EC2 results.
2. Run `HARDWARE_PROFILE=gh200` next. This increases pressure so GH200 can find
   its own replay-deadline cliff.
3. Keep `REPORT_BUILDER_MODE=lightweight` for full multi-harness runs.

### GH200 Architecture Note

The current EC2 machine and a Grace Hopper 200 machine are not the same CPU
architecture.

| Machine | Expected CPU Architecture | What This Means |
| --- | --- | --- |
| Current EC2 GPU machine | `x86_64` | Uses normal x86 Linux packages and venv wheels. |
| GH200 | `aarch64` / `arm64` | Uses ARM64 packages and ARM64 Python/Node wheels. |

In simple terms: the repo code can move to GH200, but the installed
dependencies should not be copied from EC2. Rebuild the Python venvs, SGLang
environment, Node.js CLIs, NAT venv, and Hermes venv directly on GH200.

### GH200 Setup

After cloning or pulling this repository on the GH200 machine, rebuild the
Python environment directly on GH200:

```bash
cd ~/agentic_hardware/sglang_direct_kv

INSTALL_SYSTEM_DEPS=0 bash scripts/setup_gh200.sh
```

That script creates:

| Path | Purpose |
| --- | --- |
| `sglang_direct_kv/.venv` | Main project Python environment. Installs `requirements.txt`, editable `agentic-kv`, and analysis extras such as `scikit-learn`. |
| `$HOME/agentic_hardware/.venvs/nat_py311` | Isolated NeMo Agent Toolkit / NAT environment. |
| `$HOME/agentic_hardware/.venvs/hermes_agent_py311` | Isolated Hermes Agent environment. |

If you need a different Python binary:

```bash
PYTHON_BIN=python3.10 bash scripts/setup_gh200.sh
```

If the GH200 image already has system packages and CUDA configured:

```bash
INSTALL_SYSTEM_DEPS=0 bash scripts/setup_gh200.sh
```

If you want to add or change optional analysis packages:

```bash
EXTRA_PYTHON_PACKAGES="scikit-learn matplotlib seaborn pyarrow" \
bash scripts/setup_gh200.sh
```

The native CLI harnesses use Node.js through `npx`, so GH200 also needs an ARM64
Node.js install. If `setup_gh200.sh` says Node is missing, install Node LTS:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install --lts
node -p "process.arch"
```

The expected Node architecture on GH200 is:

```text
arm64
```

Before running experiments on GH200, check the machine:

```bash
uname -m
nvidia-smi
python3 --version
node -p "process.arch"
```

Expected GH200 architecture output:

```text
uname -m -> aarch64
node process.arch -> arm64
```

After installing dependencies on GH200, run a smoke test before the long
experiment:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/smoke_multi_harness_wireability.py \
  --harnesses codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent
```

## Most Recent Experiment: Real-Client Deadline Pressure

The newest experiment sends real coding-agent CLIs through the inspection
gateway, then forwards their requests to SGLang with normalized priority
metadata. This is stronger than the earlier adapter-only run because each
native CLI generates its own live request shape before our gateway normalizes
priority at the SGLang boundary.

Current archived native run:

```text
sglang_direct_kv/artifacts/results/reports/native_harness_deadline_pressure_backend_20260901_035640/master_report.html
```

Native client smoke status as of September 1, 2026:

```text
codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent
```

`nemo_agent_toolkit` and `hermes_agent` are installed in persistent isolated
Python 3.11 venvs on EC2 and should be passed into experiment runs with
`HARNESS_NAT_BIN` and `HARNESS_HERMES_BIN`. NAT needs the LangChain integration
extra, so install it as `nvidia-nat[langchain]`, not plain `nvidia-nat`.

Native-only run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="native_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_native_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Mixed native-plus-adapter run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent deepseek_harness" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="multi_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Pre-harness priority preservation smoke. This is the cleanest first run when
you want to see whether a harness preserves urgency before the request reaches
SGLang:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code" \
PRESSURE_LEVELS="p0_control p3_high" \
MODES="no_prefetch e2e_priority_hints pre_harness_priority_hints" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="pre_harness_priority_smoke_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

If a long EC2 run is interrupted after some cases finish, resume the same
report label without rerunning completed cases:

```bash
SKIP_EXISTING_CASES=1 \
HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent deepseek_harness" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="<existing_report_label>" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Headline result from the current native run. Values are median first-replay-token
lateness, so lower is better and anything above `0 ms` missed the replay
deadline:

| Harness | P0 no-prefetch | P0 E2E | P3 no-prefetch | P3 E2E | P5 no-prefetch | P5 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hatcher | `0.22 s` | `0.21 s` | `37.17 s` | `4.99 s` | `46.87 s` | `21.56 s` |
| Codex | `1.78 s` | `1.23 s` | `47.36 s` | `11.85 s` | `53.74 s` | `22.49 s` |
| Claude Code | `1.46 s` | `1.39 s` | `60.75 s` | `11.77 s` | `50.19 s` | `18.32 s` |
| OpenCode | `3.52 s` | `3.66 s` | `82.93 s` | `13.99 s` | `85.86 s` | `47.52 s` |
| Qwen Code | `5.86 s` | `5.69 s` | `76.81 s` | `19.90 s` | `95.11 s` | `67.11 s` |
| Pi Agent Harness | `1.42 s` | `1.30 s` | `57.52 s` | `11.77 s` | `52.60 s` | `18.72 s` |
| OpenClaw | `9.50 s` | `6.93 s` | `74.91 s` | `19.76 s` | `83.86 s` | `54.65 s` |
| NeMo Agent Toolkit / NAT | `3.62 s` | `3.62 s` | `67.66 s` | `10.57 s` | `50.88 s` | `19.73 s` |
| Hermes Agent | `12.67 s` | `12.44 s` | `72.94 s` | `19.63 s` | `67.45 s` | `40.29 s` |

Interpretation from this run: end-to-end priority hints helped every harness
under P3 queue pressure and helped every P5 boss-queue comparison, but the
priority path still missed the tight replay deadline under P3 and P5. Priority
can move a replay earlier in the queue; it cannot create extra GPU compute, KV
capacity, or host-to-device bandwidth. The P0 rows are also above zero because
real CLIs add startup/protocol overhead around the backend call. OpenCode's P3
and P5 no-prefetch rows use a backend decode-result fallback because the client
closed the stream before the gateway emitted `m27.request.end`; the raw proof
marks those rows with `first_token_source=scheduler_process_decode_result`.

## Multi-Harness Deadline Pressure

The broad experiment compares the same SGLang priority boundary across ten
non-Dynamo agent harness shapes. Eight entries now launch real native CLIs; the
DeepSeek Harness path remains adapter-backed until its native CLI exposes a
reliable headless command path.

| Harness | Current experiment path |
| --- | --- |
| `hatcher` | In-repo Hatcher / Deep Agents-style control harness. |
| `codex` | Real Codex CLI through the inspection gateway. |
| `claude_code` | Real Claude Code CLI through the inspection gateway. |
| `opencode` | Real OpenCode CLI through the inspection gateway. |
| `qwen_code` | Real Qwen Code CLI through the inspection gateway. |
| `pi_agent_harness` | Real Pi CLI with a generated OpenAI-compatible gateway provider extension. |
| `openclaw` | Real OpenClaw CLI with a generated OpenAI-compatible gateway provider config. |
| `nemo_agent_toolkit` | Real NAT CLI through a generated OpenAI-compatible workflow config. |
| `deepseek_harness` | Adapter-backed; native `dsh` CLI currently hangs during headless `--help` / `--version` probing. |
| `hermes_agent` | Real Hermes Agent CLI through a generated OpenAI-compatible provider config. |

It runs each harness in:

```text
no_prefetch
e2e_priority_hints
pre_harness_priority_hints
e2e_priority_hints_speculative_prefill
```

And uses the sentinel pressure levels:

```text
p0_control p3_high p5_boss_queue
```

Main run command:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent deepseek_harness" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints e2e_priority_hints_speculative_prefill" \
REPORT_LABEL="multi_harness_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

GH200 apples-to-apples run. Use this first after migrating the checkout to
GH200:

```bash
cd ~/agentic_hardware
./gh200/run_apples_to_apples.sh
```

GH200-scaled pressure run. Use this after the apples-to-apples run:

```bash
cd ~/agentic_hardware
./gh200/run_scaled_pressure.sh
```

GH200 controller scale-up run. Use this for Phase 7 after the sentinel passes:

```bash
cd ~/agentic_hardware
./gh200/run_controller_scaleup.sh
```

By default this uses `hatcher` as the internal DeepAgents-style control harness
and runs:

```text
baseline gateway_injected controller_scheduler controller_preload controller_targeted_prefetch controller_demote_restore controller_admission
```

with `HARDWARE_PROFILE=gh200`, all six pressure levels, and the lightweight
report builder. Override `HARNESSES`, `SIGNAL_FAMILIES`, or `PRESSURE_LEVELS`
before the command to broaden or narrow the run.

The lightweight report shows the Replay Deadline Pressure Chart as a
pressure-first overlay with three panels. Panel A measures full replay-deadline
lateness from replay due time to first token using a compressed symlog axis.
Panel B shows the same replay-deadline data with a normal linear y-axis. Panel C
shows replay TTFT, measured from replay request start at the gateway/client
boundary to first token. Pressure levels are the main x-axis sections, color
separates the signal path, and symbol shape separates harnesses.

For signal-design-space reports, the chart intentionally collapses lower-level
implementation modes into manager-facing signal buckets based on the actual
proof fields for each target replay request, not just the mode that was
attempted:

| Chart bucket | Meaning |
| --- | --- |
| `Baseline` | No signal was supplied or lowered for this replay request. Failed harness-emitted attempts fall back here. |
| `Harness Cache Emitted` | The target replay request carried a harness-emitted cache/prompt-cache signal; the gateway lowered it to speculative KV preload. |
| `Harness Priority Emitted` | The target replay request carried a harness-emitted priority/latency signal; the gateway lowered SGLang `priority`. |
| `Harness Cache + Priority Emitted` | The target replay request carried both cache/preload and priority signals. |
| `Front-End Supplied` | The experiment/front end supplied signal intent before the harness; the gateway translated what came through. |
| `Gateway Priority Injected` | The gateway attached SGLang priority at the backend boundary after the request left the harness. |

The raw lower-level modes remain in the evidence tables and CSV artifacts.

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh) | Orchestrates the multi-harness pressure experiment and writes the latest report. |
| [sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh) | Runs only the native CLI harnesses plus the Hatcher control. |
| [sglang_direct_kv/scripts/run_ec2_controller_repeatability.sh](sglang_direct_kv/scripts/run_ec2_controller_repeatability.sh) | EC2-focused controller repeatability run for DeepAgents/Hatcher across the useful controller paths. |
| [sglang_direct_kv/scripts/run_multi_harness_replay_driver.py](sglang_direct_kv/scripts/run_multi_harness_replay_driver.py) | Generates target replay and filler traffic for the selected harnesses. |
| [sglang_direct_kv/scripts/nemo_agent_toolkit_wrapper.py](sglang_direct_kv/scripts/nemo_agent_toolkit_wrapper.py) | Portable NAT wrapper that records config, process, and gateway-emission lifecycle events without patching NAT itself. |
| [sglang_direct_kv/scripts/run_nemo_nat_service_priority_probe.py](sglang_direct_kv/scripts/run_nemo_nat_service_priority_probe.py) | Runs real `nat serve` as one shared service and can also inspect NAT Dynamo-provider `nvext.agent_hints` without a full Dynamo runtime. |
| [sglang_direct_kv/scripts/harness_sglang_gateway.py](sglang_direct_kv/scripts/harness_sglang_gateway.py) | Normalizes harness requests at the SGLang boundary and injects priority metadata. |
| [sglang_direct_kv/configs/hardware/](sglang_direct_kv/configs/hardware/) | EC2 and GH200 pressure profiles. |
| [sglang_direct_kv/scripts/run_real_client_wireability_probe.py](sglang_direct_kv/scripts/run_real_client_wireability_probe.py) | Launches real client CLIs against the inspection gateway and reports the live request shape observed at the boundary. |
| [sglang_direct_kv/scripts/smoke_multi_harness_wireability.py](sglang_direct_kv/scripts/smoke_multi_harness_wireability.py) | Fast local smoke test for CLI harness wireability through the gateway. |
| [sglang_direct_kv/scripts/run_sglang_hicache_server.sh](sglang_direct_kv/scripts/run_sglang_hicache_server.sh) | Launches SGLang with HiCache, priority scheduling, and runtime telemetry flags. |
| [sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) | Builds the master HTML report, evidence tables, and Replay Deadline Pressure Chart. |
| [sglang_direct_kv/scripts/build_multi_harness_deadline_summary.py](sglang_direct_kv/scripts/build_multi_harness_deadline_summary.py) | Lightweight all-harness report builder used when the rich timeline report would be too large. |
| [sglang_direct_kv/src/agentic_kv/sglang_adapters/capabilities.py](sglang_direct_kv/src/agentic_kv/sglang_adapters/capabilities.py) | Probes the installed SGLang version, hook surface, priority support, and static cache-signal source paths. |
| [gh200/run_host_signal_design_space.sh](gh200/run_host_signal_design_space.sh) | Recommended GH200 runner: host-side harnesses and gateway, Dockerized SGLang backend. |
| [gh200/run_controller_scaleup.sh](gh200/run_controller_scaleup.sh) | Phase 7 runner: GH200-scaled controller comparison using the portable controller modes validated on EC2. |
| [gh200/run_signal_design_space_docker.sh](gh200/run_signal_design_space_docker.sh) | Docker-only fallback runner for debugging Docker-compatible harnesses. |

Smoke-test native CLI wireability without starting the real GPU server:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/smoke_multi_harness_wireability.py \
  --harnesses pi_agent_harness openclaw
```

Probe real client-generated traffic without running a full pressure sweep:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python scripts/run_real_client_wireability_probe.py \
  --clients codex claude_code opencode qwen_code pi_agent_harness openclaw \
  --out-dir artifacts/results/real_client_wireability/$(date +%Y%m%d_%H%M%S)
```

If `--target-base` is omitted, the probe uses a fake local SGLang-compatible
backend. Pass `--target-base http://127.0.0.1:30000` to observe real clients
through the gateway while forwarding to a running SGLang server.

For a stronger NAT-specific internal scheduling test, NAT also exposes
`nat serve`, which runs a workflow through the FastAPI front end. This command
uses that path and updates `artifacts/results/latest_master_report.html`:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
python scripts/run_nemo_nat_service_priority_probe.py \
  --report-label "nat_service_priority_probe_$(date +%Y%m%d_%H%M%S)" \
  --low-count 6 \
  --urgent-count 1 \
  --low-lead-ms 100 \
  --low-stagger-ms 5 \
  --fake-backend-delay-ms 1200 \
  --nat-workers 1 \
  --update-latest
```

Read `nat_service_priority_probe.csv` in the report folder. The key columns are
`submit_rank_into_nat`, `emit_rank_from_nat_to_gateway`,
`older_background_submitted_before`, `older_background_emitted_before`, and
`verdict`. If the urgent row has a lower emit rank than older background rows,
NAT emitted it earlier. If NAT also preserves structured priority fields, the
evidence is stronger; if priority is recovered only from the experiment marker,
the report says the priority cause is not proven.

To inspect NAT's Dynamo-provider hint format without installing the full Dynamo
runtime, run the same probe with NAT's Dynamo transport path:

```bash
cd ~/agentic_hardware/sglang_direct_kv

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
$HOME/agentic_hardware/.venvs/nat_py311/bin/python \
  scripts/run_nemo_nat_service_priority_probe.py \
  --report-label "nat_dynamo_direct_wireability_probe_$(date +%Y%m%d_%H%M%S)" \
  --nat-provider dynamo_direct \
  --low-count 2 \
  --urgent-count 1 \
  --low-lead-ms 100 \
  --low-stagger-ms 5 \
  --fake-backend-delay-ms 200 \
  --nat-workers 1 \
  --nvext-prefix-total-requests 10 \
  --nvext-prefix-osl 512 \
  --nvext-prefix-iat 50 \
  --update-latest
```

This is a lightweight wire-format probe. It imports NAT's Dynamo HTTP transport
and sends requests through our gateway/fake backend, so it does not require a
running Dynamo router. The expected proof is that background requests emit
`nvext.agent_hints.priority=2`, while urgent requests emit
`nvext.agent_hints.priority=100` plus prefix, OSL, IAT, total-request, and
cache-control hints. A real end-to-end Dynamo scheduling result still requires
the full NAT -> Dynamo -> SGLang stack, which is better suited for a larger
machine such as GH200.

To test NAT-only inferred priority, without putting `priority_intent` on the
frontend request and without installing Dynamo, run:

```bash
cd ~/agentic_hardware/sglang_direct_kv

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
$HOME/agentic_hardware/.venvs/nat_py311/bin/python \
  scripts/run_nemo_nat_service_priority_probe.py \
  --report-label "nat_inferred_priority_probe_$(date +%Y%m%d_%H%M%S)" \
  --nat-provider dynamo_inferred \
  --prompt-tokens 1024 \
  --fake-backend-delay-ms 100 \
  --nvext-prefix-total-requests 10 \
  --nvext-prefix-osl 512 \
  --nvext-prefix-iat 50 \
  --update-latest
```

This probe gives NAT a workflow prediction profile. The profile says some NAT
workflow nodes are low sensitivity and some are high sensitivity. The request
entering NAT does not say "urgent"; it only runs under a workflow path. NAT's
Dynamo transport looks up that workflow path, emits `nvext.agent_hints.priority`,
and our gateway translates that emitted value into the SGLang `priority` field.
The profile is saved beside the report as
`nat_inferred_priority_profile.json`, and it is also rendered as a table in the
HTML report. The proof lives in `nat_service_priority_probe.csv`; look for
`frontend_priority_intent_present=no`, `expected_inferred_priority`,
`emitted_nvext_priority`, `gateway_translated_priority`, and `verdict`.

To run the main replay-deadline pressure experiment for NAT only, compare the
baseline against NAT-inferred priority across all P0-P5 pressure levels:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARDWARE_PROFILE=ec2_a10g \
HARNESSES="nemo_agent_toolkit" \
PRESSURE_LEVELS="p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue" \
MODES="no_prefetch nat_inferred_priority_hints" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="nat_inferred_deadline_pressure_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

In this run, the gateway does not attach priority from the experiment marker.
For `nat_inferred_priority_hints`, NAT emits `nvext.agent_hints.priority` from
the workflow profile, and the gateway only translates that emitted value into
SGLang's `priority` field. The chart to inspect is the Replay Deadline Pressure
Chart; the raw proof fields are `harness_emit_priority_signal`,
`gateway_priority_translation_source`, and `sglang_priority`.

## Harness-Native Cache Signal Experiment

The compact entry point for the current signal design space is:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_NAT_BIN=$HOME/agentic_hardware/.venvs/nat_py311/bin/nat \
HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
SIGNAL_FAMILIES="baseline harness_emitted frontend_supplied gateway_injected" \
HARNESSES="hatcher codex claude_code opencode qwen_code pi_agent_harness openclaw nemo_agent_toolkit hermes_agent" \
PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
FILLER_REPLAY_DEADLINES=1 \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="signal_design_space_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_signal_design_space.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

The compact signal-family selector is the preferred interface for current
experiments. Keep the experiment surface to these three public knobs:
`HARNESSES`, `PRESSURE_LEVELS`, and `SIGNAL_FAMILIES`. For system-cost runs,
also set `FILLER_REPLAY_DEADLINES=1` so filler/background work has measurable
resume deadlines.

| `SIGNAL_FAMILIES` value | Meaning |
| --- | --- |
| `baseline` | No signal is supplied or lowered. |
| `harness_emitted` | The harness creates the signal. If it emits cache intent, the gateway lowers that to gateway speculative KV preload. If it emits priority, the gateway lowers that to SGLang priority. If it emits both, the gateway lowers both. |
| `frontend_supplied` | The experiment/front end gives signal intent to the harness first. The gateway translates whatever survives the harness path for SGLang. |
| `gateway_injected` | The request leaves the harness normally. The gateway then attaches only SGLang priority before forwarding to SGLang. |
| `all` | Alias for `baseline harness_emitted frontend_supplied gateway_injected`. |

The wrapper prints the detailed mode expansion before it starts. Today that
expands each family to one lower-level mode:

| Family piece | Lower-level modes |
| --- | --- |
| `baseline` | `no_prefetch` |
| `harness_emitted` | `harness_emitted_signals` |
| `frontend_supplied` | `pre_harness_priority_hints` |
| `gateway_injected` | `e2e_priority_hints` |

In the Replay Deadline Pressure Chart, these lower-level modes are collapsed
into signal-family colors: `Baseline`, `Harness Emitted`,
`Front-End Supplied`, and `Gateway Priority Injected`. A row appears as
`Harness Emitted` only when the target replay request actually carried a
harness-emitted cache or priority signal and the gateway lowered it. Use the
evidence tables when you need to know which exact signal was emitted and which
backend action was applied.

Use `DRY_RUN=1` to preview the expansion without launching SGLang:

```bash
DRY_RUN=1 \
SIGNAL_FAMILIES="baseline harness_emitted frontend_supplied gateway_injected" \
HARNESSES="qwen_code pi_agent_harness nemo_agent_toolkit" \
bash scripts/run_harness_signal_design_space.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Legacy detailed cache-salt runs are still available for reproducing earlier
experiments, but they are no longer the recommended manager-facing path. The
current path is the unified `harness_emitted` family above, where cache signals
lower to gateway speculative KV preload.

The legacy detailed cache-signal experiment asks a narrower question:

> Can the harness decide what cache signal to emit, while the gateway only
> translates that harness-emitted signal for SGLang?

The rule is:

```text
harness decides -> gateway translates -> SGLang executes
```

The gateway stays in the path for both modes, but it is not allowed to invent a
cache hint from hidden experiment knowledge such as replay phase, pressure
level, or prompt size.

| Mode | Meaning |
| --- | --- |
| `no_cache_signal` | Baseline. The gateway observes traffic and records whether a harness emitted cache fields, but it does not lower cache signals to SGLang. |
| `harness_native_cache_lowered` | The gateway translates only cache fields that the harness emitted, such as `cache_control`, `prompt_cache_key`, `promptCacheKey`, `cacheRetention`, or `prompt_cache_retention`. |

Harness-specific notes:

| Harness | Native cache signal path used by this testbed |
| --- | --- |
| `qwen_code` | Uses the OpenAI-compatible wire path for `no_cache_signal`, because that path does not emit explicit cache-control fields. Uses the Anthropic-compatible wire path for `harness_native_cache_lowered`, where Qwen emits `cache_control` markers on stable prompt sections. |
| `pi_agent_harness` | Uses Pi session persistence plus `PI_CACHE_RETENTION=long`. In native-cache mode Pi emits Anthropic-style `cache_control`, OpenAI-style `prompt_cache_key` / `prompt_cache_retention`, and session-affinity headers. |
| `openclaw` | Uses OpenClaw cache-retention environment toggles and provider compat flags. |
| `opencode` | Uses OpenCode's own emitted prompt-cache key when available. |

Recommended first run on EC2:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARDWARE_PROFILE=ec2_a10g \
HARNESSES="opencode qwen_code pi_agent_harness openclaw" \
PRESSURE_LEVELS="p0_control p2_medium p3_high p5_boss_queue" \
MODES="no_cache_signal harness_native_cache_lowered" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="harness_native_cache_signals_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

For a faster smoke test:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARDWARE_PROFILE=ec2_a10g \
HARNESSES="qwen_code pi_agent_harness" \
PRESSURE_LEVELS="p0_control" \
MODES="no_cache_signal harness_native_cache_lowered" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="qwen_pi_native_cache_fix_smoke_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_deadline_pressure.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Read `harness_native_cache_signal_proof.csv` in the report directory. The key
columns are `native_cache_signal_seen`, `native_cache_signal_source`,
`gateway_cache_lowered`, `gateway_cache_translation_source`, and
`gateway_invented_signal`. A clean positive result has
`native_cache_signal_seen=yes`, `gateway_cache_lowered=yes`,
`gateway_cache_translation_source=harness_emitted_cache_signal`, and
`gateway_invented_signal=false`.

The lightweight report also writes `cache_action_proof.csv`. This table stays
focused on the target replay requests in the chart, not filler traffic. It
checks whether the target replay showed SGLang cache-path evidence such as
prefix match, load-back, host-to-device copy, prefill attribution, or cache
commit events. A positive row means the cache signal reached the backend and
SGLang cache machinery ran for that replay. The low-level SGLang cache events
now also try to expose runtime namespace fields such as `extra_key`,
`cache_salt`, and `RadixKey.extra_key`; older traces may not contain these
columns. Without namespace evidence this is transport-plus-action proof, not
full causality proof. For causality, compare against `no_cache_signal` or run a
follow-up cache-disabled/random-cache-key A/B.

For a short current cache/preload probe, use P1/P2 and the unified
`harness_emitted` family. This run asks whether harness-emitted cache signals
trigger gateway speculative KV preload and improve backend TTFT, without P3
queue delay hiding the preload effect:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

HARNESS_HERMES_BIN=$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes \
HARDWARE_PROFILE=ec2_a10g \
SIGNAL_FAMILIES="baseline harness_emitted" \
HARNESSES="opencode qwen_code pi_agent_harness openclaw hermes_agent" \
PRESSURE_LEVELS="p1_mild p2_medium" \
REPORT_BUILDER_MODE=lightweight \
REPORT_LABEL="harness_emitted_preload_probe_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_harness_signal_design_space.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

The main outputs for this question are the Replay Deadline Pressure Chart, the
TTFT panel, `harness_native_cache_signal_proof.csv`,
`speculative_prefill_proof.csv`, and `cache_action_proof.csv`. Use these
together to check whether the target replay emitted a cache signal, whether the
gateway started speculative KV preload, whether the preload finished before the
replay, and whether TTFT moved versus baseline.

Legacy harness-native cache runs can still lower explicit harness cache keys
such as `prompt_cache_key`, `promptCacheKey`, or `cache_key` into SGLang's
native top-level `cache_salt` field. This keeps the gateway in translator mode:
it uses a key the harness emitted and does not invent a cache namespace from the
pressure level, prompt size, or replay phase.

The same report writes `sglang_cache_signal_path_audit.csv` when
`run_environment.json` is available. This static audit inspects the installed
SGLang package on the experiment machine and checks whether fields such as
`prompt_cache_key`, `cache_salt`, `custom_params.nvext.cache_control`,
`native_cache_bridge`, and `priority` appear to reach real SGLang code paths
such as `GenerateReqInput`, `Req.extra_key`, `RadixKey.extra_key`,
`match_prefix`, cache insertion, or priority-aware eviction.

Run the source-path audit directly on EC2 or GH200 with:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

python -m agentic_kv.sglang_adapters.capabilities \
  --out artifacts/results/sglang_capabilities.json \
  --out-md artifacts/results/sglang_capabilities.md
```

Latest EC2 wireability result:

```text
sglang_direct_kv/artifacts/results/real_client_wireability/real_client_probe_six_20260901_055102/real_client_wireability_report.html
```

That EC2 smoke, run on 2026-09-01, launched the real Codex, Claude Code,
OpenCode, Qwen Code, Pi, and OpenClaw CLIs against the inspection gateway. All
six reached the gateway, all six were tagged with `sglang_priority=100`, and the
gateway recorded their live request shape without storing prompt bodies:

| Client | API shape | Request body | Prompt chars |
| --- | --- | ---: | ---: |
| Codex | `/v1/responses` | `37.6 KB` | `4.3K` |
| Claude Code | `/v1/messages?beta=true` | `5.3 KB` | `1.6K` |
| OpenCode | `/v1/chat/completions` | `3.6 KB` | `3.2K` |
| Qwen Code | `/v1/chat/completions` | `97.8 KB` | `36.6K` |
| Pi Agent Harness | `/v1/chat/completions` | `1.5 KB` | `1.3K` |
| OpenClaw | `/v1/chat/completions` | `28.0 KB` | `19.9K` |

This is a wireability probe, not a pressure result. Its purpose is to prove
that real client-generated traffic can be inspected and priority-tagged at the
SGLang boundary before we run the heavier deadline-pressure ladder through real
client CLIs.

For large all-harness runs, [run_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_harness_deadline_pressure.sh)
automatically uses the lightweight summary report when the case count is large.
Set `REPORT_BUILDER_MODE=rich` to force the full timeline report for smaller
runs, or `REPORT_BUILDER_MODE=lightweight` to force the compact all-harness
report.

Previous archived adapter run:

```text
sglang_direct_kv/artifacts/results/reports/multi_harness_no_dynamo_20260831_232717/master_report.html
```

Headline result from that run. Values are median first-replay-token lateness,
so lower is better and anything above `0 ms` missed the replay deadline:

| Harness | P0 no-prefetch | P0 E2E | P3 no-prefetch | P3 E2E | P5 no-prefetch | P5 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hatcher | `221 ms` | `223 ms` | `33.0 s` | `9.2 s` | `50.4 s` | `26.5 s` |
| Codex | `1.4 s` | `1.3 s` | `39.9 s` | `10.5 s` | `15.7 s` | `7.7 s` |
| Claude Code | `1.3 s` | `1.3 s` | `68.9 s` | `11.9 s` | `52.2 s` | `27.9 s` |
| OpenCode | `3.6 s` | `3.4 s` | `80.6 s` | `15.7 s` | `81.3 s` | `41.3 s` |
| Qwen Code | `5.9 s` | `5.9 s` | `66.6 s` | `10.9 s` | `44.5 s` | `8.3 s` |
| NeMo Agent Toolkit / NAT | `8.7 s` | `9.0 s` | `74.2 s` | `12.9 s` | `53.8 s` | `30.6 s` |
| DeepSeek Harness | `8.8 s` | `9.3 s` | `74.0 s` | `12.9 s` | `53.2 s` | `28.2 s` |
| Pi Agent Harness | `9.0 s` | `8.9 s` | `74.6 s` | `12.8 s` | `52.9 s` | `29.4 s` |
| OpenClaw | `8.9 s` | `8.8 s` | `74.6 s` | `12.8 s` | `51.8 s` | `27.4 s` |
| Hermes Agent | `9.0 s` | `8.8 s` | `69.1 s` | `13.3 s` | `52.1 s` | `27.2 s` |

Interpretation: end-to-end priority hints help, especially under P3/P5
pressure, but they do not guarantee deadline readiness. Priority metadata can
move replay requests earlier in queues; it cannot by itself create more GPU
compute, KV capacity, or transfer bandwidth.

## Hatcher Pressure Ladder

This single-harness ladder focuses on the current Hatcher / Deep Agents-style
control harness and asks when replay deadlines start failing as pressure rises.

Full P0-P5 run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

PRESSURE_LEVELS="p0_control p1_mild p2_medium p3_high p4_cliff p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_LABEL="hatcher_pressure_ladder_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_hatcher_pressure_ladder.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Fast sentinel run:

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

PRESSURE_LEVELS="p0_control p3_high p5_boss_queue" \
MODES="no_prefetch e2e_priority_hints" \
REPORT_LABEL="hatcher_pressure_ladder_quick_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_hatcher_pressure_ladder_quick.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_hatcher_pressure_ladder.sh](sglang_direct_kv/scripts/run_hatcher_pressure_ladder.sh) | Full P0-P5 Hatcher pressure ladder. |
| [sglang_direct_kv/scripts/run_hatcher_pressure_ladder_quick.sh](sglang_direct_kv/scripts/run_hatcher_pressure_ladder_quick.sh) | Reduced P0/P3/P5 Hatcher ladder. |
| [sglang_direct_kv/scripts/run_real_prompt_controlled_replay.py](sglang_direct_kv/scripts/run_real_prompt_controlled_replay.py) | Core controlled replay workload generator. |
| [sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) | Shared report builder. |

## Priority Queue Proof

This pre-flight test proves that SGLang can admit a high-priority request ahead
of older low-priority work when priority scheduling is enabled.

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

RESULT_LABEL="priority_queue_jump_sanity_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_priority_queue_sanity.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_priority_queue_sanity.sh](sglang_direct_kv/scripts/run_priority_queue_sanity.sh) | Launches the priority proof experiment. |
| [sglang_direct_kv/scripts/run_priority_queue_jump_workload.py](sglang_direct_kv/scripts/run_priority_queue_jump_workload.py) | Creates older low-priority work followed by one high-priority replay request. |
| [sglang_direct_kv/scripts/summarize_priority_queue_sanity.py](sglang_direct_kv/scripts/summarize_priority_queue_sanity.py) | Summarizes whether the high-priority request jumped ahead. |

## Controlled Replay With Priority Modes

This controlled experiment family is useful when comparing individual scheduler
or replay modes without changing the harness.

```bash
cd ~/agentic_hardware/sglang_direct_kv
source .venv/bin/activate

AGENTIC_KV_TRACE_SCHEDULER=1 \
EXPERIMENT_KIND=controlled \
REPORT_LABEL="controlled_replay_priority_$(date +%Y%m%d_%H%M%S)" \
PRESSURE_PROFILE=custom \
UPDATE_LATEST=1 \
MAX_TIMELINE_GAPS=96 \
MAX_PAIRS=2 \
MODES="no_prefetch e2e_priority_hints" \
TOOL_WAIT_LIST_MS="50" \
FILLER_LIST="32" \
REQUEST_CONCURRENCY=8 \
FILLER_PROMPT_TOKENS=1536 \
MAX_TOTAL_TOKENS=12288 \
HICACHE_SIZE_GB=8 \
MEM_FRACTION_STATIC=0.72 \
bash scripts/run_milestone27_real_prompt_controlled_replay.sh \
  Qwen/Qwen2.5-Coder-7B-Instruct
```

Primary scripts:

| Script | Purpose |
| --- | --- |
| [sglang_direct_kv/scripts/run_milestone27_real_prompt_controlled_replay.sh](sglang_direct_kv/scripts/run_milestone27_real_prompt_controlled_replay.sh) | Runs real-prompt controlled replay experiments. |
| [sglang_direct_kv/scripts/run_real_prompt_controlled_replay.py](sglang_direct_kv/scripts/run_real_prompt_controlled_replay.py) | Workload driver. |
| [sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py](sglang_direct_kv/scripts/build_milestone27_controlled_replay_report.py) | Report builder. |

## Evidence And Reporting

The report contains presentation charts near the top and raw proof tables near
the bottom. Important evidence files include:

| Artifact | What It Shows |
| --- | --- |
| `global_kv_readiness_by_mode.csv` | Replay first-token lateness by mode, harness, and pressure level. |
| `harness_priority_preservation_proof.csv` | Pre-harness priority proof: driver intent, harness input signal, emitted harness signal, gateway translation, SGLang priority, and NAT wrapper lifecycle fields when NAT is used. |
| `nat_service_priority_probe.csv` | NAT shared-service and Dynamo-provider proof: request order into `nat serve` or NAT's Dynamo transport, emitted order to the gateway, older background work ahead, `nvext.agent_hints` when present, and whether urgent work overtook older requests. |
| `speculative_prefill_proof.csv` | Dynamo-like warmup proof: hint seen, background warmup timing, cached-prefix reuse, and replay timing. |
| `replay_queue_timing.csv` | Driver release, backend receive, scheduler admission, and first-token timing. |
| `dynamo_priority_queue_effectiveness.csv` | Priority queue behavior and older lower-priority work bypass evidence. |
| `request_state_snapshots.csv` | Timestamp-centered state for the target request and surrounding system pressure. |
| `runtime_telemetry_events.csv` | Live instrumentation events emitted while SGLang is running. |
| `kv_block_ledger.csv` | KV residency and movement evidence where available. |

The most important manager-facing chart is the **Replay Deadline Pressure
Chart**. Each dot is one replay request. Lower is better. Values above `0 ms`
missed the replay deadline; values below `0 ms` were early.

## Milestone Notebook

The full milestone history lives in
[sglang_direct_kv/README.md](sglang_direct_kv/README.md). Use it when you need
the detailed lab notebook, older commands, or historical context.

| Milestone Range | Focus |
| --- | --- |
| M0-M8 | Early SGLang setup, KV stress, and basic workload control. |
| M9-M15 | HiCache, real prompts, and first report automation. |
| M16-M27 | AgentBench/DeepAgents-style prompts and controlled replay experiments. |
| M28-M37 | Replay-path ledger, exact timing attribution, H2D evidence, live telemetry, and timestamp-centered reports. |
| M38-M40 | Priority queue proof, priority retention, and E2E priority sanity checks. |
| Current | Pressure ladder and multi-harness Replay Deadline Pressure Chart. |

## Harness Backlog

Native client status:

| Harness | Status | Next action |
| --- | --- | --- |
| Codex | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| Claude Code | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| OpenCode | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| Qwen Code | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| Pi Agent Harness | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| OpenClaw | Native CLI wired and smoke-tested. | Include in real-client pressure runs. |
| NeMo Agent Toolkit / NAT | Native CLI wired and smoke-tested with `$HOME/agentic_hardware/.venvs/nat_py311/bin/nat`. | Include in real-client pressure runs with `HARNESS_NAT_BIN` set. |
| DeepSeek Harness | Adapter-backed. | Re-probe after the `dsh` CLI exposes a reliable headless command path. |
| Hermes Agent | Native CLI wired and smoke-tested with `$HOME/agentic_hardware/.venvs/hermes_agent_py311/bin/hermes`. | Include in real-client pressure runs with `HARNESS_HERMES_BIN` set. |

The next recommended path is:

1. Run the P0/P3/P5 sentinel ladder across the eight real CLI harness paths plus the Hatcher control through [run_native_harness_deadline_pressure.sh](sglang_direct_kv/scripts/run_native_harness_deadline_pressure.sh).
2. Keep DeepSeek Harness adapter-backed until `dsh` can run a non-interactive smoke request without hanging.
3. Leave Dynamo out of local EC2 runs unless the host has enough spare CPU, memory, and disk for its runtime stack.

## Current Research Claim

Priority hints are useful: they express urgency and can improve scheduler
admission. But software priority alone does not guarantee replay-deadline
readiness under heavy GPU compute pressure, KV-cache pressure, short tool waits,
or many urgent agents arriving together.

That gap is the hardware/runtime opportunity: make replay-critical KV residency,
movement, and admission cheaper, deadline-aware, and enforceable.
