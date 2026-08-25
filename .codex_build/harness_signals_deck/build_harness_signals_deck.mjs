import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "/Users/oluwolejaiyeoba/Desktop/Harness Signal Support Comparison.pptx";
const TMP = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/.codex_build/harness_signals_deck";

const W = 1280;
const H = 720;
const C = {
  ink: "#111111",
  sub: "#4B5563",
  muted: "#777777",
  rule: "#B8BCC4",
  pale: "#F2F2F2",
  pale2: "#F8F8F8",
  blue: "#3D8DFF",
  sky: "#6DCBF4",
  green: "#5AB572",
  amber: "#E5A443",
  violet: "#8E73D8",
  gray: "#D5D9DF",
  none: "#FFFFFF",
};

const harnesses = ["Dynamo", "NAT", "Claude", "Codex", "DeepSeek", "Deep Agents", "Qwen"];

const status = {
  native: { label: "Native", fill: C.green },
  partial: { label: "Partial", fill: C.sky },
  pass: { label: "Pass-through", fill: C.blue },
  custom: { label: "Custom", fill: C.amber },
  dependent: { label: "Provider/internal", fill: C.violet },
  none: { label: "None", fill: C.none },
};

const categories = [
  {
    title: "Scheduling and workload prediction",
    short: "Scheduling",
    claim: "Only Dynamo and NAT expose serving-grade workload hints; most agent harnesses stop at provider selection or custom middleware.",
    heat: ["native", "native", "none", "partial", "none", "custom", "pass"],
    rows: [
      ["Soft priority", "Dynamo: nvext.agent_hints.priority; NAT: emits priority", "Deep Agents: custom middleware; Qwen: extra_body pass-through", "Claude, Codex, DeepSeek"],
      ["Strict priority tier", "Dynamo: strict_priority", "Deep Agents: custom only; Qwen: pass-through only", "NAT, Claude, Codex, DeepSeek"],
      ["Latency sensitivity", "NAT: experimental latency_sensitivity", "Dynamo: legacy field removed from current schema; Deep Agents/Qwen: custom/pass-through only", "Claude, Codex, DeepSeek"],
      ["Expected output length", "Dynamo: osl; NAT: emits osl", "Deep Agents: could infer internally; Qwen: pass-through only", "Claude, Codex, DeepSeek"],
      ["Expected interarrival time", "NAT: custom iat", "Qwen: pass-through only", "Dynamo, Claude, Codex, DeepSeek, Deep Agents"],
      ["Predicted remaining calls", "NAT: custom total_requests", "Deep Agents: workflow may know internally; Qwen: pass-through only", "Dynamo, Claude, Codex, DeepSeek"],
      ["Prefix/workflow reuse ID", "NAT: custom prefix_id; Deep Agents: thread_id", "Dynamo: session ID closest; Claude: session/agent IDs; Codex: thread/cache key; DeepSeek: session ID; Qwen: internal only", "None explicit"],
      ["Provider speed/QoS tier", "Codex: service_tier", "Claude: fast mode/speed; NAT/Deep Agents/Qwen: provider-dependent or extra_body", "Dynamo, DeepSeek"],
    ],
  },
  {
    title: "Reasoning and inference compute",
    short: "Reasoning",
    claim: "Reasoning effort is a first-class model control in coding harnesses, while serving systems tend to expose token and sampling controls.",
    heat: ["partial", "dependent", "native", "native", "native", "dependent", "native"],
    rows: [
      ["Reasoning effort level", "Claude: output_config.effort; Codex: reasoning.effort; DeepSeek: reasoningEffort -> reasoning_effort; Qwen: model.reasoningEffort", "NAT/Deep Agents: provider setting; Dynamo: no abstract level", "None explicit"],
      ["Supported effort values", "Claude: low, medium, high, sometimes xhigh/max; Codex: minimal, low, medium, high, xhigh; DeepSeek: off, low, high, max; Qwen: low, medium, high, xhigh, max", "NAT/Deep Agents: provider-dependent", "Dynamo"],
      ["Thinking enable/disable", "Claude: thinking.type; DeepSeek: thinking.type; Qwen: enable_thinking", "Codex: mainly controlled through effort; NAT/Deep Agents: provider-dependent", "Dynamo"],
      ["Thinking-token budget", "Dynamo: max_thinking_tokens; Claude: budget_tokens where applicable; Qwen: thinking_budget / budget_tokens", "Codex: reasoning configuration; DeepSeek: primarily effort-based; NAT/Deep Agents: provider-dependent", "None explicit"],
      ["Maximum output tokens", "Dynamo/Claude: max_tokens; DeepSeek: maxTokens; Qwen: max_tokens or max_completion_tokens", "Codex: API-level control; NAT/Deep Agents: model/provider control", "None explicit"],
      ["Sampling controls", "Dynamo: OpenAI-compatible controls plus greed_sampling; DeepSeek: temperature, stop; Qwen: samplingParams", "Claude/Codex/NAT/Deep Agents: provider/model-dependent", "None explicit"],
      ["Output verbosity", "Codex: text.verbosity", "Claude: effort/output configuration; NAT/Deep Agents/Qwen: provider-dependent", "Dynamo, DeepSeek"],
    ],
  },
  {
    title: "Cache management",
    short: "Cache",
    claim: "Cache support exists nearly everywhere, but client-controllable cache policy is uneven and often provider-managed.",
    heat: ["native", "partial", "native", "partial", "dependent", "dependent", "dependent"],
    rows: [
      ["Cache key/bucket", "Codex: prompt_cache_key", "Dynamo/Claude/DeepSeek/Qwen: automatic prefix/hash matching; NAT: prefix_id is not equivalent; Deep Agents: provider middleware", "None explicit"],
      ["Cache namespace/isolation", "Dynamo: nvext.cache_salt or x-tenant-id", "Codex: key bucketing, not tenant isolation; NAT can pass salt; Deep Agents/Qwen provider-dependent", "Claude, DeepSeek"],
      ["Cache TTL", "Claude: cache_control, 5-minute or 1-hour TTL; Qwen: cacheRetention/provider cache control", "NAT: experimental nvext.cache_control.ttl; DeepSeek/Deep Agents: provider-managed; Codex: no explicit TTL", "Dynamo public schema"],
      ["Cache entry type", "Claude: type:\"ephemeral\"", "NAT: type:\"ephemeral\" experimentally; Deep Agents/Qwen provider-dependent", "Dynamo, Codex, DeepSeek"],
      ["Eviction priority", "Dynamo: priority with priority-aware SGLang cache; NAT: emits priority", "Deep Agents: custom; Qwen: pass-through only", "Claude, Codex, DeepSeek"],
      ["Speculative prefill", "Dynamo: speculative_prefill", "Codex: WebSocket prewarm, different mechanism; Qwen: UI speculation, not KV prefill", "NAT adapter, Claude, DeepSeek, Deep Agents"],
      ["Cache-hit feedback", "Dynamo: metrics/response extensions; Claude: cache-read and creation usage; Codex: cached input-token usage; DeepSeek: cacheReadTokens; Qwen: cache-hit metrics", "NAT: profiler/metrics; Deep Agents: provider trace data", "None explicit"],
      ["Cache pinning", "Claude: provider-managed TTL retention", "NAT: experimental/version-bound; DeepSeek/Deep Agents/Qwen: provider-managed/dependent", "Dynamo public feature, Codex"],
    ],
  },
  {
    title: "Identity and hierarchy",
    short: "Identity",
    claim: "Session identity is common, but parent-child hierarchy and stable turn IDs are much less consistently exposed.",
    heat: ["native", "partial", "native", "native", "partial", "native", "dependent"],
    rows: [
      ["Session ID", "Dynamo: X-Dynamo-Session-ID; Claude: x-claude-code-session-id; Codex: thread-id / metadata session_id/thread_id; DeepSeek: x-deepseek-harness-session-id; Deep Agents: thread_id", "NAT: workflow-derived prefix context; Qwen: internal session ID", "None explicit"],
      ["Parent session", "Dynamo: X-Dynamo-Parent-Session-ID; Claude: x-claude-code-parent-agent-id; Codex: x-codex-parent-thread-id", "NAT: call-stack-derived prefix; Deep Agents: parent run tracing", "DeepSeek, Qwen"],
      ["Agent/subagent ID", "Claude: x-claude-code-agent-id; Deep Agents: ls_agent_type=\"subagent\"", "Dynamo normalizes supported harness headers; NAT prefix depth/run context; Codex subagent metadata; Qwen internal state", "DeepSeek"],
      ["Turn ID", "Codex: metadata turn_id", "Dynamo: tracing/session context; NAT: profiler call index; DeepSeek: conversation state; Deep Agents: run/checkpoint identity; Qwen: internal", "Claude stable documented header"],
      ["Root/parent turn hierarchy", "Codex: root_turn_id, parent_turn_id", "Dynamo consumer-defined; NAT workflow context; Claude agent hierarchy; Deep Agents trace hierarchy", "DeepSeek, Qwen"],
      ["Anonymous user/installation", "Codex: x-codex-installation-id; DeepSeek: x-deepseek-harness-user-id", "Dynamo tenant/session headers; Claude provider/account context; Deep Agents runtime context; Qwen trace/client context", "NAT central field"],
      ["W3C trace context", "Qwen: traceparent and baggage optional", "Dynamo tracing; NAT profiler/tracing; Claude gateway/provider tracing; Codex OpenTelemetry metadata; DeepSeek request IDs; Deep Agents LangSmith/runtime tracing", "None explicit"],
    ],
  },
  {
    title: "Lifecycle and request purpose",
    short: "Lifecycle",
    claim: "Codex and DeepSeek expose purpose-style request markers; most other systems infer lifecycle from workflow state.",
    heat: ["partial", "dependent", "dependent", "native", "native", "native", "dependent"],
    rows: [
      ["Normal agent turn", "Codex: request_kind:\"turn\"", "Dynamo normal request; NAT normal workflow call; Claude normal conversation request; DeepSeek normal purpose; Deep Agents graph invocation; Qwen normal request", "None explicit"],
      ["Compaction", "Codex: request_kind:\"compaction\" plus metadata; DeepSeek: purpose:\"compaction\" and x-deepseek-harness-compact:1", "Dynamo observes normalized Codex metadata; NAT workflow-dependent; Claude internal traffic; Deep Agents summarization middleware; Qwen compactionModel internally", "None explicit"],
      ["Prewarm", "Dynamo: speculative_prefill; Codex: request_kind:\"prewarm\" / WebSocket prewarm", "Claude provider cache warming; Qwen UI speculation only", "NAT, DeepSeek, Deep Agents"],
      ["Memory request", "Codex: request_kind:\"memory\"", "NAT workflow-dependent; Claude provider-specific/internal; Deep Agents state/checkpoint operations; Qwen internal", "Dynamo, DeepSeek"],
      ["Session completion", "Dynamo: X-Dynamo-Session-Final", "Deep Agents graph/run completion", "NAT, Claude, Codex, DeepSeek, Qwen"],
      ["Session-title request", "DeepSeek: purpose:\"session-title\"", "Claude/Deep Agents/Qwen internal; Codex potential metadata/custom request kind", "Dynamo, NAT"],
      ["Subagent creation", "Deep Agents: native subagent runtime", "Dynamo parent/session hierarchy; NAT workflow stack depth; Claude agent headers; Codex subagent headers/metadata; Qwen internal agent behavior", "DeepSeek"],
    ],
  },
  {
    title: "Routing and placement",
    short: "Routing",
    claim: "Dynamo is the only harness here with native hard placement controls for backend, worker, and data-parallel rank.",
    heat: ["native", "pass", "partial", "partial", "partial", "custom", "pass"],
    rows: [
      ["Direct backend selection", "Dynamo: backend_instance_id", "NAT could pass through; Qwen static extra_body; Deep Agents custom middleware", "Claude, Codex, DeepSeek"],
      ["Prefill worker selection", "Dynamo: prefill_worker_id", "NAT could pass through; Deep Agents custom; Qwen pass-through", "Claude, Codex, DeepSeek"],
      ["Decode worker selection", "Dynamo: decode_worker_id", "NAT could pass through; Deep Agents custom; Qwen pass-through", "Claude, Codex, DeepSeek"],
      ["Data-parallel rank", "Dynamo: dp_rank, prefill_dp_rank", "NAT could pass through; Deep Agents custom; Qwen pass-through", "Claude, Codex, DeepSeek"],
      ["Sticky routing", "Dynamo: optional session affinity", "NAT custom processor/prefix; Claude session ID only; Codex x-codex-turn-state on OpenAI path; DeepSeek session ID only; Deep Agents checkpointing, not inference placement", "Qwen"],
      ["Provider routing hint", "Dynamo: native router policies", "NAT custom processor; Claude model/fast selection; Codex x-codex-routing-hint/OpenAI path; DeepSeek/Qwen provider/model selection; Deep Agents model selection", "None explicit"],
      ["Cache-aware placement", "Dynamo: native KV router", "NAT Dynamo integration; Claude/Codex/DeepSeek provider-managed; Deep Agents depends on inference provider; Qwen depends on provider", "None explicit"],
    ],
  },
  {
    title: "Observability and response",
    short: "Observability",
    claim: "Observability is widely available, but the deepest engine-level fields remain concentrated in Dynamo-style serving responses.",
    heat: ["native", "native", "dependent", "dependent", "partial", "dependent", "dependent"],
    rows: [
      ["Worker identity", "Dynamo: extra_fields[\"worker_id\"]", "NAT: Dynamo response; Deep Agents provider trace; Codex/Qwen provider/internal", "Claude normally, DeepSeek"],
      ["TTFT/ITL/queue time", "Dynamo: extra_fields[\"timing\"]", "NAT profiler metrics; Claude provider telemetry; Codex timing-metrics header; DeepSeek stream timing; Deep Agents LangSmith tracing; Qwen provider/trace metrics", "None explicit"],
      ["Cache read tokens", "DeepSeek: cacheReadTokens", "Dynamo metrics/backend response; NAT profiler; Claude Anthropic usage; Codex cached-token usage; Deep Agents provider trace; Qwen cache metrics", "None explicit"],
      ["Routed experts", "Dynamo: routed_experts", "NAT: Dynamo response; Deep Agents/Qwen provider-dependent", "Claude, Codex, DeepSeek"],
      ["Engine metadata", "Dynamo: engine_data", "NAT: Dynamo response; Codex provider metadata; DeepSeek provider request IDs; Deep Agents provider trace; Qwen provider-dependent", "Claude"],
      ["Stop reason", "Dynamo: stop_reason", "NAT/model response; Claude/Codex provider response; DeepSeek finish reason; Deep Agents model response; Qwen provider response", "None explicit"],
      ["Completion token IDs", "Dynamo: completion_token_ids", "NAT: Dynamo response; Codex/Deep Agents/Qwen provider-dependent", "Claude normally, DeepSeek"],
      ["Request IDs", "DeepSeek: x-request-id and x-deepseek-request-id", "Dynamo tracing; NAT workflow/profiler; Claude provider request IDs; Codex response IDs/turn state; Deep Agents trace IDs; Qwen W3C trace IDs", "None explicit"],
    ],
  },
];

const fieldExamples = [
  ["Dynamo", "priority", "strict_priority", "osl", "speculative_prefill", "backend_instance_id", "worker_id"],
  ["NAT", "priority", "osl", "latency_sensitivity", "iat", "total_requests", "prefix_id"],
  ["Claude", "output_config.effort", "thinking.type", "budget_tokens", "cache_control", "session-id", "max_tokens"],
  ["Codex", "reasoning.effort", "service_tier", "prompt_cache_key", "request_kind", "turn_id", "text.verbosity"],
  ["DeepSeek", "reasoningEffort", "thinking.type", "maxTokens", "cacheReadTokens", "purpose", "request-id"],
  ["Deep Agents", "thread_id", "subagents", "middleware", "LangSmith traces", "checkpointing", "interrupts"],
  ["Qwen", "model.reasoningEffort", "enable_thinking", "samplingParams", "cacheRetention", "extra_body", "traceparent"],
];

const summaries = {
  Scheduling: {
    headline: "Scheduling belongs on the serving plane",
    native: ["Dynamo: priority, strict_priority, OSL", "NAT: emits priority/OSL and experimental latency_sensitivity"],
    gaps: ["Claude, Codex, and DeepSeek have no soft-priority workload signal in the screenshot", "IAT and remaining-call prediction are NAT-specific/custom"],
    design: "Show scheduling as workload prediction, not as generic model configuration.",
  },
  Reasoning: {
    headline: "Reasoning effort is strongest in coding harnesses",
    native: ["Claude, Codex, DeepSeek, and Qwen expose explicit effort controls", "Dynamo exposes token/sampling caps rather than abstract reasoning effort"],
    gaps: ["Thinking enable/disable and budgets remain provider-dependent in NAT and Deep Agents", "Output verbosity is explicit in Codex; elsewhere it is model/provider-specific"],
    design: "Separate reasoning effort from max output tokens; they answer different control questions.",
  },
  Cache: {
    headline: "Cache policy control is fragmented",
    native: ["Claude exposes cache_control with TTL; Codex exposes prompt_cache_key", "Dynamo exposes cache salt, priority-aware eviction, and speculative_prefill"],
    gaps: ["Dynamo has no current public cache TTL field in the screenshot", "Pinning and TTL are often provider-managed rather than harness-owned"],
    design: "Use policy layers: key, isolation, TTL, entry type, eviction, prefill, feedback.",
  },
  Identity: {
    headline: "Session IDs are common; hierarchy is not.",
    native: ["Dynamo, Claude, Codex, DeepSeek, and Deep Agents expose session identity", "Codex is explicit about turn/root/parent turn metadata"],
    gaps: ["DeepSeek and Qwen show no parent-session signal", "Turn IDs and trace hierarchy are often internal, derived, or runtime-specific"],
    design: "Represent identity as a stack: installation, session, parent, agent, turn, trace.",
  },
  Observability: {
    headline: "Dynamo carries the deepest engine response fields",
    native: ["Dynamo exposes worker_id, timing, routed_experts, engine_data, stop_reason, token IDs", "DeepSeek exposes cacheReadTokens and request IDs"],
    gaps: ["Claude usually surfaces provider telemetry, not worker identity or token IDs", "Deep Agents and Qwen are mostly provider/runtime trace dependent"],
    design: "Model observability as response-envelope fields, not as runtime logs alone.",
  },
};

function addSlide(p, title, eyebrow = "Harness signal support") {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  text(slide, eyebrow, 44, 34, 520, 24, { size: 14, color: C.muted, bold: true });
  text(slide, title, 44, 68, 1050, 56, { size: 36, bold: true, color: C.ink });
  line(slide, 44, 134, 1192, C.rule);
  return slide;
}

function text(slide, value, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontSize: opts.size ?? 18,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    typeface: opts.font ?? "Helvetica Neue",
    autoFit: opts.autoFit ?? "shrinkText",
    insets: opts.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function rect(slide, left, top, width, height, fill, opts = {}) {
  return slide.shapes.add({
    geometry: opts.round ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: opts.line ?? "none", width: opts.lineWidth ?? 0 },
  });
}

function line(slide, left, top, width, color = C.rule, weight = 1) {
  return slide.shapes.add({
    geometry: "straightConnector1",
    position: { left, top, width, height: 0 },
    fill: "none",
    line: { style: "solid", fill: color, width: weight },
  });
}

function cell(slide, value, left, top, width, height, opts = {}) {
  rect(slide, left, top, width, height, opts.fill ?? C.none, { line: opts.line ?? "#E5E7EB", lineWidth: 1 });
  text(slide, value, left + 7, top + 6, width - 14, height - 10, {
    size: opts.size ?? 15.5,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    valign: "top",
  });
}

function addFooter(slide, idx, label = "Source: user-provided screenshots; None means unsupported") {
  line(slide, 44, 666, 1192, "#E5E7EB");
  text(slide, label, 44, 676, 860, 18, { size: 11.5, color: C.muted });
  text(slide, String(idx), 1190, 676, 46, 18, { size: 11.5, color: C.muted, align: "right" });
  slide.speakerNotes.textFrame.setText(`[Sources]\n- User-provided screenshots: signal comparison matrices across Dynamo, NeMo Agent Toolkit, Claude Code, Codex, DeepSeek Harness, Deep Agents, and Qwen Code.\n`);
}

function drawLegend(slide, left, top) {
  Object.entries(status).forEach(([key, s], i) => {
    const x = left + i * 175;
    rect(slide, x, top, 18, 18, s.fill, { line: key === "none" ? C.rule : "none", lineWidth: key === "none" ? 1 : 0 });
    text(slide, s.label, x + 26, top - 1, 140, 22, { size: 14, color: C.sub });
  });
}

function wrapList(items) {
  return items.map((x) => `- ${x}`).join("\n");
}

function buildTitle(p) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  text(slide, "LLM inference harness signals", 44, 186, 900, 120, { size: 62, bold: true });
  text(slide, "Comparison across Dynamo, NeMo Agent Toolkit, Claude Code, Codex, DeepSeek Harness, Deep Agents, and Qwen Code", 48, 330, 760, 86, { size: 25, color: C.sub });
  rect(slide, 924, 174, 260, 260, C.pale, { line: "none" });
  ["priority", "reasoning", "cache", "routing", "trace"].forEach((t, i) => {
    const y = 204 + i * 42;
    rect(slide, 954, y, 190, 28, i % 2 === 0 ? "#FFFFFF" : "#E8F4FB", { line: "#D7DCE2", lineWidth: 1 });
    text(slide, t, 968, y + 5, 160, 18, { size: 15, bold: true });
  });
  text(slide, "Built from the attached comparison screenshots", 48, 592, 580, 24, { size: 16, color: C.muted });
  addFooter(slide, 1);
}

function buildTaxonomy(p) {
  const slide = addSlide(p, "The signals fall into seven operating layers");
  const intro = "The useful slide structure is not one giant table. It is a layered model: decisions first, exhaustive signal detail second.";
  text(slide, intro, 52, 156, 900, 50, { size: 23, color: C.sub });
  const cols = [
    ["Serving plane", ["Scheduling", "Cache", "Routing"]],
    ["Agent/runtime plane", ["Reasoning", "Identity", "Lifecycle"]],
    ["Feedback plane", ["Observability"]],
  ];
  cols.forEach((col, i) => {
    const x = 70 + i * 400;
    text(slide, col[0], x, 250, 310, 34, { size: 25, bold: true });
    line(slide, x, 292, 300, i === 0 ? C.green : i === 1 ? C.blue : C.amber, 4);
    col[1].forEach((item, j) => {
      text(slide, item, x, 326 + j * 58, 310, 34, { size: 28, bold: true });
    });
  });
  text(slide, "Use this taxonomy in the main deck; use the full matrices only as reference evidence.", 70, 592, 930, 30, { size: 21, bold: true });
  addFooter(slide, 2);
}

function buildHeatmap(p) {
  const slide = addSlide(p, "Coverage is strongest at the serving edge and weakest for hard placement");
  drawLegend(slide, 56, 150);
  const left = 56, top = 198, rowH = 52, colW = 133;
  cell(slide, "Signal layer", left, top, 190, 42, { fill: C.pale, bold: true, size: 16 });
  harnesses.forEach((h, i) => cell(slide, h, left + 190 + i * colW, top, colW, 42, { fill: C.pale, bold: true, size: 14 }));
  categories.forEach((cat, r) => {
    cell(slide, cat.short, left, top + 42 + r * rowH, 190, rowH, { fill: r % 2 ? "#FBFBFB" : "#FFFFFF", bold: true, size: 15 });
    cat.heat.forEach((key, c) => {
      const s = status[key];
      rect(slide, left + 190 + c * colW, top + 42 + r * rowH, colW, rowH, s.fill, { line: "#E5E7EB", lineWidth: 1 });
      text(slide, s.label, left + 198 + c * colW, top + 58 + r * rowH, colW - 16, 20, { size: 12.5, align: "center", color: key === "none" ? C.muted : "#111111" });
    });
  });
  text(slide, "Read across: green means first-class support; white means the screenshots marked the signal as None/unsupported.", 56, 604, 1000, 24, { size: 18, color: C.sub });
  addFooter(slide, 3);
}

function buildKeyTakeaways(p) {
  const slide = addSlide(p, "Four patterns matter more than the raw cell count");
  const items = [
    ["Dynamo is the hard-routing outlier.", "It owns backend, worker, rank, KV routing, priority, and engine response fields."],
    ["NAT is the richest adapter layer.", "It can emit Dynamo-style workload predictions and profiler-derived hints."],
    ["Coding harnesses expose agent controls, not infrastructure placement.", "Claude, Codex, DeepSeek, and Qwen focus on effort, tools, cache, turns, and lifecycle."],
    ["Custom/pass-through is not the same as support.", "Deep Agents and Qwen can often carry fields, but the harness does not interpret them natively."],
  ];
  items.forEach((it, i) => {
    const y = 168 + i * 102;
    text(slide, `${i + 1}`, 64, y, 38, 42, { size: 30, bold: true, color: i === 0 ? C.green : i === 1 ? C.blue : i === 2 ? C.amber : C.violet });
    text(slide, it[0], 122, y - 2, 650, 30, { size: 25, bold: true });
    text(slide, it[1], 122, y + 34, 870, 40, { size: 19, color: C.sub });
  });
  addFooter(slide, 4);
}

function buildCategorySlide(p, cat, idx) {
  const slide = addSlide(p, cat.claim, cat.title);
  const left = 52, top = 160;
  const widths = [245, 365, 365, 205];
  const headers = ["Canonical signal", "Native / explicit fields", "Partial, custom, or dependent", "None"];
  let x = left;
  headers.forEach((h, i) => {
    cell(slide, h, x, top, widths[i], 34, { fill: C.pale, bold: true, size: 14.5 });
    x += widths[i];
  });
  const rowH = Math.min(58, Math.floor((640 - top) / cat.rows.length));
  cat.rows.forEach((row, r) => {
    let cx = left;
    row.forEach((v, i) => {
      cell(slide, v, cx, top + 34 + r * rowH, widths[i], rowH, {
        fill: r % 2 ? "#FBFBFB" : "#FFFFFF",
        bold: i === 0,
        size: i === 0 ? 14 : 12.5,
        color: i === 3 ? C.muted : C.ink,
      });
      cx += widths[i];
    });
  });
  addFooter(slide, idx);
}

function buildCategorySummary(p, cat, idx) {
  const s = summaries[cat.short];
  const slide = addSlide(p, s.headline, cat.title);
  text(slide, cat.claim, 56, 154, 1000, 44, { size: 22, color: C.sub });
  const y = 236;
  const sections = [
    ["Where support is explicit", s.native, C.green],
    ["Where the gaps show up", s.gaps, C.amber],
    ["Slide treatment", [s.design], C.blue],
  ];
  sections.forEach((sec, i) => {
    const x = 68 + i * 390;
    line(slide, x, y, 310, sec[2], 4);
    text(slide, sec[0], x, y + 20, 320, 28, { size: 23, bold: true });
    text(slide, wrapList(sec[1]), x, y + 62, 320, 158, { size: 18, color: C.sub });
  });
  text(slide, "Category coverage by harness", 70, 512, 380, 26, { size: 22, bold: true });
  cat.heat.forEach((key, i) => {
    const x = 70 + i * 162;
    rect(slide, x, 558, 132, 34, status[key].fill, { line: key === "none" ? C.rule : "none", lineWidth: key === "none" ? 1 : 0 });
    text(slide, harnesses[i], x, 600, 132, 22, { size: 14, bold: true, align: "center" });
    text(slide, status[key].label, x + 6, 565, 120, 18, { size: 12.5, align: "center", color: key === "none" ? C.muted : C.ink });
  });
  addFooter(slide, idx);
}

function buildFieldExamples(p) {
  const slide = addSlide(p, "A normalized schema should preserve exact field names without treating pass-through as native");
  text(slide, "Recommendation: define one canonical signal schema, then attach per-harness adapters with explicit support levels.", 54, 154, 980, 54, { size: 23, color: C.sub });
  const left = 56, top = 242, rowH = 54;
  fieldExamples.forEach((row, i) => {
    const y = top + i * rowH;
    text(slide, row[0], left, y + 12, 150, 24, { size: 18, bold: true });
    line(slide, left + 160, y + 27, 54, i < 2 ? C.green : i < 4 ? C.blue : C.amber, 3);
    text(slide, row.slice(1).join("   "), left + 230, y + 10, 900, 28, { size: 17, font: "Courier New", color: C.ink });
  });
  text(slide, "Support levels to encode: native, partial, pass-through, custom, provider/internal, none.", 56, 618, 900, 24, { size: 20, bold: true });
  addFooter(slide, 12);
}

function buildLifecycleTimeline(p) {
  const slide = addSlide(p, "Lifecycle signals describe why the request exists, not just how it runs");
  const points = [
    ["Turn", "Normal conversation or graph call"],
    ["Compact", "Summarization / context compression"],
    ["Prewarm", "Cache warming or speculative prefill"],
    ["Memory", "State retrieval or write"],
    ["Finalize", "Session completion"],
    ["Title", "Conversation naming"],
    ["Subagent", "Delegated agent creation"],
  ];
  line(slide, 76, 354, 1088, C.rule, 2);
  points.forEach((pnt, i) => {
    const x = 82 + i * 178;
    rect(slide, x - 8, 345, 18, 18, i === 0 ? C.green : i === 1 ? C.blue : i === 2 ? C.amber : C.gray, { round: true, line: "none" });
    text(slide, pnt[0], x - 30, 282, 100, 28, { size: 20, bold: true, align: "center" });
    text(slide, pnt[1], x - 58, 390, 145, 70, { size: 16, color: C.sub, align: "center" });
  });
  text(slide, "Codex and DeepSeek expose explicit purpose markers for compaction/prewarm/title-like flows. Other systems usually infer the same lifecycle through workflow state, provider behavior, or internal metadata.", 84, 522, 1050, 70, { size: 22, color: C.sub });
  addFooter(slide, 10);
}

function buildRoutingSlide(p) {
  const slide = addSlide(p, "Hard placement is concentrated in Dynamo; adapters mostly pass it through");
  const x0 = 82, y0 = 184;
  const rows = [
    ["Client", "Harness", "Router", "Backend", "Worker / DP rank"],
    ["soft signals", "priority, OSL, effort", "queue / policy", "model endpoint", "prefill / decode"],
    ["hard placement", "session, route hint", "backend_instance_id", "selected backend", "worker_id, dp_rank"],
  ];
  rows[0].forEach((h, i) => {
    const x = x0 + i * 225;
    rect(slide, x, y0, 172, 64, i === 2 ? "#E8F4FB" : C.pale, { line: "#D7DCE2", lineWidth: 1 });
    text(slide, h, x + 12, y0 + 18, 148, 28, { size: 20, bold: true, align: "center" });
    if (i < 4) line(slide, x + 174, y0 + 32, 50, C.rule, 2);
  });
  rows.slice(1).forEach((row, r) => {
    row.slice(1).forEach((v, i) => {
      const x = x0 + (i + 1) * 225;
      text(slide, v, x - 12, y0 + 120 + r * 120, 196, 54, { size: 17, color: C.sub, align: "center" });
    });
    text(slide, row[0], x0 - 10, y0 + 120 + r * 120, 150, 30, { size: 20, bold: true, color: r === 0 ? C.blue : C.green });
  });
  text(slide, "Use routing slides to separate preference signals from binding decisions. That distinction is easy to lose in a wide table.", 86, 602, 990, 28, { size: 21, bold: true });
  addFooter(slide, 11);
}

function buildAppendixIntro(p) {
  const slide = addSlide(p, "Appendix: full signal detail by category", "Reference section");
  text(slide, "Each appendix slide keeps the explicit unsupported values visible. Cells marked None in the screenshots are represented in the right-hand column.", 70, 166, 980, 60, { size: 25, color: C.sub });
  const list = categories.map((c, i) => `${i + 1}. ${c.title}`).join("\n");
  text(slide, list, 92, 278, 720, 300, { size: 27, bold: true });
  addFooter(slide, 13);
}

async function main() {
  await fs.mkdir(TMP, { recursive: true });
  const p = Presentation.create({ slideSize: { width: W, height: H } });

  buildTitle(p);
  buildTaxonomy(p);
  buildHeatmap(p);
  buildKeyTakeaways(p);
  buildCategorySummary(p, categories[0], 5);
  buildCategorySummary(p, categories[1], 6);
  buildCategorySummary(p, categories[2], 7);
  buildCategorySummary(p, categories[3], 8);
  buildLifecycleTimeline(p);
  buildCategorySummary(p, categories[6], 10);
  buildRoutingSlide(p);
  buildFieldExamples(p);
  buildAppendixIntro(p);
  categories.forEach((cat, i) => buildCategorySlide(p, cat, 14 + i));

  for (const [index, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await p.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(`${TMP}/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${TMP}/${stem}.layout.json`, await layout.text());
  }
  const montage = await p.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(`${TMP}/deck-montage.webp`, new Uint8Array(await montage.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
  console.log(OUT);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
