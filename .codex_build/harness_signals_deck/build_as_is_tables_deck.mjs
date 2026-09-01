import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/presentation/Harness Signal Tables As-Is.pptx";
const TMP = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/.codex_build/harness_signals_deck/as_is_tables";

const W = 1920;
const H = 1080;
const C = {
  ink: "#111111",
  sub: "#555555",
  muted: "#777777",
  rule: "#D0D4DA",
  grid: "#E1E4E8",
  head: "#F1F1F1",
  alt: "#FAFAFA",
  white: "#FFFFFF",
};

const screenshots = [
  "/Users/oluwolejaiyeoba/Desktop/Screenshot 2026-08-25 at 12.15.33\u202fPM.png",
  "/Users/oluwolejaiyeoba/Desktop/Screenshot 2026-08-25 at 12.15.44\u202fPM.png",
  "/Users/oluwolejaiyeoba/Desktop/Screenshot 2026-08-25 at 12.16.08\u202fPM.png",
  "/Users/oluwolejaiyeoba/Desktop/Screenshot 2026-08-25 at 12.16.32\u202fPM.png",
  "/Users/oluwolejaiyeoba/Desktop/Screenshot 2026-08-25 at 12.16.42\u202fPM.png",
  "/Users/oluwolejaiyeoba/Desktop/Screenshot 2026-08-25 at 12.16.53\u202fPM.png",
  "/Users/oluwolejaiyeoba/Desktop/Screenshot 2026-08-25 at 12.17.01\u202fPM.png",
];

const slides = [
  {
    title: "2. Scheduling and workload-prediction signals",
    columns: ["Canonical signal", "Dynamo", "NeMo Agent Toolkit", "Claude Code", "Codex", "DeepSeek Harness", "Deep Agents", "Qwen Code"],
    rows: [
      ["Soft priority", "nvext.agent_hints.priority", "Emits priority", "None", "None", "None", "None", "None"],
      ["Strict priority tier", "strict_priority", "None", "None", "None", "None", "None", "None"],
      ["Latency sensitivity", "None", "Experimental latency_sensitivity", "None", "None", "None", "None", "None"],
      ["Provider speed/QoS tier", "None", "Provider-dependent", "Fast mode / speed", "service_tier", "None", "Provider-dependent", "Provider-specific extra_body"],
      ["Expected output length", "osl", "Emits osl", "None", "None", "None", "None", "None"],
      ["Expected interarrival time", "None", "Custom iat", "None", "None", "None", "None", "None"],
      ["Predicted remaining calls", "None", "Custom total_requests", "None", "None", "None", "None", "None"],
      ["Prefix/workflow reuse ID", "None", "Custom prefix_id", "None", "None", "None", "None", "None"],
    ],
  },
  {
    title: "3. Reasoning and inference-compute signals",
    subtitle: "This is the table corresponding directly to your reasoning-effort example.",
    columns: ["Signal type", "Dynamo", "NeMo Agent Toolkit", "Claude Code", "Codex", "DeepSeek Harness", "Deep Agents", "Qwen Code"],
    rows: [
      ["Reasoning effort level", "No abstract level", "Underlying provider configuration", "output_config.effort", "reasoning.effort", "reasoningEffort -> reasoning_effort", "Underlying model setting", "model.reasoningEffort"],
      ["Supported effort values", "N/A", "Provider-dependent", "low, medium, high, sometimes xhigh / max", "minimal, low, medium, high, xhigh", "off, low, high, max", "Provider-dependent", "low, medium, high, xhigh, max"],
      ["Thinking enabled/disable", "No generic toggle", "Provider-dependent", "thinking.type", "Mainly controlled through effort", "thinking.type", "Provider-dependent", "enable_thinking"],
      ["Thinking-token budget", "max_thinking_tokens", "Provider-dependent", "budget_tokens where applicable", "Reasoning configuration", "Primarily effort-based", "Provider-dependent", "thinking_budget / budget_tokens"],
      ["Maximum output tokens", "Standard max_tokens", "Model-call configuration", "max_tokens", "API-level control; not a main routing signal", "maxTokens", "Underlying model control", "max_tokens or max_completion_tokens"],
      ["Sampling controls", "Standard OpenAI-compatible controls; greed_sampling override", "Provider-dependent", "Temperature/top-p where model permits", "Provider/model-dependent", "temperature, stop", "Provider-dependent", "samplingParams pass-through"],
      ["Output verbosity", "None as an agent hint", "Provider-dependent", "Effort/output configuration", "text.verbosity", "None", "Provider-dependent", "Provider-dependent"],
    ],
  },
  {
    title: "4. Cache-management signals",
    columns: ["Canonical signal", "Dynamo", "NeMo Agent Toolkit", "Claude Code", "Codex", "DeepSeek Harness", "Deep Agents", "Qwen Code"],
    rows: [
      ["Cache key/bucket", "Automatic prompt hash; no client cache key", "Custom prefix_id, but not equivalent to a cache key", "Exact-prefix matching", "prompt_cache_key", "Automatic prefix matching", "Provider middleware", "Automatic prefix matching"],
      ["Cache namespace/isolation", "nvext.cache_salt or x-tenant-id", "Could pass Dynamo salt", "No equivalent public field", "Cache key provides bucketing, not tenant isolation", "None", "Provider-dependent", "Static header/body pass-through"],
      ["Cache TTL", "Not currently supported publicly", "Experimental nvext.cache_control.ttl", "cache_control; 5-minute or 1-hour TTL", "No explicit TTL", "Provider-managed automatic TTL", "Anthropic/Bedrock middleware; 5-minute/1-hour", "cacheRetention / provider cache control"],
      ["Cache entry type", "No current public context type", "type:\"ephemeral\" experimentally", "type:\"ephemeral\"", "None", "None", "Provider-dependent", "Provider-dependent"],
      ["Eviction priority", "priority; currently effective with priority-aware SGLang cache", "Emits priority", "None", "None", "None", "None unless custom", "Pass-through only"],
      ["Speculative prefill", "speculative_prefill", "Adapter does not currently expose it", "None", "WebSocket prewarm, but different mechanism", "None", "None", "UI speculation, but not KV prefill"],
      ["Cache-hit feedback", "Metrics and response extensions", "Profiler/metrics", "Cache-read and cache-creation usage", "Cached input-token usage", "cacheReadTokens", "Provider trace data", "Cache-hit metrics"],
      ["Cache pinning", "Not a current public feature", "Experimental/version-bound", "Provider-managed TTL retention", "None", "Provider-managed", "Provider middleware", "Provider-dependent"],
    ],
  },
  {
    title: "5. Identity and hierarchy signals",
    columns: ["Canonical signal", "Dynamo", "NeMo Agent Toolkit", "Claude Code", "Codex", "DeepSeek Harness", "Deep Agents", "Qwen Code"],
    rows: [
      ["Session ID", "X-Dynamo-Session-ID", "Workflow-derived prefix context", "x-claude-code-session-id", "thread-id; metadata session_id/thread_id", "x-deepseek-harness-session-id", "thread_id", "Internal session ID"],
      ["Parent session", "X-Dynamo-Parent-Session-ID", "Call-stack-derived prefix", "x-claude-code-parent-agent-id", "x-codex-parent-thread-id", "None identified", "Parent run tracing", "None automatically sent"],
      ["Agent/subagent ID", "Normalizes supported harness headers", "Prefix depth/run context", "x-claude-code-agent-id", "Subagent metadata and headers", "None identified", "ls_agent_type=\"subagent\"", "Internal agent state"],
      ["Turn ID", "Available through tracing/session context", "Profiler call index", "Not exposed as a stable documented header", "Metadata turn_id", "Conversation state", "Run/checkpoint identity", "Internal"],
      ["Root/parent turn hierarchy", "Consumer-defined", "Workflow context", "Agent hierarchy", "root_turn_id, parent_turn_id", "None", "Trace hierarchy", "None"],
      ["Anonymous user/installation", "Tenant/session headers", "None central", "Provider/account context", "x-codex-installation-id", "x-deepseek-harness-user-id", "Runtime context", "Trace/client context"],
      ["W3C trace context", "Tracing support", "Profiler/tracing", "Gateway/provider tracing", "OpenTelemetry-related metadata", "Request IDs", "LangSmith/runtime tracing", "Optional traceparent and baggage"],
    ],
  },
  {
    title: "6. Lifecycle and request-purpose signals",
    columns: ["Canonical signal", "Dynamo", "NeMo Agent Toolkit", "Claude Code", "Codex", "DeepSeek Harness", "Deep Agents", "Qwen Code"],
    rows: [
      ["Normal agent turn", "Normal request", "Normal workflow call", "Normal conversation request", "request_kind:\"turn\"", "Normal purpose", "Normal graph invocation", "Normal request"],
      ["Compaction", "Can observe normalized Codex metadata", "Profiler/workflow dependent", "Internal compaction traffic, no generic header", "request_kind:\"compaction\" plus detailed metadata", "purpose:\"compaction\" and x-deepseek-harness-compact:1", "Summarization middleware", "Separate compactionModel, internally"],
      ["Prewarm", "speculative_prefill", "Not currently emitted", "Provider cache warming through normal calls", "request_kind:\"prewarm\" / WebSocket prewarm", "None", "None", "UI speculation only"],
      ["Memory request", "No standard hint", "Workflow dependent", "Internal/provider-specific", "request_kind:\"memory\"", "None", "State/checkpoint operations", "Internal"],
      ["Session completion", "X-Dynamo-Session-Final", "None identified", "None identified", "No equivalent general final-session header", "None identified", "Graph/run completion", "None identified"],
      ["Session-title request", "No standard field", "None identified", "Internal", "Potential metadata/custom request kind", "purpose:\"session-title\"", "Internal", "Internal"],
      ["Subagent creation", "Parent/session hierarchy", "Workflow stack depth", "Agent/parent-agent headers", "Subagent headers and metadata", "None identified", "Native subagent runtime", "Internal agent behavior"],
    ],
  },
  {
    title: "7. Routing and placement signals",
    columns: ["Signal", "Dynamo", "NAT", "Claude Code", "Codex", "DeepSeek Harness", "Deep Agents", "Qwen Code"],
    rows: [
      ["Direct backend selection", "backend_instance_id", "Could pass through", "None", "None for custom providers", "None", "Custom middleware", "Static extra_body"],
      ["Prefill worker selection", "prefill_worker_id", "Could pass through", "None", "None", "None", "Custom", "Pass-through"],
      ["Decode worker selection", "decode_worker_id", "Could pass through", "None", "None", "None", "Custom", "Pass-through"],
      ["Data-parallel rank", "dp_rank, prefill_dp_rank", "Could pass through", "None", "None", "None", "Custom", "Pass-through"],
      ["Sticky routing", "Optional session affinity", "Custom processor/prefix logic", "Session ID only", "x-codex-turn-state, OpenAI path only", "Session ID only", "Checkpointing, not inference placement", "None automatically"],
      ["Provider routing hint", "Native router policies", "Custom processor", "Model/fast selection", "x-codex-routing-hint, OpenAI path", "Provider/model selection", "Model selection", "Provider/model selection"],
      ["Cache-aware placement", "Native KV router", "Dynamo integration", "Provider-managed", "Provider-managed", "Provider-managed", "Depends on inference provider", "Depends on provider"],
    ],
  },
  {
    title: "8. Observability and response signals",
    columns: ["Signal type", "Dynamo", "NAT", "Claude Code", "Codex", "DeepSeek Harness", "Deep Agents", "Qwen Code"],
    rows: [
      ["Worker identity", "extra_fields:[\"worker_id\"]", "Dynamo response", "Not normally exposed", "Provider/internal metadata", "None", "Provider trace", "Provider-dependent"],
      ["TTFT/ITL/queue time", "extra_fields:[\"timing\"]", "Profiler metrics", "Provider usage/telemetry", "Timing-metrics header", "General stream timing", "LangSmith tracing", "Provider/trace metrics"],
      ["Cache read tokens", "Metrics/backend response", "Profiler", "Anthropic usage", "OpenAI cached-token usage", "cacheReadTokens", "Provider trace", "Cache metrics"],
      ["Routed experts", "routed_experts", "Dynamo response", "None", "None", "None", "Provider-dependent", "Provider-dependent"],
      ["Engine metadata", "engine_data", "Dynamo response", "None", "Provider metadata", "Provider request IDs", "Provider trace", "Provider-dependent"],
      ["Stop reason", "stop_reason", "Model response", "Provider response", "Provider response", "Finish reason", "Model response", "Provider response"],
      ["Completion token IDs", "completion_token_ids", "Dynamo response", "Not normally exposed", "Provider-dependent", "None identified", "Provider-dependent", "Provider-dependent"],
      ["Request IDs", "Dynamo tracing", "Workflow/profiler", "Provider request IDs", "Response IDs/turn state", "x-request-id, x-deepseek-request-id", "Trace IDs", "W3C trace IDs"],
    ],
  },
];

const signalIds = {
  "Soft priority": "priority",
  "Strict priority tier": "strict_priority",
  "Latency sensitivity": "latency_sensitivity",
  "Provider speed/QoS tier": "qos_tier",
  "Expected output length": "output_length",
  "Expected interarrival time": "interarrival_time",
  "Predicted remaining calls": "remaining_calls",
  "Prefix/workflow reuse ID": "prefix_id",
  "Reasoning effort level": "reasoning_effort",
  "Supported effort values": "effort_values",
  "Thinking enabled/disable": "thinking",
  "Thinking-token budget": "thinking_budget",
  "Maximum output tokens": "max_tokens",
  "Sampling controls": "sampling",
  "Output verbosity": "verbosity",
  "Cache key/bucket": "cache_key",
  "Cache namespace/isolation": "cache_namespace",
  "Cache TTL": "cache_ttl",
  "Cache entry type": "cache_type",
  "Eviction priority": "eviction_priority",
  "Speculative prefill": "speculative_prefill",
  "Cache-hit feedback": "cache_hit_feedback",
  "Cache pinning": "cache_pinning",
  "Session ID": "session_id",
  "Parent session": "parent_session",
  "Agent/subagent ID": "agent_id",
  "Turn ID": "turn_id",
  "Root/parent turn hierarchy": "turn_hierarchy",
  "Anonymous user/installation": "installation_id",
  "W3C trace context": "trace_context",
  "Normal agent turn": "turn",
  "Compaction": "compaction",
  "Prewarm": "prewarm",
  "Memory request": "memory",
  "Session completion": "session_completion",
  "Session-title request": "session_title",
  "Subagent creation": "subagent",
  "Direct backend selection": "backend",
  "Prefill worker selection": "prefill_worker",
  "Decode worker selection": "decode_worker",
  "Data-parallel rank": "dp_rank",
  "Sticky routing": "sticky_routing",
  "Provider routing hint": "provider_routing",
  "Cache-aware placement": "cache_placement",
  "Worker identity": "worker_id",
  "TTFT/ITL/queue time": "timing",
  "Cache read tokens": "cache_read_tokens",
  "Routed experts": "routed_experts",
  "Engine metadata": "engine_metadata",
  "Stop reason": "stop_reason",
  "Completion token IDs": "completion_token_ids",
  "Request IDs": "request_ids"
};

function compactJson(obj) {
  return JSON.stringify(obj);
}

function compactCell(raw, signalLabel, harness) {
  if (!signalLabel || raw === signalLabel) return raw;
  const signal = signalIds[signalLabel] ?? signalLabel.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  const value = String(raw);
  const lower = value.toLowerCase();

  if (
    value === "None" ||
    value === "None identified" ||
    value === "N/A" ||
    lower.startsWith("no current public") ||
    lower.startsWith("no equivalent") ||
    lower.startsWith("no abstract") ||
    lower.startsWith("no generic") ||
    lower.startsWith("no native") ||
    lower.startsWith("no osl hint") ||
    lower.startsWith("no standard") ||
    lower.startsWith("none as") ||
    lower.startsWith("none for") ||
    lower.startsWith("none automatically") ||
    lower.startsWith("none central") ||
    lower.startsWith("not currently supported") ||
    lower.startsWith("not a current public") ||
    lower.startsWith("not normally exposed") ||
    lower.startsWith("not exposed") ||
    lower.startsWith("adapter does not")
  ) {
    return "-";
  }

  if (value === "Normal request" || value === "Normal workflow call" || value === "Normal conversation request" || value === "Normal purpose" || value === "Normal graph invocation") {
    return compactJson({ [signal]: "default" });
  }
  if (value === "Normal request") return compactJson({ [signal]: "default" });

  if (lower.includes("task metadata") || lower.includes("orchestration metadata")) {
    return compactJson({ via: "orchestration_metadata" });
  }
  if (lower.includes("provider-dependent") || lower.includes("provider/model-dependent") || lower.includes("provider-managed") || lower.includes("underlying provider") || lower.includes("provider cache") || lower.includes("provider request") || lower.includes("provider trace") || lower.includes("provider response") || lower.includes("provider metadata") || lower.includes("depends on provider") || lower.includes("depends on inference provider")) {
    return compactJson({ via: "provider" });
  }
  if (lower.includes("underlying model")) {
    return compactJson({ via: "model" });
  }
  if (lower.includes("mainly controlled through effort")) {
    return compactJson({ via: "reasoning_effort" });
  }
  if (lower.includes("not inference placement")) {
    return compactJson({ via: "checkpoint" });
  }
  if (lower.includes("custom middleware") || lower.includes("custom extension") || lower.includes("custom plugin") || value === "Custom only" || value === "Custom" || lower.includes("custom processor")) {
    return compactJson({ via: "middleware" });
  }
  if (lower.includes("could pass through") || lower.includes("pass-through only") || lower.includes("pass-through")) {
    const obj = { via: "pass_through" };
    if (lower.includes("extra_body")) obj[signal] = "extra_body";
    return compactJson(obj);
  }
  if (lower.includes("extra_body")) {
    return compactJson({ via: "extra_body", [signal]: "extra_body" });
  }
  if (lower.includes("internal")) {
    return compactJson({ via: "internal" });
  }
  if (lower.includes("session context only") || lower.includes("session/tree context only") || lower.includes("session id only") || lower.includes("session/model route context only")) {
    return compactJson({ via: "internal" });
  }
  if (lower.includes("middleware")) {
    return compactJson({ via: "middleware" });
  }
  if (lower.includes("profiler") || lower.includes("metrics")) {
    return compactJson({ via: "metrics" });
  }
  if (lower.includes("langsmith") || lower.includes("tracing") || lower.includes("trace")) {
    return compactJson({ via: "trace" });
  }

  const exact = {
    "Emits priority": "priority",
    "Emits osl": "osl",
    "Custom iat": "iat",
    "Custom total_requests": "total_requests",
    "Custom prefix_id": "prefix_id",
    "Provider-specific; before_provider_request can add service_tier": ["before_provider_request", "service_tier"],
    "Provider max_tokens only; no OSL hint": "max_tokens",
    "Provider model/output limits only; no OSL hint": "provider_limits",
    "Automatic provider prompt caching; no client cache key": "provider_prompt_cache",
    "Prompt tiers/cache boundaries; model participates in cache key": ["prompt_tiers", "model"],
    "Provider/account isolation only": "provider_account",
    "Provider/plugin-dependent": "provider_or_plugin",
    "Provider/session isolation only": "provider_session",
    "Provider/session/model isolation only": "provider_session_model",
    "Fast mode / speed": ["fast", "speed"],
    "Thread/cache key": ["thread_id", "cache_key"],
    "Session/agent IDs": ["session_id", "agent_id"],
    "Session ID": "session_id",
    "Session ID only": "session_id",
    "Thread/cache key": ["thread_id", "cache_key"],
    "Model-call configuration": "model_config",
    "Reasoning configuration": "reasoning",
    "Primarily effort-based": "reasoning_effort",
    "API-level control; not a main routing signal": "max_tokens",
    "Temperature/top-p where model permits": ["temperature", "top_p"],
    "Effort/output configuration": ["effort", "output"],
    "Automatic prefix matching": "prefix_hash",
    "Exact-prefix matching": "prefix_hash",
    "Could pass Dynamo salt": "nvext.cache_salt",
    "Cache key provides bucketing, not tenant isolation": "prompt_cache_key",
    "Cache-read and cache-creation usage": ["cache_read", "cache_creation"],
    "Cached input-token usage": "cached_input_tokens",
    "Cache-hit metrics": "cache_metrics",
    "Provider middleware": "middleware",
    "Workflow-derived prefix context": "prefix_context",
    "Call-stack-derived prefix": "call_stack_prefix",
    "Prefix depth/run context": ["prefix_depth", "run_context"],
    "Profiler call index": "call_index",
    "Conversation state": "conversation_state",
    "Run/checkpoint identity": "checkpoint_id",
    "Workflow context": "workflow_context",
    "Agent hierarchy": "agent_hierarchy",
    "Runtime context": "runtime_context",
    "Trace/client context": "trace_context",
    "Request IDs": "request_ids",
    "OpenTelemetry-related metadata": "otel_metadata",
    "State/checkpoint operations": "checkpoint_state",
    "Graph/run completion": "run_completion",
    "Native subagent runtime": "subagent",
    "Parent/session hierarchy": ["parent_session", "session_id"],
    "Workflow stack depth": "stack_depth",
    "Agent/parent-agent headers": ["agent_id", "parent_agent_id"],
    "Subagent headers and metadata": ["subagent_headers", "metadata"],
    "Optional session affinity": "session_affinity",
    "Model/fast selection": ["model", "fast"],
    "Provider/model selection": ["provider", "model"],
    "Model selection": "model",
    "Native router policies": "router_policy",
    "Native KV router": "kv_router",
    "Dynamo integration": "dynamo",
    "Dynamo response": "dynamo_response",
    "Model response": "model_response",
    "Finish reason": "finish_reason",
    "Response IDs/turn state": ["response_id", "turn_state"],
    "Workflow/profiler": ["workflow", "profiler"],
    "Provider/trace metrics": "trace_metrics",
    "Cache metrics": "cache_metrics",
    "Not emitted": "-"
  };
  if (Object.prototype.hasOwnProperty.call(exact, value)) {
    const mapped = exact[value];
    if (mapped === "-") return "-";
    return compactJson({ [signal]: mapped });
  }

  if (lower.includes("openai-compatible controls")) {
    return compactJson({ [signal]: ["temperature", "top_p", "greed_sampling"] });
  }
  if (lower.includes("low") && lower.includes("medium") && lower.includes("high")) {
    const values = value.match(/off|minimal|low|medium|high|xhigh|max/g) ?? [];
    return compactJson({ [signal]: [...new Set(values)] });
  }
  if (lower.includes("5-minute") || lower.includes("1-hour")) {
    return compactJson({ [signal]: ["5m", "1h"] });
  }
  if (lower.includes("websocket prewarm")) {
    return compactJson({ [signal]: "websocket_prewarm" });
  }
  if (lower.includes("not kv prefill") || lower.includes("not kv speculative")) {
    return compactJson({ via: "different_mechanism" });
  }
  if (lower.includes("ui speculation")) {
    return compactJson({ via: "ui", [signal]: "speculation" });
  }
  if (lower.includes("workflow may know") || lower.includes("could infer")) {
    return compactJson({ via: "workflow" });
  }
  if (lower.includes("legacy field")) {
    return compactJson({ legacy: "latency_sensitivity" });
  }
  if (lower.includes("session id is closest")) {
    return compactJson({ [signal]: "X-Dynamo-Session-ID" });
  }
  if (lower.includes("can observe normalized codex metadata")) {
    return compactJson({ via: "metadata" });
  }
  if (lower.includes("potential metadata")) {
    return compactJson({ via: "metadata" });
  }

  const codeLike = value.match(/X-Dynamo-[A-Za-z-]+|x-[a-z0-9-]+|[a-zA-Z][a-zA-Z0-9]*(?:[._-][a-zA-Z0-9]+)+|[a-z]+[A-Z][a-zA-Z0-9]+/g);
  if (codeLike?.length) {
    const fields = [...new Set(codeLike.map((field) => field.replace(/;$/, "")))];
    if (harness === "Dynamo" && fields.some((field) => field.startsWith("nvext."))) {
      const nvextFields = fields.filter((field) => field.startsWith("nvext."));
      const grouped = {};
      for (const field of nvextFields) {
        const parts = field.replace(/^nvext\./, "").split(".");
        const top = parts.length > 1 ? `nvext.${parts.slice(0, -1).join(".")}` : "nvext";
        const leaf = parts.at(-1);
        grouped[top] ??= [];
        grouped[top].push(leaf);
      }
      return compactJson(grouped);
    }
    return compactJson({ [signal]: fields.length === 1 ? fields[0] : fields });
  }

  return compactJson({ [signal]: value });
}

function addText(slide, value, left, top, width, height, opts = {}) {
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
    autoFit: "shrinkText",
    insets: opts.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function addRect(slide, left, top, width, height, fill, line = C.grid, lineWidth = 1) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: lineWidth },
  });
}

function drawCell(slide, value, left, top, width, height, opts = {}) {
  const displayValue = opts.compact ? compactCell(value, opts.signalLabel, opts.harness) : (value === "None" || value === "None identified" ? "-" : value);
  addRect(slide, left, top, width, height, opts.fill ?? C.white, opts.line ?? C.grid, opts.lineWidth ?? 1);
  if (opts.compact) {
    if (displayValue === "-") {
      addText(slide, "-", left + 10, top + 9, width - 20, height - 16, {
        size: 19,
        color: C.muted,
        font: "Helvetica Neue",
      });
      return;
    }
    addText(slide, value, left + 10, top + 8, width - 20, Math.max(26, height * 0.43), {
      size: opts.compactDescSize ?? 16,
      color: C.ink,
      font: "Helvetica Neue",
    });
    addText(slide, displayValue, left + 10, top + Math.max(39, height * 0.46), width - 20, Math.max(26, height * 0.48), {
      size: opts.compactJsonSize ?? 13.4,
      color: C.sub,
      font: "Courier New",
    });
    return;
  }
  const isCodeHeavy = /[_"]|nvext|extra_fields|cacheReadTokens|reasoning|traceparent|maxTokens|service_tier|request_kind|purpose:|x-/.test(displayValue);
  addText(slide, displayValue, left + 10, top + 9, width - 20, height - 16, {
    size: opts.size ?? (isCodeHeavy ? 14.5 : 17),
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    font: opts.font ?? (isCodeHeavy ? "Courier New" : "Helvetica Neue"),
  });
}

function addFooter(slide, slideNumber, sourcePath) {
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 58, top: 1016, width: 1804, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.grid, width: 1 },
  });
  addText(slide, "Sources: user screenshots and public docs in speaker notes; each populated cell shows description plus compact JSON; '-' indicates unsupported.", 58, 1030, 1450, 24, { size: 15, color: C.muted });
  addText(slide, String(slideNumber), 1814, 1030, 48, 24, { size: 15, color: C.muted, align: "right" });
  slide.speakerNotes.textFrame.setText(`[Sources]\n- ${sourcePath}\n`);
}

function buildLegendSlide(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addText(slide, "Legend — How to interpret support labels", 56, 42, 1420, 56, { size: 38, bold: true });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 56, top: 122, width: 1806, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  const definitions = [
    ["Native", "Built-in, documented support in the named platform or component."],
    ["Emits", "The harness automatically sends the signal; a compatible backend must still consume it."],
    ["Custom / Custom only", "Developer-written code, an adapter, or middleware is required."],
    ["Pass-through only", "Forwards a user-supplied field unchanged; it does not create, validate, or act on it."],
    ["Could pass through", "Generic forwarding appears possible, but no standard mapping is verified."],
    ["Provider-dependent", "Support comes from the selected model provider, not consistently from the harness."],
    ["Internal only", "Known or used internally, but not exposed as a stable outbound signal."],
    ["Automatic / provider-managed", "Capability exists, but the provider controls it automatically."],
    ["Experimental / version-bound", "Integration-specific and not a stable portable contract."],
    ["Legacy", "Previously supported but deprecated or removed from the current interface."],
    ["None / '-' / N/A", "No documented equivalent or applicable support was found."],
    ["Profiler / metrics / tracing", "Observability only; reports what happened rather than controlling inference behavior."]
  ];
  const top = 156;
  const left = 68;
  const labelW = 330;
  const descW = 590;
  const rowH = 58;
  for (let i = 0; i < definitions.length; i++) {
    const col = i < 6 ? 0 : 1;
    const r = i % 6;
    const x = left + col * 910;
    const y = top + r * rowH;
    addRect(slide, x, y, labelW + descW, rowH, r % 2 === 0 ? C.white : C.alt, C.grid, 1);
    addText(slide, definitions[i][0], x + 12, y + 10, labelW - 24, 34, { size: 18, bold: true });
    addText(slide, definitions[i][1], x + labelW + 8, y + 10, descW - 20, 34, { size: 15.6, color: C.sub });
  }
  addText(slide, "Strict comparison rule", 78, 566, 420, 28, { size: 25, bold: true });
  const classifications = [
    ["Supported", "Native; Emits when compatible downstream consumption is verified"],
    ["Conditional", "Provider-dependent, automatic, experimental"],
    ["Not natively supported", "Custom, pass-through, internal, middleware"],
    ["Unsupported", "None, '-', N/A"]
  ];
  classifications.forEach((row, i) => {
    const x = 78 + i * 438;
    addRect(slide, x, 616, 388, 78, i % 2 === 0 ? C.head : C.alt, C.grid, 1);
    addText(slide, row[0], x + 14, 626, 350, 22, { size: 18, bold: true });
    addText(slide, row[1], x + 14, 654, 350, 32, { size: 14.8, color: C.sub });
  });
  addText(slide, "Most important distinction: Native is built in; Custom must be built; Pass-through only carries the field; '-' means no native equivalent.", 78, 742, 1500, 34, { size: 20, bold: true, color: C.ink });
  addText(slide, "Cell format: human-readable description on top, compact JSON wire shape below.", 78, 800, 1050, 26, { size: 18, color: C.sub });
  addText(slide, "Example: Pass-through only  {\"via\":\"pass_through\"}", 78, 838, 780, 24, { size: 16, color: C.sub, font: "Courier New" });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 58, top: 1016, width: 1804, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.grid, width: 1 },
  });
  addText(slide, "Source: user-provided legend text in chat.", 58, 1030, 1040, 24, { size: 15, color: C.muted });
  addText(slide, "1", 1814, 1030, 48, 24, { size: 15, color: C.muted, align: "right" });
  slide.speakerNotes.textFrame.setText("[Sources]\n- User-provided legend text in chat.\n");
}

function cloneWithRows(spec, title, rowNames, subtitle = spec.subtitle) {
  const wanted = new Set(rowNames);
  return {
    ...spec,
    title,
    subtitle,
    rows: spec.rows.filter((row) => wanted.has(row[0]))
  };
}

const schedulingFocus = cloneWithRows(
  slides[0],
  "Scheduling and workload-prediction hints",
  [
    "Soft priority",
    "Strict priority tier",
    "Latency sensitivity",
    "Provider speed/QoS tier",
    "Expected output length",
    "Expected interarrival time",
    "Predicted remaining calls",
    "Prefix/workflow reuse ID"
  ],
  "Strict view: nearby task metadata, internal state, and generic pass-through are shown as '-' unless they expose a real backend/provider signal."
);

const cacheFocus = {
  ...slides[2],
  title: "KV cache and prompt-cache handling",
  subtitle: "Signals that can influence cache reuse, cache isolation, retention, eviction, prefill, or cache feedback."
};

function selectHarnessColumns(spec, title, harnessColumns, source) {
  const indexes = [0, ...harnessColumns.map((name) => spec.columns.indexOf(name)).filter((index) => index > 0)];
  return {
    spec: {
      ...spec,
      title,
      columns: indexes.map((index) => spec.columns[index]),
      rows: spec.rows.map((row) => indexes.map((index) => row[index]))
    },
    source
  };
}

const focusedSchedulingSlides = [
  selectHarnessColumns(
    schedulingFocus,
    "Scheduling hints — Dynamo, NAT, Claude, Codex",
    ["Dynamo", "NeMo Agent Toolkit", "Claude Code", "Codex"],
    screenshots[0]
  ),
  selectHarnessColumns(
    schedulingFocus,
    "Scheduling hints — DeepSeek, Deep Agents, Qwen",
    ["DeepSeek Harness", "Deep Agents", "Qwen Code"],
    screenshots[0]
  )
];

const focusedCacheSlides = [
  selectHarnessColumns(
    cacheFocus,
    "KV / prompt cache — Dynamo, NAT, Claude, Codex",
    ["Dynamo", "NeMo Agent Toolkit", "Claude Code", "Codex"],
    screenshots[2]
  ),
  selectHarnessColumns(
    cacheFocus,
    "KV / prompt cache — DeepSeek, Deep Agents, Qwen",
    ["DeepSeek Harness", "Deep Agents", "Qwen Code"],
    screenshots[2]
  )
];

const additionalHarnessSources = [
  "https://pi.dev/",
  "https://github.com/earendil-works/pi",
  "https://opencode.ai/",
  "https://opencode.ai/docs/",
  "https://docs.openclaw.ai/plugins/sdk-agent-harness",
  "https://docs.openclaw.ai/tools/plugin",
  "https://hermes-agent.nousresearch.com/docs/",
  "https://github.com/NousResearch/hermes-agent",
  "https://pi.dev/docs/latest/providers",
  "https://pi.dev/docs/latest/extensions",
  "https://pi.dev/docs/latest/models",
  "https://pi.dev/docs/latest/compaction",
  "https://opencode.ai/docs/cli/",
  "https://opencode.ai/docs/plugins/",
  "https://opencode.ai/docs/providers/",
  "https://docs.openclaw.ai/providers/openai",
  "https://docs.openclaw.ai/concepts/model-providers",
  "https://docs.openclaw.ai/plugins/sdk-agent-harness",
  "https://hermes-agent.nousresearch.com/docs/integrations/providers",
  "https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching",
  "https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation",
  "https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop"
].join("; ");

const harnessAdditions = [
  [
    "Yes",
    "Pi Agent Harness",
    "Distinct lightweight agent harness; worth evaluating separately. Pi describes itself as a minimal agent harness, and its repo identifies Pi Agent Harness components.",
    "pi.dev; GitHub"
  ],
  [
    "Yes",
    "OpenCode",
    "Distinct open-source coding agent/harness with terminal, desktop, and IDE surfaces.",
    "opencode.ai"
  ],
  [
    "Yes",
    "OpenClaw",
    "Distinct agent runtime and plugin harness ecosystem. It has explicit agent harness plugin docs, so it fits this comparison.",
    "OpenClaw docs"
  ],
  [
    "Yes",
    "Hermes Agent",
    "Distinct Nous Research agent harness with memory, skills, messaging, and long-running behavior.",
    "Hermes docs; GitHub"
  ]
];

const additionalScheduling = {
  title: "Additional harnesses — scheduling and workload hints",
  subtitle: "Strict view: nearby orchestration metadata is shown as '-' unless it maps to a concrete backend/provider request field.",
  columns: ["Canonical signal", "Pi Agent Harness", "OpenCode", "OpenClaw", "Hermes Agent"],
  rows: [
    ["Soft priority", "None", "None", "None", "None"],
    ["Strict priority tier", "None", "None", "None", "None"],
    ["Latency sensitivity", "None", "None", "None", "None"],
    ["Provider speed/QoS tier", "None", "None", "Fast mode maps to service_tier=priority on supported OpenAI requests", "request_overrides can set service_tier; /fast can layer service_tier/speed"],
    ["Expected output length", "None", "None", "None", "None"],
    ["Expected interarrival time", "None", "None", "None", "None"],
    ["Predicted remaining calls", "None", "None", "None", "None"],
    ["Prefix/workflow reuse ID", "None", "None", "None", "None"]
  ]
};

const additionalCache = {
  title: "Additional harnesses — KV cache and prompt-cache handling",
  subtitle: "These harnesses mostly preserve or configure provider prompt caching rather than controlling KV cache placement directly.",
  columns: ["Canonical signal", "Pi Agent Harness", "OpenCode", "OpenClaw", "Hermes Agent"],
  rows: [
    ["Cache key/bucket", "Automatic provider prompt caching; no client cache key", "Provider-managed; optional Helicone response-cache headers via plugin", "Provider-managed; cached WebSocket can reuse prior response state", "Prompt tiers/cache boundaries; model participates in cache key"],
    ["Cache namespace/isolation", "Provider/account isolation only", "Provider/plugin-dependent", "Provider/session isolation only", "Provider/session/model isolation only"],
    ["Cache TTL", "prompt_cache_retention or cache_control.ttl in custom model/provider settings", "No native prompt-cache TTL; response cache possible through provider/plugin", "Provider-managed TTL; no generic TTL hint", "Default 5m; 1h for long-running sessions where supported"],
    ["Cache entry type", "Provider-specific cache_control format", "None", "Provider-managed", "Anthropic prompt-caching markers"],
    ["Eviction priority", "None", "None", "None", "None"],
    ["Speculative prefill", "None", "None", "Cached WebSocket/prewarm; not KV prefill", "None"],
    ["Cache-hit feedback", "Footer shows token/cache usage and cost", "Stats/provider usage; cache feedback depends on provider/plugin", "Provider usage/tracing", "Prompt cache metrics"],
    ["Cache pinning", "Long retention where provider supports it; not pinning", "None", "Provider-managed only", "1h retention where supported; not pinning"]
  ]
};

const additionalSchedulingSlides = [
  selectHarnessColumns(
    additionalScheduling,
    "Additional scheduling hints — Pi and OpenCode",
    ["Pi Agent Harness", "OpenCode"],
    additionalHarnessSources
  ),
  selectHarnessColumns(
    additionalScheduling,
    "Additional scheduling hints — OpenClaw and Hermes",
    ["OpenClaw", "Hermes Agent"],
    additionalHarnessSources
  )
];

const additionalCacheSlides = [
  selectHarnessColumns(
    additionalCache,
    "Additional KV / prompt cache — Pi and OpenCode",
    ["Pi Agent Harness", "OpenCode"],
    additionalHarnessSources
  ),
  selectHarnessColumns(
    additionalCache,
    "Additional KV / prompt cache — OpenClaw and Hermes",
    ["OpenClaw", "Hermes Agent"],
    additionalHarnessSources
  )
];

function addDeckFooter(slide, slideNumber, textValue) {
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 58, top: 1016, width: 1804, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.grid, width: 1 },
  });
  addText(slide, textValue, 58, 1030, 1240, 24, { size: 15, color: C.muted });
  addText(slide, String(slideNumber), 1814, 1030, 48, 24, { size: 15, color: C.muted, align: "right" });
}

function buildOverviewSlide(p) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addText(slide, "Focused scope — serving-plane control signals", 56, 42, 1500, 56, { size: 38, bold: true });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 56, top: 122, width: 1806, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  addText(
    slide,
    "This deck keeps only signals that can plausibly affect when a request runs, how serving load is predicted, or how KV/cache state is reused.",
    70,
    158,
    1450,
    42,
    { size: 24, color: C.sub }
  );
  const lanes = [
    ["Scheduling", "Queue order, QoS tier, latency sensitivity, output-length and arrival predictions", ["priority", "latency_sensitivity", "osl", "prefix_id", "total_requests", "iat"]],
    ["KV / Cache", "Cache keying, namespace, TTL, entry type, eviction priority, prefill, hit feedback", ["prompt_cache_key", "cache_control", "cache_salt", "speculative_prefill"]]
  ];
  lanes.forEach((lane, i) => {
    const x = 118 + i * 860;
    addRect(slide, x, 252, 760, 350, i % 2 === 0 ? C.white : C.alt, C.grid, 1);
    addText(slide, lane[0], x + 28, 282, 600, 40, { size: 28, bold: true });
    addText(slide, lane[1], x + 28, 342, 620, 76, { size: 20, color: C.sub });
    addText(slide, lane[2].map((item) => `- ${item}`).join("\n"), x + 28, 448, 620, 120, { size: 18, color: C.ink, font: "Courier New" });
  });
  addText(slide, "Out of scope for this research pass: routing, placement, worker selection, affinity identity, general reasoning controls, lifecycle metadata, and broad observability fields.", 82, 676, 1540, 52, { size: 22, bold: true });
  addText(slide, "NeMo/NAT NVIDIA-extension hint bundle: priority, latency_sensitivity, osl, prefix_id, total_requests, iat.", 82, 748, 1540, 32, { size: 20, color: C.sub, font: "Courier New" });
  addDeckFooter(slide, 2, "Scope rule: keep signals that affect scheduling/workload prediction or KV/prompt-cache handling.");
  slide.speakerNotes.textFrame.setText("[Sources]\n- User request to narrow to KV cache handling and scheduling/workload prediction hints.\n");
}

function buildHarnessAdditionsSlide(p, slideNumber) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addText(slide, "New harnesses added to the evaluation", 56, 42, 1500, 56, { size: 38, bold: true });
  addText(slide, "These additions are distinct agent harnesses, but they are evaluated only for scheduling/workload and KV/prompt-cache signals.", 56, 104, 1540, 40, { size: 22, color: C.sub });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 56, top: 158, width: 1806, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.rule, width: 1 },
  });

  const left = 66;
  const top = 188;
  const widths = [92, 280, 1020, 392];
  const headerH = 58;
  const rowH = 142;
  const headers = ["Add?", "Harness", "Why included", "Verified source"];
  let x = left;
  headers.forEach((header, i) => {
    drawCell(slide, header, x, top, widths[i], headerH, { fill: C.head, bold: true, size: 18, font: "Helvetica Neue" });
    x += widths[i];
  });

  harnessAdditions.forEach((row, r) => {
    let cx = left;
    row.forEach((value, c) => {
      const y = top + headerH + r * rowH;
      addRect(slide, cx, y, widths[c], rowH, r % 2 === 0 ? C.white : C.alt, C.grid, 1);
      addText(slide, value, cx + 12, y + 14, widths[c] - 24, rowH - 24, {
        size: c === 2 ? 20 : 19,
        bold: c === 0 || c === 1,
        color: c === 3 ? C.sub : C.ink,
        font: "Helvetica Neue",
      });
      cx += widths[c];
    });
  });

  addText(slide, "Excluded as already covered: NeMo/NAT, Qwen Code, Codex, Claude Code, Dynamo, and Deep Agents.", 70, 852, 1450, 34, { size: 21, bold: true });
  addDeckFooter(slide, slideNumber, "New additions verified against official docs and repositories.");
  slide.speakerNotes.textFrame.setText(
    "[Sources]\n" +
      "- https://pi.dev/\n" +
      "- https://github.com/earendil-works/pi\n" +
      "- https://opencode.ai/\n" +
      "- https://opencode.ai/docs/\n" +
      "- https://docs.openclaw.ai/plugins/sdk-agent-harness\n" +
      "- https://docs.openclaw.ai/tools/plugin\n" +
      "- https://hermes-agent.nousresearch.com/docs/\n" +
      "- https://github.com/NousResearch/hermes-agent\n"
  );
}

const definitionSections = [
  {
    title: "Scheduling signals in simple English",
    subtitle: "These hints help the serving system decide when a request should run and how much capacity it may need.",
    items: [
      ["priority", "Tells the serving system how important a request is. Higher-priority requests may be run before lower-priority ones."],
      ["latency_sensitivity", "Tells the system how much the request cares about speed. A highly latency-sensitive request should avoid waiting if possible."],
      ["osl", "Usually means output sequence length or expected output length. It tells the system roughly how many tokens the model may generate, so the scheduler can plan capacity."],
      ["prefix_id", "A label for a shared prompt prefix or workflow. If multiple requests share the same beginning, this helps the system reuse cached KV state."],
      ["total_requests", "A hint about how many related requests are expected in the workflow. This helps the system predict future load."],
      ["iat", "Usually means inter-arrival time. It hints how much time may pass between related requests, helping the scheduler anticipate bursts or gaps."]
    ]
  },
  {
    title: "KV and cache signals in simple English",
    subtitle: "These hints help the serving system decide what prompt or KV work can be reused and how cache entries should be separated or retained.",
    items: [
      ["prompt_cache_key", "A stable key for identifying a reusable prompt prefix. If the same key appears again, the system may reuse cached prompt/KV work."],
      ["cache_control", "Instructions about how cache should behave. For example: whether to cache, how long to keep it, or what kind of cache entry it is."],
      ["cache_salt", "A namespace separator for cache entries. It prevents two users, tenants, or workloads from accidentally sharing the same cache entry even if their prompt text looks similar."],
      ["speculative_prefill", "A hint to prepare or precompute part of the prompt before it is definitely needed. It can reduce latency if the prediction is right."]
    ]
  }
];

function buildDefinitionSlide(p, section, slideNumber) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addText(slide, section.title, 56, 42, 1500, 56, { size: 38, bold: true });
  addText(slide, section.subtitle, 56, 104, 1540, 38, { size: 22, color: C.sub });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 56, top: 158, width: 1806, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.rule, width: 1 },
  });

  const top = 190;
  const left = 72;
  const tableW = 1776;
  const headerH = 50;
  const rowH = Math.floor(740 / section.items.length);
  const signalW = 350;
  addRect(slide, left, top, tableW, headerH, C.head, C.grid, 1);
  addText(slide, "Signal", left + 14, top + 13, signalW - 28, 24, { size: 18, bold: true });
  addText(slide, "Meaning", left + signalW + 14, top + 13, tableW - signalW - 28, 24, { size: 18, bold: true });

  section.items.forEach((item, i) => {
    const y = top + headerH + i * rowH;
    addRect(slide, left, y, tableW, rowH, i % 2 === 0 ? C.white : C.alt, C.grid, 1);
    addText(slide, item[0], left + 14, y + 14, signalW - 28, rowH - 22, { size: 19, bold: true, font: "Courier New" });
    addText(slide, item[1], left + signalW + 14, y + 12, tableW - signalW - 28, rowH - 22, { size: 20, color: C.ink });
  });

  addDeckFooter(slide, slideNumber, "Simple-English definitions generated from the focused signal overview.");
  slide.speakerNotes.textFrame.setText("[Sources]\n- User request to explain the focused serving-plane signals in simple English.\n");
}

function buildSchemaSlide(p, slideNumber) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addText(slide, "Recommended canonical serving-control schema", 56, 42, 1500, 56, { size: 38, bold: true });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 56, top: 122, width: 1806, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  addText(slide, "Use this as the normalized object, then map each harness cell to the nearest available field, pass-through path, or unsupported marker.", 66, 156, 1510, 38, { size: 23, color: C.sub });
  const schema = `{
  "scheduling": {
    "priority": "...",
    "strict_priority": "...",
    "latency_sensitivity": "...",
    "qos_tier": "...",
    "expected_output_length": "...",
    "expected_interarrival_time": "...",
    "predicted_remaining_calls": "..."
  },
  "cache": {
    "key": "...",
    "namespace": "...",
    "ttl": "...",
    "entry_type": "...",
    "eviction_priority": "...",
    "speculative_prefill": true,
    "hit_feedback": "..."
  }
}`;
  addRect(slide, 82, 238, 980, 676, C.alt, C.grid, 1);
  addText(slide, schema, 108, 264, 928, 620, { size: 19, color: C.ink, font: "Courier New" });
  const rules = [
    ["Native", "Map to the exact field/path."],
    ["Emits", "Keep the emitted field but note backend consumption separately."],
    ["Pass-through", "Use {\"via\":\"pass_through\"}; do not mark as native support."],
    ["Custom", "Use {\"via\":\"middleware\"}; implementation belongs outside the harness."],
    ["Provider-dependent", "Use {\"via\":\"provider\"}; attach provider constraints later."],
    ["Unsupported", "Use '-' in slides; use support:'none' in machine-readable registries."]
  ];
  addText(slide, "Mapping rules", 1120, 252, 420, 34, { size: 26, bold: true });
  rules.forEach((rule, i) => {
    const y = 312 + i * 86;
    addText(slide, rule[0], 1120, y, 330, 24, { size: 20, bold: true });
    addText(slide, rule[1], 1120, y + 30, 560, 42, { size: 17, color: C.sub });
  });
  addDeckFooter(slide, slideNumber, "Recommendation artifact generated from the focused serving-plane signal set.");
  slide.speakerNotes.textFrame.setText("[Sources]\n- User request and previously generated signal tables.\n");
}

function buildSlide(p, spec, slideNumber, sourcePath) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  addText(slide, spec.title, 56, 42, 1420, 56, { size: 38, bold: true });
  if (spec.subtitle) addText(slide, spec.subtitle, 56, 104, 1200, 36, { size: 22, color: C.sub });
  slide.shapes.add({
    geometry: "straightConnector1",
    position: { left: 56, top: spec.subtitle ? 156 : 122, width: 1806, height: 0 },
    fill: "none",
    line: { style: "solid", fill: C.rule, width: 1 },
  });

  const top = spec.subtitle ? 178 : 150;
  const left = 56;
  const tableW = 1806;
  const availableH = 850 - (spec.subtitle ? 20 : 0);
  const headerH = 64;
  const rowH = Math.floor((availableH - headerH) / spec.rows.length);
  const firstW = spec.columns.length <= 3 ? 310 : (spec.columns.length <= 5 ? 265 : (spec.columns[0].length > 9 ? 245 : 225));
  const restW = (tableW - firstW) / (spec.columns.length - 1);
  const widths = [firstW, ...Array(spec.columns.length - 1).fill(restW)];

  let x = left;
  spec.columns.forEach((col, i) => {
    drawCell(slide, col, x, top, widths[i], headerH, { fill: C.head, bold: true, size: i === 0 ? 21 : 20, font: "Helvetica Neue" });
    x += widths[i];
  });

  spec.rows.forEach((row, r) => {
    let cx = left;
    row.forEach((value, c) => {
      const fill = r % 2 === 0 ? C.white : C.alt;
      drawCell(slide, value, cx, top + headerH + r * rowH, widths[c], rowH, {
        fill,
        bold: c === 0,
        size: c === 0 ? 18 : 16,
        font: c === 0 ? "Helvetica Neue" : undefined,
        color: value === "None" || value === "N/A" || value === "None identified" ? C.muted : C.ink,
        compact: c > 0,
        signalLabel: row[0],
        harness: spec.columns[c],
        compactDescSize: spec.columns.length <= 3 ? 18 : (spec.columns.length <= 5 ? 16.5 : 16),
        compactJsonSize: spec.columns.length <= 3 ? 15 : (spec.columns.length <= 5 ? 14 : 13.4),
      });
      cx += widths[c];
    });
  });

  addFooter(slide, slideNumber, sourcePath);
}

async function main() {
  await fs.mkdir(TMP, { recursive: true });
  const p = Presentation.create({ slideSize: { width: W, height: H } });
  buildLegendSlide(p);
  buildOverviewSlide(p);
  buildHarnessAdditionsSlide(p, 3);

  let slideNumber = 4;
  buildDefinitionSlide(p, definitionSections[0], slideNumber++);
  [...focusedSchedulingSlides, ...additionalSchedulingSlides].forEach((entry) => {
    buildSlide(p, entry.spec, slideNumber++, entry.source);
  });

  buildDefinitionSlide(p, definitionSections[1], slideNumber++);
  [...focusedCacheSlides, ...additionalCacheSlides].forEach((entry) => {
    buildSlide(p, entry.spec, slideNumber++, entry.source);
  });

  buildSchemaSlide(p, slideNumber++);

  for (const [index, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await p.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(`${TMP}/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
  }

  const montage = await p.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(`${TMP}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
  console.log(OUT);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
