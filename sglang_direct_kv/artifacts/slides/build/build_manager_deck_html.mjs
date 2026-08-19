import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "../../../..");
const slidesDir = path.join(repoRoot, "sglang_direct_kv/artifacts/slides");
const imgDir = path.join(slidesDir, "images");
const outPath = path.join(slidesDir, "agent_aware_kv_movement_manager_deck.html");

const PROJECT_GOAL =
  "Coding agents naturally pause during tool calls, then resume with tight latency expectations. Current memory movement paths are largely context-agnostic: they can move KV pages, but they do not know which agent needs them, when they are needed, or whether missing the deadline will stall replay. This project quantifies that gap and prototypes hint-guided KV movement as a path toward smarter DMA engines, KV-aware memory scheduling, and hardware/runtime co-design.";

async function dataUrl(name) {
  const filePath = path.join(imgDir, name);
  const bytes = await fs.readFile(filePath);
  return `data:image/png;base64,${bytes.toString("base64")}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function bullets(items) {
  return `<ul class="bullets">${items
    .map((item) => `<li style="--dot:${item.color || "#2563eb"}">${escapeHtml(item.text)}</li>`)
    .join("")}</ul>`;
}

function stat(value, label, className = "") {
  return `<div class="stat ${className}"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`;
}

function impactTable(headers, rows, className = "") {
  return `<table class="impact-table ${className}">
    <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
    <tbody>${rows
      .map(
        (row) =>
          `<tr>${row
            .map(
              (cell) =>
                `<td${cell.className ? ` class="${escapeHtml(cell.className)}"` : ""}>${escapeHtml(cell.text)}</td>`,
            )
            .join("")}</tr>`,
      )
      .join("")}</tbody>
  </table>`;
}

function flow(labels) {
  return `<div class="plain-flow">${labels.map(escapeHtml).join('<span>-&gt;</span>')}</div>`;
}

function flowDiagram(labels) {
  return `<div class="flow-diagram">${labels
    .map((label, index) => {
      const arrow = index < labels.length - 1 ? '<div class="flow-arrow">-&gt;</div>' : "";
      return `<div class="flow-step step-${index}">${escapeHtml(label)}</div>${arrow}`;
    })
    .join("")}</div>`;
}

function slide({ title, subtitle = "", body, source = "", number }) {
  void source;
  return `<section class="slide" id="slide-${number}">
    <h1>${escapeHtml(title)}</h1>
    ${subtitle ? `<p class="subtitle">${escapeHtml(subtitle)}</p>` : ""}
    <div class="slide-body">${body}</div>
    <div class="footer">Agent-aware KV movement</div>
    <div class="slide-number">${String(number).padStart(2, "0")}</div>
  </section>`;
}

async function main() {
  const readable = await dataUrl("readable_phase_timeline_4rows_wide.png");
  const globalPrefetch = await dataUrl("global_prefetch_margin_backup.png");
  const syntheticMechanism = await dataUrl("synthetic_profiled_mechanism_timeline_compact.png");
  const h2dReadiness = await dataUrl("global_h2d_readiness.png");
  const queueTimeline = await dataUrl("replay_queue_timeline.png");

  const slides = [
    slide({
      number: 1,
      title: "Making KV Movement Agent-Aware",
      subtitle: PROJECT_GOAL,
      body: `
        <hr>
        ${bullets([
          { text: "Core problem: memory movement sees bytes, not agent deadlines.", color: "#dc2626" },
          { text: "Research target: make KV movement deadline-aware and session-aware.", color: "#2563eb" },
          { text: "Hardware angle: smarter DMA queues, KV metadata, residency protection, and telemetry.", color: "#16a34a" },
        ])}
      `,
    }),
    slide({
      number: 2,
      title: "The testbed has controlled and real traffic",
      subtitle: "Controlled runs expose mechanisms; real traces keep the workload credible.",
      body: `
        ${bullets([
          { text: "Controlled synthetic path: prompt size, tool wait, filler pressure, and replay deadline are known.", color: "#2563eb" },
          { text: "Real coding-agent path: SWE-bench / AgentBench traces feed DeepAgents and SGLang.", color: "#9333ea" },
          { text: "Both paths measure the same question: was useful KV ready before replay?", color: "#16a34a" },
        ])}
        ${flow(["agent task", "tool wait", "replay", "KV ready?"])}
      `,
      source: "Source: latest_master_report.html, Experiment Setup And Manager Summary",
    }),
    slide({
      number: 3,
      title: "Experiment Testbed Setup",
      subtitle: "The real-request path is intentionally simple.",
      body: `
        ${flowDiagram(["SWE-bench traces", "DeepAgents tool loop", "SGLang server", "KV/cache observations"])}
        ${bullets([
          { text: "DeepAgents produces model turns and tool-call gaps.", color: "#9333ea" },
          { text: "SGLang serves the model and exposes KV/cache behavior through our hooks.", color: "#0891b2" },
          { text: "Reports compare replay timing, H2D movement, recompute, and prefetch readiness.", color: "#16a34a" },
        ])}
      `,
      source: "Source: latest_master_report.html, Experiment Setup And Manager Summary",
    }),
    slide({
      number: 4,
      title: "Today’s DMA engines move bytes, not intent",
      subtitle: "That limits how well software hints can be enforced.",
      body: `
        ${bullets([
          { text: "They do not know which KV belongs to which agent session.", color: "#2563eb" },
          { text: "They do not know when the agent will replay after a tool call.", color: "#9333ea" },
          { text: "They do not know whether evicting prefetched KV wastes the hint.", color: "#dc2626" },
          { text: "They do not expose enough semantic telemetry: useful, late, wasted, or evicted-before-use.", color: "#16a34a" },
        ])}
        <hr>
        <p class="takeaway">Hardware opportunity: add KV/session context to the memory movement path.</p>
      `,
    }),
    slide({
      number: 5,
      title: "Replay path exposes the bottleneck",
      subtitle: "Each row is one tool-gap replay; the replay path shows where time goes before the first token.",
      body: `
        <img class="chart full teaser" src="${readable}" alt="Readable phase timeline crop showing G00 through G03">
        <p class="ellipsis-note">... more replay gaps observed in the full report</p>
      `,
      source: "Source: latest_master_report.html controlled run",
    }),
    slide({
      number: 6,
      title: "Software prefetch often missed replay",
      subtitle: "In the live prefetch-margin run, almost every hint completed after replay was due.",
      body: `
        <div class="chart-row">
          <img class="chart" src="${globalPrefetch}" alt="Global prefetch margin dot chart from backup report">
          <div class="stats">
            ${stat("112 / 114", "prefetch attempts were late", "danger")}
            ${stat("98.25%", "missed the replay deadline", "warn")}
            <p class="takeaway small">Meaning: hints alone are not enough when the movement path cannot act predictably.</p>
          </div>
        </div>
      `,
      source: "Source: backups/latest_master_report-1.html, Global Prefetch Margin",
    }),
    slide({
      number: 7,
      title: "Mechanism traces explain the miss",
      subtitle: "The hint existed, but KV readiness still landed too late for replay.",
      body: `
        <div class="chart-row">
          <img class="chart" src="${syntheticMechanism}" alt="Synthetic profiled mechanism timeline">
          <div class="stats">
            ${stat("0 / 6", "ready before replay", "danger")}
            ${stat("3 / 6", "visible CUDA HtoD", "cyan")}
            ${stat("6 / 6", "replay reloaded KV", "warn")}
          </div>
        </div>
        <p class="note">Use this as mechanism evidence, not clean TTFT evidence.</p>
      `,
      source: "Source: latest_synthetic_master_report.html, Profiled Mechanism Timelines",
    }),
    slide({
      number: 8,
      title: "Replay-side KV loads missed the deadline",
      subtitle: "For this controlled no-prefetch run, visible H2D loads finished after replay was due.",
      body: `
        <div class="chart-row">
          <img class="chart" src="${h2dReadiness}" alt="Global replay H2D readiness dot chart">
          <div class="stats">
            ${stat("8 / 8", "visible replay-side H2D loads finished late", "danger")}
            <p class="takeaway small">Meaning: memory movement happened, but not early enough for the replay deadline.</p>
          </div>
        </div>
      `,
      source: "Source: latest_master_report.html, Global Replay H2D Readiness",
    }),
    slide({
      number: 9,
      title: "The delay is not just copy time",
      subtitle: "The request passes through normal runtime scheduling before visible KV H2D begins.",
      body: `
        <img class="chart full" src="${queueTimeline}" alt="Replay queue timeline versus H2D start chart">
        <p class="note">Stage markers separate client submit, SGLang receive, scheduler queue/admit, H2D start, and H2D finish.</p>
      `,
      source: "Source: latest_master_report.html, Replay Queue Timeline vs H2D Start",
    }),
    slide({
      number: 10,
      title: "Key findings make the hardware case",
      subtitle: "The pattern points to missing context, deadlines, residency control, and telemetry.",
      body: `
        ${bullets([
          { text: "Agent tool gaps can be very short.", color: "#2563eb" },
          { text: "Correct hints can still finish late.", color: "#9333ea" },
          { text: "Visible copy time is only part of end-to-end delay.", color: "#0891b2" },
          { text: "KV can be written to host, evicted from GPU, then lost from host before replay.", color: "#ca8a04" },
          { text: "Replay-side H2D movement can miss the deadline even when the replay request exists.", color: "#dc2626" },
        ])}
        <hr>
        <p class="takeaway">Conclusion: this is a scheduling and enforceability problem, not just a bandwidth problem.</p>
      `,
      source: "Source: project traces and latest master/synthetic report observations",
    }),
    slide({
      number: 11,
      title: "Potential hardware impact",
      subtitle: "Simple targets for tool-heavy coding-agent workloads.",
      body: `
        <div class="impact-bullets">
          <div class="impact-item blue">
            <div class="impact-number">1.</div>
            <div>
              <h2>Faster agent resume after tools</h2>
              <p class="target">Target: 10-30% lower post-tool latency</p>
              <p>Why: our traces show the next model turn can wait because useful KV is not ready when the agent resumes.</p>
            </div>
          </div>
          <div class="impact-item purple">
            <div class="impact-number">2.</div>
            <div>
              <h2>Fewer wasted memory movements</h2>
              <p class="target">Target: 20-50% fewer late or wasted KV reloads</p>
              <p>Why: memory movement can finish too late, or useful KV can be moved but not reused in time.</p>
            </div>
          </div>
          <div class="impact-item green">
            <div class="impact-number">3.</div>
            <div>
              <h2>More predictable tail latency</h2>
              <p class="target">Target: lower p95/p99 agent-resume stalls</p>
              <p>Why: urgent agent KV currently competes with ordinary memory traffic without deadline or priority context.</p>
            </div>
          </div>
        </div>
        <hr class="impact-rule">
        <p class="impact-support"><strong>Hardware support to test:</strong> KV/session metadata, deadline-aware movement queues, temporary KV protection, and useful/late/wasted telemetry.</p>
      `,
    }),
    slide({
      number: 12,
      title: "Hardware support can make hints enforceable",
      subtitle: "Treat KV as deadline-sensitive memory, not generic bytes.",
      body: `
        ${bullets([
          { text: "KV page/session metadata: which agent owns this KV and how urgent is it?", color: "#2563eb" },
          { text: "Deadline-aware migration queue: move urgent KV before less urgent traffic.", color: "#9333ea" },
          { text: "Residency protection: keep useful prefetched KV from being evicted too early.", color: "#16a34a" },
          { text: "Telemetry: count useful, late, wasted, and evicted-before-use movement.", color: "#0891b2" },
        ])}
        <hr>
        <p class="takeaway">Research question: how much replay latency and wasted movement can be avoided when KV movement has session context, priority, deadline, protection, and telemetry?</p>
      `,
    }),
  ];

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent-aware KV Movement Manager Deck</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #0f172a;
      --body: #334155;
      --muted: #64748b;
      --line: #cbd5e1;
      --paper: #ffffff;
      --bg: #eef2f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--body);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 22px;
      background: rgba(255, 255, 255, 0.94);
      border-bottom: 1px solid #dbe3ee;
      backdrop-filter: blur(8px);
    }
    .toolbar strong { color: var(--ink); }
    .toolbar span { color: var(--muted); font-size: 14px; }
    .toolbar nav { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .toolbar a {
      color: var(--body);
      text-decoration: none;
      border-bottom: 2px solid #cbd5e1;
      padding: 3px 5px;
      font-size: 13px;
    }
    main {
      width: min(1280px, 96vw);
      margin: 24px auto 80px;
      display: grid;
      gap: 28px;
    }
    .slide {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      padding: 38px 44px;
      background: var(--paper);
      border: 1px solid #d5dde8;
      box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
      overflow: hidden;
    }
    h1 {
      margin: 0;
      max-width: 1128px;
      color: var(--ink);
      font-size: clamp(34px, 4.0vw, 54px);
      line-height: 1.02;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 14px 0 0;
      max-width: 1110px;
      color: var(--muted);
      font-size: clamp(16px, 1.55vw, 21px);
      line-height: 1.35;
    }
    .slide-body {
      margin-top: 34px;
      height: calc(100% - 194px);
    }
    hr {
      border: 0;
      border-top: 1px solid var(--line);
      margin: 28px 0;
    }
    .bullets {
      list-style: none;
      margin: 0;
      padding: 0 0 0 38px;
      display: grid;
      gap: 28px;
      max-width: 1080px;
      font-size: 25px;
      line-height: 1.28;
    }
    .bullets li {
      position: relative;
    }
    .bullets li::before {
      content: "•";
      position: absolute;
      left: -34px;
      top: -2px;
      color: var(--dot);
      font-size: 32px;
      font-weight: 800;
      line-height: 1;
    }
    .plain-flow {
      margin: 44px auto 0;
      max-width: 1120px;
      text-align: center;
      color: var(--ink);
      font-size: clamp(26px, 3.2vw, 40px);
      font-weight: 800;
      line-height: 1.35;
    }
    .plain-flow span {
      color: var(--muted);
      padding: 0 22px;
    }
    .flow-diagram {
      display: grid;
      grid-template-columns: 1fr 36px 1fr 36px 1fr 36px 1fr;
      gap: 12px;
      align-items: center;
      margin: 54px auto 48px;
      max-width: 1128px;
    }
    .flow-step {
      min-height: 104px;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--ink);
      font-size: 22px;
      font-weight: 800;
      line-height: 1.16;
      border: 2px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .step-0 { border-color: #2563eb; }
    .step-1 { border-color: #9333ea; }
    .step-2 { border-color: #0891b2; }
    .step-3 { border-color: #16a34a; }
    .flow-arrow {
      text-align: center;
      color: var(--muted);
      font-size: 30px;
      font-weight: 800;
    }
    .takeaway {
      margin: 0;
      color: var(--ink);
      font-size: 24px;
      line-height: 1.28;
      font-weight: 800;
      text-align: center;
    }
    .takeaway.small {
      text-align: left;
      font-size: 21px;
      margin-top: 8px;
    }
    .takeaway.left {
      text-align: left;
      max-width: 1120px;
      font-size: 22px;
    }
    .impact-bullets {
      display: grid;
      gap: 24px;
      margin: 6px 18px 0 18px;
    }
    .impact-item {
      display: grid;
      grid-template-columns: 44px 1fr;
      gap: 18px;
      padding-bottom: 20px;
      border-bottom: 1px solid #e2e8f0;
    }
    .impact-item:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }
    .impact-number {
      font-size: 26px;
      line-height: 1.05;
      font-weight: 900;
    }
    .impact-item h2 {
      margin: 0 0 8px;
      color: var(--ink);
      font-size: 27px;
      line-height: 1.12;
    }
    .impact-item p {
      margin: 0;
      max-width: 1060px;
      color: var(--body);
      font-size: 18px;
      line-height: 1.24;
    }
    .impact-item .target {
      margin-bottom: 8px;
      font-size: 21px;
      font-weight: 850;
    }
    .impact-rule {
      margin: 22px 0 22px;
    }
    .impact-support {
      margin: 0;
      max-width: 1120px;
      color: var(--ink);
      font-size: 20px;
      line-height: 1.2;
    }
    .impact-item.blue .impact-number,
    .impact-item.blue .target {
      color: #2563eb;
    }
    .impact-item.purple .impact-number,
    .impact-item.purple .target {
      color: #9333ea;
    }
    .impact-item.green .impact-number,
    .impact-item.green .target {
      color: #16a34a;
    }
    .chart {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .chart.full {
      display: block;
      height: 472px;
      width: 100%;
    }
    .chart.full.teaser {
      height: 438px;
    }
    .ellipsis-note {
      margin: 8px 0 0;
      text-align: center;
      color: var(--muted);
      font-size: 24px;
      line-height: 1.1;
      font-weight: 800;
    }
    .chart-row {
      display: grid;
      grid-template-columns: 1fr 270px;
      gap: 34px;
      height: 410px;
      align-items: center;
    }
    .stats {
      display: grid;
      gap: 26px;
      align-content: center;
    }
    .stat strong {
      display: block;
      color: #dc2626;
      font-size: 45px;
      line-height: 1;
    }
    .stat.warn strong { color: #c2410c; }
    .stat.cyan strong { color: #0891b2; }
    .stat span {
      display: block;
      margin-top: 10px;
      color: var(--body);
      font-size: 20px;
      line-height: 1.2;
    }
    .note {
      margin: 12px 0 0;
      text-align: center;
      color: var(--muted);
      font-size: 18px;
    }
    .impact-table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      color: var(--body);
      font-size: 16px;
      line-height: 1.18;
    }
    .impact-table th {
      background: #eef2ff;
      color: var(--ink);
      text-align: left;
      font-size: 15px;
      line-height: 1.1;
      padding: 11px 12px;
      border: 1px solid #cbd5e1;
    }
    .impact-table td {
      padding: 12px;
      border: 1px solid #cbd5e1;
      vertical-align: middle;
    }
    .impact-table tbody tr:nth-child(even) td {
      background: #f8fafc;
    }
    .impact-table.compact {
      margin-top: 6px;
      font-size: 15px;
    }
    .impact-table.compact th {
      background: #f1f5f9;
      padding: 9px 12px;
      font-size: 14px;
    }
    .impact-table.compact td {
      padding: 9px 12px;
    }
    .impact-table .strong {
      font-weight: 800;
      color: var(--ink);
    }
    .impact-table .blue { color: #2563eb; }
    .impact-table .red { color: #dc2626; }
    .impact-table .purple { color: #9333ea; }
    .impact-table .green { color: #16a34a; }
    .impact-table .cyan { color: #0891b2; }
    .impact-table .gold { color: #ca8a04; }
    .mini-heading {
      margin: 18px 0 8px;
      color: var(--ink);
      font-size: 22px;
      line-height: 1;
    }
    .table-note {
      margin-top: 16px;
      font-size: 20px;
    }
    .source {
      position: absolute;
      left: 44px;
      bottom: 74px;
      color: var(--muted);
      font-size: 12px;
    }
    .footer {
      position: absolute;
      left: 44px;
      bottom: 42px;
      color: var(--muted);
      font-size: 14px;
    }
    .slide-number {
      position: absolute;
      right: 44px;
      bottom: 42px;
      color: var(--muted);
      font-size: 14px;
    }
    @media print {
      body { background: #fff; }
      .toolbar { display: none; }
      main { width: 100%; margin: 0; gap: 0; }
      .slide {
        width: 100vw;
        height: 100vh;
        aspect-ratio: auto;
        border: none;
        box-shadow: none;
        page-break-after: always;
      }
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <div>
      <strong>Agent-aware KV Movement</strong>
      <span>HTML slide deck. Same content as the PowerPoint deck.</span>
    </div>
    <nav>${slides.map((_, index) => `<a href="#slide-${index + 1}">${index + 1}</a>`).join("")}</nav>
  </div>
  <main>
    ${slides.join("\n")}
  </main>
  <script>
    window.addEventListener("keydown", (event) => {
      const slides = [...document.querySelectorAll(".slide")];
      const current = slides.findIndex((slide) => slide.getBoundingClientRect().top > -window.innerHeight / 2);
      if (event.key === "ArrowRight" || event.key === "PageDown") {
        slides[Math.min(slides.length - 1, current + 1)]?.scrollIntoView({ behavior: "smooth" });
      }
      if (event.key === "ArrowLeft" || event.key === "PageUp") {
        slides[Math.max(0, current - 1)]?.scrollIntoView({ behavior: "smooth" });
      }
    });
  </script>
</body>
</html>`;

  await fs.writeFile(outPath, html, "utf8");
  console.log(outPath);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
