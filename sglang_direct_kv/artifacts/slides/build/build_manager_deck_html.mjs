import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "../../../..");
const slidesDir = path.join(repoRoot, "sglang_direct_kv/artifacts/slides");
const imgDir = path.join(slidesDir, "images");
const outPath = path.join(slidesDir, "agent_aware_kv_movement_manager_deck.html");

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

function flow(nodes) {
  return `<div class="flow">${nodes
    .map((node, index) => {
      const arrow = index < nodes.length - 1 ? '<div class="arrow">-&gt;</div>' : "";
      return `<div class="flow-node ${node.className || ""}"><strong>${escapeHtml(node.title)}</strong>${node.body ? `<span>${escapeHtml(node.body)}</span>` : ""}</div>${arrow}`;
    })
    .join("")}</div>`;
}

function evidenceCard(value, label, className = "") {
  return `<div class="evidence-card ${className}"><div class="evidence-value">${escapeHtml(value)}</div><div class="evidence-label">${escapeHtml(label)}</div></div>`;
}

function slide({ eyebrow = "Agent-aware KV movement", title, subtitle = "", body, source = "", number }) {
  return `<section class="slide" id="slide-${number}">
    <div class="slide-eyebrow">${escapeHtml(eyebrow)}</div>
    <h1>${escapeHtml(title)}</h1>
    ${subtitle ? `<p class="subtitle">${escapeHtml(subtitle)}</p>` : ""}
    <div class="slide-body">${body}</div>
    ${source ? `<div class="source">${escapeHtml(source)}</div>` : ""}
    <div class="slide-number">${String(number).padStart(2, "0")}</div>
  </section>`;
}

async function main() {
  const readable = await dataUrl("readable_phase_timeline_2rows.png");
  const globalPrefetch = await dataUrl("global_prefetch_margin_backup.png");
  const h2dReadiness = await dataUrl("global_h2d_readiness.png");
  const queueTimeline = await dataUrl("replay_queue_timeline.png");

  const slides = [
    slide({
      number: 1,
      title: "Making KV Movement Agent-Aware",
      subtitle:
        "Coding agents pause during tool calls, then resume with tight latency expectations. The question is whether the memory system can move the right KV before replay arrives.",
      body: `
        ${flow([
          { title: "model turn", className: "blue" },
          { title: "tool wait", className: "gray" },
          { title: "replay request", className: "red" },
          { title: "KV ready?", className: "green" },
        ])}
        <div class="goal-box">
          <h2>Project goal</h2>
          <p>Quantify the gap and prototype hint-guided KV movement as a path toward smarter DMA engines and runtime co-design.</p>
        </div>
      `,
    }),
    slide({
      number: 2,
      title: "Today's DMA engines move bytes, not intent",
      subtitle: "The hardware path is fast, but it usually lacks agent/session semantics.",
      body: `
        <div class="two-col">
          <div class="panel">
            <h2>Context-agnostic movement</h2>
            <ul>
              <li>sees memory ranges</li>
              <li>does not know the agent session</li>
              <li>does not know replay deadline</li>
              <li>does not protect useful prefetched KV</li>
            </ul>
          </div>
          <div class="panel blue-panel">
            <h2>Hint-aware movement</h2>
            <ul>
              <li>tags KV by session and priority</li>
              <li>schedules against replay deadlines</li>
              <li>moves hot KV before replay</li>
              <li>reports late, useful, and wasted movement</li>
            </ul>
          </div>
        </div>
      `,
    }),
    slide({
      number: 3,
      title: "The replay path exposes the bottleneck",
      subtitle: "One tool-gap row shows where replay spends time before the first useful token.",
      body: `
        <img class="chart-img wide" src="${readable}" alt="Readable phase timeline crop from latest master report">
        <p class="caption">Read left to right: model turn -> tool wait -> replay path. Cyan is host-to-device KV load; magenta/gold are work before first token.</p>
      `,
      source: "Source: latest_master_report.html controlled run",
    }),
    slide({
      number: 4,
      title: "Software prefetch often missed the agent replay deadline",
      subtitle: "Across live prefetch attempts, nearly every hint completed after the agent had already resumed.",
      body: `
        <div class="chart-with-callouts">
          <img class="chart-img" src="${globalPrefetch}" alt="Global prefetch margin dot chart from backup report">
          <div class="callout-stack">
            ${evidenceCard("112 of 114", "prefetch attempts were late", "danger")}
            ${evidenceCard("98.25%", "missed replay due deadline", "warning")}
            <p class="manager-note">Semantic knowledge alone is not enough if the movement path cannot act on hints predictably.</p>
          </div>
        </div>
      `,
      source: "Source: backups/latest_master_report-1.html, Global Prefetch Margin",
    }),
    slide({
      number: 5,
      title: "Replay-side KV loads missed the deadline in this run",
      subtitle: "The aggregate view shows whether KV H2D finished before or after replay was due.",
      body: `
        <div class="chart-with-callouts">
          <img class="chart-img" src="${h2dReadiness}" alt="Global replay H2D readiness dot chart">
          <div class="callout-stack">
            ${evidenceCard("8 / 8", "H2D loads finished late", "danger")}
            <p class="manager-note">All visible replay-side KV loads finished below the 0 ms deadline line.</p>
          </div>
        </div>
      `,
      source: "Source: latest_master_report.html, Global Replay H2D Readiness",
    }),
    slide({
      number: 6,
      title: "The delay is not just copy time",
      subtitle: "The request enters normal software/runtime scheduling before visible KV H2D begins.",
      body: `
        <img class="chart-img wide" src="${queueTimeline}" alt="Replay queue timeline versus H2D start chart">
        <p class="caption">The stage markers separate submission, SGLang receive, scheduler queue/admit, and visible KV H2D movement.</p>
      `,
      source: "Source: latest_master_report.html, Replay Queue Timeline vs H2D Start",
    }),
    slide({
      number: 7,
      title: "Hardware support can make hints enforceable",
      subtitle: "The opportunity is to treat KV as deadline-sensitive memory, not generic bytes.",
      body: `
        ${flow([
          { title: "Runtime hint", body: "session, priority, deadline", className: "blue" },
          { title: "KV-aware queue", body: "order urgent KV first", className: "purple" },
          { title: "DMA / copy engine", body: "throttle and prioritize", className: "cyan" },
          { title: "Residency control", body: "protect useful KV", className: "green" },
          { title: "Telemetry", body: "late / useful / wasted", className: "gold" },
        ])}
        <div class="research-question">
          <h2>Resulting research question</h2>
          <p>How much replay latency and wasted movement can be avoided when KV movement has session context, priority, deadline, protection, and telemetry?</p>
        </div>
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
      --blue: #2563eb;
      --purple: #a855f7;
      --cyan: #06b6d4;
      --green: #16a34a;
      --red: #ef4444;
      --gold: #eab308;
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
      background: rgba(255, 255, 255, 0.92);
      border-bottom: 1px solid #dbe3ee;
      backdrop-filter: blur(8px);
    }
    .toolbar strong { color: var(--ink); }
    .toolbar span { color: var(--muted); font-size: 14px; }
    .toolbar nav { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .toolbar a {
      color: var(--body);
      text-decoration: none;
      border: 1px solid #dbe3ee;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 13px;
      background: #fff;
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
      padding: 38px 42px;
      background: var(--paper);
      border: 1px solid #d5dde8;
      box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
      overflow: hidden;
    }
    .slide-eyebrow {
      position: absolute;
      left: 42px;
      bottom: 42px;
      color: var(--muted);
      font-size: 14px;
    }
    .slide-number {
      position: absolute;
      right: 42px;
      bottom: 42px;
      color: var(--muted);
      font-size: 14px;
    }
    h1 {
      margin: 0;
      max-width: 1110px;
      color: var(--ink);
      font-size: clamp(34px, 4.2vw, 56px);
      line-height: 0.98;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 18px 0 0;
      max-width: 1020px;
      color: var(--muted);
      font-size: clamp(17px, 1.8vw, 24px);
      line-height: 1.34;
    }
    .slide-body {
      margin-top: 28px;
      height: calc(100% - 188px);
    }
    .flow {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-top: 62px;
    }
    .flow-node {
      min-height: 82px;
      width: 170px;
      display: grid;
      place-items: center;
      text-align: center;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f8fafc;
      padding: 12px;
    }
    .flow-node strong {
      display: block;
      color: var(--ink);
      font-size: 18px;
      line-height: 1.15;
    }
    .flow-node span {
      display: block;
      margin-top: 7px;
      color: var(--body);
      font-size: 14px;
      line-height: 1.2;
    }
    .arrow { color: var(--muted); font-size: 30px; font-weight: 800; }
    .blue { background: #dbeafe; border-color: #93c5fd; }
    .purple { background: #f5f3ff; border-color: #c4b5fd; }
    .cyan { background: #ecfeff; border-color: #67e8f9; }
    .green { background: #dcfce7; border-color: #86efac; }
    .gray { background: #f1f5f9; border-color: #cbd5e1; }
    .red { background: #fee2e2; border-color: #fca5a5; }
    .gold { background: #fff7ed; border-color: #fed7aa; }
    .goal-box {
      position: absolute;
      right: 74px;
      top: 326px;
      width: 292px;
      min-height: 194px;
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 10px;
      padding: 24px;
    }
    .goal-box h2,
    .research-question h2,
    .panel h2 {
      margin: 0 0 14px;
      color: var(--ink);
      font-size: 25px;
      line-height: 1.1;
    }
    .goal-box p,
    .research-question p {
      margin: 0;
      font-size: 18px;
      line-height: 1.35;
    }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 60px;
      height: 360px;
      margin-top: 34px;
      padding: 0 16px;
    }
    .panel {
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 10px;
      padding: 34px 40px;
    }
    .blue-panel { background: #eff6ff; border-color: #bfdbfe; }
    .panel ul { margin: 20px 0 0; padding-left: 22px; font-size: 22px; line-height: 1.75; }
    .chart-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #fff;
    }
    .chart-img.wide {
      height: 450px;
    }
    .chart-with-callouts {
      display: grid;
      grid-template-columns: 1fr 270px;
      gap: 30px;
      height: 412px;
      align-items: center;
    }
    .callout-stack { display: grid; gap: 24px; align-content: start; }
    .evidence-card {
      min-height: 118px;
      border: 1px solid #fed7aa;
      border-radius: 10px;
      background: #fff7ed;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 18px;
    }
    .evidence-card.danger { background: #fef2f2; border-color: #fecaca; }
    .evidence-value {
      color: #b91c1c;
      font-size: 44px;
      font-weight: 800;
      line-height: 1;
    }
    .evidence-card.warning .evidence-value { color: #c2410c; }
    .evidence-label {
      margin-top: 12px;
      color: var(--body);
      font-size: 19px;
      line-height: 1.2;
    }
    .manager-note {
      margin: 0;
      color: var(--body);
      font-size: 22px;
      line-height: 1.22;
    }
    .caption {
      margin: 14px 30px 0;
      color: var(--body);
      font-size: 18px;
    }
    .source {
      position: absolute;
      left: 42px;
      bottom: 74px;
      color: var(--muted);
      font-size: 12px;
    }
    .research-question {
      margin: 72px auto 0;
      width: 860px;
      min-height: 84px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 10px;
      display: grid;
      grid-template-columns: 290px 1fr;
      gap: 24px;
      align-items: center;
      padding: 20px 26px;
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
      <span>HTML slide deck. Refresh after rebuilding from latest report charts.</span>
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
