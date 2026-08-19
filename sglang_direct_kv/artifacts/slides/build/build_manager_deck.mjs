import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const repoRoot = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware";
const outDir = path.join(repoRoot, "sglang_direct_kv/artifacts/slides");
const buildDir = path.join(outDir, "build");
const imgDir = path.join(outDir, "images");
const finalPptx = path.join(outDir, "agent_aware_kv_movement_manager_deck.pptx");
const reportPath = path.join(repoRoot, "sglang_direct_kv/artifacts/results/latest_master_report.html");
const syntheticReportPath = path.join(repoRoot, "sglang_direct_kv/artifacts/results/latest_synthetic_master_report.html");
const backupReportPath = path.join(repoRoot, "backups/latest_master_report-1.html");

const W = 1280;
const H = 720;
const C = {
  ink: "#0f172a",
  body: "#334155",
  muted: "#64748b",
  rule: "#cbd5e1",
  blue: "#2563eb",
  purple: "#9333ea",
  cyan: "#0891b2",
  green: "#16a34a",
  red: "#dc2626",
  gold: "#ca8a04",
};

const PROJECT_GOAL =
  "Coding agents naturally pause during tool calls, then resume with tight latency expectations. Current memory movement paths are largely context-agnostic: they can move KV pages, but they do not know which agent needs them, when they are needed, or whether missing the deadline will stall replay. This project quantifies that gap and prototypes hint-guided KV movement as a path toward smarter DMA engines, KV-aware memory scheduling, and hardware/runtime co-design.";

async function imageBytes(name) {
  const bytes = await fs.readFile(path.join(imgDir, name));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opts.size ?? 20,
    bold: opts.bold ?? false,
    color: opts.color ?? C.body,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
  };
  return shape;
}

function addTitle(slide, title, subtitle = "") {
  addText(slide, title, 44, 34, 1128, 58, { size: 39, bold: true, color: C.ink });
  if (subtitle) addText(slide, subtitle, 46, 96, 1100, 48, { size: 20, color: C.muted });
}

function addFooter(slide, number) {
  addText(slide, "Agent-aware KV movement", 44, 672, 380, 22, { size: 13, color: C.muted });
  addText(slide, String(number).padStart(2, "0"), 1188, 672, 46, 22, { size: 13, color: C.muted, align: "right" });
}

function addSource(slide, source) {
  // Keep source provenance in speaker notes, not on the slide canvas.
  void slide;
  void source;
}

function addRule(slide, top, color = C.rule) {
  slide.shapes.add({
    geometry: "rect",
    position: { left: 44, top, width: 1192, height: 1.2 },
    fill: color,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addBox(slide, text, left, top, width, height, opts = {}) {
  slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height },
    fill: opts.fill ?? "#ffffff",
    line: { style: "solid", fill: opts.line ?? C.rule, width: opts.width ?? 1.4 },
    borderRadius: "rounded-md",
  });
  addText(slide, text, left + 14, top + 18, width - 28, height - 28, {
    size: opts.size ?? 21,
    bold: true,
    color: opts.color ?? C.ink,
    align: "center",
    valign: "middle",
  });
}

function addFlowDiagram(slide, labels, left, top, width) {
  const gap = 34;
  const arrowW = 28;
  const boxW = (width - gap * (labels.length - 1) - arrowW * (labels.length - 1)) / labels.length;
  labels.forEach((label, idx) => {
    const x = left + idx * (boxW + gap + arrowW);
    addBox(slide, label, x, top, boxW, 104, { line: idx === 0 ? C.blue : idx === 1 ? C.purple : idx === 2 ? C.cyan : C.green });
    if (idx < labels.length - 1) {
      addText(slide, "→", x + boxW + 8, top + 28, arrowW + 18, 40, { size: 30, bold: true, color: C.muted, align: "center" });
    }
  });
}

function addBullets(slide, items, left, top, width, opts = {}) {
  const gap = opts.gap ?? 52;
  const size = opts.size ?? 24;
  items.forEach((item, idx) => {
    const y = top + idx * gap;
    addText(slide, "•", left, y - 2, 20, 32, { size: size + 2, bold: true, color: item.color ?? C.blue });
    addText(slide, item.text, left + 30, y, width - 30, opts.height ?? 42, {
      size,
      bold: item.bold ?? false,
      color: item.textColor ?? C.body,
    });
  });
}

function addNumberLine(slide, value, label, left, top, color) {
  addText(slide, value, left, top, 190, 48, { size: 38, bold: true, color });
  addText(slide, label, left, top + 48, 250, 42, { size: 19, color: C.body });
}

function addImpactTable(slide, rows, left, top, colWidths, rowHeight, opts = {}) {
  const headers = opts.headers ?? [];
  const headerHeight = opts.headerHeight ?? 40;
  const totalWidth = colWidths.reduce((sum, width) => sum + width, 0);
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width: totalWidth, height: headerHeight },
    fill: opts.headerFill ?? "#eef2ff",
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  let x = left;
  headers.forEach((header, idx) => {
    addText(slide, header, x + 9, top + 10, colWidths[idx] - 18, headerHeight - 12, {
      size: opts.headerSize ?? 15,
      bold: true,
      color: opts.headerColor ?? C.ink,
      valign: "middle",
    });
    x += colWidths[idx];
  });
  rows.forEach((row, rowIdx) => {
    const y = top + headerHeight + rowIdx * rowHeight;
    slide.shapes.add({
      geometry: "rect",
      position: { left, top: y, width: totalWidth, height: rowHeight },
      fill: rowIdx % 2 === 0 ? "#ffffff" : "#f8fafc",
      line: { style: "solid", fill: C.rule, width: 0.8 },
    });
    let cellX = left;
    row.forEach((cell, cellIdx) => {
      const color = cell.color ?? C.body;
      addText(slide, cell.text, cellX + 9, y + 9, colWidths[cellIdx] - 18, rowHeight - 14, {
        size: cell.size ?? opts.size ?? 15,
        bold: cell.bold ?? false,
        color,
        valign: "middle",
      });
      cellX += colWidths[cellIdx];
    });
  });
}

async function addImage(slide, name, left, top, width, height, alt, fit = "contain") {
  slide.images.add({
    blob: await imageBytes(name),
    contentType: "image/png",
    alt,
    fit,
    position: { left, top, width, height },
  });
}

function addPlainFlow(slide, labels, left, top, width, opts = {}) {
  const size = opts.size ?? 31;
  const text = labels.join("   →   ");
  addText(slide, text, left, top, width, opts.height ?? 62, {
    size,
    bold: true,
    color: opts.color ?? C.ink,
    align: opts.align ?? "center",
    valign: "middle",
  });
}

function addNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText([
    ...lines,
    "",
    "[Sources]",
    `Local report: ${reportPath}`,
    `Synthetic report: ${syntheticReportPath}`,
    `Backup report: ${backupReportPath}`,
  ]);
  slide.speakerNotes.setVisible(true);
}

async function main() {
  await fs.mkdir(outDir, { recursive: true });
  await fs.mkdir(path.join(buildDir, "rendered"), { recursive: true });
  await fs.writeFile(
    path.join(buildDir, "source-notes.txt"),
    `Deck uses local report evidence from ${reportPath}, ${syntheticReportPath}, and ${backupReportPath}.\n`,
    "utf8",
  );

  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addText(slide, "Making KV Movement Agent-Aware", 44, 58, 1030, 124, { size: 56, bold: true, color: C.ink });
    addText(slide, PROJECT_GOAL, 48, 190, 1090, 142, { size: 22, color: C.body });
    addRule(slide, 372, C.rule);
    addBullets(
      slide,
      [
        { text: "Core problem: memory movement sees bytes, not agent deadlines.", color: C.red },
        { text: "Research target: make KV movement deadline-aware and session-aware.", color: C.blue },
        { text: "Hardware angle: smarter DMA queues, KV metadata, residency protection, and telemetry.", color: C.green },
      ],
      92,
      422,
      980,
      { size: 24, gap: 58 },
    );
    addFooter(slide, 1);
    addNotes(slide, [
      "Opening slide. Keep the framing hardware-facing: software can provide hints, but hardware/runtime support can make those hints enforceable.",
      `Goal: ${PROJECT_GOAL}`,
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "The testbed has controlled and real traffic", "Controlled runs expose mechanisms; real traces keep the workload credible.");
    addBullets(
      slide,
      [
        { text: "Controlled synthetic path: prompt size, tool wait, filler pressure, and replay deadline are known.", color: C.blue },
        { text: "Real coding-agent path: SWE-bench / AgentBench traces feed DeepAgents and SGLang.", color: C.purple },
        { text: "Both paths measure the same question: was useful KV ready before replay?", color: C.green },
      ],
      84,
      188,
      1060,
      { size: 25, gap: 74, height: 58 },
    );
    addPlainFlow(slide, ["agent task", "tool wait", "replay", "KV ready?"], 96, 476, 1060, { size: 32 });
    addSource(slide, "Source: latest_master_report.html, Experiment Setup And Manager Summary");
    addFooter(slide, 2);
    addNotes(slide, [
      "This slide replaces the earlier card-heavy setup with plain points.",
      "Synthetic runs give clean control. Real DeepAgents/SWE-bench traces preserve coding-agent behavior.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Experiment Testbed Setup", "The real-request path is intentionally simple.");
    addFlowDiagram(
      slide,
      ["SWE-bench\ntraces", "DeepAgents\ntool loop", "SGLang\nserver", "KV/cache\nobservations"],
      76,
      204,
      1128,
    );
    addBullets(
      slide,
      [
        { text: "DeepAgents produces model turns and tool-call gaps.", color: C.purple },
        { text: "SGLang serves the model and exposes KV/cache behavior through our hooks.", color: C.cyan },
        { text: "Reports compare replay timing, H2D movement, recompute, and prefetch readiness.", color: C.green },
      ],
      120,
      410,
      980,
      { size: 23, gap: 52 },
    );
    addSource(slide, "Source: latest_master_report.html, Experiment Setup And Manager Summary");
    addFooter(slide, 3);
    addNotes(slide, [
      "This slide is deliberately sparse. It should be easy to explain in one sentence.",
      "The user asked to remove extra hardware/context-length details from this setup slide.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Today’s DMA engines move bytes, not intent", "That limits how well software hints can be enforced.");
    addBullets(
      slide,
      [
        { text: "They do not know which KV belongs to which agent session.", color: C.blue },
        { text: "They do not know when the agent will replay after a tool call.", color: C.purple },
        { text: "They do not know whether evicting prefetched KV wastes the hint.", color: C.red },
        { text: "They do not expose enough semantic telemetry: useful, late, wasted, or evicted-before-use.", color: C.green },
      ],
      90,
      184,
      1050,
      { size: 26, gap: 76, height: 56 },
    );
    addRule(slide, 540, C.rule);
    addText(slide, "Hardware opportunity: add KV/session context to the memory movement path.", 92, 572, 1030, 38, {
      size: 25,
      bold: true,
      color: C.ink,
    });
    addFooter(slide, 4);
    addNotes(slide, [
      "This is the core problem slide. Keep it concise and hardware-focused.",
      "Do not argue software cannot do this. The claim is that hardware can make enforcement cheaper, more predictable, and scalable.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Replay path exposes the bottleneck", "Each row is one tool-gap replay; the replay path shows where time goes before the first token.");
    await addImage(slide, "readable_phase_timeline_4rows_wide.png", 36, 136, 1208, 438, "Readable phase timeline crop showing G00 through G03", "contain");
    addText(slide, "... more replay gaps observed in the full report", 100, 590, 1080, 28, {
      size: 23,
      bold: true,
      color: C.muted,
      align: "center",
    });
    addSource(slide, "Source: latest_master_report.html controlled run");
    addFooter(slide, 5);
    addNotes(slide, [
      "This is the main timeline slide. It now shows a readable crop of G00 through G03 instead of compressing all rows.",
      "The cropped chart shows initial model turn, tool wait, replay path, H2D, recompute, prefill, and decode.",
      "The full timeline remains available in latest_master_report.html.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Software prefetch often missed replay", "In the live prefetch-margin run, almost every hint completed after replay was due.");
    await addImage(slide, "global_prefetch_margin_backup.png", 40, 150, 850, 390, "Global prefetch margin dot chart from backup report", "contain");
    addNumberLine(slide, "112 / 114", "prefetch attempts were late", 940, 196, C.red);
    addNumberLine(slide, "98.25%", "missed the replay deadline", 940, 340, "#c2410c");
    addText(slide, "Meaning: hints alone are not enough when the movement path cannot act predictably.", 940, 494, 245, 62, {
      size: 20,
      bold: true,
      color: C.ink,
    });
    addSource(slide, "Source: backups/latest_master_report-1.html, Global Prefetch Margin");
    addFooter(slide, 6);
    addNotes(slide, [
      "Global prefetch margin evidence: 114 matched attempts, 112 late, 98.25% late.",
      "This chart is useful for showing deadline misses at a glance.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Mechanism traces explain the miss", "The hint existed, but KV readiness still landed too late for replay.");
    await addImage(slide, "synthetic_profiled_mechanism_timeline_compact.png", 34, 154, 800, 366, "Synthetic profiled mechanism timeline showing hint, HtoD copy, replay due, and replay reload behavior", "contain");
    addBullets(
      slide,
      [
        { text: "0 / 6 ready before replay", color: C.red, bold: true },
        { text: "3 / 6 visible CUDA HtoD", color: C.cyan, bold: true },
        { text: "6 / 6 replay reloaded KV", color: C.gold, bold: true },
      ],
      874,
      214,
      300,
      { size: 24, gap: 74 },
    );
    addText(slide, "Use this as mechanism evidence, not clean TTFT evidence.", 96, 552, 1040, 24, {
      size: 18,
      color: C.muted,
      align: "center",
    });
    addSource(slide, "Source: latest_synthetic_master_report.html, Profiled Mechanism Timelines");
    addFooter(slide, 7);
    addNotes(slide, [
      "This is profiled synthetic mechanism evidence.",
      "It supports the claim that software/runtime scheduling and residency behavior can defeat a correct hint.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Replay-side KV loads missed the deadline", "For this controlled no-prefetch run, visible H2D loads finished after replay was due.");
    await addImage(slide, "global_h2d_readiness.png", 44, 150, 860, 390, "Global replay H2D readiness dot chart", "contain");
    addNumberLine(slide, "8 / 8", "visible replay-side H2D loads finished late", 956, 248, C.red);
    addText(slide, "Meaning: the memory movement happened, but not early enough for the replay deadline.", 956, 390, 238, 74, {
      size: 20,
      bold: true,
      color: C.ink,
    });
    addSource(slide, "Source: latest_master_report.html, Global Replay H2D Readiness");
    addFooter(slide, 8);
    addNotes(slide, [
      "Aggregate no-prefetch evidence: 8 visible replay H2D gaps and 8 late H2D finishes.",
      "This supports the deadline-aware scheduling argument.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "The delay is not just copy time", "The request passes through normal runtime scheduling before visible KV H2D begins.");
    await addImage(slide, "replay_queue_timeline.png", 42, 144, 1120, 432, "Replay queue timeline versus H2D start chart", "contain");
    addText(slide, "Stage markers separate client submit, SGLang receive, scheduler queue/admit, H2D start, and H2D finish.", 78, 596, 1040, 30, {
      size: 19,
      color: C.body,
    });
    addSource(slide, "Source: latest_master_report.html, Replay Queue Timeline vs H2D Start");
    addFooter(slide, 9);
    addNotes(slide, [
      "This chart separates queue timing from H2D timing.",
      "Manager takeaway: a hint-aware system should affect admission, ordering, priority, and residency, not just copy calls.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Key findings make the hardware case", "The pattern points to missing context, deadlines, residency control, and telemetry.");
    addBullets(
      slide,
      [
        { text: "Agent tool gaps can be very short.", color: C.blue },
        { text: "Correct hints can still finish late.", color: C.purple },
        { text: "Visible copy time is only part of end-to-end delay.", color: C.cyan },
        { text: "KV can be written to host, evicted from GPU, then lost from host before replay.", color: C.gold },
        { text: "Replay-side H2D movement can miss the deadline even when the replay request exists.", color: C.red },
      ],
      92,
      168,
      1030,
      { size: 25, gap: 72, height: 54 },
    );
    addRule(slide, 570, C.rule);
    addText(slide, "Conclusion: this is a scheduling and enforceability problem, not just a bandwidth problem.", 92, 594, 1040, 28, {
      size: 22,
      bold: true,
      color: C.ink,
      align: "center",
    });
    addSource(slide, "Source: project traces and latest master/synthetic report observations");
    addFooter(slide, 10);
    addNotes(slide, [
      "Use this slide as the synthesis of the research so far.",
      "The finding list is intentionally short enough to present without reading dense boxes.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Potential hardware impact", "Simple targets for tool-heavy coding-agent workloads.");

    const impacts = [
      {
        color: C.blue,
        title: "Faster agent resume after tools",
        target: "Target: 10-30% lower post-tool latency",
        why: "Why: our traces show the next model turn can wait because useful KV is not ready when the agent resumes.",
      },
      {
        color: C.purple,
        title: "Fewer wasted memory movements",
        target: "Target: 20-50% fewer late or wasted KV reloads",
        why: "Why: memory movement can finish too late, or useful KV can be moved but not reused in time.",
      },
      {
        color: C.green,
        title: "More predictable tail latency",
        target: "Target: lower p95/p99 agent-resume stalls",
        why: "Why: urgent agent KV currently competes with ordinary memory traffic without deadline or priority context.",
      },
    ];

    impacts.forEach((impact, idx) => {
      const y = 164 + idx * 128;
      addText(slide, `${idx + 1}.`, 72, y, 56, 36, { size: 25, bold: true, color: impact.color });
      addText(slide, impact.title, 118, y, 560, 34, { size: 25, bold: true, color: C.ink });
      addText(slide, impact.target, 118, y + 38, 690, 30, { size: 21, bold: true, color: impact.color });
      addText(slide, impact.why, 118, y + 72, 980, 45, { size: 18, color: C.body });
      if (idx < impacts.length - 1) addRule(slide, y + 114, "#e2e8f0");
    });

    addRule(slide, 560, C.rule);
    addText(slide, "Hardware support to test", 92, 584, 310, 28, { size: 20, bold: true, color: C.ink });
    addText(slide, "KV/session metadata, deadline-aware movement queues, temporary KV protection, and useful/late/wasted telemetry.", 92, 616, 1030, 36, {
      size: 19,
      bold: true,
      color: C.ink,
    });
    addFooter(slide, 11);
    addNotes(slide, [
      "This slide intentionally avoids internal terms such as H2D and replay-prefill on the visible canvas.",
      "Translate for technical discussion: faster agent resume corresponds to lowering post-tool replay latency when KV is ready earlier.",
      "Fewer wasted memory movements corresponds to reducing late prefetches, evicted-before-use KV, and redundant reloads.",
      "More predictable tail latency corresponds to reducing p95/p99 stalls caused by context-agnostic memory scheduling.",
      "Potential benefit ranges are conservative internal proposal targets based on observed late prefetch/replay H2D patterns and expected gains from deadline-aware KV movement.",
    ]);
  }

  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Hardware support can make hints enforceable", "Treat KV as deadline-sensitive memory, not generic bytes.");
    addBullets(
      slide,
      [
        { text: "KV page/session metadata: which agent owns this KV and how urgent is it?", color: C.blue },
        { text: "Deadline-aware migration queue: move urgent KV before less urgent traffic.", color: C.purple },
        { text: "Residency protection: keep useful prefetched KV from being evicted too early.", color: C.green },
        { text: "Telemetry: count useful, late, wasted, and evicted-before-use movement.", color: C.cyan },
      ],
      90,
      178,
      1050,
      { size: 25, gap: 72, height: 58 },
    );
    addRule(slide, 552, C.rule);
    addText(slide, "Research question: how much replay latency and wasted movement can be avoided when KV movement has session context, priority, deadline, protection, and telemetry?", 90, 580, 1070, 50, {
      size: 21,
      bold: true,
      color: C.ink,
      align: "center",
    });
    addFooter(slide, 12);
    addNotes(slide, [
      "Closing slide. Translate the evidence into concrete hardware support candidates.",
      "These ideas map to the hardware proposal: KV metadata, semantic prefetch queue, priority-aware migration, residency protection, and telemetry.",
    ]);
  }

  for (const [i, slide] of deck.slides.items.entries()) {
    const png = await deck.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(buildDir, "rendered", `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(buildDir, "rendered", `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
  }
  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(buildDir, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(finalPptx);
  console.log(finalPptx);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
