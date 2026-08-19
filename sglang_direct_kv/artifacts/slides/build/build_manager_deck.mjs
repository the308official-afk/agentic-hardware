import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const repoRoot = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware";
const outDir = path.join(repoRoot, "sglang_direct_kv/artifacts/slides");
const buildDir = path.join(outDir, "build");
const imgDir = path.join(outDir, "images");
const finalPptx = path.join(outDir, "agent_aware_kv_movement_manager_deck.pptx");
const reportPath = path.join(repoRoot, "sglang_direct_kv/artifacts/results/latest_master_report.html");
const backupReportPath = path.join(repoRoot, "backups/latest_master_report-1.html");

const W = 1280;
const H = 720;
const PROJECT_GOAL =
  "Coding agents naturally pause during tool calls, then resume with tight latency expectations. Current memory movement paths are largely context-agnostic: they can move KV pages, but they do not know which agent needs them, when they are needed, or whether missing the deadline will stall replay. This project aims to quantify that gap and prototype hint-guided KV movement as a path toward smarter DMA engines, KV-aware memory scheduling, and hardware/runtime co-design.";
const C = {
  ink: "#0f172a",
  body: "#334155",
  muted: "#64748b",
  light: "#f8fafc",
  panel: "#e2e8f0",
  rule: "#cbd5e1",
  blue: "#2563eb",
  purple: "#a855f7",
  cyan: "#06b6d4",
  green: "#16a34a",
  red: "#ef4444",
  magenta: "#db2777",
  gold: "#eab308",
};

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

function addRect(slide, left, top, width, height, fill = C.light, line = C.rule, radius = "rounded-lg") {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function addFooter(slide, number) {
  addText(slide, "Agent-aware KV movement", 42, 670, 360, 24, { size: 13, color: C.muted });
  addText(slide, String(number).padStart(2, "0"), 1190, 670, 48, 24, { size: 13, color: C.muted, align: "right" });
}

function addTitle(slide, title, subtitle = "") {
  addText(slide, title, 42, 34, 1120, 84, { size: 38, bold: true, color: C.ink });
  if (subtitle) {
    addText(slide, subtitle, 44, 108, 1020, 44, { size: 18, color: C.muted });
  }
}

function addSource(slide, source = "Source: latest_master_report.html controlled run") {
  addText(slide, source, 42, 646, 820, 20, { size: 11, color: C.muted });
}

function addNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText([
    ...lines,
    "",
    "[Sources]",
    `Local report: ${reportPath}`,
    `Backup report: ${backupReportPath}`,
  ]);
  slide.speakerNotes.setVisible(true);
}

async function addImage(slide, name, left, top, width, height, alt, fit = "contain") {
  addRect(slide, left - 8, top - 8, width + 16, height + 16, "#ffffff", "#e5e7eb", "rounded-md");
  slide.images.add({
    blob: await imageBytes(name),
    contentType: "image/png",
    alt,
    fit,
    position: { left, top, width, height },
  });
}

function addBullets(slide, items, left, top, width, gap = 46, size = 22) {
  items.forEach((item, idx) => {
    const y = top + idx * gap;
    slide.shapes.add({
      geometry: "ellipse",
      position: { left, top: y + 8, width: 9, height: 9 },
      fill: item.color ?? C.blue,
      line: { style: "solid", fill: "none", width: 0 },
    });
    addText(slide, item.text, left + 24, y, width - 24, 42, { size, color: item.textColor ?? C.body, bold: item.bold ?? false });
  });
}

function addFlowNode(slide, text, left, top, width, height, fill, line = C.rule) {
  addRect(slide, left, top, width, height, fill, line, "rounded-md");
  addText(slide, text, left + 14, top + 15, width - 28, height - 20, {
    size: 20,
    bold: true,
    color: C.ink,
    align: "center",
    valign: "middle",
  });
}

async function main() {
  await fs.mkdir(outDir, { recursive: true });
  await fs.mkdir(path.join(buildDir, "rendered"), { recursive: true });
  await fs.writeFile(
    path.join(buildDir, "source-notes.txt"),
    `Deck uses local report evidence from ${reportPath}\nChart images extracted from latest_master_report.html.\n`,
    "utf8",
  );

  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  // Slide 1
  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addText(slide, "Making KV Movement Agent-Aware", 42, 54, 880, 128, { size: 58, bold: true, color: C.ink });
    addText(
      slide,
      PROJECT_GOAL,
      46,
      180,
      1080,
      128,
      { size: 21, color: C.body },
    );
    const y = 402;
    const nodes = [
      ["model turn", 52, "#dbeafe"],
      ["tool wait", 285, "#f1f5f9"],
      ["replay request", 518, "#fee2e2"],
      ["KV ready?", 751, "#dcfce7"],
    ];
    for (let i = 0; i < nodes.length; i++) {
      addFlowNode(slide, nodes[i][0], nodes[i][1], y, 170, 82, nodes[i][2]);
      if (i < nodes.length - 1) addText(slide, "→", nodes[i][1] + 182, y + 21, 40, 40, { size: 32, bold: true, color: C.muted });
    }
    addFooter(slide, 1);
    addNotes(slide, [
      "Opening slide. Position this as a hardware/runtime co-design question, not just an SGLang software optimization.",
      `Goal: ${PROJECT_GOAL}`,
    ]);
  }

  // Slide 2
  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "The testbed uses both controlled and real agent traffic", "Synthetic request generators give clean knobs; SWE-bench/DeepAgents traces give realistic coding-agent behavior.");
    await addImage(slide, "simple_experiment_setup_flow.png", 52, 158, 1176, 172, "Experiment setup flow chart from latest master report", "contain");
    addRect(slide, 86, 392, 478, 150, "#f8fafc", C.rule, "rounded-md");
    addText(slide, "Controlled synthetic path", 116, 416, 390, 30, { size: 24, bold: true, color: C.ink });
    addText(slide, "Synthetic request generator creates known prompt sizes, tool-wait windows, filler pressure, and replay deadlines.", 116, 458, 388, 58, { size: 18, color: C.body });
    addRect(slide, 640, 392, 478, 150, "#eff6ff", "#bfdbfe", "rounded-md");
    addText(slide, "Real coding-agent path", 670, 416, 390, 30, { size: 24, bold: true, color: C.ink });
    addText(slide, "SWE-bench / AgentBench tasks feed DeepAgents, which emits model turns and tool-call gaps into SGLang.", 670, 458, 388, 58, { size: 18, color: C.body });
    addSource(slide, "Source: latest_master_report.html, Experiment Setup And Manager Summary");
    addFooter(slide, 2);
    addNotes(slide, [
      "This slide explains the two traffic feeds. Synthetic runs let us control pressure and timing; real DeepAgents/SWE-bench traces preserve realistic coding-agent prompts and tool gaps.",
      "The setup flow image comes from latest_master_report.html.",
    ]);
  }

  // Slide 3
  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Today’s DMA engines move bytes, not intent", "The hardware path is fast, but it usually lacks agent/session semantics.");
    addRect(slide, 58, 188, 520, 360, "#f8fafc", C.rule, "rounded-md");
    addText(slide, "Context-agnostic movement", 88, 218, 440, 34, { size: 26, bold: true, color: C.ink });
    addBullets(
      slide,
      [
        { text: "sees memory ranges", color: C.muted },
        { text: "does not know the agent session", color: C.muted },
        { text: "does not know replay deadline", color: C.muted },
        { text: "does not protect useful prefetched KV", color: C.muted },
      ],
      92,
      284,
      420,
      54,
      20,
    );
    addText(slide, "→", 604, 324, 70, 70, { size: 54, bold: true, color: C.muted, align: "center" });
    addRect(slide, 702, 188, 520, 360, "#eff6ff", "#bfdbfe", "rounded-md");
    addText(slide, "Hint-aware movement", 732, 218, 420, 34, { size: 26, bold: true, color: C.ink });
    addBullets(
      slide,
      [
        { text: "tags KV by session and priority", color: C.blue },
        { text: "schedules against replay deadlines", color: C.purple },
        { text: "moves hot KV before replay", color: C.cyan },
        { text: "reports late, useful, and wasted movement", color: C.green },
      ],
      736,
      284,
      420,
      54,
      20,
    );
    addFooter(slide, 3);
    addNotes(slide, [
      "This slide frames the hardware opportunity: hints must become enforceable by the memory movement path.",
      "Hardware examples include SDMA, CPU-side DMA, NIC DMA, SSD DMA, and copy/migration engines.",
    ]);
  }

  // Slide 4
  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "The replay path exposes the bottleneck", "Several tool-gap sessions show where replay spends time before the first useful token.");
    await addImage(slide, "readable_phase_timeline_8rows_wide.png", 32, 132, 1216, 506, "Readable phase timeline crop showing several gap sessions", "contain");
    addSource(slide);
    addFooter(slide, 4);
    addNotes(slide, [
      "Use this as the narrative chart. It shows the initial model turn, tool wait, replay path, replay-side H2D, recompute, prefill, and decode.",
      "The image is a cropped view of the Readable Phase Timeline in latest_master_report.html.",
    ]);
  }

  // Slide 5
  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Software prefetch often missed the agent replay deadline", "Across live prefetch attempts, nearly every hint completed after the agent had already resumed.");
    await addImage(slide, "global_prefetch_margin_backup.png", 42, 158, 884, 392, "Global prefetch margin dot chart from backup report", "contain");
    addRect(slide, 958, 168, 234, 136, "#fef2f2", "#fecaca", "rounded-md");
    addText(slide, "112 of 114", 976, 190, 198, 46, { size: 36, bold: true, color: "#b91c1c", align: "center" });
    addText(slide, "prefetch attempts", 982, 244, 186, 24, { size: 18, color: C.body, align: "center" });
    addText(slide, "were late", 982, 270, 186, 24, { size: 18, color: C.body, align: "center" });
    addRect(slide, 958, 330, 234, 112, "#fff7ed", "#fed7aa", "rounded-md");
    addText(slide, "98.25%", 982, 350, 186, 44, { size: 40, bold: true, color: "#c2410c", align: "center" });
    addText(slide, "missed replay due deadline", 982, 402, 186, 30, { size: 17, color: C.body, align: "center" });
    addText(slide, "This shows that semantic knowledge alone is not enough if the movement path cannot act on hints predictably.", 958, 474, 236, 72, { size: 20, color: C.body });
    addSource(slide, "Source: backups/latest_master_report-1.html, Global Prefetch Margin");
    addFooter(slide, 5);
    addNotes(slide, [
      "This slide is the global prefetch-attempt evidence. The backup report shows 114 matched prefetch attempts, 112 late, and 98.25% late.",
      "Use it to show that software prefetch through the normal serving path often misses tight replay windows.",
    ]);
  }

  // Slide 6
  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Replay-side KV loads missed the deadline in this run", "The aggregate view shows whether KV H2D finished before or after replay was due.");
    await addImage(slide, "global_h2d_readiness.png", 42, 158, 900, 390, "Global replay H2D readiness dot chart", "contain");
    addRect(slide, 972, 194, 220, 116, "#fef2f2", "#fecaca", "rounded-md");
    addText(slide, "8 / 8", 998, 218, 168, 50, { size: 48, bold: true, color: "#b91c1c", align: "center" });
    addText(slide, "H2D loads finished late", 998, 276, 168, 28, { size: 17, color: C.body, align: "center" });
    addText(slide, "All visible replay-side KV loads finished below the 0 ms deadline line.", 972, 370, 220, 78, { size: 21, color: C.body });
    addSource(slide, "Source: latest_master_report.html, Global Replay H2D Readiness");
    addFooter(slide, 6);
    addNotes(slide, [
      "This is the aggregate evidence slide. The current controlled no-prefetch run reports 8 no-prefetch replay H2D gaps and 8 late H2D finishes.",
      "Measured claim source: Global Replay H2D Readiness table and dot plot in latest_master_report.html.",
    ]);
  }

  // Slide 7
  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "The delay is not just copy time", "The request enters normal software/runtime scheduling before visible KV H2D begins.");
    await addImage(slide, "replay_queue_timeline.png", 42, 148, 1120, 440, "Replay queue timeline versus H2D start chart", "contain");
    addText(slide, "The stage markers separate submission, SGLang receive, scheduler queue/admit, and visible KV H2D movement.", 72, 608, 1040, 24, { size: 17, color: C.body });
    addSource(slide, "Source: latest_master_report.html, Replay Queue Timeline vs H2D Start");
    addFooter(slide, 7);
    addNotes(slide, [
      "This chart separates replay due, client submit, SGLang receive, scheduler queue/admit, H2D start, and H2D finish.",
      "The key manager takeaway: a hint-aware design should influence scheduling and priority, not merely request more copies.",
    ]);
  }

  // Slide 8
  {
    const slide = deck.slides.add();
    slide.background.fill = "#ffffff";
    addTitle(slide, "Hardware support can make hints enforceable", "The opportunity is to treat KV as deadline-sensitive memory, not generic bytes.");
    const top = 206;
    const nodes = [
      ["Runtime hint", "session, priority, deadline", 58, "#eff6ff", C.blue],
      ["KV-aware queue", "order urgent KV first", 300, "#f5f3ff", C.purple],
      ["DMA / copy engine", "throttle and prioritize", 542, "#ecfeff", C.cyan],
      ["Residency control", "protect useful KV", 784, "#f0fdf4", C.green],
      ["Telemetry", "late / useful / wasted", 1026, "#fff7ed", "#f97316"],
    ];
    nodes.forEach(([title, body, x, fill, line], idx) => {
      addRect(slide, x, top, 178, 178, fill, line, "rounded-md");
      addText(slide, title, x + 16, top + 28, 146, 34, { size: 22, bold: true, color: C.ink, align: "center" });
      addText(slide, body, x + 18, top + 92, 142, 56, { size: 17, color: C.body, align: "center" });
      if (idx < nodes.length - 1) addText(slide, "→", x + 184, top + 62, 50, 50, { size: 34, bold: true, color: C.muted, align: "center" });
    });
    addRect(slide, 210, 474, 860, 84, "#f8fafc", C.rule, "rounded-md");
    addText(slide, "Resulting research question", 236, 498, 260, 26, { size: 22, bold: true, color: C.ink });
    addText(slide, "How much replay latency and wasted movement can be avoided when KV movement has session context, priority, deadline, protection, and telemetry?", 520, 492, 500, 44, { size: 19, color: C.body });
    addFooter(slide, 8);
    addNotes(slide, [
      "Closing slide. This translates observed software/runtime behavior into concrete hardware support candidates.",
      "Hardware ideas come from the project proposal: KV metadata, deadline-aware queues, priority-aware migration, residency protection, and telemetry.",
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
