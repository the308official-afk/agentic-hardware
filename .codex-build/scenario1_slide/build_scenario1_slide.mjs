import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware";
const SKILL_DIR = "/Users/oluwolejaiyeoba/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-build/scenario1_slide");
const FINAL_PPTX = path.join(workspaceDir, "presentation/scenario_1_better_deadline_scheduling.pptx");
const RUNTIME_PYTHON = "/Users/oluwolejaiyeoba/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const buildStamp = new Date().toISOString().replaceAll(":", "").replaceAll(".", "");

const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

const W = 1280;
const H = 720;
const C = {
  ink: "#111827",
  body: "#334155",
  muted: "#64748b",
  line: "#d7dee8",
  panel: "#f8fafc",
  panelAlt: "#f3f6fa",
  accent: "#1f4e79",
  success: "#2f6f56",
  subtleWarn: "#6f5a2a",
};

const family = resolvePresentationFont();

function addText(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: opts.size ?? 18,
    bold: opts.bold ?? false,
    color: opts.color ?? C.body,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    autoFit: "none",
  };
  return shape;
}

function addRect(slide, left, top, width, height, opts = {}) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: opts.fill ?? "#ffffff",
    line: { style: "solid", fill: opts.line ?? C.line, width: opts.lineWidth ?? 1 },
  });
}

function addRule(slide, left, top, width, color = C.line) {
  addRect(slide, left, top, width, 1.4, { fill: color, line: "none", lineWidth: 0 });
}

function addSection(slide, title, lines, left, top, width, accent) {
  addText(slide, title, left, top, width, 28, { size: 19, bold: true, color: C.ink });
  let y = top + 40;
  for (const line of lines) {
    addText(slide, "", left, y + 8, 8, 8, { size: 8, color: accent, bold: true });
    addRect(slide, left, y + 9, 8, 8, { fill: accent, line: accent, lineWidth: 0 });
    addText(slide, line, left + 20, y, width - 20, 34, { size: 17, color: C.body });
    y += 38;
  }
}

function addMetric(slide, label, value, note, left, top, width, fill, color) {
  addRect(slide, left, top, width, 92, { fill, line: "#dbe4ef", lineWidth: 1 });
  addText(slide, label, left + 16, top + 14, width - 32, 22, { size: 15, bold: true, color: C.muted });
  addText(slide, value, left + 16, top + 38, width - 32, 28, { size: 24, bold: true, color });
  addText(slide, note, left + 16, top + 68, width - 32, 18, { size: 13, color: C.body });
}

async function main() {
  await fs.mkdir(TMP_DIR, { recursive: true });
  await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";

  addText(slide, "Scenario 1: Deadline-Aware Replay Ordering", 56, 36, 840, 48, {
    size: 38,
    bold: true,
    color: C.ink,
  });
  addText(
    slide,
    "Harness timing estimates when agents resume, then releases replay work in deadline order.",
    58,
    88,
    900,
    28,
    { size: 19, color: C.muted },
  );
  addText(slide, "Clear win", 1080, 50, 120, 26, { size: 18, bold: true, color: C.success, align: "right" });
  addRule(slide, 56, 126, 1168);

  addSection(
    slide,
    "What the setup tests",
    [
      "32 replay requests run in both modes.",
      "Both modes use the same tool-wait timeline.",
      "Tool waits range from very short to long.",
      "Baseline submits replay work in arrival order.",
    ],
    66,
    160,
    455,
    C.accent,
  );

  addRect(slide, 66, 374, 455, 154, { fill: "#f8fafc", line: "#dbe4ef", lineWidth: 1 });
  addText(slide, "What the controller does", 88, 394, 390, 26, { size: 19, bold: true, color: C.ink });
  addText(
    slide,
    "It estimates when each agent will resume, then releases ready replay requests in deadline order. The request payload stays the same; only the release sequence changes.",
    88,
    428,
    390,
    76,
    { size: 17, color: C.body },
  );

  addText(slide, "Results across all replay requests", 590, 160, 560, 28, {
    size: 21,
    bold: true,
    color: C.ink,
  });
  addText(slide, "Latest clear-win run: deadline_predictive_queue_submitprio_20260916_221604", 590, 192, 560, 24, {
    size: 13,
    color: C.muted,
  });

  addMetric(slide, "Total TTFT", "15.1% better", "101.5 s to 86.2 s", 590, 228, 270, C.panel, C.ink);
  addMetric(slide, "Avg TTFT per request", "0.5 s faster", "3.2 s to 2.7 s", 882, 228, 270, C.panel, C.ink);
  addMetric(slide, "Total lateness", "22.1% better", "278.7 s to 217.1 s", 590, 340, 270, C.panel, C.ink);
  addMetric(slide, "P95 lateness", "37.6% better", "31.3 s to 19.5 s", 882, 340, 270, C.panel, C.ink);
  addMetric(slide, "Trade-off", "23.0 s longer", "Workload window: 234.7 s to 257.7 s", 590, 452, 562, C.panelAlt, C.body);

  addRule(slide, 56, 580, 1168);
  addText(slide, "Takeaway", 66, 608, 140, 28, { size: 20, bold: true, color: C.ink });
  addText(
    slide,
    "Using harness timing to order replay work reduced replay delay across the full workload.",
    206,
    608,
    900,
    38,
    { size: 21, bold: true, color: C.ink },
  );
  addText(slide, "Source: Harness-Aware Scenario Tracker", 58, 674, 440, 20, {
    size: 12,
    color: C.muted,
  });

  slide.speakerNotes.textFrame.setText(`Scenario 1 source: harness_aware_scenario_tracker.html.

Setup: same 32 measured replay requests in baseline and controller modes, using the same generated timeline. Baseline submits replay work in arrival order. Controller mode estimates expected replay timing during tool wait, then releases ready replay requests in deadline order before they enter SGLang. The request payload stays the same; only the release sequence changes.

Signals: session_id, phase=tool_wait, expected_tool_return_ms, next_ready_eta_ms, deadline_after_ready_ms, tool_wait_class, task_replay_steps.

Results: Total TTFT improved from 101.5 s to 86.2 s, or 15.1%. Average TTFT improved from 3.2 s to 2.7 s per replay request. Total lateness improved from 278.7 s to 217.1 s, or 22.1%. Average replay lateness improved from 8.7 s to 6.8 s. P95 lateness improved from 31.3 s to 19.5 s, or 37.6%. Max lateness improved from 31.6 s to 19.6 s, or 38.0%. Trade-off: full workload window increased from 234.7 s to 257.7 s, or 23.0 s longer.

[Sources]
Local file: /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/harness_aware_scenario_tracker.html`);
  slide.speakerNotes.setVisible(true);

  const candidatePath = path.join(TMP_DIR, `scenario_1_candidate_${buildStamp}.pptx`);
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

  const result = await finalizePresentation({
    explicitTotalSlideCount: 1,
    requiredNativeTableOwnerSlides: [],
    requiredNativeChartOwnerSlides: [],
    workspaceDir,
    candidatePath,
    finalPath: FINAL_PPTX,
    pythonExecutable: RUNTIME_PYTHON,
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: [
      "--expected-slide-size-emu",
      "12192000,6858000",
      "--validate-bullet-geometry",
      "--validate-heading-fit",
    ],
    fontPolicy: { basis: "design", families: [family] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(TMP_DIR, `scenario_1_validation_${buildStamp}.json`),
  });

  console.log(`Finalized ${result.finalPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
