import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware";
const SKILL_DIR = "/Users/oluwolejaiyeoba/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-build/scenario2_slide");
const FINAL_PPTX = path.join(workspaceDir, "presentation/scenario_2_proactive_kv_preparation.pptx");
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

function addSection(slide, title, lines, left, top, width, accent = C.accent) {
  addText(slide, title, left, top, width, 28, { size: 19, bold: true, color: C.ink });
  let y = top + 40;
  for (const line of lines) {
    addRect(slide, left, y + 9, 8, 8, { fill: accent, line: accent, lineWidth: 0 });
    addText(slide, line, left + 20, y, width - 20, 34, { size: 17, color: C.body });
    y += 38;
  }
}

function addMetric(slide, label, value, note, left, top, width, fill = C.panel, valueColor = C.ink) {
  addRect(slide, left, top, width, 88, { fill, line: "#dbe4ef", lineWidth: 1 });
  addText(slide, label, left + 16, top + 12, width - 32, 22, { size: 14, bold: true, color: C.muted });
  addText(slide, value, left + 16, top + 35, width - 32, 28, { size: 23, bold: true, color: valueColor });
  addText(slide, note, left + 16, top + 64, width - 32, 18, { size: 12.5, color: C.body });
}

async function main() {
  await fs.mkdir(TMP_DIR, { recursive: true });
  await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";

  addText(slide, "Scenario 2: Proactive KV Preparation", 56, 36, 820, 48, {
    size: 38,
    bold: true,
    color: C.ink,
  });
  addText(
    slide,
    "During tool wait, the backend prepares useful host-backed KV before replay arrives.",
    58,
    88,
    940,
    28,
    { size: 19, color: C.muted },
  );
  addText(slide, "Modest win", 1080, 50, 120, 26, { size: 18, bold: true, color: C.success, align: "right" });
  addRule(slide, 56, 126, 1168);

  addSection(
    slide,
    "What the setup tests",
    [
      "32 replay requests run in both modes.",
      "Both modes use the same long tool-wait timeline.",
      "Baseline waits until replay arrives.",
      "Controller mode prepares only useful KV.",
    ],
    66,
    160,
    455,
  );

  addRect(slide, 66, 374, 455, 154, { fill: C.panel, line: "#dbe4ef", lineWidth: 1 });
  addText(slide, "What the controller does", 88, 394, 390, 26, { size: 19, bold: true, color: C.ink });
  addText(
    slide,
    "It checks SGLang's live cache state before moving anything. Already-on-GPU prefixes are skipped. Small or unsafe loads are skipped. Useful host-backed KV is prepared through SGLang's own path.",
    88,
    428,
    390,
    84,
    { size: 16.5, color: C.body },
  );

  addText(slide, "Results across all replay requests", 590, 160, 560, 28, {
    size: 21,
    bold: true,
    color: C.ink,
  });
  addText(slide, "Latest all-request run: proactive_kv_management_selective_admission_20260917_062007", 590, 192, 600, 24, {
    size: 13,
    color: C.muted,
  });

  addMetric(slide, "Total TTFT", "5.3% better", "123.1 s to 116.5 s", 590, 228, 270);
  addMetric(slide, "Avg TTFT per request", "0.2 s faster", "3.8 s to 3.6 s", 882, 228, 270);
  addMetric(slide, "Total lateness", "5.3% better", "123.4 s to 116.9 s", 590, 334, 270);
  addMetric(slide, "Max lateness", "2.2 s better", "10.5 s to 8.3 s", 882, 334, 270);
  addMetric(slide, "P95 lateness", "0.1 s worse", "Roughly flat: 7.3 s to 7.4 s", 590, 440, 270, C.panelAlt, C.body);
  addMetric(slide, "Workload window", "about same", "328.6 s to 328.5 s", 882, 440, 270, C.panelAlt, C.body);

  addRule(slide, 56, 580, 1168);
  addText(slide, "Mechanism proof", 66, 604, 190, 28, { size: 20, bold: true, color: C.ink });
  addText(
    slide,
    "21 cache checks, 6 admitted direct loads, 48,889 prepared tokens, and 560 H2D copy events before replay compute.",
    256,
    604,
    860,
    42,
    { size: 20, bold: true, color: C.ink },
  );
  addText(slide, "Source: Harness-Aware Scenario Tracker", 58, 674, 440, 20, {
    size: 12,
    color: C.muted,
  });

  slide.speakerNotes.textFrame.setText(`Scenario 2 source: harness_aware_scenario_tracker.html.

Setup: same 32 replay requests in both modes using the same generated timeline. Each replay session has its own session_id and prefix_id. Baseline waits until replay arrives. Controller mode uses ETA, reuse probability, and recompute cost during tool wait, then requests SGLang-side KV preparation only when the window is safe enough for the load to finish before replay.

Run: proactive_kv_management_selective_admission_20260917_062007 compared no_prefetch against controller_proactive_kv_management.

Results: Total TTFT improved from 123.1 s to 116.5 s, or 5.3%. Average replay TTFT improved from 3.8 s to 3.6 s. Total lateness improved from 123.4 s to 116.9 s, or 5.3%. Average replay lateness improved from 3.9 s to 3.7 s. P95 lateness was roughly flat, moving from 7.3 s to 7.4 s. Max lateness improved from 10.5 s to 8.3 s. Full workload window was essentially unchanged, moving from 328.6 s to 328.5 s.

Mechanism evidence: 21 prefetch requests, 21 backend actions, direct hook available for 21 of 21, 9 prepare plans would load and 12 were already resident. Selective admission skipped 12 already-resident prefixes and 3 host-resident prefixes that were too small to justify loading. It attempted direct load on 6 prefixes. The proof table observed 111 load-back events plus 560 H2D copy events before replay compute, with 48,889 prepared tokens via load_back.

Important caveat: the aggregate timing win is modest. Only 11 of 32 paired requests improved individually, so the next tuning direction is better replay-reuse prediction.

[Sources]
Local file: /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/harness_aware_scenario_tracker.html`);
  slide.speakerNotes.setVisible(true);

  const candidatePath = path.join(TMP_DIR, `scenario_2_candidate_${buildStamp}.pptx`);
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
    receiptPath: path.join(TMP_DIR, `scenario_2_validation_${buildStamp}.json`),
  });

  console.log(`Finalized ${result.finalPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
