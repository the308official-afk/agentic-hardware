import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware";
const SKILL_DIR = "/Users/oluwolejaiyeoba/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-build/scenario_master_deck");
const FINAL_PPTX = path.join(workspaceDir, "presentation/harness_aware_scenario_results_master.pptx");
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
  winFill: "#eef7f1",
  tradeFill: "#fff1f1",
  neutralFill: "#f3f6fa",
  accent: "#1f4e79",
  success: "#2f6f56",
  tradeText: "#7f1d1d",
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

function addMetric(slide, label, value, note, left, top, width, fill = C.panel, valueColor = C.ink, height = 92) {
  addRect(slide, left, top, width, height, { fill, line: "#dbe4ef", lineWidth: 1 });
  addText(slide, label, left + 16, top + 12, width - 32, 22, { size: 14, bold: true, color: C.muted });
  addText(slide, value, left + 16, top + 36, width - 32, 28, { size: 23, bold: true, color: valueColor });
  addText(slide, note, left + 16, top + 66, width - 32, 18, { size: 12.5, color: C.body });
}

function addCommonSource(slide) {
  return slide;
}

function addHarnessSignals(slide, signalsText) {
  addRule(slide, 56, 580, 1168);
  addText(slide, "Harness signals", 66, 608, 190, 28, { size: 20, bold: true, color: C.ink });
  addText(slide, signalsText, 256, 606, 900, 46, {
    size: 18,
    bold: true,
    color: C.ink,
  });
}

function addScenario1(presentation) {
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
  );

  addRect(slide, 66, 374, 455, 154, { fill: C.panel, line: "#dbe4ef", lineWidth: 1 });
  addText(slide, "What the controller does", 88, 394, 390, 26, { size: 19, bold: true, color: C.ink });
  addText(
    slide,
    "It uses expected tool-return times to decide which replay should enter next, so a soon-returning agent is not stuck behind later work.",
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
  addText(slide, "Latest clear-win run: scenario1_replay_friction_20260917_192617", 590, 192, 560, 24, {
    size: 13,
    color: C.muted,
  });

  addMetric(slide, "Total TTFT", "33.7% better", "112.2 s to 74.4 s", 590, 228, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Avg TTFT per request", "1.2 s faster", "3.5 s to 2.3 s", 882, 228, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Total lateness", "25.7% better", "287.3 s to 213.6 s", 590, 334, 270, C.winFill, C.ink, 88);
  addMetric(slide, "P95 lateness", "39.4% better", "30.0 s to 18.2 s", 882, 334, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Max lateness", "38.0% better", "31.6 s to 19.7 s", 590, 440, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Workload window", "5.0 s longer", "37.1 s to 42.0 s", 882, 440, 270, C.tradeFill, C.tradeText, 88);

  addHarnessSignals(
    slide,
    "Session ID, tool-wait phase, expected return time, replay deadline, wait class, replay steps.",
  );
  addCommonSource(slide);

  slide.speakerNotes.textFrame.setText(`Scenario 1 source: harness_aware_scenario_tracker.html.

Setup: same 32 measured replay requests in baseline and controller modes, using the same generated timeline. Baseline submits replay work in arrival order. Controller mode estimates expected replay timing during tool wait, then releases ready replay requests in deadline order before they enter SGLang. The request payload stays the same; only the release sequence changes.

Signals: session_id, phase=tool_wait, expected_tool_return_ms, next_ready_eta_ms, deadline_after_ready_ms, tool_wait_class, task_replay_steps.

Results: Total TTFT improved from 112.2 s to 74.4 s, or 33.7%. Average TTFT improved from 3.5 s to 2.3 s per replay request. Total lateness improved from 287.3 s to 213.6 s, or 25.7%. Average replay lateness improved from 9.0 s to 6.7 s. P95 lateness improved from 30.0 s to 18.2 s, or 39.4%. Max lateness improved from 31.6 s to 19.7 s, or 38.0%. Trade-off: full workload window increased from 37.1 s to 42.0 s, or 5.0 s longer.

Mechanism evidence: early priorities assigned 32 / 32 replay requests, SGLang priority rows 32 / 32, local submission priority 32 / 32. Tool-wait buckets were 7 very short, 8 short, 14 medium, and 3 long. Replay friction deep dive was trace-rich for 64 / 64 rows. Controller average scheduler friction was 2.6 versus baseline 2.7, and controller average memory friction was 2.7 versus baseline 2.8. The main remaining issue was mixed scheduler and memory friction.

[Sources]
Local file: /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/harness_aware_scenario_tracker.html`);
  slide.speakerNotes.setVisible(true);
}

function addScenario2(presentation) {
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
    "It prepares likely-needed KV during the tool wait, but only when SGLang says the KV is worth loading.",
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

  addMetric(slide, "Total TTFT", "5.3% better", "123.1 s to 116.5 s", 590, 228, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Avg TTFT per request", "0.2 s faster", "3.8 s to 3.6 s", 882, 228, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Total lateness", "5.3% better", "123.4 s to 116.9 s", 590, 334, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Max lateness", "2.2 s better", "10.5 s to 8.3 s", 882, 334, 270, C.winFill, C.ink, 88);
  addMetric(slide, "P95 lateness", "0.1 s worse", "Roughly flat: 7.3 s to 7.4 s", 590, 440, 270, C.tradeFill, C.tradeText, 88);
  addMetric(slide, "Workload window", "about same", "328.6 s to 328.5 s", 882, 440, 270, C.neutralFill, C.body, 88);

  addHarnessSignals(
    slide,
    "Session ID, prefix ID, expected return time, ETA uncertainty, replay deadline, reuse probability, recompute cost.",
  );
  addCommonSource(slide);

  slide.speakerNotes.textFrame.setText(`Scenario 2 source: harness_aware_scenario_tracker.html.

Setup: same 32 replay requests in both modes using the same generated timeline. Each replay session has its own session_id and prefix_id. Baseline waits until replay arrives. Controller mode uses ETA, reuse probability, and recompute cost during tool wait, then requests SGLang-side KV preparation only when the window is safe enough for the load to finish before replay.

Run: proactive_kv_management_selective_admission_20260917_062007 compared no_prefetch against controller_proactive_kv_management.

Results: Total TTFT improved from 123.1 s to 116.5 s, or 5.3%. Average replay TTFT improved from 3.8 s to 3.6 s. Total lateness improved from 123.4 s to 116.9 s, or 5.3%. Average replay lateness improved from 3.9 s to 3.7 s. P95 lateness was roughly flat, moving from 7.3 s to 7.4 s. Max lateness improved from 10.5 s to 8.3 s. Full workload window was essentially unchanged, moving from 328.6 s to 328.5 s.

Mechanism evidence: 21 prefetch requests, 21 backend actions, direct hook available for 21 of 21, 9 prepare plans would load and 12 were already resident. Selective admission skipped 12 already-resident prefixes and 3 host-resident prefixes that were too small to justify loading. It attempted direct load on 6 prefixes. The proof table observed 111 load-back events plus 560 H2D copy events before replay compute, with 48,889 prepared tokens via load_back.

Important caveat: the aggregate timing win is modest. Only 11 of 32 paired requests improved individually, so the next tuning direction is better replay-reuse prediction.

[Sources]
Local file: /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/harness_aware_scenario_tracker.html`);
  slide.speakerNotes.setVisible(true);
}

function addScenario3(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";

  addText(slide, "Scenario 3: Value-Aware KV Eviction", 56, 36, 820, 48, {
    size: 38,
    bold: true,
    color: C.ink,
  });
  addText(
    slide,
    "When memory pressure forces evictions, SGLang keeps KV that is more likely to matter soon.",
    58,
    88,
    940,
    28,
    { size: 19, color: C.muted },
  );
  addText(slide, "Request win + trade-off", 990, 50, 210, 26, {
    size: 18,
    bold: true,
    color: C.success,
    align: "right",
  });
  addRule(slide, 56, 126, 1168);

  addSection(
    slide,
    "What the setup tests",
    [
      "32 replay requests run in both modes.",
      "Concurrent long-context sessions force KV eviction.",
      "Baseline uses ordinary eviction behavior.",
      "Controller mode marks cached prefixes by value.",
    ],
    66,
    160,
    455,
  );

  addRect(slide, 66, 374, 455, 154, { fill: C.panel, line: "#dbe4ef", lineWidth: 1 });
  addText(slide, "What the controller does", 88, 394, 390, 26, { size: 19, bold: true, color: C.ink });
  addText(
    slide,
    "It helps SGLang avoid evicting KV that a soon-returning agent is likely to need.",
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
  addText(slide, "Latest all-request run: value_aware_eviction_all_requests_fixed_20260917_163518", 590, 192, 600, 24, {
    size: 13,
    color: C.muted,
  });

  addMetric(slide, "Total TTFT", "51.6% better", "482.3 s to 233.5 s", 590, 228, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Avg TTFT per request", "7.8 s faster", "15.1 s to 7.3 s", 882, 228, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Total lateness", "34.9% better", "1014.0 s to 659.9 s", 590, 334, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Avg lateness", "11.1 s less late", "31.7 s to 20.6 s", 882, 334, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Value alignment", "0 mismatches", "23 high, 16 normal, 9 low", 590, 440, 270, C.winFill, C.ink, 88);
  addMetric(slide, "Workload window", "63.1 s longer", "205.6 s to 268.8 s", 882, 440, 270, C.tradeFill, C.tradeText, 88);

  addHarnessSignals(
    slide,
    "Session ID, prefix ID, expected return time, reuse probability, rebuild cost, replay deadline, work value, cancelable flag.",
  );
  addCommonSource(slide);

  slide.speakerNotes.textFrame.setText(`Scenario 3 source: harness_aware_scenario_tracker.html.

Setup: same 32 replay requests in both modes using the same generated timeline. The load generator creates enough concurrent long-context sessions to force KV eviction decisions. Baseline mode runs without value-aware eviction signals. Controller mode scores every replay-capable prefix as high, normal, or low value from harness signals, and SGLang's native priority radix eviction path performs the real eviction.

Run: value_aware_eviction_all_requests_fixed_20260917_163518 compared no_prefetch against controller_value_aware_eviction.

Results: Total TTFT improved from 482.3 s to 233.5 s, or 51.6%. Average replay TTFT improved from 15.1 s to 7.3 s. Total lateness improved from 1014.0 s to 659.9 s, or 34.9%. Average replay lateness improved from 31.7 s to 20.6 s. The trade-off is that the overall workload window increased from 205.6 s to 268.8 s, or 63.1 s longer.

Mechanism evidence: value classes reached SGLang as 23 high, 16 normal, and 9 low values with zero mismatches. SGLang ran with radix priority eviction. Native SGLang and HiCache eviction events were observed: 324 device eviction events and 77 host eviction events.

Important caveat: this produced a strong replay responsiveness win, but not a no-cost system win. The next tuning question is how to keep the TTFT and lateness benefit while reducing the workload-window cost.

[Sources]
Local file: /Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware/harness_aware_scenario_tracker.html`);
  slide.speakerNotes.setVisible(true);
}

async function main() {
  await fs.mkdir(TMP_DIR, { recursive: true });
  await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  addScenario1(presentation);
  addScenario2(presentation);
  addScenario3(presentation);

  const candidatePath = path.join(TMP_DIR, `scenario_master_candidate_${buildStamp}.pptx`);
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

  const result = await finalizePresentation({
    explicitTotalSlideCount: 3,
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
    receiptPath: path.join(TMP_DIR, `scenario_master_validation_${buildStamp}.json`),
  });

  console.log(`Finalized ${result.finalPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
