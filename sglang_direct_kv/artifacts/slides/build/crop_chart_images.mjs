import path from "node:path";
import sharp from "sharp";

const repoRoot = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware";
const imgDir = path.join(repoRoot, "sglang_direct_kv/artifacts/slides/images");

await sharp(path.join(imgDir, "readable_phase_timeline.png"))
  .extract({ left: 0, top: 0, width: 3404, height: 1850 })
  .toFile(path.join(imgDir, "readable_phase_timeline_top.png"));

await sharp(path.join(imgDir, "readable_phase_timeline.png"))
  .extract({ left: 0, top: 0, width: 3404, height: 1320 })
  .toFile(path.join(imgDir, "readable_phase_timeline_2rows.png"));

await sharp(path.join(imgDir, "readable_phase_timeline.png"))
  .extract({ left: 0, top: 0, width: 3404, height: 3000 })
  .toFile(path.join(imgDir, "readable_phase_timeline_8rows.png"));

await sharp(path.join(imgDir, "readable_phase_timeline.png"))
  .extract({ left: 0, top: 0, width: 3404, height: 4242 })
  .resize({ width: 3404, height: 1400, fit: "fill" })
  .toFile(path.join(imgDir, "readable_phase_timeline_8rows_wide.png"));

await sharp(path.join(imgDir, "readable_phase_timeline.png"))
  .extract({ left: 0, top: 0, width: 3404, height: 2160 })
  .resize({ width: 3404, height: 1400, fit: "fill" })
  .toFile(path.join(imgDir, "readable_phase_timeline_4rows_wide.png"));

await sharp(path.join(imgDir, "synthetic_profiled_mechanism_timeline.png"))
  .extract({ left: 0, top: 0, width: 3000, height: 1372 })
  .resize({ width: 3000, height: 1100, fit: "fill" })
  .toFile(path.join(imgDir, "synthetic_profiled_mechanism_timeline_compact.png"));
