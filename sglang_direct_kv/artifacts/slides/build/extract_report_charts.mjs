import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

const repoRoot = "/Users/oluwolejaiyeoba/Documents/GitHub/agentic_hardware";
const reportPath = path.join(repoRoot, "sglang_direct_kv/artifacts/results/latest_master_report.html");
const backupReportPath = path.join(repoRoot, "backups/latest_master_report-1.html");
const outDir = path.join(repoRoot, "sglang_direct_kv/artifacts/slides/images");

const charts = [
  {
    name: "readable_phase_timeline.png",
    selector: 'svg[aria-label="Readable phase timeline with local timing inside each column"]',
  },
  {
    name: "global_h2d_readiness.png",
    selector: 'svg[aria-label="Global replay H2D readiness dot plot"]',
  },
  {
    name: "replay_queue_timeline.png",
    selector: 'svg[aria-label="Replay request versus H2D start timeline plot"]',
  },
  {
    name: "replay_execution_timeline.png",
    selector: 'svg[aria-label="Replay execution timeline aligned at actual resume start"]',
  },
];

const backupCharts = [
  {
    name: "global_prefetch_margin_backup.png",
    selector: 'svg[aria-label="Global prefetch margin dot plot"]',
  },
];

async function main() {
  await fs.mkdir(outDir, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  });
  const page = await browser.newPage({ viewport: { width: 1800, height: 1400 }, deviceScaleFactor: 2 });
  await page.goto(`file://${reportPath}`, { waitUntil: "load" });
  await page.evaluate(() => {
    for (const details of document.querySelectorAll("details")) {
      details.open = true;
    }
    document.body.style.background = "#ffffff";
  });

  for (const chart of charts) {
    const locator = page.locator(chart.selector).first();
    await locator.waitFor({ state: "visible" });
    await locator.screenshot({
      path: path.join(outDir, chart.name),
      omitBackground: false,
    });
  }

  await page.goto(`file://${backupReportPath}`, { waitUntil: "load" });
  await page.evaluate(() => {
    for (const details of document.querySelectorAll("details")) {
      details.open = true;
    }
    document.body.style.background = "#ffffff";
  });

  for (const chart of backupCharts) {
    const locator = page.locator(chart.selector).first();
    await locator.waitFor({ state: "visible" });
    await locator.screenshot({
      path: path.join(outDir, chart.name),
      omitBackground: false,
    });
  }
  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
