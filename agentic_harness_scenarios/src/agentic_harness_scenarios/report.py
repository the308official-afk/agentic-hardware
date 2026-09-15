from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .metrics import improvement_label
from .models import ScenarioRun


def write_outputs(runs: list[ScenarioRun], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(runs, out / "scenario_results.json")
    write_csv(runs, out / "scenario_results.csv")
    (out / "scenario_report.html").write_text(render_html(runs), encoding="utf-8")


def write_json(runs: list[ScenarioRun], path: str | Path) -> None:
    payload = []
    for run in runs:
        payload.append(
            {
                "scenario": {
                    "scenario_id": run.scenario.scenario_id,
                    "brief_claim": run.scenario.brief_claim,
                    "scenario_type": run.scenario.scenario_type,
                    "description": run.scenario.description,
                    "expected_benefit": run.scenario.expected_benefit,
                    "harness_signals": run.scenario.harness_signals,
                    "backend_state": run.scenario.backend_state,
                },
                "baseline": _result_dict(run.baseline),
                "harness_aware": _result_dict(run.harness_aware),
                "primary_metric": run.primary_metric,
                "improvement": run.improvement,
            }
        )
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(runs: list[ScenarioRun], path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for item in (run.baseline, run.harness_aware):
            row = {
                "scenario_id": item.scenario_id,
                "brief_claim": item.brief_claim,
                "mode": item.mode,
                "policy": item.policy,
                "decision": item.decision,
                "benefit_summary": item.benefit_summary,
                "harness_signals_used": " ".join(item.harness_signals_used),
                "backend_state_used": " ".join(item.backend_state_used),
            }
            row.update(item.metrics)
            rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_html(runs: list[ScenarioRun]) -> str:
    cards = "\n".join(_render_card(run) for run in runs)
    score_rows = "\n".join(_render_score_row(run) for run in runs)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Harness-Aware Scenario Report</title>
<style>
body {{ margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f8fafc; }}
header {{ padding: 28px 36px; background: #101827; color: white; }}
main {{ padding: 28px 36px 48px; max-width: 1180px; margin: 0 auto; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin: 0 0 14px; font-size: 20px; }}
h3 {{ margin: 0 0 8px; font-size: 16px; }}
.panel {{ background: white; border: 1px solid #dbe4ee; border-radius: 8px; padding: 18px; margin: 18px 0; box-shadow: 0 1px 2px rgba(15, 23, 42, .04); }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5edf5; vertical-align: top; }}
th {{ color: #475569; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }}
.decision {{ border-left: 4px solid #94a3b8; padding: 10px 12px; background: #f8fafc; border-radius: 6px; }}
.aware {{ border-left-color: #16a34a; }}
.baseline {{ border-left-color: #64748b; }}
.metric {{ color: #0f766e; font-weight: 700; }}
.muted {{ color: #64748b; }}
.bar-label {{ font-size: 12px; fill: #334155; font-weight: 700; }}
.note {{ color: #475569; }}
code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<header>
<h1>Harness-Aware Inference Scenario Report</h1>
<div>Minimal synthetic demonstrations of harness-level signals changing backend decisions.</div>
</header>
<main>
<section class="panel">
<h2>Executive Scorecard</h2>
<table>
<thead><tr><th>Claim</th><th>Primary Metric</th><th>Baseline</th><th>Harness-Aware</th><th>Improvement</th></tr></thead>
<tbody>
{score_rows}
</tbody>
</table>
</section>
{cards}
</main>
</body>
</html>
"""


def _render_score_row(run: ScenarioRun) -> str:
    key = run.primary_metric
    base = run.baseline.metrics.get(key, "")
    aware = run.harness_aware.metrics.get(key, "")
    return (
        "<tr>"
        f"<td>{html.escape(run.scenario.brief_claim)}</td>"
        f"<td><code>{html.escape(key)}</code></td>"
        f"<td>{html.escape(str(base))}</td>"
        f"<td>{html.escape(str(aware))}</td>"
        f"<td class=\"metric\">{html.escape(improvement_label(run))}</td>"
        "</tr>"
    )


def _render_card(run: ScenarioRun) -> str:
    key = run.primary_metric
    base = run.baseline.metrics.get(key)
    aware = run.harness_aware.metrics.get(key)
    return f"""<section class="panel">
<h2>{html.escape(run.scenario.brief_claim)}</h2>
<p class="note">{html.escape(run.scenario.description)}</p>
{_render_bar_svg(str(key), base, aware)}
<div class="grid">
<div class="decision baseline"><h3>Baseline</h3><p>{html.escape(run.baseline.decision)}</p><p class="muted">{html.escape(run.baseline.benefit_summary)}</p></div>
<div class="decision aware"><h3>Harness-Aware</h3><p>{html.escape(run.harness_aware.decision)}</p><p class="muted">{html.escape(run.harness_aware.benefit_summary)}</p></div>
</div>
<p><strong>Harness signal exposed:</strong> {html.escape(", ".join(run.baseline.harness_signals_used))}</p>
<p><strong>Backend state held constant:</strong> {html.escape(", ".join(run.baseline.backend_state_used))}</p>
</section>"""


def _render_bar_svg(metric: str, baseline: Any, aware: Any) -> str:
    if not isinstance(baseline, (int, float)) or not isinstance(aware, (int, float)):
        return ""
    max_value = max(abs(float(baseline)), abs(float(aware)), 1.0)
    base_w = 420 * abs(float(baseline)) / max_value
    aware_w = 420 * abs(float(aware)) / max_value
    return f"""<svg viewBox="0 0 620 96" width="100%" height="96" role="img" aria-label="{html.escape(metric)} comparison">
<text x="0" y="16" class="bar-label">{html.escape(metric)}</text>
<text x="0" y="42" class="bar-label">Baseline</text>
<rect x="140" y="28" width="{base_w:.1f}" height="18" rx="3" fill="#64748b"/>
<text x="{150 + base_w:.1f}" y="42" class="bar-label">{html.escape(str(baseline))}</text>
<text x="0" y="74" class="bar-label">Harness-aware</text>
<rect x="140" y="60" width="{aware_w:.1f}" height="18" rx="3" fill="#16a34a"/>
<text x="{150 + aware_w:.1f}" y="74" class="bar-label">{html.escape(str(aware))}</text>
</svg>"""


def _result_dict(result: Any) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "policy": result.policy,
        "decision": result.decision,
        "metrics": result.metrics,
        "benefit_summary": result.benefit_summary,
        "harness_signals_used": list(result.harness_signals_used),
        "backend_state_used": list(result.backend_state_used),
    }
