"""Encoding evidence from gateway events; no engine implementation imports."""
from __future__ import annotations

import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_encoding_report(root: Path, out_dir: Path, replay_rows: list[dict], *, reuse_existing: bool = False) -> str:
    by_request = {(row["case_id"], row["request_id"]): row for row in replay_rows}
    evidence = []
    proof_path = out_dir / "prompt_encoding_proof.csv"
    if reuse_existing and proof_path.exists():
        with proof_path.open(newline="") as handle:
            evidence = list(csv.DictReader(handle))
        for row in evidence:
            for key in ("encoding_saved_tokens", "encoding_elapsed_ms", "first_token_lateness_ms"):
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
    for case in sorted(root.iterdir()) if root.exists() and not reuse_existing else []:
        if not case.is_dir():
            continue
        path = case / "harness_gateway_events.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") != "gateway.forwarded_request" or not event.get("encoding_config_hash"):
                continue
            row = {k: v for k, v in event.items() if k.startswith("encoding_")}
            row.update(case_id=case.name, request_id=event.get("request_id"), phase=event.get("phase"),
                       harness=event.get("harness"), mode=event.get("mode"),
                       backend_status=event.get("status"), error=event.get("error"),
                       gateway_to_first_content_ms=event.get("gateway_to_first_content_ms"),
                       backend_call_ttft_ms=event.get("ttft_ms"), task_quality="not_measured")
            replay = by_request.get((case.name, row["request_id"]), {})
            for key in ("pressure_level", "first_token_lateness_ms", "first_token_source", "sglang_receive_to_first_token_ms", "prefill_full_input_tokens", "prefill_cached_prefix_tokens", "prefill_uncached_token_count"):
                row[key] = replay.get(key, "")
            evidence.append(row)
    write_csv(out_dir / "prompt_encoding_proof.csv", evidence)
    grouped = defaultdict(list)
    for row in evidence:
        grouped[tuple(row.get(k, "") for k in ("harness", "mode", "phase", "pressure_level", "encoding_codec", "encoding_config_hash", "encoding_scope"))].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        row = dict(zip(("harness", "mode", "phase", "pressure_level", "codec", "config_hash", "scope"), key))
        row.update(requests=len(group), applied=sum(r["encoding_status"] == "applied" for r in group),
                   bypassed=sum(r["encoding_status"] != "applied" for r in group),
                   backend_failures=sum(bool(r.get("error")) for r in group))
        for metric in ("encoding_saved_tokens", "encoding_elapsed_ms", "first_token_lateness_ms"):
            values = [r[metric] for r in group if isinstance(r.get(metric), (int, float))]
            row["median_" + metric] = statistics.median(values) if values else ""
            if metric != "encoding_saved_tokens":
                row["p95_" + metric] = percentile(values, 0.95)
        if row["phase"] == "replay":
            row["on_time_success_rate"] = sum(not r.get("error") and isinstance(r.get("first_token_lateness_ms"), (float, int))
                                             and r["first_token_lateness_ms"] <= 0 for r in group) / len(group)
        row["task_quality"] = "not_measured"
        summaries.append(row)
    write_csv(out_dir / "prompt_encoding_summary.csv", summaries)
    columns = list(summaries[0]) if summaries else []
    table = "<table><thead><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in columns) + "</tr></thead><tbody>"
    table += "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in columns) + "</tr>" for row in summaries)
    table += "</tbody></table>"
    return ("<section><h2>Prompt encoding</h2><p>All requests, including bypasses and failures. "
            "Backend first content is observed at the gateway; this experiment gateway buffers client responses. "
            "Task quality requires the separate evaluation suite.</p>" + table + "</section>")
