from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class MetricsWriter:
    def __init__(self, results_dir: str, mode: str) -> None:
        root = Path(results_dir)
        root.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        self.csv_path = root / f"{mode}_{timestamp}.csv"
        self.jsonl_path = root / f"{mode}_{timestamp}.jsonl"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "session_id",
                "mode",
                "priority",
                "tool_wait_ms",
                "ttft_ms",
                "total_latency_ms",
                "prefetch_attempted",
                "prefetch_success",
            ],
        )
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.jsonl_file.write(json.dumps(row) + "\n")
        self.jsonl_file.flush()
        self.csv_file.flush()

    def close(self) -> None:
        self.csv_file.close()
        self.jsonl_file.close()
        print(f"Wrote metrics to {self.csv_path}")
