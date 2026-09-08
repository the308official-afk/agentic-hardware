from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TimingEstimator:
    """Small P50/P95/P99 estimator used by the portable controller policy."""

    default_p99_ms: int = 25
    _samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, metric: str, value_ms: float) -> None:
        if value_ms >= 0:
            self._samples[metric].append(float(value_ms))

    def percentile(self, metric: str, percentile: float, default_ms: float | None = None) -> float:
        values = sorted(self._samples.get(metric, []))
        if not values:
            return float(default_ms if default_ms is not None else self.default_p99_ms)
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * percentile
        lower = int(rank)
        upper = min(lower + 1, len(values) - 1)
        weight = rank - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    def p50(self, metric: str, default_ms: float | None = None) -> float:
        return self.percentile(metric, 0.50, default_ms)

    def p95(self, metric: str, default_ms: float | None = None) -> float:
        return self.percentile(metric, 0.95, default_ms)

    def p99(self, metric: str, default_ms: float | None = None) -> float:
        return self.percentile(metric, 0.99, default_ms)
