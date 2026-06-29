"""Generic vLLM Prometheus metrics collector for benchmark windows."""

from __future__ import annotations

import logging
import statistics
import threading
import time

import requests
from prometheus_client.parser import text_string_to_metric_families

from ..benchmarks.config import BenchmarkConfig

logger = logging.getLogger(__name__)


def strip_prometheus_namespace(name: str) -> str:
    """Strip a ``vllm:`` namespace prefix from a Prometheus metric name."""
    if name.startswith("vllm:"):
        return name[len("vllm:") :]
    return name


def percentile(values: list[float], p: float) -> float:
    """Compute a percentile with linear interpolation (p in [0, 100])."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compute_stat(values: list[float], stat: str) -> float:
    """Compute a single statistic over a list of scalar samples."""
    if not values:
        raise ValueError("compute_stat requires at least one value")
    if stat == "mean":
        return statistics.mean(values)
    if stat == "median":
        return statistics.median(values)
    if stat == "min":
        return min(values)
    if stat == "max":
        return max(values)
    if stat == "std_dev":
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values)
    if stat == "p90":
        return percentile(values, 90.0)
    if stat == "p95":
        return percentile(values, 95.0)
    if stat == "p99":
        return percentile(values, 99.0)
    raise ValueError(f"Unknown stat: {stat!r}")


def resolve_benchmark_window_seconds(
    benchmark_config: BenchmarkConfig,
) -> tuple[float, float]:
    """Return absolute warmup and cooldown durations in seconds."""
    warmup_s = 0.0
    cooldown_s = 0.0

    if benchmark_config.warmup is not None:
        if 0 < benchmark_config.warmup < 1:
            warmup_s = benchmark_config.max_seconds * benchmark_config.warmup
        else:
            warmup_s = float(benchmark_config.warmup)

    if benchmark_config.cooldown is not None:
        if 0 < benchmark_config.cooldown < 1:
            cooldown_s = benchmark_config.max_seconds * benchmark_config.cooldown
        else:
            cooldown_s = float(benchmark_config.cooldown)

    return warmup_s, cooldown_s


def filter_samples_to_window(
    samples: list[tuple[float, float, str]],
    started_at: float,
    ended_at: float,
    warmup_s: float,
    cooldown_s: float,
) -> list[tuple[float, float, str]]:
    """Keep samples whose timestamps fall inside the benchmark measurement window."""
    window_start = started_at + warmup_s
    window_end = ended_at - cooldown_s
    if window_end < window_start:
        return []
    return [sample for sample in samples if window_start <= sample[0] <= window_end]


def aggregate_counter_stat(
    values: list[float],
    stat: str,
    window_duration: float,
) -> float | None:
    """Aggregate counter samples over a window (rate for mean, delta otherwise)."""
    if len(values) < 2:
        return None
    delta = values[-1] - values[0]
    if stat == "mean":
        if window_duration <= 0:
            return None
        return delta / window_duration
    return delta


def parse_required_metrics(
    text: str,
    required: dict[str, set[str]],
) -> dict[str, tuple[float, str]]:
    """
    Parse Prometheus text and return per-metric (value, type) for this scrape.

    Metrics are matched by sample name (after stripping ``vllm:``), because counter
    families may omit the ``_total`` suffix in ``MetricFamily.name``.
    Multi-label series are collapsed with max per metric name.
    """
    parsed: dict[str, tuple[float, str]] = {}
    for family in text_string_to_metric_families(text):
        metric_type = family.type
        for sample in family.samples:
            metric_name = strip_prometheus_namespace(sample.name)
            if metric_name not in required:
                continue
            current = parsed.get(metric_name)
            if current is None or sample.value > current[0]:
                parsed[metric_name] = (sample.value, metric_type)
    return parsed


class VLLMMetricsCollector:
    """Scrape vLLM ``/metrics`` during a benchmark and aggregate requested stats."""

    def __init__(
        self,
        metrics_url: str,
        required: dict[str, set[str]],
        interval_seconds: float,
        benchmark_config: BenchmarkConfig,
        align_with_benchmark_window: bool = True,
        custom_logger: logging.Logger | None = None,
    ):
        self.metrics_url = metrics_url
        self.required = required
        self.interval_seconds = interval_seconds
        self.benchmark_config = benchmark_config
        self.align_with_benchmark_window = align_with_benchmark_window
        self._logger = custom_logger or logger

        self._samples: dict[str, list[tuple[float, float, str]]] = {
            name: [] for name in required
        }
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._stopped = False

    def start(self) -> None:
        """Start background scraping."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._started_at = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scrape_loop,
            daemon=True,
            name="vllm-metrics-collector",
        )
        self._thread.start()
        self._logger.info(
            "Started vLLM metrics collector: url=%s interval=%ss required=%s",
            self.metrics_url,
            self.interval_seconds,
            sorted(self.required.keys()),
        )

    def stop_and_collect(self) -> dict[str, float]:
        """Stop scraping and return aggregated metrics keyed by log_metrics ids."""
        if self._stopped:
            return {}

        self._stopped = True
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

        if self._started_at is None:
            return {}

        ended_at = time.monotonic()
        warmup_s, cooldown_s = resolve_benchmark_window_seconds(self.benchmark_config)
        window_duration = ended_at - self._started_at
        if self.align_with_benchmark_window:
            window_duration = max(
                0.0,
                window_duration - warmup_s - cooldown_s,
            )

        results: dict[str, float] = {}
        for metric_name, stats_requested in self.required.items():
            samples = self._samples.get(metric_name, [])
            if self.align_with_benchmark_window:
                samples = filter_samples_to_window(
                    samples,
                    self._started_at,
                    ended_at,
                    warmup_s,
                    cooldown_s,
                )
            if not samples:
                self._logger.warning(
                    "vLLM metrics: no samples in window for %r; skipping",
                    metric_name,
                )
                continue

            values = [sample[1] for sample in samples]
            metric_type = samples[-1][2]

            for stat in stats_requested:
                key = f"vllm_{metric_name}_{stat}"
                if metric_type == "counter":
                    aggregated = aggregate_counter_stat(values, stat, window_duration)
                    if aggregated is None:
                        self._logger.warning(
                            "vLLM metrics: insufficient counter samples for %r; "
                            "skipping %s",
                            metric_name,
                            key,
                        )
                        continue
                    results[key] = aggregated
                else:
                    results[key] = compute_stat(values, stat)

        return results

    def _scrape_loop(self) -> None:
        period = self.interval_seconds
        next_deadline = time.monotonic()

        while not self._stop_event.is_set():
            self._scrape_once()
            next_deadline += period
            now = time.monotonic()
            while next_deadline <= now:
                next_deadline += period
            sleep_duration = max(0.0, next_deadline - now)
            if self._stop_event.wait(timeout=sleep_duration):
                break

        # Final scrape on stop for fresher end-of-window samples
        self._scrape_once()

    def _scrape_once(self) -> None:
        timestamp = time.monotonic()
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            self._logger.warning(
                "vLLM metrics scrape failed for %s: %s",
                self.metrics_url,
                exc,
            )
            return

        try:
            parsed = parse_required_metrics(
                response.text,
                self.required,
            )
        except Exception as exc:
            self._logger.warning(
                "vLLM metrics parse failed for %s: %s",
                self.metrics_url,
                exc,
            )
            return

        for metric_name, (value, metric_type) in parsed.items():
            self._samples[metric_name].append((timestamp, value, metric_type))
