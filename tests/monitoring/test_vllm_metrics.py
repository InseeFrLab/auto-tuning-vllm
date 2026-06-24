"""Unit tests for vLLM Prometheus metrics collector."""

from __future__ import annotations

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.monitoring.vllm_metrics import (
    VLLMMetricsCollector,
    aggregate_counter_stat,
    compute_stat,
    filter_samples_to_window,
    parse_required_metrics,
    strip_prometheus_namespace,
)

PROMETHEUS_FIXTURE = """
# HELP foo_bar Example gauge with labels
# TYPE foo_bar gauge
foo_bar{engine="a"} 10.0
foo_bar{engine="b"} 20.0
# HELP vllm:events_total Example counter
# TYPE vllm:events_total counter
vllm:events_total{source="a"} 80.0
vllm:events_total{source="b"} 100.0
"""


def test_strip_prometheus_namespace():
    assert strip_prometheus_namespace("vllm:events_total") == "events_total"
    assert strip_prometheus_namespace("foo_bar") == "foo_bar"


def test_parse_required_metrics_filters_and_collapses_labels():
    required = {"foo_bar": {"max"}, "events_total": {"mean"}}
    parsed = parse_required_metrics(PROMETHEUS_FIXTURE, required)
    assert parsed["foo_bar"] == (20.0, "gauge")
    assert parsed["events_total"] == (100.0, "counter")


def test_compute_stat_helpers():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert compute_stat(values, "mean") == 3.0
    assert compute_stat(values, "median") == 3.0
    assert compute_stat(values, "min") == 1.0
    assert compute_stat(values, "max") == 5.0
    assert compute_stat(values, "p95") == 4.8
    assert compute_stat([2.0], "std_dev") == 0.0
    assert compute_stat([1.0, 3.0], "std_dev") == 1.4142135623730951


def test_aggregate_counter_stat_mean_is_rate_other_stats_are_delta():
    values = [100.0, 160.0]
    assert aggregate_counter_stat(values, "mean", 30.0) == 2.0
    assert aggregate_counter_stat(values, "p95", 30.0) == 60.0
    assert aggregate_counter_stat([100.0], "mean", 30.0) is None


def test_filter_samples_to_window_excludes_warmup_and_cooldown():
    samples = [
        (0.0, 1.0, "gauge"),
        (2.0, 2.0, "gauge"),
        (5.0, 3.0, "gauge"),
        (8.0, 4.0, "gauge"),
        (10.0, 5.0, "gauge"),
    ]
    filtered = filter_samples_to_window(
        samples,
        started_at=0.0,
        ended_at=10.0,
        warmup_s=2.0,
        cooldown_s=2.0,
    )
    assert [sample[1] for sample in filtered] == [2.0, 3.0, 4.0]


def test_stop_and_collect_gauge_and_counter(monkeypatch):
    collector = VLLMMetricsCollector(
        metrics_url="http://localhost:8000/metrics",
        required={"foo_bar": {"mean", "p95"}, "events_total": {"mean", "max"}},
        interval_seconds=10.0,
        benchmark_config=BenchmarkConfig(model="m", max_seconds=100),
        align_with_benchmark_window=False,
    )
    collector._started_at = 0.0
    collector._stopped = False
    collector._samples = {
        "foo_bar": [
            (1.0, 10.0, "gauge"),
            (2.0, 20.0, "gauge"),
            (3.0, 30.0, "gauge"),
        ],
        "events_total": [
            (1.0, 100.0, "counter"),
            (3.0, 160.0, "counter"),
        ],
    }

    monkeypatch.setattr(
        "auto_tune_vllm.monitoring.vllm_metrics.time.monotonic",
        lambda: 3.0,
    )

    results = collector.stop_and_collect()
    assert results["vllm_foo_bar_mean"] == 20.0
    assert results["vllm_foo_bar_p95"] == 29.0
    assert results["vllm_events_total_mean"] == 20.0
    assert results["vllm_events_total_max"] == 60.0


def test_stop_and_collect_skips_missing_in_window_samples(monkeypatch):
    collector = VLLMMetricsCollector(
        metrics_url="http://localhost:8000/metrics",
        required={"foo_bar": {"mean"}},
        interval_seconds=10.0,
        benchmark_config=BenchmarkConfig(
            model="m",
            max_seconds=100,
            warmup=2,
            cooldown=2,
        ),
        align_with_benchmark_window=True,
    )
    collector._started_at = 0.0
    collector._stopped = False
    collector._samples = {
        "foo_bar": [
            (0.5, 10.0, "gauge"),
            (9.5, 20.0, "gauge"),
        ],
    }

    monkeypatch.setattr(
        "auto_tune_vllm.monitoring.vllm_metrics.time.monotonic",
        lambda: 10.0,
    )

    results = collector.stop_and_collect()
    assert results == {}
