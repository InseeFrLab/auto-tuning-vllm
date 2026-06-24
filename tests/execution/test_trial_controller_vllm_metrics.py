"""Unit tests for vLLM metrics integration in trial controller."""

from __future__ import annotations

from unittest.mock import MagicMock

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.core.config import ObjectiveConfig, OptimizationConfig
from auto_tune_vllm.core.trial import ExecutionInfo, TrialConfig
from auto_tune_vllm.execution.trial_controller import LocalTrialController


class _StubBenchmarkProvider:
    def parse_results(self):
        return {"request_latency_p95": 42.0}


def _make_trial_config(log_metrics: list[str] | None = None) -> TrialConfig:
    return TrialConfig(
        study_name="test_study",
        trial_id="trial_0",
        trial_number=0,
        benchmark_config=BenchmarkConfig(model="test-model"),
        optimization_config=OptimizationConfig(
            approach="single_objective",
            objectives=[
                ObjectiveConfig(
                    metric="output_tokens_per_second_mean",
                    direction="maximize",
                )
            ],
            log_metrics=log_metrics or [],
        ),
    )


def test_start_vllm_metrics_collector_skipped_when_no_required_metrics():
    controller = LocalTrialController()
    trial_config = _make_trial_config(log_metrics=["request_latency_p95"])

    controller._start_vllm_metrics_collector(
        trial_config,
        {"url": "http://localhost:8000/v1"},
        MagicMock(),
    )

    assert controller.vllm_metrics_collector is None


def test_handle_benchmark_running_merges_vllm_metrics(monkeypatch):
    controller = LocalTrialController()
    controller.benchmark_provider = _StubBenchmarkProvider()
    controller.vllm_metrics_collector = MagicMock()
    controller.vllm_metrics_collector.stop_and_collect.return_value = {
        "vllm_foo_p95": 1.23,
    }

    trial_config = _make_trial_config(log_metrics=["vllm_foo_p95"])
    benchmark_process = MagicMock()
    benchmark_process.poll.return_value = 0
    benchmark_process.communicate.return_value = ("", "")

    monkeypatch.setattr(
        controller,
        "_check_health_status",
        lambda: None,
    )
    monkeypatch.setattr(
        controller,
        "_extract_objectives",
        lambda benchmark_result, optimization_config: [1.0],
    )

    result = controller._handle_benchmark_running(
        benchmark_process,
        benchmark_start_time=0.0,
        trial_config=trial_config,
        execution_info=ExecutionInfo(),
        logger=MagicMock(),
    )

    assert result is not None
    assert result.detailed_metrics["request_latency_p95"] == 42.0
    assert result.detailed_metrics["vllm_foo_p95"] == 1.23
    assert controller.vllm_metrics_collector is None
