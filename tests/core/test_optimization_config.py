"""Unit tests for OptimizationConfig.log_metrics validation."""

from __future__ import annotations

import pytest

from auto_tune_vllm.core.config import ObjectiveConfig, OptimizationConfig


def test_log_metrics_default_normalized_to_empty_list():
    cfg = OptimizationConfig(
        approach="single_objective",
        objectives=[
            ObjectiveConfig(
                metric="output_tokens_per_second_mean",
                direction="maximize",
            )
        ],
    )
    assert cfg.log_metrics == []


def test_log_metrics_valid_entries():
    cfg = OptimizationConfig(
        approach="single_objective",
        objectives=[
            ObjectiveConfig(
                metric="output_tokens_per_second_mean",
                direction="maximize",
            )
        ],
        log_metrics=["time_to_first_token_ms_p95", "request_latency_median"],
    )
    assert cfg.log_metrics == [
        "time_to_first_token_ms_p95",
        "request_latency_median",
    ]


def test_log_metrics_invalid_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        OptimizationConfig(
            approach="single_objective",
            objectives=[
                ObjectiveConfig(
                    metric="output_tokens_per_second_mean",
                    direction="maximize",
                )
            ],
            log_metrics=["not_a_valid_metric_p95"],
        )


def test_log_metrics_wrong_container_type_raises():
    with pytest.raises(ValueError, match="log_metrics must be a list"):
        OptimizationConfig(
            approach="single_objective",
            objectives=[
                ObjectiveConfig(
                    metric="output_tokens_per_second_mean",
                    direction="maximize",
                )
            ],
            log_metrics="time_to_first_token_ms_p95",  # type: ignore[arg-type]
        )
