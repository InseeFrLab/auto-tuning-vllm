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


def test_log_metrics_valid_vllm_entries():
    cfg = OptimizationConfig(
        approach="single_objective",
        objectives=[
            ObjectiveConfig(
                metric="output_tokens_per_second_mean",
                direction="maximize",
            )
        ],
        log_metrics=[
            "request_latency_p95",
            "vllm_foo_bar_p90",
            "vllm_num_events_total_mean",
        ],
    )
    assert cfg.log_metrics == [
        "request_latency_p95",
        "vllm_foo_bar_p90",
        "vllm_num_events_total_mean",
    ]


def test_log_metrics_invalid_vllm_stat_raises():
    with pytest.raises(ValueError, match="unknown stat"):
        OptimizationConfig(
            approach="single_objective",
            objectives=[
                ObjectiveConfig(
                    metric="output_tokens_per_second_mean",
                    direction="maximize",
                )
            ],
            log_metrics=["vllm_foo_bar_unknown"],
        )


def test_log_metrics_invalid_vllm_empty_name_raises():
    with pytest.raises(ValueError, match="prometheus name must be non-empty"):
        OptimizationConfig(
            approach="single_objective",
            objectives=[
                ObjectiveConfig(
                    metric="output_tokens_per_second_mean",
                    direction="maximize",
                )
            ],
            log_metrics=["vllm__p95"],
        )


def test_resolve_required_vllm_metrics_empty_when_no_vllm_ids():
    cfg = OptimizationConfig(
        approach="single_objective",
        objectives=[
            ObjectiveConfig(
                metric="output_tokens_per_second_mean",
                direction="maximize",
            )
        ],
        log_metrics=["request_latency_p95"],
    )
    assert cfg.resolve_required_vllm_metrics() == {}


def test_resolve_required_vllm_metrics_groups_stats_by_name():
    cfg = OptimizationConfig(
        approach="single_objective",
        objectives=[
            ObjectiveConfig(
                metric="output_tokens_per_second_mean",
                direction="maximize",
            )
        ],
        log_metrics=[
            "vllm_foo_bar_p90",
            "vllm_foo_bar_p95",
            "vllm_events_total_mean",
        ],
    )
    assert cfg.resolve_required_vllm_metrics() == {
        "foo_bar": {"p90", "p95"},
        "events_total": {"mean"},
    }


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
