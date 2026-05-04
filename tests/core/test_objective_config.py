"""Unit tests for ObjectiveConfig._break_down_objectives."""

from __future__ import annotations

import pytest

from auto_tune_vllm.core.config import ObjectiveConfig


class TestBreakDownObjectivesValid:
    """Cases where the metric expression is valid and parsing should succeed."""

    def test_single_metric(self):
        obj = ObjectiveConfig(
            metric="output_tokens_per_second_median",
            direction="maximize",
        )
        assert obj._break_down_objectives() == ["output_tokens_per_second_median"]

    def test_two_metrics_division(self):
        obj = ObjectiveConfig(
            metric="output_tokens_per_second_mean / requests_per_second_median",
            direction="maximize",
        )
        assert obj._break_down_objectives() == [
            "output_tokens_per_second_mean",
            "requests_per_second_median",
        ]

    def test_expression_with_constant(self):
        # Constants must not appear in the returned list.
        obj = ObjectiveConfig(
            metric="output_tokens_per_second_mean / (requests_per_second_median + 1)",
            direction="maximize",
        )
        assert obj._break_down_objectives() == [
            "output_tokens_per_second_mean",
            "requests_per_second_median",
        ]

    def test_deduplicates_preserving_first_seen_order(self):
        obj = ObjectiveConfig(
            metric=(
                "output_tokens_per_second_mean "
                "+ requests_per_second_median "
                "- output_tokens_per_second_mean"
            ),
            direction="maximize",
        )
        assert obj._break_down_objectives() == [
            "output_tokens_per_second_mean",
            "requests_per_second_median",
        ]

    def test_same_metric_different_percentiles_kept_separately(self):
        obj = ObjectiveConfig(
            metric="request_latency_p95 - request_latency_p50",
            direction="minimize",
        )
        assert obj._break_down_objectives() == [
            "request_latency_p95",
            "request_latency_p50",
        ]

    def test_order_follows_ast_walk(self):
        obj = ObjectiveConfig(
            metric="requests_per_second_median * output_tokens_per_second_mean",
            direction="maximize",
        )
        assert obj._break_down_objectives() == [
            "requests_per_second_median",
            "output_tokens_per_second_mean",
        ]

    def test_all_supported_percentiles(self):
        # Sanity-check that every documented percentile parses cleanly.
        for percentile in ("median", "p50", "p90", "p95", "p99", "mean"):
            metric = f"request_latency_{percentile}"
            obj = ObjectiveConfig(metric=metric, direction="minimize")
            assert obj._break_down_objectives() == [metric]


class TestBreakDownObjectivesInvalid:
    """Cases where the metric expression is invalid; __post_init__ raises."""

    def test_unknown_metric_name(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            ObjectiveConfig(metric="unknown_metric_p95", direction="maximize")

    def test_metric_without_percentile_suffix(self):
        # "output_tokens_per_second" alone is not in valid_metrics_combined;
        # only the metric_percentile combinations are valid.
        with pytest.raises(ValueError, match="Unknown metric"):
            ObjectiveConfig(
                metric="output_tokens_per_second",
                direction="maximize",
            )

    def test_invalid_percentile_suffix(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            ObjectiveConfig(metric="request_latency_p42", direction="minimize")

    def test_unknown_identifier_in_otherwise_valid_expression(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            ObjectiveConfig(
                metric="output_tokens_per_second_mean / bogus_metric_p95",
                direction="maximize",
            )

    def test_syntax_error(self):
        with pytest.raises(ValueError, match="Invalid metric expression"):
            ObjectiveConfig(metric="output_tokens_per_second_mean +", direction="maximize")

    def test_empty_expression(self):
        with pytest.raises(ValueError, match="Invalid metric expression"):
            ObjectiveConfig(metric="", direction="maximize")


class TestBreakDownObjectivesDirectCall:
    """Calling _break_down_objectives after mutating self.metric.

    __post_init__ runs at construction, so to exercise the method standalone
    we build a valid instance first, then swap in the metric under test.
    """

    @staticmethod
    def _make_obj() -> ObjectiveConfig:
        return ObjectiveConfig(
            metric="output_tokens_per_second_median",
            direction="maximize",
        )

    def test_method_returns_list_type(self):
        obj = self._make_obj()
        result = obj._break_down_objectives()
        assert isinstance(result, list)
        assert all(isinstance(m, str) for m in result)

    def test_method_raises_on_invalid_after_mutation(self):
        obj = self._make_obj()
        obj.metric = "not_a_metric_p95"
        with pytest.raises(ValueError, match="Unknown metric"):
            obj._break_down_objectives()

    def test_method_handles_complex_expression_after_mutation(self):
        obj = self._make_obj()
        obj.metric = (
            "(output_tokens_per_second_mean - request_latency_p95) "
            "/ (requests_per_second_median + 1)"
        )
        assert obj._break_down_objectives() == [
            "output_tokens_per_second_mean",
            "request_latency_p95",
            "requests_per_second_median",
        ]
