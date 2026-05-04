"""Unit tests for BaseTrialController.evaluate_metric_expression."""

from __future__ import annotations

import pytest

from auto_tune_vllm.core.trial import TrialConfig, TrialResult
from auto_tune_vllm.execution.trial_controller import BaseTrialController


class _StubController(BaseTrialController):
    """Minimal concrete subclass so the abstract base can be instantiated.

    evaluate_metric_expression does not touch any of these methods, so they
    can stay as no-ops for the purposes of these tests.
    """

    def run_trial(self, trial_config: TrialConfig) -> TrialResult:  # pragma: no cover
        raise NotImplementedError

    def cleanup_resources(self) -> None:  # pragma: no cover
        return None

    def request_cancellation(self) -> None:  # pragma: no cover
        return None


@pytest.fixture
def controller() -> _StubController:
    return _StubController()


class TestEvaluateMetricExpressionValid:
    """Expressions that should evaluate to a numeric result."""

    def test_single_metric(self, controller: _StubController):
        assert controller.evaluate_metric_expression("x", {"x": 5.0}) == 5.0

    def test_returns_float_type(self, controller: _StubController):
        # Even when inputs are ints, the result is coerced to float.
        result = controller.evaluate_metric_expression("a", {"a": 7})
        assert isinstance(result, float)
        assert result == 7.0

    def test_integer_literal(self, controller: _StubController):
        assert controller.evaluate_metric_expression("42", {}) == 42.0

    def test_float_literal(self, controller: _StubController):
        assert controller.evaluate_metric_expression("3.14", {}) == pytest.approx(3.14)

    def test_addition(self, controller: _StubController):
        assert controller.evaluate_metric_expression("a + b", {"a": 2, "b": 3}) == 5.0

    def test_subtraction(self, controller: _StubController):
        assert controller.evaluate_metric_expression("a - b", {"a": 10, "b": 4}) == 6.0

    def test_multiplication(self, controller: _StubController):
        assert controller.evaluate_metric_expression("a * b", {"a": 6, "b": 7}) == 42.0

    def test_division(self, controller: _StubController):
        result = controller.evaluate_metric_expression("a / b", {"a": 9, "b": 4})
        assert result == pytest.approx(2.25)

    def test_power(self, controller: _StubController):
        assert controller.evaluate_metric_expression("a ** 2", {"a": 3}) == 9.0

    def test_unary_minus(self, controller: _StubController):
        assert controller.evaluate_metric_expression("-a", {"a": 5}) == -5.0

    def test_unary_minus_combined(self, controller: _StubController):
        result = controller.evaluate_metric_expression("a + -b", {"a": 5, "b": 3})
        assert result == 2.0

    def test_parentheses_change_precedence(self, controller: _StubController):
        result = controller.evaluate_metric_expression(
            "(a + b) * c", {"a": 1, "b": 2, "c": 3}
        )
        assert result == 9.0

    def test_complex_expression(self, controller: _StubController):
        # ((a + b) / (c - 1)) ** 2
        result = controller.evaluate_metric_expression(
            "((a + b) / (c - 1)) ** 2",
            {"a": 4, "b": 6, "c": 6},
        )
        assert result == pytest.approx(4.0)

    def test_real_world_metric_names(self, controller: _StubController):
        result = controller.evaluate_metric_expression(
            "output_tokens_per_second / requests_per_second",
            {"output_tokens_per_second": 1000.0, "requests_per_second": 10.0},
        )
        assert result == pytest.approx(100.0)

    def test_repeated_metric_reference(self, controller: _StubController):
        # The same metric may legitimately appear more than once.
        result = controller.evaluate_metric_expression("a + a + a", {"a": 4})
        assert result == 12.0

    def test_negative_metric_value(self, controller: _StubController):
        assert controller.evaluate_metric_expression("a + 1", {"a": -5}) == -4.0

    def test_extra_unused_metrics_ignored(self, controller: _StubController):
        # Unreferenced entries in the dictionary are harmless.
        result = controller.evaluate_metric_expression(
            "a", {"a": 1.0, "unused": 999.0}
        )
        assert result == 1.0


class TestEvaluateMetricExpressionErrors:
    """Expressions that should raise."""

    def test_missing_metric_value(self, controller: _StubController):
        with pytest.raises(ValueError, match=r"Missing value for metric: 'z'"):
            controller.evaluate_metric_expression("z + 1", {"a": 1})

    def test_division_by_zero(self, controller: _StubController):
        with pytest.raises(ZeroDivisionError):
            controller.evaluate_metric_expression("a / b", {"a": 1, "b": 0})

    def test_function_call_not_allowed(self, controller: _StubController):
        with pytest.raises(ValueError, match="Construct not allowed"):
            controller.evaluate_metric_expression("abs(a)", {"a": -1})

    def test_modulo_operator_not_allowed(self, controller: _StubController):
        with pytest.raises(ValueError, match="Operator not allowed"):
            controller.evaluate_metric_expression("a % b", {"a": 5, "b": 2})

    def test_floor_division_not_allowed(self, controller: _StubController):
        with pytest.raises(ValueError, match="Operator not allowed"):
            controller.evaluate_metric_expression("a // b", {"a": 5, "b": 2})

    def test_bitwise_and_not_allowed(self, controller: _StubController):
        with pytest.raises(ValueError, match="Operator not allowed"):
            controller.evaluate_metric_expression("a & b", {"a": 5, "b": 3})

    def test_unary_not_not_allowed(self, controller: _StubController):
        # `not a` parses to UnaryOp(Not, ...); Not is outside ALLOWED_OPERATORS.
        with pytest.raises(ValueError, match="Unary operator not allowed"):
            controller.evaluate_metric_expression("not a", {"a": 1})

    def test_string_literal_not_allowed(self, controller: _StubController):
        # Constant guard rejects non-numeric values, so this falls to the
        # "Construct not allowed" arm.
        with pytest.raises(ValueError, match="Construct not allowed"):
            controller.evaluate_metric_expression('"hello"', {})

    def test_comparison_not_allowed(self, controller: _StubController):
        with pytest.raises(ValueError, match="Construct not allowed"):
            controller.evaluate_metric_expression("a > b", {"a": 1, "b": 2})

    def test_attribute_access_not_allowed(self, controller: _StubController):
        with pytest.raises(ValueError, match="Construct not allowed"):
            controller.evaluate_metric_expression("a.b", {"a": 1})

    def test_syntax_error_propagates(self, controller: _StubController):
        # The function does not catch SyntaxError from ast.parse; it just
        # propagates. Document that behavior here.
        with pytest.raises(SyntaxError):
            controller.evaluate_metric_expression("a +", {"a": 1})
