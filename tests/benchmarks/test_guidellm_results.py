"""Reject reports that would silently turn inference failures into objectives."""

import pytest

from auto_tune_vllm.benchmarks.providers import GuideLLMBenchmark


@pytest.mark.parametrize("successful_count", [0, 1])
def test_report_requires_successful_requests(successful_count):
    provider = GuideLLMBenchmark()
    metrics = {
        metric: {category: {"count": successful_count, "mean": 1.0, "median": 1.0}}
        for metric, category in provider.METRIC_CATEGORIES.items()
    }
    report = {"benchmarks": [{"metrics": metrics}]}
    if successful_count == 0:
        with pytest.raises(RuntimeError, match="without successful requests"):
            provider._parse_guidellm_results(report)
    else:
        assert provider._parse_guidellm_results(report)["request_latency"] == 1.0
