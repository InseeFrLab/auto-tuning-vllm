"""Unit tests for GuideLLM CLI command construction."""

from __future__ import annotations

import pytest

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.benchmarks.providers import GuideLLMBenchmark


def _build_cmd(**kwargs) -> list[str]:
    config = BenchmarkConfig(model="test-model", **kwargs)
    return GuideLLMBenchmark()._build_guidellm_command(
        "http://localhost:8000/v1", config, "/tmp/results.json"
    )


def test_warmup_and_cooldown_included_when_set():
    cmd = _build_cmd(warmup=0.1, cooldown=0.1)
    assert cmd[cmd.index("--warmup") + 1] == "0.1"
    assert cmd[cmd.index("--cooldown") + 1] == "0.1"


def test_warmup_and_cooldown_omitted_when_unset():
    cmd = _build_cmd()
    assert "--warmup" not in cmd
    assert "--cooldown" not in cmd


def test_absolute_warmup_value_passed_through():
    cmd = _build_cmd(warmup=30, cooldown=None)
    assert cmd[cmd.index("--warmup") + 1] == "30"
    assert "--cooldown" not in cmd


def test_warmup_zero_rejected():
    with pytest.raises(ValueError, match="warmup"):
        BenchmarkConfig(model="m", warmup=0)


def test_negative_cooldown_rejected():
    with pytest.raises(ValueError, match="cooldown"):
        BenchmarkConfig(model="m", cooldown=-0.1)


def test_fraction_warmup_plus_cooldown_must_leave_measured_window():
    with pytest.raises(ValueError, match="sum to less than 1"):
        BenchmarkConfig(model="m", warmup=0.6, cooldown=0.5)


def test_mixed_fraction_and_absolute_skips_sum_check():
    BenchmarkConfig(model="m", warmup=0.1, cooldown=10)
