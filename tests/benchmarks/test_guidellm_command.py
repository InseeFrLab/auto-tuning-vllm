"""Unit tests for GuideLLM CLI command construction."""

from __future__ import annotations

import json

import pytest

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.benchmarks.providers import GuideLLMBenchmark


def _build_cmd(**kwargs) -> list[str]:
    config = BenchmarkConfig(model="test-model", **kwargs)
    return GuideLLMBenchmark()._build_guidellm_command(
        "http://localhost:8000/v1", config, "/tmp/results.json"
    )


def _profile_value(cmd: list[str]) -> str:
    return cmd[cmd.index("--profile") + 1]


def test_command_uses_guidellm_run():
    cmd = _build_cmd()
    assert cmd[0:2] == ["guidellm", "run"]
    assert "--backend" in cmd
    assert "kind=openai_http" in cmd[cmd.index("--backend") + 1]
    assert "--constraint" in cmd
    assert cmd[cmd.index("--constraint") + 1] == "kind=max_duration,seconds=300"
    assert "--output" in cmd
    assert cmd[cmd.index("--output") + 1] == "kind=json,path=/tmp/results.json"
    assert "--metrics" in cmd
    assert cmd[cmd.index("--metrics") + 1] == "kind=generative,sample_size=0"


def test_warmup_and_cooldown_included_in_profile_when_set():
    cmd = _build_cmd(warmup=0.1, cooldown=0.1)
    profile = _profile_value(cmd)
    assert "kind=concurrent" in profile
    assert "streams=50" in profile
    assert "warmup=0.1" in profile
    assert "cooldown=0.1" in profile


def test_warmup_and_cooldown_omitted_from_profile_when_unset():
    cmd = _build_cmd()
    profile = _profile_value(cmd)
    assert "warmup=" not in profile
    assert "cooldown=" not in profile


def test_absolute_warmup_value_passed_through_in_profile():
    cmd = _build_cmd(warmup=30, cooldown=None)
    profile = _profile_value(cmd)
    assert "warmup=30" in profile
    assert "cooldown=" not in profile


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


def test_rampup_included_in_profile_when_set():
    cmd = _build_cmd(rampup=10)
    profile = _profile_value(cmd)
    assert "rampup_duration=10" in profile


def test_rampup_omitted_from_profile_when_unset():
    cmd = _build_cmd()
    profile = _profile_value(cmd)
    assert "rampup_duration=" not in profile


def test_rampup_zero_rejected():
    with pytest.raises(ValueError, match="rampup"):
        BenchmarkConfig(model="m", rampup=0)


def test_negative_rampup_rejected():
    with pytest.raises(ValueError, match="rampup"):
        BenchmarkConfig(model="m", rampup=-1)


def test_synthetic_data_uses_synthetic_text_and_data_loader():
    cmd = _build_cmd(dataset=None)
    data = json.loads(cmd[cmd.index("--data") + 1])
    assert data["kind"] == "synthetic_text"
    assert data["prompt_tokens"] == 1000
    assert data["output_tokens"] == 1000
    assert "--data-loader" in cmd
    assert cmd[cmd.index("--data-loader") + 1] == "kind=pytorch,samples=1000"
