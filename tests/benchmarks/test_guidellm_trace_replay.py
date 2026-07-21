"""Unit tests for trace replay GuideLLM benchmark path."""

from __future__ import annotations

import json

import pytest

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.benchmarks.profiles import (
    ReplayProfile,
    profile_from_dict,
    render_replay_data_cli,
)
from auto_tune_vllm.benchmarks.trace_replay import GuideLLMTraceReplayBenchmark

_TRACE_DATASET = "examples/trace_replay/sample.jsonl"
_REPLAY_PROFILE = {"trace_format": "trace_synthetic"}


def _build_cmd(**kwargs) -> list[str]:
    defaults = {
        "benchmark_type": "guidellm_trace_replay",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": _TRACE_DATASET,
    }
    defaults.update(kwargs)
    config = BenchmarkConfig(**defaults)
    return GuideLLMTraceReplayBenchmark()._build_guidellm_command(
        "http://localhost:8000/v1", config, "/tmp/results.json"
    )


def _profile_value(cmd: list[str]) -> str:
    return cmd[cmd.index("--profile") + 1]


def test_trace_replay_injects_profile_kind_from_benchmark_type():
    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset=_TRACE_DATASET,
    )
    assert config.profile == {"kind": "replay"}


def test_trace_replay_rejects_conflicting_profile_kind_at_config_load():
    with pytest.raises(ValueError, match="incompatible with benchmark.profile.kind"):
        BenchmarkConfig(
            benchmark_type="guidellm_trace_replay",
            model="test-model",
            dataset=_TRACE_DATASET,
            profile={"kind": "concurrent"},
        )


def test_trace_replay_rejects_replay_profile_on_guidellm_at_config_load():
    with pytest.raises(
        ValueError, match="requires benchmark_type='guidellm_trace_replay'"
    ):
        BenchmarkConfig(
            model="test-model",
            dataset=_TRACE_DATASET,
            profile={"kind": "replay"},
        )


def test_trace_replay_requires_dataset_at_config_load():
    with pytest.raises(ValueError, match="requires a trace dataset"):
        BenchmarkConfig(
            benchmark_type="guidellm_trace_replay",
            model="test-model",
        )


def test_trace_replay_rejects_non_positive_rate_at_config_load():
    with pytest.raises(ValueError, match="benchmark.rate must be greater than 0"):
        BenchmarkConfig(
            benchmark_type="guidellm_trace_replay",
            model="test-model",
            dataset=_TRACE_DATASET,
            rate=0,
        )


def test_trace_replay_accepts_fractional_rate_at_config_load():
    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset=_TRACE_DATASET,
        rate=0.5,
    )
    assert config.rate == 0.5


def test_trace_replay_rejects_warmup_at_config_load():
    with pytest.raises(ValueError, match="warmup.*not supported"):
        BenchmarkConfig(
            benchmark_type="guidellm_trace_replay",
            model="test-model",
            dataset=_TRACE_DATASET,
            warmup=0.1,
        )


def test_trace_replay_command_without_explicit_profile_section():
    cmd = _build_cmd(rate=2)

    profile = _profile_value(cmd)
    assert profile == "kind=replay,time_scale=2.0"

    data = json.loads(cmd[cmd.index("--data") + 1])
    assert data == {
        "kind": "trace_synthetic",
        "path": _TRACE_DATASET,
        "timestamp_column": "timestamp",
        "prompt_tokens_column": "input_length",
        "output_tokens_column": "output_length",
    }


def test_trace_replay_command_uses_trace_format_from_profile():
    cmd = _build_cmd(profile={"trace_format": "trace_synthetic"}, rate=1)

    data = json.loads(cmd[cmd.index("--data") + 1])
    assert data["kind"] == "trace_synthetic"


def test_trace_replay_fractional_rate_used_as_time_scale():
    cmd = _build_cmd(rate=0.5)

    assert _profile_value(cmd) == "kind=replay,time_scale=0.5"


def test_trace_replay_profile_time_scale_overrides_rate():
    cmd = _build_cmd(rate=2, profile={"time_scale": 0.5})

    assert _profile_value(cmd) == "kind=replay,time_scale=0.5"


def test_trace_replay_mooncake_data_includes_hash_fields():
    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset=_TRACE_DATASET,
        profile={
            "trace_format": "mooncake",
            "hash_ids_column": "block_hashes",
            "hash_id_block_size": 256,
        },
    )
    profile = profile_from_dict(config.profile)
    assert isinstance(profile, ReplayProfile)
    data_args = render_replay_data_cli(config, profile)
    data = json.loads(data_args[data_args.index("--data") + 1])

    assert data["kind"] == "mooncake"
    assert data["hash_ids_column"] == "block_hashes"
    assert data["hash_id_block_size"] == 256


def test_trace_replay_data_loader_when_data_samples_set():
    cmd = _build_cmd(profile={"data_samples": 100})

    assert "--data-loader" in cmd
    assert cmd[cmd.index("--data-loader") + 1] == "kind=pytorch,samples=100"
