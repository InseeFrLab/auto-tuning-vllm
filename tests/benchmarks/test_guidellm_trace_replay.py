"""Unit tests for trace replay GuideLLM benchmark path."""

from __future__ import annotations

import json
import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.benchmarks.profiles import (
    ReplayProfile,
    compute_trace_token_stats,
    default_prewarm_token_stats,
    profile_from_dict,
    render_prewarm_args,
    render_replay_data_cli,
    resolve_prewarm_token_stats,
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


def _backend_value(cmd: list[str]) -> str:
    return cmd[cmd.index("--backend") + 1]


def test_trace_replay_default_rate_when_omitted():
    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset=_TRACE_DATASET,
    )
    assert config.rate == 1.0


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


def test_trace_replay_command_includes_request_format_when_set():
    cmd = _build_cmd(request_format="/v1/completions")
    assert "request_format=/v1/completions" in _backend_value(cmd)


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


def test_prewarm_rejected_on_non_trace_replay_benchmark():
    with pytest.raises(ValueError, match="benchmark.prewarm is only supported"):
        BenchmarkConfig(
            model="test-model",
            prewarm={"duration": 30, "concurrency": 4},
        )


def test_prewarm_requires_duration_and_concurrency():
    with pytest.raises(ValueError, match="benchmark.prewarm.duration is required"):
        BenchmarkConfig(
            benchmark_type="guidellm_trace_replay",
            model="test-model",
            dataset=_TRACE_DATASET,
            prewarm={"concurrency": 4},
        )


def test_compute_trace_token_stats_from_sample_trace():
    stats = compute_trace_token_stats(_TRACE_DATASET, "input_length", "output_length")

    assert stats["prompt_mean"] >= 1
    assert stats["output_mean"] >= 1
    assert stats["prompt_stdev"] >= 1
    assert stats["output_stdev"] >= 1
    assert isinstance(stats["prompt_stdev"], int)
    assert isinstance(stats["output_stdev"], int)


def test_resolve_prewarm_token_stats_uses_trace_file():
    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset=_TRACE_DATASET,
    )
    profile = ReplayProfile()

    stats = resolve_prewarm_token_stats(config, profile)

    assert stats == compute_trace_token_stats(
        _TRACE_DATASET, profile.prompt_tokens_column, profile.output_tokens_column
    )


def test_resolve_prewarm_token_stats_hf_dataset_uses_defaults(caplog):
    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset="hf://demo/trace",
        prompt_tokens=512,
        output_tokens=128,
    )
    profile = ReplayProfile()
    logger = logging.getLogger("test.prewarm.hf")

    with caplog.at_level(logging.WARNING, logger="test.prewarm.hf"):
        stats = resolve_prewarm_token_stats(config, profile, logger)

    assert stats == default_prewarm_token_stats(config)
    assert "HuggingFace dataset" in caplog.text


def test_resolve_prewarm_token_stats_missing_file_uses_defaults(caplog):
    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset="/tmp/nonexistent_trace.jsonl",
        prompt_tokens=256,
        output_tokens=64,
    )
    profile = ReplayProfile()
    logger = logging.getLogger("test.prewarm.missing")

    with caplog.at_level(logging.WARNING, logger="test.prewarm.missing"):
        stats = resolve_prewarm_token_stats(config, profile, logger)

    assert stats == default_prewarm_token_stats(config)
    assert "not found" in caplog.text


def test_render_prewarm_args_uses_token_stats():
    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset=_TRACE_DATASET,
        samples=50,
        sample_requests=0,
    )
    stats = {
        "prompt_mean": 400,
        "prompt_stdev": 120,
        "output_mean": 80,
        "output_stdev": 30,
    }
    args = render_prewarm_args(
        config, {"duration": 15, "concurrency": 2}, stats, "/tmp/prewarm.json"
    )

    assert args[args.index("--profile") + 1] == "kind=concurrent,streams=2"
    assert args[args.index("--constraint") + 1] == "kind=max_duration,seconds=15"
    data = json.loads(args[args.index("--data") + 1])
    assert data["prompt_tokens"] == 400
    assert data["output_tokens"] == 80
    assert data["prompt_tokens_stdev"] == 120
    assert data["output_tokens_stdev"] == 30
    assert isinstance(data["prompt_tokens_stdev"], int)
    assert isinstance(data["output_tokens_stdev"], int)


def test_run_prewarm_failure_aborts_trial():
    benchmark = GuideLLMTraceReplayBenchmark()
    benchmark._logger = logging.getLogger("test.prewarm.fail")
    benchmark._results_file = "/tmp/trial_benchmark_results.json"

    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset=_TRACE_DATASET,
        prewarm={"duration": 10, "concurrency": 2},
    )

    mock_process = MagicMock()
    mock_process.wait.return_value = 1

    with patch.object(
        GuideLLMTraceReplayBenchmark,
        "_launch_subprocess",
        return_value=mock_process,
    ):
        with pytest.raises(RuntimeError, match="Prewarm failed with exit code 1"):
            benchmark._run_prewarm("http://localhost:8000/v1", config)


def test_run_prewarm_timeout_aborts_trial():
    benchmark = GuideLLMTraceReplayBenchmark()
    benchmark._logger = logging.getLogger("test.prewarm.timeout")
    benchmark._results_file = "/tmp/trial_benchmark_results.json"

    config = BenchmarkConfig(
        benchmark_type="guidellm_trace_replay",
        model="test-model",
        dataset=_TRACE_DATASET,
        prewarm={"duration": 10, "concurrency": 2},
    )

    mock_process = MagicMock()
    mock_process.wait.side_effect = subprocess.TimeoutExpired(
        cmd=["guidellm"], timeout=50
    )

    with patch.object(
        GuideLLMTraceReplayBenchmark,
        "_launch_subprocess",
        return_value=mock_process,
    ):
        with patch.object(GuideLLMTraceReplayBenchmark, "terminate_benchmark"):
            with pytest.raises(RuntimeError, match="Prewarm did not finish"):
                benchmark._run_prewarm("http://localhost:8000/v1", config)
