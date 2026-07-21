"""GuideLLM benchmark profile abstractions for CLI command construction."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .config import BenchmarkConfig

TRACE_FORMATS = frozenset({"trace_synthetic", "mooncake"})


class BenchmarkProfile(ABC):
    """Abstract benchmark profile that renders GuideLLM ``--profile`` CLI args."""

    @abstractmethod
    def render_cli_profile(self, config: BenchmarkConfig) -> str:
        """Return the comma-separated ``--profile`` value for GuideLLM."""

    def validate(self, config: BenchmarkConfig) -> None:
        """Validate profile-specific constraints against ``config``."""


@dataclass
class ConcurrentProfile(BenchmarkProfile):
    """Concurrent request load profile (GuideLLM ``kind=concurrent``)."""

    def render_cli_profile(self, config: BenchmarkConfig) -> str:
        profile_parts = ["kind=concurrent", f"streams={config.rate}"]
        if config.warmup is not None:
            profile_parts.append(f"warmup={config.warmup}")
        if config.cooldown is not None:
            profile_parts.append(f"cooldown={config.cooldown}")
        if config.rampup is not None:
            profile_parts.append(f"rampup_duration={config.rampup}")
        return ",".join(profile_parts)


@dataclass
class ReplayProfile(BenchmarkProfile):
    """Trace replay profile (GuideLLM ``kind=replay``).

    ``time_scale`` scales inter-arrival intervals from the trace file.
    When ``time_scale`` is omitted, ``config.rate`` is used as the scale factor.
    """

    trace_format: Literal["trace_synthetic", "mooncake"] = "trace_synthetic"
    time_scale: float | None = None
    data_samples: int | None = None
    timestamp_column: str = "timestamp"
    prompt_tokens_column: str = "input_length"
    output_tokens_column: str = "output_length"
    hash_ids_column: str = "hash_ids"
    hash_id_block_size: int = 512

    def effective_time_scale(self, config: BenchmarkConfig) -> float:
        """Return ``time_scale`` or fall back to ``config.rate``."""
        return self.time_scale if self.time_scale is not None else float(config.rate)

    def render_cli_profile(self, config: BenchmarkConfig) -> str:
        scale = self.effective_time_scale(config)
        return f"kind=replay,time_scale={scale}"

    def validate(self, config: BenchmarkConfig) -> None:
        if config.dataset is None:
            raise ValueError(
                "benchmark profile 'replay' requires a trace dataset; "
                "set benchmark.dataset to a JSONL file path"
            )
        scale = self.effective_time_scale(config)
        if scale <= 0:
            raise ValueError(
                f"benchmark replay time_scale must be greater than 0; got {scale}"
            )
        if self.trace_format not in TRACE_FORMATS:
            raise ValueError(
                f"Unsupported trace_format {self.trace_format!r}; "
                f"expected one of {sorted(TRACE_FORMATS)}"
            )
        for name in ("warmup", "cooldown", "rampup"):
            if getattr(config, name) is not None:
                raise ValueError(
                    f"benchmark.{name} is not supported with profile kind='replay'"
                )
        if self.data_samples is not None and self.data_samples <= 0:
            raise ValueError(
                f"benchmark profile data_samples must be greater than 0; "
                f"got {self.data_samples}"
            )


def profile_from_dict(data: dict[str, Any] | None) -> BenchmarkProfile:
    """Build a profile instance from a YAML ``benchmark.profile`` mapping."""
    if data is None:
        return ConcurrentProfile()

    kind = data.get("kind", "concurrent")
    if kind == "concurrent":
        return ConcurrentProfile()
    if kind == "replay":
        replay_fields = {
            key: data[key]
            for key in (
                "trace_format",
                "time_scale",
                "data_samples",
                "timestamp_column",
                "prompt_tokens_column",
                "output_tokens_column",
                "hash_ids_column",
                "hash_id_block_size",
            )
            if key in data
        }
        return ReplayProfile(**replay_fields)

    raise ValueError(
        f"Unsupported benchmark profile kind {kind!r}; "
        "expected 'concurrent' or 'replay'"
    )


def render_replay_data_cli(
    config: BenchmarkConfig, profile: ReplayProfile
) -> list[str]:
    """Build ``--data`` and optional ``--data-loader`` args for trace replay."""
    if config.dataset is None:
        raise ValueError("Trace replay requires benchmark.dataset to be set")

    if not config.dataset.startswith("hf://") and not os.path.exists(config.dataset):
        raise FileNotFoundError(f"Trace dataset file not found: {config.dataset}")

    data_entry: dict[str, Any] = {
        "kind": profile.trace_format,
        "path": config.dataset,
        "timestamp_column": profile.timestamp_column,
        "prompt_tokens_column": profile.prompt_tokens_column,
        "output_tokens_column": profile.output_tokens_column,
    }
    if profile.trace_format == "mooncake":
        data_entry["hash_ids_column"] = profile.hash_ids_column
        data_entry["hash_id_block_size"] = profile.hash_id_block_size

    cmd = ["--data", json.dumps(data_entry)]
    if profile.data_samples is not None:
        cmd.extend(
            [
                "--data-loader",
                f"kind=pytorch,samples={profile.data_samples}",
            ]
        )
    return cmd


def _read_trace_dataframe(dataset_path: str) -> pd.DataFrame:
    """Load a local trace file into a pandas DataFrame."""
    path = Path(dataset_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(dataset_path, lines=True)
    if suffix == ".json":
        return pd.read_json(dataset_path)
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    raise ValueError(
        f"Unsupported trace dataset format {suffix!r}; "
        "expected .jsonl, .json, .csv, or .parquet"
    )


def default_prewarm_token_stats(config: BenchmarkConfig) -> dict[str, float | int]:
    """Fallback token statistics for prewarm synthetic data from benchmark defaults."""
    prompt_stdev = (
        float(config.prompt_tokens_stdev)
        if config.prompt_tokens_stdev is not None
        else 1.0
    )
    output_stdev = (
        float(config.output_tokens_stdev)
        if config.output_tokens_stdev is not None
        else 1.0
    )
    return {
        "prompt_mean": max(1, config.prompt_tokens),
        "prompt_stdev": max(1.0, prompt_stdev),
        "output_mean": max(1, config.output_tokens),
        "output_stdev": max(1.0, output_stdev),
    }


def resolve_prewarm_token_stats(
    config: BenchmarkConfig,
    profile: ReplayProfile,
    logger: logging.Logger | None = None,
) -> dict[str, float | int]:
    """Derive prewarm token stats from a local trace file, or fall back to defaults."""
    dataset = config.dataset
    if dataset is None:
        raise ValueError("Trace replay requires benchmark.dataset to be set")

    if dataset.startswith("hf://"):
        if logger is not None:
            logger.warning(
                "Cannot derive prewarm token stats from HuggingFace dataset %r; "
                "using benchmark.prompt_tokens / benchmark.output_tokens defaults",
                dataset,
            )
        return default_prewarm_token_stats(config)

    if not os.path.exists(dataset):
        if logger is not None:
            logger.warning(
                "Trace dataset file not found at %r; using benchmark.prompt_tokens / "
                "benchmark.output_tokens defaults for prewarm",
                dataset,
            )
        return default_prewarm_token_stats(config)

    try:
        return compute_trace_token_stats(
            dataset,
            profile.prompt_tokens_column,
            profile.output_tokens_column,
        )
    except (ValueError, FileNotFoundError) as exc:
        if logger is not None:
            logger.warning(
                "Failed to derive prewarm token stats from trace file: %s; "
                "using benchmark.prompt_tokens / benchmark.output_tokens defaults",
                exc,
            )
        return default_prewarm_token_stats(config)


def compute_trace_token_stats(
    dataset_path: str,
    prompt_col: str,
    output_col: str,
) -> dict[str, float | int]:
    """Compute mean and stdev of prompt/output token lengths from a trace file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Trace dataset file not found: {dataset_path}")

    df = _read_trace_dataframe(dataset_path)
    for col in (prompt_col, output_col):
        if col not in df.columns:
            raise ValueError(
                f"Trace dataset missing required column {col!r}; "
                f"available columns: {list(df.columns)}"
            )

    prompt_series = df[prompt_col]
    output_series = df[output_col]

    prompt_mean = max(1, int(round(prompt_series.mean())))
    output_mean = max(1, int(round(output_series.mean())))
    prompt_stdev = max(1.0, float(prompt_series.std(ddof=0) or 0.0))
    output_stdev = max(1.0, float(output_series.std(ddof=0) or 0.0))

    return {
        "prompt_mean": prompt_mean,
        "prompt_stdev": prompt_stdev,
        "output_mean": output_mean,
        "output_stdev": output_stdev,
    }


def render_prewarm_args(
    config: BenchmarkConfig,
    prewarm: dict[str, Any],
    stats: dict[str, float | int],
    results_file: str,
) -> list[str]:
    """Build GuideLLM CLI args for a concurrent prewarm run before trace replay."""
    duration = prewarm["duration"]
    concurrency = prewarm["concurrency"]
    data_config: dict[str, Any] = {
        "kind": "synthetic_text",
        "prompt_tokens": stats["prompt_mean"],
        "output_tokens": stats["output_mean"],
        "prompt_tokens_stdev": stats["prompt_stdev"],
        "output_tokens_stdev": stats["output_stdev"],
    }

    return [
        "--profile",
        f"kind=concurrent,streams={concurrency}",
        "--constraint",
        f"kind=max_duration,seconds={duration}",
        "--metrics",
        f"kind=generative,sample_size={config.sample_requests}",
        "--output",
        f"kind=json,path={results_file}",
        "--data",
        json.dumps(data_config),
        "--data-loader",
        f"kind=pytorch,samples={config.samples}",
    ]
