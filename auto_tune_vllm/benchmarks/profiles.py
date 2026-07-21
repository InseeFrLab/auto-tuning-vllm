"""GuideLLM benchmark profile abstractions for CLI command construction."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

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
