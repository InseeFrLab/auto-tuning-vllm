"""Benchmark configuration."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    benchmark_type: str = "guidellm"  # concurrent-profile GuideLLM (see profiles.py)
    model: str = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
    max_seconds: int = 300
    dataset: Optional[str] = None  # HF dataset or file path
    prompt_tokens: int = 1000  # For synthetic data
    output_tokens: int = 1000  # For synthetic data
    concurrency: int = 50  # Benchmark concurrency level (legacy, use rates instead)

    # Advanced GuideLLM parameters
    processor: Optional[str] = None  # Processor model, defaults to model if not set
    rate: float = 3  # Concurrent streams (guidellm) or replay time_scale fallback
    samples: int = 1000  # Number of samples to take

    # Token statistics for synthetic data - only used when explicitly specified
    prompt_tokens_stdev: Optional[int] = None
    prompt_tokens_min: Optional[int] = None
    prompt_tokens_max: Optional[int] = None
    output_tokens_stdev: Optional[int] = None
    output_tokens_min: Optional[int] = None
    output_tokens_max: Optional[int] = None

    # GuideLLM warmup/cooldown (excluded from reported metrics). See GuideLLM docs:
    # values in (0, 1) are a fraction of run time/requests; values >= 1 are absolute.
    warmup: Optional[float] = None
    cooldown: Optional[float] = None

    # GuideLLM ramp-up duration in seconds (linear increase to target rate). Omit to disable.
    rampup: Optional[float] = None

    # Max detailed request samples stored in benchmark JSON output (GuideLLM sample_size).
    sample_requests: int = 0

    # Multimodal / advanced data pipeline (benchmark_type: guidellm_multimodal)
    data_args: Optional[dict] = None
    data_column_mapper: Optional[dict] = None
    data_preprocessors: Optional[list[str]] = None
    data_preprocessors_kwargs: Optional[dict] = None
    data_finalizer: Optional[str] = None
    request_format: Optional[str] = None

    # GuideLLM benchmark profile (concurrent or replay). Omitted => concurrent.
    profile: Optional[dict] = None

    # Set in benchmark section of study config
    # Logging level for GuideLLM
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    def __post_init__(self) -> None:
        for name, value in (
            ("warmup", self.warmup),
            ("cooldown", self.cooldown),
            ("rampup", self.rampup),
        ):
            if value is None:
                continue
            if value <= 0:
                raise ValueError(
                    f"benchmark.{name} must be greater than 0 or omitted; got {value}"
                )
        if (
            self.warmup is not None
            and self.cooldown is not None
            and 0 < self.warmup < 1
            and 0 < self.cooldown < 1
            and self.warmup + self.cooldown >= 1
        ):
            raise ValueError(
                "benchmark warmup and cooldown fractions must sum to less than 1 "
                f"(got warmup={self.warmup}, cooldown={self.cooldown})"
            )
        if self.sample_requests < 0:
            raise ValueError(
                f"benchmark.sample_requests must be >= 0; got {self.sample_requests}"
            )
        if self.rate <= 0:
            raise ValueError(f"benchmark.rate must be greater than 0; got {self.rate}")

        self._normalize_benchmark_profile()

        if self.profile is not None:
            from .profiles import profile_from_dict

            profile = profile_from_dict(self.profile)
            profile.validate(self)

    def _normalize_benchmark_profile(self) -> None:
        """Align ``benchmark.profile`` with ``benchmark_type`` (single source of truth)."""
        if self.benchmark_type == "guidellm_trace_replay":
            profile = dict(self.profile or {})
            kind = profile.get("kind", "replay")
            if kind != "replay":
                raise ValueError(
                    "benchmark_type 'guidellm_trace_replay' is incompatible with "
                    f"benchmark.profile.kind={kind!r}; omit profile.kind or set "
                    "it to 'replay'"
                )
            profile["kind"] = "replay"
            self.profile = profile
            return

        if self.profile and self.profile.get("kind") == "replay":
            raise ValueError(
                "benchmark.profile.kind='replay' requires "
                "benchmark_type='guidellm_trace_replay'"
            )

    @property
    def use_synthetic_data(self) -> bool:
        """Whether to use synthetic data instead of a dataset."""
        return self.dataset is None
