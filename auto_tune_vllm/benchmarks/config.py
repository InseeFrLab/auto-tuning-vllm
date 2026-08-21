"""Benchmark configuration."""

from dataclasses import dataclass
from typing import Literal, Optional

CONCURRENT_DEFAULT_RATE = 50
REPLAY_DEFAULT_RATE = 1.0

# OpenAI-compatible endpoints GuideLLM can target. Restricted to generative text
# endpoints: the results parser expects TTFT/ITL/output-token metrics, which
# non-generative endpoints (embeddings, pooling) do not produce.
SUPPORTED_REQUEST_FORMATS = (
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/responses",
)


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
    rate: Optional[float] = None  # Concurrent streams or replay time_scale fallback
    samples: int = 1000  # Number of samples to take

    # OpenAI endpoint targeted by the benchmark. None => GuideLLM default
    # (/v1/chat/completions); left unset the emitted command is unchanged.
    request_format: Optional[str] = None

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

    # GuideLLM benchmark profile (concurrent or replay). Omitted => concurrent.
    profile: Optional[dict] = None

    # Optional kernel prewarm before trace replay (guidellm_trace_replay only).
    prewarm: Optional[dict] = None

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
        self._normalize_benchmark_profile()
        self._apply_default_rate()

        if self.rate <= 0:
            raise ValueError(f"benchmark.rate must be greater than 0; got {self.rate}")

        self._validate_prewarm()
        self._validate_request_format()
        if self.profile is not None:
            from .profiles import profile_from_dict

            profile = profile_from_dict(self.profile)
            profile.validate(self)

    def _validate_prewarm(self) -> None:
        """Validate optional prewarm settings for trace replay benchmarks."""
        if self.prewarm is None:
            return

        if self.benchmark_type != "guidellm_trace_replay":
            raise ValueError(
                "benchmark.prewarm is only supported with "
                "benchmark_type='guidellm_trace_replay'"
            )

        allowed_keys = frozenset({"duration", "concurrency"})
        unknown_keys = set(self.prewarm.keys()) - allowed_keys
        if unknown_keys:
            raise ValueError(
                "benchmark.prewarm supports only duration and concurrency; "
                f"got unexpected keys: {sorted(unknown_keys)}"
            )

        for key in ("duration", "concurrency"):
            if key not in self.prewarm:
                raise ValueError(f"benchmark.prewarm.{key} is required")
            value = self.prewarm[key]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"benchmark.prewarm.{key} must be a number; got {value!r}"
                )
            if value <= 0:
                raise ValueError(
                    f"benchmark.prewarm.{key} must be greater than 0; got {value}"
                )

    def _validate_request_format(self) -> None:
        """Reject unsupported endpoints before a GPU trial is started."""
        if self.request_format is None:
            return
        if self.request_format not in SUPPORTED_REQUEST_FORMATS:
            raise ValueError(
                f"benchmark.request_format={self.request_format!r} is not supported; "
                f"expected one of {list(SUPPORTED_REQUEST_FORMATS)}"
            )

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

    def _apply_default_rate(self) -> None:
        """Apply profile-specific defaults when ``rate`` is omitted."""
        if self.rate is not None:
            return
        if self.benchmark_type == "guidellm_trace_replay":
            self.rate = REPLAY_DEFAULT_RATE
        else:
            self.rate = CONCURRENT_DEFAULT_RATE

    @property
    def use_synthetic_data(self) -> bool:
        """Whether to use synthetic data instead of a dataset."""
        return self.dataset is None
