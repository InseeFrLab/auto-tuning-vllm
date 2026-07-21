"""Speculative decoding configuration for vLLM synthetic benchmarking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import optuna

from .parameters import ParameterConfig

VALID_METHODS = frozenset({"mtp", "eagle", "eagle3", "dflash"})
STATIC_SPEC_KEYS = frozenset({"draft_tensor_parallel_size", "max_model_len"})
MIN_VLLM_VERSION = (0, 20)


def parse_vllm_version(version: str) -> tuple[int, ...]:
    """Parse a vLLM version string into a comparable tuple of integers."""
    cleaned = version.strip()
    match = re.match(r"^(\d+(?:\.\d+)*)", cleaned)
    if not match:
        raise ValueError(f"Invalid vLLM version string: {version!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def vllm_version_at_least(version: str, minimum: tuple[int, ...]) -> bool:
    """Return True when ``version`` is greater than or equal to ``minimum``."""
    parsed = parse_vllm_version(version)
    min_len = max(len(parsed), len(minimum))
    padded = parsed + (0,) * (min_len - len(parsed))
    min_padded = minimum + (0,) * (min_len - len(minimum))
    return padded >= min_padded


@dataclass
class SpeculativeMethod:
    """Maps a speculation method to its draft/auxiliary model identifier."""

    method: str
    model: str

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"Invalid speculative method {self.method!r}. "
                f"Valid options: {sorted(VALID_METHODS)}"
            )
        if not self.model:
            raise ValueError(
                f"speculative_decoding.methods entry for {self.method!r} "
                "requires a non-empty model"
            )


@dataclass
class SpeculativeDecodingConfig:
    """Study-level speculative decoding search space."""

    enabled: bool = False
    allow_disabled: bool = True
    methods: list[SpeculativeMethod] = field(default_factory=list)
    synthetic_acceptance_rates: Optional[list[float]] = None
    synthetic_acceptance_length: Optional[int] = None
    num_speculative_tokens: Optional[int] = None
    static_parameters: dict[str, int] = field(default_factory=dict)
    draft_tensor_parallel_size: Optional[ParameterConfig] = None
    max_model_len: Optional[ParameterConfig] = None

    def __post_init__(self) -> None:
        for key, value in self.static_parameters.items():
            if key not in STATIC_SPEC_KEYS:
                raise ValueError(
                    f"Invalid speculative_decoding.static_parameters key {key!r}. "
                    f"Valid options: {sorted(STATIC_SPEC_KEYS)}"
                )
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"speculative_decoding.static_parameters.{key} must be a "
                    f"positive integer; got {value!r}"
                )

        if (
            "draft_tensor_parallel_size" in self.static_parameters
            and self.draft_tensor_parallel_size is not None
            and self.draft_tensor_parallel_size.enabled
        ):
            raise ValueError(
                "draft_tensor_parallel_size cannot be both fixed in "
                "speculative_decoding.static_parameters and enabled for tuning"
            )
        if (
            "max_model_len" in self.static_parameters
            and self.max_model_len is not None
            and self.max_model_len.enabled
        ):
            raise ValueError(
                "max_model_len cannot be both fixed in "
                "speculative_decoding.static_parameters and enabled for tuning"
            )

        if not self.enabled:
            return

        if not self.methods:
            raise ValueError(
                "speculative_decoding.methods must contain at least one entry "
                "when enabled"
            )

        has_rates = self.synthetic_acceptance_rates is not None
        has_length = self.synthetic_acceptance_length is not None
        if has_rates == has_length:
            raise ValueError(
                "speculative_decoding requires exactly one of "
                "synthetic_acceptance_rates or synthetic_acceptance_length"
            )

        if has_rates:
            if not self.synthetic_acceptance_rates:
                raise ValueError(
                    "speculative_decoding.synthetic_acceptance_rates must be non-empty"
                )
            for rate in self.synthetic_acceptance_rates:
                if not isinstance(rate, (int, float)) or not 0 <= float(rate) <= 1:
                    raise ValueError(
                        "speculative_decoding.synthetic_acceptance_rates entries "
                        f"must be floats in [0, 1]; got {rate!r}"
                    )
        else:
            length = self.synthetic_acceptance_length
            if not isinstance(length, int) or length <= 0:
                raise ValueError(
                    "speculative_decoding.synthetic_acceptance_length must be "
                    f"a positive integer; got {length!r}"
                )
            if self.num_speculative_tokens is None:
                raise ValueError(
                    "speculative_decoding.num_speculative_tokens is required when "
                    "using synthetic_acceptance_length"
                )
            if self.num_speculative_tokens <= 0:
                raise ValueError(
                    "speculative_decoding.num_speculative_tokens must be > 0; "
                    f"got {self.num_speculative_tokens}"
                )

    def suggest(self, trial: optuna.Trial) -> str | None:
        """Register Optuna sub-params and return the speculative-config JSON."""
        if self.allow_disabled:
            enabled = trial.suggest_categorical("spec_enabled", [True, False])
            if not enabled:
                trial.set_user_attr("speculative_config", "disabled")
                return None

        method_names = [entry.method for entry in self.methods]
        chosen_method = trial.suggest_categorical("spec_method", method_names)
        model = next(
            entry.model for entry in self.methods if entry.method == chosen_method
        )

        if self.synthetic_acceptance_rates is not None:
            rates = self.synthetic_acceptance_rates
            k = trial.suggest_int("spec_num_speculative_tokens", low=1, high=len(rates))
            acceptance_rates = rates[:k]
            spec_dict: dict[str, Any] = {
                "method": chosen_method,
                "model": model,
                "num_speculative_tokens": k,
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_rates": acceptance_rates,
            }
        else:
            k = self.num_speculative_tokens
            spec_dict = {
                "method": chosen_method,
                "model": model,
                "num_speculative_tokens": k,
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_length": self.synthetic_acceptance_length,
            }

        spec_dict.update(self.static_parameters)

        if (
            "draft_tensor_parallel_size" not in spec_dict
            and self.draft_tensor_parallel_size is not None
            and self.draft_tensor_parallel_size.enabled
        ):
            spec_dict["draft_tensor_parallel_size"] = (
                self.draft_tensor_parallel_size.generate_optuna_suggest(trial)
            )

        if (
            "max_model_len" not in spec_dict
            and self.max_model_len is not None
            and self.max_model_len.enabled
        ):
            spec_dict["max_model_len"] = self.max_model_len.generate_optuna_suggest(
                trial
            )

        json_str = json.dumps(spec_dict, separators=(",", ":"))
        trial.set_user_attr("speculative_config", json_str)
        return json_str
