"""Speculative decoding configuration for vLLM synthetic benchmarking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import optuna

from .parameters import ListParameter, ParameterConfig

VALID_METHODS = frozenset({"mtp", "eagle", "eagle3", "dflash", "qwen3_next_mtp"})
# Native MTP methods where the draft head lives in the target; `model` is optional
# in study YAML and omitted from --speculative-config when unset.
OPTIONAL_MODEL_METHODS = frozenset({"qwen3_next_mtp"})
STATIC_SPEC_KEYS = frozenset(
    {"draft_tensor_parallel_size", "max_model_len", "num_speculative_tokens"}
)
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
        if not self.model and self.method not in OPTIONAL_MODEL_METHODS:
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
    num_speculative_tokens: Optional[ParameterConfig] = None
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

        self._reject_static_and_tunable_conflict(
            "draft_tensor_parallel_size", self.draft_tensor_parallel_size
        )
        self._reject_static_and_tunable_conflict("max_model_len", self.max_model_len)
        self._reject_static_and_tunable_conflict(
            "num_speculative_tokens", self.num_speculative_tokens
        )

        if not self.enabled:
            return

        if not self.methods:
            raise ValueError(
                "speculative_decoding.methods must contain at least one entry "
                "when enabled"
            )

        method_names = [entry.method for entry in self.methods]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in method_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                "speculative_decoding.methods contains duplicate method values: "
                f"{sorted(duplicates)}. Each method may appear only once."
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
            max_k = len(self.synthetic_acceptance_rates)
            static_k = self.static_parameters.get("num_speculative_tokens")
            if static_k is not None:
                self._validate_k(static_k, max_k)
            if (
                self.num_speculative_tokens is not None
                and self.num_speculative_tokens.enabled
            ):
                self._validate_num_speculative_tokens_param(max_k)
        else:
            length = self.synthetic_acceptance_length
            if not isinstance(length, int) or length <= 0:
                raise ValueError(
                    "speculative_decoding.synthetic_acceptance_length must be "
                    f"a positive integer; got {length!r}"
                )
            if (
                self.num_speculative_tokens is not None
                and self.num_speculative_tokens.enabled
            ):
                raise ValueError(
                    "speculative_decoding.num_speculative_tokens cannot be enabled "
                    "when using synthetic_acceptance_length"
                )
            static_k = self.static_parameters.get("num_speculative_tokens")
            if static_k is None:
                raise ValueError(
                    "speculative_decoding.static_parameters.num_speculative_tokens "
                    "is required when using synthetic_acceptance_length"
                )
            self._validate_k(static_k, length)

    def _reject_static_and_tunable_conflict(
        self, key: str, param: ParameterConfig | None
    ) -> None:
        if key in self.static_parameters and param is not None and param.enabled:
            raise ValueError(
                f"{key} cannot be both fixed in "
                f"speculative_decoding.static_parameters and enabled for tuning"
            )

    def _validate_num_speculative_tokens_param(self, max_k: int) -> None:
        param = self.num_speculative_tokens
        if param is None or not param.enabled:
            return
        if not isinstance(param, ListParameter):
            raise ValueError(
                "speculative_decoding.num_speculative_tokens only supports "
                "list options (enabled: true with options: [...])"
            )
        if not param.options:
            raise ValueError(
                "speculative_decoding.num_speculative_tokens.options must be "
                "non-empty when enabled"
            )
        seen_k: set[int] = set()
        for option in param.options:
            if not isinstance(option, int):
                raise ValueError(
                    "speculative_decoding.num_speculative_tokens.options entries "
                    f"must be integers; got {option!r}"
                )
            self._validate_k(option, max_k)
            if option in seen_k:
                raise ValueError(
                    "speculative_decoding.num_speculative_tokens.options contains "
                    f"duplicate values: {option}"
                )
            seen_k.add(option)

    @staticmethod
    def _validate_k(k: int, max_k: int) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                "speculative_decoding num_speculative_tokens values must be "
                f"positive integers; got {k!r}"
            )
        if k > max_k:
            raise ValueError(
                f"speculative_decoding num_speculative_tokens value {k} exceeds "
                f"the available synthetic acceptance slots ({max_k})"
            )

    def _resolve_num_speculative_tokens(self, trial: optuna.Trial, max_k: int) -> int:
        """Resolve k for synthetic_acceptance_rates mode."""
        static_k = self.static_parameters.get("num_speculative_tokens")
        if static_k is not None:
            return static_k
        if (
            self.num_speculative_tokens is not None
            and self.num_speculative_tokens.enabled
        ):
            return int(self.num_speculative_tokens.generate_optuna_suggest(trial))
        return max_k

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
            k = self._resolve_num_speculative_tokens(trial, len(rates))
            acceptance_rates = rates[:k]
            spec_dict: dict[str, Any] = {
                "method": chosen_method,
                "num_speculative_tokens": k,
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_rates": acceptance_rates,
            }
            if model:
                spec_dict["model"] = model
        else:
            k = self.static_parameters["num_speculative_tokens"]
            spec_dict = {
                "method": chosen_method,
                "num_speculative_tokens": k,
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_length": self.synthetic_acceptance_length,
            }
            if model:
                spec_dict["model"] = model

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
