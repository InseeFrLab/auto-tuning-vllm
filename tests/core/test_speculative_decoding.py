"""Unit tests for speculative decoding configuration and Optuna integration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import optuna
import pytest
from optuna.trial import FixedTrial

from auto_tune_vllm.core.config import StudyConfig
from auto_tune_vllm.core.parameters import ListParameter
from auto_tune_vllm.core.speculative import (
    SpeculativeDecodingConfig,
    SpeculativeMethod,
    parse_vllm_version,
    vllm_version_at_least,
)
from auto_tune_vllm.core.study_controller import _validate_speculative_vllm_version
from auto_tune_vllm.utils.vllm_cli_parser import VLLMCLIParser


def _base_spec_config(**kwargs) -> SpeculativeDecodingConfig:
    """Minimal valid enabled config using synthetic_acceptance_rates."""
    defaults = {
        "enabled": True,
        "methods": [SpeculativeMethod(method="eagle3", model="org/draft")],
        "synthetic_acceptance_rates": [0.8, 0.7, 0.6],
    }
    defaults.update(kwargs)
    return SpeculativeDecodingConfig(**defaults)


class TestVllmVersionHelpers:
    def test_parse_vllm_version_strips_suffix(self):
        assert parse_vllm_version("0.20.1+cu124") == (0, 20, 1)

    def test_vllm_version_at_least(self):
        assert vllm_version_at_least("0.20.0", (0, 20))
        assert vllm_version_at_least("0.21.0", (0, 20))
        assert not vllm_version_at_least("0.19.9", (0, 20))

    def test_parse_vllm_version_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid vLLM version"):
            parse_vllm_version("not-a-version")


class TestSpeculativeDecodingConfigValidation:
    def test_enabled_requires_methods(self):
        with pytest.raises(ValueError, match="methods must contain at least one"):
            SpeculativeDecodingConfig(
                enabled=True,
                synthetic_acceptance_rates=[0.8],
            )

    def test_duplicate_methods_rejected(self):
        with pytest.raises(ValueError, match="duplicate method"):
            _base_spec_config(
                methods=[
                    SpeculativeMethod(method="eagle3", model="a"),
                    SpeculativeMethod(method="eagle3", model="b"),
                ],
            )

    def test_requires_exactly_one_synthetic_mode(self):
        with pytest.raises(ValueError, match="exactly one of"):
            _base_spec_config(synthetic_acceptance_length=4)

        with pytest.raises(ValueError, match="exactly one of"):
            SpeculativeDecodingConfig(
                enabled=True,
                methods=[SpeculativeMethod(method="eagle3", model="org/draft")],
            )

    def test_static_and_tunable_conflict(self):
        with pytest.raises(ValueError, match="cannot be both fixed"):
            _base_spec_config(
                static_parameters={"max_model_len": 8192},
                max_model_len=ListParameter(
                    name="spec_max_model_len",
                    enabled=True,
                    options=[4096, 8192],
                ),
            )

    def test_k_exceeds_rates_length(self):
        with pytest.raises(ValueError, match="exceeds"):
            _base_spec_config(
                static_parameters={"num_speculative_tokens": 5},
            )

    def test_acceptance_length_requires_static_k(self):
        with pytest.raises(ValueError, match="num_speculative_tokens.*required"):
            SpeculativeDecodingConfig(
                enabled=True,
                methods=[SpeculativeMethod(method="eagle3", model="org/draft")],
                synthetic_acceptance_length=4,
            )

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError, match="Invalid speculative method"):
            SpeculativeMethod(method="unknown", model="org/draft")

    def test_qwen3_next_mtp_allows_empty_model(self):
        method = SpeculativeMethod(method="qwen3_next_mtp", model="")
        assert method.model == ""


class TestSpeculativeDecodingSuggest:
    def test_allow_disabled_returns_none_and_user_attr(self):
        spec = _base_spec_config(allow_disabled=True)
        trial = FixedTrial({"spec_enabled": False})

        assert spec.suggest(trial) is None
        assert trial.user_attrs["speculative_config"] == "disabled"

    def test_suggest_with_rates_and_static_k(self):
        spec = _base_spec_config(
            allow_disabled=False,
            static_parameters={
                "draft_tensor_parallel_size": 1,
                "num_speculative_tokens": 2,
            },
        )
        trial = FixedTrial({"spec_method": "eagle3"})
        result = json.loads(spec.suggest(trial))

        assert result["method"] == "eagle3"
        assert result["model"] == "org/draft"
        assert result["num_speculative_tokens"] == 2
        assert result["synthetic_acceptance_rates"] == [0.8, 0.7]
        assert result["rejection_sample_method"] == "synthetic"
        assert result["draft_tensor_parallel_size"] == 1

    def test_suggest_default_k_is_full_rates_length(self):
        spec = _base_spec_config(allow_disabled=False)
        trial = FixedTrial({"spec_method": "eagle3"})
        result = json.loads(spec.suggest(trial))

        assert result["num_speculative_tokens"] == 3
        assert result["synthetic_acceptance_rates"] == [0.8, 0.7, 0.6]

    def test_suggest_tunable_k_and_draft_max_model_len(self):
        spec = _base_spec_config(
            allow_disabled=False,
            synthetic_acceptance_rates=[0.8, 0.7, 0.6, 0.5],
            static_parameters={"draft_tensor_parallel_size": 1},
            num_speculative_tokens=ListParameter(
                name="spec_num_speculative_tokens",
                enabled=True,
                options=[2, 4],
            ),
            max_model_len=ListParameter(
                name="spec_max_model_len",
                enabled=True,
                options=[4096, 8192],
            ),
        )
        trial = FixedTrial(
            {
                "spec_method": "eagle3",
                "spec_num_speculative_tokens": 4,
                "spec_max_model_len": 8192,
            }
        )
        result = json.loads(spec.suggest(trial))

        assert result["num_speculative_tokens"] == 4
        assert result["synthetic_acceptance_rates"] == [0.8, 0.7, 0.6, 0.5]
        assert result["max_model_len"] == 8192

    def test_suggest_acceptance_length_mode(self):
        spec = SpeculativeDecodingConfig(
            enabled=True,
            allow_disabled=False,
            methods=[SpeculativeMethod(method="eagle3", model="org/draft")],
            synthetic_acceptance_length=4,
            static_parameters={"num_speculative_tokens": 4},
        )
        trial = FixedTrial({"spec_method": "eagle3"})
        result = json.loads(spec.suggest(trial))

        assert result["synthetic_acceptance_length"] == 4
        assert "synthetic_acceptance_rates" not in result

    def test_qwen3_next_mtp_omits_model_from_json(self):
        spec = SpeculativeDecodingConfig(
            enabled=True,
            allow_disabled=False,
            methods=[SpeculativeMethod(method="qwen3_next_mtp", model="")],
            synthetic_acceptance_rates=[0.9],
        )
        trial = FixedTrial({"spec_method": "qwen3_next_mtp"})
        result = json.loads(spec.suggest(trial))

        assert result["method"] == "qwen3_next_mtp"
        assert "model" not in result

    def test_suggest_stores_json_user_attr(self):
        spec = _base_spec_config(allow_disabled=False)
        trial = FixedTrial({"spec_method": "eagle3"})
        json_str = spec.suggest(trial)

        assert trial.user_attrs["speculative_config"] == json_str


class TestStudyConfigSpeculativeParsing:
    _MINIMAL_SPEC_YAML = """
study:
  prefix: spec_test
optimization:
  preset: high_throughput
  sampler: tpe
  n_trials: 5
benchmark:
  benchmark_type: guidellm
  model: "test/model"
  max_seconds: 60
  prompt_tokens: 100
  output_tokens: 100
speculative_decoding:
  enabled: true
  methods:
    - method: eagle3
      model: "org/draft"
  synthetic_acceptance_rates: [0.8, 0.7]
  static_parameters:
    draft_tensor_parallel_size: 1
"""

    def test_load_speculative_example_yaml(self):
        cfg = StudyConfig.from_file(
            "examples/study_config_speculative_decoding.yaml",
            vllm_version="0.20.0",
        )
        spec = cfg.speculative_decoding
        assert spec is not None
        assert spec.enabled
        assert spec.methods[0].method == "eagle3"
        assert spec.synthetic_acceptance_rates == [0.8, 0.7, 0.6, 0.5]

    def test_parse_speculative_block_from_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(self._MINIMAL_SPEC_YAML)
        cfg = StudyConfig.from_file(str(path), vllm_version="0.20.0")

        assert cfg.speculative_decoding is not None
        assert cfg.speculative_decoding.enabled
        assert (
            cfg.speculative_decoding.static_parameters["draft_tensor_parallel_size"]
            == 1
        )

    def test_grid_sampler_rejected_when_speculative_enabled(self, tmp_path):
        yaml_content = self._MINIMAL_SPEC_YAML.replace("sampler: tpe", "sampler: grid")
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        with pytest.raises(ValueError, match="not compatible with sampler: grid"):
            StudyConfig.from_file(str(path), vllm_version="0.20.0")

    def test_qwen3_next_mtp_optional_model_in_yaml(self, tmp_path):
        yaml_content = self._MINIMAL_SPEC_YAML.replace(
            '    - method: eagle3\n      model: "org/draft"',
            "    - method: qwen3_next_mtp",
        )
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)
        cfg = StudyConfig.from_file(str(path), vllm_version="0.20.0")

        assert cfg.speculative_decoding.methods[0].method == "qwen3_next_mtp"
        assert cfg.speculative_decoding.methods[0].model == ""


class TestValidateSpeculativeVllmVersion:
    def test_skipped_when_speculative_disabled(self):
        config = SimpleNamespace(
            speculative_decoding=SimpleNamespace(enabled=False),
        )
        _validate_speculative_vllm_version(config)

    def test_raises_when_vllm_too_old(self):
        config = SimpleNamespace(
            speculative_decoding=SimpleNamespace(enabled=True),
        )
        with patch(
            "auto_tune_vllm.utils.vllm_cli_parser.VLLMCLIParser.get_vllm_version",
            return_value="0.19.0",
        ):
            with pytest.raises(RuntimeError, match="requires vLLM >= 0.20"):
                _validate_speculative_vllm_version(config)

    def test_passes_when_vllm_meets_minimum(self):
        config = SimpleNamespace(
            speculative_decoding=SimpleNamespace(enabled=True),
        )
        with patch(
            "auto_tune_vllm.utils.vllm_cli_parser.VLLMCLIParser.get_vllm_version",
            return_value="0.20.1",
        ):
            _validate_speculative_vllm_version(config)


class TestVllmCliParserVersionNormalization:
    def test_normalize_version_from_multiline_output(self):
        raw = "vLLM release\n0.20.1\n"
        assert VLLMCLIParser._normalize_version_output(raw) == "0.20.1"

    def test_normalize_version_single_line(self):
        assert VLLMCLIParser._normalize_version_output("0.10.1.1") == "0.10.1.1"

    def test_normalize_version_invalid_raises(self):
        with pytest.raises(ValueError, match="Could not parse vLLM version"):
            VLLMCLIParser._normalize_version_output("no version here")


class TestStudyControllerTrialConfigIntegration:
    def test_build_trial_config_includes_speculative_config(self):
        from auto_tune_vllm.core.study_controller import StudyController

        spec = _base_spec_config(
            allow_disabled=False,
            static_parameters={"draft_tensor_parallel_size": 1},
        )
        config = SimpleNamespace(
            study_name="test-study",
            static_parameters={},
            parameters={},
            static_environment_variables={},
            benchmark=SimpleNamespace(),
            optimization=SimpleNamespace(),
            logging_config=None,
            metrics_scraping=SimpleNamespace(),
            speculative_decoding=spec,
        )
        controller = StudyController.__new__(StudyController)
        controller.config = config

        study = optuna.create_study()
        trial = study.ask(
            fixed_distributions={
                "spec_method": optuna.distributions.CategoricalDistribution(["eagle3"]),
            }
        )
        trial_config = controller._build_trial_config(trial)

        assert "speculative_config" in trial_config.parameters
        parsed = json.loads(trial_config.parameters["speculative_config"])
        assert parsed["method"] == "eagle3"
