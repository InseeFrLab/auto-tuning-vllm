"""Unit tests for multimodal GuideLLM benchmark path."""

from __future__ import annotations

import json

import pytest

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.benchmarks.guidellm_multimodal import GuideLLMMultimodalBenchmark
from auto_tune_vllm.benchmarks.preprocessors import FlattenImageListsPreprocessor


def _build_cmd(**kwargs) -> list[str]:
    config = BenchmarkConfig(
        benchmark_type="guidellm_multimodal",
        model="Qwen/Qwen2-VL-2B-Instruct",
        dataset="hf://demo-dataset",
        data_preprocessors=["flatten_image_lists", "encode_media"],
        **kwargs,
    )
    return GuideLLMMultimodalBenchmark()._build_guidellm_command(
        "http://localhost:8000/v1", config, "/tmp/results.json"
    )


def test_multimodal_requires_real_dataset():
    config = BenchmarkConfig(
        benchmark_type="guidellm_multimodal",
        model="Qwen/Qwen2-VL-2B-Instruct",
        dataset=None,
        data_preprocessors=["flatten_image_lists", "encode_media"],
    )
    with pytest.raises(ValueError, match="requires a real dataset"):
        GuideLLMMultimodalBenchmark()._build_guidellm_command(
            "http://localhost:8000/v1", config, "/tmp/results.json"
        )


def test_multimodal_requires_data_preprocessors():
    config = BenchmarkConfig(
        benchmark_type="guidellm_multimodal",
        model="Qwen/Qwen2-VL-2B-Instruct",
        dataset="hf://demo-dataset",
        data_preprocessors=None,
    )
    with pytest.raises(ValueError, match="requires data_preprocessors"):
        GuideLLMMultimodalBenchmark()._build_guidellm_command(
            "http://localhost:8000/v1", config, "/tmp/results.json"
        )


def test_multimodal_command_includes_multimodal_args():
    cmd = _build_cmd(
        request_format="/v1/chat/completions",
        data_column_mapper={"text_column": "prompt", "image_column": "image"},
        data_preprocessors_kwargs={"base_dirs": ["examples/vlm_multi_image"]},
        data_args={"split": "train"},
        data_finalizer="generative",
    )

    assert "--request-format" in cmd
    assert cmd[cmd.index("--request-format") + 1] == "/v1/chat/completions"
    assert "--data-preprocessors" in cmd
    assert (
        cmd[cmd.index("--data-preprocessors") + 1] == "flatten_image_lists,encode_media"
    )
    assert json.loads(cmd[cmd.index("--data-column-mapper") + 1]) == {
        "text_column": "prompt",
        "image_column": "image",
    }
    assert json.loads(cmd[cmd.index("--data-preprocessors-kwargs") + 1]) == {
        "base_dirs": ["examples/vlm_multi_image"]
    }
    assert json.loads(cmd[cmd.index("--data-args") + 1]) == {"split": "train"}
    assert cmd[cmd.index("--data-finalizer") + 1] == "generative"


def test_flatten_image_lists_preprocessor_flattens_nested_images():
    preprocessor = FlattenImageListsPreprocessor(base_dirs=[])
    items = [{"image_column": [["image1.png", ""], "image2.png", None]}]

    processed = preprocessor(items)

    assert processed[0]["image_column"] == ["image1.png", "image2.png"]
