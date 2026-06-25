"""GuideLLM benchmark provider for multi-image VLM workloads (GuideLLM 0.6+)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

from .config import BenchmarkConfig
from .providers import GuideLLMBenchmark


class GuideLLMMultimodalBenchmark(GuideLLMBenchmark):
    """Parallel benchmark path for JSONL datasets with multiple images per request."""

    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        self._logger.info(f"Starting GuideLLM multimodal benchmark for {config.model}")

        self._results_file = self._get_results_file_path()
        cmd = self._build_guidellm_command(model_url, config, self._results_file)

        if importlib.util.find_spec("guidellm") is None:
            raise RuntimeError(
                "GuideLLM is not installed for this Python interpreter. "
                f"Install with: {sys.executable} -m pip install 'guidellm>=0.6.0,<0.7.0'"
            )
        if not (model_url.startswith("http://") or model_url.startswith("https://")):
            raise ValueError(f"Invalid model_url: {model_url!r} (expected http/https)")

        self._logger.info(f"Running: {' '.join(cmd)}")
        self._logger.info(f"Results will be saved to: {self._results_file}")

        env = os.environ.copy()
        env["GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL"] = config.logging_level

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )

        self._process_pid = self._process.pid
        try:
            self._process_pgid = os.getpgid(self._process_pid)
            self._logger.debug(
                f"Started GuideLLM process {self._process_pid} "
                f"in process group {self._process_pgid}"
            )
        except (OSError, ProcessLookupError):
            self._logger.warning(
                f"Failed to get process group for GuideLLM process {self._process_pid}"
            )
            self._process_pgid = None

        return self._process

    def _build_guidellm_command(
        self, model_url: str, config: BenchmarkConfig, results_file: str
    ) -> list[str]:
        if config.use_synthetic_data:
            raise ValueError(
                "benchmark_type 'guidellm_multimodal' requires a real dataset; "
                "set benchmark.dataset to a JSONL file path"
            )
        if not config.data_preprocessors:
            raise ValueError(
                "benchmark_type 'guidellm_multimodal' requires data_preprocessors, "
                "e.g. ['flatten_image_lists', 'encode_media']"
            )

        processor = config.processor if config.processor is not None else config.model

        cmd = [
            sys.executable,
            "-m",
            "auto_tune_vllm.benchmarks._guidellm_shim",
            "benchmark",
            "run",
            "--target",
            model_url,
            "--model",
            config.model,
            "--processor",
            processor,
            "--backend",
            "openai_http",
            "--profile",
            "concurrent",
            "--max-seconds",
            str(config.max_seconds),
            "--rate",
            str(config.rate),
            "--output-path",
            results_file,
            "--processor-args",
            '{"trust-remote-code":"true"}',
            "--sample-requests",
            str(config.sample_requests),
        ]

        if config.request_format is not None:
            cmd.extend(["--request-format", config.request_format])

        if config.warmup is not None:
            cmd.extend(["--warmup", str(config.warmup)])
        if config.cooldown is not None:
            cmd.extend(["--cooldown", str(config.cooldown)])

        if config.data_args is not None:
            cmd.extend(["--data-args", json.dumps(config.data_args)])
        if config.data_column_mapper is not None:
            cmd.extend(["--data-column-mapper", json.dumps(config.data_column_mapper)])
        cmd.extend(["--data-preprocessors", ",".join(config.data_preprocessors)])
        if config.data_preprocessors_kwargs is not None:
            cmd.extend(
                [
                    "--data-preprocessors-kwargs",
                    json.dumps(config.data_preprocessors_kwargs),
                ]
            )
        if config.data_finalizer is not None:
            cmd.extend(["--data-finalizer", config.data_finalizer])

        if config.dataset.startswith("hf://"):
            dataset_name = config.dataset[5:]
            cmd.extend(["--data", dataset_name])
        else:
            if not os.path.exists(config.dataset):
                raise FileNotFoundError(f"Dataset file not found: {config.dataset}")
            cmd.extend(["--data", config.dataset])

        return cmd
