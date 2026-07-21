"""GuideLLM benchmark provider for trace replay workloads (GuideLLM >= 0.7.1)."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

from .config import BenchmarkConfig
from .profiles import (
    ReplayProfile,
    compute_trace_token_stats,
    profile_from_dict,
    render_prewarm_args,
    render_replay_data_cli,
)
from .providers import GuideLLMBenchmark


class GuideLLMTraceReplayBenchmark(GuideLLMBenchmark):
    """Benchmark provider that replays trace files via GuideLLM ``kind=replay``."""

    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        """Start trace replay benchmark, optionally after a concurrent prewarm run."""
        self._logger.info(
            f"Starting GuideLLM trace replay benchmark for {config.model}"
        )

        self._validate_runtime(model_url, config)

        self._results_file = self._get_results_file_path()

        if config.prewarm is not None:
            self._run_prewarm(model_url, config)

        cmd = self._build_guidellm_command(model_url, config, self._results_file)

        self._logger.info(f"Running: {' '.join(cmd)}")
        self._logger.info(f"Results will be saved to: {self._results_file}")

        return self._launch_subprocess(cmd, config)

    def _build_guidellm_command(
        self, model_url: str, config: BenchmarkConfig, results_file: str
    ) -> list[str]:
        profile = profile_from_dict(config.profile)
        if not isinstance(profile, ReplayProfile):
            raise ValueError(
                "benchmark_type 'guidellm_trace_replay' requires profile.kind='replay'"
            )

        cmd = self._build_guidellm_base_command(model_url, config, results_file)
        cmd.extend(render_replay_data_cli(config, profile))
        return cmd

    def _build_prewarm_command(
        self,
        model_url: str,
        config: BenchmarkConfig,
        profile: ReplayProfile,
        prewarm_results_file: str,
    ) -> list[str]:
        """Build a short concurrent GuideLLM run to warm up vLLM kernels."""
        stats = compute_trace_token_stats(
            config.dataset,
            profile.prompt_tokens_column,
            profile.output_tokens_column,
        )
        processor = config.processor if config.processor is not None else config.model

        cmd = [
            "guidellm",
            "run",
            "--backend",
            f"kind=openai_http,target={model_url},model={config.model}",
            "--tokenizer",
            json.dumps(
                {
                    "kind": "huggingface_auto",
                    "model": processor,
                    "load_kwargs": {"trust_remote_code": True},
                }
            ),
        ]
        cmd.extend(
            render_prewarm_args(config, config.prewarm, stats, prewarm_results_file)
        )
        return cmd

    def _prewarm_results_path(self) -> str:
        results_path = Path(self._results_file)
        if results_path.name.endswith("_benchmark_results.json"):
            return str(
                results_path.with_name(
                    results_path.name.replace(
                        "_benchmark_results.json", "_prewarm_results.json"
                    )
                )
            )
        return str(
            results_path.with_name(f"{results_path.stem}_prewarm{results_path.suffix}")
        )

    def _run_prewarm(self, model_url: str, config: BenchmarkConfig) -> None:
        profile = profile_from_dict(config.profile)
        if not isinstance(profile, ReplayProfile):
            raise ValueError(
                "benchmark_type 'guidellm_trace_replay' requires profile.kind='replay'"
            )

        prewarm_results_file = self._prewarm_results_path()
        prewarm_log_path = str(Path(prewarm_results_file).with_suffix(".prewarm.log"))
        cmd = self._build_prewarm_command(
            model_url, config, profile, prewarm_results_file
        )

        duration = float(config.prewarm["duration"])
        self._logger.info(
            f"Running prewarm ({duration:.0f}s, "
            f"concurrency={config.prewarm['concurrency']}): {' '.join(cmd)}"
        )

        process = self._launch_subprocess(cmd, config, log_path=prewarm_log_path)
        timeout = max(duration * 2, duration + 30)

        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._logger.warning(
                f"Prewarm did not finish within {timeout:.0f}s; terminating prewarm"
            )
            self.terminate_benchmark()
            self._clear_process_handles()
            return

        self._clear_process_handles()

        if returncode == 0:
            self._logger.info("Prewarm completed successfully")
            return

        if returncode < 0 and -returncode in (signal.SIGTERM, signal.SIGKILL):
            raise KeyboardInterrupt("Prewarm cancelled")

        log_tail = self.get_last_log_lines()
        self._logger.warning(
            f"Prewarm exited with code {returncode}; continuing with trace replay. "
            f"Log tail:\n{log_tail}"
        )

    def _clear_process_handles(self) -> None:
        self._process = None
        self._process_pid = None
        self._process_pgid = None
