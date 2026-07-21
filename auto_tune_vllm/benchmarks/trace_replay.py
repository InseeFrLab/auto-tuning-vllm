"""GuideLLM benchmark provider for trace replay workloads (GuideLLM >= 0.7.1)."""

from __future__ import annotations

from .config import BenchmarkConfig
from .profiles import ReplayProfile, profile_from_dict, render_replay_data_cli
from .providers import GuideLLMBenchmark


class GuideLLMTraceReplayBenchmark(GuideLLMBenchmark):
    """Benchmark provider that replays trace files via GuideLLM ``kind=replay``."""

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
