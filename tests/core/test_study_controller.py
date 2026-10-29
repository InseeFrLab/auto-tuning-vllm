"""Unit tests for ``StudyController`` (baseline budget, orchestration hooks, etc.)."""

from __future__ import annotations

from pathlib import Path

from typing_extensions import override

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.core.config import (
    BaselineConfig,
    OptimizationConfig,
    StudyConfig,
)
from auto_tune_vllm.core.study_controller import StudyController
from auto_tune_vllm.core.trial import TrialConfig, TrialResult
from auto_tune_vllm.execution.backends import ExecutionBackend, JobHandle


class _ImmediateCompleteBackend(ExecutionBackend):
    """Backend that completes any submitted trial on the first poll (no GPU / vLLM)."""

    def __init__(self) -> None:
        self._pending: TrialConfig | None = None

    @override
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        self._pending = trial_config
        return JobHandle(trial_config.trial_id, "immediate-job")

    @override
    def poll_trials(
        self, job_handles: list[JobHandle]
    ) -> tuple[list[TrialResult], list[JobHandle]]:
        if not job_handles or self._pending is None:
            return [], list(job_handles)
        cfg = self._pending
        self._pending = None
        result = TrialResult(
            trial_id=cfg.trial_id,
            trial_number=cfg.trial_number,
            trial_type=cfg.trial_type,
            objective_values=[1.0],
            success=True,
        )
        return [result], []

    @override
    def shutdown(self) -> None:
        return None

    @override
    def cleanup_all_trials(self) -> None:
        return None


def test_baseline_trials_do_not_increment_completed_trials(tmp_path: Path) -> None:
    """``n_trials`` counts optimization trials only; baselines must not shrink that budget."""
    optimization = OptimizationConfig(
        preset="high_throughput",
        sampler="random",
        n_trials=5,
        n_startup_trials=0,
        max_concurrent_trials=1,
    )
    config = StudyConfig(
        study_name="test_study_controller_baseline_budget",
        database_url=None,
        optimization=optimization,
        benchmark=BenchmarkConfig(model="dummy-model", max_seconds=1, rate=50),
        baseline=BaselineConfig(
            enabled=True,
            concurrency_levels=[50, 100],
        ),
        storage_file=str(tmp_path / "optuna.db"),
    )
    backend = _ImmediateCompleteBackend()
    controller = StudyController.create_from_config(backend, config, create_db=False)

    assert controller.completed_trials == 0
    controller._run_baseline_trials()  # pyright: ignore[reportPrivateUsage]
    assert controller.completed_trials == 0
