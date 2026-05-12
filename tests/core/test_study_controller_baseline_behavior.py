"""Generic baseline behavior tests for StudyController."""

from __future__ import annotations

from types import SimpleNamespace

from auto_tune_vllm.core.study_controller import StudyController
from auto_tune_vllm.core.trial import TrialResult


class _StubTrial:
    def __init__(self, number: int):
        self.number = number
        self.user_attrs: dict[str, object] = {}

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


class _StubStudy:
    def __init__(self):
        self.enqueued_parameters: list[dict[str, object]] = []
        self.tell_calls: list[dict[str, object]] = []
        self._trial = _StubTrial(number=0)

    def enqueue_trial(self, params: dict[str, object]) -> None:
        self.enqueued_parameters.append(params)

    def ask(self) -> _StubTrial:
        return self._trial

    def tell(self, trial: int, values: list[float] | None, state) -> None:
        self.tell_calls.append({"trial": trial, "values": values, "state": state})


class _StubBackend:
    def __init__(self):
        self.submitted_trial_configs = []
        self._job_handle = object()
        self._result = TrialResult(
            trial_id="baseline_concurrency_1",
            trial_number=0,
            trial_type="baseline",
            objective_values=[1.0],
            success=True,
        )

    def submit_trial(self, trial_config):
        self.submitted_trial_configs.append(trial_config)
        return self._job_handle

    def poll_trials(self, job_handles):
        return [self._result], []


def _make_config(
    *,
    static_env: dict[str, str] | None = None,
    static_parameters: dict[str, object] | None = None,
    baseline_parameters: dict[str, object] | None = None,
    concurrency_levels: list[int] | None = None,
    baseline_enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        study_name="test-study",
        baseline=SimpleNamespace(
            enabled=baseline_enabled,
            concurrency_levels=concurrency_levels or [1],
            parameters=baseline_parameters or {},
        ),
        static_parameters=static_parameters or {},
        parameters={},
        static_environment_variables=static_env or {},
        benchmark=SimpleNamespace(rate=1),
        optimization=SimpleNamespace(objectives=[]),
        logging_config=None,
    )


def test_baseline_uses_static_env_startup_timeout() -> None:
    backend = _StubBackend()
    study = _StubStudy()
    config = _make_config(static_env={"VLLM_STARTUP_TIMEOUT": "777"})
    controller = StudyController(backend=backend, study=study, config=config)

    controller._run_baseline_trials()

    assert len(backend.submitted_trial_configs) == 1
    assert backend.submitted_trial_configs[0].vllm_startup_timeout == 777


def test_baseline_uses_default_startup_timeout_when_missing() -> None:
    backend = _StubBackend()
    study = _StubStudy()
    config = _make_config()
    controller = StudyController(backend=backend, study=study, config=config)

    controller._run_baseline_trials()

    assert len(backend.submitted_trial_configs) == 1
    assert backend.submitted_trial_configs[0].vllm_startup_timeout == 300


def test_baseline_merges_static_and_baseline_parameters() -> None:
    backend = _StubBackend()
    study = _StubStudy()
    config = _make_config(
        static_parameters={
            "gpu_memory_utilization": 0.8,
            "max_num_batched_tokens": 1024,
        },
        baseline_parameters={"gpu_memory_utilization": 0.9, "block_size": 16},
    )
    controller = StudyController(backend=backend, study=study, config=config)

    controller._run_baseline_trials()

    assert len(backend.submitted_trial_configs) == 1
    submitted = backend.submitted_trial_configs[0].parameters
    assert submitted["gpu_memory_utilization"] == 0.9
    assert submitted["max_num_batched_tokens"] == 1024
    assert submitted["block_size"] == 16


def test_baseline_adds_max_num_seqs_for_high_concurrency() -> None:
    backend = _StubBackend()
    study = _StubStudy()
    config = _make_config(concurrency_levels=[300])
    controller = StudyController(backend=backend, study=study, config=config)

    controller._run_baseline_trials()

    assert len(backend.submitted_trial_configs) == 1
    submitted = backend.submitted_trial_configs[0].parameters
    assert submitted["max_num_seqs"] == 300


def test_baseline_skips_when_disabled() -> None:
    backend = _StubBackend()
    study = _StubStudy()
    config = _make_config(baseline_enabled=False)
    controller = StudyController(backend=backend, study=study, config=config)

    controller._run_baseline_trials()

    assert backend.submitted_trial_configs == []
