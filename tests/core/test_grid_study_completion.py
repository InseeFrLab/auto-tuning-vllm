"""Regression: the final GridSampler result must count in ask/tell studies."""

from types import SimpleNamespace

import optuna
import pytest

from auto_tune_vllm.core.study_controller import StudyController
from auto_tune_vllm.core.trial import TrialResult


@pytest.mark.parametrize("success", [True, False])
def test_final_grid_trial_is_counted_and_removed(success):
    study = optuna.create_study(sampler=optuna.samplers.GridSampler({"x": [1]}))
    trial = study.ask()
    trial.suggest_categorical("x", [1])
    result = TrialResult(
        trial_id="trial_0",
        trial_number=trial.number,
        trial_type="optimization",
        objective_values=[2.0] if success else [],
        success=success,
    )
    backend = SimpleNamespace(poll_trials=lambda handles: ([result], []))
    config = SimpleNamespace(optimization=SimpleNamespace(log_metrics=[]))
    controller = StudyController(backend, study, config)
    controller.active_trials[result.trial_id] = object()
    controller.trial_objects[trial.number] = trial

    assert controller._collect_completed_trials() == 1
    assert study.trials[0].state == (
        optuna.trial.TrialState.COMPLETE if success else optuna.trial.TrialState.FAIL
    )
    assert study.trials[0].value == (2.0 if success else None)
    assert not controller.active_trials
    assert not controller.trial_objects
