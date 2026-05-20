# Skill: Test writer

## Run

```bash
source venv/bin/activate
pytest -v tests/
```

**Do not** use full `auto-tune-vllm optimize` in tests or agent workflows.

## Priority

1. `StudyConfig.from_file` / validation (`tests/core/`)
2. Metric expressions (`tests/execution/test_evaluate_metric_expression.py`)
3. `StudyController` with fake `ExecutionBackend` (no vLLM)
4. Mock `subprocess` for GuideLLM in `providers.py`
5. GPU integration only when explicitly requested by maintainer

## Patterns

- Optuna: `sqlite:///:memory:`
- Backend fake: return `TrialResult` on first `poll_trials`
- No CUDA in default CI matrix
