# Skill: PR reviewer

Review = **read the diff**, **reason about behavior**, optionally **lint + unit tests**.
**Never run the autotuner** (`auto-tune-vllm optimize`, `resume`, or any command that starts vLLM / GuideLLM / GPU work). Maintainers run end-to-end studies manually.

## Allowed commands (agents)

```bash
source venv/bin/activate
ruff check .
pytest -v tests/
# optional: basedpyright (if enabled locally)
```

## Review workflow

1. Read PR description and linked issues.
2. Walk changed files; trace call path from `cli/main.py` or `StudyController` when relevant.
3. Run `ruff check .` and `pytest -v tests/` if environment is available.
4. Record findings in the output format below.

## Config & CLI

- [ ] `StudyConfig.from_file()` — new fields validated; errors actionable.
- [ ] `examples/*.yaml` + `docs/configuration.md` aligned.
- [ ] Typer options in `cli/main.py` documented when added.

## Local execution path (primary)

- [ ] `LocalExecutionBackend` — submit/poll/cancel/cleanup semantics still coherent.
- [ ] `trial_controller.py` — vLLM + GuideLLM lifecycle, cancellation, `cleanup_resources()`.
- [ ] No regression for install **without** Ray (`pip install -e .` only).

## Optuna

- [ ] `study.ask()` / `study.tell()` paired; failures → `FAIL` + user attrs.
- [ ] Baseline vs optimization trial counting (`n_trials`, PR #21 context).
- [ ] Grid / sampler / multi-objective values consistent.

## Benchmarks & metrics

- [ ] `benchmarks/providers.py` — GuideLLM CLI args from `BenchmarkConfig`.
- [ ] Objective expressions match `ObjectiveConfig.valid_metrics_combined`.

## Tests & docs

- [ ] New behavior covered in `tests/` without mandatory GPU.
- [ ] User-facing docs updated when behavior or YAML changes.

## Legacy Ray (only if PR touches `RayExecutionBackend`)

- [ ] Optional import still works; no new hard dependency on `ray` in core install path.
- [ ] No Ray-specific review steps unless the diff is explicitly Ray-related.

## Output format

```markdown
### Blockers
- ...

### Questions
- ...

### Nits
- ...

### Checks run
- [ ] ruff
- [ ] pytest
```
