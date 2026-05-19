# Skill: PR writer

## Before opening

1. Branch from `main`; one logical change per PR.
2. Run `ruff check .` and `pytest -v tests/` (`venv`).
3. Update `docs/configuration.md` + `examples/*.yaml` if schema or CLI changed.
4. **Do not** list `auto-tune-vllm optimize` as agent-run validation; maintainers run E2E manually.

## Title convention

- `[FEAT]` / `[FIX]` / `[CI]` / `[Docs]`

## Template

```markdown
## Summary
<what changed>

## Why
<problem or goal>

## What changed
- `path/module.py` — …
- `tests/...` — …
- `docs/` or `examples/` — …

## How tested
- [ ] `ruff check .`
- [ ] `pytest -v tests/...`
- [ ] Manual E2E (maintainer): auto-tune-vllm optimize …

## Risks / limitations
- …

## Links
- Closes #…
```

## Fork checks

- Baseline vs `n_trials` if `study_controller.py` touched.
- Optuna storage (SQLite vs PostgreSQL).
- `pyproject.toml` pins (vLLM / GuideLLM).
- Structural change → update `docs/architecture.md` (see `architecture-diagrams.md`).

## After merge

Update `.ai/context/current-work.md` and `.ai/context/known-issues.md` when relevant.
