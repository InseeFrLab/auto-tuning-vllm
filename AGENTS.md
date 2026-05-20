# Agent guide — auto-tuning-vllm (InseeFrLab fork)

Hyperparameter optimization for **vLLM** serving: YAML configs, **Optuna**, **GuideLLM** benchmarks, **local** execution (`LocalExecutionBackend`). PostgreSQL storage optional.

> **Agents:** do not run `auto-tune-vllm optimize` / `resume` (GPU, vLLM, long jobs). Use lint + `pytest` only.

## Context files

| File | Purpose |
|------|---------|
| [.ai/context/repo-map.md](.ai/context/repo-map.md) | Directories and entry points |
| [.ai/context/architecture.md](.ai/context/architecture.md) | Execution flow (prose); diagrams in `docs/architecture.md` |
| [.ai/context/current-work.md](.ai/context/current-work.md) | Open PRs, roadmap |
| [.ai/context/known-issues.md](.ai/context/known-issues.md) | Bugs and limitations |
| [.ai/context/history.md](.ai/context/history.md) | Design decisions |
| [.ai/context/external-links.md](.ai/context/external-links.md) | External docs |

## Skills

| Skill | Use when |
|-------|----------|
| [.ai/skills/pr-writer.md](.ai/skills/pr-writer.md) | Drafting a PR |
| [.ai/skills/pr-reviewer.md](.ai/skills/pr-reviewer.md) | Reviewing a PR (diff + ruff + pytest) |
| [.ai/skills/test-writer.md](.ai/skills/test-writer.md) | Adding tests |
| [.ai/skills/docs-writer.md](.ai/skills/docs-writer.md) | README / `docs/` |
| [.ai/skills/architecture-diagrams.md](.ai/skills/architecture-diagrams.md) | Updating `docs/architecture.md` |

## Priorities

1. **Local backend** — `LocalExecutionBackend`, `BaseTrialController` / `LocalTrialController`.
2. **Config** — `StudyConfig`; sync `docs/configuration.md` + `examples/*.yaml`.
3. **Trial lifecycle** — vLLM subprocess, GuideLLM, cancellation, cleanup (`trial_controller.py`, `backends.py`).
4. **Optuna** — `ask`/`tell`, baselines, `core/storage/utils.py`.
5. **Tests** — `pytest` under `tests/`; no GPU in default CI.
6. **Small diffs** — match existing patterns.

## Commands (safe for agents)

```bash
source venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -v tests/
```

Fork: https://github.com/InseeFrLab/auto-tuning-vllm
