# Current work

_Last updated from local `gh` / git — refresh before large changes._

## Open pull requests (InseeFrLab)

| PR | Branch | Objective | Status | Next step |
|----|--------|-----------|--------|-----------|
| [#22](https://github.com/InseeFrLab/auto-tuning-vllm/pull/22) | `FEAT/optuna-user-attrs-log-metrics` | `optimization.log_metrics` → Optuna user attrs for dashboard | OPEN | Review + merge; ensure docs/example match `StudyConfig` validation |
| [#21](https://github.com/InseeFrLab/auto-tuning-vllm/pull/21) | `fix/exclude-baseline-trials-budget` | Baselines must not increment `completed_trials` / consume `n_trials` | OPEN | Merge; run `pytest tests/core/test_study_controller.py` |
| [#17](https://github.com/InseeFrLab/auto-tuning-vllm/pull/17) | `fix/guidellm-cli-preflight` | GuideLLM CLI preflight + pin `vllm<=0.19` | OPEN | Resolve overlap with issue #19 / current `pyproject` vllm pin |
| [#13](https://github.com/InseeFrLab/auto-tuning-vllm/pull/13) | `fix/local-backend-cleanup` | Cooperative cancel + cleanup on local backend | OPEN | Merge after manual interrupt test |

## Remote branches (not all have open PRs)

| Branch | Notes |
|--------|--------|
| `origin/FEAT/custom-metrics` | Merged as #18 on main |
| `origin/FEAT/grid-cardinality-auto-switch` | Merged as #7 |
| `origin/FEAT/ray-optional` | Legacy: Ray optional extra (merged) |
| `origin/add-optuna-dashboard-example` | Dashboard launcher (#14 merged) |
| `origin/add-startup-timeout-baseline-run` | Startup timeout for baselines — **verify if merged or stale** |
| `origin/ci-setup` | CI workflow (#8) |
| `origin/renovate/configure` | Dependency bot config |

## README roadmap (main)

| Item | Status | Next step |
|------|--------|-----------|
| Comprehensive test suite | In progress (small `tests/` tree) | Add controller/backend tests per PR #21 pattern |
| CI runs tests strictly | Partial | Remove `pytest ... \|\| true` in `ci.yml` when suite is stable |
| Dependency pinning / hygiene | In progress | Align `pyproject.toml` with supported vLLM/GuideLLM matrix (GuideLLM `>=0.7.1`) |
| CLI validation / error messages | Open | Extend `StudyConfig` errors + Typer messages |
| Speculative decoding params | Future | Design parameter module + example YAML |
| Extra benchmark providers | Future | Implement `BenchmarkProvider` subclass |

## Maintainer TODO (fill if stale)

- **Active local branch:** `FEAT/optuna-user-attrs-log-metrics` — confirm whether uncommitted edits on `config.py` / `study_controller.py` belong in PR #22.
- **Production study configs:** _Add paths or naming convention used internally._
- **Target vLLM version for production:** _e.g. 0.19 vs 0.20+ — drives issue #19 resolution._
