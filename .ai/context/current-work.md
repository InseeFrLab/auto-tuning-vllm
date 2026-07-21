# Current work

_Last updated: 2026-07-21 — refresh before large changes._

## Open pull requests (InseeFrLab)

| PR | Branch | Objective | Status | Next step |
|----|--------|-----------|--------|-----------|
| [#32](https://github.com/InseeFrLab/auto-tuning-vllm/pull/32) | `add-no-repeat-feature` | `optimization.no_repeat` — skip duplicate parameter combos | OPEN | Review + merge |
| [#28](https://github.com/InseeFrLab/auto-tuning-vllm/pull/28) | `FEAT/n-repeats-benchmark` | `optimization.n_repeats` — repeat benchmark per trial | OPEN | Review + merge |
| [#21](https://github.com/InseeFrLab/auto-tuning-vllm/pull/21) | `fix/exclude-baseline-trials-budget` | Baselines must not increment `completed_trials` / consume `n_trials` | OPEN | Merge; run `pytest tests/core/test_study_controller.py` |
| [#17](https://github.com/InseeFrLab/auto-tuning-vllm/pull/17) | `fix/guidellm-cli-preflight` | GuideLLM CLI preflight + pin `vllm<=0.19` | OPEN | Resolve overlap with issue #19 / current `pyproject` vllm pin |
| [#13](https://github.com/InseeFrLab/auto-tuning-vllm/pull/13) | `fix/local-backend-cleanup` | Cooperative cancel + cleanup on local backend | OPEN | Merge after manual interrupt test |

## Recently merged (main)

| PR | Summary |
|----|---------|
| [#41](https://github.com/InseeFrLab/auto-tuning-vllm/pull/41) | Speculative decoding limitations docs |
| [#40](https://github.com/InseeFrLab/auto-tuning-vllm/pull/40) | Speculative decoding optimization (`core/speculative.py`, example YAML) |
| [#39](https://github.com/InseeFrLab/auto-tuning-vllm/pull/39) | Optional prewarm before trace replay benchmarks |
| [#38](https://github.com/InseeFrLab/auto-tuning-vllm/pull/38) | GuideLLM trace replay benchmark profile |
| [#37](https://github.com/InseeFrLab/auto-tuning-vllm/pull/37) | GuideLLM `>= 0.7.1` migration, subprocess deadlock fix |
| [#35](https://github.com/InseeFrLab/auto-tuning-vllm/pull/35) | vLLM Prometheus metrics scraping for `log_metrics` |
| [#34](https://github.com/InseeFrLab/auto-tuning-vllm/pull/34) | GuideLLM multimodal path (multi-image VLM) |
| [#31](https://github.com/InseeFrLab/auto-tuning-vllm/pull/31) | GuideLLM concurrent benchmark ramp-up |
| [#29](https://github.com/InseeFrLab/auto-tuning-vllm/pull/29) | Parse prompt/total token throughput from GuideLLM |
| [#27](https://github.com/InseeFrLab/auto-tuning-vllm/pull/27) | `benchmark.sample_requests` |
| [#24](https://github.com/InseeFrLab/auto-tuning-vllm/pull/24) | `benchmark.warmup` / `benchmark.cooldown` |
| [#22](https://github.com/InseeFrLab/auto-tuning-vllm/pull/22) | `optimization.log_metrics` → Optuna user attrs |
| [#20](https://github.com/InseeFrLab/auto-tuning-vllm/pull/20) | Baseline trials honor `VLLM_STARTUP_TIMEOUT` |

## Examples layout (post-cleanup)

| File | Role |
|------|------|
| `examples/study_config_local_exec.yaml` | Primary full local study |
| `examples/study_config.yaml` | Basic multi-objective starter |
| `examples/study_config_minimal.yaml` | Quick smoke test |
| `examples/study_config_speculative_decoding.yaml` | EAGLE3 speculative decoding |
| `examples/study_config_trace_replay.yaml` | Trace replay + sample JSONL |
| `examples/study_config_vlm_multi_image.yaml` | Multimodal VLM + assets |
| `examples/README.md` | Index; `README_optimization_guide.md` for objectives |

Removed legacy: `trial_config_*`, `test_defaults_config.yaml`, `study_config_no_postgres.yaml`, `study_config_optimization_examples.yaml`, `README_trial_configs.md`.

## README roadmap (main)

| Item | Status | Next step |
|------|--------|-----------|
| Comprehensive test suite | In progress | Add controller/backend tests per PR #21 pattern |
| CI runs tests strictly | Partial | Remove `pytest ... \|\| true` in `ci.yml` when suite is stable |
| Dependency pinning / hygiene | In progress | Align `pyproject.toml` with supported vLLM/GuideLLM matrix |
| CLI validation / error messages | Open | Extend `StudyConfig` errors + Typer messages |
| Speculative decoding params | **Done** (#40) | Monitor vLLM API drift |
| Extra benchmark providers | Future | Implement `BenchmarkProvider` subclass |
| `n_repeats` / `no_repeat` | PRs open | Merge #28, #32 |

## Maintainer TODO (fill if stale)

- **Target vLLM version for production:** _e.g. 0.19 vs 0.20+ — drives issue #19 resolution._
- **Production study configs:** _Add paths or naming convention used internally._
