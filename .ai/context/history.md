# History — decisions to preserve

## Execution model

| Decision | Reference |
|----------|-----------|
| **Local backend is the product path** | `LocalExecutionBackend`; fork README |
| **Do not kill parent process group** on vLLM cleanup | upstream PR #92 |
| Ray backend kept as **legacy / optional** extra | `db1e9ab`; issue #3 — not primary development |

## Optuna / study

| Decision | Reference |
|----------|-----------|
| Baselines visible in Optuna dashboard | upstream PR #111 |
| Failed trial attrs for sampler | #93, #97 |
| Constraint sampling | #101 |
| Grid cardinality auto-switch | fork PR #7 |
| Custom metric expressions | fork PR #18 |
| `max_concurrent_trials` naming | upstream #122, #125 |

## Benchmarking

| Decision | Reference |
|----------|-----------|
| GuideLLM as default provider | `benchmarks/providers.py` |
| Process-group benchmark terminate | `BenchmarkProvider` |

## Config / vLLM

| Decision | Reference |
|----------|-----------|
| Versioned defaults in `schemas/vllm_defaults/` | `version_manager.py` |
| Config validation in Python (no separate JSON schema) | upstream #110 |

## Tooling

| Decision | Reference |
|----------|-----------|
| CI: Ruff + pytest matrix | fork PR #8 |
| Optuna Dashboard script | fork PR #14 |

Upstream: [openshift-psap/auto-tuning-vllm](https://github.com/openshift-psap/auto-tuning-vllm). Fork emphasizes **local execution**, tests, and dependency control.
