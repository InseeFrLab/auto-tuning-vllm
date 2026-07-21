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
| `optimization.log_metrics` → user attrs | fork PR #22 |
| Baseline startup timeout from env | fork PR #20 |

## Benchmarking

| Decision | Reference |
|----------|-----------|
| GuideLLM as default provider | `benchmarks/providers.py` |
| GuideLLM `>= 0.7.1` CLI subprocess runner | fork PR #37 |
| Warmup / cooldown / ramp-up / sample_requests | fork PRs #24, #27, #31 |
| Prompt + total token throughput parsing | fork PR #29 |
| Multimodal VLM profile (`guidellm_multimodal`) | fork PR #34 |
| Trace replay profile + optional prewarm | fork PRs #38, #39 |
| Prometheus scrape during benchmark for vLLM metrics | fork PR #35 |
| Process-group benchmark terminate | `BenchmarkProvider` |

## Config / vLLM

| Decision | Reference |
|----------|-----------|
| Versioned defaults in `schemas/vllm_defaults/` | `version_manager.py` |
| Config validation in Python (no separate JSON schema) | upstream #110 |
| Speculative decoding YAML block + search space | fork PR #40; `core/speculative.py` |
| MTP vs EAGLE3 model compatibility documented | fork PR #41 |

## Tooling

| Decision | Reference |
|----------|-----------|
| CI: Ruff + pytest matrix | fork PR #8 |
| Optuna Dashboard script | fork PR #14 |
| Agent onboarding (`AGENTS.md`, `.ai/`) | fork PR #23 |
| Examples trimmed to feature-focused YAMLs + `examples/README.md` | 2026-07 maintenance |

Upstream: [openshift-psap/auto-tuning-vllm](https://github.com/openshift-psap/auto-tuning-vllm). Fork emphasizes **local execution**, tests, GuideLLM benchmark profiles, and dependency control.
