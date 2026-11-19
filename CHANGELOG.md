# Changelog

All notable changes to this project are documented here.

The fork baseline is commit [`3c4264d`](https://github.com/InseeFrLab/auto-tuning-vllm/commit/3c4264d93a594e6ae4b2e919741a455f282c2701) (CI setup, pre-commit, unified workflow).

## 0.1.0

### Execution

- **Local backend by default** — run on a single machine without Ray (`pip install -e .`); Ray remains available via `pip install -e ".[ray]"` and `--backend ray` ([#3](https://github.com/InseeFrLab/auto-tuning-vllm/issues/3))
- **`max_concurrent_trials` defaults to 1** on the local backend; Python environment flags (`--venv-path`, etc.) are only required for Ray

### Optimization

- **Custom metric expressions** — compose objectives from benchmark identifiers (e.g. `output_tokens_per_second_mean / requests_per_second_median`) instead of a single named metric ([#18](https://github.com/InseeFrLab/auto-tuning-vllm/pull/18))
- **Grid cardinality auto-switch** — `validate` reports search-space size; the CLI switches to grid search when `n_trials` exceeds all combinations, or to random search when `n_trials <= n_startup_trials` ([#7](https://github.com/InseeFrLab/auto-tuning-vllm/pull/7))
- **`optimization.log_metrics`** — copy extra benchmark scalars to Optuna user attributes (`metric_<name>`) for dashboard visibility without affecting objectives ([#22](https://github.com/InseeFrLab/auto-tuning-vllm/pull/22))

### Benchmarking (GuideLLM)

- **`benchmark.warmup` / `benchmark.cooldown`** — exclude cold-start and shutdown phases from reported metrics to reduce variance ([#24](https://github.com/InseeFrLab/auto-tuning-vllm/pull/24))
- **`benchmark.rampup`** — linear load ramp-up duration in seconds before reaching target concurrency ([#31](https://github.com/InseeFrLab/auto-tuning-vllm/pull/31))
- **`benchmark.sample_requests`** — control per-request samples in benchmark JSON output (default `0` keeps files small; requires GuideLLM `>= 0.7.1`) ([#27](https://github.com/InseeFrLab/auto-tuning-vllm/pull/27))
- **GuideLLM `>= 0.7.1` migration** — subprocess-based CLI runner, deadlock fix ([#37](https://github.com/InseeFrLab/auto-tuning-vllm/pull/37))
- **Prompt / total token throughput** — parsed from GuideLLM output for objectives and `log_metrics` ([#29](https://github.com/InseeFrLab/auto-tuning-vllm/pull/29))
- **Multimodal VLM benchmarks** — `guidellm_multimodal` profile for multi-image workloads ([#34](https://github.com/InseeFrLab/auto-tuning-vllm/pull/34))
- **Trace replay benchmarks** — `guidellm_trace_replay` profile with optional prewarm ([#38](https://github.com/InseeFrLab/auto-tuning-vllm/pull/38), [#39](https://github.com/InseeFrLab/auto-tuning-vllm/pull/39))
- **Prometheus metrics scraping** — optional vLLM `/metrics` scrape during benchmarks for `log_metrics` ([#35](https://github.com/InseeFrLab/auto-tuning-vllm/pull/35))

### Speculative decoding

- **Speculative decoding search space** — tune EAGLE3 / ngram / Medusa / MTP parameters from YAML ([#40](https://github.com/InseeFrLab/auto-tuning-vllm/pull/40))
- **Limitations documented** — model-specific MTP vs EAGLE3 guidance in configuration reference ([#41](https://github.com/InseeFrLab/auto-tuning-vllm/pull/41))

### Bug fixes

- **Baseline startup timeout** — baseline trials now honor `static_environment_variables.VLLM_STARTUP_TIMEOUT` like optimization trials ([#20](https://github.com/InseeFrLab/auto-tuning-vllm/pull/20))

### Tooling and documentation

- **Optuna Dashboard launcher** — `optuna_dashboard/start_optuna_dashboard.sh` with a sample `study.db` to explore results immediately ([#14](https://github.com/InseeFrLab/auto-tuning-vllm/pull/14))
- **Architecture guide** — [docs/architecture.md](docs/architecture.md) with Mermaid diagrams ([#23](https://github.com/InseeFrLab/auto-tuning-vllm/pull/23))
- **Agent onboarding** — `AGENTS.md`, `.ai/context/`, and `.ai/skills/` for contributors and coding assistants ([#23](https://github.com/InseeFrLab/auto-tuning-vllm/pull/23))

### Testing

- **Unit test suite** under `tests/` — optimization config, custom metrics, baseline behavior, GuideLLM CLI args (no GPU required)
- **CI** runs Ruff, pytest (Python 3.10–3.12), pre-commit, and import smoke tests on every push/PR

### Completed roadmap items

- [x] Local execution backend (Ray optional)
- [x] Custom metric expressions for objectives
- [x] Grid cardinality and sampler auto-switch
- [x] GuideLLM warmup/cooldown, ramp-up, and `sample_requests`
- [x] GuideLLM trace replay and multimodal VLM profiles
- [x] Prometheus metrics scraping for `log_metrics`
- [x] Optuna Dashboard example launcher
- [x] `optimization.log_metrics` for dashboard visibility
- [x] Unit test suite (core + benchmarks)
- [x] CI workflow (lint, pytest matrix, pre-commit)
- [x] Architecture documentation and agent onboarding
- [x] Speculative decoding parameter search space
