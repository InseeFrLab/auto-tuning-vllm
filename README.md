# Auto-Tune vLLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A hyperparameter optimization framework for vLLM serving with local and optional Ray execution backends, built with Optuna.

> **Note: This is a maintained fork**
>
> This repository is a fork of the [openshift-psap/auto-tuning-vllm](https://github.com/openshift-psap/auto-tuning-vllm) project.
> We are grateful to the original authors for providing the foundation this fork builds upon.
> This fork was created to address specific needs in our environment that differ from the original project's scope.

## Why this fork?

This fork was created to adapt the original framework to our specific requirements:

- **Simpler deployment** for single-node scenarios — Ray is optional, not required
- **Testing infrastructure** to support safe evolution of the codebase
- **Active maintenance** for our production use cases (dependency updates, bug fixes)
- **Feature expansion** as needed for our workloads (additional inference engines, benchmark tools)

## What's changed since the fork

The fork baseline is commit [`3c4264d`](https://github.com/InseeFrLab/auto-tuning-vllm/commit/3c4264d93a594e6ae4b2e919741a455f282c2701) (CI setup, pre-commit, unified workflow). Since then:

### Execution

- **Local backend by default** — run on a single machine without Ray (`pip install -e .`); Ray remains available via `pip install -e ".[ray]"` and `--backend ray` ([#3](https://github.com/InseeFrLab/auto-tuning-vllm/issues/3))
- **`max_concurrent_trials` defaults to 1** on the local backend; Python environment flags (`--venv-path`, etc.) are only required for Ray

### Optimization

- **Custom metric expressions** — compose objectives from benchmark identifiers (e.g. `output_tokens_per_second_mean / requests_per_second_median`) instead of a single named metric ([#18](https://github.com/InseeFrLab/auto-tuning-vllm/pull/18))
- **Grid cardinality auto-switch** — `validate` reports search-space size; the CLI switches to grid search when `n_trials` exceeds all combinations, or to random search when `n_trials <= n_startup_trials` ([#7](https://github.com/InseeFrLab/auto-tuning-vllm/pull/7))
- **`optimization.log_metrics`** — copy extra benchmark scalars to Optuna user attributes (`metric_<name>`) for dashboard visibility without affecting objectives ([#22](https://github.com/InseeFrLab/auto-tuning-vllm/pull/22))

### Benchmarking (GuideLLM)

- **`benchmark.warmup` / `benchmark.cooldown`** — exclude cold-start and shutdown phases from reported metrics to reduce variance ([#24](https://github.com/InseeFrLab/auto-tuning-vllm/pull/24))
- **`benchmark.rampup`** — linear load ramp-up duration in seconds before reaching target concurrency
- **`benchmark.sample_requests`** — control per-request samples in benchmark JSON output (default `0` keeps files small; requires GuideLLM `>= 0.5.4`) ([#27](https://github.com/InseeFrLab/auto-tuning-vllm/pull/27))

### Bug fixes

- **Baseline startup timeout** — baseline trials now honor `static_environment_variables.VLLM_STARTUP_TIMEOUT` like optimization trials ([#20](https://github.com/InseeFrLab/auto-tuning-vllm/pull/20))

### Tooling and documentation

- **Optuna Dashboard launcher** — `optuna_dashboard/start_optuna_dashboard.sh` with a sample `study.db` to explore results immediately ([#14](https://github.com/InseeFrLab/auto-tuning-vllm/pull/14))
- **Architecture guide** — [docs/architecture.md](docs/architecture.md) with Mermaid diagrams ([#23](https://github.com/InseeFrLab/auto-tuning-vllm/pull/23))
- **Agent onboarding** — `AGENTS.md`, `.ai/context/`, and `.ai/skills/` for contributors and coding assistants ([#23](https://github.com/InseeFrLab/auto-tuning-vllm/pull/23))

### Testing

- **Unit test suite** under `tests/` — optimization config, custom metrics, baseline behavior, GuideLLM CLI args (no GPU required)
- **CI** runs Ruff, pytest (Python 3.10–3.12), pre-commit, and import smoke tests on every push/PR

## Features

- 🎯 **Flexible backends**: Local execution by default; optional Ray for distributed runs
- 📊 **GuideLLM benchmarking**: Warmup/cooldown, output size control, synthetic or custom datasets
- 🧮 **Rich objectives**: Multi-objective optimization with arithmetic metric expressions
- 🔀 **Smart sampler selection**: Automatic grid or random mode based on search-space size
- 📈 **Optuna integration**: User attributes for extra metrics; dashboard launcher included
- 🗄️ **Flexible storage**: SQLite for local use, PostgreSQL for production (optional)
- ⚙️ **Easy configuration**: YAML-based study and parameter configuration
- ✅ **Tested**: Unit tests and CI on Python 3.10–3.12

## Quick Start (5 minutes)

For a detailed starter guide, see the [Quick Start Guide](docs/quick_start.md).

### Installation

Install the base package for local execution. Add the optional `ray` extra only if you want distributed execution.

```bash
# Clone the maintained fork
git clone https://github.com/InseeFrLab/auto-tuning-vllm.git
cd auto-tuning-vllm

# Basic installation (local execution only)
pip install -e .

# Optional: Install with Ray support for distributed execution
pip install -e ".[ray]"

# Optional: Install with PostgreSQL support
pip install -e ".[postgresql]"
```

### Basic Usage

```bash
# Run optimization study locally (default backend)
auto-tune-vllm optimize --config config.yaml --max-concurrent-trials 2

# Run optimization study on Ray
auto-tune-vllm optimize --config config.yaml --backend ray --venv-path ./venv --max-concurrent-trials 2

# Validate config and preview grid cardinality / sampler auto-switch
auto-tune-vllm validate --config config.yaml

# Resume interrupted study
auto-tune-vllm resume --study-name study_35884

# Stream live logs
auto-tune-vllm logs --study-name study_35884

# Explore results with Optuna Dashboard (sample database included)
./optuna_dashboard/start_optuna_dashboard.sh
```

## Documentation

- [Quick Start Guide](docs/quick_start.md) — Get running in 5 minutes
- [Architecture overview](docs/architecture.md) — How the framework works (diagrams)
- [Configuration Reference](docs/configuration.md) — Complete YAML configuration guide
- [Ray Cluster Setup](docs/ray_cluster_setup.md) — For distributed optimization (optional)
- [AGENTS.md](AGENTS.md) — Guide for coding assistants and maintainers

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (for running vLLM)
- SQLite (included) or PostgreSQL (optional)

Core dependencies are installed with `pip install -e .`. Ray is optional and available via `pip install -e ".[ray]"`.

## Roadmap

This fork is actively being improved.

### Completed since fork

- [x] Local execution backend (Ray optional)
- [x] Custom metric expressions for objectives
- [x] Grid cardinality and sampler auto-switch
- [x] GuideLLM warmup/cooldown and `sample_requests`
- [x] Optuna Dashboard example launcher
- [x] `optimization.log_metrics` for dashboard visibility
- [x] Unit test suite (core + benchmarks)
- [x] CI workflow (lint, pytest matrix, pre-commit)
- [x] Architecture documentation and agent onboarding

### In progress

- [ ] Expand test coverage (controller, backends, trial lifecycle)
- [ ] Make CI fail strictly on pytest errors (remove `|| true` workaround)
- [ ] Dependency hygiene — pin versions, reduce heavy core dependencies
- [ ] Improve CLI error messages and validation

### Future work

- [ ] Support for speculative decoding parameters
- [ ] Additional benchmark providers beyond GuideLLM
- [ ] Support for alternative inference engines (e.g., SGLang)
- [ ] Better parameter validation against vLLM CLI args

## Contributing

This fork welcomes contributions. Priority areas:

1. **Testing** — Extending coverage for controllers, backends, and edge cases
2. **Documentation** — Improving guides and examples
3. **Core stability** — Bug fixes and edge case handling

## License

Apache License 2.0 — see [LICENSE](LICENSE) file for details.
