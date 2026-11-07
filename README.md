# Auto-Tune vLLM

[![CI](https://github.com/InseeFrLab/auto-tuning-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/InseeFrLab/auto-tuning-vllm/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Automatic performance tuning for vLLM deployments — Optuna-driven search over vLLM server flags, benchmarked with GuideLLM (throughput, TTFT, ITL, trace replay, speculative decoding). Runs on a single GPU node; Ray optional.

## Features

- 🎯 **Flexible backends**: Local execution by default; optional Ray for distributed runs
- 📊 **GuideLLM benchmarking**: Warmup/cooldown, ramp-up, trace replay, multimodal VLM, synthetic or custom datasets
- 🧮 **Rich objectives**: Multi-objective optimization with arithmetic metric expressions
- 🔀 **Smart sampler selection**: Automatic grid or random mode based on search-space size
- ⚡ **Speculative decoding**: Tune EAGLE3, ngram, Medusa, and MTP parameters from YAML
- 📈 **Optuna integration**: User attributes for extra metrics; dashboard launcher included
- 🗄️ **Flexible storage**: SQLite for local use, PostgreSQL for production (optional)
- ⚙️ **Easy configuration**: YAML-based study and parameter configuration
- ✅ **Tested**: Unit tests and CI on Python 3.10–3.12

## Quick Start (5 minutes)

For a detailed starter guide, see the [Quick Start Guide](docs/quick_start.md).

### Installation

Install the base package for local execution. Add the optional `ray` extra only if you want distributed execution.

```bash
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
- [Examples](examples/README.md) — Sample study YAMLs and Python demos
- [Ray Cluster Setup](docs/ray_cluster_setup.md) — For distributed optimization (optional)
- [AGENTS.md](AGENTS.md) — Guide for coding assistants and maintainers

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (for running vLLM)
- SQLite (included) or PostgreSQL (optional)

Core dependencies are installed with `pip install -e .`. Ray is optional and available via `pip install -e ".[ray]"`.

## Manual integration test

The opt-in autotuner integration suite runs complete optimization studies through
`auto-tune-vllm optimize`, using real GuideLLM benchmarks and SQLite/Optuna storage.
Only vLLM is substituted with a small streaming HTTP server inspired by GuideLLM's
mock server. No GPU, model download, or external service is required.

Each scenario runs four sequential trials: concurrent, trace replay, multi-image,
server startup failure, and inference failure. The tests check metrics against
raw reports, objective calculation, failure recovery, request payloads, and process
cleanup between trials. Benchmarks are capped at 20 seconds each; the full suite
can take several minutes including startup and cleanup.

The suite is excluded from default pytest discovery and requires explicit opt-in:

```bash
pip install -e ".[dev]"
AUTOTUNER_E2E=1 pytest -v manual_tests/test_autotuner_e2e.py
```

Run on Linux (process cleanup assertions use `/proc`). To select one scenario, add
`-k trace_replay`, `-k multi_image`, or `-k recovers`.
After the workflow is merged into the default branch, use GitHub Actions:
**Manual autotuner integration tests → Run workflow**. This workflow has no push,
pull request, or scheduled trigger. Its logs and reports are uploaded as artifacts.
These tests cover orchestration and API integration; GPU execution remains a
separate validation step.

## Roadmap

### In progress

- [ ] Expand test coverage (controller, backends, trial lifecycle)
- [ ] Make CI fail strictly on pytest errors (remove `|| true` workaround)
- [ ] Dependency hygiene — pin versions, reduce heavy core dependencies
- [ ] Improve CLI error messages and validation

### Future work

- [ ] Additional benchmark providers beyond GuideLLM
- [ ] Support for alternative inference engines (e.g., SGLang)
- [ ] Better parameter validation against vLLM CLI args
- [ ] `optimization.n_repeats` and `optimization.no_repeat` (PRs [#28](https://github.com/InseeFrLab/auto-tuning-vllm/pull/28), [#32](https://github.com/InseeFrLab/auto-tuning-vllm/pull/32) open)

## Contributing

Contributions are welcome. Priority areas:

1. **Testing** — Extending coverage for controllers, backends, and edge cases
2. **Documentation** — Improving guides and examples
3. **Core stability** — Bug fixes and edge case handling

## Project history

This project originated as a fork of [openshift-psap/auto-tuning-vllm](https://github.com/openshift-psap/auto-tuning-vllm). We are grateful to the original authors for providing the foundation. This fork focuses on simpler single-node deployment (Ray optional), active maintenance, test coverage, and feature expansion for production workloads.

See [CHANGELOG.md](CHANGELOG.md) for the full list of changes since the fork.

## License

Apache License 2.0 — see [LICENSE](LICENSE) file for details.
