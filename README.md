# Auto-Tune vLLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A hyperparameter optimization framework for vLLM serving, built with Optuna.

> **Note: This is a maintained fork**
>
> This repository is a fork of the [openshift-psap/auto-tuning-vllm](https://github.com/openshift-psap/auto-tuning-vllm) project.
> We are grateful to the original authors for providing the foundation this fork builds upon.
> This fork was created to address specific needs in our environment that differ from the original project's scope.

## Why this fork?

This fork was created to adapt the original framework to our specific requirements:

- **Simpler deployment** for single-node scenarios - Ray is optional, not required
- **Testing infrastructure** to support safe evolution of the codebase
- **Active maintenance** for our production use cases (dependency updates, bug fixes)
- **Feature expansion** as needed for our workloads (additional inference engines, benchmark tools)


## Features

- 🎯 **Flexible Backends**: Run locally (default) or optionally on Ray clusters
- 📊 **Benchmarking**: Built-in GuideLLM support
- 🗄️ **Flexible Storage**: SQLite for local use, PostgreSQL for production (optional)
- ⚙️ **Easy Configuration**: YAML-based study and parameter configuration
- 📈 **Multi-Objective**: Support for throughput vs latency trade-offs

## Quick Start (5 minutes)

For a detailed starter guide, see the [Quick Start Guide](docs/quick_start.md).

### Installation

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
# Run optimization study with local execution (default)
auto-tune-vllm optimize --config config.yaml

# Resume interrupted study
auto-tune-vllm resume --study-name study_35884

# Stream live logs
auto-tune-vllm logs --study-name study_35884
```

## Documentation

- [Quick Start Guide](docs/quick_start.md) - Get running in 5 minutes
- [Configuration Reference](docs/configuration.md) - Complete YAML configuration guide
- [Ray Cluster Setup](docs/ray_cluster_setup.md) - For distributed optimization (optional)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (for running vLLM)
- SQLite (included) or PostgreSQL (optional)

## Roadmap

This fork is actively being improved. Current work in progress:

### Immediate priorities

- [ ] Add comprehensive test suite 
- [ ] Expand CI/CD to run tests, not just linting
- [ ] Dependency hygiene - pin versions, reduce heavy core dependencies
- [ ] Improve CLI error messages and validation

### Future work

- [ ] Support for speculative decoding parameters
- [ ] Additional benchmark providers beyond GuideLLM
- [ ] Support for alternative inference engines (e.g., SGLang)
- [ ] Better parameter validation against vLLM CLI args

## Contributing

This fork welcomes contributions. Priority areas:

1. **Testing** - Adding tests for existing functionality
2. **Documentation** - Improving guides and examples
3. **Core stability** - Bug fixes and edge case handling

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.
