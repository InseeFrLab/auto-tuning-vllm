# Repository map

| Path | Role |
|------|------|
| `auto_tune_vllm/` | Python package |
| `auto_tune_vllm/cli/main.py` | Typer CLI: `optimize`, `resume`, `logs`, `validate` |
| `auto_tune_vllm/core/config.py` | `StudyConfig.from_file()` |
| `auto_tune_vllm/core/study_controller.py` | Optuna loop, baselines, concurrency |
| `auto_tune_vllm/core/speculative.py` | Speculative decoding search space from YAML |
| `auto_tune_vllm/core/trial.py` | `TrialConfig`, `TrialResult` |
| `auto_tune_vllm/core/parameters.py` | Search-space types |
| `auto_tune_vllm/core/storage/` | Optuna storage, PostgreSQL helpers |
| `auto_tune_vllm/execution/backends.py` | `LocalExecutionBackend` (+ legacy Ray class) |
| `auto_tune_vllm/execution/trial_controller.py` | vLLM + GuideLLM + cleanup |
| `auto_tune_vllm/benchmarks/` | `GuideLLMBenchmark`, trace replay, multimodal runners |
| `auto_tune_vllm/logging/` | Centralized trial logs |
| `auto_tune_vllm/utils/` | Grid cardinality, vLLM CLI, versioned defaults |
| `auto_tune_vllm/schemas/vllm_defaults/` | Per-version default YAML |
| `docs/` | `quick_start.md`, `architecture.md`, `configuration.md` |
| `examples/` | Study YAMLs, sample datasets, demos — `examples/README.md` |
| `tests/` | Pytest (`core/`, `execution/`, `benchmarks/`) |
| `optuna_dashboard/` | Dashboard launcher + sample DB |
| `.github/workflows/ci.yml` | Ruff, pytest matrix |
| `pyproject.toml` | Dependencies and tooling |
| `README.md` | Install and usage |

**CLI:** `auto-tune-vllm` → `auto_tune_vllm.cli:main`

**Example YAMLs:** `study_config_local_exec.yaml` (primary), `study_config_minimal.yaml` (smoke), plus feature configs for speculative decoding, trace replay, and multimodal VLM.
