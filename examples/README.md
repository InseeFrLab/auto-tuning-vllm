# Examples

Sample study configurations and small Python demos for auto-tune-vllm.

## Study YAML files

| File | Purpose |
|------|---------|
| [`study_config_local_exec.yaml`](study_config_local_exec.yaml) | **Start here** — full local study with presets, many tunable parameters, and static env vars |
| [`study_config.yaml`](study_config.yaml) | Basic multi-objective study on a small model |
| [`study_config_minimal.yaml`](study_config_minimal.yaml) | Minimal config for quick smoke tests (short benchmark, few trials) |
| [`study_config_speculative_decoding.yaml`](study_config_speculative_decoding.yaml) | EAGLE3 speculative decoding search space (vLLM ≥ 0.20) |
| [`study_config_trace_replay.yaml`](study_config_trace_replay.yaml) | GuideLLM trace replay profile with sample JSONL |
| [`study_config_vlm_multi_image.yaml`](study_config_vlm_multi_image.yaml) | Multi-image VLM benchmark (GuideLLM multimodal) |

Supporting data:

- [`trace_replay/sample.jsonl`](trace_replay/sample.jsonl) — synthetic trace for replay benchmarks
- [`vlm_multi_image/`](vlm_multi_image/) — sample JSONL and images for multimodal benchmarks

## Guides

- [`README_optimization_guide.md`](README_optimization_guide.md) — objectives, presets, custom metric expressions, sampler tips

## Python demos

| Script | Purpose |
|--------|---------|
| [`basic_usage.py`](basic_usage.py) | Run a study via the Python API (`StudyController` + `LocalExecutionBackend`) |
| [`versioned_defaults_demo.py`](versioned_defaults_demo.py) | Explore versioned vLLM CLI defaults |
| [`vllm_cli_demo.py`](vllm_cli_demo.py) | Parse vLLM CLI args and generate parameter schemas |

## Quick commands

```bash
# Validate before launching (no GPU required)
auto-tune-vllm validate --config examples/study_config_local_exec.yaml

# Local optimization
auto-tune-vllm optimize --config examples/study_config_minimal.yaml

# Feature-specific examples (require GPU + vLLM)
auto-tune-vllm optimize --config examples/study_config_speculative_decoding.yaml
auto-tune-vllm optimize --config examples/study_config_trace_replay.yaml
auto-tune-vllm optimize --config examples/study_config_vlm_multi_image.yaml
```

See [docs/quick_start.md](../docs/quick_start.md) for environment setup and [docs/configuration.md](../docs/configuration.md) for the full YAML reference.
