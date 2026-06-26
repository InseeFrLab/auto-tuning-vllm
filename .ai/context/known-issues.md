# Known issues

Update this file when merging fixes (no separate triage skill).

| Title | Status | Link | Component | Next action |
|-------|--------|------|-----------|-------------|
| GuideLLM + vLLM ≥ 0.20 | open | [#19](https://github.com/InseeFrLab/auto-tuning-vllm/issues/19) | `providers.py`, deps | Merge #17 or document pins |
| GuideLLM version pin | documented | `pyproject.toml` | deps | Pinned to `>=0.6.0,<0.7.0`; multimodal path uses `_guidellm_multimodal_runner` (GuideLLM Python API) |
| GuideLLM + transformers ≥ 5 | open | [#15](https://github.com/InseeFrLab/auto-tuning-vllm/issues/15) | GuideLLM | Reproduce; track upstream |
| Orphan vLLM on parent stop | open | [#2](https://github.com/InseeFrLab/auto-tuning-vllm/issues/2) | `trial_controller.py` | Merge #13 |
| Local backend cleanup | fix pending | [#13](https://github.com/InseeFrLab/auto-tuning-vllm/pull/13) | `backends.py` | Merge PR |
| Baselines consume `n_trials` | fix pending | [#21](https://github.com/InseeFrLab/auto-tuning-vllm/pull/21) | `study_controller.py` | Merge PR |
| CI pytest non-blocking | open | `ci.yml` | CI | Remove `\|\| true` when stable |
| Basic usage tests | open | [#4](https://github.com/InseeFrLab/auto-tuning-vllm/issues/4) | `tests/` | Expand pytest |
| Ray removal / deprecation | open | [#3](https://github.com/InseeFrLab/auto-tuning-vllm/issues/3) | `backends.py` | Legacy only; local path default |

## Code TODOs

| File | Note |
|------|------|
| `cli/main.py` | Sync log streaming |
| `trial_controller.py` | Remove debug health logging |
| `config.py` | Split int/float range parameter types |
