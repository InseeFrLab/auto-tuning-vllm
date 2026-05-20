# Architecture — execution flow

**User diagrams:** [`docs/architecture.md`](../../docs/architecture.md).

## 1. Configuration loading

- YAML study file (`examples/study_config_local_exec.yaml` reference).
- `StudyConfig.from_file()` in `core/config.py` — parse, validate objectives, storage, parameters.
- CLI `optimize` (`cli/main.py`) copies config beside SQLite when `storage_file` is set.

## 2. Study setup

- `StudyController.create_from_config(LocalExecutionBackend, config)`:
  - Optional PostgreSQL (`storage/postgres_utils.py`).
  - `get_storage(config)` → SQLite / PostgreSQL (`storage/utils.py`).
  - Optuna `Study` + sampler (TPE, Grid, NSGA-II, …).
  - `CentralizedLogger` if logging block present.

## 3. Study loop

- Baselines: `_run_baseline_trials` → enqueue + run with default/static params.
- Optimization: `study.ask()` → `TrialConfig` → `backend.submit_trial()` → poll → `study.tell()`.
- Failures: error classification → trial user attrs.
- Optional `optimization.log_metrics` → extra user attrs (PR #22).

## 4. Backend (supported path)

**`LocalExecutionBackend`** (`execution/backends.py`): thread pool, `LocalTrialController`, `poll_trials`, `cleanup_all_trials`.

Legacy: `RayExecutionBackend` exists for upstream compatibility; not the fork focus.

## 5. Single trial

`BaseTrialController.run_trial()` (`execution/trial_controller.py`):

1. Validate imports (vllm, guidellm, optuna).
2. `GuideLLMBenchmark` from `benchmarks/providers.py`.
3. `_start_vllm_server()` → `_wait_for_server_ready()`.
4. State machine: `WAITING_FOR_VLLM` → `RUNNING_BENCHMARK`.
5. Metrics → objectives; `cleanup_resources()` on exit/cancel.

## 6. Storage & logs

- Optuna: `study.storage_file` or `study.database_url`.
- Logs: `logging/manager.py` (file and/or DB).
- Dashboard: `optuna_dashboard/start_optuna_dashboard.sh`.

## 7. Cleanup

- Per trial: kill vLLM + benchmark process group.
- Study end / interrupt: `backend.cleanup_all_trials()`, `shutdown()`.
- Known gap: orphan vLLM if parent killed abruptly (issue #2).
