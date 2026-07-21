# Architecture overview

How **auto-tune-vllm** fits together: YAML configuration, Optuna studies, local trial execution, vLLM serving, and GuideLLM benchmarks.

The default path uses **`LocalExecutionBackend`** on a single machine. An optional Ray backend exists for legacy distributed setups (see [Ray Cluster Setup](ray_cluster_setup.md)).

---

## End-to-end flow

1. You provide a **study YAML** (see [Configuration Reference](configuration.md)).
2. The CLI loads **`StudyConfig`** and creates a **`StudyController`** with an Optuna study (SQLite or PostgreSQL).
3. Optional **baseline trials** run with default vLLM parameters.
4. The optimizer loops: suggest parameters → run trial → record metrics in Optuna.
5. Each trial starts **vLLM**, waits until healthy, runs **GuideLLM**, then cleans up processes.

```mermaid
flowchart TD
  A[Study YAML] --> B["StudyConfig.from_file()"]
  B --> C["auto-tune-vllm optimize / resume"]
  C --> D["StudyController"]
  D --> E["Optuna study + storage"]
  D --> F["LocalExecutionBackend"]

  F --> G[Baseline trials]
  G --> H["Loop: ask → run trial → tell"]

  H --> I[TrialConfig]
  I --> J["TrialController.run_trial()"]

  J --> K[vLLM subprocess]
  K --> L[Server healthy]
  L --> M[GuideLLM benchmark]
  M --> N[Objectives + metrics]
  N --> O["Optuna tell()"]
  O --> P[Cleanup]

  E -.-> O
```

---

## Repository layout

```mermaid
flowchart TB
  subgraph root["Repository"]
    README["README.md"]
    PY["pyproject.toml"]
    PKG["auto_tune_vllm/"]
    DOCS["docs/"]
    EX["examples/"]
    TESTS["tests/"]
    DASH["optuna_dashboard/"]
  end

  subgraph pkg["auto_tune_vllm package"]
    CLI["cli/"]
    CORE["core/"]
    EXEC["execution/"]
    BENCH["benchmarks/"]
    LOG["logging/"]
    UTIL["utils/"]
  end

  PKG --> CLI
  PKG --> CORE
  PKG --> EXEC
  PKG --> BENCH
  PKG --> LOG
  PKG --> UTIL

  CORE --> CFG["config.py — YAML model"]
  CORE --> SC["study_controller.py — Optuna loop"]
  EXEC --> BE["backends.py — local execution"]
  EXEC --> TC["trial_controller.py — vLLM + benchmark"]
  BENCH --> PROV["providers.py — GuideLLM"]
```

| Area | Responsibility |
|------|----------------|
| `cli/` | Commands: `optimize`, `resume`, `logs` |
| `core/` | Config, study orchestration, Optuna storage |
| `execution/` | Backends and per-trial runtime |
| `benchmarks/` | GuideLLM integration |
| `logging/` | Centralized trial logs |
| `examples/` | Study YAMLs, sample datasets, Python demos — see `examples/README.md` |

---

## Study orchestration

```mermaid
sequenceDiagram
  participant User
  participant CLI as CLI
  participant SC as StudyController
  participant BE as LocalExecutionBackend
  participant TC as TrialController
  participant O as Optuna

  User->>CLI: optimize --config study.yaml
  CLI->>SC: create_from_config()
  SC->>O: create or load study
  SC->>SC: run baselines
  loop optimization trials
    SC->>O: ask()
    SC->>BE: submit_trial()
    BE->>TC: run_trial()
    TC-->>BE: TrialResult
    BE-->>SC: poll completed
    SC->>O: tell(metric values)
  end
  CLI->>BE: cleanup / shutdown
```

Concurrency is controlled by **`--max-concurrent-trials`**: several trials may run in parallel, each with its own vLLM process (subject to GPU memory).

---

## Single trial lifecycle

```mermaid
stateDiagram-v2
  [*] --> ValidateEnv
  ValidateEnv --> StartVLLM
  StartVLLM --> WaitReady: process started
  WaitReady --> RunBenchmark: HTTP health OK
  RunBenchmark --> ParseMetrics: GuideLLM finished
  ParseMetrics --> Cleanup
  Cleanup --> [*]

  StartVLLM --> Cleanup: error or cancel
  WaitReady --> Cleanup: timeout or cancel
  RunBenchmark --> Cleanup: error or cancel
```

On failure, error details are stored on the Optuna trial (user attributes) to help the sampler avoid repeating bad configurations.

---

## Module dependencies (simplified)

```mermaid
flowchart LR
  CLI["cli/main.py"] --> CFG["core/config.py"]
  CLI --> SC["core/study_controller.py"]
  CLI --> BE["execution/backends.py"]

  SC --> CFG
  SC --> BE
  BE --> TC["execution/trial_controller.py"]
  TC --> PROV["benchmarks/providers.py"]
  TC --> LOGM["logging/manager.py"]
```

---

## Outputs per study

```mermaid
flowchart LR
  YAML[study_config.yaml] --> DB[(Optuna DB)]
  YAML --> LOGS[Trial log directory]
  TC2[Trial run] --> VLOG[vLLM logs]
  TC2 --> GJSON[GuideLLM results]
  GJSON --> MET[Metrics]
  MET --> DB
```

Typical locations:

- **Optuna database** — path from `study.storage_file` (SQLite) or `study.database_url` (PostgreSQL).
- **Logs** — `logging.file_path` in your YAML.
- **Dashboard** — `./optuna_dashboard/start_optuna_dashboard.sh path/to/study.db`

---

## Related docs

- [Quick Start](quick_start.md)
- [Configuration Reference](configuration.md)
- [Ray Cluster Setup](ray_cluster_setup.md) (optional, legacy distributed path)
