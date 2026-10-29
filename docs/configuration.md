# Configuration Guide

This guide provides detailed explanations of all configuration options available in auto-tune-vllm for optimizing vLLM performance.

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Study Configuration](#study-configuration)
3. [Optimization Configuration](#optimization-configuration)
4. [Benchmark Configuration](#benchmark-configuration)
5. [Logging Configuration](#logging-configuration)
6. [Parameter Configuration](#parameter-configuration)
7. [Baseline Configuration](#baseline-configuration)
8. [Environment Variables](#environment-variables)
9. [Static Parameters](#static-parameters)
10. [Speculative Decoding Configuration](#speculative-decoding-configuration)
11. [Configuration Examples](#configuration-examples)

## Configuration File Structure

Auto-tune-vllm uses YAML configuration files with several main sections. Each section controls a different aspect of the optimization process:

- **`study`**: Defines study metadata, naming, and storage backend
- **`optimization`**: Specifies what metrics to optimize and how
- **`benchmark`**: Configures how performance benchmarks are executed
- **`logging`**: Controls logging output and verbosity (optional)
- **`parameters`**: Defines which vLLM parameters to optimize and their ranges
- **`static_parameters`**: Defines vLLM parameters that remain constant across all trials (optional)
- **`speculative_decoding`**: Configures speculative decoding search space (optional)
- **`baseline`**: Configures baseline performance trials (optional)
- **`static_environment_variables`**: Defines environment variables for all trials (optional)

The basic structure looks like:

```yaml
study:
  # Study identity and storage

optimization:
  # What to optimize and optimization strategy

benchmark:
  # How to measure performance

logging:
  # Where and how to log (optional)

baseline:
  # Baseline trial configuration (optional - enabled by default)

static_parameters:
  # Fixed vLLM parameters for all trials (optional)

static_environment_variables:
  # Environment variables for all trials (optional)

parameters:
  # Which vLLM parameters to tune
```

## Study Configuration

The `study` section controls study identity, naming, and where optimization results are stored. This section is required in every configuration file.

### Study Naming

Studies need unique identifiers to track optimization progress. Auto-tune-vllm provides flexible naming options:

#### `name` (string, optional)
Specifies an explicit study name that must be unique. If a study with this name already exists, the optimization will fail unless you're resuming it. Use this when you need predictable, exact study names.

#### `prefix` (string, optional)
Used for auto-generating unique study names in the format `{prefix}_{timestamp}`. This ensures uniqueness while providing meaningful prefixes. If omitted, defaults to "study".

**Naming Rules:**
- You cannot specify both `name` and `prefix` - choose one approach
- If neither is specified, auto-generates names like `study_123456`
- Explicit names (`name`) fail if the study already exists
- Prefixed names (`prefix`) automatically create unique variants

### Storage Backend

Auto-tune-vllm supports two storage backends for persisting optimization results:

#### `database_url` (string, optional)
PostgreSQL connection URL for production environments. Supports concurrent optimization workers and provides robust persistence. The URL format is: `postgresql://username:password@host:port/database`

#### `storage_file` (string, optional)
Path to SQLite database file for single-machine optimization. Simpler to set up than PostgreSQL but doesn't support concurrent workers. If not specified, defaults to `./optuna_studies/{study_name}/study.db`.

**Storage Rules:**
- You cannot specify both `database_url` and `storage_file`
- If neither is specified, uses SQLite with default file location
- PostgreSQL is recommended for production with multiple workers
- SQLite is suitable for development and single-worker optimization

### Additional Options

#### `study_prefix` (string, optional)
Internal option for advanced study naming scenarios. Generally not needed in user configurations.

#### `use_explicit_name` (boolean, optional)
Internal flag that controls study loading behavior. Automatically set based on your naming choice.

## Optimization Configuration

The `optimization` section defines what performance metrics to optimize and the strategy for finding optimal parameter combinations. This section is required and controls the core optimization behavior.

### Configuration Approaches

Auto-tune-vllm supports three ways to configure optimization, from simple to advanced:

#### Preset-Based Configuration (Recommended for Beginners)

Use `preset` for common optimization scenarios:

##### `preset` (string, optional)
Pre-configured optimization strategies for typical use cases:

- **`"high_throughput"`**: Maximizes token generation rate (output_tokens_per_second)
- **`"low_latency"`**: Minimizes 95th percentile request latency
- **`"balanced"`**: Multi-objective optimization balancing throughput and latency

When using presets, you only need to specify `n_trials`. The preset automatically configures the approach, objectives, and sampler.

#### Structured Configuration (Advanced)

Use `approach` and `objectives` for full control over optimization:

##### `approach` (string, optional)
Defines the optimization strategy:

- **`"single_objective"`**: Optimize one metric only. Best when you have a clear primary goal.
- **`"multi_objective"`**: Optimize multiple metrics simultaneously, finding trade-off solutions.

##### `objectives` (list, required when using approach)
List of optimization objectives. Each objective specifies:

**`metric`** (string, required): The performance metric to optimize. Available metrics:
- `output_tokens_per_second`: Token generation throughput (tokens/sec)
- `prompt_tokens_per_second`: Prompt (input/prefill) token throughput (tokens/sec)
- `tokens_per_second`: Combined prompt + output token throughput (tokens/sec)
- `request_latency`: End-to-end request latency (milliseconds)
- `time_to_first_token_ms`: Time until first token appears (milliseconds)
- `inter_token_latency_ms`: Latency between consecutive tokens (milliseconds)
- `requests_per_second`: Request processing throughput (requests/sec)

**`direction`** (string, required): Optimization direction:
- `"maximize"`: Increase the metric value (for throughput metrics)
- `"minimize"`: Decrease the metric value (for latency metrics)

**`percentile`** (string, optional): Which percentile of the metric to optimize:
- `"median"` or `"p50"`: 50th percentile (most stable, default)
- `"p95"`: 95th percentile (good for SLA optimization)
- `"p90"`: 90th percentile (balanced approach)
- `"p99"`: 99th percentile (extreme tail optimization)

#### Legacy Configuration (Backward Compatibility)

##### `objective` (string, optional)
Legacy single-objective format:
- `"maximize"`: Defaults to maximizing throughput
- `"minimize"`: Defaults to minimizing latency

This format is deprecated but still supported for backward compatibility.

### Optimization Algorithm Settings

#### `sampler` (string, optional)
The optimization algorithm to use. Default is "tpe":

- **`"tpe"`**: Tree-structured Parzen Estimator. Best general-purpose sampler for single-objective optimization.
- **`"nsga2"`**: Non-dominated Sorting Genetic Algorithm II. Recommended for multi-objective optimization.
- **`"botorch"`**: Bayesian Optimization with Torch. Advanced sampler that can be faster but may get stuck in local optima.
- **`"random"`**: Random sampling. Useful for baselines and quick testing only.
- **`"grid"`**: Exhaustive grid search. Tests all parameter combinations (use with small parameter spaces only).

#### `n_trials` (integer, required)
Number of optimization trials to run. Each trial tests one parameter combination:
- **Development**: 10-50 trials for quick testing
- **Production**: 100-500 trials for thorough optimization
- **Multi-objective**: Typically needs 2x more trials than single-objective

#### `n_startup_trials` (integer, optional)
Number of random trials to run before starting the main sampler algorithm. Only supported by some samplers (TPE, BoTorch). Helps initialize the sampler with diverse data points.

#### `no_repeat` (boolean, optional)
When `true` (default), the study skips any trial whose exact parameter combination has already been used in the study (completed, running, failed, or pruned optimization trials). Duplicate suggestions are rejected immediately and the sampler is asked again; the skipped suggestion is recorded in Optuna as **PRUNED** with user attribute `skip_reason: duplicate_parameters` (it does not count toward `n_trials` and is not marked as a failure). If the search space is exhausted (e.g. grid search or a converged sampler), the study stops early after 15 consecutive duplicate suggestions. Set to `false` to allow re-running identical configurations that already completed successfully (failed/pruned combinations remain skipped).

#### `log_metrics` (list of strings, optional)
Extra benchmark scalars to copy onto each **Optuna trial** as [user attributes](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.set_user_attr), mainly so tools like **Optuna Dashboard** can plot or filter on them alongside objectives.

- **Semantics**: This does **not** change the optimization objective. It only stores additional numbers on the trial record after a successful benchmark.
- **Identifiers**: Each list entry must be either:
  - a GuideLLM metric id in the same `<metric>_<percentile>` form as in objective expressions (see **`objectives`** above), e.g. `request_latency_p95`, `output_tokens_per_second_median`; or
  - a vLLM runtime metric id `vllm_<prometheus_name>_<stat>`, where `<prometheus_name>` is the Prometheus metric name **without** the `vllm:` namespace prefix (e.g. `kv_cache_usage_perc`, `num_preemptions_total`) and `<stat>` is one of `mean`, `median`, `p90`, `p95`, `p99`, `min`, `max`, `std_dev`.
- **Storage**: For each configured name, the runner writes `trial.set_user_attr("metric_<name>", float_value)` using the value from the trial’s `detailed_metrics`. If a name is missing from `detailed_metrics`, or the value cannot be converted to a float, a warning is logged and that attribute is skipped.
- **Trials**: Applied to **optimization** and **baseline** trials when the run succeeds and detailed metrics are present. Omitted or unset `log_metrics` is treated as an empty list.
- **vLLM scraping**: Metrics whose ids start with `vllm_` are populated by scraping the vLLM server `/metrics` endpoint during the benchmark (see **`metrics_scraping.vllm`** below). No metric names are hardcoded: any name vLLM exposes can be requested. Scraping is **disabled** when `log_metrics` contains no `vllm_*` entries.

Example:

```yaml
optimization:
  preset: "balanced"
  n_trials: 50
  log_metrics:
    - "prompt_tokens_per_second_median"
    - "inter_token_latency_ms_p95"
    - "time_to_first_token_ms_median"
    - "vllm_kv_cache_usage_perc_p90"
```

Example with vLLM runtime metrics:

```yaml
optimization:
  preset: "balanced"
  n_trials: 50
  log_metrics:
    - "request_latency_p95"
    - "vllm_num_preemptions_total_mean"
metrics_scraping:
  vllm:
    scrape_interval_seconds: 10
    align_with_benchmark_window: true
```

### Preset Configurations Explained

#### High Throughput Preset
Optimizes for maximum token generation speed using median throughput. Best for batch processing and high-volume serving where latency is less critical.

#### Low Latency Preset
Optimizes for minimal 95th percentile request latency. Best for interactive applications where response time matters more than maximum throughput.

#### Balanced Preset
Multi-objective optimization finding the best trade-offs between throughput and latency. Provides a Pareto front of solutions rather than a single optimum. Best when you need to balance performance characteristics.

## Benchmark Configuration

The `benchmark` section controls how performance measurements are conducted. This section is required and defines the workload used to evaluate different vLLM parameter combinations.

### Core Benchmark Settings

#### `benchmark_type` (string, optional)
The benchmarking framework and workload mode to use:

| Value | GuideLLM profile | Role |
|-------|------------------|------|
| `"guidellm"` (default) | **Concurrent** (`kind=concurrent`) | Fixed concurrency load via `rate` → `streams` |
| `"guidellm_multimodal"` | Concurrent | Multi-image VLM JSONL workloads |
| `"guidellm_trace_replay"` | **Replay** (`kind=replay`) | Trace file replay via `rate` / `time_scale` |

`benchmark_type: "guidellm"` is **not** a generic catch-all benchmark mode: it specifically runs GuideLLM with the **concurrent profile** (omitted `benchmark.profile` defaults to `kind: concurrent`). Trace replay and multimodal workloads use separate providers and profiles.

Optional explicit profile override via `benchmark.profile`:

```yaml
benchmark:
  benchmark_type: "guidellm"
  profile:
    kind: concurrent   # default when omitted
  rate: 50             # concurrent streams
```

#### `model` (string, required)
The HuggingFace model identifier to benchmark. This should match the model you plan to serve in production. Examples:
- `"facebook/opt-125m"` (small model for testing)
- `"Qwen/Qwen3-30B-A3B-FP8"` (production model)
- `"microsoft/DialoGPT-medium"` (conversational model)

#### `max_seconds` (integer, optional)
Duration in seconds for each benchmark run. Longer benchmarks provide more accurate measurements but take more time. Typical values:
- **Development**: 60-120 seconds for quick feedback
- **Production**: 300-600 seconds for stable measurements
Default: 300 seconds

#### `request_format` (string, optional)
OpenAI-compatible endpoint path targeted by GuideLLM's `openai_http` backend. Applies to all built-in benchmark providers (`guidellm`, `guidellm_trace_replay`, `guidellm_multimodal`).

| Value | Endpoint |
|-------|----------|
| `"/v1/completions"` | Text completions (no chat template) |
| `"/v1/chat/completions"` | Chat completions (default when omitted) |
| `"/v1/responses"` | OpenAI Responses API |

When omitted, GuideLLM defaults to `/v1/chat/completions` and the emitted CLI command does not include `request_format` (unchanged behavior for existing configs).

**Note:** `/v1/completions` does not apply the chat template, so prompt token counts and TTFT are not directly comparable with `/v1/chat/completions`. Keep `request_format` constant within a study; do not compare results across studies that use different endpoints.

### Workload Configuration

The benchmark needs a workload to test against. Auto-tune-vllm supports both synthetic data generation and real datasets.

#### Synthetic Data (Recommended)

##### `dataset` (null)
Set to `null` to use synthetic data generation. This creates artificial prompts and responses based on your specifications.

##### `prompt_tokens` (integer, optional)
Base number of tokens in generated prompts. Default: 1000

##### `output_tokens` (integer, optional)
Base number of tokens in generated responses. Default: 1000

##### Advanced Synthetic Data Distribution

For more realistic workloads, control the distribution of prompt and output lengths:

##### `prompt_tokens_stdev` (integer, optional)
Standard deviation for prompt token lengths. Creates variation around the base `prompt_tokens` value. Default: 128

##### `prompt_tokens_min` (integer, optional)
Minimum prompt length in tokens. Default: 256

##### `prompt_tokens_max` (integer, optional)
Maximum prompt length in tokens. Default: 1024

##### `output_tokens_stdev` (integer, optional)
Standard deviation for output token lengths. Default: 512

##### `output_tokens_min` (integer, optional)
Minimum output length in tokens. Default: 1024

##### `output_tokens_max` (integer, optional)
Maximum output length in tokens. Default: 3072

#### Real Dataset Configuration

##### `dataset` (string)
Path to a real dataset file or HuggingFace dataset identifier. Supported formats:
- Local JSONL files: `"path/to/dataset.jsonl"`
- HuggingFace datasets: `"hf://dataset_name"` (prefix with `hf://`)

When using real datasets, the `prompt_tokens` and `output_tokens` settings are ignored.

#### Multimodal Datasets (VLM / multi-image)

Use a **separate benchmark provider** so the concurrent-profile `guidellm` path stays unchanged:

```yaml
benchmark:
  benchmark_type: "guidellm_multimodal"  # not "guidellm"
  model: "Qwen/Qwen2-VL-2B-Instruct"
  dataset: "/path/to/data.jsonl"
  ...
```

For vision models served via vLLM, point `dataset` at a JSONL file where each line has a text prompt and an `image` field that may be a single path/URL string or a list of paths:

```jsonl
{"prompt": "Describe this image", "image": "images_test/9.png"}
{"prompt": "Compare these images", "image": ["images_test/9.png", "images_test/11.png"]}
```

`guidellm_multimodal` launches `auto_tune_vllm.benchmarks._guidellm_multimodal_runner`, which calls GuideLLM >= 0.7.1 through the Python API (`benchmark_generative_text`). The runner resolves the custom `flatten_image_lists` preprocessor locally and passes remaining preprocessors (e.g. `encode_media`) to GuideLLM by name. Setting `data_preprocessors` overrides GuideLLM defaults, so list both preprocessors explicitly.

| Field | Description |
|-------|-------------|
| `benchmark_type` | Must be `"guidellm_multimodal"` for multi-image JSONL workloads |
| `request_format` | OpenAI endpoint path, e.g. `"/v1/chat/completions"` for VLMs |
| `data_column_mapper` | Maps JSONL columns to GuideLLM fields as a flat mapping, e.g. `text_column: prompt`, `image_column: image` |
| `data_preprocessors` | Ordered list, e.g. `["flatten_image_lists", "encode_media"]` |
| `data_preprocessors_kwargs` | Arguments passed to preprocessors, e.g. `base_dirs` for resolving relative image paths (used as provided; for local JSONL datasets the runner also searches `Path.cwd()` and the dataset file's parent directory) |
| `data_finalizer` | Optional finalizer (GuideLLM default: `"generative"`) |
| `data_args` | Optional HuggingFace `load_dataset` arguments when using `hf://` datasets |

Set vLLM static parameters so the server accepts multiple images per request, for example `limit_mm_per_prompt: '{"image": 4}'`.

Tune multimodal server memory with `mm_processor_cache_gb` in the `parameters` section (passed to vLLM as `--mm-processor-cache-gb`). Example:

```yaml
parameters:
  mm_processor_cache_gb:
    enabled: true
    options: [0, 2, 4, 8]
```

See [examples/study_config_vlm_multi_image.yaml](../examples/study_config_vlm_multi_image.yaml) and [examples/vlm_multi_image/data.jsonl](../examples/vlm_multi_image/data.jsonl) for a full Qwen2-VL-2B-Instruct example.

#### Trace Replay Benchmarking

Use a **separate benchmark provider** with GuideLLM's **replay profile** (not the concurrent profile used by `benchmark_type: "guidellm"`) to replay production or synthetic trace files (GuideLLM >= 0.7.1):

```yaml
benchmark:
  benchmark_type: "guidellm_trace_replay"
  model: "Qwen/Qwen2.5-0.5B-Instruct"
  max_seconds: 300
  rate: 1
  dataset: "examples/trace_replay/sample.jsonl"
  profile:
    trace_format: trace_synthetic
```

Each trace row is a JSON object with timestamp and token lengths. GuideLLM sorts rows by timestamp and schedules requests at the recorded inter-arrival intervals:

```jsonl
{"timestamp": 0.0, "input_length": 256, "output_length": 128}
{"timestamp": 0.5, "input_length": 512, "output_length": 96}
```

Prompts are generated synthetically to match `input_length`; the replay profile does not use literal prompt text from the trace file.

| Field | Description |
|-------|-------------|
| `benchmark_type` | Must be `"guidellm_trace_replay"` — selects the replay profile automatically |
| `dataset` | Path to a JSONL, JSON, CSV, or Parquet trace file (required) |
| `rate` | Replay time scale when `profile.time_scale` is omitted (`1.0` = real-time by default, `2.0` = half speed, `0.5` = double speed). Supports fractional values. |
| `profile.trace_format` | Trace deserializer: `"trace_synthetic"` (default) or `"mooncake"` |
| `profile.time_scale` | Optional explicit time scale; overrides `rate` when set |
| `profile.data_samples` | Optional cap on trace rows loaded via `--data-loader kind=pytorch,samples=N` |
| `profile.timestamp_column` | Trace timestamp column name (default: `"timestamp"`) |
| `profile.prompt_tokens_column` | Prompt token count column (default: `"input_length"`) |
| `profile.output_tokens_column` | Output token count column (default: `"output_length"`) |
| `profile.hash_ids_column` | Mooncake-only hash ID list column (default: `"hash_ids"`) |
| `profile.hash_id_block_size` | Mooncake-only tokens per hash ID block (default: `512`) |

`warmup`, `cooldown`, and `rampup` are not supported with the replay profile. Replay speed is controlled by `benchmark.rate` (or `profile.time_scale`), not by `baseline.concurrency_levels` — see [Baseline Configuration](#baseline-configuration).

#### Prewarm (kernel warmup)

Optional pre-run phase for `guidellm_trace_replay` only. Distinct from `benchmark.warmup`: prewarm runs **before** the replay benchmark as a separate short GuideLLM subprocess, so vLLM kernel autotuners (CUDA graphs, torch.compile, etc.) can settle before timed replay requests.

When `benchmark.prewarm` is set, auto-tune-vllm:

1. Derives mean/stdev of prompt and output token lengths from the local trace file when possible (using `profile.prompt_tokens_column` and `profile.output_tokens_column`). If the dataset is remote (`hf://`) or cannot be read locally, a warning is logged and `benchmark.prompt_tokens` / `benchmark.output_tokens` (plus optional stdev fields) are used instead.
2. Launches a concurrent GuideLLM run (`kind=concurrent`) for `duration` seconds with `concurrency` streams.
3. Uses `kind=synthetic_text` data whose token mean/stdev match those statistics.
4. Discards prewarm results and continues with the normal trace replay run only when prewarm succeeds.

If prewarm fails (non-zero exit, timeout, or cancellation), the trial is marked as failed and trace replay does **not** run.

```yaml
benchmark:
  benchmark_type: "guidellm_trace_replay"
  dataset: "examples/trace_replay/sample.jsonl"
  prewarm:
    duration: 30      # seconds for the prewarm run
    concurrency: 4    # concurrent streams during prewarm
```

| Field | Description |
|-------|-------------|
| `prewarm.duration` | Wall-clock seconds for the prewarm subprocess (`> 0`) |
| `prewarm.concurrency` | Concurrent streams during prewarm (`> 0`) |

Token statistics are derived from the trace file when it is available locally; otherwise defaults from `benchmark.prompt_tokens` / `benchmark.output_tokens` apply.

See [examples/study_config_trace_replay.yaml](../examples/study_config_trace_replay.yaml) and [examples/trace_replay/sample.jsonl](../examples/trace_replay/sample.jsonl) for a full example.

### Load Configuration

#### `rate` (float, optional)
With `benchmark_type: "guidellm"` (concurrent profile), number of concurrent requests to maintain (`rate` → GuideLLM `streams`). This simulates realistic server load:
- **Light load**: 10-20 requests
- **Moderate load**: 50-100 requests
- **Heavy load**: 200+ requests
Default when omitted: **50** for concurrent profiles (`guidellm`, `guidellm_multimodal`); **1.0** for `guidellm_trace_replay` (real-time replay).

For `guidellm_trace_replay`, `rate` is reused as the replay profile's `time_scale` when `profile.time_scale` is not set.

### Advanced Options

#### `processor` (string, optional)
Separate model for request processing if different from the served model. Rarely needed - defaults to the same value as `model`.

#### `warmup` (number, optional)
GuideLLM warmup period, excluded from benchmark metrics. Reduces variance from cold GPU/KV cache at the start of each run.

- **Fraction** (recommended): value strictly between `0` and `1`, e.g. `0.1` = first 10% of the run used for warmup only.
- **Absolute**: value `>= 1` = fixed number of requests or seconds (see [GuideLLM](https://github.com/vllm-project/guidellm) docs).

Omit or set to `null` to disable (GuideLLM default). Applies to optimization and baseline trials. Requires [GuideLLM](https://github.com/vllm-project/guidellm) `>= 0.7.1` (warmup/cooldown are passed via the concurrent profile in `guidellm run`).

When both `warmup` and `cooldown` are fractional, their sum must stay below `1` so a measured window remains.

**Measured duration:** warmup and cooldown are taken from the same `max_seconds` budget (they do not extend wall-clock time). With `max_seconds: 300`, `warmup: 0.1`, and `cooldown: 0.1`, roughly 240 seconds contribute to reported metrics—increase `max_seconds` if you need a longer steady-state phase.

#### `cooldown` (number, optional)
GuideLLM cooldown period at the end of the run, also excluded from metrics. Same format as `warmup` (e.g. `0.1` for the last 10%).

#### `rampup` (number, optional)
GuideLLM ramp-up duration in seconds. Requests are spread linearly from zero up to the target concurrency (`rate`) over this period at the start of the benchmark.

- **Seconds**: e.g. `rampup: 10` = 10 seconds to reach full concurrency.
- Omit or set to `null` to disable (GuideLLM default).

Unlike `warmup`, ramp-up requests are included in reported metrics — it controls how load increases, not which phase is measured. See [GuideLLM benchmark docs](https://github.com/vllm-project/guidellm/blob/main/docs/getting-started/benchmark.md).

#### `sample_requests` (integer, optional)
Maximum number of detailed request samples stored in GuideLLM benchmark JSON output (`sample_size` in the generative metrics config). Default: `0` (no per-request samples; keeps result files small). Set to a positive value when you need request-level timings for debugging or deeper analysis. Requires [GuideLLM](https://github.com/vllm-project/guidellm) `>= 0.7.1`.

Example:

```yaml
benchmark:
  model: "Qwen/Qwen2.5-0.5B-Instruct"
  max_seconds: 300
  rate: 16
  warmup: 0.1
  cooldown: 0.1
  rampup: 10
  # sample_requests: 20  # optional; default 0
```

## Logging Configuration

The `logging` section controls where and how detailed logging information is recorded. This section is optional - if omitted, logs are only displayed on the console.

#### `file_path` (string, optional)
Directory path where log files should be written. Auto-tune-vllm will create log files in this directory for:
- Optimization progress and results
- vLLM server output
- Benchmark execution details
- Error and debugging information

If not specified, no log files are created and all output goes to the console only.

#### `log_level` (string, optional)
Controls the verbosity of logging output. Available levels:
- **`"DEBUG"`**: Detailed debugging information, including parameter values and internal state
- **`"INFO"`**: General information about optimization progress (default)
- **`"WARNING"`**: Only warnings and errors
- **`"ERROR"`**: Only error messages

**Note**: File logging is recommended for production optimization runs to preserve detailed results and troubleshooting information.

## Metrics scraping configuration

The optional `metrics_scraping` section controls vLLM `/metrics` scraping during trials. It is only active when `optimization.log_metrics` includes at least one id starting with `vllm_`.

### `metrics_scraping.vllm`

Scrapes the vLLM Prometheus `/metrics` endpoint while the GuideLLM benchmark runs.

#### `scrape_interval_seconds` (number, optional)
Seconds between scrapes. Default: `10`.

#### `align_with_benchmark_window` (boolean, optional)
When `true` (default), scraped samples are filtered to the same measurement window as GuideLLM reported metrics, using `benchmark.warmup`, `benchmark.cooldown`, and `benchmark.max_seconds`. Warmup/cooldown values in `(0, 1)` are treated as fractions of `max_seconds`; values `>= 1` are absolute seconds.

### vLLM metric ids in `log_metrics`

Use the form `vllm_<prometheus_name>_<stat>`:

- `<prometheus_name>`: name as exposed by vLLM on `/metrics`, without the `vllm:` namespace prefix.
- `<stat>`: `mean`, `median`, `p90`, `p95`, `p99`, `min`, `max`, or `std_dev`.

No metric list is enforced at config load time; missing metrics at runtime produce warnings and the corresponding user attribute is skipped.

### Aggregation behavior

| Prometheus type | Aggregation |
|-----------------|-------------|
| **Gauge** | The requested stat is computed over the in-window time series (multi-label series are collapsed with **max** per scrape). |
| **Counter** | `mean` → rate `(last - first) / window_duration`; all other stats → window delta `(last - first)`. |

## Parameter Configuration

The `parameters` section defines which vLLM server parameters should be optimized and the ranges or options to explore for each parameter. This section controls the parameter space that the optimization algorithm searches through.

### Parameter Structure

Each parameter has a common structure with required and optional fields:

#### `enabled` (boolean, required)
Controls whether this parameter should be included in optimization:
- `true`: Include this parameter in optimization
- `false`: Skip this parameter (use vLLM defaults)

#### Parameter Type-Specific Configuration

Auto-tune-vllm supports three types of parameters, each with different configuration options:

### Range Parameters

Range parameters define continuous or discrete numeric ranges to optimize over. Used for parameters like memory utilization or batch sizes.

#### Configuration Fields:
- **`min`** (number, optional): Minimum value to test. Uses schema default if not specified.
- **`max`** (number, optional): Maximum value to test. Uses schema default if not specified.
- **`step`** (number, optional): Step size between values. Uses schema default if not specified.

The optimizer will test values between `min` and `max` in increments of `step`. For floating-point parameters, the step can be a decimal value.

### List Parameters

List parameters define categorical choices from a fixed set of options. Used for parameters like data types or scheduling policies.

#### Configuration Fields:
- **`options`** (array, optional): List of valid values to test. Uses schema defaults if not specified.

The optimizer will test each value in the options list.

### Boolean Parameters

Boolean parameters test both true and false values. Used for feature flags and enable/disable options.

#### Configuration Fields:
No additional configuration needed - automatically tests both `true` and `false` values.

### Available Parameters

Auto-tune-vllm supports optimization of 27+ vLLM server parameters. For detailed descriptions of each parameter, run:

```bash
vllm serve --help
```

The available parameters include:

**Memory & Cache**: `gpu_memory_utilization`, `swap_space`, `block_size`, `kv_cache_dtype`
**Model & Computation**: `dtype`, `enforce_eager`, `max_seq_len_to_capture`, `compilation_config`
**Batching & Scheduling**: `max_num_batched_tokens`, `scheduling_policy`, `scheduler_delay_factor`, `max_num_partial_prefills`
**CUDA Graphs**: `cuda_graph_sizes`, `long_prefill_token_threshold`
**Parallelism**: `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`
**Caching**: `enable_prefix_caching`

### Parameter Configuration Guidelines

#### Schema Defaults
If you don't specify `min`, `max`, `step`, or `options` for a parameter, auto-tune-vllm uses built-in schema defaults based on typical vLLM usage patterns.

#### Important Notes
- **`gpu_memory_utilization`**: Should not go below 0.9 as it significantly reduces performance
- **Parallelism parameters**: `tensor_parallel_size * pipeline_parallel_size * data_parallel_size` must not exceed your GPU count

#### Performance Impact
Focus on high-impact parameters first:
- **High impact**: `gpu_memory_utilization`, `max_num_batched_tokens`, `kv_cache_dtype`
- **Medium impact**: `block_size`, `dtype`
- **Low impact**: `scheduler_delay_factor`, `compilation_config`

#### Performance Notes for High-Impact Parameters

**`kv_cache_dtype`**: FP8 typically provides 2x+ throughput improvement over "auto". Consider using FP8 if your model supports it.

**`gpu_memory_utilization`**: Values below 0.9 significantly reduce performance. Start optimization around 0.9-0.95 range.

**`max_num_batched_tokens`**: Higher values generally improve throughput but increase memory usage. Balance with `gpu_memory_utilization`.

## Baseline Configuration

Baseline trials establish performance baselines using pure vLLM defaults before running optimization. This helps measure optimization improvements and provides reference performance data.

**Default Behavior**: Baseline trials are **enabled by default**. If no baseline configuration is provided, the system automatically creates a baseline trial using the benchmark's configured rate as the concurrency level.

### Baseline Configuration Fields

```yaml
baseline:
  enabled: true  # Default: true (can be set to false to disable)
  concurrency_levels: [50, 100, 150]  # Optional: defaults to benchmark.rate if not specified
  parameters:  # Optional: Custom parameters for baseline trials - Will override static_parameters if defined in both
    tensor_parallel_size: 1
    max_model_len: 16384
```

#### Configuration Fields:
- **`enabled`** (boolean, default: `true`): Enable baseline trials. Set to `false` to disable.
- **`concurrency_levels`** (array, optional): For concurrent benchmarks (`guidellm`, `guidellm_multimodal`), list of concurrency levels to run as separate baseline trials. If not specified, defaults to `[benchmark.rate]`. For trace replay (`guidellm_trace_replay`), defaults to `[1]` (one baseline trial). **This field does not set replay speed** — replay `time_scale` comes from `benchmark.rate` / `profile.time_scale` instead. Baseline trials always reuse the benchmark section as-is; `concurrency_levels` only controls how many baseline trials are enqueued and affects `--max-num-seqs` when a level exceeds 256.
- **`parameters`** (dict, optional): Custom vLLM parameters to use for all baseline trials - Will override static_parameters if defined in both

### Baseline Trial Behavior

By default, baseline trials use **pure vLLM defaults** with only one parameter modified:
- `--max-num-seqs` is set to the concurrency level being tested (when concurrency > 256)
- All other parameters use vLLM's built-in defaults (not hardcoded values)

Optionally, you can specify custom parameters in the `parameters` field to set specific vLLM arguments for baseline trials. This is useful when you need consistent baseline parameters across all runs (e.g., `tensor_parallel_size`, `max_model_len`).

This provides clean baseline performance data for comparison with optimized configurations.

### Disabling Baseline Trials

To disable baseline trials entirely, explicitly set `enabled: false`:

```yaml
baseline:
  enabled: false  # Disable baseline trials
```

**Note**: If you completely omit the baseline section from your config, baseline trials will still run by default using your benchmark's configured rate.

### Example Configuration

```yaml
baseline:
  enabled: true
  concurrency_levels: [50, 100]

optimization:
  preset: "high_throughput"
  n_trials: 50

parameters:
  gpu_memory_utilization:
    enabled: true
    min: 0.88
    max: 0.95

  kv_cache_dtype:
    enabled: true
    options: ["auto", "fp8"]
    # PERFORMANCE NOTE: fp8 typically provides 2x+ throughput improvement
```

This configuration will:
1. **First** run baseline trials at concurrency 50 and 100 using pure vLLM defaults
2. **Then** run 50 optimization trials to find the best parameter settings
3. **Finally** compare the best optimized performance against the baseline

## Environment Variables

Auto-tune-vllm configuration files support environment variable expansion, allowing you to externalize sensitive information and environment-specific settings.

### Variable Expansion Syntax

Environment variables are referenced using `${VARIABLE_NAME}` syntax within YAML values:

#### Basic Expansion: `${VAR_NAME}`
Expands to the environment variable value, or an empty string if the variable is not set.

#### Default Values: `${VAR_NAME:-default_value}`
Expands to the environment variable value if set, otherwise uses the provided default value.

### Usage Patterns

#### Database Credentials
Keep database passwords out of configuration files:
```yaml
study:
  database_url: "postgresql://user:${POSTGRES_PASSWORD}@localhost:5432/optuna"
```

#### Environment-Specific Settings
Use different log levels per environment:
```yaml
logging:
  log_level: "${LOG_LEVEL:-INFO}"
```

#### Model Configuration
Allow model selection via environment:
```yaml
benchmark:
  model: "${MODEL_NAME:-facebook/opt-125m}"
```

### Common Environment Variables

These environment variables are commonly used with auto-tune-vllm:

- **`POSTGRES_PASSWORD`**: Database password for PostgreSQL storage
- **`DATABASE_URL`**: Complete database connection URL
- **`LOG_LEVEL`**: Logging verbosity (DEBUG/INFO/WARNING/ERROR)
- **`LOG_PATH`**: Directory for log files
- **`MODEL_NAME`**: HuggingFace model identifier
- **`BENCHMARK_DURATION`**: Benchmark runtime in seconds

Set these in your shell or deployment environment before running auto-tune-vllm.

## vLLM Environment Variables

Auto-tune-vllm supports passing environment variables to vLLM processes in two ways:

- **Environment parameters**: Add `type: environment` in the `parameters` section (list-only options required)
- **Static environment variables**: Use `static_environment_variables` section for consistent key-value pairs

```yaml
parameters:
  VLLM_ATTENTION_BACKEND:
    enabled: true
    type: environment
    options: ["FLASH_ATTN", "XFORMERS"]

static_environment_variables:
  VLLM_CACHE_ROOT: "/tmp/vllm_cache"
  VLLM_DEBUG: "0"
```

## Static Parameters

Static parameters are vLLM command-line arguments that remain constant across **all trials** (both baseline and optimization trials). Use this section when you need specific vLLM settings applied consistently without optimizing them.

### When to Use Static Parameters

Static parameters are useful for:
- **Hardware constraints**: Fixed parallelism settings like `tensor_parallel_size` based on your GPU setup
- **Model requirements**: Parameters like `max_model_len` that must match your model's specifications
- **Consistency requirements**: Settings that should remain constant for fair comparisons across trials

### Configuration

```yaml
static_parameters:
  tensor_parallel_size: 1       # Use 1 GPU for all trials
  max_model_len: 16384          # Fixed context length
  enable_chunked_prefill: true  # Always enable chunked prefill
```

### How Static Parameters Work

1. Static parameters are applied to **every trial** (baseline and optimization)
2. They can be overridden by baseline-specific parameters or optimization parameters if needed
3. The merge order is: `static_parameters` → `baseline.parameters` (for baseline trials) → optimized parameters (for optimization trials)

### Example with Optimization

```yaml
# Fixed parameters for all trials
static_parameters:
  tensor_parallel_size: 2
  max_model_len: 8192

# Parameters to optimize (will be added to static parameters)
parameters:
  gpu_memory_utilization:
    enabled: true
    min: 0.85
    max: 0.95

  max_num_batched_tokens:
    enabled: true
    options: [2048, 4096, 8192]
```

In this example, all trials will use:
- Fixed: `tensor_parallel_size=2`, `max_model_len=8192`
- Optimized: varying `gpu_memory_utilization` and `max_num_batched_tokens`

### Static Parameters vs Baseline Parameters

- **`static_parameters`**: Applied to ALL trials (baseline + optimization)
- **`baseline.parameters`**: Applied ONLY to baseline trials (on top of static_parameters)

```yaml
static_parameters:
  tensor_parallel_size: 1  # All trials use this

baseline:
  enabled: true
  concurrency_levels: [50]
  parameters:
    gpu_memory_utilization: 0.9  # Only baseline trials use this
```

## Speculative Decoding Configuration

The `speculative_decoding` section enables tuning vLLM speculative decoding via **synthetic rejection sampling** (`rejection_sample_method: synthetic`). This lets you benchmark speculative decoding performance without a real draft model acceptance distribution.

**Requirements:** vLLM **>= 0.20**. The optimizer checks the installed version at study startup and fails with a clear error if the version is too old.

**Reference:** [vLLM speculative decoding docs](https://docs.vllm.ai/en/stable/features/speculative_decoding/)

### What Gets Optimized

When `speculative_decoding.enabled: true`, each optimization trial may:

1. Run with speculative decoding **off** (when `allow_disabled: true`)
2. Run with speculative decoding **on**, tuning:
   - **Method**: `mtp`, `qwen3_next_mtp`, `eagle`, `eagle3`, or `dflash`
   - **k** (`num_speculative_tokens`): fixed via `static_parameters`, tuned via `enabled`/`options`, or defaults to `len(rates)` (see below)
   - **Draft model** settings inside `--speculative-config`: `draft_tensor_parallel_size`, `max_model_len`
   - **Target model** settings via the normal `parameters:` block: `tensor_parallel_size`, `max_model_len`

Baseline trials never use speculative decoding (natural reference without speculation).

### Method-to-Model Mapping

Each entry in `methods` maps a speculation method to its draft or auxiliary model:

```yaml
speculative_decoding:
  methods:
    - method: eagle3
      model: "RedHatAI/Qwen3-8B-speculator.eagle3"
```

**EAGLE3 / EAGLE / dflash:** set `model` to the external speculator or draft checkpoint (see the model card or vLLM docs).

**MTP (native multi-token prediction):** only for targets that ship MTP support in vLLM (e.g. Qwen3-Next, Qwen3.5, DeepSeek, Gemma 4). **Qwen/Qwen3-8B does not support native MTP** — use EAGLE3 instead, as in [`examples/study_config_speculative_decoding.yaml`](../examples/study_config_speculative_decoding.yaml).

**Qwen3-Next** uses the dedicated vLLM method `qwen3_next_mtp` (MTP head built into the target; `model` in `methods` is optional):

```yaml
benchmark:
  model: "Qwen/Qwen3-Next-80B-A3B-Instruct"
speculative_decoding:
  methods:
    - method: qwen3_next_mtp
```

When MTP applies for other families, set `model` as follows:

| Case | `benchmark.model` | `methods[].model` |
|------|-------------------|-------------------|
| MTP heads in the target | same HF id | same as `benchmark.model` |
| Separate assistant checkpoint (Gemma 4) | target model | assistant/auxiliary checkpoint |

```yaml
# Pattern A — MTP baked into the target
benchmark:
  model: "Qwen/Qwen3.5-xxx"
speculative_decoding:
  methods:
    - method: mtp
      model: "Qwen/Qwen3.5-xxx"

# Pattern B — Gemma 4 assistant checkpoint
benchmark:
  model: "google/gemma-4-E2B-it"
speculative_decoding:
  methods:
    - method: mtp
      model: "gg-hf-am/gemma-4-E2B-it-assistant"
```

List multiple `methods` entries to let Optuna compare MTP vs EAGLE3 (or other methods) within one study.

### Static Parameters (Fixed Draft Settings)

Use `speculative_decoding.static_parameters` to fix draft-level keys inside `--speculative-config` without tuning them:

```yaml
speculative_decoding:
  static_parameters:
    draft_tensor_parallel_size: 1
    max_model_len: 8192
```

Allowed keys: `draft_tensor_parallel_size`, `max_model_len`, `num_speculative_tokens`.

A key cannot appear in both `static_parameters` and an enabled tunable sub-block. To tune one setting while fixing the other:

```yaml
speculative_decoding:
  static_parameters:
    draft_tensor_parallel_size: 1

  max_model_len:
    enabled: true
    options: [4096, 8192, 16384]
```

### Synthetic Acceptance Modes

Provide **exactly one** of:

#### `synthetic_acceptance_rates` (k is configurable)

The rates list defines the maximum acceptance profile length (`len(rates)`). Control **k** like other speculative sub-parameters:

| Configuration | Behavior |
|---------------|----------|
| omit `num_speculative_tokens` and no static k | fixed k = `len(rates)` (uses the full rates list) |
| `static_parameters.num_speculative_tokens: 2` | fixed k = 2 |
| `num_speculative_tokens.enabled: true` with `options` | Optuna categorical over those k values |

Each k must satisfy `1 <= k <= len(rates)`. vLLM receives the first k rates:

```yaml
synthetic_acceptance_rates: [0.8, 0.7, 0.6, 0.5]
num_speculative_tokens:
  enabled: true
  options: [2, 4]
# k=2 -> synthetic_acceptance_rates: [0.8, 0.7]
# k=4 -> synthetic_acceptance_rates: [0.8, 0.7, 0.6, 0.5]
```

Fixed k example:

```yaml
synthetic_acceptance_rates: [0.8, 0.7, 0.6, 0.5]
static_parameters:
  num_speculative_tokens: 2
```

#### `synthetic_acceptance_length` (k is fixed)

k is **not** tunable. Set it in `static_parameters`:

```yaml
synthetic_acceptance_length: 4
static_parameters:
  num_speculative_tokens: 4
```

### Full Example

See [`examples/study_config_speculative_decoding.yaml`](../examples/study_config_speculative_decoding.yaml).

```yaml
speculative_decoding:
  enabled: true
  allow_disabled: true

  methods:
    - method: eagle3
      model: "RedHatAI/Qwen3-8B-speculator.eagle3"

  synthetic_acceptance_rates: [0.8, 0.7, 0.6]
  num_speculative_tokens:
    enabled: true
    options: [2, 4]

  static_parameters:
    draft_tensor_parallel_size: 1

  max_model_len:
    enabled: true
    options: [4096, 8192, 16384]

parameters:
  tensor_parallel_size:
    enabled: true
    options: [1]
  max_model_len:
    enabled: true
    options: [8192, 16384]
```

For native **MTP**, swap `benchmark.model` and `methods` per the patterns in [Method-to-Model Mapping](#method-to-model-mapping) (commented snippets in the example YAML).

At runtime, enabled trials receive a single vLLM flag:

```bash
--speculative-config '{"method":"eagle3","model":"...","num_speculative_tokens":2,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.8,0.7],...}'
```

### Limitations

#### Grid sampler incompatible

The **grid** sampler enumerates every combination of the top-level `parameters:` block as a fixed Cartesian product. That works when every trial explores the same dimensions with the same shape.

Speculative decoding adds a **conditional** search space that grid search cannot represent:

| Dimension | Why it breaks grid enumeration |
|-----------|-------------------------------|
| **`allow_disabled: true`** | Each trial first chooses `spec_enabled` ∈ `{true, false}`. When `false`, no `--speculative-config` is passed at all; when `true`, several extra sub-parameters are sampled. These are two different trial shapes, not one product grid. |
| **Method choice** | Optuna samples `spec_method` from `methods:` (e.g. `eagle3` vs `mtp`). Different methods may require different draft models and vLLM flags inside the JSON. |
| **k (`num_speculative_tokens`)** | With `synthetic_acceptance_rates`, valid k values depend on `len(rates)` (each k slices the rates list). Options like `[2, 4]` are not independent of the rates profile. |
| **Cardinality under-count** | `auto-tune-vllm validate` reports grid cardinality from `parameters:` only and appends *(parameters only; excludes speculative search space)* when speculative decoding is enabled — the true trial count is higher. |

Because of this, the optimizer:

1. **Rejects** `optimization.sampler: grid` at config load time when `speculative_decoding.enabled: true`.
2. **Skips** the automatic grid switch in the CLI (normally triggered when `n_trials >=` parameter grid cardinality).

Use **`tpe`**, **`random`**, **`gp`**, **`botorch`**, or **`nsga2`** instead. For exhaustive search over target-model parameters only, disable speculative decoding (`enabled: false`) or run a separate study without the `speculative_decoding` block.

#### Optuna constraints cannot reference speculative sub-parameters

Study-level `constraints:` are arithmetic expressions evaluated against the **trial parameter dict** passed to vLLM — the same keys you would see in a trial config (e.g. `max_model_len`, `tensor_parallel_size`, `speculative_config`).

During sampling, speculative decoding registers internal Optuna names such as `spec_enabled`, `spec_method`, and `spec_num_speculative_tokens`. Those names exist in `trial.params` for the Optuna dashboard, but they are **not** copied into the trial parameter dict. Instead, enabled trials receive a single composed flag:

```yaml
# In trial parameters (what constraints see):
speculative_config: '{"method":"eagle3","num_speculative_tokens":2,...}'

# NOT available to constraints:
# spec_method, spec_enabled, spec_num_speculative_tokens, spec_max_model_len
```

Constraint evaluation uses `eval(expression, parameters)` on that dict. A constraint like `spec_method == "eagle3"` or `spec_num_speculative_tokens <= 2` will fail (missing variable) or cannot express method-specific rules.

**What works today:** constraints among top-level `parameters:` and `static_parameters` keys, e.g. `max_num_batched_tokens - max_model_len` or `tensor_parallel_size - 4`.

**What does not work:** rules that depend on whether speculation is on, which method was chosen, or draft-level k / `max_model_len`. Encode those relationships in the YAML itself (fixed `static_parameters`, non-overlapping `options`, separate studies per method) rather than in `constraints:`.

#### Optuna user attributes

Each trial stores `speculative_config` as a user attribute (`"disabled"` or the JSON string) for dashboard visibility. Use this column in the Optuna UI to inspect speculation settings post-hoc; it is not used by constraint evaluation.

## Configuration Examples

Ready-to-run YAML files live under [`examples/`](../examples/README.md): `study_config_local_exec.yaml` (full local study), `study_config_minimal.yaml` (smoke test), plus feature configs for speculative decoding, trace replay, and multimodal VLM.

The snippets below are inline alternatives you can copy into your own config.

### Basic Development Configuration

A minimal configuration for quick testing and development:

```yaml
study:
  prefix: "dev_test"

optimization:
  preset: "high_throughput"
  n_trials: 10

benchmark:
  model: "facebook/opt-125m"
  max_seconds: 60
  dataset: null
  prompt_tokens: 100
  output_tokens: 100

parameters:
  gpu_memory_utilization:
    enabled: true
  max_num_batched_tokens:
    enabled: true
```

### Production Configuration

A comprehensive production setup with PostgreSQL storage and multi-objective optimization:

```yaml
study:
  name: "production_optimization_v1"
  database_url: "postgresql://tuner:${POSTGRES_PASSWORD}@localhost:5432/optuna"

optimization:
  approach: "multi_objective"
  objectives:
    - metric: "output_tokens_per_second"
      direction: "maximize"
      percentile: "median"
    - metric: "request_latency"
      direction: "minimize"
      percentile: "p95"
  sampler: "nsga2"
  n_trials: 200

benchmark:
  model: "Qwen/Qwen3-30B-A3B-FP8"
  max_seconds: 300
  dataset: null
  prompt_tokens: 1000
  output_tokens: 1000
  rate: 100
  # warmup: 0.1
  # cooldown: 0.1
  # rampup: 10
  # sample_requests: 0  # default; increase to keep detailed request samples in benchmark JSON

logging:
  file_path: "/var/log/auto-tune-vllm"
  log_level: "INFO"

parameters:
  gpu_memory_utilization:
    enabled: true
    min: 0.85
    max: 0.95
    step: 0.01
  max_num_batched_tokens:
    enabled: true
  kv_cache_dtype:
    enabled: true
    options: ["auto", "fp8"]
```

### SQLite-Only Configuration

A file-based configuration without PostgreSQL requirements:

```yaml
study:
  name: "local_optimization"
  storage_file: "/tmp/optimization/study.db"

optimization:
  preset: "balanced"
  n_trials: 50

benchmark:
  model: "facebook/opt-125m"
  max_seconds: 120
  dataset: null

logging:
  file_path: "/tmp/optimization/logs"
  log_level: "INFO"

parameters:
  gpu_memory_utilization:
    enabled: true
  max_num_batched_tokens:
    enabled: true
  kv_cache_dtype:
    enabled: true
```

---

## Configuration Validation

Before running optimization, validate your configuration file:

```bash
auto-tune-vllm validate --config your_config.yaml
```

The validation checks:
- YAML syntax correctness
- Required field presence
- Parameter range and option validity
- Optimization configuration consistency
- Environment variable expansion
- Study naming conflicts

## Best Practices

1. **Start simple**: Begin with preset configurations before customizing
2. **Use meaningful names**: Choose descriptive study names for production
3. **Choose appropriate storage**: PostgreSQL for production, SQLite for development
4. **Scale trials appropriately**: 10-50 for development, 100+ for production
5. **Select relevant percentiles**: p95 for SLAs, median for general optimization
6. **Enable file logging**: Essential for troubleshooting and result analysis
7. **Secure credentials**: Use environment variables for sensitive information
8. **Start with high-impact parameters**: Focus on memory and batching parameters first
