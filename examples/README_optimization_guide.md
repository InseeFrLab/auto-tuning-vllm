# Optimization Configuration Guide

This guide explains how to configure optimization objectives for auto-tune-vllm to get the best results for your specific use case.

## Quick Start

### Simple Presets (Recommended for Beginners)

```yaml
# High throughput (maximize tokens/second)
optimization:
  preset: "high_throughput"
  n_trials: 100

# Low latency (minimize response time)
optimization:
  preset: "low_latency"
  n_trials: 100

# Balanced (find throughput vs latency trade-offs)
optimization:
  preset: "balanced"
  n_trials: 200
```

### Advanced Configuration

For full control, use the explicit configuration format:

```yaml
optimization:
  approach: "single_objective"  # or "multi_objective"
  objective:  # For single objective
    metric: "output_tokens_per_second_median"
    direction: "maximize"
  sampler: "tpe"
  n_trials: 100
```

## The `metric` Field

`metric` is an **arithmetic expression** built from identifiers of the form
`<base_metric>_<percentile>`. There is no separate `percentile` field — the
percentile is part of each identifier.

The recommended default percentile is `_median`.

### Base Metrics

| Base metric | Description | Typical Goal | Units |
|---|---|---|---|
| `output_tokens_per_second` | Token generation throughput | Maximize | tokens/sec |
| `request_latency` | End-to-end request latency | Minimize | milliseconds |
| `time_to_first_token_ms` | Time until first token (TTFT) | Minimize | milliseconds |
| `inter_token_latency_ms` | Latency between tokens (ITL) | Minimize | milliseconds |
| `requests_per_second` | Request throughput | Maximize | requests/sec |

### Percentile Suffixes

| Suffix | Description | When to Use |
|---|---|---|
| `_median` or `_p50` | 50th percentile | Recommended default — stable optimization |
| `_p90` | 90th percentile | Good balance between median and extreme cases |
| `_p95` | 95th percentile | SLA optimization, tail latency |
| `_p99` | 99th percentile | Extreme tail latency optimization |
| `_mean` | Arithmetic mean | When you want average-case behavior |

### Custom Expressions

`metric` can be any arithmetic expression over the identifiers above using
`+`, `-`, `*`, `/`, `**`, parentheses and numeric constants. Examples:

```yaml
# Tokens generated per request
metric: "output_tokens_per_second_mean / requests_per_second_median"

# Composite latency score
metric: "time_to_first_token_ms_p95 + inter_token_latency_ms_p99"

# Throughput penalised by latency
metric: "(output_tokens_per_second_mean - request_latency_median) * 2"
```

## Optimization Approaches

### 1. Single Objective Optimization

Optimize for one metric only. Best when you have a clear primary goal.

#### Example: Maximize Throughput
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "output_tokens_per_second_median"
    direction: "maximize"
  sampler: "tpe"
  n_trials: 100
```

#### Example: Minimize P95 Latency
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "request_latency_p95"  # Optimize for 95th percentile SLA
    direction: "minimize"
  sampler: "tpe"
  n_trials: 100
```

#### Example: Minimize Time-To-First-Token
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "time_to_first_token_ms_p95"
    direction: "minimize"
  sampler: "tpe"
  n_trials: 100
```

### 2. Multi-Objective Optimization

Find optimal trade-offs between multiple metrics. Returns Pareto-optimal solutions.

#### Example: Throughput vs Latency (Most Common)
```yaml
optimization:
  approach: "multi_objective"
  objectives:
    - metric: "output_tokens_per_second_median"
      direction: "maximize"
    - metric: "request_latency_median"
      direction: "minimize"
  sampler: "nsga2"  # Recommended for multi-objective
  n_trials: 200
```

#### Example: Throughput vs TTFT
```yaml
optimization:
  approach: "multi_objective"
  objectives:
    - metric: "output_tokens_per_second_median"
      direction: "maximize"
    - metric: "time_to_first_token_ms_p95"
      direction: "minimize"
  sampler: "nsga2"
  n_trials: 200
```

#### Example: TTFT vs End-to-End Latency
```yaml
optimization:
  approach: "multi_objective"
  objectives:
    - metric: "time_to_first_token_ms_p95"
      direction: "minimize"
    - metric: "request_latency_p95"
      direction: "minimize"
  sampler: "nsga2"
  n_trials: 200
```

## Sampler Selection

| Sampler | Best For | Description |
|---------|----------|-------------|
| `"tpe"` | Single objective | Tree-structured Parzen Estimator, good default |
| `"nsga2"` | Multi-objective | Non-dominated Sorting Genetic Algorithm II - Default sampler for multi-objective optimization|
| `"botorch"` | Single/Multi objective (advanced) | Bayesian Optimization - Fast but not necessarily the best result. Can get stuck in local optima. |
| `"random"` | Testing/baseline | Random sampling, good for quick tests - Do not use outside of testing |
| `"grid"` | Exhaustive search | Grid search over all parameter combinations |

## Use Case Examples

### High-Performance Serving
```yaml
# Focus on maximum throughput
optimization:
  preset: "high_throughput"
  # OR explicit:
  # approach: "single_objective"
  # objective:
  #   metric: "output_tokens_per_second_mean"
  #   direction: "maximize"
```

### Interactive Applications
```yaml
# Minimize time to first token for responsiveness
optimization:
  approach: "single_objective"
  objective:
    metric: "time_to_first_token_ms_p95"
    direction: "minimize"
  sampler: "tpe"
  n_trials: 100
```

### Production SLA Optimization
```yaml
# Minimize P95 latency for SLA compliance
optimization:
  approach: "single_objective"
  objective:
    metric: "request_latency_p95"
    direction: "minimize"
  sampler: "tpe"
  n_trials: 150
```

### Balanced Production Setup
```yaml
# Find best throughput vs latency trade-offs
optimization:
  preset: "balanced"
  # OR explicit:
  # approach: "multi_objective"
  # objectives:
  #   - metric: "output_tokens_per_second_mean"
  #     direction: "maximize"
  #   - metric: "request_latency_median"
  #     direction: "minimize"
```

### Streaming Applications
```yaml
# Optimize inter-token latency for smooth streaming
optimization:
  approach: "single_objective"
  objective:
    metric: "inter_token_latency_ms_p95"
    direction: "minimize"
  sampler: "tpe"
  n_trials: 100
```

## Migration from Old Format

### Old Format (Deprecated but still supported)
```yaml
optimization:
  objective: "maximize"  # Vague, only optimizes throughput
  sampler: "tpe"
  n_trials: 100
```

### Old Structured Format (no longer supported — `percentile` field removed)
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "output_tokens_per_second"  # bare name (no longer accepted)
    direction: "maximize"
    percentile: "median"                # field removed
```

### New Format (Recommended)
```yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "output_tokens_per_second_median"  # percentile baked into the identifier
    direction: "maximize"
  sampler: "tpe"
  n_trials: 100

# OR use preset for simplicity
optimization:
  preset: "high_throughput"
  n_trials: 100
```

## Advanced Tips

### 1. Choosing Percentiles
- **`_median` (p50)**: Most stable, good for general optimization
- **`_p95`**: Good for SLA requirements and tail latency
- **`_p99`**: Use sparingly, can be noisy and lead to overfitting

### 2. Trial Count Guidelines
- **Single objective**: 50-150 trials usually sufficient
- **Multi-objective**: 150-300 trials for good Pareto frontier
- **Quick testing**: 10-20 trials with random sampler

### 3. Multi-Objective Interpretation
Multi-objective optimization returns multiple solutions on the Pareto frontier. Each solution represents a different trade-off. Review the results and choose the solution that best fits your requirements.

### 4. Parameter Space Considerations
More complex parameter spaces need more trials. If you have many parameters enabled, increase `n_trials` accordingly.

## Troubleshooting

### Issue: Optimization seems random
- **Solution**: Increase `n_trials`, ensure parameters have reasonable ranges
- **Check**: Parameter space isn't too large relative to trial count

### Issue: No improvement over baseline
- **Solution**: Check if parameters are actually affecting the chosen metric
- **Try**: Different metrics or multi-objective approach

### Issue: Multi-objective results unclear
- **Solution**: Use visualization tools to explore Pareto frontier
- **Consider**: Single objective if one metric is clearly most important

## Examples in This Repository

- `study_config_local_exec.yaml`: High-throughput optimization
- `study_config_no_postgres.yaml`: Low-latency optimization
- `study_config_minimal.yaml`: Balanced multi-objective
- `study_config.yaml`: Advanced throughput vs TTFT optimization
- `study_config_optimization_examples.yaml`: Comprehensive examples
