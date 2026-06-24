"""vLLM Prometheus metrics scraping during benchmarks."""

from .vllm_metrics import VLLMMetricsCollector

__all__ = ["VLLMMetricsCollector"]
