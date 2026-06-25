"""Wrapper that registers custom preprocessors then dispatches to the GuideLLM CLI."""

from guidellm.__main__ import cli

from auto_tune_vllm.benchmarks import preprocessors  # noqa: F401

if __name__ == "__main__":
    cli()
