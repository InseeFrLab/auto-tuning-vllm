"""Smoke-test an installed distribution; run with python -I outside the checkout."""

import importlib
import subprocess
import sys
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path

import yaml

import auto_tune_vllm
from auto_tune_vllm.utils.version_manager import VLLMVersionManager

dist = distribution("auto-tune-vllm")
assert auto_tune_vllm.__version__ == dist.version
assert (
    Path(auto_tune_vllm.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
)
for name in (
    "cli",
    "core.config",
    "core.study_controller",
    "execution.backends",
    "benchmarks.providers",
    "benchmarks.trace_replay",
    "logging.manager",
    "metrics_scraping.vllm_metrics",
    "utils.vllm_cli_parser",
):
    importlib.import_module(f"auto_tune_vllm.{name}")

defaults = files("auto_tune_vllm").joinpath("schemas/vllm_defaults")
resources = sorted(p for p in defaults.iterdir() if p.name.endswith(".yaml"))
assert resources, "Installed package has no vLLM defaults"
for resource in resources:
    data = yaml.safe_load(resource.read_text())
    assert data["defaults"], f"Empty defaults in {resource.name}"
manager = VLLMVersionManager()
assert len(manager.list_available_versions()) == len(resources)
assert manager.load_defaults()["defaults"]

entry_point = next(ep for ep in dist.entry_points if ep.name == "auto-tune-vllm")
assert entry_point.value == "auto_tune_vllm.cli:main"
assert callable(entry_point.load())
cli = Path(sys.executable).parent / "auto-tune-vllm"
result = subprocess.run(
    [str(cli), "--help"], check=True, capture_output=True, text=True
)
assert "Usage" in result.stdout and "optimize" in result.stdout
print(result.stdout)
print(
    f"Installed {dist.metadata['Name']} {dist.version}: imports, YAML defaults and CLI OK"
)
