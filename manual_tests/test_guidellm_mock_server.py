"""Opt-in integration test: real GuideLLM against its mock inference server."""

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from auto_tune_vllm.benchmarks.config import BenchmarkConfig
from auto_tune_vllm.benchmarks.providers import GuideLLMBenchmark


def test_guidellm_benchmark_against_mock_server(tmp_path):
    # A tiny local tokenizer keeps this test independent of model downloads.
    vocabulary = {"[UNK]": 0, **{f"word{i}": i + 1 for i in range(1024)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        tmp_path / "tokenizer"
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    server_log = tmp_path / "server.log"
    with server_log.open("w") as log:
        server = subprocess.Popen(
            ["guidellm", "mock-server", "--port", str(port), "--model", "mock-model"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        provider = GuideLLMBenchmark()
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    raise AssertionError(f"Mock server exited:\n{server_log.read_text()}")
                try:
                    if requests.get(f"{url}/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() >= deadline:
                    raise AssertionError(f"Mock server not ready:\n{server_log.read_text()}")
                time.sleep(0.2)

            config = BenchmarkConfig(
                model="mock-model",
                processor=str(tmp_path / "tokenizer"),
                request_format="/v1/completions",
                max_seconds=10,
                rate=1,
                samples=5,
                prompt_tokens=16,
                output_tokens=8,
            )
            provider._get_results_file_path = lambda: str(tmp_path / "results.json")
            process = provider.start_benchmark(url, config)
            try:
                code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                raise AssertionError(f"GuideLLM timed out:\n{provider.get_last_log_lines()}")
            assert code == 0, provider.get_last_log_lines()
            metrics = provider.parse_results()
            assert metrics["requests_per_second"] > 0
            assert metrics["output_tokens_per_second"] > 0
            assert Path(provider._results_file).is_file()
        finally:
            if provider._process is not None and provider._process.poll() is None:
                provider.terminate_benchmark()
                provider._process.wait(timeout=10)
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
