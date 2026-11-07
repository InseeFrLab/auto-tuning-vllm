"""Manual full-stack autotuner tests. Only vLLM inference is simulated."""

import json
import math
import os
import shutil
import signal
import socket
import subprocess
from pathlib import Path
from uuid import uuid4

import optuna
import pytest
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    "output_tokens_per_second_mean",
    "requests_per_second_mean",
    "time_to_first_token_ms_p95",
]
pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOTUNER_E2E") != "1",
    reason="Set AUTOTUNER_E2E=1 to run manual integration tests",
)


@pytest.fixture
def study_inputs(tmp_path):
    # Only the vLLM import/entry point is substituted, in the child environment.
    shim = tmp_path / "shim"
    package = shim / "vllm" / "entrypoints" / "openai"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text('__version__ = "0.11.0"\n')
    (package / "api_server.py").write_text(
        "from manual_tests.mock_vllm import main\nmain()\n"
    )
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, **{f"word{i}": i + 1 for i in range(1024)}}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    fast.save_pretrained(tmp_path / "tokenizer")
    (tmp_path / "events").mkdir()
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(shim), str(ROOT)]),
        MOCK_EVENTS=str(tmp_path / "events"),
        HF_HUB_OFFLINE="1",
        HF_HOME=str(tmp_path / "hf-cache"),
        TOKENIZERS_PARALLELISM="false",
    )
    env.pop("MOCK_FAIL_PARAMETER", None)
    env.pop("MOCK_FAULT", None)
    return env


def run_study(tmp_path, env, benchmark, *, fault=None):
    name = f"manual_{uuid4().hex}"
    config = {
        "study": {"name": name, "storage_file": str(tmp_path / "study.db")},
        "optimization": {
            "approach": "single_objective",
            "objective": {
                "metric": "output_tokens_per_second_mean / requests_per_second_mean",
                "direction": "maximize",
            },
            "sampler": "grid",
            "n_trials": 4,
            "log_metrics": METRICS,
        },
        "benchmark": {
            "model": "mock-model",
            "processor": str(tmp_path / "tokenizer"),
            "max_seconds": 20,
            "rate": 1,
            "samples": 4,
            "prompt_tokens": 16,
            "output_tokens": 8,
            "request_format": "/v1/completions",
        }
        | benchmark,
        "baseline": {"enabled": False},
        "parameters": {"max_num_seqs": {"enabled": True, "options": [1, 2, 3, 4]}},
        "static_environment_variables": {"VLLM_STARTUP_TIMEOUT": "5"},
        "logging": {"file_path": str(tmp_path / "logs")},
    }
    if fault:
        env["MOCK_FAIL_PARAMETER"] = "2"
        env["MOCK_FAULT"] = fault
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(config))
    result_dir = Path("/tmp/auto-tune-vllm-local-run/logs") / name / "benchmark_results"
    log_path = tmp_path / "autotuner.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                "auto-tune-vllm",
                "optimize",
                "--config",
                str(path),
                "--max-concurrent-trials",
                "1",
            ],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            assert process.wait(timeout=240) == 0, log_path.read_text()[-12000:]
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            # Assert cleanup before this fallback; always remove leaks after a failure.
            records = [
                json.loads(line)
                for f in (tmp_path / "events").glob("*.jsonl")
                for line in f.read_text().splitlines()
            ]
            leaked = []
            # Every spawned server/benchmark inherits this unique environment marker.
            # Inspect before cleanup so a test fallback cannot hide an autotuner leak.
            marker = f"MOCK_EVENTS={env['MOCK_EVENTS']}".encode()
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    if marker not in (entry / "environ").read_bytes().split(b"\0"):
                        continue
                    pid = int(entry.name)
                    leaked.append(pid)
                    os.kill(pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
            if result_dir.exists():
                shutil.copytree(result_dir, tmp_path / "reports")
            assert not leaked, f"Autotuner leaked child processes: {leaked}"
    assert len([r for r in records if r["kind"] == "start"]) == 4
    assert len([r for r in records if r["kind"] == "stop"]) == (
        3 if fault == "startup" else 4
    )
    lifecycle = sorted(
        (r for r in records if r["kind"] in {"start", "stop", "startup_error"}),
        key=lambda r: r["time"],
    )
    assert [r["kind"] for r in lifecycle][::2] == ["start"] * 4
    assert all(r["kind"] in {"stop", "startup_error"} for r in lifecycle[1::2])
    # A closed port independently confirms that the server no longer listens.
    for file in (tmp_path / "events").glob("*.jsonl"):
        events = [json.loads(line) for line in file.read_text().splitlines()]
        start = events[0]
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", start["port"])) != 0
    study = optuna.load_study(
        study_name=name, storage=f"sqlite:///{tmp_path / 'study.db'}"
    )
    assert len(study.trials) == 4
    assert {t.params["max_num_seqs"] for t in study.trials} == {1, 2, 3, 4}
    for trial in study.trials:
        if fault and trial.params["max_num_seqs"] == 2:
            assert trial.state == optuna.trial.TrialState.FAIL
            assert trial.value is None
            assert trial.user_attrs.get("error_message")
            if fault == "benchmark":
                assert trial.user_attrs["error_type"] == "Benchmark_Error"
            assert not any(k.startswith("metric_") for k in trial.user_attrs)
            continue
        assert trial.state == optuna.trial.TrialState.COMPLETE, (
            trial.number,
            trial.user_attrs,
            log_path.read_text()[-12000:],
        )
        raw = json.loads(
            (result_dir / f"trial_{trial.number}_benchmark_results.json").read_text()
        )["benchmarks"][0]["metrics"]
        for metric in METRICS:
            name, stat = metric.rsplit("_", 1)
            category = "total" if "per_second" in name else "successful"
            expected = raw[name][category].get(
                stat, raw[name][category].get("percentiles", {}).get(stat)
            )
            actual = trial.user_attrs[f"metric_{metric}"]
            assert math.isfinite(actual) and actual > 0
            assert actual == pytest.approx(expected)
        throughput = trial.user_attrs["metric_output_tokens_per_second_mean"]
        rate = trial.user_attrs["metric_requests_per_second_mean"]
        assert trial.value == pytest.approx(throughput / rate)
        assert raw["output_token_count"]["successful"]["median"] == 8
        assert trial.user_attrs["metric_time_to_first_token_ms_p95"] >= 30
        assert 0 < trial.user_attrs["benchmark_duration_seconds"] < 35
    if fault:
        failed = next(
            t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        )
        assert any(
            t.number > failed.number and t.state == optuna.trial.TrialState.COMPLETE
            for t in study.trials
        )
    return records


@pytest.mark.parametrize("kind", ["concurrent", "trace_replay", "multi_image"])
def test_autotuner_four_trials(tmp_path, study_inputs, kind):
    benchmark = {}
    if kind == "trace_replay":
        trace = tmp_path / "trace.jsonl"
        trace.write_text(
            "\n".join(
                json.dumps(
                    {"timestamp": i * 0.25, "input_length": 16, "output_length": 8}
                )
                for i in range(4)
            )
        )
        benchmark = {
            "benchmark_type": "guidellm_trace_replay",
            "dataset": str(trace),
            "profile": {"trace_format": "trace_synthetic"},
        }
    elif kind == "multi_image":
        dataset = tmp_path / "images.jsonl"
        dataset.write_text(
            "\n".join(
                json.dumps(
                    {
                        "prompt": "word1 word2",
                        "image": [
                            str(ROOT / "examples/vlm_multi_image/image1.png"),
                            str(ROOT / "examples/vlm_multi_image/image2.png"),
                        ],
                        "output_tokens_count": 8,
                    }
                )
                for _ in range(4)
            )
        )
        benchmark = {
            "benchmark_type": "guidellm_multimodal",
            "dataset": str(dataset),
            "data_column_mapper": {
                "text_column": "prompt",
                "image_column": "image",
                "output_tokens_count_column": "output_tokens_count",
            },
            "data_preprocessors": ["flatten_image_lists", "encode_media"],
        }
    if kind == "multi_image":
        # Override the default endpoint without duplicate keyword arguments.
        benchmark["request_format"] = "/v1/chat/completions"
    records = run_study(tmp_path, study_inputs, benchmark)
    for start in [r for r in records if r["kind"] == "start"]:
        requests = [
            r for r in records if r["kind"] == "request" and r["pid"] == start["pid"]
        ]
        assert requests
        if kind == "multi_image":
            for request in requests:
                content = request["body"]["messages"][0]["content"]
                images = [part for part in content if part["type"] == "image_url"]
                assert len(images) == 2
                assert all(
                    p["image_url"]["url"].startswith("data:image/") for p in images
                )
        if kind == "trace_replay":
            assert len(requests) >= 4


@pytest.mark.parametrize("fault", ["startup", "benchmark"])
def test_autotuner_recovers_from_failure(tmp_path, study_inputs, fault):
    records = run_study(tmp_path, study_inputs, {}, fault=fault)
    assert sum(r["kind"] == "startup_error" for r in records) == (
        1 if fault == "startup" else 0
    )
