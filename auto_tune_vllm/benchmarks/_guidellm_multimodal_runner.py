"""Run multimodal GuideLLM benchmarks through Python API entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

# Register custom preprocessors before GuideLLM benchmark schemas are imported.
from auto_tune_vllm.benchmarks import preprocessors  # noqa: F401


def _parse_json_arg(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    return json.loads(raw)


def _build_base_dirs(
    dataset: str, preprocessors_kwargs: dict[str, Any] | None
) -> list[Path]:
    base_dirs: list[Path] = []

    if preprocessors_kwargs:
        for base_dir in preprocessors_kwargs.get("base_dirs", []):
            base_dirs.append(Path(str(base_dir)))

    if not dataset.startswith("hf://"):
        dataset_path = Path(dataset)
        base_dirs.extend([Path.cwd(), dataset_path.parent])

    unique: dict[str, Path] = {}
    for base_dir in base_dirs:
        unique[str(base_dir)] = base_dir
    return list(unique.values())


def _normalize_column_mapper(mapper: dict[str, Any] | None) -> dict[str, Any]:
    if not mapper:
        return {"kind": "generative_column_mapper"}

    normalized = dict(mapper)
    normalized.pop("type", None)
    normalized.setdefault("kind", "generative_column_mapper")

    if "column_mappings" not in normalized and any(
        key.endswith("_column") for key in normalized if key != "kind"
    ):
        column_mappings = {
            key: value
            for key, value in normalized.items()
            if key not in {"kind", "type"}
        }
        return {"kind": "generative_column_mapper", "column_mappings": column_mappings}

    return normalized


def _resolve_preprocessors(
    names: list[str], dataset: str, preprocessors_kwargs: dict[str, Any] | None
) -> list[dict[str, Any]]:
    base_dirs = [str(path) for path in _build_base_dirs(dataset, preprocessors_kwargs)]
    resolved: list[dict[str, Any]] = []

    for name in names:
        if name == "flatten_image_lists":
            resolved.append({"kind": "flatten_image_lists", "base_dirs": base_dirs})
        else:
            resolved.append({"kind": name})

    return resolved


def _normalize_request_format(request_format: str) -> str:
    legacy_aliases = {
        "text_completions": "/v1/completions",
        "chat_completions": "/v1/chat/completions",
        "audio_transcriptions": "/v1/audio/transcriptions",
        "audio_translations": "/v1/audio/translations",
    }
    return legacy_aliases.get(request_format, request_format)


def _file_data_kind(dataset: str) -> str:
    suffix = Path(dataset).suffix.lower()
    if suffix in {".csv"}:
        return "csv_file"
    if suffix in {".json", ".jsonl"}:
        return "json_file"
    if suffix in {".parquet"}:
        return "parquet_file"
    return "json_file"


_FILE_KINDS_NEEDING_TRAIN_SPLIT = frozenset(
    {"json_file", "csv_file", "parquet_file", "arrow_file"}
)


def _build_data_entries(dataset: str, data_args: dict[str, Any] | None) -> list[Any]:
    load_kwargs = dict(data_args or {})
    if dataset.startswith("hf://"):
        return [
            {
                "kind": "huggingface",
                "source": dataset[5:],
                "load_kwargs": load_kwargs,
            }
        ]

    kind = _file_data_kind(dataset)
    # GuideLLM 0.7+ column mappers expect a Dataset (with .info), but
    # datasets.load_dataset() returns a DatasetDict for local files unless
    # a split is selected.
    if kind in _FILE_KINDS_NEEDING_TRAIN_SPLIT and "split" not in load_kwargs:
        load_kwargs["split"] = "train"

    return [
        {
            "kind": kind,
            "path": dataset,
            "load_kwargs": load_kwargs,
        }
    ]


def _copy_generated_report(output_path: Path) -> None:
    generated_report = output_path.parent / "benchmarks.json"
    if not generated_report.exists():
        raise RuntimeError(f"GuideLLM report was not generated: {generated_report}")
    shutil.copyfile(generated_report, output_path)


async def _run(args: argparse.Namespace) -> None:
    from guidellm.benchmark.entrypoints import benchmark_generative_text
    from guidellm.benchmark.schemas.entrypoints import BenchmarkScenario

    processor_args = _parse_json_arg(args.processor_args) or {}
    data_args = _parse_json_arg(args.data_args)
    data_column_mapper = _parse_json_arg(args.data_column_mapper)
    data_preprocessors_kwargs = _parse_json_arg(args.data_preprocessors_kwargs) or {}

    profile: dict[str, Any] = {
        "kind": "concurrent",
        "streams": [int(args.rate)],
    }
    if args.warmup is not None:
        profile["warmup"] = args.warmup
    if args.cooldown is not None:
        profile["cooldown"] = args.cooldown

    data_finalizer = args.data_finalizer or "generative"
    if isinstance(data_finalizer, str):
        data_finalizer = {"kind": data_finalizer}

    spec: dict[str, Any] = {
        "backend": {
            "kind": "openai_http",
            "target": args.target,
            "model": args.model,
            "request_format": _normalize_request_format(args.request_format),
        },
        "profile": profile,
        "constraints": [{"kind": "max_duration", "seconds": args.max_seconds}],
        "tokenizer": {
            "kind": "huggingface_auto",
            "model": args.processor,
            "load_kwargs": processor_args,
        },
        "data": _build_data_entries(args.dataset, data_args),
        "data_column_mapper": _normalize_column_mapper(data_column_mapper),
        "data_preprocessors": _resolve_preprocessors(
            args.data_preprocessors.split(","),
            args.dataset,
            data_preprocessors_kwargs,
        ),
        "data_finalizer": data_finalizer,
        "data_loader": {"kind": "pytorch", "num_workers": 0},
        "outputs": [
            {
                "kind": "json",
                "path": str(args.output_path.parent / "benchmarks.json"),
            }
        ],
        "metrics": {
            "kind": "generative",
            "sample_size": args.sample_requests,
        },
    }

    bench_args = BenchmarkScenario.create(scenario=None, spec=spec)
    await benchmark_generative_text(args=bench_args)
    _copy_generated_report(args.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--request-format", required=True)
    parser.add_argument("--max-seconds", required=True, type=float)
    parser.add_argument("--rate", required=True, type=float)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--sample-requests", required=True, type=int)
    parser.add_argument("--data-preprocessors", required=True)
    parser.add_argument("--processor-args")
    parser.add_argument("--data-args")
    parser.add_argument("--data-column-mapper")
    parser.add_argument("--data-preprocessors-kwargs")
    parser.add_argument("--data-finalizer")
    parser.add_argument("--warmup", type=float)
    parser.add_argument("--cooldown", type=float)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.output_path = args.output_path.resolve()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
