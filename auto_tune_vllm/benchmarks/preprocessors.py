"""Custom GuideLLM dataset preprocessors registered for CLI use."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)

try:
    from guidellm.data.schemas import DataPreprocessorArgs
    from pydantic import Field

    @DataPreprocessorArgs.register("flatten_image_lists")
    class FlattenImageListsArgs(DataPreprocessorArgs):
        """Arguments for the flatten_image_lists preprocessor (GuideLLM 0.7+)."""

        kind: Literal["flatten_image_lists"] = Field(default="flatten_image_lists")
        base_dirs: list[str] = Field(default_factory=list)

except ImportError:
    FlattenImageListsArgs = None  # type: ignore[assignment,misc]

__all__ = ["FlattenImageListsPreprocessor", "resolve_image_path"]


def resolve_image_path(path: str, base_dirs: list[Path]) -> str:
    """Resolve a local image path against optional base directories."""
    if path.startswith(("http://", "https://", "data:image/")):
        return path

    candidate = Path(path)
    if candidate.is_file():
        return str(candidate.resolve())

    for base in base_dirs:
        resolved = base / path
        if resolved.is_file():
            return str(resolved.resolve())

    return path


@PreprocessorRegistry.register("flatten_image_lists")
class FlattenImageListsPreprocessor(DatasetPreprocessor):
    """Expand nested image lists so each image is encoded separately."""

    def __init__(
        self,
        config: Any | None = None,
        base_dirs: list[str | Path] | None = None,
        **_: Any,
    ) -> None:
        if config is not None and hasattr(config, "base_dirs"):
            base_dirs = config.base_dirs
        self.base_dirs = [Path(directory) for directory in (base_dirs or [])]

    def __call__(self, items: list[dict[str, list[Any]]]) -> list[dict[str, list[Any]]]:
        for turn in items:
            image_column = turn.get("image_column")
            if not image_column:
                continue

            values = [image_column] if isinstance(image_column, str) else image_column

            flattened: list[Any] = []
            for value in values:
                if isinstance(value, list):
                    flattened.extend(item for item in value if item)
                elif value:
                    flattened.append(value)

            turn["image_column"] = [
                resolve_image_path(image, self.base_dirs)
                if isinstance(image, str)
                else image
                for image in flattened
            ]

        return items
