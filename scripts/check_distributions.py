"""Check release metadata and shipped files without importing runtime dependencies."""

import argparse
import ast
import configparser
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--tag", help="Require this exact release tag: v<project.version>"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if args.tag is not None and args.tag != f"v{version}":
        parser.error(f"Tag {args.tag!r} must equal 'v{version}'")

    module = ast.parse((root / "auto_tune_vllm/__init__.py").read_text())
    package_version = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    )
    assert package_version == version, "Package and pyproject versions differ"

    wheels = list(args.dist_dir.glob("*.whl"))
    sdists = list(args.dist_dir.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Expected exactly one wheel and one sdist"
    expected = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "auto_tune_vllm").rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".json"}
    }
    assert any(name.endswith(".yaml") for name in expected), "No defaults found"

    def check_metadata(data):
        metadata = Parser().parsestr(data.decode("utf-8"))
        assert metadata["Name"] == project["name"] == "auto-tune-vllm"
        assert metadata["Version"] == version
        assert metadata["Requires-Python"] == project["requires-python"]
        assert metadata["License-Expression"] == project["license"] == "Apache-2.0"
        assert "LICENSE" in metadata.get_all("License-File", [])
        assert metadata["Description-Content-Type"] == "text/markdown"
        assert (
            metadata.get_payload().strip() == (root / "README.md").read_text().strip()
        )

    with zipfile.ZipFile(wheels[0]) as wheel:
        for name, content in expected.items():
            assert wheel.read(name) == content, f"Missing or changed wheel file: {name}"
        metadata_name = next(n for n in wheel.namelist() if n.endswith("/METADATA"))
        info_dir = metadata_name.rsplit("/", 1)[0]
        check_metadata(wheel.read(metadata_name))
        assert (
            wheel.read(f"{info_dir}/licenses/LICENSE")
            == (root / "LICENSE").read_bytes()
        )
        entry_points = configparser.ConfigParser()
        entry_points.read_string(wheel.read(f"{info_dir}/entry_points.txt").decode())
        assert dict(entry_points["console_scripts"]) == project["scripts"]

    with tarfile.open(sdists[0]) as sdist:
        prefix = sdist.getnames()[0].split("/")[0]
        source_files = dict(expected)
        for name in ("README.md", "LICENSE", "CHANGELOG.md", "pyproject.toml"):
            source_files[name] = (root / name).read_bytes()
        for directory in ("examples", "docs"):
            for path in (root / directory).rglob("*"):
                if path.is_file() and path.suffix in {
                    ".py",
                    ".yaml",
                    ".md",
                    ".rst",
                    ".jsonl",
                    ".jpg",
                }:
                    source_files[path.relative_to(root).as_posix()] = path.read_bytes()
        for name, content in source_files.items():
            member = sdist.extractfile(f"{prefix}/{name}")
            assert member is not None and member.read() == content, (
                f"Invalid sdist file: {name}"
            )
        metadata_file = sdist.extractfile(f"{prefix}/PKG-INFO")
        assert metadata_file is not None
        check_metadata(metadata_file.read())

    print(
        f"Validated {wheels[0].name} and {sdists[0].name}: version {version}, "
        f"metadata, CLI entry point, license and {len(expected)} package files"
    )


if __name__ == "__main__":
    main()
