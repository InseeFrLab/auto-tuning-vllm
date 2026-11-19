# Publishing a release

The PyPI distribution is **auto-tune-vllm**; the import is `auto_tune_vllm`
and the CLI is `auto-tune-vllm`. The initial version is `0.1.0`.

## One-time setup (maintainer)

1. In this repository's **Settings → Environments**, create `pypi`.
   Restrict deployments to **selected tags** matching `v*` (do not allow branches).
   Configure required reviewers where available, and restrict who can create or
   update release tags through a repository ruleset.
2. On PyPI, configure a GitHub Trusted Publisher for these exact values:

   | Field | Value |
   | --- | --- |
   | PyPI project | `auto-tune-vllm` |
   | Owner | `InseeFrLab` |
   | Repository | `auto-tuning-vllm` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

   If the project does not exist yet, add a **pending publisher** at
   [PyPI account publishing](https://pypi.org/manage/account/publishing/).
   The first successful upload creates the project. A pending publisher does
   not reserve the project name; confirm availability before releasing.
   If the project already exists and you own it, add the publisher in its
   publishing settings instead.
3. Ensure GitHub Actions can run the actions used in these workflows under the
   organization/repository policy. No PyPI API token, username or password is
   needed in GitHub secrets.

References: [PyPI publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/),
[first publication](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/),
[GitHub OIDC](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi).

## Validate a release PR

1. Update `project.version` in `pyproject.toml` and `__version__` in
   `auto_tune_vllm/__init__.py` together. Update `CHANGELOG.md` for the release.
   For the first release, both version fields already contain `0.1.0`.
2. Open a PR against `main`. `package-ci.yml` builds the sdist and wheel, runs
   `twine check --strict`, checks metadata, modules, YAML defaults, the license
   and sdist documentation/example assets, then installs the wheel **with all
   dependencies** in a fresh Python 3.12 environment on Linux. It checks
   dependencies with `pip check`, imports, installed defaults and `--help`
   outside the checkout. It has no OIDC permission or publishing step.
3. Require successful package CI and review before merging. Existing business
   tests remain in `ci.yml`; no GPU optimization study is started by package CI.

The build and archive checks can also run locally (Python 3.12):

```bash
python -m venv /tmp/auto-tune-build
/tmp/auto-tune-build/bin/python -m pip install --upgrade build twine
# Use an empty dist/ directory for each release build.
/tmp/auto-tune-build/bin/python -m build
/tmp/auto-tune-build/bin/python -m twine check --strict dist/*
/tmp/auto-tune-build/bin/python scripts/check_distributions.py --tag v0.1.0
```

On a Linux platform supported by vLLM, validate the complete installation:

```bash
python -m venv /tmp/auto-tune-install
/tmp/auto-tune-install/bin/python -m pip install --upgrade pip
/tmp/auto-tune-install/bin/python -m pip install dist/*.whl
/tmp/auto-tune-install/bin/python -m pip check
check_script="$PWD/scripts/check_installed_package.py"
(cd /tmp && /tmp/auto-tune-install/bin/python -I "$check_script")
```

To additionally check source installation, repeat in another fresh environment
with `dist/*.tar.gz` in place of `dist/*.whl`. macOS or a machine without
compatible vLLM dependencies may only permit partial checks. A `--no-deps`
installation does **not** validate the full dependency set, even if imports and
CLI help succeed. Build and installation checks do not validate GPU execution
or compatibility of every vLLM/GuideLLM version combination.

## Publish v0.1.0 after merge (maintainer)

Complete the one-time setup and merge the approved PR with package CI passing.
Then tag the merged release commit from an up-to-date, clean `main` checkout:

```bash
git switch main
git pull --ff-only origin main
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

Pushing the tag is the publication trigger. `publish.yml` only runs on pushed
`v*` tags. Its build job rejects a tag unless it is **exactly** `v` followed by
the literal `project.version`, and rejects a different Python package version.
It builds and validates both distributions, then passes them as an artifact to
a separate `pypi` environment job. Only that job has `id-token: write`; it uses
`pypa/gh-action-pypi-publish@release/v1` without permanent credentials.
Approve the environment deployment if required, and inspect the workflow and
the [PyPI project](https://pypi.org/project/auto-tune-vllm/) after publication.

For later releases, substitute the new version throughout. PyPI versions/files
cannot be overwritten: fix a failed release with a new version and release PR
when an upload has already succeeded. There is no manual workflow dispatch or
PR publication path.
