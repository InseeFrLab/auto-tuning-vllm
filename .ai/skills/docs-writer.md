# Skill: Docs writer

## Scope

| Audience | Files |
|----------|-------|
| Users | `README.md`, `docs/quick_start.md`, `docs/configuration.md` |
| Examples | `examples/*.yaml` |
| Agents | `.ai/context/*` |

## Rules

1. Runnable commands: `pip install -e .`, `ruff`, `pytest`, `auto-tune-vllm --help` — E2E optimize only in maintainer sections.
2. YAML keys match `StudyConfig` in `core/config.py`.
3. Link GitHub issues instead of long incident writeups.
4. Structural changes → update `docs/architecture.md` per `architecture-diagrams.md`.

## Ray

Legacy user docs live in `docs/ray_*.md`; do not expand Ray in agent context unless deprecating.

## Agents must not document

“Run optimize to verify” as an agent step — see `AGENTS.md`.
