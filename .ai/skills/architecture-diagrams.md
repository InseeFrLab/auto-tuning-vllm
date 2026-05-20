# Skill: Architecture diagrams

## User doc (source of truth)

**[`docs/architecture.md`](../../docs/architecture.md)** — Mermaid diagrams for contributors and users. Also linked from `README.md`.

Agent context: [`.ai/context/architecture.md`](../context/architecture.md) (prose only).

## When to update `docs/architecture.md`

| Change | Section |
|--------|---------|
| New package / module layout | Repository layout |
| Study or trial flow | End-to-end flow, Study orchestration, Single trial lifecycle |
| Storage / logs | Outputs per study |
| Import graph | Module dependencies |

## Rules

1. Use real module paths (`study_controller.py`).
2. Default path = **local** backend; Ray at most one sentence, no extra diagrams.
3. Mermaid only (GitHub renders natively).
4. PR: note “updated docs/architecture.md” when structure changes.

## Do not

- Duplicate full diagrams under `.ai/context/`.
- Run `auto-tune-vllm optimize` to validate diagrams.
