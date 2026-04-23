# Workflow

## 1) Gentle-AI + SDD

- Explore the codebase before making assumptions.
- Understand the architecture before changing it.
- For non-trivial changes, split work into proposal, design, tasks, implementation, and verification.
- Keep changes incremental and traceable.
- Prefer contract tests at boundaries when moving responsibilities.

## 2) Clean Code and SOLID

- Prefer the smallest correct change.
- Keep responsibilities isolated.
- Prefer readability over cleverness.
- Add abstractions only when they reduce real complexity.
- Stop refactoring when the cost outweighs the benefit.

## 3) Imports and Module Boundaries

- Use barrel files only for stable public APIs.
- Prefer direct imports in internal modules when they improve clarity.
- Do not grow `__init__.py` files into hidden dependency hubs.
- Use explicit consumer migration only when the barrel clearly improves usage.
- Keep import structure aligned with ownership: `strategy`, `helpers`, `runtime`, `flow`.
