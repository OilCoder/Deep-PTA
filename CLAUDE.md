# Deep PTA

Neural network that interprets pressure transient tests: reads the Δp + Bourdet
derivative curve (log-log) and diagnoses the reservoir model, the boundary, and the
key parameters (k·h, S, C, ω, λ, L, x_f) — the senior interpreter's job, automated.
Includes an interactive app (CSV → diagnosis) and an optional LLM narrator agent.

Design source of truth: `documentation/deep-pta.md` (project overview) and
`documentation/deep-pta-interpretacion-pruebas-presion.md` (full design — data,
formulas, model, limitations, closed decisions).

## How this project is governed

This project uses the four-layer model from `claude-project-base`:

| Layer | Where | Behavior |
|---|---|---|
| **Rules** | `.claude/rules/*.md` | Advisory context, loaded into every session |
| **Skills** | `.claude/skills/*/SKILL.md` | On-demand workflows (`/checkpoint`, `/bug-fix`, …) |
| **Agents** | `.claude/agents/*.md` | Review/design in a fresh context |
| **Hooks** | `.claude/settings.json` + `.claude/hooks/*.sh` | Deterministic enforcement |

Rules guide. Skills orchestrate. Agents review or design in isolation. Hooks enforce.
Start at `.claude/rules/project-guidelines.md` for the full index.

## Language conventions

- **Code** (identifiers + comments): **English**.
- **Docstrings**: **NumPy style** (NumPy/SciPy/PyTorch ecosystem).
- **Plans and bitácoras** (`todo/`): **Spanish** (user's working language).
- **Study notes** (`aprendizaje/`): **Spanish** prose, English technical terms.
- **Commits**: Conventional Commits subset (9 prefixes, includes `learn:`) — see
  `.claude/rules/commit-style.md`.

## Stack and environment

- Python 3.11+ (managed with `uv`), NumPy / SciPy / PyTorch.
- Analytical engine: Laplace-space solutions + Bessel `K₀/K₁` (`scipy.special`) +
  Stehfest inversion. No commercial simulator (OPM Flow optional, later phase).
- HPO: Optuna. Tracking: W&B / TensorBoard. App: Gradio or Streamlit.
- Engine validation: `AnaFlow` / `welltestpy` + published Bourdet/Gringarten type curves.
  A GPU-batched port (`gpu_engine.py`, torch) is validated to 4e-7 against the
  certified CPU engine and generates the million-curve training sets.
- Hardware: NVIDIA RTX 4080 (CUDA) for training and data generation, on WSL2.
- Environment: `uv`-managed `.venv` (use `.venv/bin/{python,pytest,mypy,ruff}`).
  No Docker config in this repo yet, despite the cross-project convention.

## Verification gate (mandatory)

No task is complete until verification passes (`.claude/rules/verification.md`):

```text
test:        pytest -q
type-check:  mypy src/
lint:        ruff check .
format:      ruff format --check .
```

Fase 1 has a stronger gate: the analytical engine must be **certified against
published type curves and AnaFlow/welltestpy** before any model is trained.

## Folder convention

Minimum scaffold (`.claude/`, `todo/`, `documentation/`, `aprendizaje/`, `docs/`).
Code folders (`src/`, `tests/`, `data/`, `models/`, `outputs/`, `debug/`) emerge per
phase, not up front. `documentation/` = code docs (never `docs/`); `docs/` = GitHub
Pages only; `aprendizaje/` = study material (target of `/study`).

## Current state

**Ciclo v2 completo (2026-06-08)** — engine certified (9 type curves + AnaFlow),
GPU engine validated (4e-7), 5M-curve training sets, honest balanced metrics.
Headline (ResNet32+PatchTST ensemble, in-dist stratified test): reservoir bal-acc
**0.649**, boundary **0.765**, MAE **0.357**; extrapolation stress test 0.553/0.733.
Validation mode: `warn`. Next: v3 cycle — physics-informed channels (Δp−Δp′
separation + local slope) to break the homogeneous↔inf-fracture confusion.
See `todo/PLAN.md` and `documentation/reporte-mejora-accuracy.md`.
