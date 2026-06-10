# Deep PTA

Neural network that interprets pressure transient tests: reads the Δp + Bourdet
derivative curve (log-log) and diagnoses the reservoir model, the boundary, and the
key parameters (k·h, S, C, ω, λ, L, x_f) — the senior interpreter's job, automated.
Includes an interactive app (CSV → diagnosis) and an optional LLM narrator agent.

Design source of truth: `documentation/01_overview.md` (project overview) and
`documentation/02_diseno_interpretacion_pruebas_presion.md` (full design — data,
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

**Ciclo v3 completo (2026-06-10)** — the architecture benchmark (the project's core
question) is done and published: 6 architectures × 2 seeds under identical
conditions on the 5-channel physics input (Δp−Δp′ separation + local slope).
Large PatchTST wins in-distribution (0.639/0.762/MAE 0.322); the **TCN finalist at
5M is the shipped interpreter** (`models/best.pt`): in-dist 0.632/0.771/MAE 0.340,
**extrapolation 0.623/0.806/MAE 0.363** (near in-dist). MoE = published negative
result (flat gates). Bilingual results site (report + decision logbook) in `docs/`.
`todo/`, `aprendizaje/`, `debug/`, `.claude/` are local-only (not published).
Validation mode: `warn`. Next: real-data validation (`real_cases.py` ready).
See `documentation/10_reporte_banco_arquitecturas.md`.
