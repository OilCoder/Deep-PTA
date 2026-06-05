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
- **Commits**: Conventional Commits subset — see `.claude/rules/commit-style.md`.

## Stack and environment

- Python 3.11+ (managed with `uv`), NumPy / SciPy / PyTorch.
- Analytical engine: Laplace-space solutions + Bessel `K₀/K₁` (`scipy.special`) +
  Stehfest inversion. No commercial simulator (OPM Flow optional, later phase).
- HPO: Optuna. Tracking: W&B / TensorBoard. App: Gradio or Streamlit.
- Engine validation: `AnaFlow` / `welltestpy` + published Bourdet/Gringarten type curves.
- Hardware: NVIDIA RTX 4080 (CUDA) for training; Devcontainer + Docker on WSL2.

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

Minimum scaffold (`.claude/`, `todo/`, `documentation/`, `docs/`). Code folders
(`src/`, `tests/`, `data/`, `models/`, `outputs/`, `debug/`) emerge per phase, not
up front. `documentation/` = code docs (never `docs/`); `docs/` = GitHub Pages only.

## Current state

**Fase 0/1** — design fully landed, base configured. Validation mode: `suggest`.
Next: build the analytical engine (Laplace solutions + Stehfest). See `todo/PLAN.md`.
