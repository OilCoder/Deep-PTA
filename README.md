# Deep PTA

Neural network that interprets pressure transient tests (PTA): it reads the Δp +
Bourdet derivative curve (log-log) and diagnoses the reservoir model, the boundary,
and the key parameters (k·h, S, C, ω, λ, L, x_f) — the senior interpreter's job,
automated. Trained on a 100% synthetic dataset produced by a certified analytical
engine (Laplace solutions + Stehfest inversion). Ships with an interactive app
(CSV → diagnosis) and an optional LLM narrator agent.

Design source of truth: `documentation/deep-pta.md`,
`documentation/deep-pta-interpretacion-pruebas-presion.md`,
`documentation/plan-implementacion.md`, and `documentation/referencias.md`.

## Status

Under construction. See `todo/PLAN.md` for the phase plan and progress.
