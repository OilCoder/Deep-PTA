# Deep PTA

**Which neural architecture best interprets a well pressure test?** Six networks —
ResNet-1D, PatchTST (small and large), InceptionTime, TCN, and a mixture of
experts — compete under identical conditions at the senior interpreter's job:
reading the Δp + Bourdet derivative curve (log-log) and diagnosing the reservoir
model, the boundary, and the key parameters. Trained on millions of synthetic
curves from a **certified** analytical engine (Laplace solutions + Stehfest
inversion). Ships with an interactive app (CSV → diagnosis).

**Results site (ES/EN):** https://oilcoder.github.io/Deep-PTA/ — report + decision logbook.

## The benchmark (the project's core result)

Identical budget (40k steps on the same frozen 2M-curve set, 5-channel physics
input, validation-based selection, 2 seeds, same stratified test). Two numbers per
model: in-distribution and **extrapolation** to never-trained high wellbore storage.

| Architecture | Params | Reservoir | Boundary | MAE | Extrapolation |
|---|---|---|---|---|---|
| Large PatchTST | 6.35M | **0.639** | **0.762** | **0.322** | 0.526 |
| TCN | 0.27M | 0.634 | 0.756 | 0.378 | 0.578 |
| PatchTST | 0.16M | 0.621 | 0.740 | 0.410 | 0.548 |
| InceptionTime | 0.42M | 0.621 | 0.743 | 0.409 | 0.569 |
| ResNet-1D | 0.24M | 0.610 | 0.745 | 0.428 | **0.582** |
| MoE (dense gating) | 0.12M | 0.563 | 0.704 | 0.544 | 0.470 |

Finalists at 5M curves / 80k steps: the **TCN** reaches 0.632 / 0.771 / MAE 0.340
in-distribution and — the property that matters in the field —
**0.623 / 0.806 / MAE 0.363 under extrapolation**, nearly matching its
in-distribution numbers. It ships as the app's interpreter (`models/best.pt`).

Three findings: (1) **capacity pays only with a rich input** — with the old
3-channel input a 60× larger model gained nothing; with the physics channels the
large Transformer leads in-distribution; (2) **in-distribution and OOD robustness
are different axes** — the big model drops 11 points under extrapolation, the TCN
~1; (3) the **MoE underperformed** and its routing analysis shows why (nearly
uniform gates — a published negative result). Full method, the
reservoir-engineering↔code mapping, and every experiment row:
`documentation/10_reporte_banco_arquitecturas.md` + `outputs/ablation/*.json`.

## The input is half the result

The biggest single jump didn't come from data or capacity but from the
**representation**: per-curve standardization destroyed the pressure/derivative
separation (exactly log₁₀2 in fracture linear flow — the interpreter's classic
discriminator). Restoring it as a fixed-normalization channel, plus a local
log-log-slope channel, beat scaling the training set from 2M to 5M curves.
History of the metric-honesty and data-scale cycles:
`documentation/07_reporte_mejora_accuracy.md`.

## What's inside

- **Certified analytical engine** — 4 reservoir models × 4 boundaries with wellbore
  storage and skin, via Laplace + Stehfest; verified against published
  Bourdet/Gringarten type curves and AnaFlow. A GPU-batched port (validated to
  ~4e-7) generates 5M curves in ~14 min.
- **Domain-aware generator** — gauge noise, drift, truncation; a validity filter
  that relabels undeveloped boundaries as "infinite" (the interpreter's rule,
  codified); hybrid `C_D` split (i.i.d. holdout + extrapolation band).
- **Six architectures** sharing one multi-task head contract (reservoir, boundary,
  masked heteroscedastic parameter regression) and one training loop.
- **Honest evaluation** — balanced accuracy on a class-stratified test, per-class
  recall, in-distribution and extrapolation always reported together, and a
  confusion-concentration diagnostic separating physical non-uniqueness from
  pipeline error.
- **App** — Gradio CSV → diagnosis with fitted-curve overlay and entropy-based
  confidence; optional LLM narrator.

## Run it

```bash
uv sync                                  # or: pip install -e .[dev,ml,app]

pytest -q                                # verification gate (also: mypy src/, ruff check .)

# Generate the 5-channel training superset on the GPU:
python debug/dbg_make_train_gpu.py --total 2000000 --out data/synthetic_train_2m_v3.h5 \
    --extra-channels sep,slope --float16

# Reproduce the benchmark (12 runs) and the finalists:
bash debug/dbg_run_bench.sh && bash debug/dbg_run_finalists.sh
python debug/dbg_bench_analysis.py       # table + routing/attention figures

python -m deep_pta.app.app               # launch the Gradio app
```

## Status

The architecture benchmark — the project's core question — is complete and
published. Next: validation on real cases (Lee/Horne/Volve; the inference path
`src/deep_pta/data/real_cases.py` is ready) and longer-window / deconvolution
levers against the physical non-uniqueness ceiling.
