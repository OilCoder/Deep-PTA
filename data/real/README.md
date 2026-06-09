# Real PTA cases

Real pressure-transient cases for the sim-to-real validation (Fase 4). Each case is a
two-column CSV plus an optional ground-truth JSON.

## Format

- `<case>.csv` — header `t,p` then rows of time and pressure change `(t, Δp)`.
- `<case>.json` — ground truth, e.g.
  `{"reservoir": "homogeneous", "boundary": "sealing fault", "source": "Horne 1995, p.123"}`.

Load with `deep_pta.data.real_cases.load_real_case` / `load_ground_truth`; both feed the
same Bourdet + representation pipeline as the synthetic data.

## How to populate (needs external sources — manual)

1. **Tabulated cases** — transcribe pressure-time tables with published interpretation
   from Lee (1982) `[Lee1982]` and Horne (1995) `[Horne1995]`.
2. **WebPlotDigitizer** — digitize 10-15 classic log-log curves from papers and Bourdet
   (2002) `[Bourdet2002]` (~30-45 min each).
3. **Volve (Equinor, open)** — extract DST pressures from the public reports.

Target: 20-30 cases. See `documentation/04_referencias.md` for the citation keys.

## Files here

- `example_synthetic_case.csv` / `.json` — a **synthetic** example (engine-generated with
  realistic noise) so the pipeline and app can be demonstrated end-to-end. It is *not* a
  real case; it only exercises the CSV → diagnosis flow.
