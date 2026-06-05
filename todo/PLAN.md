# Deep PTA

## Goal
Construir una red neuronal que interprete pruebas de presión (curva Δp + derivada de Bourdet en log-log) clasificando modelo de yacimiento y frontera, y estimando los parámetros clave, entrenada con un dataset sintético generado por un motor analítico certificado.

## Stack
| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ (uv) |
| Motor analítico | NumPy, SciPy (`scipy.special` Bessel K₀/K₁, inversión Stehfest), h5py |
| Modelos | PyTorch (ResNet-1D → Transformer 1D a mano) |
| HPO / tracking | Optuna, W&B / TensorBoard |
| App | Gradio o Streamlit |
| Validación motor | AnaFlow / welltestpy + type curves Bourdet/Gringarten |
| Hardware | RTX 4080 (CUDA), Devcontainer + Docker en WSL2 |

## Structure
```
.claude/                      → rules, skills, agents, hooks (gitignored)
todo/                         → PLAN.md y bitácoras
documentation/                → diseño + teoría + bibliografía
  deep-pta.md                 → overview
  deep-pta-interpretacion-*   → diseño completo
  plan-implementacion.md      → plan detallado (fórmulas, mermaid, rangos)
  referencias.md              → bibliografía con claves [clave]
  teoria-pta.md               → repaso teórico por fase
docs/                         → reservado para GitHub Pages
src/deep_pta/
  engine/   → stehfest, laplace_base, reservoir_models, fractures, boundaries, solution
  data/     → sampling, realism, bourdet, representation, generator
  models/   → resnet1d, patchtst, losses
  train/    → train_cnn, hpo, compare
  app/      → app, narrator
tests/      → test_stehfest, test_engine_typecurves, test_engine_vs_anaflow,
              test_sampling, test_bourdet, test_generator, test_train_smoke
data/       → synthetic_test.h5 (gitignored), real/ (casos)
outputs/    → figuras, matrices de confusión, mapas de atención (gitignored)
models/     → checkpoints entrenados (gitignored)
```

## Phases

### Phase 0 — Arranque y configuración (COMPLETED)
- [x] Aterrizar el diseño completo (notas en documentation/) (2026-06-04)
- [x] Configurar la base claude-project-base en .claude/ (rules, skills, agents, hooks) (2026-06-05)
- [x] Inicializar git y .gitignore (2026-06-05)
- [x] Configurar entorno Python: pyproject.toml + uv, deps base (numpy, scipy, h5py, pytest, ruff, mypy, matplotlib) (pyproject.toml) (2026-06-05)
- [x] Escribir nota de estudio con la teoría mínima por fase (documentation/teoria-pta.md) (2026-06-05)
- [x] Reubicar plan-implementacion.md y referencias.md a documentation/ (2026-06-05)
- ~~Crear devcontainer + Dockerfile con CUDA para WSL2~~ (discarded 2026-06-05: infraestructura local opcional, se documenta no se construye; el motor corre en CPU y el entrenamiento usa la RTX 4080 ya disponible en WSL2)

### Phase 1 — Motor analítico + generador (COMPLETED)
- [x] Implementar inversión Stehfest (N=8-12, default 12, Vᵢ cacheados, vectorizado) [Stehfest1970] (src/deep_pta/engine/stehfest.py) (2026-06-05)
- [x] Implementar solución base en Laplace con almacenamiento C y skin S (x=√(u·f(u)), K₀/K₁) [Agarwal1970][vanEverdingenHurst1949][MavorCincoLey1979] (src/deep_pta/engine/laplace_base.py) (2026-06-05)
- [x] Implementar f(u) por modelo: homogéneo, doble porosidad Warren-Root [WarrenRoot1963] (src/deep_pta/engine/reservoir_models.py) (2026-06-05)
- [x] Implementar fracturas: conductividad infinita (½, x_f) [Gringarten1974] y finita (¼, F_CD) [CincoLey1978] (src/deep_pta/engine/fractures.py) (2026-06-05)
- [x] Implementar fronteras por pozos imagen: sellante, presión constante, cerrado [Horne1995][vanEverdingenHurst1949] (src/deep_pta/engine/boundaries.py) (2026-06-05)
- [x] Implementar composición evaluate(model, boundary, well, params, t_D) → (p_wD, dp_wD) (src/deep_pta/engine/solution.py) (2026-06-05)
- [x] Test Stehfest contra transformadas conocidas (1/u→1, 1/u²→t, 1/(u+a)→e^{-at}, 1/√u) (tests/test_stehfest.py) (2026-06-05)
- [x] Certificar pendientes diagnósticas del motor (meseta, ½, ¼, duplicación, caída, unitaria) + referencia Theis [Bourdet1983][Gringarten1974] (tests/test_engine_typecurves.py) (2026-06-05)
- [x] Certificar el motor contra AnaFlow/welltestpy en el subconjunto común (importorskip; corre en local con extra [validation]) (tests/test_engine_vs_anaflow.py) (2026-06-05)
- [x] Implementar muestreo log-uniforme con filtro de validez + etiquetas factorizadas + máscara (src/deep_pta/data/sampling.py) (2026-06-05)
- [x] Implementar capa de realismo: muestreo temporal, ruido gauge, deriva, truncamiento, outliers (src/deep_pta/data/realism.py) (2026-06-05)
- [x] Implementar derivada de Bourdet con ventana L variable + tiempo de Agarwal [Bourdet1989][Agarwal1980] (src/deep_pta/data/bourdet.py) (2026-06-05)
- [x] Implementar representación 1D 2 canales sobre malla log de 256 puntos + normalización (src/deep_pta/data/representation.py) (2026-06-05)
- [x] Generador on-the-fly (seed reproducible, splits por rangos disjuntos, export h5) (src/deep_pta/data/generator.py) (2026-06-05)

### Phase 2 — Baseline CNN (COMPLETED)
- [x] Implementar ResNet-1D multi-task: cabezas yacimiento(4) + frontera(4) + regresión + incertidumbre (log-varianza) [DiscoverAppSci2024][JPSE2021] (src/deep_pta/models/resnet1d.py) (2026-06-05)
- [x] Implementar loss conjunta CE + CE + NLL gaussiana enmascarada (src/deep_pta/models/losses.py) (2026-06-05)
- [x] Smoke test de overfit de 1 batch en CPU (forward/backward + máscaras) (tests/test_train_smoke.py) (2026-06-05)
- [x] Loop de entrenamiento con DataLoader on-the-fly + AMP/CUDA (src/deep_pta/train/train_cnn.py) (2026-06-05)
- [x] Optimización de hiperparámetros con Optuna (src/deep_pta/train/hpo.py) (2026-06-05)
- [x] Entrenar baseline en RTX 4080 + matriz de confusión + scatter params (acc yac 0.42 / front 0.84; outputs/) (2026-06-05)

### Phase 3 — Transformer 1D
- [x] Implementar encoder Transformer a mano: patches, positional encoding, self-attention (estilo PatchTST) [Nie2023] (src/deep_pta/models/patchtst.py) (2026-06-05)
- [x] Comparación honesta CNN vs Transformer con mismas condiciones + mapas de atención (acc yac 0.41; outputs/attention_map.png) (src/deep_pta/train/compare.py) (2026-06-05)
- [ ] Redactar post LinkedIn #1 (documentation/)

### Phase 4 — Validación real (sim-to-real)
- [x] Crear estructura data/real/ (CSV (t,p) + ground truth JSON) + pipeline de inferencia (src/deep_pta/data/real_cases.py, src/deep_pta/app/inference.py) (2026-06-05)
- [ ] Transcribir casos tabulados de Lee y Horne [Lee1982][Horne1995] (data/real/) (requiere fuentes externas)
- [ ] Digitalizar 10-15 casos clásicos con WebPlotDigitizer [Bourdet2002] (data/real/) (requiere fuentes externas)
- [ ] Extraer DSTs del dataset Volve (Equinor) (data/real/) (requiere fuentes externas)
- [x] Reporte sim-to-real: metodología + baseline sintético (documentation/reporte-sim-to-real.md) (2026-06-05)
- [ ] Redactar post LinkedIn #2 (documentation/)

### Phase 5 — App + cierre
- [x] App Gradio: CSV (t,p) → preprocesamiento → derivada → diagnóstico + parámetros + curva ajustada superpuesta (src/deep_pta/app/app.py) (2026-06-05)
- [x] Agente LLM narrador opcional [anthropic SDK] (src/deep_pta/app/narrator.py) (2026-06-05)
- [ ] README final + página GitHub Pages en docs/ (README.md, docs/) (README base creado; Pages pendiente)
- [ ] Posts finales (documentation/)

## Conventions
- Código e identificadores en inglés; docstrings NumPy style con clave [clave] de la fuente; planes y bitácoras en español.
- Diseño de referencia: documentation/plan-implementacion.md (fórmulas, mermaid, rangos) y documentation/referencias.md (bibliografía con claves).
- Carpetas de código (src/, tests/, data/, models/, outputs/) se crean cuando la fase las necesita, no antes.
- Gate de verificación por tarea: pytest -q, mypy src/, ruff check ., ruff format --check .
- Gate reforzado de Fase 1: el motor debe estar certificado (Stehfest analítico + pendientes diagnósticas; AnaFlow donde haya red) ANTES de entrenar cualquier modelo.
- Hito de salida temprana: si el ciclo se corta en Fase 2, generador + clasificador ya son proyecto publicable.
- El filtro de validez del muestreo ES conocimiento de dominio: documentarlo (reclasifica fronteras no desarrolladas como "infinito").
- Alcance MVP: monofásico petróleo, drawdown + buildup vía Agarwal, representación solo 1D.
