# Deep PTA

## Goal
Construir una red neuronal que interprete pruebas de presión (curva Δp + derivada de Bourdet en log-log) clasificando modelo de yacimiento y frontera, y estimando los parámetros clave, entrenada con un dataset sintético generado por un motor analítico certificado.

## Stack
| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ (uv) |
| Motor analítico | NumPy, SciPy (`scipy.special` Bessel K₀/K₁, inversión Stehfest) |
| Modelos | PyTorch (ResNet-1D → Transformer 1D a mano) |
| HPO / tracking | Optuna, W&B / TensorBoard |
| App | Gradio o Streamlit |
| Validación motor | AnaFlow / welltestpy + type curves Bourdet/Gringarten |
| Hardware | RTX 4080 (CUDA), Devcontainer + Docker en WSL2 |

## Structure
```
.claude/         → rules, skills, agents, hooks
todo/            → PLAN.md y bitácoras
documentation/   → docs de código + notas de diseño
docs/            → reservado para GitHub Pages
src/             → motor, generador, modelos, app (emerge en Fase 1+)
tests/           → suite pytest (certificación del motor, generadores, modelos)
data/            → test sets congelados, casos reales (Volve, digitalizados)
outputs/         → figuras, matrices de confusión, mapas de atención (gitignored)
models/          → checkpoints entrenados (gitignored)
```

## Phases

### Phase 0 — Arranque y configuración
- [x] Aterrizar el diseño completo (notas en documentation/) (2026-06-04)
- [x] Configurar la base claude-project-base en .claude/ (rules, skills, agents, hooks) (2026-06-05)
- [x] Inicializar git y .gitignore (2026-06-05)
- [ ] Configurar entorno Python: pyproject.toml + uv, deps base (numpy, scipy, pytest, ruff, mypy) (pyproject.toml)
- [ ] Crear devcontainer + Dockerfile con CUDA para WSL2 (.devcontainer/)
- [ ] Escribir nota de estudio con la teoría mínima por fase (documentation/)

### Phase 1 — Motor analítico + generador
- [ ] Implementar inversión Stehfest (N=8-12) (src/engine/stehfest.py)
- [ ] Implementar solución base en Laplace con almacenamiento C y skin S (van Everdingen-Hurst, Agarwal) (src/engine/laplace_base.py)
- [ ] Implementar f(s) por modelo: homogéneo, doble porosidad Warren-Root (src/engine/reservoir_models.py)
- [ ] Implementar fracturas: conductividad infinita (Gringarten) y finita (Cinco-Ley) (src/engine/fractures.py)
- [ ] Implementar fronteras por pozos imagen: falla sellante, presión constante, cerrado (src/engine/boundaries.py)
- [ ] Certificar el motor contra type curves Bourdet/Gringarten publicadas (tests/test_engine_typecurves.py)
- [ ] Certificar el motor contra AnaFlow/welltestpy en el subconjunto común (tests/test_engine_vs_anaflow.py)
- [ ] Implementar muestreo de parámetros log-uniforme con filtro de validez (src/data/sampling.py)
- [ ] Implementar capa de realismo: muestreo temporal, ruido gauge, deriva, truncamiento, outliers (src/data/realism.py)
- [ ] Implementar derivada de Bourdet con ventana de suavizado L variable (src/data/bourdet.py)
- [ ] Implementar representación 1D 2 canales sobre malla log de 256 puntos (src/data/representation.py)
- [ ] Generador on-the-fly (~80k curvas, seed reproducible, splits por rangos disjuntos) (src/data/generator.py)

### Phase 2 — Baseline CNN
- [ ] Implementar ResNet-1D multi-task: 2 cabezas clasificación (yacimiento 4 + frontera 4) + regresión enmascarada (src/models/resnet1d.py)
- [ ] Implementar loss conjunta CE + MSE enmascarado (src/models/losses.py)
- [ ] Loop de entrenamiento con tracking W&B/TensorBoard (src/train/train_cnn.py)
- [ ] Optimización de hiperparámetros con Optuna (src/train/hpo.py)
- [ ] Reportar matriz de confusión + scatter parámetros estimados vs verdaderos (outputs/)

### Phase 3 — Transformer 1D
- [ ] Implementar encoder Transformer a mano: patches, positional encoding, self-attention (estilo PatchTST) (src/models/patchtst.py)
- [ ] Comparación honesta CNN vs Transformer con mismas condiciones (src/train/compare.py)
- [ ] Extraer y visualizar mapas de atención sobre la derivada (outputs/)
- [ ] Redactar post LinkedIn #1 (documentation/)

### Phase 4 — Validación real (sim-to-real)
- [ ] Transcribir casos tabulados de Lee y Horne (data/real/)
- [ ] Digitalizar 10-15 casos clásicos con WebPlotDigitizer (data/real/)
- [ ] Extraer DSTs del dataset Volve (Equinor) (data/real/)
- [ ] Reporte sim-to-real: accuracy sintético vs real (documentation/)
- [ ] Redactar post LinkedIn #2 (documentation/)

### Phase 5 — App + cierre
- [ ] App Gradio: CSV (t,p) → preprocesamiento → derivada → diagnóstico + parámetros + curva ajustada superpuesta (src/app/app.py)
- [ ] Agente LLM narrador opcional (src/app/narrator.py)
- [ ] README final + página GitHub Pages en docs/ (README.md, docs/)
- [ ] Posts finales (documentation/)

## Conventions
- Código e identificadores en inglés; docstrings NumPy style; planes y bitácoras en español.
- Carpetas de código (src/, tests/, data/, models/, outputs/) se crean cuando la fase las necesita, no antes.
- Hito de salida temprana: si el ciclo se corta en Fase 2, generador + clasificador ya son proyecto publicable.
- El filtro de validez del muestreo ES conocimiento de dominio: documentarlo (reclasifica fronteras no desarrolladas como "infinito").
- Alcance MVP: monofásico petróleo, drawdown + buildup vía Agarwal, representación solo 1D.
