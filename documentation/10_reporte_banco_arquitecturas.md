# Reporte — Banco comparativo de arquitecturas (ciclo v3, 2026-06-10)

Evaluación de seis arquitecturas neuronales interpretando pruebas de presión bajo
condiciones idénticas, sobre la entrada de 5 canales validada en el prototipo v3.
Este reporte documenta la metodología, los resultados y — central para este
proyecto — **cómo cada necesidad de la ingeniería de yacimientos está representada
en el código**.

## 1. Del yacimiento al tensor: qué representa cada pieza del código

La pregunta de diseño de todo el proyecto es: ¿cómo se convierte el problema del
intérprete (leer una derivada de Bourdet) en un problema de aprendizaje bien
planteado? Cada requisito físico tiene una contraparte concreta en el código:

| Necesidad de ingeniería de yacimientos | Representación en el código |
|---|---|
| Las soluciones de la ecuación de difusividad (4 yacimientos × 4 fronteras, con C y S acoplados en la frontera interna) | `src/deep_pta/engine/` — Laplace + Stehfest + Bessel K₀/K₁; certificado contra type curves publicadas y AnaFlow (`tests/test_engine_typecurves.py`, `test_engine_vs_anaflow.py`) |
| Los datos de campo son ruidosos: gauge, deriva, truncamiento, outliers | `src/deep_pta/data/realism.py` — capa de realismo aplicada DESPUÉS del motor limpio |
| El diagnóstico se hace sobre la derivada de Bourdet con suavizado variable | `src/deep_pta/data/bourdet.py` — ventana L ∈ [0.1, 0.3] sorteada por curva, como elige un intérprete |
| El lenguaje del diagnóstico son las pendientes log-log y la separación Δp−Δp′ | `src/deep_pta/data/representation.py` — canales `slope` (pendiente local) y `sep` (separación con normalización FIJA: el nivel absoluto es la señal) |
| El storage enmascara la firma temprana (`t_fin ≈ 50·C_D`) | canal de tiempo absoluto + test de **extrapolación** (banda `log10 C_D ≥ 3.5` jamás entrenada, `generator.split_of`) |
| Una frontera no desarrollada en la ventana ES "infinito" para el intérprete | filtro de validez del muestreo (`src/deep_pta/data/sampling.py`) — la regla de dominio codificada como reetiquetado |
| Cada modelo tiene SUS parámetros (ω,λ solo en doble porosidad; F_CD solo en fractura finita…) | máscara de parámetros activos (`active_param_mask`) + NLL gaussiana enmascarada (`models/losses.py`) |
| El intérprete reporta incertidumbre, no certezas | cabeza de log-varianza heterocedástica + confianza por entropía del softmax (`app/inference.py`) |
| La no-unicidad es del problema, no del modelo | ratio de concentración de confusiones (`debug/dbg_confusion_concentration.py`) — separa error físico de error de pipeline |

## 2. Metodología del banco

- **Entrada común**: 5 canales × 256 puntos (log Δp, log Δp′, log t, separación,
  pendiente), validada con prototipo dual-seed (+0.025 yacimiento, +0.038
  extrapolación vs 3 canales).
- **Presupuesto idéntico**: 40k pasos, batch 256, AdamW + warmup/cosine, early
  stopping por balanced accuracy en validación, sobre el MISMO superset congelado de
  2M de curvas (float16, round-trip validado: error máx 0.0039 ≈ piso de ruido).
- **2 semillas** por arquitectura; el test estratificado (300/clase, 1200 curvas) se
  toca una vez por fila. La extrapolación NUNCA se usa para seleccionar.
- **Finalistas**: las 2 mejores re-entrenadas a 80k pasos sobre 5M de curvas.
- Driver: `debug/dbg_train_ablation.py --arch {resnet1d,patchtst,patchtst_big,inception,tcn,moe}`;
  filas JSON en `outputs/ablation/bench_*.json`; agregación en
  `debug/dbg_bench_analysis.py`.

## 3. Resultados (media de 2 semillas)

| Arquitectura | Params | bal-acc yac | bal-acc front | MAE | extrap yac |
|---|---|---|---|---|---|
| PatchTST-grande | 6.35M | **0.6391** | **0.7621** | **0.3223** | 0.5262 |
| TCN | 0.27M | 0.6341 | 0.7558 | 0.3782 | 0.5775 |
| PatchTST | 0.16M | 0.6212 | 0.7404 | 0.4101 | 0.5483 |
| InceptionTime | 0.42M | 0.6209 | 0.7429 | 0.4093 | 0.5687 |
| ResNet-1D | 0.24M | 0.6096 | 0.7446 | 0.4276 | **0.5821** |
| MoE (4 expertos, gating denso) | 0.12M | 0.5625 | 0.7042 | 0.5437 | 0.4696 |

**Finalistas a 5M (80k pasos, 1 semilla):**

| Finalista | bal-acc yac | bal-acc front | MAE | extrap yac | extrap MAE |
|---|---|---|---|---|---|
| **TCN (embarcado)** | 0.6317 | 0.7708 | 0.3399 | **0.6233** | **0.3631** |
| PatchTST-grande | **0.6517** | **0.7833** | **0.2781** | 0.5125 | 0.8924 |

## 4. Hallazgos

1. **La entrada rica desbloqueó la capacidad.** Con 3 canales (v2), ResNet64 (15.4M)
   ≈ ResNet32 (0.24M). Con 5 canales, el PatchTST de 6.35M lidera in-dist y rompe el
   récord de MAE (0.322). La capacidad no sobraba: la entrada era el límite.
2. **Robustez OOD e in-dist son ejes distintos.** El Transformer grande pierde 11
   puntos al extrapolar a storage alto; el TCN pierde ~1 — y a 5M la brecha se agudiza: el finalista grande marca récords in-dist (0.652/0.783/MAE 0.278) pero colapsa a 0.513/MAE 0.892 al extrapolar (su sesgo inductivo —
   convoluciones dilatadas + el canal slope invariante a log-t — viaja mejor). Por
   eso el TCN es el intérprete embarcado (`models/best.pt`,
   `app/inference.py::DEFAULT_*`).
3. **El MoE es un resultado negativo con diagnóstico.** Gating denso estilo Jacobs,
   4 expertos: quedó último (0.5625) y el análisis de enrutamiento
   (`outputs/bench_moe_routing.png`) muestra compuertas casi uniformes — solo
   fractura-finita capturó un experto (peso 0.55). La especialización por régimen
   que motivaba la hipótesis no emergió a esta escala.
4. **Las clases se reparten por arquitectura** (`outputs/bench_recall_by_class.png`):
   fractura finita es fácil para todas (~0.75-0.79); el par homogéneo↔inf-fractura
   es el que separa contendientes.
5. **vs ciclo v2**: el mejor extrapolador pasó de 0.553/MAE 0.735 (ensemble 3ch) a
   0.623/MAE 0.363 (TCN 5ch) — la mitad del error de parámetros fuera de
   distribución.

## 5. Limitaciones declaradas

- Sin HPO por arquitectura (presupuesto idéntico > óptimo por modelo): los números
  comparan FAMILIAS bajo las mismas reglas, no máximos teóricos de cada una.
- Finalistas a 1 semilla (los screenings son dual-seed).
- El MoE probado es la variante densa a esta escala; un top-k con load-balancing a
  mayor escala queda como pregunta abierta.
- Validación con datos reales pendiente (infra en `src/deep_pta/data/real_cases.py`).

## 6. Reproducibilidad

```bash
# superset 2M (motor GPU, ~6 min en RTX 4080)
python debug/dbg_make_train_gpu.py --total 2000000 --out data/synthetic_train_2m_v3.h5 \
    --extra-channels sep,slope --float16
# banco completo (12 corridas)
bash debug/dbg_run_bench.sh
# finalistas 5M + análisis
bash debug/dbg_run_finalists.sh && python debug/dbg_bench_analysis.py
```

Fuente de cada número: `outputs/ablation/*.json`. Figuras on-brand vía
`docs/assets/deep_pta.mplstyle`.

Ratios de concentración del TCN embarcado (test in-dist): yacimiento 1.27–1.32 en
homogéneo/doble-porosidad/fractura-finita (subió desde ~1.0–1.26 del v2 — el error
residual migró hacia los pares físicamente ambiguos), frontera en su piso físico.
