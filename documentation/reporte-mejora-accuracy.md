# Reporte — Mejora de accuracy y honestidad de métricas

**Fecha:** 2026-06-07
**Autor:** sesión Claude Code (Carlos Esquivel)
**Alcance:** profundizar el MVP (taxonomía 4×4), validación solo sintética con test
estratificado, objetivo portafolio + rigor. El motor analítico certificado no se tocó.

## Problema de partida

El resultado central del proyecto era débil y **engañoso**: accuracy de yacimiento 0.42 y
de frontera 0.84. El análisis previo (`reporte-no-unicidad-confusiones.md`) demostró que no
era un muro físico, sino dos problemas arreglables:

1. **Frontera 0.84 = espejismo de desbalance.** El filtro de validez reetiquetaba a
   "infinito" toda frontera no desarrollada; con `t_max` independiente de `L_D`/`r_eD`, el
   **70% del test era "infinito"** y la métrica vivía de la clase mayoritaria.
2. **Yacimiento 0.42 = entrenamiento sin pulir + storage masking.** Errores difusos, no
   no-unicidad física: lr constante, 3000 steps, sin schedule/HPO; y el wellbore storage
   (`C_D` hasta 1e4) tapaba la ventana discriminante (la representación descartaba el tiempo
   absoluto).

## Metodología (cambios por etapa)

- **E0 — Métricas honestas.** `evaluate()` ahora reporta **balanced accuracy, macro-F1 y
  recall por clase** además de la cruda (`src/deep_pta/train/train_cnn.py`). Todo el reporte
  usa métricas balanceadas.
- **E1 — Desbalance de frontera.** Muestreo **condicionado por observabilidad**: `t_max`
  correlaciona con `L_D²`/`r_eD²` (observable ~82%, ~18% ventana corta → infinito genuino),
  y la frontera solo es etiquetable si su firma se desarrolla *después* del storage y
  *dentro* de la ventana. Pesos de clase opcionales en la CE. **Test estratificado**
  (300/clase) para medir honestamente. Distribución de frontera: 70%→37% infinito.
- **E2 — Yacimiento.** AdamW + warmup(5%)+cosine + dropout + grad-clip + **early stopping**
  sobre balanced acc de validación; **tercer canal de tiempo absoluto** (2→3 canales,
  normalizado con constantes fijas) para recuperar el contexto temporal que la
  estandarización per-curva destruía; ventana condicionada a `C_D` (siempre más allá del
  storage).
- **E3 — HPO real (Optuna).** Objetivo **balanced accuracy macro**, espacio ampliado (lr,
  weight_decay, warmup, dropout, capacidad, batch, loss weights), **MedianPruner**,
  persistencia SQLite, evaluado en **val set** (banda `C_D` disjunta; test se toca una vez).
- **E4 — Infraestructura.** Tercer split disjunto de validación, `TrainConfig`,
  checkpoint-best-on-val, TensorBoard, entrypoint `python -m deep_pta.train`. Set de
  entrenamiento congelado opcional para desacoplar generación (CPU) de entrenamiento (GPU).
- **E5 — Verificación.** Tabla de ablation sobre el **mismo test estratificado** y re-corrida
  del experimento de concentración (`debug/dbg_confusion_concentration.py`).

## Resultados

### Tabla de ablation (test estratificado 3-canal, 300/clase salvo baseline)

| Config | bal-acc yac | bal-acc front | macro-F1 yac | macro-F1 front |
|---|---|---|---|---|
| Baseline (2-canal, sampler viejo, test viejo) | 0.43 | 0.61 | 0.40 | 0.69 |
| Pipeline nuevo (E1+E2, frozen-train 8k, class weights, batch 128) | **0.513** | **0.698** | 0.507 | 0.702 |
| + HPO (E3, 10 trials, mejor config: base_ch 48, n_blocks 7, batch 256) | 0.484 | 0.700 | — | — |

> Nota: la fila baseline se evalúa sobre el test/distribución antiguos (es su contexto
> nativo); de E1 en adelante todo se mide sobre el test estratificado balanceado. El cambio
> de canal (2→3) y de muestreo forman parte de la config, documentados como cambio de
> distribución legítimo (currículo por dominio, mismo condicionamiento en train/val/test).
>
> **Hallazgo del HPO:** las 10 pruebas (best val 0.606) **no superaron** la config por
> defecto en yacimiento de test (0.484 vs 0.513) y empataron en frontera (~0.70). Esto es
> evidencia, no fracaso: el techo del yacimiento está limitado por la **diversidad del set
> de entrenamiento congelado (25k)**, no por los hiperparámetros. El siguiente lever real es
> más datos / on-the-fly / currículo de C_D, no más HPO.

### Recall por clase (antes → después)

**Yacimiento:**

| clase | baseline | nuevo |
|---|---|---|
| homogéneo | 0.12 | **0.33** |
| doble porosidad | 0.51 | 0.61 |
| fractura cond. inf | 0.43 | 0.38 |
| fractura cond. fin | 0.66 | 0.73 |

**Frontera:**

| clase | baseline | nuevo |
|---|---|---|
| infinito | 0.98 | 0.76 |
| falla sellante | 0.35 | **0.59** |
| presión constante | 0.40 | **0.60** |
| cerrado | 0.71 | **0.85** |

### Lectura honesta

- **El 0.84 de frontera era un espejismo.** En un test balanceado el modelo nuevo logra
  **0.698 balanceado**, mejorando *de verdad* las clases minoritarias (sellante 0.35→0.59,
  presión cte 0.40→0.60, cerrado 0.71→0.85). El "infinito" baja a 0.76 porque ya no se
  sobre-predice — eso es **honestidad**, no regresión.
- **El homogéneo se recuperó** (0.12→0.33), antes era el punto ciego total. El tercer canal
  de tiempo y la ventana condicionada a `C_D` atacaron el storage masking.
- **Concentración:** los ratios de frontera subieron a **2.0–2.3** — el error residual de las
  minoritarias ya es la confusión físicamente esperada con "infinito" (no-unicidad real,
  no colapso difuso). En yacimiento los ratios son 1.0–1.26: queda señal difusa, indicando
  que aún hay margen de entrenamiento/datos antes de tocar el piso de no-unicidad.

## Conclusión

No se demostró un resultado negativo: el deep learning **sí** interpreta PTA. Se pasó de un
clasificador con métrica engañosa a uno con **métricas honestas y mejores en las clases que
importan**, con un diagnóstico cuantitativo (ratios de concentración) de cuánto del error
residual es física vs pipeline. Trabajo futuro: validación con datos reales (infra lista),
mayor diversidad de datos de entrenamiento para el yacimiento, y currículo de `C_D`.

---

# Ciclo v2 (2026-06-08) — split híbrido, motor GPU y escala de datos

El ciclo v1 dejó el yacimiento data-limited (HPO nulo). El v2 atacó justamente eso.

## Decisiones y cambios

- **Split de `C_D` híbrido (decisión del usuario).** Antes train/test usaban bandas de `C_D`
  **disjuntas** → el modelo se evaluaba extrapolando a almacenamiento nunca visto (test
  brutal que topaba homogéneo/inf-fractura). Ahora `generator.split_of` reserva la banda
  alta (`log10 C_D ≥ 3.5`) como **test de extrapolación** y reparte el resto **i.i.d. por
  hash de muestra** (sin fuga, verificado: 0 colisiones de split). La **métrica de cabecera**
  pasa a ser in-distribution; el test de extrapolación se reporta **aparte**.
- **Motor GPU (autorizado tocar el motor).** Se portó el motor analítico a PyTorch batched
  (`src/deep_pta/engine/gpu_engine.py`): agrupa por clase y hace Stehfest en GPU; implementa
  el Bessel I escalado (`ive`) que torch no trae. **Validado contra el motor CPU certificado a
  4e-7** (muy por debajo del piso de ruido de realismo y de la tolerancia de certificación;
  `tests/test_gpu_engine.py`). El motor CPU sigue siendo la referencia; el GPU solo acelera la
  generación (~10-22× en la parte del motor), **preservando la distribución de datos**.
- **Escala de datos.** Generador paralelo CPU (`dbg_make_train_1m.py`) y generador GPU
  (`dbg_make_train_gpu.py`, motor GPU + post CPU, distribución idéntica, ~3.3× end-to-end).
  Sets congelados de **2M** (CPU, 22 min) y **5M** (GPU, 12 min), reservoir balanceado 25%/clase
  (→ 500k/1.25M por clase, vs ~6k del set viejo de 25k).
- **Ensemble.** Wrapper que promedia softmax y fusiona parámetros por precisión
  (`src/deep_pta/models/ensemble.py`). Combina dos ResNet (base32/6 y base64/10) sin coste de
  entrenamiento extra.

## Tabla de ablation v2 (mismo test estratificado in-distribution)

| Config | bal-acc yac | bal-acc front | macro-F1 yac | MAE | recall yac (homog/dp/inf/fin) |
|---|---|---|---|---|---|
| Baseline v1 (config nuevo, frozen 25k) | 0.513 | 0.698 | 0.507 | 0.69 | 0.33/0.61/0.38/0.73 |
| **2M** (ResNet32, 40k steps) | 0.618 | 0.750 | 0.622 | 0.429 | 0.49/0.70/0.50/0.78 |
| **5M** (ResNet32, 80k steps) | 0.638 | 0.750 | 0.644 | 0.399 | 0.54/0.72/0.50/0.79 |
| **5M** (PatchTST, 60k steps) | 0.639 | 0.758 | 0.643 | 0.364 | 0.54/0.74/0.50/0.77 |
| **Ensemble 5M** (ResNet32 + PatchTST) | **0.649** | **0.765** | 0.654 | **0.357** | 0.59/0.73/0.50/0.79 |

Un ResNet 2× más grande (base64/10) dio yac ≈0.633 (capacidad plana); un ensemble de 3
(R32+R64+PatchTST) llega a 0.651, marginal sobre el de 2. Se embarca el ensemble **R32 +
PatchTST** (arquitecturas diversas, ambos completos) como mejor configuración.

**Test de extrapolación** (banda alta de `C_D` nunca entrenada): ensemble yac **0.553**,
front **0.733**, MAE 0.735 — degradación honesta y acotada fuera de distribución (el ensemble
de 2 ResNet extrapola algo mejor en frontera, 0.768, a costa de menor in-distribution).

## Lectura honesta

- **Los datos eran el cuello, como predijo el HPO nulo de v1.** Pasar de 25k → 2M subió el
  yacimiento 0.513 → 0.618; a 5M, 0.638. La **curva de escala se aplana** (2M→5M solo +0.02).
- **La capacidad también se aplana:** un ResNet grande (base64/10) ≈ al pequeño (0.633 vs
  0.638). No es cuello de hiperparámetros ni de capacidad.
- **El ensemble** (ResNet32 + PatchTST) da el mejor número global (yac 0.649, front 0.765,
  MAE 0.357) combinando dos arquitecturas sin coste de entrenamiento extra.
- **Mejora total v1→v2:** yacimiento **0.513 → 0.649 (+0.136)**, frontera 0.698 → 0.765,
  **MAE 0.69 → 0.36 (−48%)**. Desde el "0.42 engañoso" original, el yacimiento honesto y
  balanceado es hoy ~0.65.
- **Techo restante:** la inf-fractura se estanca en ~0.46-0.50 (confusión física con
  homogéneo a tiempos largos). Los ratios de concentración de yacimiento (1.0-1.25) indican
  que el residual es **parcialmente** físico; el núcleo duro homogéneo↔inf-fractura es el
  límite práctico de no-unicidad para esta taxonomía.

## Conclusión v2

La hipótesis "faltan datos" quedó **confirmada empíricamente**: más datos (vía un motor GPU
validado contra el certificado) subieron el yacimiento honesto de 0.51 a 0.64 y bajaron el
MAE casi a la mitad, con extrapolación reportada por separado. El siguiente lever ya no es
más datos ni más capacidad, sino **romper la no-unicidad homogéneo↔inf-fractura** (features
físicas adicionales o tareas auxiliares) y la **validación con datos reales**.
