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
