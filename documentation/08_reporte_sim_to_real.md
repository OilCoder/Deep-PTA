# Reporte sim-to-real

Documento de validación de la Fase 4: cómo se mide el desempeño del modelo y qué tan
bien transfiere del dataset sintético a casos reales. Las claves `[clave]` remiten a
`documentation/04_referencias.md`.

## 1. Métricas

Para cada cabeza se reporta:

- **Accuracy de clasificación** de modelo de yacimiento (4 clases) y frontera (4 clases),
  sobre el **test set sintético congelado** (`data/synthetic_test.h5`, split por rango
  disjunto de `C_D`) y sobre los **casos reales** (`data/real/`).
- **MAE de regresión** sobre los parámetros activos (enmascarados por clase).
- **Confianza** (1 − entropía normalizada de la softmax) como señal de fuera-de-taxonomía.

Las matrices de confusión por cabeza se guardan en `outputs/` (`cm_*_reservoir.png`,
`cm_*_boundary.png`) y las métricas numéricas en `outputs/metrics.json`.

## 2. Resultados sintéticos (baseline)

Entrenamiento en RTX 4080 (CUDA), generación on-the-fly, 3000 pasos/modelo, test set
congelado de 1200 muestras con split por banda disjunta de `C_D` (mide generalización,
no solo ruido nuevo). Valores exactos en `outputs/metrics.json`; figuras en `outputs/`
(`cm_*.png`, `scatter_params_cnn.png`, `attention_map.png`).

| Modelo | Acc. yacimiento (4) | Acc. frontera (4) | MAE regresión |
|---|---|---|---|
| CNN ResNet-1D `[DiscoverAppSci2024]` `[JPSE2021]` | 0.424 | 0.838 | 1.036 |
| Transformer PatchTST a mano `[Nie2023]` | 0.413 | 0.833 | 1.009 |

**Lectura honesta del baseline:**

- **Frontera (~0.84)**: razonable, aunque el filtro de validez sesga las clases hacia
  "infinito", así que parte del acierto es de clase mayoritaria (ver `cm_cnn_boundary.png`).
- **Yacimiento (~0.42)** vs azar 0.25: aprende señal pero es claramente débil. Causa
  **física, no un bug**: con almacenamiento `C_D` alto la firma temprana de fractura
  (pendiente ½ lineal, ¼ bilineal) queda **enmascarada** por el flujo de almacenamiento
  (pendiente unitaria), y el test usa justo la banda de `C_D` alto retenida → homogéneo,
  doble porosidad y fracturas se vuelven difíciles de separar. La matriz de confusión
  (`cm_cnn_reservoir.png`) lo muestra.
- CNN y Transformer quedan **a la par** en este presupuesto corto; el Transformer es algo
  mejor en regresión (MAE 1.01 vs 1.04).

**Próximos pasos para subir el yacimiento** (no ejecutados en este baseline):
extender el filtro de validez para reetiquetar fractura→homogéneo cuando el almacenamiento
enmascara la firma (análogo al filtro de frontera); balanceo/sobremuestreo de clases
difíciles; más pasos + HPO con Optuna (`train/hpo.py`); modelo de mayor capacidad.

## 3. Validación real (pendiente de datos externos)

El gap sim-to-real **es un hallazgo publicable, no un fracaso**: el ruido sintético nunca
replica todos los artefactos reales. El protocolo:

1. Poblar `data/real/` (ver `data/real/README.md`): casos tabulados de Lee `[Lee1982]` y
   Horne `[Horne1995]`, digitalización con WebPlotDigitizer de clásicos `[Bourdet2002]`,
   y DSTs de Volve (Equinor). Meta: 20-30 casos con ground truth.
2. Correr el pipeline de inferencia (`deep_pta.app.inference.diagnose`) sobre cada caso
   y tabular aciertos por cabeza + error de parámetros, junto a la confianza.
3. Reportar **ambos** números (sintético y real) y analizar el gap por clase.

**Mitigación de fuera-de-taxonomía**: la cabeza de incertidumbre (entropía de
clasificación + log-varianza de regresión) marca baja confianza cuando el caso real cae
fuera de las clases implementadas.

## 4. Limitaciones declaradas

- Monofásico, drawdown + buildup vía tiempo equivalente de Agarwal `[Agarwal1980]`.
- La fractura de conductividad finita usa una aproximación fiel a la pendiente bilineal
  (¼), documentada en `engine/fractures.py`.
- La derivada depende del suavizado `L`; el modelo se entrena con `L` variable
  `[Bourdet1989]` para ser robusto a esa elección.
