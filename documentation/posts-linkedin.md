# Posts de LinkedIn — Deep PTA

Borradores para difusión del proyecto. Tono: honesto, técnico, sin exagerar resultados.

---

## Post #1 — CNN vs Transformer y por qué la métrica importa

**Gancho:** "Mi modelo tenía 84% de accuracy. Era mentira. Así lo descubrí."

Construí una red neuronal que interpreta pruebas de presión (PTA): lee la curva Δp +
derivada de Bourdet en log-log y diagnostica el modelo de yacimiento y la frontera —
el trabajo de un intérprete senior.

El clasificador de frontera marcaba **0.84 de accuracy**. Sonaba genial. Hasta que miré
el soporte por clase: el **70% del test era "frontera infinita"**. El modelo simplemente
apostaba a la clase mayoritaria y acertaba el 98% de ella. En las fronteras que de verdad
importan (falla sellante, presión constante) acertaba 35-40%.

Lección: **la accuracy cruda miente bajo desbalance.** Cambié a *balanced accuracy*
(recall promedio por clase) y a un test estratificado. El 0.84 se volvió 0.61 honesto.

Luego arreglé la causa: el muestreo generaba fronteras que nunca se desarrollaban en la
ventana de tiempo y se reetiquetaban como "infinito". Condicioné la ventana a la física
del problema. Resultado en test balanceado: sellante 0.35→0.59, presión cte 0.40→0.60.

No es magia. Es medir bien y arreglar la causa raíz.

`#MachineLearning #PetroleumEngineering #WellTesting #DeepLearning`

---

## Post #2 — ¿No-unicidad física o pipeline sin pulir?

**Gancho:** "¿El deep learning falló, o yo no lo entrené bien? Lo medí en vez de adivinar."

El clasificador de yacimiento de mi proyecto de PTA daba 42% (4 clases, azar 25%).
Mejor que el azar, pero pobre. La pregunta de fondo: ¿es un muro físico (curvas que de
verdad se parecen) o un pipeline a medio pulir?

En vez de discutirlo, lo **medí**: para cada clase, ¿los errores se concentran en los
pares físicamente ambiguos (no-unicidad real) o están repartidos al azar (problema de
entrenamiento)? Definí un ratio: errores-en-pares-ambiguos / esperado-por-azar.

- ratio ≫ 1 → el modelo aprendió física; lo que queda es no-unicidad real.
- ratio ≈ 1 → error difuso → falta entrenamiento/datos.

Los errores eran **difusos**. Así que no era un muro: era pipeline. Añadí un canal de
tiempo absoluto (el storage temprano tapaba la firma), AdamW con schedule, early stopping
y HPO honesto sobre balanced accuracy. El homogéneo pasó de 0.12 a 0.33 de recall.

El deep learning **sí** puede con PTA. El error más caro no fue del modelo: fue creer una
métrica sin auditarla.

`#MachineLearning #PetroleumEngineering #ReservoirEngineering #MLOps`
