# Reporte — ¿No-unicidad física o pipeline sin pulir?

**Fecha:** 2026-06-07
**Autor:** sesión Claude Code (Carlos Esquivel)
**Experimento:** `debug/dbg_confusion_concentration.py`
**Insumos:** `models/cnn_baseline.pt`, `models/patchtst.pt`, `data/synthetic_test.h5` (1200 casos)

## Pregunta que motiva el experimento

Tras observar una accuracy de yacimiento de **0.42** y de frontera de **0.84**, surge la
duda de fondo: *¿demostramos que el deep learning no puede interpretar pruebas de presión,
o lo que construimos simplemente no está lo bastante pulido?*

La hipótesis a contrastar: si el modelo aprendió física real, sus errores deberían
**concentrarse en los pares de regímenes físicamente no-únicos** (curvas que producen
derivadas de Bourdet casi idénticas). Si los errores están **repartidos al azar**, el
problema es de pipeline/entrenamiento, no un muro físico.

## Método

Para cada clase verdadera se mide:

- `recall` — fracción correctamente clasificada (diagonal de la matriz de confusión).
- `%err→ambig` — qué fracción de los errores cae en clases físicamente ambiguas con la verdadera.
- `baseline` — esa fracción esperada si el error fuera uniforme entre las clases incorrectas.
- `ratio = %err→ambig / baseline` — **el indicador clave**:
  - `ratio ≫ 1` → errores concentrados en pares físicos → no-unicidad real (el modelo aprendió).
  - `ratio ≈ 1` → errores difusos → artefacto de pipeline/entrenamiento.
  - `ratio < 1` → errores que *evitan* los pares físicos → sesgo artificial del modelo.

Pares físicamente no-únicos declarados (conocimiento de dominio):

- **Yacimiento:** homogéneo↔fractura cond. infinita (storage temprano tapa la media pendiente),
  homogéneo↔doble porosidad (dip de interporosidad débil/no visto),
  doble porosidad↔fractura cond. infinita, fractura cond. infinita↔fractura cond. finita (½ vs ¼).
- **Frontera:** infinito↔{falla sellante, presión constante, cerrado} (frontera no desarrollada en la ventana).

## Resultados crudos

### CNN (acc_res = 0.424, acc_bnd = 0.838)

```
[RESERVOIR]   support: homogeneous=308, double_poros=290, inf_fracture=311, fin_fracture=291
true class      recall   off  %err→ambig baseline ratio
homogeneous       0.12   271     0.81     0.67    1.22
double_poros      0.51   143     0.64     0.67    0.95
inf_fracture      0.43   177     1.00     1.00    1.00
fin_fracture      0.66   100     0.49     0.33    1.47

[BOUNDARY]    support: infinite=836, sealing_fault=113, const_press=107, closed=144
infinite          0.98    16     1.00     1.00    1.00
sealing_fault     0.35    73     0.97     0.33    2.92
const_press       0.40    64     0.94     0.33    2.81
closed            0.71    42     1.00     0.33    3.00
```

### Transformer (acc_res = 0.413, acc_bnd = 0.833)

```
[RESERVOIR]
homogeneous       0.19   250     0.59     0.67    0.89
double_poros      0.43   166     0.41     0.67    0.61
inf_fracture      0.28   223     1.00     1.00    1.00
fin_fracture      0.78    65     0.26     0.33    0.78

[BOUNDARY]
sealing_fault     0.32    77     0.96     0.33    2.88
const_press       0.40    64     0.89     0.33    2.67
closed            0.67    47     0.98     0.33    2.94
```

## Hallazgos

### 1. El 0.84 de frontera está inflado por desbalance — confirmado

El test set contiene **836 de 1200 casos (70%) de clase "infinito"**, producto del filtro de
validez que reetiqueta como infinito toda frontera no desarrollada en la ventana de tiempo.
El modelo acierta el 98% de esa mayoría y eso arrastra el promedio hacia arriba. El promedio
no ponderado de las cuatro clases ronda **0.61**, no 0.84. En las fronteras reales
(minoritarias) el skill es pobre: sellante 0.35, presión constante 0.40.

### 2. Los errores de frontera son físicamente coherentes, pero confundidos con el desbalance

`ratio ≈ 2.8–3.0` (el máximo posible es 3.0): casi el 100% de los errores de frontera van a
"infinito". Es *exactamente* lo que predice la física (frontera no desarrollada ≡ infinito).
**Pero** el desbalance 70/30 empuja en la misma dirección, así que no-unicidad física y sesgo
por desbalance quedan **confundidos** y no se pueden separar con este test. El desbalance es
100% arreglable.

### 3. Los errores de yacimiento son difusos, no concentrados

`ratio` entre 0.95 y 1.47 (CNN), promediando ~1. Hay señal física **débil** (fin_fracture→
inf_fracture `ratio` 1.47 — la confusión ½ vs ¼ de pendiente; homogéneo 1.22), pero no domina.
En el Transformer los `ratio` caen por debajo de 1 (doble porosidad 0.61, fin 0.78) porque el
modelo tiene un **sesgo de sobre-predecir fin_fracture**, volcando masa ahí sin importar la
clase verdadera. Esto **no parece no-unicidad limpia — parece modelo a medio entrenar.**

## Veredicto

| Cabeza | Diagnóstico | ¿Prueba que DL no sirve? | Acción |
|---|---|---|---|
| **Frontera** | No-unicidad real + desbalance 70/30 confundidos | **No** | Rebalancear el muestreo / reportar accuracy por clase y balanceada |
| **Yacimiento** | Errores difusos, señal física débil → *under-polished* | **No** | Más entrenamiento + HPO finalizado; atacar el enmascaramiento por storage temprano |

**No se demostró un resultado negativo.** Lo que hay es: (a) un clasificador de frontera que
funciona pero cuya métrica engaña por desbalance, y (b) un clasificador de yacimiento poco
pulido (3000 steps, sin HPO finalizado), con un piso pequeño de no-unicidad genuina en los
bordes. Ambos problemas son móviles. La literatura de ML aplicado a PTA reporta accuracies de
85–95% en clasificación de modelo, lo que confirma que el techo está muy por encima del 0.42
actual: la distancia es de método, no de imposibilidad física.

## Próximos pasos sugeridos

1. **Frontera (rápido, alto impacto):** rebalancear el muestreo o reportar accuracy
   balanceada + por clase; medir cuánto del 0.84 era la mayoría "infinito".
2. **Yacimiento:** subir presupuesto de entrenamiento, finalizar HPO, e investigar si el
   wellbore storage temprano destruye la ventana discriminante (recortar/ponderar el
   tramo de storage en la representación de entrada).
3. **Repetir este experimento** tras cada cambio para ver si los `ratio` de yacimiento
   suben (señal de que el modelo empieza a capturar la física en vez de adivinar).
