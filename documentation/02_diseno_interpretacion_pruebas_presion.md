---
title: "Deep PTA: IA que interpreta pruebas de presión (derivada de Bourdet)"
date: 2026-06-04
tags: [idea, portfolio, well-testing, pta, deep-learning, cnn, transformers, sim-to-real, app-interactiva]
type: idea
status: promoted
---

# Deep PTA: IA que interpreta pruebas de presión

## La idea
Una red neuronal que lee la respuesta de presión de una prueba de pozo (curva Δp + **derivada de Bourdet** en log-log) y hace el trabajo del intérprete senior: **clasifica** el modelo de yacimiento y frontera, y **estima** los parámetros clave (k·h, skin, C, ω, λ, distancia a frontera). Envuelto en una app interactiva: subes los datos de tu prueba → diagnóstico + parámetros + interpretación narrada.

> [!info] Arco narrativo del perfil
> "En 2014 corría pruebas P/T en campo con wireline. En 2016 construí el software de intrusión de agua en MATLAB. En 2026 entrené una IA que interpreta las pruebas." Es el proyecto que une la experiencia de campo + el software MATLAB + el stack ML actual. Nadie más puede contar esta historia.

## Problema que resuelve
- La interpretación de PTA es experta, lenta y subjetiva: identificar regímenes de flujo en la derivada requiere años de experiencia
- Los papers demuestran viabilidad ([CNN supera FCNN 0.91 vs 0.81](https://link.springer.com/article/10.1007/s42452-024-06089-5), hasta 98% accuracy con DL) pero **no existe ningún portafolio open source** con pipeline completo reproducible
- Well testing es un área del perfil sin explotar — todo el portafolio actual es petrofísica

## Generación del dataset (el corazón del proyecto)

**No existe dataset público etiquetado.** Se genera sintéticamente con soluciones analíticas — etiquetas perfectas por construcción. Plan exacto:

### 1. Motor analítico (espacio de Laplace + inversión Stehfest)
Implementar la solución de presión adimensional p_D(t_D) en espacio de Laplace para cada combinación modelo × frontera, e invertir numéricamente con el algoritmo de **Stehfest** (N=8-12). Todo con NumPy/SciPy — sin simulador comercial.

**Modelos de yacimiento (4):**
1. Radial homogéneo infinito
2. Doble porosidad (Warren-Root, flujo interporoso pseudo-estacionario: ω, λ)
3. Pozo fracturado — fractura de conductividad infinita (flujo lineal, x_f)
4. Pozo fracturado — conductividad finita (flujo bilineal, F_CD)

**Fronteras (4):**
1. Infinito (sin frontera)
2. Falla sellante (pozo imagen, distancia L)
3. Frontera de presión constante (acuífero/casquete, distancia L)
4. Yacimiento cerrado (pseudo-estado estacionario, r_e)

**Efectos de pozo (siempre presentes):** almacenamiento C y skin S vía la convolución estándar en Laplace.

→ Taxonomía: hasta **16 clases** (4×4). MVP: empezar con las **8-10 combinaciones físicamente más comunes** (ej. doble porosidad + presión constante es raro; falla + fractura finita, descartable).

**Fórmulas concretas del motor (todas estándar de libro):**
- Solución general en Laplace de pozo con almacenamiento y skin (van Everdingen & Hurst; Agarwal et al. 1970), con funciones de Bessel `K₀, K₁` de `scipy.special`
- El tipo de yacimiento entra como una sola función `f(s)`: homogéneo `f(s)=1`; doble porosidad Warren-Root `f(s) = (ω(1−ω)s+λ)/((1−ω)s+λ)`
- Fracturas: Gringarten et al. 1974 (conductividad infinita), Cinco-Ley & Samaniego 1978 (finita)
- Fronteras por pozos imagen (superposición): falla sellante = +imagen, presión constante = −imagen; cerrado = solución acotada de van Everdingen-Hurst
- Inversión Laplace→tiempo: algoritmo de Stehfest 1970 (~15 líneas)
- Motor completo estimado: 300-400 líneas NumPy/SciPy

**Validación del motor (certificación antes de entrenar nada):**
- Reproducir punto a punto las type curves publicadas de Bourdet/Gringarten
- Contrastar contra `AnaFlow`/`welltestpy` en el subconjunto que comparten (radial homogéneo, doble porosidad)

**Decisión de lenguaje (2026-06-04):** Python puro. Se evaluaron alternativas: Octave CLI (viable pero repo bilingüe y MATLAB ya olvidado), MRST (innecesariamente pesado para soluciones analíticas), OPM Flow (descartado como generador masivo — lento y con artefactos de malla en tiempo temprano; **queda como fase opcional**: 50-100 casos numéricos con heterogeneidad como test set "difícil", puente sim-to-real intermedio).

### 2. Muestreo de parámetros (rangos realistas, log-uniforme)
| Parámetro | Rango | Escala |
|---|---|---|
| k | 0.1 – 1,000 mD | log |
| h | 5 – 100 m | lineal |
| S | -5 – +20 | lineal |
| C | 10⁻⁴ – 1 bbl/psi | log |
| ω | 0.01 – 0.5 | log |
| λ | 10⁻⁹ – 10⁻⁴ | log |
| L (frontera) | 50 – 1,500 ft | log |
| x_f | 50 – 500 ft | log |
| Duración prueba | 6 – 720 h | log |

Filtro de validez: descartar muestras donde la firma característica no alcanza a desarrollarse dentro de la duración de la prueba (ej. frontera demasiado lejana = inetiquetable → reclasificar como "infinito"). Este filtro ES conocimiento de dominio y hay que documentarlo.

### 3. Realismo (de curva de libro a curva de campo)
Aplicado en orden sobre la curva limpia:
1. **Muestreo temporal realista** — gauge con frecuencia variable (denso al inicio, ralo después), 100-1,000 puntos
2. **Ruido de gauge** — gaussiano σ = 0.01-0.5 psi + deriva térmica lenta (random walk suave)
3. **Truncamiento** — pruebas cortadas antes de tiempo (lo más común en campo)
4. **Outliers** — picos espurios aislados (0-2% de los puntos)
5. **Derivada de Bourdet** calculada como en la práctica real: diferenciación con ventana de suavizado L = 0.1-0.3 ciclos log — la derivada ruidosa es parte del problema real

### 4. Representación de entrada al modelo
**Secuencia 1D de 2 canales** sobre malla logarítmica fija de 256 puntos: [log Δp, log (t·dΔp/dt)], normalizados. Ventajas sobre imagen del plot: sin pérdida por rasterización, más barato, y la malla log-log fija preserva la invariancia que el intérprete humano usa.

### 5. Volumen y splits
- **~80,000 curvas** (≈5,000 por clase + sobremuestreo de clases difíciles), generadas con seed reproducible. Generación on-the-fly durante entrenamiento como augmentation infinita (mismos modelos, nuevo ruido/parámetros cada época)
- Split por **rangos de parámetros disjuntos** (no solo aleatorio) para medir generalización real
- Test set sintético congelado + **test set real** (ver Validación)

## Modelo

### Fase A — Baseline: CNN 1D (ResNet-style)
ResNet-1D de ~6-8 bloques sobre los 2 canales, **multi-task**: cabeza de clasificación (modelo×frontera, softmax) + cabeza de regresión (parámetros del modelo predicho, normalizados log). Loss conjunta: CE + MSE enmascarado (solo los parámetros válidos para la clase). Es la arquitectura que los papers validan y entrena en minutos en la RTX 4080. Optuna para hiperparámetros (terreno conocido).

### Fase B — Transformer 1D (el hueco de aprendizaje)
Encoder Transformer construido a mano (patches de la secuencia, positional encoding, self-attention) — estilo PatchTST. La pregunta empírica del post: ¿el attention global identifica mejor las transiciones de régimen (que son relaciones de largo alcance en la curva) que los kernels locales de la CNN? Aquí el transformer entra con justificación física real: un régimen se define por su relación con lo que pasa antes y después.

### Fase C — App + capa de agente
- **Gradio/Streamlit**: subes CSV (t, p) → preprocesamiento → derivada → diagnóstico + parámetros + curva del modelo ajustado superpuesta
- **Agente LLM** (opcional, conecta con [[agente-llm-registros]]): narra la interpretación en lenguaje de ingeniero ("la derivada muestra estabilización radial seguida de duplicación — consistente con falla sellante a ~300 ft")

## Validación con datos reales (sim-to-real)
- Casos clásicos digitalizados de la literatura: Bourdet 1983, Horne, Lee — ~20-40 casos con interpretación publicada como ground truth
- Reportes DST del dataset **Volve** (Equinor, abierto) — extraer presiones de los reportes
- Métrica honesta: accuracy sintético vs accuracy real, reportar ambas. El gap sim-to-real ES un hallazgo publicable, no un fracaso

## Alcance MVP y limitaciones conocidas
> [!warning] Límites declarados antes de empezar
> - **Monofásico, petróleo ligeramente compresible** — gas requiere pseudopresión (extensión futura, terreno conocido por el proyecto MATLAB)
> - **Drawdown a tasa constante + buildup vía tiempo equivalente de Agarwal** — pruebas multi-tasa con superposición completa quedan fuera del MVP
> - El modelo solo conoce las **clases implementadas** — un yacimiento real fuera de la taxonomía será forzado a la clase más parecida (mitigación: cabeza de incertidumbre/entropía para señalar baja confianza)
> - El modelo de ruido sintético nunca replica todos los artefactos reales de gauge — por eso el test real es obligatorio
> - La derivada depende del suavizado L — entrenar con L variable para robustez
> - Pozos horizontales y composite reservoirs: fuera del MVP

## Qué se puede mostrar
- Demo en vivo: CSV de prueba → diagnóstico en segundos (el momento "wow" para reclutadores técnicos y no técnicos)
- Matriz de confusión sobre 16 clases + scatter de parámetros estimados vs verdaderos
- Comparación CNN vs Transformer con análisis de mapas de atención sobre la derivada (¿el modelo "mira" donde mira el intérprete?)
- Validación sobre casos clásicos de libro — interpretación IA vs interpretación publicada
- 2-3 posts LinkedIn + GitHub con generador de datos reproducible (entregable independiente que nadie tiene)

## Qué se aprende
- **Transformers 1D construidos a mano** (attention, patches, positional encoding) — hueco principal
- Diseño de **datasets sintéticos** y el problema sim-to-real — habilidad central en ML científico
- Multi-task learning (clasificación + regresión conjunta)
- Despliegue de app interactiva (Gradio/Streamlit) — hueco de "demo usable"
- (Opcional) integración de agente LLM con herramientas

## Duración estimada
5-6 semanas:
1. **Sem 1-2** — motor analítico + generador de datos con realismo (la fase crítica; el resto fluye de aquí)
2. **Sem 3** — CNN baseline + Optuna
3. **Sem 4** — Transformer + comparación
4. **Sem 5** — validación real (digitalizar casos, Volve) + app
5. **Sem 6** — agente narrador (opcional) + posts + README

Hito de salida temprana: si el ciclo se corta en la semana 3, ya hay generador + clasificador funcionando = proyecto publicable.

## Conexiones
- [[pinn-curvas-sinteticas]] — antecesor en física+ML; aquí la física genera los datos en vez de regularizar la loss
- [[agente-llm-registros]] — la capa de agente narrador es el mismo patrón aplicado a well testing
- [[laboratorio-arquitecturas-vision-f3]] — absorbe su Fase 1 (CNN vs Transformer) con justificación natural; el laboratorio queda como referencia
- [[cuantificacion-incertidumbre-curvas]] — la cabeza de incertidumbre del MVP conecta con Conformal Prediction
- Referencias: Bourdet et al. 1983 (derivada), Stehfest 1970 (inversión), [DL para PTA 2024](https://link.springer.com/article/10.1007/s42452-024-06089-5), [CNN well-test interpretation](https://www.sciencedirect.com/science/article/abs/pii/S0920410521009190), PatchTST (ICLR 2023)

## Decisiones tomadas (2026-06-04) — sin pendientes para arrancar

> [!tip] Taxonomía: dos cabezas, no 16 clases planas
> Clasificación factorizada: una cabeza predice **modelo de yacimiento** (4 clases) y otra **frontera** (4 clases). Elimina la explosión combinatoria, maneja el desbalance por dimensión, y las combinaciones raras no necesitan miles de muestras propias. Es además como diagnostica un intérprete real: primero yacimiento, luego frontera. El filtro de validez reclasifica como "infinito" las fronteras no desarrolladas.

> [!tip] Representación: solo 1D en el MVP
> El experimento ViT sobre imagen del plot queda fuera del MVP (disciplina de alcance). La secuencia 1D es superior (sin pérdida por rasterización) y el hueco de transformers se cubre con el Transformer 1D de la Fase B. Posible apéndice post-MVP si sobra tiempo.

> [!tip] Casos reales: plan escalonado (semana 5, ~2-3 días)
> 1. Datos **tabulados** en libros: Lee (*Well Testing*, SPE) y Horne (*Modern Well Test Analysis*) traen tablas presión-tiempo con interpretación publicada — transcripción directa
> 2. WebPlotDigitizer para 10-15 casos clásicos de papers (~30-45 min c/u)
> 3. DSTs de Volve (Equinor)
> Meta: 20-30 casos reales — suficiente para el reporte sim-to-real.
