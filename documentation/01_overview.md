---
title: "Deep PTA"
date: 2026-06-04
tags: [project, python, pytorch, deep-learning, well-testing, pta, transformers, gradio]
type: project
status: active
repo: "pendiente — se creará en WSL (~/deep-pta)"
---

# Deep PTA

## Qué problema resuelve
Red neuronal que interpreta pruebas de presión: lee la curva Δp + derivada de Bourdet (log-log) y diagnostica el modelo de yacimiento, la frontera y los parámetros clave (k·h, S, C, ω, λ, L, x_f) — el trabajo del intérprete senior, automatizado. Incluye app interactiva (CSV → diagnóstico) y opcionalmente un agente LLM que narra la interpretación.

Detalle completo del diseño en [[deep-pta-interpretacion-pruebas-presion]] (nota de idea, ya aterrizada — datos, fórmulas, modelo, limitaciones, decisiones).

## Stack
- Python (NumPy/SciPy para el motor analítico — Bessel `K₀/K₁`, inversión Stehfest)
- PyTorch (ResNet-1D multi-task → Transformer 1D a mano)
- Optuna (hiperparámetros), W&B/Tensorboard (tracking)
- Gradio o Streamlit (app)
- Validación del motor: type curves Bourdet/Gringarten + `AnaFlow`/`welltestpy`

## Estructura
Pendiente — el repo vivirá en WSL con la convención estándar: carpeta `todo/` en la raíz (PLAN.md + bitácoras) como fuente de verdad del estado.

## Entorno de desarrollo
Devcontainer + Docker en WSL (convención de todos los proyectos). RTX 4080 para entrenamiento.

## Cómo correr
Pendiente — repo aún no creado.

## Estado actual
**Fase 0 — arranque.** Idea completamente aterrizada, sin pendientes de diseño. Próximo paso: crear repo en WSL y comenzar Fase 1 (motor analítico).

## Plan de fases (5-6 semanas)

- [ ] **Fase 1 (sem 1-2) — Motor analítico + generador.** Soluciones en Laplace (4 yacimientos × 4 fronteras, C y S siempre), Stehfest, capa de realismo (ruido gauge, deriva, truncamiento, outliers, derivada con L variable). Certificación contra type curves publicadas y AnaFlow. ~80k curvas, generación on-the-fly. *Hito: motor certificado = entregable publicable por sí solo.*
- [ ] **Fase 2 (sem 3) — Baseline CNN.** ResNet-1D, 2 canales × 256 puntos log. Dos cabezas de clasificación (yacimiento 4 + frontera 4) + cabeza de regresión enmascarada. Optuna. *Hito: matriz de confusión + scatter parámetros.*
- [ ] **Fase 3 (sem 4) — Transformer 1D.** Encoder a mano (patches, positional encoding, attention) estilo PatchTST. Comparación honesta vs CNN + mapas de atención sobre la derivada. *Hito: post LinkedIn #1.*
- [ ] **Fase 4 (sem 5) — Validación real.** Casos tabulados de Lee/Horne, WebPlotDigitizer para clásicos, DSTs de Volve (meta: 20-30 casos). Reporte sim-to-real. *Hito: post LinkedIn #2.*
- [ ] **Fase 5 (sem 6) — App + cierre.** Gradio (CSV → diagnóstico + curva ajustada superpuesta), agente narrador opcional, README, posts finales.

Salida temprana: si el ciclo se corta en Fase 2, generador + clasificador ya son proyecto publicable.

## Notas
- Decisiones de diseño cerradas el 2026-06-04 (ver nota de idea): Python puro, taxonomía factorizada en dos cabezas, representación solo 1D en MVP, OPM Flow solo como fase opcional de robustez
- Limitaciones declaradas: monofásico petróleo, buildup vía Agarwal, sin pozos horizontales ni anisotropía areal (flujo elíptico 0.36 fuera del MVP — candidato para extensión con segmentación de regímenes)
- Pendiente al arrancar: nota de estudio en Topics/ con la teoría mínima por fase (repaso de Bourdet, Stehfest, regímenes y pendientes)

## Conexiones
- [[deep-pta-interpretacion-pruebas-presion]] — nota de idea con el diseño completo
- [[Projects/pinn-curvas-sinteticas/pinn-curvas-sinteticas]] — proyecto anterior; aquí la física genera los datos en vez de regularizar la loss
- [[flow-boundary-analysis]] — el software MATLAB de intrusión de agua (2016), antecesor directo del arco narrativo
- [[agente-llm-registros]] — patrón del agente narrador
- [[cuantificacion-incertidumbre-curvas]] — la cabeza de incertidumbre conecta con Conformal Prediction
