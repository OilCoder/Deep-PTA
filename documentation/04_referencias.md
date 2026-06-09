---
title: "Deep PTA — Bibliografía"
date: 2026-06-05
tags: [referencias, bibliografia, pta, well-testing]
type: reference
---

# Bibliografía — Deep PTA

Fuente de verdad de las referencias del proyecto. Cada entrada tiene una **clave de
cita** (`[clave]`) que se usa en `plan-implementacion.md`, en los docstrings de los
módulos del motor (`src/deep_pta/engine/*`) y en el reporte sim-to-real. Tres bloques:
(A) fundacionales del motor analítico, (B) libros para validación y casos tabulados,
(C) ML aplicado a PTA (estado del arte a superar).

## A. Fundacionales — motor analítico

| Clave | Referencia | Aporta al motor |
|---|---|---|
| `[vanEverdingenHurst1949]` | van Everdingen, A.F. & Hurst, W. (1949). *The Application of the Laplace Transformation to Flow Problems in Reservoirs.* Trans. AIME 186, 305–324. | Base del enfoque en espacio de Laplace; solución de yacimiento cerrado (acotada). |
| `[Agarwal1970]` | Agarwal, R.G., Al-Hussainy, R. & Ramey, H.J. (1970). *An Investigation of Wellbore Storage and Skin Effect in Unsteady Liquid Flow: I. Analytical Treatment.* SPEJ 10(3), 279–290. | Almacenamiento de pozo `C` + skin `S` en la condición de frontera interna. |
| `[Stehfest1970]` | Stehfest, H. (1970). *Algorithm 368: Numerical Inversion of Laplace Transforms.* Comm. ACM 13(1), 47–49. | Inversión numérica Laplace→tiempo (coeficientes `V_i`, `N` par). |
| `[WarrenRoot1963]` | Warren, J.E. & Root, P.J. (1963). *The Behavior of Naturally Fractured Reservoirs.* SPEJ 3(3), 245–255. | Doble porosidad: función `f(s)` con `ω` (storativity ratio) y `λ` (interporosity flow). |
| `[MavorCincoLey1979]` | Mavor, M.J. & Cinco-Ley, H. (1979). *Transient Pressure Behavior of Naturally Fractured Reservoirs.* SPE 7977. | **Forma a implementar**: doble porosidad con `C` y `S` acoplados en Laplace vía `f(s)`. |
| `[Gringarten1974]` | Gringarten, A.C., Ramey, H.J. & Raghavan, R. (1974). *Unsteady-State Pressure Distributions Created by a Well With a Single Infinite-Conductivity Vertical Fracture.* SPEJ 14(4), 347–360. | Fractura de conductividad infinita: flujo lineal (pendiente ½), `x_f`. |
| `[CincoLey1978]` | Cinco-Ley, H., Samaniego, F. & Domínguez, N. (1978). *Transient Pressure Behavior for a Well With a Finite-Conductivity Vertical Fracture.* SPEJ 18(4), 253–264. | Fractura de conductividad finita: flujo bilineal (pendiente ¼), `F_CD`. |
| `[Bourdet1983]` | Bourdet, D., Whittle, T.M., Douglas, A.A. & Pirard, Y.M. (1983). *A New Set of Type Curves Simplifies Well Test Analysis.* World Oil 196(6), 95–106. | Derivada de Bourdet + type curves de certificación. |
| `[Bourdet1989]` | Bourdet, D., Ayoub, J.A. & Pirard, Y.M. (1989). *Use of Pressure Derivative in Well-Test Interpretation.* SPEFE 4(2), 293–302. | Algoritmo práctico de la derivada con ventana de suavizado `L` (ciclos log). |
| `[Agarwal1980]` | Agarwal, R.G. (1980). *A New Method To Account for Producing Time Effects When Drawdown Type Curves Are Used To Analyze Pressure Buildup and Other Test Data.* SPE 9289. | Tiempo equivalente de Agarwal para buildup: `Δt_e = Δt·t_p/(t_p+Δt)`. |

## B. Libros — validación y casos tabulados

| Clave | Referencia | Uso |
|---|---|---|
| `[Lee1982]` | Lee, J. (1982). *Well Testing.* SPE Textbook Series Vol. 1. | Tablas presión-tiempo con interpretación publicada (casos reales tabulados). |
| `[Horne1995]` | Horne, R.N. (1995). *Modern Well Test Analysis: A Computer-Aided Approach* (2ª ed.). Petroway. | Casos tabulados; diagnóstico por derivada; pozos imagen. |
| `[Bourdet2002]` | Bourdet, D. (2002). *Well Test Analysis: The Use of Advanced Interpretation Models.* Elsevier. | Forma consolidada de la solución en Laplace (`f(s)` + `C` + `S`); type curves. |

## C. ML aplicado a PTA — estado del arte a superar

| Clave | Referencia | Relevancia |
|---|---|---|
| `[DiscoverAppSci2024]` | *Enhancing pressure transient analysis through deep learning neural networks.* Discover Applied Sciences (2024). | Baseline a batir: CNN 0.91 vs FCNN 0.81 accuracy. |
| `[JPSE2021]` | *Application of deep learning on well-test interpretation.* J. Petroleum Science & Engineering (2021). | CNN para interpretación de well testing; antecedente directo. |
| `[DRL2021]` | *Deep Reinforcement Learning para interpretación de well testing* (2021). | Enfoque alternativo; contexto del estado del arte. |
| `[Nie2023]` | Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).* ICLR 2023. | Referencia del Transformer 1D (patches + self-attention) de la Fase 3. |

## D. Opcional — extensión futura (TDS / flujo elíptico)

| Clave | Referencia | Relevancia |
|---|---|---|
| `[Tiab1995]` | Tiab, D. (1995). *Analysis of Pressure and Pressure Derivative without Type-Curve Matching (Tiab's Direct Synthesis).* | TDS: diagnóstico directo por pendientes/intersecciones de la derivada. |
| `[Escobar]` | Escobar, F.H. et al. *Trabajos sobre TDS y flujo elíptico (pendiente 0.36).* | Régimen elíptico (candidato a extensión post-MVP con segmentación de regímenes). |