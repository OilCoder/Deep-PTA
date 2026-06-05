# Teoría mínima de PTA por fase

Nota de estudio: la física justa para leer y construir cada fase del proyecto. Las claves
`[clave]` remiten a `documentation/referencias.md`. El diagnóstico de PTA se hace sobre la
**derivada de Bourdet** en log-log, porque cada régimen de flujo deja una **pendiente
característica** que el ojo del intérprete (y la red) reconoce.

## 1. La derivada de Bourdet y los regímenes de flujo

La derivada de Bourdet adimensional es `p_D' = dp_D / d(ln t_D) = t_D · dp_D/dt_D`
`[Bourdet1989]`. En un gráfico log-log de `p_D'` vs `t_D`, cada régimen aparece como un
tramo de pendiente fija:

| Régimen | Pendiente de la derivada | Firma | Referencia |
|---|---|---|---|
| Almacenamiento de pozo (wellbore storage) | unitaria (1) inicial | Δp y derivada coinciden, recta de pendiente 1 | `[Agarwal1970]` |
| Flujo radial (actuación infinita) | 0 (meseta) | nivel constante `= 0.5` en adimensional | `[Bourdet1983]` |
| Flujo lineal (fractura cond. infinita) | 1/2 | recta de pendiente ½ temprana | `[Gringarten1974]` |
| Flujo bilineal (fractura cond. finita) | 1/4 | recta de pendiente ¼ temprana | `[CincoLey1978]` |
| Doble porosidad (Warren-Root) | meseta → valle → meseta | dos mesetas con transición; profundidad ∝ ω, posición ∝ λ | `[WarrenRoot1963]` |
| Falla sellante | salto de meseta (×2) | la meseta radial **duplica** su nivel | `[Horne1995]` |
| Presión constante (acuífero/casquete) | caída | la derivada **cae** hacia cero | `[Horne1995]` |
| Frontera cerrada (pseudo-estacionario) | unitaria (1) tardía | derivada sube con pendiente 1 al final | `[vanEverdingenHurst1949]` |

La red aprende a clasificar **modelo de yacimiento** (homogéneo, doble porosidad, fractura
cond. infinita, fractura cond. finita) y **frontera** (infinito, sellante, presión cte,
cerrado) leyendo qué pendientes aparecen y en qué orden.

## 2. Espacio de Laplace y por qué se usa

Las soluciones de difusividad para pozo con almacenamiento `C` y skin `S` son cerradas en
**espacio de Laplace** (variable `u` conjugada a `t_D`), no en tiempo. El motor:

1. evalúa `p̄_wD(u)` en Laplace (combinando yacimiento `f(u)`, almacenamiento, skin y
   frontera);
2. la invierte a tiempo con el algoritmo de **Stehfest** `[Stehfest1970]`:

   `f(t) ≈ (ln 2 / t) · Σ_{i=1..N} V_i · F(i · ln 2 / t)`, con `N` par (8–12).

Esto evita simuladores numéricos: las etiquetas son **perfectas por construcción**.

## 3. Solución base pozo (almacenamiento + skin)

Forma consolidada `[Bourdet2002]` `[MavorCincoLey1979]`, con `x = √(u · f(u))` y las
Bessel modificadas `K₀, K₁` (`scipy.special`):

```
            K₀(x) + S·x·K₁(x)
p̄_wD(u) = ─────────────────────────────────────────────
          u·x·K₁(x) + C_D·u² · [ K₀(x) + S·x·K₁(x) ]
```

El **acoplamiento de yacimiento** entra solo por `f(u)`:
- Homogéneo infinito: `f(u) = 1`.
- Doble porosidad Warren-Root (interporoso pseudo-estacionario):
  `f(u) = ( ω(1−ω)·u + λ ) / ( (1−ω)·u + λ )`, con `ω ∈ (0,1)` (storativity ratio) y
  `λ` (interporosity flow) `[WarrenRoot1963]`.

## 4. Fracturas

- **Conductividad infinita** `[Gringarten1974]`: flujo lineal temprano (pendiente ½),
  parámetro semilongitud `x_f`; tiempo adimensional escalado por `x_f` (`t_Dxf`).
- **Conductividad finita** `[CincoLey1978]`: flujo bilineal (pendiente ¼), parámetro
  conductividad adimensional `F_CD`; entra como solución propia en Laplace.

## 5. Fronteras por superposición de pozos imagen `[Horne1995]`

- Infinito: sin imagen.
- Falla sellante: + imagen a distancia `2L` → la meseta radial **se duplica**.
- Presión constante: − imagen → la derivada **cae**.
- Cerrado: solución acotada de van Everdingen-Hurst → pendiente **unitaria** tardía.

## 6. Buildup vía tiempo equivalente de Agarwal `[Agarwal1980]`

Un buildup se analiza con type curves de drawdown usando el tiempo equivalente
`Δt_e = Δt · t_p / (t_p + Δt)`, donde `t_p` es el tiempo de producción. Así el MVP cubre
drawdown a tasa constante + buildup sin superposición multi-tasa completa.

## 7. Filtro de validez (conocimiento de dominio)

Si una firma característica no alcanza a desarrollarse dentro de la duración de la prueba
(p. ej. una frontera demasiado lejana), es **inetiquetable**: el muestreo la reclasifica
como "infinito". Esto NO es un bug — es exactamente la ambigüedad que enfrenta el
intérprete real, y se documenta como parte del problema.
