# documentation/ — índice

Documentación de código y diseño del proyecto (audiencia: mantenedores). Los archivos
siguen el patrón `NN_<slug>.md` en orden de lectura. El material de estudio didáctico
vive en `aprendizaje/`; la landing pública en `docs/` (GitHub Pages).

| Archivo | Contenido |
|---|---|
| `01_overview.md` | Qué es Deep PTA: problema, alcance MVP, decisiones cerradas |
| `02_diseno_interpretacion_pruebas_presion.md` | Diseño completo: datos, fórmulas, modelo, limitaciones |
| `03_plan_implementacion.md` | Plan detallado de implementación (fórmulas, mermaid, rangos de muestreo) |
| `04_referencias.md` | Bibliografía con claves `[clave]` citadas en docstrings y docs |
| `05_teoria_pta.md` | Repaso teórico mínimo por fase (Stehfest, Bourdet, type curves) |
| `06_reporte_no_unicidad_confusiones.md` | Diagnóstico: ratio de concentración de confusiones (físico vs pipeline) |
| `07_reporte_mejora_accuracy.md` | Ciclos v1/v2/v3: metodología, tablas de ablation, lectura honesta |
| `08_reporte_sim_to_real.md` | Metodología de validación con datos reales (trabajo futuro) |
| `09_posts_linkedin.md` | Borradores de posts de divulgación |
| `10_reporte_banco_arquitecturas.md` | Ciclo v3: banco de 6 arquitecturas, mapeo yacimiento↔código, hallazgos |

Convenciones en `.claude/rules/docs-style.md` y `.claude/rules/file-naming.md`.
