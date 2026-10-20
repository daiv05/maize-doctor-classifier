# Backlog de investigación

| Prioridad | Línea | Criterio de cierre | Estado |
|---:|---|---|---|
| 1 | Optuna EfficientNet-Lite0, 25 intentos (enmienda de 60) | SQLite y trials auditados; test único después de selection lock | COMPLETADO; [resultado y límites](./HPO_BASELINE_COMPARISON.md) |
| 2 | Multi-seed baseline frente a HPO trial 0 | 5 pares predeclarados, validation-only, media/SD y runs individuales | EN CURSO; 2/10 runs completas, 1/5 pares; [primer resultado](../reproducibilidad/evidencia/multiseed_baseline_hpo_seed42/REVIEW.md) |
| 3 | Entrenamiento formal posterior | decisión sustentada + run contractual + protocolo nuevo | Regla de selección [fijada](../experimentos/seleccion-modelo-final.md); decisión y entrenamiento PENDIENTES hasta completar los 4 pares restantes |
| 4 | Auditoría perceptual | candidatos entre splits revisados; tasa/decisión documentada | PENDIENTE |
| 5 | CV estratificada vigente | K y protocolo fijados; resultados por pliegue | PENDIENTE |
| 6 | Source-grouped final | holdout y/o CV agrupada del modelo elegido | PENDIENTE |
| 7 | LOSO final | una fuente retenida por ejecución; clases evaluables explícitas | PENDIENTE |
| 8 | Análisis N/P/K | revisión de errores + métrica 9-clases y agrupada separadas | PENDIENTE |
| 9 | Calibración | método elegido solo con validación; ECE/Brier antes/después | PENDIENTE |
| 10 | Segmentación/doble perspectiva | comparación pareada full/segmented y quality gate predefinido | PENDIENTE |
| 11 | Exportación y dispositivo | paridad, test completo exportado, latencia y memoria | PENDIENTE |
| 12 | Verificación bibliográfica | DOI/metadatos de referencias marcadas | PENDIENTE |

No mover una fila a completada sin enlazar el artefacto correspondiente en `EVIDENCE_REGISTRY.md`.
