# Backlog de investigación

| Prioridad | Línea | Criterio de cierre | Estado |
|---:|---|---|---|
| 1 | Optuna EfficientNet-Lite0, 25 intentos (enmienda de 60) | SQLite y trials auditados; test único después de selection lock | COMPLETADO; [resultado y límites](./HPO_BASELINE_COMPARISON.md) |
| 2 | Multi-seed baseline frente a HPO trial 0 | 5 pares predeclarados, validation-only, media/SD y runs individuales | IMPLEMENTADO; 0/10; reanudación prevista 2026-10-01; [protocolo](../experimentos/multiseed.md) |
| 3 | Entrenamiento formal posterior | decisión sustentada + run contractual + protocolo nuevo | PENDIENTE; no autorizado en esta fase |
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
