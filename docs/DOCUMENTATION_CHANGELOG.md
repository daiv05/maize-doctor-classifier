# Changelog de documentación

## 2026-09-22 — Auditoría integral

| Documento/área | Problema encontrado | Cambio realizado | Razón |
|---|---|---|---|
| README y portada de resultados | baseline histórico, vigente y CV agrupada mezclados | separar estado, protocolo y materialización | evitar comparaciones inválidas |
| `LOCAL.md` | loop principal descrito como pendiente | actualizar comandos de entrenamiento y evaluación | reflejar Makefile actual |
| `CLAUDE.md` | faltaba mapa canónico de identidad/protocolos | enlazar metodología y fijar invariantes | proteger contra regresiones |
| Metodología | especificación distribuida | crear pipeline de datos y protocolos canónicos | una sola fuente de verdad |
| Experimentos | HPO futuro confundible con estudios previos | crear registro, estado, HPO, entrenamiento formal y multi-seed | preparar siguiente fase sin inventar resultados |
| Resultados | run source-grouped sin ficha propia | añadir ficha separada y evidencia | conservar resultado negativo útil |
| Evidencia | métricas actuales no estaban versionadas por dimensión | añadir CSV/JSON de clase, confusión, ambiente, fuente, calibración y N/P/K | trazabilidad cuantitativa |
| Historia | evolución reconstruida de forma dispersa | crear evolución, resumen, decisiones e hitos | base verificable para tesis |
| Reproducibilidad | sin auditoría de figuras/referencias | crear registros y marcar pendientes | no inventar procedencia ni bibliografía |
| `docs/etapa_2` | solo auxiliares LaTeX sin fuente | clasificar como no verificable; no borrar | preservar historia sin tratarla como fuente |

No se modificaron modelos, dataset, splits ni código funcional; no se ejecutaron entrenamientos ni Optuna.
