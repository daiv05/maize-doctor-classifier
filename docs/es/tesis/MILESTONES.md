# Hitos del proyecto

| ID | Fecha verificable | Hito | Evidencia | Estado |
|---|---|---|---|---|
| M1 | 2026-05-19 | objetivo y documentación inicial | `0c4f331`, `fc841c2` | HISTÓRICO |
| M2 | 2026-06-03 a 12 | recopilación, limpieza y deduplicación inicial | commits de datos; informe fase 1 | HISTÓRICO |
| M3 | 2026-06-16 a 22 | pipeline de preparación y `CornDataset` | `e9fbc5e`, `d4c1806`, `a47cb94` | EVOLUCIONADO |
| M4 | 2026-07-08 | baselines principales | `a2562df` | HISTÓRICO |
| M5 | 2026-07-21 a 2026-09-07 | segmentación y corpus presegmentado | `9501c52`, `be8e028`, run `20260907_163546` | EXPERIMENTAL |
| M6 | 2026-08-11 | ampliación del dataset | `0476b4c` | REEMPLAZADO por manifiesto actual |
| M7 | 2026-08-12 | Lite0 histórico luego desplegado | run `20260812_221429` | HISTÓRICO desplegado |
| M8 | 2026-08-14 a 29 | exportación, móvil y OOD | commits export/OOD | HISTÓRICO funcional |
| M9 | 2026-09-08 a 13 | Optuna histórico, ensamble y CV | artefactos resultados | HISTÓRICO |
| M10 | 2026-09-09 a 12 | auditoría de procedencia y LOSO | evidencia provenance | HISTÓRICO útil |
| M11 | 2026-09-18 | `sample_id` y predicciones trazables | `80bff4` | VIGENTE |
| M12 | 2026-09-18 | manifest, integridad, grupos y fail-fast | `711bd10`, `600ebb9` | VIGENTE |
| M13 | 2026-09-21 | separación desarrollo/cross-source | auditorías + run `180112` | VIGENTE |
| M14 | 2026-09-21 | baseline estratificado actual | run `20260921_204608` | VIGENTE |
| M15 | 2026-09-22 | contratos de runs y auditoría documental | `96be6f1` + este corte | VIGENTE |
| M16 | 2026-09-24 | Optuna 25 trials | HPO_REPORT.md | COMPLETADO |
| M17 | preparación: 2026-09-24 | comparación multi-seed previa al entrenamiento formal | [protocolo y verificación local](../experimentos/multiseed.md) | 0/10 runs; reanudación prevista 2026-10-01 |
| M18 | pendiente | CV/source-grouped/LOSO final | protocolo | PENDIENTE |

<!-- hpo-lite0-seed42-completed -->

## M16 — cierre verificado (2026-09-24)

**COMPLETADO: HPO Optuna 25 trials.** Study `efficientnet_lite0_seed42_hpo_v1`: 25 intentos, ganador trial 0, validation Macro-F1 0.957292225; baseline 0.956086266; delta +0.120596 pp.

Son 8 completos, 15 podados y 2 interrumpidos. Test Macro-F1 = 0.943125073,
inferior al baseline 0.948002144. Completar M16 significa cerrar el protocolo y
su evidencia, **no demostrar superioridad en test ni completar M17/M18**.
La fecha de cierre usa UTC; en El Salvador fue el 2026-09-23 a las 20:11.

[Evidencia](../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md).

## M17 — Preparación validada; ejecución pendiente

Baseline y HPO trial 0 se compararán mediante cinco semillas emparejadas, sin
evaluar test ni cambiar los splits. La implementación está validada localmente;
el hito experimental se cerrará únicamente con diez runs verificadas, sus métricas
de validación y el análisis pareado. La reanudación manual está prevista para el
1 de octubre de 2026. El entrenamiento formal posterior queda fuera de esta fase.
