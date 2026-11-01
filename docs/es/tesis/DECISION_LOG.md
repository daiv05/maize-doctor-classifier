# Registro de decisiones técnicas

## DEC-001 — `clean/` como fuente de verdad

- **Fecha:** 2026-06-16 (pipeline inicial).
- **Problema:** fuentes y estructuras incompatibles.
- **Alternativas:** entrenar desde cada fuente; normalizar una copia; mutar originales.
- **Decisión:** preservar `raw/` y entrenar desde `clean/<clase>/{lab,real}`.
- **Impacto:** preparación uniforme y regenerable.
- **Evidencia:** pipeline de preparación en `src/data/`.

## DEC-002 — Macro-F1 como métrica primaria

- **Fecha:** verificable en documentación de baselines, julio de 2026.
- **Problema:** desbalance hasta 32.9× históricamente.
- **Decisión:** seleccionar por Macro-F1; accuracy queda secundaria.
- **Impacto:** una clase pequeña pesa igual que una mayoritaria.
- **Evidencia:** `docs/es/pipeline-baselines/evaluacion.md`.

## DEC-003 — Tres arquitecturas principales

- **Fecha:** 2026-07-08.
- **Problema:** equilibrar precisión, tamaño y cuantización.
- **Decisión:** conservar B0, ShuffleNet-V2-x1.0 y Lite0 tras exploración de ocho modelos.
- **Impacto:** comparación reproducible y ensamble posterior.
- **Evidencia:** documentación de baselines.

## DEC-004 — Segmentación opt-in, no universal

- **Fecha:** 2026-09-07, run segmentado.
- **Problema:** fondo y escenas multihoja frente a pérdida de contexto/errores de máscara.
- **Alternativas:** máscara negra, bbox crop, crop+mask, letterbox, imagen original.
- **Decisión:** perfiles explícitos con selección/threshold/fallback; no promover el run segmentado.
- **Impacto:** se evita que una máscara dudosa gobierne todos los datos.
- **Evidencia:** `src/segmentation/leaf_processor.py`, run `20260907_163546`.

## DEC-005 — No transferir ciegamente HP de B0 a Lite0

- **Fecha:** 2026-09-12/13.
- **Problema:** parámetros buenos para B0 redujeron el resultado de Lite0.
- **Decisión:** mantener configuración base de Lite0 en esa etapa.
- **Impacto:** preservó el run histórico desplegado.
- **Evidencia:** `optuna_*best_params.json`, `run_afinada_*summary.json`.

## DEC-006 — Separar desarrollo de generalización por fuente

- **Fecha:** 2026-09-21 (fecha de los runs; integración en código 2026-09-18).
- **Problema:** once fuentes indivisibles dieron 49.52/23.36/27.12 y clases ausentes en validación.
- **Alternativas:** mantener source grouping como split único; ignorar procedencia; mantener dos protocolos.
- **Decisión:** `seed_42` estratificado para desarrollo y `seed_42_source_grouped` para cross-source.
- **Razón:** selección balanceada sin perder una evaluación honesta de cambio de dominio.
- **Impacto:** las métricas deben reportarse por protocolo.
- **Evidencia:** locks/auditorías y runs `20260921_180112`, `20260921_204608`.

## DEC-007 — Identidad por ruta; integridad por contenido

- **Fecha:** 2026-09-18.
- **Problema:** orden/sampler/workers no identifican establemente una muestra.
- **Alternativas:** índice de fila; SHA del contenido como identidad; hash de ruta normalizada.
- **Decisión:** `sample_id=SHA256(ruta lógica UTF-8)` y `sha256` separado.
- **Impacto:** reemplazar bytes no cambia identidad, pero sí detecta integridad; duplicados conservan identidad propia.
- **Evidencia:** `src/data/identity.py`.

## DEC-008 — Fallar sobre la misma muestra

- **Fecha:** 2026-09-18.
- **Problema:** `idx` ilegible provocaba intento de filas siguientes.
- **Alternativas:** devolver `None`; filtrar en collate; sustituir; fail-fast.
- **Decisión:** error trazable de la misma muestra.
- **Impacto:** no se desacoplan imagen, etiqueta, ID y predicción.
- **Evidencia:** `tests/data/test_dataset_fail_fast.py`.

## DEC-009 — Integración selectiva de dos líneas de desarrollo

- **Fecha:** 2026-09-18 a 22.
- **Problema:** las dos líneas de desarrollo tenían avances y regresiones distintos.
- **Alternativas:** cherry-pick total; descartar rama; portar capacidades.
- **Decisión:** adaptar identidad/manifests/run contracts y conservar caché, tope, PHash y splitter de la línea principal.
- **Impacto:** se añadieron garantías sin reemplazar `CornDataset`.
- **Evidencia:** `tests/data/` y `tests/training/`.

## DEC-010 — No afirmar fuga perceptual cero

- **Fecha:** corte documental 2026-09-22.
- **Problema:** PHash existe, pero el lock registra `deduplicate_perceptual=false`.
- **Decisión:** afirmar solo cero duplicados exactos y dejar auditoría perceptual pendiente.
- **Impacto:** limita correctamente el alcance del baseline.
- **Evidencia:** `seed_42_manifest_lock.json`.

## DEC-011 — HPO Lite0 con test cerrado

- **Fecha:** 2026-09-22, antes de los resultados.
- **Problema:** el baseline es sólido, pero falta una optimización sistemática sobre los splits vigentes sin contaminar el test.
- **Decisión:** Optuna TPE, objetivo Macro-F1 de validación en `seed_42`, seis dimensiones con AdamW y cosine fijos, MedianPruner (startup 5, warmup 8) y máximo de 60 épocas con patience 8. Presupuesto inicial de 60 intentos, reducido después (DEC-013). El test se evalúa una sola vez sobre el checkpoint ganador, tras el selection lock; ningún trial lo utiliza.
- **Evidencia:** `scripts/pipeline/tune.py`, `src/training/tuning_study.py`, `tests/training/test_hpo_protocol.py` y [protocolo](../experimentos/hpo.md).

## DEC-013 — Enmienda de presupuesto HPO: 60 → 25

- **Fecha:** 2026-09-23, enmienda durante la ejecución.
- **Alcance:** reducción del número máximo de intentos; se mantienen el objetivo y el espacio de búsqueda.
- **Decisión:** 25 intentos totales incluyendo COMPLETE, PRUNED y FAIL, conservando IDs y RNG de TPE.
- **Estado al cambio:** trials 0 y 1 completos; trial 2 interrumpido al detener la app para migrar. Se conserva como FAIL; no se repite su ID ni se presenta como entrenamiento completo.
- **Alcance:** splits, dataset, transforms, modelo, lógica de entrenamiento y máximo de 60 épocas intactos. El test sigue reservado para el ganador bloqueado al agotar los 25 intentos.
- **Transparencia:** se habían observado métricas parciales de validation. No presentar el nuevo presupuesto como prefijado desde el inicio. DEC-011 conserva el plan original.
- **Respaldo:** SQLite, código y preflight originales en `budget_revisions/60-to-25/`; comparación de tablas de trials y RNG antes/después sin cambios.
- **Evidencia:** [enmienda y hashes](../reproducibilidad/evidencia/hpo_lite0_seed42/BUDGET_AMENDMENT.json).

## Cierre de DEC-011 y DEC-013 (2026-09-24)

Study `efficientnet_lite0_seed42_hpo_v1`: 25 intentos, ganador trial 0, validation Macro-F1 0.957292; baseline 0.956086; delta +0.12 pp. [Configuración seleccionada y resultados](../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md).

## DEC-014 — Interpretar el HPO sin promoverlo por validación solamente

- **Fecha:** revisión documental del 2026-09-24.
- **Evidencia:** +0.120596 pp en Macro-F1 de validación, pero −0.487707 pp en Macro-F1 de test frente al baseline vigente.
- **Criterio de reporte:** conservar trial 0 como ganador contractual por validación; no reordenar candidatos ni reabrir la búsqueda después de observar test.
- **Límite:** no afirmar superioridad de generalización, significancia estadística ni aptitud de producción. No reemplazar automáticamente el baseline/checkpoint desplegado.
- **Fases futuras:** entrenamiento formal, multi-seed, CV/LOSO y calibración requieren un protocolo posterior; no se ejecutaron en este cierre.
- **Trazabilidad:** [comparación y fuentes](./HPO_BASELINE_COMPARISON.md).

## DEC-015 — Comparación pareada multi-seed, exclusivamente validación

- **Fecha:** 2026-09-24.
- **Diseño:** baseline `20260921_204608` frente a HPO trial 0, seeds 42, 123, 2026, 3407 y 7777, diez slots, split `seed_42` congelado.
- **Implementación:** reutiliza `train.py` y `fit`; `--skip-test` es explícito y bloquea la exportación. Sin cambios en `CornDataset`.
- **No reutilización histórica:** la equivalencia completa no es verificable para ambos antecedentes.
- **Recuperación:** se saltan las runs completas e íntegras; una interrumpida requiere diagnóstico.
- **Estado:** 2 de 10 entrenamientos completados el 2026-10-19 (seed 42 de cada configuración); sin conclusión experimental ni promoción. [Protocolo y estado](../experimentos/multiseed.md).
