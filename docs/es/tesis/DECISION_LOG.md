# Registro de decisiones técnicas

## DEC-001 — `clean/` como fuente de verdad

- **Fecha:** 2026-06-16 (pipeline inicial).
- **Problema:** fuentes y estructuras incompatibles.
- **Alternativas:** entrenar desde cada fuente; normalizar una copia; mutar originales.
- **Decisión:** preservar `raw/` y entrenar desde `clean/<clase>/{lab,real}`.
- **Impacto:** preparación uniforme y regenerable.
- **Evidencia:** `CLAUDE.md`, `e9fbc5e`.

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
- **Evidencia:** `a2562df`, documentación de baselines.

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
- **Evidencia:** `80bff4`, `src/data/identity.py`.

## DEC-008 — Fallar sobre la misma muestra

- **Fecha:** 2026-09-18.
- **Problema:** `idx` ilegible provocaba intento de filas siguientes.
- **Alternativas:** devolver `None`; filtrar en collate; sustituir; fail-fast.
- **Decisión:** error trazable de la misma muestra.
- **Impacto:** no se desacoplan imagen, etiqueta, ID y predicción.
- **Evidencia:** `master:1110207`, corrección `600ebb9`, tests fail-fast.

## DEC-009 — Integración selectiva de `dev-abner`

- **Fecha:** 2026-09-18 a 22.
- **Problema:** `master` y `dev-abner` tenían avances y regresiones distintos.
- **Alternativas:** cherry-pick total; descartar rama; portar capacidades.
- **Decisión:** adaptar identidad/manifests/run contracts y conservar caché, tope, PHash y splitter de `master`.
- **Impacto:** se añadieron garantías sin reemplazar `CornDataset`.
- **Evidencia:** merge-base `8a19ff9`; commits `80bff4`, `711bd10`, `600ebb9`, `96be6f1`.

## DEC-010 — No afirmar fuga perceptual cero

- **Fecha:** corte documental 2026-09-22.
- **Problema:** PHash existe, pero el lock registra `deduplicate_perceptual=false`.
- **Decisión:** afirmar solo cero duplicados exactos y dejar auditoría perceptual pendiente.
- **Impacto:** limita correctamente el alcance del baseline.
- **Evidencia:** `seed_42_manifest_lock.json`.

## DEC-011 — HPO nuevo con test cerrado

- **Fecha:** plan aprobado en este corte documental.
- **Problema:** seleccionar HP sin contaminar el test.
- **Decisión:** 60 trials de Optuna, objetivo Macro-F1 de validación en `seed_42`, test no usado.
- **Impacto:** el resultado de test se reserva para entrenamiento formal.
- **Evidencia:** `docs/es/experimentos/hpo.md`.

## DEC-012 — Protocolo ejecutable HPO Lite0, 60 intentos

- **Fecha:** 2026-09-22, implementación previa a los resultados.
- **Problema:** el baseline actual es sólido, pero falta optimización sistemática sobre los splits vigentes.
- **Alternativas:** ajuste manual, grid search y Optuna.
- **Decisión:** reutilizar Optuna TPE, 60 intentos totales, validation Macro-F1 como único objetivo, semilla 42; seis dimensiones y AdamW/cosine fijos.
- **Razón:** búsqueda reproducible sobre variables continuas y discretas, con SQLite y estado RNG del sampler para reanudar.
- **Pruning:** MedianPruner, startup=5, warmup=8, interval=1; patience=8, máximo 60 épocas.
- **Test:** la nueva solicitud precisa DEC-011: se evalúa una vez el checkpoint exacto del HPO winner después del selection lock. El entrenamiento formal posterior es otro experimento. Ningún trial utiliza test.
- **Estado:** implementado y en validación técnica; no declara 60 trials completos ni mejora.
- **Evidencia:** `scripts/pipeline/tune.py`, `src/training/tuning_study.py`, `tests/training/test_hpo_protocol.py`, [protocolo](../experimentos/hpo.md).

## DEC-013 — Enmienda de presupuesto HPO: 60 → 25

- **Fecha:** 2026-09-23, enmienda durante la ejecución.
- **Alcance:** reducción del número máximo de intentos; se mantienen el objetivo y el espacio de búsqueda.
- **Decisión:** 25 intentos totales incluyendo COMPLETE, PRUNED y FAIL, conservando IDs y RNG de TPE.
- **Estado al cambio:** trials 0 y 1 completos; trial 2 interrumpido al detener la app para migrar. Se conserva como FAIL; no se repite su ID ni se presenta como entrenamiento completo.
- **Alcance:** splits, dataset, transforms, modelo, lógica de entrenamiento y máximo de 60 épocas intactos. El test sigue reservado para el ganador bloqueado al agotar los 25 intentos.
- **Transparencia:** se habían observado métricas parciales de validation. No presentar el nuevo presupuesto como prefijado desde el inicio. DEC-012 permanece como historial del plan original.
- **Respaldo:** SQLite, código y preflight originales en `budget_revisions/60-to-25/`; comparación de tablas de trials y RNG antes/después sin cambios.
- **Evidencia:** [enmienda y hashes](../reproducibilidad/evidencia/hpo_budget_amendment_25.json), [smoke revisado](../reproducibilidad/evidencia/hpo_smoke_complete_25.json), [continuidad entre workspaces](../reproducibilidad/hpo-continuidad.md).

<!-- hpo-lite0-seed42-completed -->

## Cierre de DEC-012/DEC-013 — Resultado experimental (2026-09-24)

Study `efficientnet_lite0_seed42_hpo_v1`: 25 intentos, ganador trial 0, validation Macro-F1 0.957292225; baseline 0.956086266; delta +0.120596 pp.

[Configuración seleccionada y resultados](../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md).

## DEC-014 — Interpretar el HPO sin promoverlo por validación solamente

- **Fecha:** revisión documental del 2026-09-24.
- **Evidencia:** +0.120596 pp en Macro-F1 de validación, pero −0.487707 pp en Macro-F1 de test frente al baseline vigente.
- **Criterio de reporte:** conservar trial 0 como ganador contractual por validación; no reordenar candidatos ni reabrir la búsqueda después de observar test.
- **Límite:** no afirmar superioridad de generalización, significancia estadística ni aptitud de producción. No reemplazar automáticamente el baseline/checkpoint desplegado.
- **Fases futuras:** entrenamiento formal, multi-seed, CV/LOSO y calibración requieren un protocolo posterior; no se ejecutaron en este cierre.
- **Trazabilidad:** [comparación y fuentes](./HPO_BASELINE_COMPARISON.md).

## DEC-015 — Comparación pareada multi-seed, exclusivamente validation

- **Fecha:** 2026-09-24. **Rama:** dev-abner2.
- **Diseño:** baseline 20260921_204608 frente a HPO trial 0,
  seeds 42/123/2026/3407/7777, diez slots, split seed_42 congelado.
- **Implementación:** reutiliza train.py/fit; `--skip-test` explícito preserva
  el flujo general; export bloqueado en ese modo. Sin cambios en CornDataset.
- **No reutilización histórica:** equivalencia completa no verificable para
  ambos antecedentes; no mezclar diagnósticos/entornos incompletos.
- **Recuperación:** saltar completos íntegros; intentos interrumpidos requieren
  diagnóstico, pues el loop no permite restauración completa por época.
- **Calendario:** reanudación manual prevista para el 2026-10-01, precedida de
  verificación de fuentes, manifiestos y resultados existentes. Sin lanzamiento automático.
- **Estado:** implementación validada localmente; 0/10 runs. No conclusión
  experimental ni promoción. [Protocolo y comandos](../experimentos/multiseed.md).
