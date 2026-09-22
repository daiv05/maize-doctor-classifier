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
