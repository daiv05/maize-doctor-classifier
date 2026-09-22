# Estado experimental actual

Punto de corte: **22 de septiembre de 2026**. No implica que todo artefacto histórico se haya repetido sobre el corpus vigente.

## Completado

- `seed_42`: 33 429 muestras elegibles, 23 400/5 014/5 015, nueve clases y auditoría de cero solapamiento exacto.
- `seed_42_source_grouped`: benchmark separado de 16 554/7 809/9 066 con once fuentes indivisibles.
- Baseline principal de desarrollo EfficientNet-Lite0 `20260921_204608`.
- Benchmark source-grouped `20260921_180112`, útil como resultado negativo/domain shift.
- Identidad `sample_id`, SHA-256 de contenido, manifests/locks y contratos de runs.
- Experimentos históricos de baselines, HPO reducido, ensamble, CV, procedencia/LOSO, equidad, XAI, segmentación y exportación.

## Pendiente

- estudio Optuna nuevo de 60 trials sobre el protocolo vigente;
- entrenamiento formal con la configuración elegida;
- validación multi-seed de esa configuración;
- CV estratificada y source-grouped repetidas sobre la materialización vigente;
- LOSO final y benchmark cross-source del modelo elegido;
- auditoría perceptual entre splits;
- calibración explícita y validación de umbrales;
- error analysis cualitativo N/P/K;
- exportación Int8, evaluación completa y dispositivo para el checkpoint vigente;
- comparación controlada `full_image` frente a perspectivas segmentadas con *quality gate*.

El orden y los criterios de salida se mantienen en [Backlog de investigación](../tesis/RESEARCH_BACKLOG.md).
