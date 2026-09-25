# Estado experimental actual

Punto de corte: **24 de septiembre de 2026**. No implica que todo artefacto histórico se haya repetido sobre el corpus vigente.

## Completado

- `seed_42`: 33 429 muestras elegibles, 23 400/5 014/5 015, nueve clases y auditoría de cero solapamiento exacto.
- `seed_42_source_grouped`: benchmark separado de 16 554/7 809/9 066 con once fuentes indivisibles.
- Baseline principal de desarrollo EfficientNet-Lite0 `20260921_204608`.
- HPO `efficientnet_lite0_seed42_hpo_v1`: **25 intentos**, 8 COMPLETE, 15 PRUNED y 2 FAIL por interrupción; ganador trial 0, mejor época 44. Presupuesto original de 60 reducido con autorización registrada.
- Selección bloqueada y test único del checkpoint ganador: accuracy 0.975274177 y Macro-F1 0.943125073 sobre 5 015 muestras. Test no intervino en búsqueda ni poda.
- Benchmark source-grouped `20260921_180112`, útil como resultado negativo/domain shift.
- Identidad `sample_id`, SHA-256 de contenido, manifests/locks y contratos de runs.
- Experimentos históricos de baselines, HPO reducido, ensamble, CV, procedencia/LOSO, equidad, XAI, segmentación y exportación.

## Interpretación del HPO cerrado

El ganador superó al baseline en validation Macro-F1 (0.957292225 frente a
0.956086266), pero quedó por debajo en test (0.943125073 frente a 0.948002144).
No demuestra una mejora de generalización ni sustituye automáticamente al baseline.
El resultado es de una sola semilla; no establece significancia estadística.
[Comparación y límites](../tesis/HPO_BASELINE_COMPARISON.md).

Al revisar el cierre no había aplicaciones de entrenamiento activas.

## Comparación multi-seed preparada

[Baseline frente a HPO trial 0](./multiseed.md): cinco seeds emparejadas
(42, 123, 2026, 3407, 7777), mismo split, validation-only. Runner y smoke local
implementados; **0/10 runs iniciadas**. La reanudación manual está prevista para
el **1 de octubre de 2026**, tras verificar código, manifiestos y artefactos.
No se volvió a evaluar test ni se seleccionó una configuración para la fase siguiente.

## Pendiente

- entrenamiento formal con la configuración elegida;
- completar la comparación multi-seed baseline/HPO preparada;
- CV estratificada y source-grouped repetidas sobre la materialización vigente;
- LOSO final y benchmark cross-source del modelo elegido;
- auditoría perceptual entre splits;
- calibración explícita y validación de umbrales;
- error analysis cualitativo N/P/K;
- exportación Int8, evaluación completa y dispositivo para el checkpoint vigente;
- comparación controlada `full_image` frente a perspectivas segmentadas con *quality gate*.

El orden y los criterios de salida se mantienen en [Backlog de investigación](../tesis/RESEARCH_BACKLOG.md).
