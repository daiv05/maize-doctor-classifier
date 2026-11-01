# Estado experimental actual

Punto de corte: **1 de noviembre de 2026**. No implica que todo artefacto histórico se haya repetido sobre el corpus vigente.

## Completado

- `seed_42`: 33 429 muestras elegibles, 23 400 / 5 014 / 5 015, nueve clases y auditoría de cero solapamiento exacto.
- `seed_42_source_grouped`: benchmark separado de 16 554 / 7 809 / 9 066 con once fuentes indivisibles.
- Modelo oficial EfficientNet-Lite0 `20260921_204608`: validación Macro-F1 0.9561, test Macro-F1 0.9480 y accuracy 97.81 %. Reproducido bit a bit el 2026-10-19 (`20261019_203033`).
- Exportación móvil del oficial: ONNX y TFLite FP32 con paridad aprobada; TFLite int8 de 3.74 MB con Macro-F1 0.9489 y accuracy 97.87 % sobre el test completo, y detector OOD con umbral 106.37. Evidencia en [`export_movil_20260921_204608/`](../reproducibilidad/evidencia/export_movil_20260921_204608/export_summary_int8.json). La app despliega este modelo desde 2026-11-01.
- HPO `efficientnet_lite0_seed42_hpo_v1`: **25 intentos** (8 COMPLETE, 15 PRUNED, 2 FAIL por interrupción); ganador trial 0, mejor época 44. Presupuesto original de 60 reducido a 25. Test único del ganador: accuracy 0.9753 y Macro-F1 0.9431 sobre 5 015 muestras; el test no intervino en la búsqueda ni en la poda. No se adoptó.
- Benchmark source-grouped `20260921_180112`, útil como resultado negativo de cambio de dominio.
- Identidad `sample_id`, SHA-256 de archivo, manifests, locks y contratos de runs.
- [Auditoría de píxeles](../reproducibilidad/evidencia/pixel_duplicate_audit/PIXEL_DUPLICATE_AUDIT.md): 33 429 elegibles, cero grupos duplicados exactos, cero cross-split y 0 % del test afectado.
- [Análisis por fuente](../reproducibilidad/evidencia/source_analysis/SOURCE_ANALYSIS.md): 11 fuentes y asociación fuente–etiqueta V de Cramér 0.5030, descriptiva.
- [LOSO baseline de dos fuentes](../reproducibilidad/evidencia/loso_baseline/LOSO_RESULTS.md): Macro-F1 externo 0.9931 (`maize-diseases`) y 0.9055 (`multicrop-disease-maiz`) con seed 42.
- [Estabilidad de baseline y LOSO con tres semillas](../reproducibilidad/evidencia/multiseed_source_stability/MULTISEED_SOURCE_STABILITY.md): baseline 0.9435 ± 0.0052, LOSO `maize-diseases` 0.9933 ± 0.0006 y LOSO `multicrop-disease-maiz` 0.9019 ± 0.0044 (SD muestral, n = 3).
- [Ensamble histórico bajo LOSO multicrop](../reproducibilidad/evidencia/ensemble_loso/ENSEMBLE_LOSO_REPORT.md): 0.9284 ± 0.0042 frente a B0 0.9240 ± 0.0094; Δ pareado +0.0045 ± 0.0080, positivo en dos seeds y negativo en una. Resultado mixto.
- [Síntesis experimental](../tesis/EXPERIMENTAL_RESULTS_SUMMARY.md) y [respaldo de modelos](../tesis/MODEL_PACKAGE_README.md): 18 checkpoints cotejados con hashes de evidencia, distribuidos como `.tar.gz`.
- Evidencia de la etapa 2 (Lite0 `20260812_221429`, equidad por fuente y validación cruzada) en [`etapa2_lite0_20260812_221429/`](../resultados/evidencia/etapa2_lite0_20260812_221429/summary.json).
- Evaluación en 97 fotos de campo externas al corpus: 41/97 aciertos. Ver [Resultados](../resultados/index.md).

## Interpretación del HPO cerrado

El ganador superó al baseline en validación Macro-F1 (0.957292 frente a 0.956086) pero quedó por debajo en test (0.943125 frente a 0.948002). No demuestra una mejora de generalización, es de una sola semilla y no establece significancia estadística. [Comparación y límites](../tesis/HPO_BASELINE_COMPARISON.md).

## Estudios multi-seed

El [estudio de estabilidad baseline/LOSO](../reproducibilidad/evidencia/multiseed_source_stability/MULTISEED_SOURCE_STABILITY.md) está cerrado: nueve runs verificadas (seeds 42, 123 y 2026 por escenario), tres históricas reutilizadas y seis nuevas, cada una con selection lock previo al holdout y `evaluation_count=1`. El intento interrumpido de `maize-diseases/seed_123` se conserva aparte y no entra en las cifras.

La [comparación pareada baseline/HPO](./multiseed.md) tiene 2 de 10 entrenamientos de validación completados (seed 42 de cada configuración, 2026-10-19). La diferencia de +0.0012 no es interpretable y el estudio no continuó.

## Estado de las fases

| Actividad | Estado |
|---|---|
| HPO de 25 intentos | COMPLETADO |
| Auditorías de `sample_id`, `file_sha256` y `pixel_sha256` | COMPLETADAS; duplicados exactos |
| Análisis por fuente | COMPLETADO; descriptivo |
| Baseline multi-seed y LOSO de dos fuentes (42 / 123 / 2026) | COMPLETADOS |
| Comparación pareada baseline/HPO | PARCIAL; 2 de 10 entrenamientos de validación |
| Ensamble histórico bajo LOSO multicrop | COMPLETADO; resultado mixto |
| Exportación del oficial: ONNX, TFLite FP32 e int8 | COMPLETADA; paridad FP32 aprobada |
| Paridad de muestra del int8 (30 imágenes) | NO APROBADA por una imagen límite (max \|Δprob\| 0.198 frente al umbral 0.15) |
| Evaluación int8 sobre el test completo | COMPLETADA; Macro-F1 0.9489 |
| Detector OOD del oficial | COMPLETADO; 5/5 imágenes sintéticas detectadas y 3.8 % de falsos rechazos en 500 imágenes de test |
| Medición en dispositivo | A CARGO DE LA SESIÓN DE DISPOSITIVO; no se registra aquí |
| LOSO final general y validación cruzada final | PENDIENTES |
| Entrenamiento y evaluación formal | PENDIENTES |

La auditoría exacta no sustituye el análisis por fuente ni la revisión de casi duplicados. El hallazgo histórico de unos 15 grupos y 0.68 % del test no se reprodujo en la materialización vigente.

## Pendiente

- Analizar los errores persistentes de `lethal_necrosis` sin ajustar el modelo con el holdout.
- CV estratificada y source-grouped repetidas sobre la materialización vigente.
- LOSO final y benchmark cross-source del modelo elegido.
- Auditoría perceptual entre splits (PHash).
- Mejorar la generalización fuera del corpus: en fotos de campo el oficial acierta 41/97.
- Comparación controlada de imagen completa frente a perspectivas segmentadas con *quality gate*.

## Cierre de las cinco observaciones

1. **Una sola semilla:** completada para baseline estándar y dos LOSO prioritarios con seeds 42, 123 y 2026; se reporta SD muestral.
2. **Procedencia y fuentes:** analizadas las dos fuentes prioritarias; `multicrop-disease-maiz` mantiene la caída LOSO en tres seeds. La asociación no prueba causalidad.
3. **Grupos píxel-idénticos con etiquetas contradictorias:** cerrada para la materialización vigente; cuatro conflictos conocidos quedan en exclusiones contractuales.
4. **Duplicados exactos train/test:** cerrada para `seed_42`; cero solapamiento exacto y 0 % del test afectado, sin garantía sobre casi duplicados.
5. **Ensamble fuera de fuente:** medido con resultado mixto bajo LOSO multicrop; mejora en dos seeds, empeora en una y tiene peor ECE que B0.
