# Resultados experimentales

Resumen de los experimentos con checkpoint respaldado. Los test estándar, los holdouts LOSO y las validaciones internas miden cosas distintas y no se comparan entre sí. `N/A` indica que no hay una medición comparable.

## Corpus y protocolos

El corpus tiene 33 429 muestras elegibles de nueve clases. El desarrollo estratificado `seed_42` reparte 23 400 de entrenamiento, 5 014 de validación y 5 015 de prueba. Los experimentos son un baseline EfficientNet-Lite0, su optimización de hiperparámetros (HPO), una comparación histórica de arquitecturas y ensamble, la repetición con tres semillas, la exclusión de una fuente (LOSO, *Leave-One-Source-Out*: se entrena y selecciona sin esa fuente y se evalúa sobre todas sus muestras) y el ensamble bajo LOSO.

La [auditoría de píxeles](evidencia/pixel_duplicate_audit/PIXEL_DUPLICATE_AUDIT.md) no encontró solapamientos exactos entre particiones. El [análisis por fuente](evidencia/source_analysis/SOURCE_ANALYSIS.md) describe once fuentes y una asociación fuente–etiqueta de V de Cramér 0.5030, que es descriptiva. Las fuentes de cada cifra están en el [registro de evidencias](evidencias.md), el [resumen de estabilidad multi-semilla y LOSO](evidencia/multiseed_source_stability/multiseed_summary.json), el [informe de HPO](evidencia/hpo_lite0_seed42/HPO_REPORT.md) y el [resumen del ensamble LOSO](evidencia/ensemble_loso/ensemble_loso_summary.json). Rutas y SHA-256 de los checkpoints: [MODEL_INDEX.csv](evidencia/model_package/MODEL_INDEX.csv) y [MODEL_COMPARISON_RESULTS.csv](evidencia/model_package/MODEL_COMPARISON_RESULTS.csv).

## Comparación por protocolo

Con tres semillas se informa media ± desviación estándar muestral; en los demás casos es una sola evaluación. Los valores con más decimales están en `MODEL_COMPARISON_RESULTS.csv`. El Macro-F1 N/P/K agrupado usa otra taxonomía y no reemplaza al de nueve clases.

| Experimento | Semillas | Macro-F1 validación | Macro-F1 test | Macro-F1 LOSO | ECE | N/P/K agrupado | Bytes por `best.pth` |
|---|---|---:|---:|---:|---:|---:|---:|
| Lite0 baseline | 42, 123, 2026 | 0.9564 ± 0.0037 | 0.9435 ± 0.0052 | N/A | 0.1370 ± 0.0025 | 0.9784 (semilla 42) | 13 785 701 |
| Lite0 HPO trial 0 | 42 | 0.9573 | 0.9431 | N/A | 0.0600 | 0.9754 | 13 801 629 |
| B0 histórico | sin registro | 0.9610 | 0.9483 | N/A | 0.1393 | 0.9783 | 16 355 557 |
| ShuffleNet-V2-x1.0 histórico | sin registro | 0.9502 | 0.9330 | N/A | 0.1415 | 0.9744 | 5 221 267 |
| Ensamble histórico de tres modelos | sin registro | N/A | 0.9567 | N/A | N/A | N/A | sin checkpoint único |
| Lite0 LOSO `maize-diseases` | 42, 123, 2026 | 0.9519 ± 0.0028 | N/A | 0.9933 ± 0.0006 | 0.1311 ± 0.0087 | N/A | 13 785 701 |
| Lite0 LOSO `multicrop-disease-maiz` | 42, 123, 2026 | 0.9543 ± 0.0045 | N/A | 0.9019 ± 0.0044 | 0.0462 ± 0.0056 | N/A | 13 785 701 |
| B0 LOSO `multicrop-disease-maiz` | 42, 123, 2026 | ver CSV | N/A | 0.9240 ± 0.0094 | 0.0446 ± 0.0122 | N/A | 16 355 557 |
| ShuffleNet LOSO `multicrop-disease-maiz` | 42, 123, 2026 | ver CSV | N/A | 0.8959 ± 0.0149 | ver CSV | N/A | 5 221 267 |
| Ensamble LOSO `multicrop-disease-maiz` | 42, 123, 2026 | N/A | N/A | 0.9284 ± 0.0042 | 0.0891 ± 0.0064 | N/A | tres miembros por semilla |

Los 18 checkpoints respaldados son las corridas individuales de la tabla: 16 del inventario principal y los B0 y ShuffleNet históricos. El Lite0 del ensamble histórico (`20260812_221429`) no está en este respaldo, pero sí en `docs/es/resultados/evidencia/etapa2_lite0_20260812_221429/` (resumen y evaluación int8) y en la descarga de artefactos de la etapa 2. Las métricas históricas proceden de una materialización anterior del corpus y no son comparables punto a punto con el corpus vigente.

## HPO y semillas

El estudio HPO ejecutó 25 intentos: 8 completos, 15 podados y 2 fallidos por interrupción. El trial 0 se eligió por validación (0.9573 frente a 0.9561 del baseline) y se evaluó una sola vez en prueba: 0.9431 frente a 0.9480 del baseline. La mejora en validación no se trasladó a prueba y el baseline sigue siendo el modelo oficial.

Las tres semillas del baseline dan Macro-F1 de prueba 0.9480, 0.9379 y 0.9446, con media 0.9435 ± 0.0052. En el test estándar, el F1 medio por clase fue 0.8915 ± 0.0139 para nitrógeno, 0.9278 ± 0.0066 para fósforo y 0.8089 ± 0.0145 para potasio, con soportes 127, 140 y 93. Los valores por semilla están en [baseline_class_multiseed.csv](evidencia/multiseed_source_stability/baseline_class_multiseed.csv).

## Fuente excluida y ensamble

Lite0 sin `maize-diseases` obtuvo 0.9931, 0.9927 y 0.9939 (media 0.9933 ± 0.0006). Lite0 sin `multicrop-disease-maiz` obtuvo 0.9055, 0.8970 y 0.9033 (media 0.9019 ± 0.0044). Esa fuente aporta 958 `common_rust`, 1 627 `fall_armyworm` y 3 231 `lethal_necrosis` al holdout de 5 816 muestras, y su exclusión también cambia la distribución de entrenamiento.

El ensamble promedia las probabilidades softmax de Lite0, B0 y ShuffleNet con pesos iguales y decide por `argmax`. En el test estratificado histórico superó a B0 por +0.0083. Bajo LOSO se reutilizaron los tres Lite0 y se entrenaron B0 y ShuffleNet de cada semilla sin acceso a la fuente excluida; los locks conjuntos se fijaron antes de inferir.

| Semilla | Lite0 | B0 | ShuffleNet | Ensamble | Ensamble − B0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.9055 | 0.9173 | 0.9130 | 0.9301 | +0.0127 |
| 123 | 0.8970 | 0.9348 | 0.8865 | 0.9316 | −0.0032 |
| 2026 | 0.9033 | 0.9198 | 0.8880 | 0.9237 | +0.0039 |

La diferencia media frente a B0 es +0.0045 ± 0.0080: positiva en dos semillas y negativa en una. El ECE del ensamble (0.0891) es peor que el de B0 (0.0446). Los resultados completos están en [ensemble_loso_results.csv](evidencia/ensemble_loso/ensemble_loso_results.csv), [ensemble_component_results.csv](evidencia/ensemble_loso/ensemble_component_results.csv) y [ensemble_loso_class_metrics.csv](evidencia/ensemble_loso/ensemble_loso_class_metrics.csv).

F1 por clase de B0 y del ensamble en la fuente excluida (CR `common_rust`, FA `fall_armyworm`, LN `lethal_necrosis`):

| Semilla | CR B0 → ensamble | FA B0 → ensamble | LN B0 → ensamble |
|---:|---:|---:|---:|
| 42 | 0.9897 → 0.9933 | 0.9021 → 0.9087 | 0.8602 → 0.8882 |
| 123 | 0.9953 → 0.9943 | 0.9115 → 0.9033 | 0.8975 → 0.8972 |
| 2026 | 0.9897 → 0.9927 | 0.9037 → 0.8934 | 0.8662 → 0.8850 |

En `lethal_necrosis` el recall medio pasa de 0.7778 ± 0.0320 (B0) a 0.8025 ± 0.0103 (ensamble) y el F1 de 0.8746 ± 0.0200 a 0.8901 ± 0.0063. La mejora es de esa clase; `fall_armyworm` empeora en F1 medio.

## Otros experimentos

| Experimento | Resultado | Alcance |
|---|---|---|
| [Benchmark source-grouped](../resultados/generalizacion.md), corrida `20260921_180112` | Macro-F1 de validación 0.2669 y de prueba 0.4007 | Fuentes completas por partición; no es LOSO ni el test estándar |
| [CV estratificada y agrupada](../resultados/evidencia/kfold_2x2_comparacion.csv) | 0.9311 ± 0.0099 estratificada y 0.6026 ± 0.1240 agrupada (± es la semiamplitud del IC95; la agrupada es el Macro-F1 sobre las clases presentes en cada pliegue, el convencional es 0.4359) | Cinco pliegues, Lite0 de la etapa 2 |
| [Corpus segmentado](../resultados/evidencia/manifiesto_corridas.csv), corrida `20260907_163546` | Macro-F1 de prueba 0.7191 y accuracy 0.8903 | Exploratorio; summary y checkpoint no versionados |

## Integridad de los datos

- Los quince grupos de píxeles idénticos y el 0.68 % de prueba con gemelo en entrenamiento que reportó una materialización anterior no se reproducen en las 33 429 muestras vigentes: hay 0 de 5 015 imágenes de prueba con duplicado exacto entre particiones.
- Cuatro grupos con etiquetas contradictorias (ocho archivos) están excluidos del contrato de datos.
- No se ejecutó PHash, de modo que los casi duplicados no están medidos.
- El LOSO se midió con dos fuentes y el ensamble bajo una sola exclusión; la comparación pareada baseline/HPO de cinco semillas completó 2 de 10 entrenamientos (solo validación).
