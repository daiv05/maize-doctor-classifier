# Resultados experimentales consolidados

Estado al 2 de octubre de 2026. Este documento reúne resultados **ya registrados** para que el equipo compare alternativas. No selecciona una arquitectura ni autoriza entrenamiento o evaluación formal. Cada cifra corresponde al protocolo indicado; los test estándar, holdouts LOSO y validaciones internas no son intercambiables. `N/A` significa que no existe una medición comparable en la evidencia consultada.

## Corpus y preguntas experimentales

La materialización canónica tiene 33.429 muestras elegibles de nueve clases. El desarrollo estratificado `seed_42` contiene 23.400 train, 5.014 validation y 5.015 test. Se ensayaron: un baseline EfficientNet-Lite0, HPO de ese modelo, comparación histórica de arquitecturas y ensamble, repetición con tres semillas, exclusión de fuentes y el ensamble bajo exclusión de `multicrop-disease-maiz`. El objetivo de LOSO (*Leave-One-Source-Out*) es entrenar y seleccionar sin una fuente y después medir sobre todas sus muestras elegibles. No es el test estándar, un cambio de semilla ni K-Fold.

La [auditoría de integridad](../reproducibilidad/evidencia/pixel_duplicate_audit/PIXEL_DUPLICATE_AUDIT.md) comprobó identidad por `sample_id`, SHA-256 de archivo y píxeles: no encontró solapamientos exactos cross-split en el corpus vigente. Ocho imágenes de cuatro grupos con etiquetas contradictorias ya estaban excluidas. El [análisis por fuente](../reproducibilidad/evidencia/source_analysis/SOURCE_ANALYSIS.md) describió once fuentes y una asociación fuente–etiqueta V de Cramér 0,5030; es una asociación descriptiva, no una demostración de que el modelo use un atajo de procedencia.

Las referencias primarias son el [registro de evidencias](EVIDENCE_REGISTRY.md), la [estabilidad multi-seed/LOSO](../reproducibilidad/evidencia/multiseed_source_stability/MULTISEED_SOURCE_STABILITY.md), el [HPO](../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md) y la [evaluación del ensamble LOSO](../reproducibilidad/evidencia/ensemble_loso/ENSEMBLE_LOSO_REPORT.md). Las rutas y SHA-256 de los checkpoints respaldados están además en el [índice de modelos](evidencia/model_package/MODEL_INDEX.csv) y la [tabla comparativa](evidencia/model_package/MODEL_COMPARISON_RESULTS.csv), copiados también en el paquete de modelos.

## Comparación por protocolo

Macro-F1 y ECE son media ± desviación estándar **muestral** cuando aparecen tres semillas; de lo contrario se trata de una sola evaluación. El tamaño exacto, la seed, la mejor época, el checkpoint y el SHA-256 de cada run están en `MODEL_INDEX.csv`. N/P/K agrupado usa otra taxonomía; no sustituye Macro-F1 de nueve clases.

| Arquitectura / experimento | Seeds | Validation Macro-F1 | Test estándar Macro-F1 | LOSO Macro-F1 | ECE final | N/P/K agrupado | Checkpoints | Bytes por `best.pth` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Lite0 baseline vigente | 42, 123, 2026 | 0,956408 ± 0,003723 | 0,943489 ± 0,005158 | N/A | 0,136975 ± 0,002464 | 0,978385 solo seed 42 | 3 | 13.785.701 |
| Lite0 HPO trial 0 | 42 | 0,957292 | 0,943125 | N/A | 0,060042 | 0,975380 | 1 | 13.801.629 |
| B0 histórico, test estratificado | no registrada | 0,960982 | 0,948333 | N/A | 0,139307 | 0,978265 | 1 | 16.355.557 |
| ShuffleNet-V2-x1.0 histórico | no registrada | 0,950249 | 0,932980 | N/A | 0,141539 | 0,974449 | 1 | 5.221.267 |
| Ensamble histórico de tres modelos | no registrada | N/A | 0,956657 | N/A | N/A | N/A | sin checkpoint único | N/A |
| Lite0 LOSO `maize-diseases` | 42, 123, 2026 | 0,951942 ± 0,002815 | N/A | 0,993264 ± 0,000600 | 0,131112 ± 0,008749 | N/A | 3 | 13.785.701 |
| Lite0 LOSO `multicrop-disease-maiz` | 42, 123, 2026 | 0,954282 ± 0,004452 | N/A | 0,901906 ± 0,004422 | 0,046155 ± 0,005557 | N/A | 3 | 13.785.701 |
| B0 LOSO `multicrop-disease-maiz` | 42, 123, 2026 | ver cada run | N/A | 0,923978 ± 0,009443 | 0,044573 ± 0,012196 | N/A | 3 | 16.355.557 |
| ShuffleNet LOSO `multicrop-disease-maiz` | 42, 123, 2026 | ver cada run | N/A | ver resultados por seed | ver cada run | N/A | 3 | 5.221.267 |
| Ensamble LOSO `multicrop-disease-maiz` | 42, 123, 2026 | N/A | N/A | 0,928446 ± 0,004168 | 0,089088 ± 0,006372 | N/A | tres miembros por seed | N/A |

Los 18 checkpoints físicos respaldados corresponden a las runs individuales de la tabla: 16 exigidos por el inventario principal y dos históricos B0/ShuffleNet incluidos para preservar la comparación original. El Lite0 desplegado en el ensamble histórico (`20260812_221429`) no está disponible aquí; no se sustituyó por el baseline vigente ni se le atribuyó un hash. Por esa razón el ensamble histórico figura como resultado documentado, no como modelo completo reconstituible en el respaldo. Las métricas históricas proceden de una materialización anterior y no deben leerse como una comparación pareada con el corpus vigente.

## HPO y repetición

El estudio HPO solicitó 25 intentos: 8 `COMPLETE`, 15 `PRUNED` y 2 `FAIL` por interrupción. El trial 0 se seleccionó **solo por validation** (0,957292 frente a 0,956086 del baseline seed 42); después se hizo una evaluación final única del test, con 0,943125 frente a 0,948002 del baseline seed 42. El test no participó en la búsqueda ni en la poda. Este resultado describe una mejora pequeña en validation que no se trasladó al test; no se lo interpreta como selección de modelo final.

Las tres semillas 42/123/2026 del baseline dieron test Macro-F1 0,948002 / 0,937867 / 0,944597, media 0,943489 ± 0,005158 y rango 0,937867–0,948002. Para N/P/K en el test estándar, F1 medio por clase fue N 0,891506 ± 0,013902, P 0,927756 ± 0,006590 y K 0,808883 ± 0,014451; los soportes fueron 127, 140 y 93 por seed. La [tabla por clase](../reproducibilidad/evidencia/multiseed_source_stability/baseline_class_multiseed.csv) conserva los valores individuales. Tres semillas permiten observar variación, no establecer significancia estadística fuerte.

## Fuente excluida y ensamble

Lite0 LOSO `maize-diseases` obtuvo 0,993124 / 0,992747 / 0,993922 (media 0,993264 ± 0,000600). Lite0 LOSO `multicrop-disease-maiz` obtuvo 0,905490 / 0,896964 / 0,903264 (media 0,901906 ± 0,004422). La segunda fuente contiene 958 `common_rust`, 1.627 `fall_armyworm` y 3.231 `lethal_necrosis` en su holdout de 5.816 muestras. El cambio de fuente también cambia cantidad y distribución de train; la brecha no prueba por sí sola un atajo causal.

El ensamble histórico promediaba uniformemente las probabilidades softmax de Lite0, B0 y ShuffleNet y decidía por `argmax`. Su delta exacto frente a B0 en el test estratificado anterior fue +0,008323935. Para LOSO multicrop se reutilizaron los tres Lite0 válidos y se entrenaron B0 y ShuffleNet de cada seed sin acceso a esa fuente en train/validation. Los tres locks conjuntos se fijaron antes de inferir; no se ajustaron miembros ni pesos con el holdout.

| Seed | Lite0 | B0 | ShuffleNet | Ensamble | Δ ensamble − B0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0,905490 | 0,917318 | 0,913037 | 0,930055 | +0,012737 |
| 123 | 0,896964 | 0,934785 | 0,886505 | 0,931569 | −0,003216 |
| 2026 | 0,903264 | 0,919831 | 0,888009 | 0,923713 | +0,003883 |

El delta pareado medio fue +0,004468 ± 0,007993: dos seeds positivas y una negativa. El resultado es **mixto/inestable**; no demuestra superioridad robusta. El ECE medio del ensamble (0,089088) fue peor que el de B0 (0,044573). Los CSV completos de [resultados](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_results.csv), [componentes](../reproducibilidad/evidencia/ensemble_loso/ensemble_component_results.csv) y [clases](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_class_metrics.csv) permiten revisar cada medida.

## Otros experimentos documentados

Estas cifras responden preguntas distintas y no entran al inventario principal de 18 checkpoints. Se conservan aquí para dar contexto, no para ordenar modelos.

| Experimento | Resultado registrado | Alcance |
|---|---|---|
| [Benchmark source-grouped](../resultados/run-20260921-source-grouped.md), run `20260921_180112` | Validation Macro-F1 0,266927; test Macro-F1 0,400665 | Fuentes indivisibles por split; no es LOSO ni el test estándar |
| [CV estratificada y agrupada histórica](../resultados/evidencia/kfold_2x2_comparacion.csv) | 0,931145 ± 0,009872 estratificada; 0,602575 ± 0,123966 agrupada | Cinco pliegues; no sustituye las tres seeds ni el LOSO vigente |
| [Clasificación con corpus segmentado](../resultados/evidencia/manifiesto_corridas.csv), run `20260907_163546` | Test Macro-F1 0,719082; accuracy 0,890329 | Resultado exploratorio; summary/checkpoint no versionados aquí |

Las pruebas históricas de explicabilidad, equidad y exportación tienen evidencias en el [registro general](EVIDENCE_REGISTRY.md), pero no se agregan a la tabla principal porque no ofrecen el mismo par protocolo–métrica–checkpoint. Tampoco se vuelve a inferir ni se recalcula el test durante esta síntesis.

| Seed | CR F1 B0 → ensamble | FA F1 B0 → ensamble | LN F1 B0 → ensamble |
|---:|---:|---:|---:|
| 42 | 0,989659 → 0,993254 | 0,902130 → 0,908673 | 0,860166 → 0,888239 |
| 123 | 0,995320 → 0,994286 | 0,911542 → 0,903268 | 0,897493 → 0,897152 |
| 2026 | 0,989659 → 0,992739 | 0,903670 → 0,893421 | 0,866164 → 0,884980 |

En `lethal_necrosis`, B0 tuvo recall medio 0,777778 ± 0,031967 y F1 0,874607 ± 0,020045; el ensamble 0,802538 ± 0,010260 y 0,890124 ± 0,006301. Es una mejora **descriptiva de esa clase** en el agregado, no una conclusión global: FA empeora en F1 medio y LN no mejora en la seed 123.

## Cierre de las cinco observaciones

1. **Una sola semilla:** atendida con tres seeds para baseline estándar y las dos fuentes LOSO prioritarias; se informa SD muestral.
2. **Procedencia:** `maize-diseases` y `multicrop-disease-maiz` medidas con LOSO multi-seed. La asociación fuente–etiqueta y la caída multicrop son descriptivas, no causales.
3. **Quince grupos píxel-idénticos históricos:** no reproducidos entre las 33.429 muestras elegibles vigentes. Cuatro grupos contradictorios (ocho archivos) ya estaban excluidos del contrato de datos.
4. **0,68 % de test con gemelo en train histórico:** no reproducido; la auditoría vigente halló `0/5015 = 0 %` de duplicados exactos por píxeles entre splits.
5. **Ensamble fuera de fuente:** medido en LOSO multicrop con tres seeds; resultado mixto frente a B0. El control extra `maize-diseases` no se ejecutó porque requeriría dos entrenamientos nuevos.

## Alcance pendiente

No se ejecutó PHash, por lo que los casi duplicados siguen sin medir. Solo dos fuentes fueron priorizadas en el LOSO inicial y el ensamble se midió bajo una sola exclusión de fuente. El test histórico se ha observado repetidamente; ninguna cifra de esta síntesis se empleó para reajustar configuraciones. Quedan separados y pendientes la comparación formal baseline/HPO de cinco semillas (0/10 runs), CV y LOSO final general, selección por el equipo, entrenamiento formal y evaluación final formal. El respaldo de modelos es una copia verificable de artefactos existentes, no una nueva ejecución experimental.
