# Registro de evidencias

Origen de cada cifra del proyecto: cada afirmación apunta a un artefacto versionado o a una ruta remota estable. Los hashes son SHA-256.

## Corpus y particiones vigentes

| Afirmación verificable | Artefacto | Run/protocolo | Hash disponible | Documento que la usa |
|---|---|---|---|---|
| 33 437 muestras descubiertas y válidas; 8 exclusiones; 33 429 elegibles | [`split_protocols_summary.json`](../reproducibilidad/evidencia/split_protocols_summary.json) y auditoría remota | preparación vigente | resumen versionado: `9e24dc42…656f0a`; fuente en `corn-outputs:/splits/.../preparation_audit.json` | pipeline de datos, estado actual, evolución |
| `seed_42` = 23 400 / 5 014 / 5 015, nueve clases por split | mismo resumen + lock exacto | `seed_42` | [`manifest.lock.json`](../reproducibilidad/evidencia/seed_42_manifest_lock.json): `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` | metodología, README, baseline actual |
| Cero solapamiento de `sample_id`, SHA-256 y `effective_group_id` | `preparation_audit.json` remoto, extracto en resumen | `seed_42` | incluido en el lock anterior | pipeline de datos y protocolos |
| 33 429 elegibles, cero duplicados por pixel_sha256, cero cross-split, 0 % de test afectado; 4 grupos/8 archivos ya excluidos | [resumen e inventario](../reproducibilidad/evidencia/pixel_duplicate_audit/pixel_duplicate_audit.md) | `seed_42` vigente | master: `64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163`; lock: `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` | metodología y estado actual |
| 11 fuentes, distribución de 33 429 muestras, V de Cramér 0.5030 y métricas baseline/HPO por fuente | [paquete de análisis por fuente](../reproducibilidad/evidencia/source_analysis/source_analysis.md), [resumen y hashes](../reproducibilidad/evidencia/source_analysis/source_analysis_summary.json) | `seed_42`; validation comparativa y test descriptivo post-hoc | hashes SHA-256 de manifests y cuatro CSV de predicciones registrados en el resumen | estado actual y análisis de procedencia; no selecciona modelo |
| LOSO baseline: `maize-diseases` 6 989, F1 0.993124; `multicrop-disease-maiz` 5 816, F1 0.905490; locks previos al holdout | [resultados](../reproducibilidad/evidencia/loso_baseline/loso_results.csv), [resumen y timestamps](../reproducibilidad/evidencia/loso_baseline/loso_summary.json) | `loso_baseline_seed42_v1`, dos runs seed 42 | checkpoints `b16b1ef8…dcc` y `bd778a28…517`; hashes completos de splits, locks y runs en [protocolo](../reproducibilidad/evidencia/loso_baseline/loso_summary.json) y CSV | comparación descriptiva con validation histórica; no causalidad ni significancia con una semilla |
| `config_sha256` del lock = SHA-256 de `split_parameters`; regenerar `seed_42` con la configuración actual reproduce el lock byte a byte | [`manifest.lock.json`](../reproducibilidad/evidencia/seed_42_manifest_lock.json) | configuración de `seed_42` | `config_sha256`: `8f0e560f…`; lock: `0db3ff3e…` | pipeline de datos |
| PHash no ejecutado; solapamiento perceptual desconocido | `preparation_audit.json`: `deduplicate_perceptual=false`, conteo `null` | ambos protocolos actuales | rutas declaradas dentro del resumen | metodología y limitaciones |
| `seed_42_source_grouped` = 16 554 / 7 809 / 9 066; 5/3/3 fuentes | resumen de protocolos | benchmark actual | fuente remota declarada en JSON | protocolos y ficha cross-source |

## Baseline de desarrollo `20260921_204608`

Ruta original: `corn-outputs:/main/efficientnet_lite0/20260921_204608/`.

| Afirmación verificable | Artefacto versionado | Hash del artefacto | Sección usuaria |
|---|---|---|---|
| 47/60 épocas; mejor época 39; val Macro-F1 0.956086; test Macro-F1 0.948002; accuracy 0.978066 | [`hpo_baseline_summary.json`](../reproducibilidad/evidencia/hpo_baseline_summary.json) | `20bb0945574913ffa8c736fecbcf25d932e6b87ce3e7a7c59d98439558432346` | ficha del run, registro experimental, README |
| checkpoint `best.pth` | mismo summary contractual | checkpoint: `860180f33bf1749ffee9f3cccb9569a47afd9ca0fcef52713e18f99f82a57859` | ficha del run |
| configuración efectiva | mismo summary contractual | config: `53cc091e505057f6751a7b6151f403e03f830fac3c0966df9ba51e2df2814e8b` | ficha del run y registro experimental |
| precision, recall, F1 y soporte de las nueve clases | [`classification_report.csv`](../resultados/evidencia/run_20260921_204608_test_classification_report.csv) | `9b18915ab63447f71f572f18f52e443d1b337c20baff33114f4e9b502d49378b` | ficha del run |
| confusiones exactas, incluidas GLS→NCLB 18 y K→N 15 | [`confusion_matrix.csv`](../resultados/evidencia/run_20260921_204608_test_confusion_matrix.csv) | `0dfdf12e180078d150f48bb96726538da224aa5a96e5b1d910596d2755e65bea` | análisis por clase/N-P-K |
| ECE 0.139656; confianza aciertos 0.844379 y errores 0.687191 | [`test_calibration.json`](../resultados/evidencia/run_20260921_204608_test_calibration.json) | `8ede2e9d839fdcc87763d1dd4c0a93d235af6b95d0b1f6b1c5cf83422b7cf15d` | calibración |
| 15 errores con confianza ≥ 0.90 | `predictions.csv` remoto; recuento auditado | `predictions.csv` no versionado | ficha del run |
| Macro-F1 N/P/K agrupado 0.978385 y accuracy 0.984048 | [`test_grouped_metrics.json`](../resultados/evidencia/run_20260921_204608_test_grouped_metrics.json) | `ca94652860fb7cc53c73fbef5aa7e65c21a560666420f00eeca53bc8f8b44e35` | análisis N/P/K |
| métricas por `lab` y `real` | [`test_by_environment.csv`](../resultados/evidencia/run_20260921_204608_test_by_environment.csv) | `a35eed7ed4d87648dce66171c53ed20a6fabd3f82118817d03d267b1a2bd0f30` | ficha del run |
| métricas por `source_id` | [`test_by_source.csv`](../resultados/evidencia/run_20260921_204608_test_by_source.csv) | `d744894c8b57c451144fbfdbcf34a63d9c270bed7e8bbe7e882237f49af97e7e` | ficha del run |

## Benchmark source-grouped `20260921_180112`

| Afirmación verificable | Artefacto | Hash | Sección usuaria |
|---|---|---|---|
| 12/60 épocas, mejor época 4, val Macro-F1 0.266927, test Macro-F1 0.400665, accuracy 0.585705 | [`summary curado`](../resultados/evidencia/run_20260921_180112_summary.json) | `c20b87f5429e080448301950060202ba5c584e83084fdc82d87a539010350c9b` | ficha source-grouped e historia |
| train Macro-F1 final 0.966368 | [`train_history.csv`](../resultados/evidencia/run_20260921_180112_train_history.csv), época 12 | `09860e69e7b6c7cf26b38b62fb0b5715db739eda1e3ff0b3ccef2da7acadfdfe` | ficha source-grouped e historia |
| checkpoint histórico | summary original remoto | `fc2b57f388b2546846fd43039b483e3dbc1c51a2629a5b0b965d48f08de8d713` | ficha source-grouped |
| materialización source-grouped histórica | summary contractual | split lock: `ac9c441ff949c7d863d725e59453f0eddff33f385378e5677c78b796f011b95b` | registro experimental |

## Resultados históricos preservados

| Afirmación | Artefacto | Hash disponible | Estado/uso |
|---|---|---|---|
| Lite0 de la etapa 2 `20260812_221429`: Macro-F1 0.946789, accuracy 0.979063; int8 0.9474 | [`manifiesto_corridas.csv`](../resultados/evidencia/manifiesto_corridas.csv) y [evidencia de la etapa 2](../resultados/evidencia/etapa2_lite0_20260812_221429/summary.json) | `best.pth`: `298b73cab4a509788902c5b3165ff314b0c14d056e4221eab7800f06ef603193`; int8: `0a77ad7937f73e24b795288c68741d14dd00226ffe546d5ddf5b99125fbef24d` | HISTÓRICO; materialización de 33 433 muestras |
| ensamble: Macro-F1 0.956657, accuracy 0.982851 | [`ensamble_resumen.json`](../resultados/evidencia/ensamble_resumen.json) | `dcafbb4c21e2d69646b8a04cd1bbf425474b615f5eb6b44d594abd469b6f90c2` | HISTÓRICO, test estratificado |
| CV estratificada actual 0.931145 ± 0.009872 y CV agrupada actual 0.602575 ± 0.123966 | [`kfold_2x2_comparacion.csv`](../resultados/evidencia/kfold_2x2_comparacion.csv) | `0990f734edcf6b4aa5d924f8509df11d4a1a6b83761f477e0dd18b60ef60dc95` | HISTÓRICO; CV de cinco pliegues, no LOSO |
| LOSO histórico por clase/fuente | [`loso_gate_3seeds.csv`](../provenance/evidencia/loso_gate_3seeds.csv) y archivos vecinos | `24b6445058c25a62c8f717c059570b780b96988f80786074a5295bf9bd3ff72b` | HISTÓRICO exploratorio; no resumir como 0.6026 |
| corpus segmentado `20260907_163546`: Macro-F1 0.719082, accuracy 0.890329 | manifiesto de corridas | summary y checkpoint no versionados | EXPERIMENTAL |

## Reglas de uso

- Las cifras históricas se marcan como tales y no se mezclan con las del modelo oficial.
- Las métricas por ambiente o fuente indican su soporte y las clases evaluables.
- `0.6026 ± 0.1240` significa CV agrupada por fuente (Macro-F1 sobre las clases presentes en cada pliegue; ± es la semiamplitud del IC95; el Macro-F1 convencional es 0.4359), no LOSO ni el holdout source-grouped actual.
- El resultado agrupando N/P/K responde a otra taxonomía y se reporta separado del Macro-F1 de nueve clases.
- El multi-seed baseline/LOSO de tres semillas está en su sección. El estudio pareado baseline/HPO tiene 2 de 10 entrenamientos de validación. La validación cruzada no se repitió sobre la materialización vigente.
- El ensamble histórico se midió bajo LOSO multicrop con tres semillas: resultado mixto frente a B0 y peor ECE.

## HPO formal Lite0

Study `efficientnet_lite0_seed42_hpo_v1`: 25 intentos, ganador trial 0, validation Macro-F1 0.957292225; baseline 0.956086266; delta +0.120596 pp.

[Registro con storage, trials, parámetros, lock, checkpoint, figuras y test final](../reproducibilidad/evidencia/hpo_lite0_seed42/hpo_report.md). Base y pesos completos en `corn-outputs:/hpo/efficientnet_lite0/efficientnet_lite0_seed42_hpo_v1/`.

Resultado final: 8 COMPLETE, 15 PRUNED y 2 FAIL. Test único: Macro-F1 0.9431250727951999, accuracy 0.9752741774675973; no supera el Macro-F1 test del baseline (0.9480021439111501). [Interpretación y límites](/es/experimentos/hpo).

| Evidencia del cierre | SHA-256 | Alcance |
|---|---|---|
| [Summary original del baseline](../reproducibilidad/evidencia/hpo_baseline_summary.json) | `20bb0945574913ffa8c736fecbcf25d932e6b87ce3e7a7c59d98439558432346` | Comparación descriptiva de validación/test; no cambia el ganador |

`scripts/pipeline/report_hpo.py` valida la concordancia SQLite/CSV, la enmienda 60 → 25, el código archivado, los splits, el ganador por validación, el hash del checkpoint y los siete archivos de test. El marcador final registra una única evaluación de test posterior al lock.

## Multi-seed pareado baseline/HPO (2 de 10 entrenamientos)

[Protocolo y estado](../experimentos/multiseed.md). Diseño solo de validación con semillas 42, 123, 2026, 3407 y 7777 para baseline y HPO trial 0; el 2026-10-19 se completó la semilla 42 de cada configuración. No hay inferencia sobre el test.

| Artefacto | SHA-256 | Alcance |
|---|---|---|
| [multiseed_results.csv](../reproducibilidad/evidencia/multiseed_baseline_vs_hpo/multiseed_results.csv) | `7e3cb23126189bf07e0426e4c1f11c498990f05c7de1a12d87f33f4dca9f0a95` | Dos runs COMPLETE y ocho sin ejecutar |
| [multiseed_summary.json](../reproducibilidad/evidencia/multiseed_baseline_vs_hpo/multiseed_summary.json) | `8892a7b7806e73926dc584288cda91b0e6eaa3f1f95a28fa8d9e411bf2b0afd8` | Estadísticos con n = 1 |
| [protocol.json](../reproducibilidad/evidencia/multiseed_baseline_vs_hpo/protocol.json) | `b654609f33bdbda65099c26d3b8475edd061d3e93a49fc2316c6fe4639ddd081` | Configuración, fuentes y hashes |
| [preflight.json](../reproducibilidad/evidencia/multiseed_baseline_vs_hpo/preflight.json) | `347a19c5ada41cf99d615d8919303184659b34a15d1d347d4bc463688df58b64` | Verificación de splits |
| [report_hashes.json](../reproducibilidad/evidencia/multiseed_baseline_vs_hpo/report_hashes.json) | `1319445679b55ffc3a0d3544bf4c9a75980508b87190882f684faade31263473` | Hashes de los reportes |
| [COMPLETE baseline `20261019_203033`](../reproducibilidad/evidencia/multiseed_baseline_vs_hpo/complete_baseline_20261019_203033.json) | `74e540b57d7925b18858888dc468671fe185ab799063f553be3ceae01430250c` | Val Macro-F1 0.956086; `best.pth` `860180f3…` idéntico al oficial |
| [COMPLETE HPO trial 0 `20261019_211317`](../reproducibilidad/evidencia/multiseed_baseline_vs_hpo/complete_hpo_trial_0_20261019_211317.json) | `2fa43a41c00c0b9e8eb191f15df74187bb5bf2bfcd40cdc72b2984797136c230` | Val Macro-F1 0.957292; `best.pth` `00ff9aa5…` |

Las sumas SHA-256 de los paquetes de cada run están en `sha256sums_baseline` y `sha256sums_hpo_trial_0` de la misma carpeta.

## Multi-seed baseline/LOSO

[Resumen](../reproducibilidad/evidencia/multiseed_source_stability/multiseed_summary.json). Nueve resultados: tres de `seed=42` reutilizados y seis entrenamientos nuevos (semillas 123 y 2026 por escenario). El primer intento de `maize-diseases/seed_123` se interrumpió, se conserva aparte y no entra en ningún agregado.

Cada `selection.lock.json` nuevo precede a su evaluación final, que se ejecuta una vez (`evaluation_count=1`). Los dos LOSO de la semilla 42 tienen locks previos al holdout; el baseline de la semilla 42 no tiene un lock previo al test. El [registro por evaluación](../reproducibilidad/evidencia/multiseed_source_stability/evaluation_registry.csv) guarda timestamps, IDs y hashes.

| Afirmación | Evidencia | SHA-256 |
|---|---|---|
| Baseline estándar: test Macro-F1 0.943489 ± 0.005158; accuracy 0.976205 ± 0.002722; n=3, SD muestral | [baseline_multiseed.csv](../reproducibilidad/evidencia/multiseed_source_stability/baseline_multiseed.csv) | `04f90fc453775aa74d7762d19e34c67801c7e5671d4ee6b9657e02006aad71ae` |
| LOSO maize-diseases: Macro-F1 0.993264 ± 0.000600; multicrop-disease-maiz: 0.901906 ± 0.004422; n=3 | [loso_multiseed.csv](../reproducibilidad/evidencia/multiseed_source_stability/loso_multiseed.csv) | `4b8c3ffbd360d8e01917138eec7f8f384ba90205215af15984cc61e2bfc79350` |
| F1 por clase, incluida lethal_necrosis en multicrop: 0.845444 ± 0.005487 | [loso_class_multiseed.csv](../reproducibilidad/evidencia/multiseed_source_stability/loso_class_multiseed.csv) | `4749a7943edf61a6298762942d7e9327c2d87eda1765d0a65ef9bb29bb783815` |
| Media, mediana, SD ddof=1, rango y comparaciones descriptivas | [multiseed_summary.json](../reproducibilidad/evidencia/multiseed_source_stability/multiseed_summary.json) | `2a259cb6a92add4bc2c187987001c7d70106b5dd047bd1b1f9b3bd64efe5515d` |
| Orden lock→evaluación, conteo de las seis runs nuevas y procedencia | [evaluation_registry.csv](../reproducibilidad/evidencia/multiseed_source_stability/evaluation_registry.csv) | `00d31902d04945a885d2f437f9bbba0ee148bffacde455d8bcbe327603318ad8` |
| Rutas y hashes de los nueve checkpoints, manifests, predicciones y locks | [manifest_references.json](../reproducibilidad/evidencia/multiseed_source_stability/manifest_references.json) | `eca316fc57c22384ebf2bc0f012f1c9d9a5fae65ed72c4198c5173dec621d8b5` |

Los locks de manifests son `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` (estándar), `6486473932ae2937ff5a97881953d6ffbfac3725aa81170ef2b20a6bb49cbc07` (maize-diseases) y `d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b` (multicrop-disease-maiz). El [preflight de integridad](../reproducibilidad/evidencia/multiseed_source_stability/integrity_preflight.json) registró cero solapamientos por `sample_id`, SHA de archivo y SHA de píxel en cada escenario. Esto descarta fuga exacta en la materialización vigente, no casi duplicados ni atajos de fuente.

## Ensamble histórico bajo LOSO multicrop

El [resumen](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_summary.json) registra la regla del ensamble (promedio de tres softmax con pesos iguales), nueve locks de componente previos a la evaluación (tres Lite0 existentes y seis nuevos), tres evaluaciones únicas, seis entrenamientos nuevos y tres Lite0 reutilizados. Un intento B0/semilla 42 interrumpido se conserva y no entra al resultado.

| Afirmación | Evidencia | SHA-256 |
|---|---|---|
| Tres deltas pareados de Macro-F1 ensamble − B0: +0,012737, −0,003216, +0,003883; media +0,004468 ± 0,007993, n=3, SD muestral | [resultados por seed](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_results.csv) y [resumen](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_summary.json) | CSV `5a3d12f1b679a7c304f493087c57670cb00c0b70ee0210759047da474f55dbba`; JSON `be4320c64c9349a5abf224de244cf9bdf5827b39ac6daf315524bf7e582e8c7c` |
| Componentes y métricas globales, incluido ECE medio B0 0,044573 frente a ensamble 0,089088 | [componentes](../reproducibilidad/evidencia/ensemble_loso/ensemble_component_results.csv) | `bf89b7d5b3c5ecdc27c3fed1f33e833f8943c75dc53d105715311c1ef1688d41` |
| Precision, recall, F1 y soporte por clase/modelo/seed; LN recall ensamble 0,802538 ± 0,010260 | [clases](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_class_metrics.csv) | `526db4ac4d8161917fe3e6a5b197eb8161c0b5ca864a92fde48879c156866d1d` |
| Timestamps lock→guard→resultado, conteo único y hashes de predicciones | [registro](../reproducibilidad/evidencia/ensemble_loso/ensemble_evaluation_registry.csv) | `7beb03b4e5c06e78502f348f4b549927ed517bd6695d31332c6f4d0b7f67b8da` |
| Nueve rutas de checkpoints y sus SHA-256, manifest LOSO y materialización congelada | [referencias](../reproducibilidad/evidencia/ensemble_loso/manifest_checkpoint_references.json) | `45100d2b766be46a95f7a6e229031a35d6e5de509f126874215ff70c884c1c28` |

Los CSV completos de predicción, checkpoints, locks y guards permanecen en `corn-outputs:/ensemble_loso/multicrop-disease-maiz/`; la copia local se ignora en Git bajo `outputs/ensemble_loso/multicrop-disease-maiz/`. Dos semillas favorecen al ensamble y una a B0. Los deltas del test estándar histórico y del LOSO actual no son comparables porque cambian la materialización y las clases presentes.

## Síntesis y respaldo de modelos

La [síntesis experimental](resultados-por-protocolo.md) reúne los resultados por protocolo. El [índice de respaldo](evidencia/model_package/model_index.csv) enumera 18 checkpoints `best.pth` cotejados con el SHA-256 de su evidencia: los 16 del inventario principal y dos B0/ShuffleNet históricos. La [comparación por run](evidencia/model_package/model_comparison_results.csv) conserva test, LOSO, ECE y clases únicamente cuando existe registro. El archivo de checksums `evidencia/model_package/model_checksums.sha256` permite verificar los modelos al reunir las carpetas del respaldo.

| Artefacto | SHA-256 | Alcance |
|---|---|---|
| [Síntesis](resultados-por-protocolo.md) | `c5606ebe36f1f6cbe5eb5bc1f19387b9234e20729bc943773e5b1d4594301baf` | Protocolos, resultados y límites |
| [Guía del paquete](paquete-de-modelos.md) | `1bad6189a660ac2762f8b280f8dd055c4963867a8e7d4e84c49a43d0e0f49842` | Contenido, carga, integridad y distribución |
| [model_index.csv](evidencia/model_package/model_index.csv) | `77454998beb78524d495893fa25fa2f2918c77129fc112445094948280fa00b4` | 18 checkpoints verificados, rutas, hashes y tamaños |
| [model_comparison_results.csv](evidencia/model_package/model_comparison_results.csv) | `67f3359868b46d5fe68dac217ccaffae006ab4510ccfcd4e3f771eda297c8fbb` | 18 runs y cuatro resultados de ensamble sin checkpoint único |
| `model_checksums.sha256` | `77940bbbfd9dd8a5a37ee2a65fec618ed28187b7debb45e95370684c4149bb1f` | Comprobación local de los 18 `best.pth` |

El respaldo físico se distribuyó como ocho archivos `.tar.gz` (uno por carpeta `DoctorMaiz_*`); el `sha256sum -c` se ejecuta desde una raíz común que contiene las siete carpetas de checkpoints. Los checkpoints no se versionan en Git.

| Artefacto | SHA-256 | Alcance |
|---|---|---|
| [failed_runs_manifest.json](evidencia/model_package/failed_runs_manifest.json) | `aaab278ad44978252aa32a5a52c5482a515bad3cffa71963477842390983273e` | Dos intentos interrumpidos excluidos del respaldo |
| [evidence_references.json](evidencia/model_package/evidence_references.json) | `69c69d2d7ab614f43d31d4f204f1e680211ac364c108de577728ea5e6e550184` | Hashes de los documentos fuente |

## Exportación móvil del modelo oficial

Run `20260921_204608`, checkpoint `860180f3…`. Los archivos pesados (`model.onnx`, `model.tflite`, `model_int8.tflite`, `ood_stats.json`) no se versionan; `ood_stats_metadata.json` conserva los campos escalares y el SHA-256 del archivo completo.

| Afirmación | Artefacto | SHA-256 |
|---|---|---|
| ONNX y TFLite FP32 con paridad aprobada (max \|Δprob\| 3e-6 y 4e-7) | [export_summary.json](../reproducibilidad/evidencia/export_movil_20260921_204608/export_summary.json) | `8cf93465d6dd17188264e7491bf462ca6fd136f9dbe67005b18dd648ae81965f` |
| TFLite int8 `bb31467a…3c56b`, 3 743 728 B; paridad de muestra de 30 imágenes no aprobada (max \|Δprob\| 0.198 > 0.15) | [export_summary_int8.json](../reproducibilidad/evidencia/export_movil_20260921_204608/export_summary_int8.json) | `850a0e6c2476b96a2f7c05aea4d0f2d8c6ef6bf83f3d1e28788c7a86e8d99b62` |
| int8 sobre el test completo (5 015): Macro-F1 0.9489, accuracy 97.87 % | [eval_tflite_int8.json](../reproducibilidad/evidencia/export_movil_20260921_204608/eval_tflite_int8.json) | `f1430c9bbfff8a0ff9b2bb4c2133455269c1cfc9668c7f68441408222e2b86e2` |
| Orden de las nueve clases | [labels.json](../reproducibilidad/evidencia/export_movil_20260921_204608/labels.json) | `01eedc9c93ff7cd4ee195062534e04ac1101378b43a7d0bff785ecfc180f9642` |
| OOD: schema 4, PCA 182 (99.01 % de varianza), umbral 106.37 (percentil 95 de validación) | [ood_stats_metadata.json](../reproducibilidad/evidencia/export_movil_20260921_204608/ood_stats_metadata.json) | `f9b1c4a4c22e3c68fd4d005d656ddb1c5332beeaa5f25306fc2b2fdf083a47fd`; `ood_stats.json` completo: `5eb6cdd1ab2298b99f37fe563e62237685cb58870eaa7fd509b1839c3d91f6e1` |

## Etapa 2 — Lite0 `20260812_221429`

Evidencia en [`resultados/evidencia/etapa2_lite0_20260812_221429/`](../resultados/evidencia/etapa2_lite0_20260812_221429/summary.json): resumen del run, reporte de clasificación, matriz de confusión y calibración de test, evaluación int8 (Macro-F1 0.9474), equidad por fuente y las cuatro celdas del k-fold 2x2 (`kfold_summary.json` y `kfold_metrics.csv`). El corpus de esa etapa tiene 33 433 muestras (23 403 / 5 015 / 5 015) y comparte 2 539 imágenes de test con el split vigente; sobre ellas, McNemar da 23 frente a 22 discordancias (p = 1.00). Hashes de referencia:

| Artefacto | SHA-256 |
|---|---|
| `summary.json` | `0d8a5d91708b5ad10869ad3159df7772e694c82697f16fdf63aa487a9d283e1c` |
| `export/export_summary_int8.json` | `7a3016c3324e903d8e97fb2db7bb3e68768037566fff64c3ae65575023383225` |
| `export/eval_tflite_int8.json` | `942c1f35c4404c1e08eb104fa079da8dc8864a951957889d38c3dc77344f3fac` |
| `fairness_por_fuente/fairness_metrics.json` | `1246f3ab71dc808d4c10c7897e22163114789c47122f436d465113d1a236cfc8` |
| `predictions.csv` (predicción por imagen del test) | `1fe736982c49bbdd509ae7aa121eb772ed88579e707da1f62d71044a51b70e5b` |

## Equidad entre laboratorio y campo

DIR calculado sobre las tres clases presentes en ambos entornos con `scripts/experiments/equidad_dir.py`: modelo oficial 0.9636 (Macro-F1) y 0.9956 (accuracy); etapa 2, 0.9604 y 0.9901. Incluye el F1 por clase y entorno.

| Artefacto | SHA-256 |
|---|---|
| [`equidad_dir_clases_compartidas.json`](../resultados/evidencia/equidad_dir_clases_compartidas.json) | `e10c22711659c9df322a943cbed070b6c1016255f56ebf055146e03498ac9f27` |
