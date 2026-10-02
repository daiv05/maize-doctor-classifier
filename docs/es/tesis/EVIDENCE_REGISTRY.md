# Registro de evidencias para tesis

Este registro responde **de dónde sale cada cifra**. Una afirmación puede usarse como resultado cuantitativo solo si apunta a un artefacto versionado o a una ruta remota estable. Los hashes son SHA-256; `PENDIENTE` significa que no se recuperó evidencia suficiente y no autoriza completar el dato por inferencia.

## Corpus y particiones vigentes

| Afirmación verificable | Artefacto | Run/protocolo | Hash disponible | Documento que la usa |
|---|---|---|---|---|
| 33 437 muestras descubiertas y válidas; 8 exclusiones; 33 429 elegibles | [`split_protocols_summary.json`](../reproducibilidad/evidencia/split_protocols_summary.json) y auditoría remota | preparación vigente | resumen versionado: `9e24dc42…656f0a`; fuente en `corn-outputs:/splits/.../preparation_audit.json` | pipeline de datos, estado actual, evolución |
| `seed_42` = 23 400 / 5 014 / 5 015, nueve clases por split | mismo resumen + lock exacto | `seed_42` | [`manifest.lock.json`](../reproducibilidad/evidencia/seed_42_manifest_lock.json): `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` | metodología, README, baseline actual |
| Cero solapamiento de `sample_id`, SHA-256 y `effective_group_id` | `preparation_audit.json` remoto, extracto en resumen | `seed_42` | incluido en el lock anterior | pipeline de datos y protocolos |
| 33 429 elegibles, cero duplicados por pixel_sha256, cero cross-split, 0 % de test afectado; 4 grupos/8 archivos ya excluidos | [resumen e inventario](../reproducibilidad/evidencia/pixel_duplicate_audit/PIXEL_DUPLICATE_AUDIT.md) | `seed_42` vigente | master: `64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163`; lock: `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` | metodología y estado actual |
| 11 fuentes, distribución de 33 429 muestras, V de Cramér 0.5030 y métricas baseline/HPO por fuente | [paquete de análisis por fuente](../reproducibilidad/evidencia/source_analysis/SOURCE_ANALYSIS.md), [resumen y hashes](../reproducibilidad/evidencia/source_analysis/source_analysis_summary.json) | `seed_42`; validation comparativa y test descriptivo post-hoc | hashes SHA-256 de manifests y cuatro CSV de predicciones registrados en el resumen | estado actual y análisis de procedencia; no selecciona modelo |
| LOSO baseline: `maize-diseases` 6 989, F1 0.993124; `multicrop-disease-maiz` 5 816, F1 0.905490; locks previos al holdout | [resultado y límites](../reproducibilidad/evidencia/loso_baseline/LOSO_RESULTS.md), [CSV](../reproducibilidad/evidencia/loso_baseline/loso_results.csv), [resumen y timestamps](../reproducibilidad/evidencia/loso_baseline/loso_summary.json) | `loso_baseline_seed42_v1`, dos runs seed 42 | checkpoints `b16b1ef8…dcc` y `bd778a28…517`; hashes completos de splits, locks y runs en [protocolo](../reproducibilidad/evidencia/loso_baseline/LOSO_REPORT.md) y CSV | comparación descriptiva con validation histórica; no causalidad ni significancia con una semilla |
| Hash de dataset.yaml distinto del registrado en el lock; causa textual no recuperada | [informe de drift](../reproducibilidad/evidencia/pixel_duplicate_audit/dataset_yaml_drift_report.md) | configuración de `seed_42` | lock: `8f0e560f…`; actual: `8e3f5a4f…` | límite de reproducción de la preparación |
| PHash no ejecutado; solapamiento perceptual desconocido | `preparation_audit.json`: `deduplicate_perceptual=false`, conteo `null` | ambos protocolos actuales | rutas declaradas dentro del resumen | metodología y limitaciones |
| `seed_42_source_grouped` = 16 554 / 7 809 / 9 066; 5/3/3 fuentes | resumen de protocolos | benchmark actual | fuente remota declarada en JSON | protocolos y ficha cross-source |

## Baseline de desarrollo `20260921_204608`

Ruta original: `corn-outputs:/main/efficientnet_lite0/20260921_204608/`.

| Afirmación verificable | Artefacto versionado | Hash del artefacto | Sección usuaria |
|---|---|---|---|
| 47/60 épocas; mejor época 39; val Macro-F1 0.956086; test Macro-F1 0.948002; accuracy 0.978066 | [`run_20260921_lite0_summary.json`](../resultados/evidencia/run_20260921_lite0_summary.json) | `20bb0945574913ffa8c736fecbcf25d932e6b87ce3e7a7c59d98439558432346` | ficha del run, registro experimental, README |
| checkpoint `best.pth` | mismo summary contractual | checkpoint: `860180f33bf1749ffee9f3cccb9569a47afd9ca0fcef52713e18f99f82a57859` | ficha del run |
| configuración efectiva | mismo summary contractual | config: `53cc091e505057f6751a7b6151f403e03f830fac3c0966df9ba51e2df2814e8b` | ficha del run y registro experimental |
| precision, recall, F1 y soporte de las nueve clases | [`classification_report.csv`](../resultados/evidencia/run_20260921_204608_test_classification_report.csv) | `9b18915ab63447f71f572f18f52e443d1b337c20baff33114f4e9b502d49378b` | ficha del run |
| confusiones exactas, incluidas GLS→NCLB 18 y K→N 15 | [`confusion_matrix.csv`](../resultados/evidencia/run_20260921_204608_test_confusion_matrix.csv) | `0dfdf12e180078d150f48bb96726538da224aa5a96e5b1d910596d2755e65bea` | análisis por clase/N-P-K |
| ECE 0.139656; confianza aciertos 0.844379 y errores 0.687191 | [`test_calibration.json`](../resultados/evidencia/run_20260921_204608_test_calibration.json) | `8ede2e9d839fdcc87763d1dd4c0a93d235af6b95d0b1f6b1c5cf83422b7cf15d` | calibración |
| 15 errores con confianza ≥ 0.90 | `predictions.csv` remoto; recuento auditado | `PENDIENTE — versionar predictions.csv o su consulta` | ficha del run; no usar sin esta salvedad |
| Macro-F1 N/P/K agrupado 0.978385 y accuracy 0.984048 | [`test_grouped_metrics.json`](../resultados/evidencia/run_20260921_204608_test_grouped_metrics.json) | `ca94652860fb7cc53c73fbef5aa7e65c21a560666420f00eeca53bc8f8b44e35` | análisis N/P/K; nunca como métrica principal |
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
| Lite0 desplegado `20260812_221429`: Macro-F1 0.946789, accuracy 0.979063 | [`manifiesto_corridas.csv`](../resultados/evidencia/manifiesto_corridas.csv) y summary remoto | `PENDIENTE — hash del checkpoint histórico no versionado aquí` | HISTÓRICO; materialización distinta |
| ensamble: Macro-F1 0.956657, accuracy 0.982851 | [`ensamble_resumen.json`](../resultados/evidencia/ensamble_resumen.json) | `dcafbb4c21e2d69646b8a04cd1bbf425474b615f5eb6b44d594abd469b6f90c2` | HISTÓRICO, test estratificado |
| CV estratificada actual 0.931145 ± 0.009872 y CV agrupada actual 0.602575 ± 0.123966 | [`kfold_2x2_comparacion.csv`](../resultados/evidencia/kfold_2x2_comparacion.csv) | `0990f734edcf6b4aa5d924f8509df11d4a1a6b83761f477e0dd18b60ef60dc95` | HISTÓRICO; CV de cinco pliegues, no LOSO |
| LOSO histórico por clase/fuente | [`loso_gate_3seeds.csv`](../provenance/evidencia/loso_gate_3seeds.csv) y archivos vecinos | `24b6445058c25a62c8f717c059570b780b96988f80786074a5295bf9bd3ff72b` | HISTÓRICO exploratorio; no resumir como 0.6026 |
| corpus segmentado `20260907_163546`: Macro-F1 0.719082, accuracy 0.890329 | manifiesto de corridas | `PENDIENTE — summary/checkpoint no versionados aquí` | EXPERIMENTAL |

## Reglas de uso

- Una cifra histórica no se convierte en vigente por aparecer en una página actual.
- Una métrica por ambiente o fuente debe indicar su soporte y las clases evaluables.
- `0.6026 ± 0.1240` significa **CV agrupada por fuente**, no LOSO ni el holdout source-grouped actual.
- El resultado agrupando N/P/K responde a otra taxonomía y se reporta separado del Macro-F1 de nueve clases.
- El multi-seed baseline/LOSO de tres semillas ya está cerrado más abajo. Siguen pendientes el estudio pareado baseline/HPO (0/10 runs), CV/LOSO final general, entrenamiento formal, exportación y dispositivo. El HPO de 25 intentos no completa esas fases.

<!-- hpo-lite0-seed42-completed -->

## HPO formal Lite0 — 2026-09-24

Study `efficientnet_lite0_seed42_hpo_v1`: 25 intentos, ganador trial 0, validation Macro-F1 0.957292225; baseline 0.956086266; delta +0.120596 pp.

[Registro con storage, trials, parámetros, lock, checkpoint, figuras y test final](../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md). Base y pesos completos en `corn-outputs:/hpo/efficientnet_lite0/efficientnet_lite0_seed42_hpo_v1/`.

Resultado final: **8 COMPLETE / 15 PRUNED / 2 FAIL**. Test único: Macro-F1
0.9431250727951999, accuracy 0.9752741774675973; no supera el Macro-F1 test del
baseline (0.9480021439111501). [Interpretación y límites](./HPO_BASELINE_COMPARISON.md).

| Evidencia añadida al cierre | SHA-256 | Alcance |
|---|---|---|
| [Summary original del baseline](../reproducibilidad/evidencia/hpo_baseline_summary.json) | `20bb0945574913ffa8c736fecbcf25d932e6b87ce3e7a7c59d98439558432346` | Comparación descriptiva de validación/test; no cambia el ganador |

La revisión comprobó concordancia SQLite/CSV, enmienda 60 → 25, código archivado,
splits inmutables, ganador por validación, hash del checkpoint y los siete archivos
de test. El marcador final registra una evaluación y su lock previo; no se volvió
a inferir sobre test durante la revisión/documentación.

## Multi-seed pareado — preparación 2026-09-24

[Protocolo, auditoría, calendario y comandos](../experimentos/multiseed.md).
Implementación validation-only; seeds 42/123/2026/3407/7777 para baseline y HPO
trial 0. Estado: **0/10 runs**, sin nuevas métricas experimentales.

Artefactos locales bajo `outputs/multiseed/efficientnet_lite0_baseline_vs_hpo/`:
`PROTOCOL.json`, `PREFLIGHT.json`, `source_snapshot.zip`, `REPORT_HASHES.json` y
`MULTISEED_SUMMARY.json`, `MULTISEED_RESULTS.csv`, `MULTISEED_REPORT.md`.
El reporte explicita pendientes; no importa resultados históricos como nuevos.
JUnit de 146 pruebas en `outputs/multiseed-checks/tests.xml`.
Este plan sigue separado del estudio baseline/LOSO ya cerrado y conserva 0/10 runs.
No se ejecutó inferencia del test real durante su preparación.

## Multi-seed baseline/LOSO — cerrado 2026-10-01

[Informe completo y límites](../reproducibilidad/evidencia/multiseed_source_stability/MULTISEED_SOURCE_STABILITY.md).
Nueve resultados: tres `seed=42` reutilizados y seis entrenamientos nuevos
(seeds 123 y 2026 por escenario), sin HPO, ensamble, CV ni entrenamiento
formal. El primer intento de `maize-diseases/seed_123` fue interrumpido,
se preservó fuera de `runs/` y no contribuye a ningún agregado; el segundo
terminó y es el resultado válido.

Los seis `selection.lock.json` nuevos preceden a cualquier test/holdout
nuevo. Seis guards registran `evaluation_count=1`, con predicciones
reconciliadas por `sample_id`, hash del checkpoint y resultados calculados
de nuevo desde el CSV. Los dos LOSO históricos tienen locks previos al
holdout; el baseline histórico no tiene un lock pre-test documentado y no se
le atribuyó uno retroactivamente. El
[registro por evaluación](../reproducibilidad/evidencia/multiseed_source_stability/evaluation_registry.csv)
guarda timestamps, IDs y hashes; los tres conteos históricos sin guard nuevo
quedan sin valor formal.

| Afirmación | Evidencia | SHA-256 |
|---|---|---|
| Baseline estándar: test Macro-F1 0.943489 ± 0.005158; accuracy 0.976205 ± 0.002722; n=3, SD muestral | [baseline_multiseed.csv](../reproducibilidad/evidencia/multiseed_source_stability/baseline_multiseed.csv) | `04f90fc453775aa74d7762d19e34c67801c7e5671d4ee6b9657e02006aad71ae` |
| LOSO maize-diseases: Macro-F1 0.993264 ± 0.000600; multicrop-disease-maiz: 0.901906 ± 0.004422; n=3 | [loso_multiseed.csv](../reproducibilidad/evidencia/multiseed_source_stability/loso_multiseed.csv) | `4b8c3ffbd360d8e01917138eec7f8f384ba90205215af15984cc61e2bfc79350` |
| F1 por clase, incluida lethal_necrosis en multicrop: 0.845444 ± 0.005487 | [loso_class_multiseed.csv](../reproducibilidad/evidencia/multiseed_source_stability/loso_class_multiseed.csv) | `4749a7943edf61a6298762942d7e9327c2d87eda1765d0a65ef9bb29bb783815` |
| Media, mediana, SD ddof=1, rango y comparaciones descriptivas | [multiseed_summary.json](../reproducibilidad/evidencia/multiseed_source_stability/multiseed_summary.json) | `2a259cb6a92add4bc2c187987001c7d70106b5dd047bd1b1f9b3bd64efe5515d` |
| Orden lock→evaluación, conteo de las seis runs nuevas y procedencia | [evaluation_registry.csv](../reproducibilidad/evidencia/multiseed_source_stability/evaluation_registry.csv) | `00d31902d04945a885d2f437f9bbba0ee148bffacde455d8bcbe327603318ad8` |
| Rutas y hashes de los nueve checkpoints, manifests, predicciones y locks | [manifest_references.json](../reproducibilidad/evidencia/multiseed_source_stability/manifest_references.json) | `eca316fc57c22384ebf2bc0f012f1c9d9a5fae65ed72c4198c5173dec621d8b5` |

Los locks de manifests son
`0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8`
(estándar),
`6486473932ae2937ff5a97881953d6ffbfac3725aa81170ef2b20a6bb49cbc07`
(maize-diseases) y
`d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b`
(multicrop-disease-maiz). El
[preflight de integridad](../reproducibilidad/evidencia/multiseed_source_stability/integrity_preflight.json)
registró cero solapamientos por `sample_id`, SHA de archivo y SHA de píxel
en cada escenario. Esto descarta fuga **exacta** en la materialización
vigente, no casi duplicados ni atajos de fuente. El resultado multicrop es
compatible con sensibilidad a la exclusión de esa fuente, sin atribución
causal.
