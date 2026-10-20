# Respaldo de modelos DoctorMaiz — guía para el equipo

Este paquete contiene checkpoints **ya entrenados** y una copia de los resultados registrados hasta el 2 de octubre de 2026. No establece un modelo elegido. La comparación y la decisión posterior corresponden al equipo.

`MODEL_INDEX.csv` identifica cada `best.pth`: arquitectura, protocolo, seed, holdout, época, Macro-F1, ruta original, ruta dentro de este paquete, tamaño y SHA-256. Las 18 filas `VERIFIED` fueron cotejadas byte a byte con sus hashes de evidencia; no se incluyeron intentos interrumpidos. `MODEL_COMPARISON_RESULTS.csv` añade métricas de test/LOSO, ECE y clases cuando están registradas. `RESULTS_SUMMARY.md` explica qué preguntas responde cada experimento, sus cifras y límites. Los campos `N/A` no deben completarse suponiendo que otro protocolo produjo la misma medida.

## Carpetas y archivos para compartir

| Grupo | Contenido | Archivo de distribución previsto |
|---|---|---|
| `baseline_lite0/` | Lite0 estándar, seeds 42/123/2026 | `DoctorMaiz_Baseline_Lite0.rar` |
| `hpo_lite0/` | Lite0 HPO, trial 0 | `DoctorMaiz_HPO_Lite0.rar` |
| `loso_maize_diseases/` | Lite0 sin `maize-diseases` en entrenamiento/selección, tres seeds | `DoctorMaiz_LOSO_MaizeDiseases.rar` |
| `loso_multicrop/lite0/` | Lite0 sin `multicrop-disease-maiz`, tres seeds | `DoctorMaiz_LOSO_Multicrop_Lite0.rar` |
| `loso_multicrop/b0/` | B0 LOSO multicrop, tres seeds | `DoctorMaiz_LOSO_Multicrop_B0.rar` |
| `loso_multicrop/shufflenet/` | ShuffleNet-V2-x1.0 LOSO multicrop, tres seeds | `DoctorMaiz_LOSO_Multicrop_ShuffleNet.rar` |
| `historical_standard/` | B0 y ShuffleNet históricos del test estratificado original | `DoctorMaiz_Historicos_B0_ShuffleNet.rar` |
| Archivos de la raíz y `metadata/` | Índices, checksums, este README, resultados, procedencia, intentos excluidos | `DoctorMaiz_Modelos_Metadata_Resultados.rar` |

La carpeta de cada run tiene `best.pth`, `summary.json`, `config_snapshot.json` extraído del summary y `checkpoint_reference.json`. Cuando existe, también tiene `selection.lock.json` y métricas finales por clase. El baseline seed 42 es histórico y **no** tiene un lock pre-test documentado; no se inventó. El trial HPO lleva su `HPO_SELECTION_LOCK.json`. En los B0/ShuffleNet multicrop se incluye el lock conjunto del ensamble. La configuración de cada run está en su summary/snapshot, no en un `dataset.yaml` actual que pudiera haber cambiado.

## Cómo comparar y cargar

1. Leer `RESULTS_SUMMARY.md` y filtrar `MODEL_COMPARISON_RESULTS.csv` por `experiment` y `evaluation_split`. El test estándar y un holdout LOSO no se comparan como si fueran el mismo test.
2. Para un checkpoint concreto, buscar su `model_id` en `MODEL_INDEX.csv` y abrir el `local_path` relativo a esta carpeta. El archivo a cargar es siempre `best.pth`, no `last.pth`.
3. Leer `summary.json` y `config_snapshot.json` antes de cargarlo: contienen arquitectura, `class_to_idx`, preprocessing y fingerprint del split cuando esa run los registró. Usar el código del proyecto compatible con la run y el cargador contractual; este respaldo no incluye dataset, entorno virtual ni ejecutables autónomos.
4. Verificar integridad desde la raíz del paquete: `sha256sum -c MODEL_CHECKSUMS.sha256`. La suma de cada checkpoint también aparece en el índice.

El ensamble histórico de tres miembros se documenta por sus métricas, pero **no** se presenta como paquete totalmente reconstruible: falta el checkpoint Lite0 desplegado de agosto. Los B0 y ShuffleNet históricos sí están respaldados para consulta. El ensamble LOSO multicrop no tiene `best.pth` único: se forma con los tres checkpoints de la misma seed y promedio uniforme de softmax, sin reajuste de pesos. Las predicciones completas y guards de evaluación permanecen en los artefactos originales; este respaldo prioriza modelos y metadatos pequeños.

`metadata/evidence_references.json` enumera los hashes de documentos fuente. `metadata/FAILED_RUNS_MANIFEST.json` identifica los dos intentos interrumpidos excluidos. Los archivos `.rar`, cuando exista la utilidad `rar`, se crean con `bash CREATE_RARS.sh`; el script prueba cada RAR con `rar t` y genera `RAR_CHECKSUMS.sha256` en la carpeta de archivos. Si `rar` no está disponible, `bash CREATE_TARS.sh` genera ocho `.tar.gz` reales, los prueba y registra `TAR_CHECKSUMS.sha256`. No se debe renombrar un `.tar.gz` o `.zip` como `.rar`.
