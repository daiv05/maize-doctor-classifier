# Respaldo de modelos DoctorMaiz — guía para el equipo

Este paquete contiene checkpoints **ya entrenados** y una copia de los resultados registrados hasta el 2 de octubre de 2026. No establece un modelo elegido. La comparación y la decisión posterior corresponden al equipo.

`MODEL_INDEX.csv` identifica cada `best.pth`: arquitectura, protocolo, seed, holdout, época, Macro-F1, ruta original, ruta dentro de este paquete, tamaño y SHA-256. Las 18 filas `VERIFIED` fueron cotejadas byte a byte con sus hashes de evidencia; no se incluyeron intentos interrumpidos. `MODEL_COMPARISON_RESULTS.csv` añade métricas de test/LOSO, ECE y clases cuando están registradas. `RESULTS_SUMMARY.md` explica qué preguntas responde cada experimento, sus cifras y límites. Los campos `N/A` no deben completarse suponiendo que otro protocolo produjo la misma medida.

## Carpetas y archivos para compartir

| Grupo | Contenido | Archivo de distribución |
|---|---|---|
| `baseline_lite0/` | Lite0 estándar, seeds 42/123/2026 | `DoctorMaiz_Baseline_Lite0.tar.gz` |
| `hpo_lite0/` | Lite0 HPO, trial 0 | `DoctorMaiz_HPO_Lite0.tar.gz` |
| `loso_maize_diseases/` | Lite0 sin `maize-diseases` en entrenamiento/selección, tres seeds | `DoctorMaiz_LOSO_MaizeDiseases.tar.gz` |
| `loso_multicrop/lite0/` | Lite0 sin `multicrop-disease-maiz`, tres seeds | `DoctorMaiz_LOSO_Multicrop_Lite0.tar.gz` |
| `loso_multicrop/b0/` | B0 LOSO multicrop, tres seeds | `DoctorMaiz_LOSO_Multicrop_B0.tar.gz` |
| `loso_multicrop/shufflenet/` | ShuffleNet-V2-x1.0 LOSO multicrop, tres seeds | `DoctorMaiz_LOSO_Multicrop_ShuffleNet.tar.gz` |
| `historical_standard/` | B0 y ShuffleNet históricos del test estratificado original | `DoctorMaiz_Historicos_B0_ShuffleNet.tar.gz` |
| Archivos de la raíz y `metadata/` | Índices, checksums, este README, resultados, procedencia, intentos excluidos | `DoctorMaiz_Modelos_Metadata_Resultados.tar.gz` |

La carpeta de cada run tiene `best.pth`, `summary.json`, `config_snapshot.json` extraído del summary y `checkpoint_reference.json`. Cuando existe, también tiene `selection.lock.json` y métricas finales por clase. El baseline seed 42 es histórico y **no** tiene un lock pre-test documentado; no se inventó. El trial HPO lleva su `HPO_SELECTION_LOCK.json`. En los B0/ShuffleNet multicrop se incluye el lock conjunto del ensamble. La configuración de cada run está en su summary/snapshot, no en un `dataset.yaml` actual que pudiera haber cambiado.

## Cómo comparar y cargar

1. Leer `RESULTS_SUMMARY.md` y filtrar `MODEL_COMPARISON_RESULTS.csv` por `experiment` y `evaluation_split`. El test estándar y un holdout LOSO no se comparan como si fueran el mismo test.
2. Para un checkpoint concreto, buscar su `model_id` en `MODEL_INDEX.csv` y abrir el `local_path` relativo a esta carpeta. El archivo a cargar es siempre `best.pth`, no `last.pth`.
3. Leer `summary.json` y `config_snapshot.json` antes de cargarlo: contienen arquitectura, `class_to_idx`, preprocessing y fingerprint del split cuando esa run los registró. Usar el código del proyecto compatible con la run y el cargador contractual; este respaldo no incluye dataset, entorno virtual ni ejecutables autónomos.
4. Verificar integridad con `sha256sum -c MODEL_CHECKSUMS.sha256` desde una raíz común que contenga las siete carpetas `DoctorMaiz_*` de checkpoints junto a `DoctorMaiz_Modelos_Metadata_Resultados`. Ejecutado desde una sola carpeta falla por rutas inexistentes. La suma de cada checkpoint también aparece en el índice.

El ensamble histórico de tres miembros (Lite0 `20260812_221429`, B0 `20260910_170120` y ShuffleNet `20260910_184521`) se documenta por sus métricas. Es reconstruible: los B0 y ShuffleNet históricos están en este respaldo y el Lite0 de agosto está en la descarga de artefactos de la etapa 2. El ensamble LOSO multicrop no tiene `best.pth` único: se forma con los tres checkpoints de la misma seed y promedio uniforme de softmax, sin reajuste de pesos. Las predicciones completas y guards de evaluación permanecen en los artefactos originales; este respaldo prioriza modelos y metadatos pequeños.

`metadata/evidence_references.json` enumera los hashes de documentos fuente y `metadata/FAILED_RUNS_MANIFEST.json` identifica los dos intentos interrumpidos excluidos; ambos están versionados en `docs/es/reproducibilidad/evidencia/model_package/`. El paquete se distribuyó como ocho archivos `.tar.gz` (uno por carpeta de la tabla anterior). `bash CREATE_TARS.sh` los genera, los prueba con `tar -tzf` y registra `TAR_CHECKSUMS.sha256`.
