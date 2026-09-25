# Registro de procedencia de figuras

Una figura marcada `PENDIENTE` puede conservarse como ilustración histórica, pero no debe presentarse como evidencia cuantitativa reproducible hasta recuperar su generador y fuente exactos. Las rutas son relativas a `public/`.

## EDA

Todas estas figuras proceden de la exploración del corpus ampliado y aparecen en `docs/es/exploratory-data-analysis/`.

| Figura | Run/experimento | Generador | Artefacto fuente | Estado |
|---|---|---|---|---|
| `eda/eda_00_muestra_visual.png` | EDA agosto 2026 | `notebooks/01_eda.ipynb` | corpus `clean/` de esa fecha | HISTÓRICA |
| `eda/eda_01_distribucion_clases.png` | EDA agosto 2026 | mismo notebook | inventario del corpus histórico de 33 438 | HISTÓRICA |
| `eda/eda_02_lab_vs_real.png` | EDA agosto 2026 | mismo notebook | inventario por ambiente | HISTÓRICA |
| `eda/eda_02b_heatmap_fuente_clase.png` | EDA agosto 2026 | mismo notebook | inferencia histórica de fuentes | HISTÓRICA |
| `eda/eda_03_resoluciones.png` | EDA agosto 2026 | mismo notebook | metadatos de imágenes | HISTÓRICA |
| `eda/eda_03b_resolucion_por_clase.png` | EDA agosto 2026 | mismo notebook | metadatos de imágenes | HISTÓRICA |
| `eda/eda_04_calidad.png` | EDA agosto 2026 | mismo notebook | métricas de calidad del notebook | HISTÓRICA |
| `eda/eda_04b_calidad_boxplots.png` | EDA agosto 2026 | mismo notebook | métricas de calidad del notebook | HISTÓRICA |
| `eda/eda_04c_color_hsv.png` | EDA agosto 2026 | mismo notebook | estadísticas HSV | HISTÓRICA |
| `eda/eda_04d_hue_lab_vs_real.png` | EDA agosto 2026 | mismo notebook | estadísticas HSV/ambiente | HISTÓRICA |
| `eda/eda_05_duplicados.png` | limpieza histórica | mismo notebook | registro histórico de deduplicación | HISTÓRICA |
| `eda/eda_06_sesgos.png` | EDA agosto 2026 | mismo notebook | conteos por clase/ambiente | HISTÓRICA |

No deben reinterpretarse estas imágenes como gráficos de `master_manifest.csv` vigente: los conteos pertenecen a la instantánea anterior de 33 438 imágenes.

## Baselines

| Figura | Run/experimento | Generador | Artefacto fuente | Estado |
|---|---|---|---|---|
| `baselines/samples/aug_potasio_minority.png` | ejemplo de transforms baseline | `src/explainability/augmentation_preview.py` | imagen concreta/config exacta `PENDIENTE` | REVISAR |
| `baselines/samples/lime_common_rust_ok.png` | XAI baseline | pipeline LIME/Grad-CAM histórico | run e imagen exactos `PENDIENTE` | REVISAR |
| `baselines/samples/lime_potasio_error_nitrogeno.png` | XAI baseline | pipeline LIME/Grad-CAM histórico | run e imagen exactos `PENDIENTE` | REVISAR |
| `baselines/baseline_f1_por_clase.png` | comparación baseline histórica | generador exacto `PENDIENTE` | reportes históricos baseline | REVISAR |
| `baselines/baseline_confusion_b0.png` | B0 baseline histórico | generador exacto `PENDIENTE` | matriz/predicciones exactas `PENDIENTE` | REVISAR |
| `baselines/baseline_convergencia.png` | entrenamiento baseline histórico | generador exacto `PENDIENTE` | history exacto `PENDIENTE` | REVISAR |

## Resultados, optimización y ensamble

| Figura | Run/experimento | Generador | Artefacto fuente | Estado |
|---|---|---|---|---|
| `resultados/optimizacion_tres_arquitecturas.png` | HPO histórico | generador exacto `PENDIENTE` | JSON/CSV Optuna en `resultados/evidencia/` | REVISAR |
| `resultados/diferencias_en_sigmas.png` | multi-seed histórico | generador exacto `PENDIENTE` | summaries de semillas + manifiesto | REVISAR |
| `resultados/kfold_2x2.png` | CV 2×2 histórica | generador de la imagen `PENDIENTE`; `comparar_kfold.py` solo consolida tablas | `kfold_2x2_comparacion.csv` y pliegues | REVISAR |
| `ensemble/ensemble_comparison_bar.png` | ensamble histórico | generador exacto `PENDIENTE` | `ensamble_comparacion.csv` | REVISAR |
| `ensemble/confusion_matrix_ensemble.png` | ensamble histórico | `scripts/pipeline/evaluate_ensemble.py` | `ensamble_predicciones.csv`/resumen | HISTÓRICA REPRODUCIBLE EN PARTE |
| `training/training_convergence.png` | pipeline principal histórico | generador exacto `PENDIENTE` | train history exacto `PENDIENTE` | REVISAR |

## Equidad y explicabilidad

| Figura | Run/experimento | Generador | Artefacto fuente | Estado |
|---|---|---|---|---|
| `resultados/ablacion_control_nulo.png` | ablación histórica | `scripts/experiments/figura_ablacion.py` | `equidad_metricas.json` y predicciones fairness | HISTÓRICA |
| `resultados/equidad_por_fuente.png` | equidad por fuente histórica | generador exacto `PENDIENTE` | `equidad_por_fuente.csv` | REVISAR |
| `fairness/fairness_disparity.png` | auditoría por ambiente histórica | `scripts/experiments/figura_equidad_entorno.py` o `scripts/pipeline/evaluate_fairness.py`; ejecución exacta `PENDIENTE` | fairness JSON/CSV | REVISAR |
| `fairness/disaggregated_confusion_matrices.png` | auditoría fairness histórica | `scripts/pipeline/evaluate_fairness.py` | fairness predictions | HISTÓRICA |
| `fairness/gradcam_samples.png` | auditoría fairness histórica | `scripts/pipeline/evaluate_fairness.py` | checkpoint/imágenes exactos `PENDIENTE` | REVISAR |
| `xai/panel_compare_common_rust.png` | comparación LIME/SHAP/Grad-CAM histórica | `src/explainability/compare_report.py` | run/imagen exactos `PENDIENTE` | REVISAR |
| `xai/class_profile.png` | perfil global XAI, 270 imágenes | `src/explainability/global_report.py` | `xai_global_summary.csv`, `xai_global_per_image.csv` | HISTÓRICA |
| `xai/mask_audit.png` | auditoría de máscara XAI | `src/explainability/global_report.py` | mismos CSV XAI | HISTÓRICA |

## Imágenes de clases

Estas nueve imágenes son ilustraciones del catálogo, no muestras declaradas de un split ni evidencia de rendimiento. La ruta original por imagen y su licencia no están anotadas en el archivo.

| Figura | Origen declarado | Run | Generador | Estado |
|---|---|---|---|---|
| `dataset-classes/healthy.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |
| `dataset-classes/northern_corn_leaf_blight.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |
| `dataset-classes/lethal_necrosis.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |
| `dataset-classes/fall_armyworm.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |
| `dataset-classes/common_rust.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |
| `dataset-classes/gray_leaf_spot.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |
| `dataset-classes/phosphorus_deficiency.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |
| `dataset-classes/nitrogen_deficiency.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |
| `dataset-classes/potassium_deficiency.jpg` | páginas de datasets del repositorio | N/A | N/A | REVISAR ATRIBUCIÓN |

## Prototipo móvil

| Figura | Origen | Run/modelo | Generador | Estado |
|---|---|---|---|---|
| `app/home.png` | captura manual de `maize-doctor-app` | versión/commit `PENDIENTE` | captura manual | REVISAR |
| `app/camara-marco-guia.png` | captura manual de `maize-doctor-app` | versión/commit `PENDIENTE` | captura manual | REVISAR |
| `app/resultado-mancha-gris.png` | captura manual de `maize-doctor-app` | modelo/run `PENDIENTE` | captura manual | REVISAR |
| `app/resultado-potasio.png` | captura manual de `maize-doctor-app` | modelo/run `PENDIENTE` | captura manual | REVISAR |
| `app/resultado-no-reconocida.png` | captura manual de `maize-doctor-app` | estadísticas OOD/run `PENDIENTE` | captura manual | REVISAR |
| `app/contribuir.png` | captura manual de `maize-doctor-app` | versión/commit `PENDIENTE` | captura manual | REVISAR |

## Acción pendiente

Para cerrar una fila `REVISAR`, registrar commit del generador, comando, hash o ruta del artefacto de entrada, run/protocolo y hash de la imagen final. No reasignar una figura histórica a `20260921_204608` por similitud visual.

<!-- hpo-lite0-seed42-completed -->

## HPO Lite0 seed42 — 2026-09-24

Study `efficientnet_lite0_seed42_hpo_v1`; generador `src/training/tuning.py::save_optimization_plots`; fuente `study.db`/`trials.csv`. Código exacto en `source_code.zip`.

| figure | SHA256 |
| --- | --- |
| parameter_importance.png | 95c61e4b41dbcd40ed9ff983ab4d7606f13f4bcd39d8ab6f2e8cf92fe87d6c92 |
| optimization_history.png | 6f134caffabab5aaff8bdf774de2ce8312a0b3f154009391dbce2c7a80fbac60 |
