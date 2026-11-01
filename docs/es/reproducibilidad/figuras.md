# Procedencia de figuras

Figuras de `public/` cuyo generador y artefacto fuente están registrados. Las rutas son relativas a `public/`.

## Análisis exploratorio

Generadas por `notebooks/01_eda.ipynb` sobre el corpus ampliado de agosto de 2026 (33 438 imágenes). Los conteos no corresponden al `master_manifest.csv` vigente. Aparecen en `docs/es/exploratory-data-analysis/`.

| Figura | Artefacto fuente |
|---|---|
| `eda/eda_00_muestra_visual.png` | corpus `clean/` de esa fecha |
| `eda/eda_01_distribucion_clases.png` | inventario del corpus |
| `eda/eda_02_lab_vs_real.png` | inventario por ambiente |
| `eda/eda_02b_heatmap_fuente_clase.png` | inferencia de fuentes |
| `eda/eda_03_resoluciones.png`, `eda/eda_03b_resolucion_por_clase.png` | metadatos de imágenes |
| `eda/eda_04_calidad.png`, `eda/eda_04b_calidad_boxplots.png` | métricas de calidad |
| `eda/eda_04c_color_hsv.png`, `eda/eda_04d_hue_lab_vs_real.png` | estadísticas HSV |
| `eda/eda_05_duplicados.png` | registro de deduplicación |
| `eda/eda_06_sesgos.png` | conteos por clase y ambiente |

## Resultados, equidad y explicabilidad

| Figura | Generador | Artefacto fuente |
|---|---|---|
| `ensemble/confusion_matrix_ensemble.png` | `scripts/pipeline/evaluate_ensemble.py` | `ensamble_predicciones.csv` y `ensamble_resumen.json` |
| `resultados/ablacion_control_nulo.png` | `scripts/experiments/figura_ablacion.py` | `equidad_metricas.json` y predicciones de equidad |
| `fairness/disaggregated_confusion_matrices.png` | `scripts/pipeline/evaluate_fairness.py` | predicciones de equidad |
| `xai/class_profile.png`, `xai/mask_audit.png` | `src/explainability/global_report.py` | `xai_global_summary.csv` y `xai_global_per_image.csv` |
| `xai/panel_compare_common_rust.png` | `src/explainability/compare_report.py` | run e imagen de la comparación LIME, SHAP y Grad-CAM |

## HPO Lite0 seed 42

Study `efficientnet_lite0_seed42_hpo_v1`. Generador `src/training/tuning.py::save_optimization_plots`; fuente `study.db` y `trials.csv`; código exacto en `source_code.zip`.

| Figura | SHA-256 |
|---|---|
| `parameter_importance.png` | `95c61e4b41dbcd40ed9ff983ab4d7606f13f4bcd39d8ab6f2e8cf92fe87d6c92` |
| `optimization_history.png` | `6f134caffabab5aaff8bdf774de2ce8312a0b3f154009391dbce2c7a80fbac60` |
