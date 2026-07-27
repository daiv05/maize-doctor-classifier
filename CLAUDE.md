# CLAUDE.md - Corn Leaf Disease Project

## Reglas de datos

- **Nunca modificar `raw/`.** Es inmutable, solo fuente original.
- `clean/` es la única fuente de verdad para entrenamiento. Estructura: `clean/<clase>/{lab,real}/`.
- Los CSV de `splits/` son derivados reproducibles (`make splits` / `make splits-baseline`). No editarlos a mano. Viven en `outputs/splits/` (ver más abajo), no bajo `DATASET_ROOT`.

## Arquitectura (`src/`)

- **Único punto de entrada a imagen:** `load_and_normalize_image()` (`src/data/loader.py`).
- **Rutas:** dataset fuente vía `get_dataset_root()`, artefactos generados vía `get_output_root()` (ambas en `src/config.py`) - nunca hardcodear paths ni usar la constante `DATASET_ROOT` directo.
- **Config centralizada:** `config/dataset.yaml` (clases, `target_size`, seed, perfil `baseline`). Nunca hardcodear constantes de dominio.
- **Sin `sys.path.append`.** Paquete editable (`pip install -e .`); los imports `src.*` resuelven directo.
- Convenciones detalladas de carga/rutas/`target_size`/clases minoritarias → skill `corn-data-pipeline`. Sampler de balanceo, utilidades de entrenamiento y versionado de runs → skill `corn-training-internals`.
- Para ubicar símbolos, llamadas o impacto de cambios en `src/`, usa CodeGraph (si está disponible) en vez de grep/lectura manual.

## Pipelines

- **Datos:** `clean/<clase>/{lab,real}/` → `create_splits.py` (valida integridad PIL, deduplica por SHA-256 con escaneo `sorted()` - determinista entre máquinas -, estratifica por `label+environment`) → `outputs/splits/seed_42/` (9 clases) o `outputs/splits/seed_42_baseline/` (`--baseline`, subset de `config/dataset.yaml -> baseline:`).
- **Baselines (funcional, PyTorch):** `CornDataset` → `WeightedRandomSampler` → `DataLoader` → `MODEL_REGISTRY.build(<efficientnet_b0|efficientnet_lite0|mobilenet_v3_large|fastvit_t8|ghostnetv2_100|shufflenet_v2_x1_0>)` vía `train_baselines.py`. Pese al nombre, no es un pipeline sklearn - es DL completo, pensado para comparar arquitecturas rápido y barato. Cada run también escribe `predictions.csv` (predicción + confianza por imagen de test), usado por `explain_report.py` para el análisis de errores.
- **Principal (`train.py`):** comparte toda la infraestructura de datos/modelos con baselines. Entrena una arquitectura (default `shufflenet_v2_x1_0`) sobre el dataset completo (`outputs/splits/seed_42`, 31 623 imágenes) con pérdida ponderada (`sqrt_inverse`) + label smoothing, scheduler cosine con warmup, early stopping y gradient clipping. El `WeightedRandomSampler` va **desactivado**: con desbalance de 32.9x, sampler + pérdida ponderada sobre-compensaría el mismo desbalance por dos vías, y augmenta solo las clases minoritarias. Además de las métricas estándar, escribe `test_calibration.json` (incluye `brier_binary_hit`, un Brier **binario** de acierto - el multiclase no es calculable porque `predictions.csv` guarda `pred_prob` escalar), `test_by_environment.csv` (formato largo: fila agregada `class == "__all__"` con accuracy/macro-F1, más una fila por clase con su `f1` y su `n`) y `test_grouped_metrics.json`. CLAHE es opt-in vía `--clahe` (CLI) / `CLAHE=1` (Makefile).
- **Explicabilidad (post-hoc, no acoplada al entrenamiento):** `explain_lime.py` (reporte visual LIME + Grad-CAM por imagen), `explain_report.py` (fidelidad agregada y análisis de errores, cruzando con `predictions.csv`), `scripts/checks/lime_stability.py` (auditoría manual de estabilidad de LIME). Ver sección "Explicabilidad" más abajo.

## Clases del dataset

Definidas en `config/dataset.yaml -> dataset.classes` (orden canónico para `class_to_idx`). Ratios de desbalance vs. `healthy`:
`common_rust` (3.9x), `gray_leaf_spot` (7.9x), `nitrogen_deficiency` (16.8x), `phosphorus_deficiency` (14.3x), `potassium_deficiency` (32.9x).
El pipeline extendido de augmentación / el `WeightedRandomSampler` se activan con umbral estricto `max_count/count > 4.0`, así que sobre el dataset completo califican `gray_leaf_spot`, `nitrogen_deficiency`, `phosphorus_deficiency` y `potassium_deficiency` (no `common_rust`, que queda en 3.9x).
El perfil `baseline` (`config/dataset.yaml -> baseline:`) usa las 9 clases con un tope de 1500 img/clase (`max_images_per_class`).

## Explicabilidad

Post-hoc, no acoplada al entrenamiento: `explain_lime.py` (reporte visual LIME + Grad-CAM), `explain_report.py` (fidelidad agregada / análisis de errores vía `predictions.csv`), `scripts/checks/lime_stability.py` (auditoría manual). LIME ya no corre automáticamente al entrenar (usar flag `--lime` puntual en `train_baselines.py`).

- Flujo LIME (`make explain-lime`/`explain-report`/`explain-errors`, `lime_stability.py`) → skill `corn-lime-explainability`.
- Grad-CAM (`GRADCAM_TARGET_LAYERS`, requisito al añadir modelos nuevos) → skill `corn-gradcam`.

## Dataset: hosting y descarga

`clean/` (~25k imágenes) vive en Hugging Face Datasets Hub (fuente primaria) con Google Drive de respaldo;
`download_dataset.py --source auto` resuelve cuál usar. `scripts/download_datasets.sh` es un flujo distinto:
ingesta de fuentes crudas nuevas (Kaggle/Mendeley/Roboflow) hacia `raw/`, no toca `clean/`.


## Comandos frecuentes

Convención de nombres: prefijo `modal-` = GPU en la nube, sin prefijo = local; sufijo
`-baselines` = runs de `outputs/baselines` (var `MODELS`), sufijo `-main` = runs de
`outputs/main` (var `MAIN_MODELS`). Los targets sin sufijo son los genéricos y apuntan a
baselines salvo que se pase `OUTPUT_DIR` (local) o `PIPELINE=main` (Modal). `make help`
lista todo agrupado.

```bash
make install                          # pip install -e ".[dev,analysis,xai,cloud]"
make download-dataset                 # clean/ (HF Hub, fallback Google Drive)
make splits / make splits-baseline    # regenera splits CSV
make train-baselines [MODELS=<nombre> NO_CAP=1|MAX_PER_CLASS=<n>]
make train-main [MAIN_MODELS=<nombre> MAIN_EPOCHS=<n> CLAHE=1 CLASS_WEIGHTS=<estrategia>]  # alias: make train
make modal-train-baselines [MODELS=<nombre> EPOCHS=<n>]        # baselines en GPU
make modal-train-main [MAIN_MODELS=<nombre> MAIN_EPOCHS=<n>]   # principal en GPU (alias: modal-train)
make explain-lime-baselines [MODELS=<nombre>]   # reporte visual LIME+Grad-CAM post-hoc
make explain-report-baselines [MODELS=<nombre> SAMPLE_SIZE=<n>]  # fidelidad agregada
make explain-errors-baselines [MODELS=<nombre>] # LIME dirigido a falsos positivos/negativos
make explain-lime-main / explain-report-main / explain-errors-main [MAIN_MODELS=<nombre>]
make modal-explain-lime-baselines / modal-explain-report-baselines / modal-explain-errors-baselines
make modal-explain-lime-main / modal-explain-report-main / modal-explain-errors-main
make summary                          # conteo de imágenes por clase/entorno
make test-loader                      # smoke check del pipeline de carga
make lint / make fmt                  # ruff check / ruff format
```

## Setup local

Ver [LOCAL.md](LOCAL.md) para levantar el proyecto (venv, `.env`, descarga del dataset).
