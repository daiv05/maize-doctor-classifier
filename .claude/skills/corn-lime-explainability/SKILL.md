---
name: corn-lime-explainability
description: Use when running or editing scripts/pipeline/explain_lime.py, scripts/pipeline/explain_report.py, or scripts/checks/lime_stability.py, or when asked about LIME visual reports, fidelity, or error analysis for the corn leaf disease project.
---

# Corn LIME Explainability

## Flujo

LIME ya no corre automáticamente al entrenar. `train_baselines.py` mantiene el flag `--lime` (útil para encadenarlo puntualmente), pero el target `make train-baselines` no lo pasa por defecto - entrenamiento y explicabilidad son pasos separados.

## Targets: local vs. Modal, baselines vs. main

Cada script tiene cuatro entradas en el Makefile, diferenciadas por nombre: `explain-*-baselines` / `explain-*-main` (local) y `modal-explain-*-baselines` / `modal-explain-*-main` (GPU). Los targets sin sufijo son los genéricos y apuntan a baselines salvo que reciban `OUTPUT_DIR` (local, p.ej. `outputs/main`) o `PIPELINE=main` (Modal). El directorio de runs es lo único que cambia: `--output-dir` decide de dónde se leen checkpoints y `predictions.csv`, y su default es `<OUTPUT_ROOT>/baselines`.

En Modal la selección viaja como `--pipeline baselines|main` y el contenedor la traduce a `--output-dir` (`scripts/modal/explain.py::_output_dir_args`). No pasar rutas absolutas por `make` en Windows: MSYS las reescribe a rutas de Windows.

## `make explain-lime` (`scripts/pipeline/explain_lime.py`)

Reporte visual por imagen (`<run_dir>/lime_visual/`), muestreo balanceado chico (`lime.images_per_class` en `config/dataset.yaml`) o `--image` puntual. Persiste, junto al PNG, un `.json` (predicción + pesos por superpíxel) y un `.npy` (mapa de segmentos) para reanálisis sin re-ejecutar LIME.

## `make explain-report` / `make explain-errors` (`scripts/pipeline/explain_report.py`)

Fidelidad agregada sobre una muestra amplia (`lime.report_sample_size`) o, con `--errors-only`, explica específicamente las filas de `predictions.csv` donde `label != pred_label`. Requiere que el run ya tenga `predictions.csv` (lo genera `train_baselines.py`).

## `scripts/checks/lime_stability.py`

Diagnóstico manual (sin target Make) para auditar cuán estable es una explicación LIME sobre una imagen puntual corriendo varias seeds y comparando IoU/correlación.
