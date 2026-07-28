---
name: corn-xai
description: Use when running or editing scripts/pipeline/explain.py, src/explainability/*, or scripts/checks/lime_stability.py, or when asked about LIME, SHAP, Grad-CAM, fidelity, error analysis or global attribution profiles for the corn leaf disease project.
---

# Corn XAI

## Flujo

La explicabilidad es post-hoc y no corre al entrenar. `train_baselines.py` mantiene el flag `--lime` para encadenarla puntualmente, pero `make train-baselines` no lo pasa por defecto.

## CLI unificado (`scripts/pipeline/explain.py`)

Cinco subcomandos. Los nombres describen el artefacto, no la técnica: casi todos los paneles combinan varias.

| Subcomando | Produce | Directorio |
|---|---|---|
| `visual` | Panel por imagen: LIME + Grad-CAM | `<run_dir>/explain_visual/` |
| `fidelity` | Muestra amplia + resumen por clase | `<run_dir>/explain_fidelity/` |
| `errors` | Panel sobre `label != pred_label` | `<run_dir>/explain_errors/` |
| `compare` | Panel LIME \| SHAP \| Grad-CAM + acuerdo | `<run_dir>/explain_compare/` |
| `global` | Mapas medios por clase + ratio hoja/fondo | `<run_dir>/explain_global/` |

`compare` y `global` son exclusivos del pipeline principal (`outputs/main`).

## Targets: local vs. Modal, baselines vs. main

Los tres primeros subcomandos tienen cuatro entradas en el Makefile: `explain-<sub>-{baselines,main}` y `modal-explain-<sub>-{baselines,main}`. `compare` y `global` solo tienen `-main`. Los targets sin sufijo son genéricos y apuntan a baselines salvo que reciban `OUTPUT_DIR` (local) o `PIPELINE=main` (Modal). No pasar rutas absolutas por `make` en Windows: MSYS las reescribe.

## SHAP: lo que hay que saber antes de tocarlo

- **KernelSHAP sobre superpíxeles**, con línea base única configurable (`shap.background`, `black` por defecto para paridad con el `hide_color=0` de LIME).
- **La segmentación se calcula afuera** (`src/explainability/segmentation.py`) y se inyecta tanto en LIME como en SHAP. Sin eso no hay comparación por superpíxel: LIME segmenta con quickshift por su cuenta.
- **Consecuencia:** las regiones LIME de `compare` no coinciden con las de `visual`. Es esperado; `shap.segmentation: quickshift` las hace coincidir.
- **Determinismo:** `explain_with_kernel_shap` fija la semilla de NumPy y restaura el estado previo. No quitar ese bloque: el determinismo es el argumento por el que SHAP entró al pipeline.
- **Ratio hoja/fondo:** ExG+Otsu es una heurística y puede fallar en hojas cloróticas. Nunca leer `mean_leaf_attribution_ratio` sin mirar `ratio_reliable`, `n_mask_rejected` y `n_ratio_undefined` en la misma fila: son dos causas distintas del mismo síntoma (máscara de vegetación fallando vs. modelo sin atribución positiva que repartir) y `ratio_reliable` se apaga cuando su suma supera el umbral.

## `scripts/checks/lime_stability.py`

Diagnóstico manual (sin target Make) para auditar la estabilidad de LIME sobre una imagen corriendo varias seeds y comparando IoU/correlación.
