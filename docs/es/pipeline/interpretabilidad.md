# Interpretabilidad

La interpretabilidad del pipeline principal combina **LIME** (regiones de superpíxeles que sostienen el diagnóstico), **Grad-CAM** (mapa de activación de la clase predicha) y **SHAP** (valores de Shapley por superpíxel), post-hoc y no acopladas al entrenamiento. Los tres se ejecutan vía `scripts/pipeline/explain.py` (ver [teoría de interpretabilidad](../deep-learning/interpretability)).

## Panel comparado (`compare`)

`make explain-compare-main` genera, por imagen, un panel LIME | SHAP | Grad-CAM sobre una **segmentación compartida**: el mapa de superpíxeles se calcula una sola vez (`src/explainability/segmentation.py`, SLIC por defecto) y se inyecta tanto en LIME como en KernelSHAP, así que las dos técnicas explican exactamente las mismas regiones y sus atribuciones son comparables término a término. Esto es distinto del panel `visual` (LIME + Grad-CAM), donde LIME sigue segmentando con quickshift por su cuenta - las regiones de un panel y otro no coinciden, y es esperado.

Sobre esa base común se calculan tres métricas de acuerdo entre LIME y SHAP:

| Métrica | Responde |
|---|---|
| `iou_topk` | ¿coinciden en **dónde mirar**? (solapamiento de los segmentos positivos top-k) |
| `spearman` | ¿coinciden en el **orden** de importancia de los segmentos? |
| `sign_agreement` | ¿coinciden en la **dirección** del empuje (a favor o en contra de la clase)? |

Un acuerdo alto en las tres refuerza la confianza en la explicación; un desacuerdo (frecuente cuando el modelo se apoya en un atajo poco robusto) es en sí mismo un hallazgo.

## Perfil global (`global`)

`make explain-global-main` no renderiza paneles por imagen: acumula valores de Shapley sobre una muestra balanceada de `predictions.csv` para producir, por clase, un mapa espacial medio de atribución (`<clase>_attribution_map.png`) y una tabla agregada (`global_summary.csv`) con, entre otras columnas, el ratio de atribución hoja/fondo (`mean_leaf_attribution_ratio`).

**Salvedad honesta sobre ese ratio:** la máscara de hoja se calcula con una heurística (ExG + Otsu, `src/explainability/leaf_mask.py`) que puede fallar en hojas cloróticas o con fondo similar en color. `global_summary.csv` nunca reporta el ratio sin contexto: junto a `mean_leaf_attribution_ratio` siempre van `n_mask_rejected` (imágenes cuya máscara se descartó por cobertura degenerada) y `n_ratio_undefined` (imágenes con máscara válida pero sin atribución positiva que repartir - un síntoma distinto, del lado del modelo y no de la máscara). `ratio_reliable` resume ambas causas: se apaga cuando su suma supera el 30% de las imágenes de esa fila. Leer el ratio sin mirar `ratio_reliable` es leerlo a ciegas.

## Alcance

`compare` y `global` son exclusivos del pipeline principal (`outputs/main`); no tienen variante `-baselines`. Los baselines se quedan con LIME + Grad-CAM (`visual`/`fidelity`/`errors`) - ver [interpretabilidad de baselines](../pipeline-baselines/interpretabilidad.md) para el porqué.
