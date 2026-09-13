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

`make explain-global-main` no renderiza paneles por imagen: acumula valores de Shapley sobre una muestra balanceada de `predictions.csv` para producir un perfil por clase (`class_profile.png`) y una tabla agregada (`global_summary.csv`) con, entre otras columnas, el ratio de atribución hoja/fondo (`mean_leaf_attribution_ratio`).

**El perfil es deliberadamente no espacial.** Promediar los mapas de atribución en coordenadas de píxel mezcla imágenes donde la hoja cae en distinta posición, ángulo y escala, así que el promedio converge a una mancha centrada que describe el **encuadre del dataset**, no dónde mira el modelo: aun con atribución perfectamente sobre la lesión en todas las imágenes, el promedio se aplana hacia el centro. Por eso `class_profile.png` reporta cantidades invariantes a la posición de la hoja - la distribución del ratio hoja/fondo y la dispersión, separando aciertos de errores -, con una línea de referencia en la cobertura media de la máscara: ése es el ratio que daría una atribución repartida al azar, y sin ese ancla el número no se puede interpretar. Los mapas promediados siguen escribiéndose en `framing_diagnostics/`, etiquetados como lo que son: un diagnóstico de encuadre.

**La máscara es auditable, no un artículo de fe.** `mask_audit.png` muestra, para unas pocas imágenes por clase, qué considera hoja la máscara (el fondo se atenúa) junto a su cobertura, fragmentación y si fue rechazada. El ratio no se puede interpretar sin mirar ese panel: la máscara es una heurística de color (ExG cromático + umbral absoluto + apertura y cierre morfológicos, `src/explainability/leaf_mask.py`), no un segmentador aprendido. El cierre es lo que reincorpora las lesiones internas: al ser marrones o amarillas no pasan el umbral de verde, y sin él la máscara excluía el síntoma (medido: 19% del interior de la hoja en `common_rust`), de modo que una atribución correcta sobre la lesión contaba como atribución al fondo.

**Salvedad honesta sobre ese ratio:** la heurística puede fallar en hojas cloróticas o con fondo similar en color. `global_summary.csv` nunca reporta el ratio sin contexto: junto a `mean_leaf_attribution_ratio` siempre van `n_mask_rejected` (imágenes cuya máscara se descartó por cobertura degenerada) y `n_ratio_undefined` (imágenes con máscara válida pero sin atribución positiva que repartir - un síntoma distinto, del lado del modelo y no de la máscara). `ratio_reliable` resume ambas causas: se apaga cuando su suma supera el 30% de las imágenes de esa fila. Leer el ratio sin mirar `ratio_reliable` es leerlo a ciegas.

## Los artefactos visuales

### Panel comparado

![Panel LIME, SHAP y Grad-CAM](/xai/panel_compare_common_rust.png)

Las tres técnicas sobre la misma imagen y la **misma segmentación**: LIME y SHAP explican exactamente los mismos superpíxeles, así que sus atribuciones son comparables término a término. El pie recoge las tres métricas de acuerdo de ese panel.

Grad-CAM usa `jet` aparte porque su magnitud es no negativa; LIME y SHAP comparten un divergente centrado en cero que evita el eje rojo-verde, donde el verde ya significa tejido sano y el contraste colapsa en daltonismo.

### Perfil global por clase

![Perfil global por clase](/xai/class_profile.png)

Distribución del ratio hoja/fondo por clase, con la línea de azar y las clases de ratio no fiable marcadas con `(!)`. Es deliberadamente **no espacial**: promediar mapas de atribución en coordenadas de píxel mezcla hojas en distinta posición y ángulo, y converge a una mancha centrada que describe el encuadre del corpus, no el modelo.

### Auditoría de la máscara

![Auditoría de la máscara foliar](/xai/mask_audit.png)

La zona atenuada es lo que la máscara considera fondo. **Ningún ratio de las tablas anteriores debe leerse sin mirar este panel.** Las imágenes marcadas `RECHAZADA` tienen cobertura 1.00 —la máscara declara hoja a la imagen entera— y quedan fuera del cómputo. Entre ellas hay casos donde la hoja efectivamente llena el encuadre y casos donde la máscara falla de verdad: en una de `nitrogen_deficiency` marca como hoja el suelo de baldosa y una pierna.

## Resultados sobre el modelo desplegado

Las cifras del perfil global y del panel comparado sobre `efficientnet_lite0/20260812_221429`
están en [Análisis de sesgos y ética](/es/resultados/equidad). En resumen:

| métrica de acuerdo LIME-SHAP | observado | esperado bajo independencia | exceso |
| --- | ---: | ---: | ---: |
| `spearman` | 0,730 | 0,000 | **+0,730** |
| `iou_topk` | 0,521 | 0,127 | **+0,395** |
| `sign_agreement` | 0,823 | **0,707** | **+0,116** |

Las tres métricas viajan con su valor esperado bajo independencia. Sin esa referencia,
`sign_agreement` parece la más fuerte de las tres y es la más débil: dos vectores
mayoritariamente del mismo signo coinciden mucho sin que eso signifique acuerdo, y **7 de 45
paneles no superan su propio nulo**.

En el perfil global, el `attribution_excess` —ratio de atribución a la hoja menos cobertura de
la máscara— reparte las seis clases fiables en tres por encima del azar y tres por debajo.

## Alcance

`compare` y `global` son exclusivos del pipeline principal (`outputs/main`); no tienen variante `-baselines`. Los baselines se quedan con LIME + Grad-CAM (`visual`/`fidelity`/`errors`) - ver [interpretabilidad de baselines](../pipeline-baselines/interpretabilidad.md) para el porqué.
