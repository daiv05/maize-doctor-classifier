# Evidencia bruta — resultados

Artefactos de las mediciones reportadas en esta sección. Cada cifra publicada se recomputa desde
estos ficheros.

## Optimización de hiperparámetros

| fichero | contenido |
| --- | --- |
| `optuna_lite0_15epocas_best_params.json` | mejor configuración del barrido con techo de 15 épocas |
| `optuna_lite0_15epocas_trials.csv` | los 25 trials con su estado, valor y duración |
| `optuna_b0_presupuesto_completo_best_params.json` | mejor configuración del barrido sobre `b0` |
| `optuna_b0_presupuesto_completo_trials.csv` | los 15 trials de ese barrido |
| `run_afinada_lite0_summary.json` | reentrenamiento a presupuesto completo de la primera |
| `run_afinada_b0_summary.json` | reentrenamiento a presupuesto completo de la segunda |
| `run_20260921_lite0_summary.json` | primer reentrenamiento de Lite0 sobre el split corregido de 33,429 muestras |

## Manifiesto de corridas

`manifiesto_corridas.csv` lista las diez corridas auditadas del pipeline principal que existen en el
Volume `corn-outputs`, con su ruta en Modal, su configuración completa, sus métricas publicadas
y el fichero desde el que cada cifra se recomputa. **Las diez verifican de forma exacta.**

Dos de ellas —`efficientnet_b0/20260910_170120` y `shufflenet_v2_x1_0/20260910_184521`— no
guardaron `predictions.csv` propio; sus cifras se verifican desde la columna correspondiente de
`ensamble_predicciones.csv`, que registra la predicción de cada modelo por imagen.

Se regenera con:

```bash
python scripts/experiments/manifiesto_auditoria.py
```

## Ensamble

| fichero | contenido |
| --- | --- |
| `ensamble_comparacion.csv` | los tres modelos individuales y el voto blando |
| `ensamble_predicciones.csv` | 5 015 filas con procedencia y la predicción de cada modelo |
| `ensamble_resumen.json` | resumen de la corrida |
| `ensamble_por_fuente.csv` | ganancia del ensamble desglosada por las 14 fuentes |

## Validación cruzada

| fichero | contenido |
| --- | --- |
| `kfold_2x2_comparacion.csv` | las cuatro celdas agregadas, con IC 95 % y control nulo |
| `kfold_estratificado_actual_pliegues.csv` | métricas por pliegue de cada celda |
| `kfold_estratificado_afinada_pliegues.csv` | |
| `kfold_agrupado_actual_pliegues.csv` | |
| `kfold_agrupado_afinada_pliegues.csv` | |

## Equidad

| fichero | contenido |
| --- | --- |
| `equidad_por_fuente.csv` | las 14 fuentes con su clase mayoritaria y macro-F1 evaluable |
| `equidad_metricas.json` | disparidad por entorno, ablación dual y control nulo |

El macro-F1 de `lab` promedia sobre las 3 clases con soporte y el de `real` sobre las 9, así que el cociente entre ambos no mide disparidad. El DIR publicado restringe los dos subgrupos a las 3 clases compartidas y se recomputa desde `fairness_predictions.csv` unido a `outputs/splits/seed_42/test.csv` por el nombre de fichero.

## Estabilidad entre semillas

| fichero | contenido |
| --- | --- |
| `semilla1_summary.json` | corrida de la configuración de producción con semilla 1 |
| `semilla2_summary.json` | corrida de la configuración de producción con semilla 2 |

La semilla 42 es la del modelo desplegado y su resumen vive en el manifiesto. Las tres comparten partición materializada, de modo que la única fuente de variación es la inicialización y el barajado: media 0.9430, $\sigma = 0.0049$, IC 95 % de ±0.0122.

## Interpretabilidad

| fichero | contenido |
| --- | --- |
| `xai_global_summary.csv` | perfil por clase y acierto: ratio de atribución foliar, cobertura de máscara, dispersión y exceso de atribución |
| `xai_global_per_image.csv` | las 270 imágenes del perfil con su procedencia, su máscara y sus métricas individuales |

`attribution_excess` es la columna que sostiene la lectura publicada: ratio de atribución sobre la hoja menos la cobertura de la máscara, que es el nivel de azar de cada imagen. Un ratio alto con cobertura igualmente alta no indica atención foliar, sino que la hoja llena el encuadre.

Ninguna fila debe leerse sin mirar `ratio_reliable`, `n_mask_rejected` y `n_ratio_undefined` en la misma fila: tres de las nueve clases tienen la máscara degradada y su ratio no es interpretable. La columna `mask_source` registra que las 270 usan la segmentación del modelo de `maize-doctor-segmenter`, no el umbral cromático ExG.

## Reproducción

Las métricas de prueba se recomputan desde el `predictions.csv` de cada corrida:

```python
import pandas as pd
from sklearn.metrics import f1_score

p = pd.read_csv("predictions.csv")
f1_score(p.label, p.pred_label, average="macro", zero_division=0)
```

La comparación del experimento 2×2 se regenera con:

```bash
python scripts/experiments/comparar_kfold.py --raiz outputs/fase_f
```
