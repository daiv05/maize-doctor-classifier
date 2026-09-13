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

## Ensamble

| fichero | contenido |
| --- | --- |
| `ensamble_comparacion.csv` | los tres modelos individuales y el voto blando |

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
