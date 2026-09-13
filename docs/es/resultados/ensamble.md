# Modelos avanzados y ensamble

## Resultado

El ensamble por voto blando de las tres arquitecturas, cada una con su mejor checkpoint
disponible, supera al mejor modelo individual en **+0,0084 de macro-F1**.

| configuración | checkpoint | macro-F1 | accuracy | precisión macro | recall macro |
| --- | --- | ---: | ---: | ---: | ---: |
| `efficientnet_lite0` | `20260812_221429` | 0,9468 | 0,9791 | 0,9533 | 0,9413 |
| `efficientnet_b0` | `20260910_170120` | 0,9483 | 0,9797 | 0,9589 | 0,9400 |
| `shufflenet_v2_x1_0` | `20260910_184521` | 0,9330 | 0,9731 | 0,9388 | 0,9298 |
| **Ensamble (voto blando)** | los tres | **0,9567** | **0,9829** | **0,9642** | **0,9506** |

Las cuatro cifras se recomputan exactamente desde `ensamble_predicciones.csv`, que guarda una
fila por imagen con la predicción del ensamble y la de cada modelo individual.

## Selección de checkpoints

El proyecto tiene ocho corridas archivadas del pipeline principal. La elección importa: el
puntero `latest.json` de `efficientnet_lite0` apunta a `20260907_163546`, la corrida con
segmentación previa, cuyo macro-F1 es 0,7191. Un ensamble construido por autodescubrimiento
habría incorporado ese modelo.

Los checkpoints se pasan explícitos. Para `efficientnet_b0` y `shufflenet_v2_x1_0` los mejores
son las corridas del 10 de septiembre, entrenadas con los hiperparámetros del barrido de Optuna
sobre `b0`:

| modelo | por defecto | modificada | delta | qué se cambió |
| --- | ---: | ---: | ---: | --- |
| `efficientnet_b0` | 0,9426 | **0,9483** | **+0,0057** | `lr` y `batch_size` |
| `shufflenet_v2_x1_0` | 0,9237 | **0,9330** | **+0,0093** | `lr` y `batch_size` |
| `efficientnet_lite0` | **0,9468** | 0,9386 | **−0,0081** | los **seis** de Optuna |

Las tres filas **no comparan la misma configuración**. Las corridas de `b0` y
`shufflenet` modificaron dos hiperparámetros; la de `lite0` modificó seis, añadiendo
`warmup_epochs` 2 en vez de 3 y `weight_decay` 1,573e-05 en vez de 1,0e-04.

El motivo es que el wrapper de Modal no propagaba `--weight-decay` ni `--warmup-epochs`
cuando se lanzaron las de septiembre. Por tanto, que `lite0` empeore **no es atribuible a la
arquitectura**: puede deberse a esos dos parámetros adicionales. Distinguirlo requiere entrenar
`lite0` cambiando sólo `lr` y `batch_size`, que no se ha hecho.

## Dónde gana y dónde pierde

La ganancia agregada de +0,0084 no está repartida. Desglosada por procedencia, **el ensamble
mejora en 4 fuentes de 14 y empeora en 5**:

| fuente | n | clases | mejor individual | ensamble | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| maize_deficiency_scanner_roboflow | 32 | 3 | 0,8831 | 0,9087 | **+0,0256** |
| corn_leaf_roboflow | 424 | 6 | 0,9128 | 0,9271 | **+0,0144** |
| multi_desease | 862 | 3 | 0,9977 | 0,9993 | +0,0017 |
| corn_leaf_diseases_classification_roboflow | 73 | 1 | 0,9931 | 0,9861 | −0,0070 |
| maize_2_roboflow | 116 | 3 | 0,9148 | 0,9068 | −0,0081 |
| cropdg | 390 | 3 | 0,9440 | 0,9286 | **−0,0154** |

Donde más gana es `corn_leaf_roboflow`, la fuente con más clases del corpus (seis) y una de las
que peor rinde individualmente. Donde más pierde es `cropdg`, donde el mejor individual ya
alcanzaba 0,9440.

El desglose completo está en `ensamble_por_fuente.csv`.

## Coste

El ensamble multiplica por tres la inferencia: tres pasadas hacia adelante por imagen y tres
juegos de pesos en memoria. La aplicación móvil despliega un único `efficientnet_lite0`
exportado a TFLite, así que el ensamble es una cifra de referencia del techo alcanzable con
estas arquitecturas, no la configuración desplegada.

## Reproducibilidad

```bash
modal run --detach scripts/modal/train.py::evaluate_ensemble_modal \
  --models "efficientnet_lite0 efficientnet_b0 shufflenet_v2_x1_0" \
  --checkpoints "/outputs/main/efficientnet_lite0/20260812_221429/best.pth \
/outputs/main/efficientnet_b0/20260910_170120/best.pth \
/outputs/main/shufflenet_v2_x1_0/20260910_184521/best.pth" \
  --num-workers 32 --output-dir /outputs/ensemble_mejores
```

## Limitaciones

Las cifras son de una sola partición y una sola semilla, sin desviación estándar asociada. La
diferencia de +0,0084 entre el ensamble y el mejor individual está por debajo de la variación
que cabría esperar entre semillas: sostiene la dirección del efecto, no su magnitud.

Todas las cifras se miden sobre la partición estándar, que comparte las catorce fuentes entre
entrenamiento y prueba. El comportamiento del ensamble bajo partición agrupada por procedencia
no se ha medido.

Los pesos del voto blando son uniformes. No se exploró ponderar por rendimiento individual.
