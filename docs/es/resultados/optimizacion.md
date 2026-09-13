# Optimización de hiperparámetros de `efficientnet_lite0`

## Resultado

Para `efficientnet_lite0`, que es la arquitectura desplegada, los valores por defecto del
pipeline ganan en el conjunto de prueba a las dos configuraciones optimizadas que se midieron.
La configuración de producción no cambia.

Esa conclusión es específica de `lite0`. La misma configuración afinada **mejora** a
`efficientnet_b0` y a `shufflenet_v2_x1_0`.

| configuración | val macro-F1 | **test macro-F1** | épocas | mejor época |
| --- | ---: | ---: | ---: | ---: |
| **Valores por defecto** | **0,9554** | **0,9468** | 43 | 35 |
| Optuna sobre `lite0`, techo 15 épocas | 0,9451 | 0,9379 | 20 | 12 |
| Optuna sobre `b0`, aplicada a `lite0` | 0,9548 | 0,9386 | 31 | 23 |

Las tres cifras de prueba se recomputan exactamente desde sus CSV de predicciones por imagen.

## Las tres configuraciones

| hiperparámetro | por defecto | Optuna `lite0` | Optuna `b0` |
| --- | --- | --- | --- |
| `learning_rate` | 1,00e-04 | 2,72e-04 | 4,55e-04 |
| `weight_decay` | 1,00e-04 | 1,11e-05 | 1,57e-05 |
| `batch_size` | 32 | 64 | 64 |
| `label_smoothing` | 0,100 | 0,056 | 0,100 |
| `warmup_epochs` | 3 | 1 | 2 |
| `class_weights` | `sqrt_inverse` | `sqrt_inverse` | `sqrt_inverse` |

## Por qué falló la búsqueda con techo de 15 épocas

El barrido sobre `lite0` evaluó cada trial con un máximo de 15 épocas. La corrida de referencia
alcanza su mejor validación en la **época 35**, así que a las 15 épocas ninguna configuración ha
terminado de converger y el criterio de selección deja de medir calidad para medir **velocidad de
convergencia**.

La configuración elegida lo refleja: triplica el learning rate, reduce el suavizado de etiquetas
a la mitad y acorta el warmup a una época. Con eso llega a 0,9451 en la época 12 y se agota; la
paciencia de 8 la detiene en la 20.

La comparación intermedia lo hace visible:

| época | por defecto | Optuna `lite0` |
| ---: | ---: | ---: |
| 12 | ~0,910 | **0,9451** |
| 15 | 0,9288 | ~0,945 |
| 35 | **0,9554** | detenida |

A 15 épocas la configuración optimizada gana por 0,0195. A presupuesto completo pierde por
0,0103. El proxy premiaba la cualidad equivocada.

## El contraejemplo

El barrido previo sobre `efficientnet_b0` produjo una configuración distinta. Aplicada a
`lite0`, sostiene el entrenamiento hasta la época 23 y alcanza 0,9548 de validación, a seis
diezmilésimas de los valores por defecto, frente al 0,9451 de la búsqueda con techo de 15
épocas. No colapsa temprano.

Sus trials completos duraron entre 1 h 24 min y 2 h 18 min, frente a los 15 minutos de los de
`lite0`. **El número de épocas por trial no está registrado en ningún artefacto del estudio**:
la documentación del pipeline indica 15, y las duraciones son compatibles con un presupuesto
mayor una vez descontado que aquel estudio corrió con cuatro núcleos de CPU y sobre una
arquitectura más pesada. No se puede determinar cuál de las dos cosas explica la diferencia.

Lo que sí está medido es que esa configuración **no transfiere entre arquitecturas**:

| modelo | por defecto | hiperparámetros de `b0` | delta |
| --- | ---: | ---: | ---: |
| `efficientnet_b0` | 0,9426 | **0,9483** | **+0,0057** |
| `shufflenet_v2_x1_0` | 0,9237 | **0,9330** | **+0,0093** |
| `efficientnet_lite0` | **0,9468** | 0,9386 | **−0,0081** |

Mejora las dos arquitecturas sobre las que no se buscó menos una: perjudica precisamente a
`lite0`, que es la desplegada. La conclusión no es que los valores por defecto sean mejores en
general, sino que **lo son para `lite0`**.

## Coste y configuración del barrido

| | |
| --- | --- |
| Trials | 25 — 6 completos, 19 podados |
| Podador | `MedianPruner(n_startup_trials=3, n_warmup_steps=3)` |
| Duración | 15 min por trial completo, 3 min por trial podado |
| Espacio | 6 hiperparámetros; CLAHE excluido |

CLAHE se retiró del espacio de búsqueda tras medir su coste: baja el pipeline de
transformaciones de 107,5 a 40,6 imágenes por segundo (9,3 → 24,7 ms por imagen). Como el
entrenamiento está limitado por CPU, cada trial que lo activaba costaba el doble —33 minutos
frente a 15— para decidir una única opción binaria de preprocesado. Queda pendiente medirlo como
comparación directa de dos corridas.

## La configuración de `b0` bajo partición agrupada

El experimento 2×2 de [Evaluación rigurosa](/es/resultados/evaluacion) mide esa misma
configuración con validación cruzada agrupada por procedencia, y allí el orden se invierte:

| protocolo | por defecto | hp de `b0` | diferencia |
| --- | ---: | ---: | ---: |
| Prueba retenida, partición estándar | **0,9468** | 0,9386 | −0,0081 |
| Validación cruzada agrupada | 0,6026 ± 0,1240 | **0,6499 ± 0,1508** | +0,0473 |

Los intervalos se solapan holgadamente, así que con cinco pliegues la ventaja bajo partición
agrupada **no es significativa**. La dirección es coherente con que una configuración que ajusta
peor a la partición con fuga se apoye menos en ella, pero con estos datos sólo puede señalarse.

## Limitaciones

No se ejecutó un barrido sobre `lite0` a presupuesto completo, que sería el cuarto brazo del
experimento y costaría unas cinco horas. La conclusión de que los valores por defecto son
suficientes se apoya en que ninguna de las dos configuraciones optimizadas medidas los supera en
prueba, no en haber agotado el espacio de búsqueda.

La diferencia entre los tres brazos en prueba —0,9468, 0,9386 y 0,9379— se midió con una sola
semilla. No hay desviación estándar asociada, así que las distancias por debajo de una centésima
no deberían leerse como diferencias reales.
