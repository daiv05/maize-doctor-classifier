# Evaluación rigurosa y métricas finales

## Las dos cifras del sistema

| | macro-F1 | protocolo |
| --- | ---: | --- |
| Modelo desplegado sobre prueba retenida | **0,9468** | partición estratificada por clase y entorno |
| Ensamble de tres modelos sobre prueba retenida | **0,9551** | la misma |
| **Generalización a una fuente no vista** | **0,6026 ± 0,1240** | validación cruzada agrupada por procedencia |

La diferencia entre la primera y la tercera —**0,344**— es el coste de evaluar sobre datasets
que el modelo no ha visto. Ambas describen el mismo modelo; miden preguntas distintas.

## El experimento 2×2

Se cruzan dos protocolos de partición con dos configuraciones de hiperparámetros. El modelo es
`efficientnet_lite0` en las cuatro celdas, con cinco pliegues cada una.

| | estratificada | agrupada por fuente | caída |
| --- | ---: | ---: | ---: |
| **Configuración de producción** | 0,9311 ± 0,0099 | **0,6026 ± 0,1240** | **−0,3286** |
| Hiperparámetros de `b0` | 0,9463 ± 0,0052 | 0,6499 ± 0,1508 | −0,2964 |
| clases por pliegue | 9 de 9 | 5 a 8 de 9 | |
| fuentes por pliegue | 14 de 14 | 2 a 3 de 14 | |

**El efecto del protocolo (−0,33) es veintidós veces mayor que el efecto de la configuración
(+0,015).** Cómo se parten los datos pesa mucho más que cómo se ajusta el modelo.

## Por qué el protocolo estratificado no mide generalización

`HierarchicalKFoldSplitter` reparte imágenes estratificando por clase y entorno. Medido sobre el
corpus, sus cinco pliegues contienen **las catorce fuentes, y las catorce aparecen a ambos lados
de cada pliegue**. El modelo valida siempre sobre datasets que ya vio entrenando.

`SourceGroupedKFoldSplitter` reparte fuentes completas mediante `StratifiedGroupKFold` y verifica
solape cero en cada pliegue.

| | fuentes en validación | solapan con entrenamiento |
| --- | ---: | ---: |
| Estratificada | 14 | **14 (100 %)** |
| Agrupada | 2-3 | **0** |

## El intervalo de confianza invierte su significado

El protocolo estratificado reporta ±0,0099 y ±0,0052. Esa precisión no describe la estabilidad
del modelo: describe la homogeneidad del protocolo. Sus cinco pliegues contienen las mismas
fuentes y las mismas nueve clases, así que son el mismo experimento repetido cinco veces.

El mismo modelo, bajo partición agrupada, da ±0,1240 y ±0,1508 — entre doce y veintinueve veces
más ancho. Esa es su variabilidad real frente a datasets distintos.

Un intervalo estrecho obtenido de pliegues que comparten procedencia es el argumento más
persuasivo a favor de una cifra que no se sostiene.

## El control nulo

Cada celda se acompaña del acierto de un predictor constante de la clase mayoritaria. Sin esa
referencia, el acierto por pliegue no es interpretable. Los cinco pliegues de la partición
agrupada con configuración de producción:

| pliegue | n | clases | fuentes | accuracy | control nulo | exceso | macro-F1 ev. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9 020 | 7 | 3 | 0,8244 | 0,3967 | +0,4277 | 0,5054 |
| 2 | 4 532 | 5 | 3 | **0,9314** | **0,8824** | **+0,0490** | 0,8182 |
| 3 | 7 126 | 6 | 2 | 0,7231 | 0,3877 | +0,3354 | 0,6537 |
| 4 | 3 044 | 7 | 3 | 0,7638 | 0,5240 | +0,2398 | 0,5675 |
| 5 | 4 696 | 8 | 3 | 0,7566 | 0,2890 | **+0,4676** | 0,7044 |

El pliegue 2 tiene el mejor acierto de los cinco y es el que menos demuestra: su clase
mayoritaria cubre ya el 88,24 %, así que el modelo aporta 0,0490 sobre no hacer nada. El
pliegue 5, con acierto casi veinte puntos menor, aporta 0,4676. Ordenar por accuracy invierte
el ranking de capacidad real.

## Métrica sobre clases con soporte

Bajo partición agrupada hay pliegues sin todas las clases: `lethal_necrosis` sólo existe en dos
fuentes, así que aparece como mucho en la validación de dos pliegues. Promediar el macro-F1
sobre las nueve clases introduce ceros de clases ausentes y hunde la cifra por artefacto.

Se reporta `macro_f1_evaluable`, promediado sobre las clases con soporte en cada pliegue, junto
con el número de clases evaluadas.

## Limitaciones

Cinco pliegues por celda y una sola semilla. Los intervalos de las dos celdas agrupadas
—±0,1240 y ±0,1508— se solapan holgadamente, así que la diferencia de +0,0473 a favor de los
hiperparámetros de `b0` bajo partición agrupada **no es significativa**. Es notable que
invierta el orden respecto a la partición estándar, donde esa configuración pierde por 0,0081,
pero con estos datos sólo puede señalarse como dirección, no afirmarse.

La partición agrupada cambia simultáneamente la agrupación y la distribución de clases de cada
pliegue. La caída de 0,3286 no es atribuible en exclusiva a la agrupación por procedencia.

Los casi-duplicados no se eliminaron de la partición estándar. Medido sobre el corpus, el
0,68 % de las imágenes de prueba tiene un gemelo de hash idéntico en entrenamiento.
