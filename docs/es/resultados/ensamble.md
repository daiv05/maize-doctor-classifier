# Modelos avanzados y ensamble

## Resultado

El ensamble por voto blando de las tres arquitecturas supera al mejor modelo individual en
**+0,0083 de macro-F1** sobre el conjunto de prueba.

| configuración | macro-F1 | accuracy | precisión macro | recall macro |
| --- | ---: | ---: | ---: | ---: |
| `efficientnet_lite0` | 0,9468 | 0,9791 | 0,9533 | 0,9413 |
| `efficientnet_b0` | 0,9426 | 0,9777 | 0,9528 | 0,9348 |
| `shufflenet_v2_x1_0` | 0,9237 | 0,9649 | 0,9300 | 0,9185 |
| **Ensamble (voto blando)** | **0,9551** | **0,9821** | **0,9623** | **0,9491** |

Las cuatro cifras se recomputan exactamente desde `ensemble_predictions.csv`, que guarda una
fila por imagen con la predicción del ensamble y la de cada modelo individual.

## Dónde gana y dónde pierde

La ganancia agregada de +0,0083 no está repartida. Desglosada por procedencia:

| fuente | n | clases | mejor individual | ensamble | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| maize_field | 133 | 3 | 0,8547 | 0,8701 | **+0,0154** |
| maize_2_roboflow | 116 | 3 | 0,8784 | 0,8928 | **+0,0145** |
| corn_leaf_roboflow | 424 | 6 | 0,9107 | 0,9224 | **+0,0117** |
| multi_desease | 862 | 3 | 0,9976 | 0,9981 | +0,0005 |
| maize_africa_v1, v1.2, leaf_roboflow, desease, corn_leaf_diseases_class. | 48-330 | 1-2 | — | — | 0,0000 |
| maize_africa | 1 499 | 3 | 0,9985 | 0,9983 | −0,0002 |
| maize_desease_v1.1 | 742 | 1 | 1,0000 | 0,9993 | −0,0007 |
| cropdg | 390 | 3 | 0,9473 | 0,9440 | −0,0033 |
| maize_nutrient | 49 | 4 | 0,8980 | 0,8897 | −0,0083 |
| maize_deficiency_scanner_roboflow | 32 | 3 | 0,9195 | 0,9087 | −0,0108 |

**El ensamble mejora en 4 fuentes de 14 y empeora en 5.** Las tres en las que más gana
—`maize_field`, `maize_2_roboflow`, `corn_leaf_roboflow`— son precisamente aquellas donde el
mejor modelo individual rinde peor. Donde pierde, o bien el rendimiento individual ya roza el
techo (`maize_desease_v1.1` con 1,0000) o bien el soporte es de decenas de imágenes.

Es decir, el ensamble aporta donde el clasificador es débil y resta de forma marginal donde ya
no queda margen. Ese comportamiento es el que justifica adoptarlo, más que el +0,0083 agregado.

## Coste

El ensamble multiplica por tres la inferencia: tres pasadas hacia adelante por imagen y tres
juegos de pesos en memoria. La aplicación móvil despliega un único `efficientnet_lite0`
exportado a TFLite, así que el ensamble es una cifra de referencia del techo alcanzable con
estas arquitecturas, no la configuración desplegada.

## Limitaciones

Las cifras son de una sola partición y una sola semilla, sin desviación estándar asociada. La
diferencia de +0,0083 entre el ensamble y el mejor individual está por debajo de la variación
que cabría esperar entre semillas, así que sostiene la dirección del efecto pero no su magnitud.

Todas las cifras de esta página se miden sobre la partición estándar, que comparte procedencia
entre entrenamiento y prueba. El comportamiento del ensamble bajo partición agrupada por fuente
no se ha medido.
