# Análisis de sesgos y ética

## La ablación con su control nulo

Se ocluye alternativamente el centro y la periferia de cada imagen de prueba, con negro real en
espacio normalizado ImageNet. La referencia es el acierto de un predictor constante de la clase
mayoritaria: **0,2614**.

| condición | qué queda visible | accuracy | exceso sobre el nulo | confianza retenida |
| --- | --- | ---: | ---: | ---: |
| Imagen completa | todo | 0,9791 | +0,7177 | — |
| Oclusión central 60 % | **sólo la periferia** | **0,7946** | **+0,5332** | 0,6916 |
| Oclusión periférica 40 % | **sólo el centro** | **0,4225** | **+0,1611** | 0,4266 |

**El fondo por sí solo predice mejor que la hoja por sí sola: 0,7946 frente a 0,4225.**

Con la lámina foliar tapada y únicamente el marco visible, el modelo acierta el 79,5 % de las
imágenes y retiene el 69 % de su confianza. Con el fondo tapado y la lesión visible, cae al
42,3 %, la tasa de cambio de predicción sube al 57,4 % y el colapso queda confirmado.

Sin la columna del control nulo ninguno de esos números es interpretable: un modelo degenerado
que emitiera siempre la clase mayoritaria alcanzaría 0,2614 sin extraer nada de la imagen.

La máscara es geométrica rectangular, no una segmentación anatómica de la lámina. Mide
sensibilidad a regiones espaciales, no causalidad biológica sobre fondo frente a lesión.

![Ablación con control nulo](/resultados/ablacion_control_nulo.png)

## Desagregación por entorno

| subgrupo | n | macro-F1 evaluable | accuracy |
| --- | ---: | ---: | ---: |
| Laboratorio | 532 | 0,9423 | 0,9680 |
| Campo real | 4 483 | 0,9215 | 0,9804 |

Disparidad de macro-F1 **0,0209**, DIR **0,9778**, cumple la regla del 80 %.

El subgrupo de laboratorio contiene únicamente tres de las nueve clases. Promediando sobre las
nueve, su macro-F1 sería 0,2955 por los ceros de las seis clases ausentes; la cifra de 0,9423
promedia sobre las clases con soporte real.

![Matrices de confusión desagregadas](/fairness/disaggregated_confusion_matrices.png)

## Desagregación por procedencia

![Rendimiento por procedencia con su control nulo](/resultados/equidad_por_fuente.png)

El entorno tiene dos categorías. La procedencia tiene catorce, y es el eje con más varianza del
corpus.

| fuente | n | clases | accuracy | clase mayoritaria | **exceso** | macro-F1 ev. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| maize_deficiency_scanner_roboflow | 32 | 3 | 0,8438 | 0,4062 | +0,4375 | 0,8571 |
| maize_2_roboflow | 116 | 3 | 0,8707 | 0,3707 | +0,5000 | 0,8743 |
| maize_field | 133 | 3 | 0,8797 | 0,6917 | **+0,1880** | 0,8483 |
| maize_nutrient | 49 | 4 | 0,9184 | 0,4898 | +0,4286 | 0,8980 |
| corn_leaf_roboflow | 424 | 6 | 0,9245 | 0,2665 | **+0,6580** | 0,9107 |
| cropdg | 390 | 3 | 0,9590 | 0,4615 | +0,4974 | 0,9440 |
| corn_leaf_diseases_classification_roboflow | 73 | **1** | 0,9726 | 1,0000 | **−0,0274** | 0,9861 |
| multi_desease | 862 | 3 | 0,9930 | 0,5429 | +0,4501 | 0,9943 |
| maize_africa | 1 499 | 3 | 0,9947 | 0,4303 | **+0,5644** | 0,9965 |
| maize_desease | 330 | 2 | 0,9970 | 0,5576 | +0,4394 | 0,9983 |
| maize_africa_v1.2 | 255 | **1** | 1,0000 | 1,0000 | **0,0000** | 1,0000 |
| maize_desease_v1.1 | 742 | **1** | 1,0000 | 1,0000 | **0,0000** | 1,0000 |
| maize_africa_v1 | 62 | **1** | 1,0000 | 1,0000 | **0,0000** | 1,0000 |
| maize_leaf_roboflow | 48 | **1** | 1,0000 | 1,0000 | **0,0000** | 1,0000 |

Brecha de accuracy **0,1562**; DIR por procedencia **0,8438**, cumple la regla del 80 %.

**Cuatro fuentes puntúan accuracy 1,0000 y las cuatro contienen una sola clase.** Un predictor
constante también acierta el 100 % en ellas: su exceso sobre el nulo es cero y no aportan
evidencia de capacidad de clasificación. `maize_field`, pese a un acierto de 0,8797, sólo aporta
+0,1880 porque su clase mayoritaria cubre ya el 69 %.

Las fuentes que de verdad demuestran capacidad son `corn_leaf_roboflow` (+0,6580 sobre el nulo,
seis clases) y `maize_africa` (+0,5644). Ordenar la tabla por accuracy invierte el ranking:
coloca arriba justo a las que no prueban nada.

## Atención visual

![Panel Grad-CAM](/fairness/gradcam_samples.png)

Grad-CAM muestra dónde se concentra el gradiente de la clase predicha, **no cuánta información
aporta cada región**. No puede confirmar ni refutar la ablación de arriba, porque no mide lo
mismo. Cuando ambos parecen discrepar, manda la medición cuantitativa.

## Fuga de procedencia

Sobre la partición estándar, un modelo entrenado y evaluado únicamente sobre el anillo exterior
del 10 % alcanza **78,3 % de acierto**. La fuga está concentrada: `maize-diseases` recupera el
85,4 % de su acierto desde el marco y `multicrop-disease-maiz` el 72,7 %, frente al 1,0 % de
`cropdg` y `maize-in-field`.

Esa medición es válida sobre partición aleatoria, donde las mismas fuentes están a ambos lados.
Bajo validación por fuente **deja de serlo**: el acierto del anillo supera a la clase mayoritaria
en 1 de 11 pliegues, porque los modelos de anillo colapsan sobre una sola clase entre el 33 % y
el 91 % de las veces y su acierto depende de si esa clase coincide con el prior del retenido.

## Coste asimétrico del error

Un falso negativo sobre un patógeno crítico y un falso positivo sobre una hoja sana no tienen el
mismo impacto. El primero arrastra pérdida de cosecha y propagación comunitaria; el segundo,
gasto en fungicida y contaminación de suelos. La tasa de falsos negativos por clase y su lectura
agronómica están en el informe de equidad del pipeline, y no dependen de la partición.

## Limitaciones

Todas las cifras de esta página se miden sobre la partición estándar, que comparte las catorce
fuentes entre entrenamiento y prueba. Describen el comportamiento del modelo sobre datasets que
ya vio. La generalización a una fuente nueva está en
[Evaluación rigurosa](/es/resultados/evaluacion).

Una sola semilla, sin desviación estándar. Los subgrupos de procedencia con soporte de decenas
de imágenes —`maize_deficiency_scanner_roboflow` con 32, `maize_nutrient` con 49— tienen
intervalos amplios que no se reportan.

El DIR por procedencia se calcula sobre accuracy, que en cuatro fuentes de una sola clase vale
1,0000 por construcción. Ese 0,8438 es optimista.
