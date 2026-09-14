# Evaluación Rigurosa y Auditoría de Equidad

Celebrar un 98 % de exactitud en un cuaderno de entrenamiento es fácil. Lo difícil es verificar si ese porcentaje se sostiene cuando desarmamos los datos, cuando evaluamos subgrupos por separado y cuando auditamos si el modelo está mirando la enfermedad real o apoyándose en trampas visuales del fondo.

La fase de evaluación del pipeline principal sometió a los modelos a un protocolo exhaustivo sobre el conjunto de prueba independiente (**5,015 imágenes retenidas** que nunca participaron en el ajuste de pesos ni en la búsqueda de hiperparámetros).

---

## Desempeño global y equidad entre entornos

La evaluación comenzó comparando cómo responde el modelo frente a dos tipos de fotografía muy distintos: las fotos tomadas en **campo real** (4,483 imágenes) frente a las tomadas en bancos de trabajo o mesas de **laboratorio** (532 imágenes).

| Subgrupo de Entorno | Muestras ($N$) | Exactitud (Accuracy) | Macro $F_1$ sobre clases evaluables |
|---|:---:|:---:|:---:|
| **Campo Real (`real`)** | 4,483 | **98.04 %** | **0.9215** |
| **Laboratorio (`lab`)** | 532 | **96.80 %** | **0.9423** |
| **Conjunto Global de Prueba** | 5,015 | **97.91 %** | **0.9468** |

*(Cifras correspondientes a `EfficientNet-Lite0`, modelo desplegado).*

![Comparativa de Disparidad por Subgrupo](/fairness/fairness_disparity.png)

A primera vista, si se calcula un promedio ciego entre todas las clases para el subgrupo de laboratorio, el número parecería desplomarse. Sin embargo, al auditar los datos encontramos una razón estructural evidente: **en los datasets públicos disponibles, solo 3 de las 9 clases tienen fotos de laboratorio** (*common_rust*, *gray_leaf_spot* y *northern_corn_leaf_blight*). Las otras seis clases provienen exclusivamente de tomas directas en campo.

Ese desequilibrio hace que el macro $F_1$ de laboratorio promedie sobre tres clases y el de campo real sobre nueve: el cociente entre ambos no mide disparidad, sino la distinta composición de cada promedio. La comparación válida restringe los dos subgrupos a las tres clases compartidas, y ahí la disparidad es mínima ($\Delta \text{Acc} = 0.97 \%$), con índices de impacto dispar de $DIR = 0.9901$ en exactitud y $DIR = 0.9604$ en macro $F_1$, ambos con holgura sobre el umbral regulatorio del 80 %.

![Matrices de Confusión Desagregadas](/fairness/disaggregated_confusion_matrices.png)

Las matrices de confusión desglosadas demuestran que el modelo no inventa diagnósticos cruzados: en laboratorio no predice síntomas de insectos o deficiencias que solo existen en fotos de campo.

---

## La prueba del contexto: ¿qué tanto pesa el fondo?

Una de las auditorías más reveladoras del proyecto fue someter al modelo a una prueba de ablación espacial (un chequeo estilo *Clever Hans*) para saber qué tanto depende de la parte central de la hoja frente al contorno y el fondo.

Tapamos partes de la foto con máscaras neutras y medimos cómo reaccionaba el modelo:

| Condición evaluada | Qué ve el modelo | Área visible | Exactitud ($Acc$) | Retención de confianza |
|---|---|:---:|:---:|:---:|
| **Imagen original completa** | Hoja y fondo completos | 100.0 % | **97.91 %** | 100.0 % |
| **Oclusión central** | **Anillo periférico** (centro tapado) | 63.7 % | **79.46 %** | 69.16 % |
| **Oclusión periférica** | **Caja central** (fondo y bordes tapados) | 36.3 % | **42.25 %** | 42.66 % |
| *Línea base de azar* | *Adivinar siempre la clase más frecuente* | 0.0 % | *26.14 %* | — |

La máscara es un rectángulo fijo que abarca del 20 % al 80 % de cada eje, así que tapar el centro oculta el 36.3 % del área y tapar la periferia oculta el 63.7 % restante.

El resultado invita a la humildad científica: **con el centro de la hoja tapado y dejando visible solo el anillo periférico, el modelo acierta el 79.5 % de las veces**, muy por encima del 26.1 % que daría el azar. Si en cambio le mostramos solo la caja central tapando todo el borde, la precisión cae al 42.3 %.

Las dos condiciones **no están igualadas en área** —la que deja el anillo muestra 1.75 veces más píxeles que la que deja la caja central—, de modo que la distancia entre ambas cifras no es atribuible únicamente a la región observada. Lo que sí queda establecido es la comparación de cada condición contra el azar.

Esto confirma que las redes convolucionales son sensibles al contexto general de la escena (la iluminación ambiental, el color de la tierra o el encuadre de la cámara) y no únicamente a la textura de la lesión. 

Probamos entrenar un modelo recortando el fondo con un segmentador automático a negro, pero el rendimiento cayó a un Macro F1 de 0.7191: borrar el fondo de golpe elimina también referencias visuales legítimas que el modelo necesita. Por eso, en lugar de amputar el fondo con algoritmos pesados, la solución práctica adoptada en la aplicación móvil fue incorporar el marco guía en la cámara, orientando al productor a llenar el cuadro con la hoja.

---

## Auditoría visual con Grad-CAM

Para corroborar visualmente dónde se concentra la atención del modelo, generamos mapas de activación sobre cientos de imágenes de prueba:

![Panel de Auditoría Visual Grad-CAM](/fairness/gradcam_samples.png)

En fotos de campo real, Grad-CAM confirma que las activaciones más intensas coinciden con las lesiones foliares: las pústulas alargadas de roya, los márgenes necróticos y las quemaduras típicas. En laboratorio, la atención se enfoca en el tejido foliar cortado. 

Esta coherencia cualitativa, sumada al detector OOD por distancia de Mahalanobis, respalda el despliegue del clasificador como una herramienta de apoyo confiable para la agricultura familiar.
