# Evaluación Rigurosa y Auditoría de Equidad (Fairness Report)

La fase de evaluación del pipeline principal valida los modelos sobre el subconjunto de prueba independiente (**`test.csv` con 5,015 imágenes** retenidas que no participaron en el entrenamiento ni en la búsqueda de hiperparámetros).

El protocolo asegura la reproducibilidad científica y atiende al **Criterio 4 de la Rúbrica de la Etapa 2 (Análisis de Sesgos y Ética)**, evaluando la capacidad diagnóstica global, la equidad de desempeño entre entornos fotográficos (*Laboratorio* vs *Campo Real*), la mitigación de atajos visuales (*Clever Hans Effect*) y la explicabilidad mediante Grad-CAM.

---

::: tip Generalización a fuentes no vistas
Las cifras de esta página se miden sobre la partición estándar, que reparte las catorce fuentes del corpus entre entrenamiento y prueba. El rendimiento del mismo modelo sobre una fuente que no vio entrenando es **0.6026 ± 0.1240**, medido con validación cruzada agrupada por procedencia en [Evaluación rigurosa y métricas finales](/es/resultados/evaluacion).
:::

## 1. Métricas Primarias y Rendimiento Global en Test

Dado el desbalance natural de patologías vegetales en el dataset, la métrica primaria oficial es el **Macro $F_1$-Score**:

$$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C F_{1, c}$$

Esta métrica asigna idéntico peso a todas las clases ($C=9$), impidiendo que el alto rendimiento en clases mayoritarias (*Healthy*, *Northern Corn Leaf Blight*) oculte deficiencias en clases minoritarias (*Potassium Deficiency*).

### Tabla Comparativa de Rendimiento Final (5,015 Muestras de Test):

| Modelo / Ensamble | Macro $F_1$ | Accuracy | Macro Precision | Macro Recall | Weighted $F_1$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **[ShuffleNet-V2-x1.0](./entrenamiento)** | 0.9330 | 0.9731 | 0.9388 | 0.9298 | 0.9728 |
| **[EfficientNet-Lite0](./entrenamiento)** (desplegada) | 0.9468 | 0.9791 | 0.9533 | 0.9413 | — |
| **[EfficientNet-B0](./entrenamiento)** | 0.9483 | 0.9797 | 0.9589 | 0.9400 | 0.9793 |
| **[Soft Voting Ensemble](./ensamble)** 🏆 | **`0.9567`** | **`0.9829`** | **`0.9642`** | **`0.9506`** | — |

::: tip Conclusión de Desempeño
El **Soft Voting Ensemble** supera la barrera del **95% en Macro $F_1$** y alcanza un **98% de exactitud diagnóstica global**. La ganancia sobre el mejor individual es de **+0.84 pp**, y está concentrada: el ensamble mejora en 4 de las 14 fuentes del corpus y empeora en 5. El desglose está en [Modelos avanzados y ensamble](/es/resultados/ensamble).

El modelo desplegado en la aplicación móvil es `EfficientNet-Lite0`, no el ensamble: la inferencia en dispositivo ejecuta un único modelo.
:::

---

## 2. Auditoría de Equidad y Análisis Desagregado por Entorno

Un riesgo ético común en visión por computadora aplicada al agro es que el modelo aprenda correlaciones espurias asociadas a fondos artificiales (como mesas de laboratorio o iluminación de estudio) en vez de aprender la patología de la hoja.

Para auditar este sesgo algorítmico, el conjunto de prueba se evaluó de forma desagregada entre los subgrupos **`real`** (condiciones naturales de cultivo) y **`lab`** (muestras en banco de trabajo).

### Métricas Desagregadas por Subgrupo:

| Subgrupo de Entorno | Muestras ($N$) | Macro $F_1$ (evaluable) | Macro $F_1$ (9 clases) | Exactitud (Accuracy) | Macro Precision | Macro Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Campo Real (`real`)** | **4,483** | **0.9215 (92.15%)** | **0.9215 (92.15%)** | **0.9804 (98.04%)** | — | — |
| **Laboratorio (`lab`)** | **532** | **0.9423 (94.23%)\*** | **0.3141 (31.41%)\*** | **0.9680 (96.80%)** | — | — |

Estas cifras corresponden a `EfficientNet-Lite0`, la arquitectura desplegada. La disparidad de Macro $F_1$ evaluable es **0.0209** y el DIR **0.9778**, que cumple la regla del 80 %.
| **Global (Test Set)** | **5,015** | **0.9468 (94.68%)** | **0.9468 (94.68%)** | **0.9791 (97.91%)** | 0.9533 | 0.9413 |

### Análisis Crítico de la Disparidad Aritmética vs. Desempeño Real

Al analizar el reporte cuantitativo original, surge una aparente disparidad en Macro $F_1$ ($\Delta F_1 = 0.6074$). Sin embargo, la inspección detallada de la distribución del dataset revela una causa estructural del corpus:

::: info Hallazgo Demográfico en el Dataset
En el conjunto de prueba, **únicamente 3 de las 9 clases** disponen de muestras en laboratorio (*common_rust*, *gray_leaf_spot* y *northern_corn_leaf_blight*). Las 6 clases restantes (*fall_armyworm*, *healthy*, *lethal_necrosis*, *nitrogen_deficiency*, *phosphorus_deficiency*, *potassium_deficiency*) provienen 100% de tomas directas en campo real.
:::

Al calcular el Macro $F_1$ no ajustado sobre el subgrupo de laboratorio dividiendo entre $C=9$, las 6 clases con 0 muestras asignan un score de $0.0$, deprimiendo aritméticamente el promedio global del subgrupo:

$$\text{Macro } F_{1, \text{lab, no ajustado}} = \frac{F_{1, \text{rust}} + F_{1, \text{gls}} + F_{1, \text{nclb}} + 0 + 0 + 0 + 0 + 0 + 0}{9} \approx 0.3141$$

Si evaluamos el comportamiento real del modelo sobre las clases que **sí existen** en laboratorio (clases evaluables con soporte $N > 0$):
1. **Roya Común (*common_rust*, $N=322$):** Tasa de falsos negativos **$FNR = 0.0\%$** (Recall perfecto del 100%).
2. **Tizón Foliar (*northern_corn_leaf_blight*, $N=133$):** Tasa de falsos negativos **$FNR = 3.01\%$** (Recall del 96.99%).
3. **Mancha Gris (*gray_leaf_spot*, $N=77$):** Tasa de falsos negativos **$FNR = 15.58\%$** (Recall del 84.42%), la peor de las tres clases presentes en laboratorio.

El **Macro $F_1$ evaluable en laboratorio alcanza 0.9423** y la **Exactitud se sitúa en 96.80%**, con una disparidad de exactitud frente a campo real de $\Delta 	ext{Acc} = 1.23\%$ y un $DIR_{F_1} = 0.9778 \ge 0.80$ sobre clases evaluables.

Conviene notar que en este modelo el subgrupo de laboratorio puntúa **por encima** del de campo real en Macro $F_1$ evaluable (0.9423 frente a 0.9215), y por debajo en exactitud (0.9680 frente a 0.9804). No es contradictorio: laboratorio contiene tres clases y campo real las nueve, así que el macro promedia sobre conjuntos distintos. Comparar ambos subgrupos con una sola cifra es, en rigor, comparar dos problemas de clasificación diferentes.

Como se analiza en la sección 4, la ausencia de 6 clases en laboratorio y la sensibilidad a regiones perimetrales aconsejan interpretar estas métricas con prudencia.

![Comparativa de Disparidad por Subgrupo](/fairness/fairness_disparity.png)

---

### Desagregación por procedencia

El entorno tiene dos categorías; la procedencia tiene **catorce**, y es el eje con más varianza del corpus.

![Disparidad por procedencia](/fairness/fairness_disparity_por_fuente.png)

Cuatro fuentes puntúan accuracy 1.0000 y las cuatro contienen **una sola clase**: un predictor constante también acierta el 100 % en ellas. La tabla completa, con el exceso de cada fuente sobre ese control nulo, está en [Análisis de sesgos y ética](/es/resultados/equidad).

---

## 3. Matrices de Confusión Desagregadas

La comparación directa de las matrices de confusión normalizadas confirma que los patrones de error son coherentes entre ambos dominios:

![Matrices de Confusión Desagregadas](/fairness/disaggregated_confusion_matrices.png)

* **En Campo Real (`real`):** La diagonal principal muestra una precisión diagnóstica superior al 95% en la gran mayoría de clases. Los únicos errores marginales ocurren entre deficiencias nutricionales adyacentes (*nitrogen* vs *potassium*), donde el patrón de clorosis foliar comparte similitudes visuales.
* **En Laboratorio (`lab`):** No se registran falsos positivos cruzados hacia las clases ausentes. El modelo no "inventa" síntomas de insectos o deficiencias ante la presencia de un fondo blanco o mesa de estudio.

---

## 4. Control Negativo de Sensibilidad Espacial (*Clever Hans Check*) y Control Inverso

Para auditar si la red convolucional depende de regiones centrales de la lámina frente a información periférica (que en algunas tomas incluye suelo, mesa o bordes foliares), se diseñó una **auditoría de ablación dual** sobre el conjunto de prueba independiente ($N = 5,015$ muestras):

1. **Control Negativo (Oclusión Central 60%):** Se aplica una máscara rectangular que cubre del 20% al 80% de cada dimensión espacial. La máscara inyecta **negro real en espacio normalizado ImageNet** (`(-mean)/std`) y mide la retención de probabilidad sobre la **misma clase predicha originalmente**.
2. **Control Inverso (Oclusión Periférica 40% - Solo Región Central):** Se oculta la periferia y se deja visible el 60% central con el objetivo de evaluar si la región central preserva capacidad diagnóstica por sí sola.

> [!NOTE]
> **Delimitación Metodológica:** La máscara empleada es geométrica rectangular, no una segmentación biológica fina de la patología. Describe patrones de sensibilidad a zonas de la imagen y no causalidad estricta sobre el fondo exterior.

### Resultados de la Auditoría de Sensibilidad Dual:

| Condición de Inferencia | Confianza Media | Exactitud ($Acc$) | $\Delta Acc$ | Retención de Confianza | *Flip Rate* |
|---|:---:|:---:|:---:|:---:|:---:|
| **Inferencia Original (Test Base)** | **84.00%** | **97.91%** | — | 100.0% | 0.0% |
| **Oclusión Central 60%** (queda la periferia) | **58.09%** | **79.46%** | `-18.44 pp` | **`69.16%`** | 19.51% |
| **Control Inverso** (queda sólo el centro) | **35.84%** | **42.25%** | **`-55.65 pp`** | 42.66% | **57.43%** |
| **Control nulo** (clase mayoritaria constante) | — | **26.14%** | — | — | — |

La última fila es la referencia sin la cual las anteriores no se pueden interpretar: un modelo degenerado que emitiera siempre la clase mayoritaria alcanzaría 26.14% sin extraer nada de la imagen.

::: warning Alerta Técnica: Sensibilidad Relevante a la Periferia de la Imagen
**El fondo por sí solo predice mejor que la hoja por sí sola.** Con el 60% central tapado —queda únicamente la periferia— el modelo acierta el **79.46%**, muy por encima del control nulo de 26.14%, y retiene el 69.16% de su confianza. Con la periferia tapada y la lesión visible cae al **42.25%**, con un 57.43% de predicciones correctas que mutan a error.

La caída al ocultar la periferia (55.65 puntos) triplica la caída al ocultar el centro (18.44 puntos). 

**Directriz Técnica y de Despliegue:** La máscara rectangular no aísla el fondo del tejido foliar perimetral, así que mide sensibilidad espacial y no causalidad sobre el fondo. Aun con esa acotación, la magnitud aconseja no leer el 97.91% de exactitud global como capacidad de diagnóstico foliar pura.

La mitigación aparente —segmentar antes de clasificar— **se midió y cuesta**: la corrida `20260907_163546`, entrenada y evaluada sobre imágenes segmentadas, rinde 0.7191 frente a 0.9468. Eliminar el fondo elimina también señal que el modelo estaba usando con éxito dentro de este corpus.
:::

### Evaluación de Mitigaciones Necesarias antes del Despliegue

Para neutralizar esta vulnerabilidad antes de considerar un despliegue operativo en los campos agrícolas de El Salvador (MAG / CENTA), el pipeline requiere la implementación de las siguientes contramedidas:

1. **Desacople de fondo vía segmentación foliar — medido, con coste:**
   Integrar `maize-doctor-segmenter` como etapa frontal y poner en negro el 100% de los píxeles de fondo. **Medido: el Macro $F_1$ baja de 0.9468 a 0.7191.** No se descarta como línea de trabajo, pero no es una mitigación gratuita ni está implantada.
2. **Entrenamiento Basado en Parches (*Patch-Based Training*):**
   Reentrenar la red convolucional utilizando parches de alta resolución ($128 \times 128$ o $224 \times 224$) muestreados estrictamente del interior del tejido vegetal enfermo, desacoplando la escala de la hoja y los artefactos de encuadre.
3. **Data augmentation destructivo de entorno — probado sobre el pipeline real:**
   Se midieron dos brazos sobre validación por fuente, con las transformaciones del proyecto: jitter de color agresivo y recorte aleatorio con escala 0.3-1.0. **Ninguno mejoró**, y ambos aumentaron la dependencia del marco respecto de la configuración base. El detalle está en [Procedencia y fuga](/es/provenance/consolidacion).
   * Quedan sin medir el reemplazo sintético de fondos (*Background Swapping*) y la inyección de ruido de sensor con recompresión JPEG.
4. **Filtro Fuera de Distribución (OOD):**
   Utilizar la distancia de Mahalanobis para rechazar imágenes donde las características de fondo interfieran con la distribución biológica aprendida.

---

## 5. Auditoría Visual de Explicabilidad con Grad-CAM

Para validar cualitativamente el foco de atención, se generaron mapas de activación de la última capa convolucional de `EfficientNet-Lite0` mediante **Grad-CAM** (*Gradient-weighted Class Activation Mapping*):

![Panel de Auditoría Visual Grad-CAM](/fairness/gradcam_samples.png)

### Observaciones del Panel Visual:
1. **Muestras de Campo Real:**
   * La atención se concentra de forma dominante sobre los focos de infección foliar: pústulas elongadas de roya y manchas necróticas.
2. **Muestras de Laboratorio:**
   * La red enfoca la textura clorótica de la hoja cortada.

::: warning El panel visual no contradice ni confirma la ablación
Una lectura cualitativa de mapas de activación **no puede** establecer que la red ignore el fondo: Grad-CAM muestra dónde se concentra el gradiente de la clase predicha, no cuánta información aporta cada región. La medición cuantitativa de la sección 4 es inequívoca en la dirección contraria: **con la hoja tapada y sólo el fondo visible, el modelo acierta el 79.46%**, frente a un control nulo de 26.14%.

Cuando el panel visual y la ablación discrepan, manda la ablación.
:::

---

## 6. Detección Fuera de Distribución (OOD - Mahalanobis)

Como mecanismo ético complementario para prevenir diagnósticos erróneos sobre imágenes no pertenecientes al cultivo de maíz (hojas de frijol, suelo sin vegetación, manos humanas), el sistema implementa un **detector OOD basado en distancia de Mahalanobis**:

$$D_M(x) = \min_{c} \sqrt{(\phi(x) - \mu_c)^T \Sigma^{-1} (\phi(x) - \mu_c)}$$

* **$\phi(x)$:** Vector de características del cuello de botella convolucional (1,280 dimensiones).
* **$\mu_c, \Sigma$:** Centroide por clase y matriz de covarianza empírica acumulada sobre el conjunto de entrenamiento.
* **Umbral Operativo ($\tau_{95\%}$):** Muestras con distancia mayor al percentil 95 son clasificadas como **`Fuera de Distribución / Indeterminada`**, instruyendo al usuario a repetir la toma fotográfica en lugar de emitir un diagnóstico agronómico erróneo.

---

## 7. Directrices Éticas para el Despliegue en Campo

A partir de esta auditoría se derivan las siguientes salvaguardas. Se distingue lo que el sistema **ya hace** de lo que queda **propuesto**:

1. **Desacople de fondo — propuesto, con coste medido.** El test Clever Hans demuestra dependencia del contexto: con el 60 % central ocluido el modelo aún acierta el 79.5 %, frente a un control nulo de 26.1 %. La mitigación natural sería segmentar la lámina foliar antes de clasificar, pero **esa vía se midió y sale cara**: entrenar y evaluar sobre imágenes segmentadas baja el Macro $F_1$ de 0.9468 a **0.7191** (corrida `20260907_163546`). La aplicación móvil **no incorpora segmentador**; empaqueta únicamente el clasificador. Queda como línea abierta, no como salvaguarda implantada.
2. **Advertencia de confianza en deficiencias nutricionales.** Sobre el modelo desplegado, las dos clases con mayor tasa de falsos negativos son `potassium_deficiency` ($FNR = 14.0\%$) y `nitrogen_deficiency` ($FNR = 15.7\%$). Ambas son minoritarias —93 y 127 muestras en prueba— y justifican solicitar una segunda toma cuando la probabilidad diagnóstica sea baja.
3. **Privacidad y Procesamiento en Dispositivo (*Edge AI*):** Todo el cómputo de inferencia (vía modelos cuantizados INT8 TFLite y ONNX) se realiza localmente en el teléfono, sin requerir conexión a internet ni almacenar datos privados del productor en servidores remotos.
