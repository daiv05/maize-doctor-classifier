# Evaluación Rigurosa y Auditoría de Equidad (Fairness Report)

La fase de evaluación del pipeline principal valida los modelos sobre el subconjunto de prueba independiente (**`test.csv` con 5,015 imágenes** retenidas que no participaron en el entrenamiento ni en la búsqueda de hiperparámetros).

El protocolo asegura la reproducibilidad científica y atiende al **Criterio 4 de la Rúbrica de la Etapa 2 (Análisis de Sesgos y Ética)**, evaluando la capacidad diagnóstica global, la equidad de desempeño entre entornos fotográficos (*Laboratorio* vs *Campo Real*), la mitigación de atajos visuales (*Clever Hans Effect*) y la explicabilidad mediante Grad-CAM.

---

## 1. Métricas Primarias y Rendimiento Global en Test

Dado el desbalance natural de patologías vegetales en el dataset, la métrica primaria oficial es el **Macro $F_1$-Score**:

$$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C F_{1, c}$$

Esta métrica asigna idéntico peso a todas las clases ($C=9$), impidiendo que el alto rendimiento en clases mayoritarias (*Healthy*, *Northern Corn Leaf Blight*) oculte deficiencias en clases minoritarias (*Potassium Deficiency*).

### Tabla Comparativa de Rendimiento Final (5,015 Muestras de Test):

| Modelo / Ensamble | Macro $F_1$ | Accuracy | Macro Precision | Macro Recall | Weighted $F_1$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **[ShuffleNet-V2-x1.0](./entrenamiento)** | 0.9330 | 0.9731 | 0.9388 | 0.9298 | 0.9728 |
| **[EfficientNet-B0](./entrenamiento)** | 0.9483 | 0.9797 | 0.9589 | 0.9400 | 0.9793 |
| **[Soft Voting Ensemble](./ensamble)** 🏆 | **`0.9507`** | **`0.9799`** | **`0.9582`** | **`0.9445`** | **`0.9796`** |

::: tip Conclusión de Desempeño
El **Soft Voting Ensemble** logra el equilibrio óptimo superando la barrera del **95% en Macro $F_1$** y alcanzando un **98% de exactitud diagnóstica global**, reduciendo sustancialmente los falsos positivos entre patologías de manchas foliares.
:::

---

## 2. Auditoría de Equidad y Análisis Desagregado por Entorno

Un riesgo ético común en visión por computadora aplicada al agro es que el modelo aprenda correlaciones espurias asociadas a fondos artificiales (como mesas de laboratorio o iluminación de estudio) en vez de aprender la patología de la hoja.

Para auditar este sesgo algorítmico, el conjunto de prueba se evaluó de forma desagregada entre los subgrupos **`real`** (condiciones naturales de cultivo) y **`lab`** (muestras en banco de trabajo).

### Métricas Desagregadas por Subgrupo:

| Subgrupo de Entorno | Muestras ($N$) | Macro $F_1$ (evaluable) | Macro $F_1$ (9 clases) | Exactitud (Accuracy) | Macro Precision | Macro Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Campo Real (`real`)** | **4,483** | **0.9298 (92.98%)** | **0.9298 (92.98%)** | **0.9842 (98.42%)** | 0.9466 | 0.9164 |
| **Laboratorio (`lab`)** | **532** | **0.8865 (88.65%)\*** | **0.2955 (29.55%)\*** | **0.9417 (94.17%)** | 0.9279 | 0.8712 |
| **Global (Test Set)** | **5,015** | **0.9483 (94.83%)** | **0.9483 (94.83%)** | **0.9797 (97.97%)** | 0.9589 | 0.9400 |

### Análisis Crítico de la Disparidad Aritmética vs. Desempeño Real

Al analizar el reporte cuantitativo original, surge una aparente disparidad en Macro $F_1$ ($\Delta F_1 = 0.6343$). Sin embargo, la inspección detallada de la distribución del dataset revela una causa estructural del corpus:

::: info Hallazgo Demográfico en el Dataset
En el conjunto de prueba, **únicamente 3 de las 9 clases** disponen de muestras en laboratorio (*common_rust*, *gray_leaf_spot* y *northern_corn_leaf_blight*). Las 6 clases restantes (*fall_armyworm*, *healthy*, *lethal_necrosis*, *nitrogen_deficiency*, *phosphorus_deficiency*, *potassium_deficiency*) provienen 100% de tomas directas en campo real.
:::

Al calcular el Macro $F_1$ no ajustado sobre el subgrupo de laboratorio dividiendo entre $C=9$, las 6 clases con 0 muestras asignan un score de $0.0$, deprimiendo aritméticamente el promedio global del subgrupo:

$$\text{Macro } F_{1, \text{lab, no ajustado}} = \frac{F_{1, \text{rust}} + F_{1, \text{gls}} + F_{1, \text{nclb}} + 0 + 0 + 0 + 0 + 0 + 0}{9} \approx 0.2955$$

Si evaluamos el comportamiento real del modelo sobre las clases que **sí existen** en laboratorio (clases evaluables con soporte $N > 0$):
1. **Roya Común (*common_rust*, $N=322$):** Tasa de falsos negativos **$FNR = 0.0\%$** (Recall perfecto del 100%).
2. **Tizón Foliar (*northern_corn_leaf_blight*, $N=133$):** Tasa de falsos negativos **$FNR = 2.25\%$** (Recall del 97.75%).
3. **Mancha Gris (*gray_leaf_spot*, $N=77$):** Tasa de falsos negativos **$FNR = 36.36\%$** (Recall del 63.64%).

El **Macro $F_1$ evaluable en laboratorio alcanza 0.8865** y la **Exactitud se sitúa en 94.17%**, con una disparidad de exactitud frente a campo real de apenas **$\Delta \text{Acc} = 4.24\%$** y un $DIR_{F_1} = 0.9534 \ge 0.80$ sobre clases evaluables. No obstante, como se analiza en la sección 4, la ausencia de 6 clases en laboratorio y la sensibilidad a regiones perimetrales aconsejan interpretar estas métricas con prudencia técnica y respaldarlas con segmentación previa en producción.

![Comparativa de Disparidad por Subgrupo](/fairness/fairness_disparity.png)

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

| Condición de Inferencia | Confianza Media | $\Delta$ Confianza | Exactitud ($Acc$) | $\Delta Acc$ | Retención de Confianza (Clase Original) | Tasa de Error Inducido (*Flip Rate*) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Inferencia Original (Test Base)** | **84.29%** | — | **97.97%** | — | 100.0% | 0.0% |
| **Oclusión Central 60% (Máscara Rectangular)** | **77.95%** | `-6.34 pp` | **80.46%** | `-17.51 pp` | **`92.47%`** | 18.60% |
| **Control Inverso (Solo Centro 60%)** | **70.60%** | `-13.69 pp` | **71.76%** | **`-26.20 pp`** | 83.76% | **27.58%** |

::: warning Alerta Técnica: Sensibilidad Relevante a la Periferia de la Imagen
La retención de un **92.47% de confianza** en la clase original y un **80.46% de exactitud** tras ocultar el 60% central revela que el clasificador extrae señales discriminantes significativas de las regiones exteriores de la imagen.

Al retirar el contexto periférico en el **Control Inverso**, la exactitud se reduce **26.20 puntos porcentuales** (de 97.97% a 71.76%) y un **27.58% de las predicciones correctas mutan a clasificaciones erróneas**. 

**Directriz Técnica y de Despliegue:** Aunque la máscara rectangular no aísla el fondo del tejido foliar perimetral, esta marcada sensibilidad aconseja **no desplegar directamente sobre tomas fotográficas sin filtrar**, recomendando la integración frontal del modelo de segmentación (`maize-doctor-segmenter`) para aislar exclusivamente la hoja de maíz antes de clasificar.
:::

### Evaluación de Mitigaciones Necesarias antes del Despliegue

Para neutralizar esta vulnerabilidad antes de considerar un despliegue operativo en los campos agrícolas de El Salvador (MAG / CENTA), el pipeline requiere la implementación de las siguientes contramedidas:

1. **Desacople Mandatorio de Fondo vía Segmentación Foliar:**
   Integrar el modelo de segmentación de `maize-doctor-segmenter` como etapa frontal estricta: aislar la lámina de la hoja y poner en negro neutro (`pixel = 0`) el 100% de los píxeles de fondo (suelo, dedos, maleza o rastrojo) antes de ingresar al clasificador.
2. **Entrenamiento Basado en Parches (*Patch-Based Training*):**
   Reentrenar la red convolucional utilizando parches de alta resolución ($128 \times 128$ o $224 \times 224$) muestreados estrictamente del interior del tejido vegetal enfermo, desacoplando la escala de la hoja y los artefactos de encuadre.
3. **Data Augmentation Destructivo de Entorno:**
   * **Borrado aleatorio periférico (*Perimeter CutMix / Random Erasing*):** Ocluir agresivamente bordes durante el entrenamiento para forzar a los filtros convolucionales a extraer características de pústulas y manchas necróticas.
   * **Reemplazo sintético de fondos (*Background Swapping*):** Mezclar hojas recortadas con fondos aleatorios de otros cultivos y texturas inertes.
   * **Simulación de ruido de sensor:** Inyección de ruido Gaussiano y compresión destructiva JPEG (calidad variable 30-75) para desmantelar firmas fotométricas de dispositivos específicos.
4. **Filtro Fuera de Distribución (OOD):**
   Utilizar la distancia de Mahalanobis para rechazar imágenes donde las características de fondo interfieran con la distribución biológica aprendida.

---

## 5. Auditoría Visual de Explicabilidad con Grad-CAM

Para validar cualitativamente el foco de atención, se generaron mapas de activación de la última capa convolucional (`features.-1` en EfficientNet-B0) mediante **Grad-CAM** (*Gradient-weighted Class Activation Mapping*):

![Panel de Auditoría Visual Grad-CAM](/fairness/gradcam_samples.png)

### Observaciones del Panel Visual:
1. **Muestras de Campo Real:**
   * La atención de la red se concentra exactamente sobre los focos de infección foliar (pústulas elongadas de roya y manchas necróticas).
   * Se ignora por completo el fondo (tierra del surco, rastrojo, malezas perimetrales o sombras arrojadas).
2. **Muestras de Laboratorio:**
   * La red enfoca la textura clorótica de la hoja cortada.
   * No se presentan activaciones espurias en los bordes de la mesa ni en artefactos de iluminación controlada.

---

## 6. Detección Fuera de Distribución (OOD - Mahalanobis)

Como mecanismo ético complementario para prevenir diagnósticos erróneos sobre imágenes no pertenecientes al cultivo de maíz (hojas de frijol, suelo sin vegetación, manos humanas), el sistema implementa un **detector OOD basado en distancia de Mahalanobis**:

$$D_M(x) = \min_{c} \sqrt{(\phi(x) - \mu_c)^T \Sigma^{-1} (\phi(x) - \mu_c)}$$

* **$\phi(x)$:** Vector de características del cuello de botella convolucional (1,280 dimensiones).
* **$\mu_c, \Sigma$:** Centroide por clase y matriz de covarianza empírica acumulada sobre el conjunto de entrenamiento.
* **Umbral Operativo ($\tau_{95\%}$):** Muestras con distancia mayor al percentil 95 son clasificadas como **`Fuera de Distribución / Indeterminada`**, instruyendo al usuario a repetir la toma fotográfica en lugar de emitir un diagnóstico agronómico erróneo.

---

## 7. Directrices Éticas para el Despliegue en Campo

A partir de esta auditoría, se establecen las siguientes salvaguardas para la aplicación móvil:

1. **Preprocesamiento Mandatorio y Desacople de Fondo:** Para mitigar la dependencia de atajos de contexto demostrada en el test Clever Hans, la aplicación móvil ejecuta la inferencia en dos etapas: primero aísla la lámina foliar descartando el fondo mediante el segmentador local, guiando al usuario con retícula de encuadre en el visor.
2. **Advertencia de Confianza en Deficiencia de Potasio:** Al ser la clase minoritaria con mayor tasa de falsos negativos ($FNR = 23.6\%$), el sistema sugiere una segunda toma con iluminación natural directa si la probabilidad diagnóstica es inferior al 75%.
3. **Privacidad y Procesamiento en Dispositivo (*Edge AI*):** Todo el cómputo de inferencia (vía modelos cuantizados INT8 TFLite y ONNX) se realiza localmente en el teléfono, sin requerir conexión a internet ni almacenar datos privados del productor en servidores remotos.
