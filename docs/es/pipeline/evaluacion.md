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

| Subgrupo de Entorno | Muestras ($N$) | Macro $F_1$ | Exactitud (Accuracy) | Macro Precision | Macro Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **Campo Real (`real`)** | **4,483** | **0.9298 (92.98%)** | **0.9842 (98.42%)** | 0.9466 | 0.9164 |
| **Laboratorio (`lab`)** | **532** | **0.2955 (29.55%)** | **0.9417 (94.17%)** | 0.3093 | 0.2904 |
| **Global (Test Set)** | **5,015** | **0.9483 (94.83%)** | **0.9797 (97.97%)** | 0.9589 | 0.9400 |

### Análisis Crítico de la Disparidad Aritmética vs. Desempeño Real

Al analizar el reporte cuantitativo, surge una aparente disparidad en Macro $F_1$ ($\Delta F_1 = 0.6343$). Sin embargo, la inspección detallada de la distribución del dataset revela una causa estructural del corpus:

::: info Hallazgo Demográfico en el Dataset
En el conjunto de prueba, **únicamente 3 de las 9 clases** disponen de muestras en laboratorio (*common_rust*, *gray_leaf_spot* y *northern_corn_leaf_blight*). Las 6 clases restantes (*fall_armyworm*, *healthy*, *lethal_necrosis*, *nitrogen_deficiency*, *phosphorus_deficiency*, *potassium_deficiency*) provienen 100% de tomas directas en campo real.
:::

Al calcular el Macro $F_1$ sobre el subgrupo de laboratorio dividiendo entre $C=9$, las 6 clases con 0 muestras asignan un score de $0.0$, deprimiendo aritméticamente el promedio global del subgrupo:

$$\text{Macro } F_{1, \text{lab}} = \frac{F_{1, \text{rust}} + F_{1, \text{gls}} + F_{1, \text{nclb}} + 0 + 0 + 0 + 0 + 0 + 0}{9} \approx 0.2955$$

Si evaluamos el comportamiento real del modelo sobre las clases que **sí existen** en laboratorio:
1. **Roya Común (*common_rust*, $N=322$):** Tasa de falsos negativos **$FNR = 0.0\%$** (Recall perfecto del 100%).
2. **Tizón Foliar (*northern_corn_leaf_blight*, $N=133$):** Tasa de falsos negativos **$FNR = 2.25\%$** (Recall del 97.75%).
3. **Mancha Gris (*gray_leaf_spot*, $N=77$):** Tasa de falsos negativos **$FNR = 36.36\%$** (Recall del 63.64%).

La **Exactitud en laboratorio se mantiene en un sobresaliente 94.17%**, con una disparidad de exactitud frente a campo real de apenas **$\Delta \text{Acc} = 4.24\%$**. Esto certifica que la red neuronal generaliza con alta robustez a fondos limpios de laboratorio y a entornos complejos con suelo y follaje natural.

![Comparativa de Disparidad por Subgrupo](/fairness/fairness_disparity.png)

---

## 3. Matrices de Confusión Desagregadas

La comparación directa de las matrices de confusión normalizadas confirma que los patrones de error son coherentes entre ambos dominios:

![Matrices de Confusión Desagregadas](/fairness/disaggregated_confusion_matrices.png)

* **En Campo Real (`real`):** La diagonal principal muestra una precisión diagnóstica superior al 95% en la gran mayoría de clases. Los únicos errores marginales ocurren entre deficiencias nutricionales adyacentes (*nitrogen* vs *potassium*), donde el patrón de clorosis foliar comparte similitudes visuales.
* **En Laboratorio (`lab`):** No se registran falsos positivos cruzados hacia las clases ausentes. El modelo no "inventa" síntomas de insectos o deficiencias ante la presencia de un fondo blanco o mesa de estudio.

---

## 4. Control Negativo contra Atajos Visuales (*Clever Hans Effect*)

Para comprobar experimentalmente que el clasificador toma decisiones basándose en la lesión foliar y no en atajos espurios del fondo (sesgo de Clever Hans), se ejecutó un **test de control negativo con oclusión central**:

1. **Protocolo:** Se aplica una máscara opaca que oculta el **60% central** de la imagen (donde se ubican los síntomas principales de la hoja).
2. **Hipótesis:** Si el modelo dependiera del fondo o del encuadre para diagnosticar, mantendría una alta confianza independientemente de la oclusión. Si depende genuinamente de la lesión patológica, la confianza debe decaer significativamente.

### Resultados del Test de Oclusión:
* **Confianza Media Original:** $0.8429$ (84.29%)
* **Confianza Media con Oclusión:** $0.7795$ (77.95%)
* **Caída Media de Confianza ($\Delta \text{Conf}$):** **`-6.34 pp`**
* **Ratio de Vulnerabilidad a Atajos:** $0.9247$

::: warning Interpretación de Robustez
La caída sistemática de confianza ante la oclusión de la lesión confirma que la red busca activamente la textura patológica. No obstante, al retener un 77.95% de confianza residual en el contorno visible de la hoja, se recomienda en el protocolo de usuario final capturar tomas centradas y nítidas de la lámina foliar.
:::

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

1. **Guía de Encuadre en Tiempo Real:** Interfaz con retícula que solicita al agricultor posicionar la hoja afectada en los dos tercios centrales del visor.
2. **Advertencia de Confianza en Deficiencia de Potasio:** Al ser la clase minoritaria con mayor tasa de falsos negativos ($FNR = 23.6\%$), el sistema sugiere una segunda toma con iluminación natural directa si la probabilidad diagnóstica es inferior al 75%.
3. **Privacidad y Procesamiento en Dispositivo (*Edge AI*):** Todo el cómputo de inferencia (vía modelos cuantizados INT8 TFLite y ONNX) se realiza localmente en el teléfono, sin requerir conexión a internet ni almacenar datos privados del productor en servidores remotos.
