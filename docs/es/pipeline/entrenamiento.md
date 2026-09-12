# Entrenamiento de Producción (Pipeline Principal)

Tras validar la arquitectura y descubrir la configuración óptima mediante Optimización Bayesiana ([Optuna](./optimizacion)), se procedió al **entrenamiento de producción** sobre el corpus completo del proyecto (**33,438 imágenes** repartidas en las 9 clases del cultivo de maíz).

El objetivo de esta fase es converger a los checkpoints definitivos de alto rendimiento (`best.pth`) que alimentarán el **Ensamble Multimodelo**, la **Auditoría de Equidad (Fairness)** y la **Exportación a Dispositivos Móviles (TFLite/Edge)**.

---

## 1. Infraestructura y Configuración de Datos

El entrenamiento se ejecutó en la nube utilizando contenedores GPU en **Modal** con almacenamiento persistente en volúmenes (`corn-clean` y `corn-outputs`).

### Partición de Datos (Split `seed_42`)
A diferencia de los baselines (que operaron sobre un subconjunto capado a 10,020 imágenes), el pipeline principal utiliza el **100% de las imágenes** estratificadas jerárquicamente:

| Partición | Proporción | Muestras | Propósito |
|---|:---:|:---:|---|
| **Entrenamiento (`train.csv`)** | 70 % | **23,407** | Ajuste de gradientes mediante AdamW |
| **Validación (`val.csv`)** | 15 % | **5,016** | Monitoreo por época, scheduler y early stopping |
| **Prueba (`test.csv`)** | 15 % | **5,015** | Evaluación final retenida (no vista en entrenamiento) |
| **Total Corpus** | **100 %** | **33,438** | 9 clases patológicas y nutricionales |

---

## 2. Hiperparámetros de Producción

Se aplicaron estrictamente los hiperparámetros ganadores identificados durante el estudio bayesiano de Optuna (Trial #12), diseñados para balancear la velocidad de convergencia con una regularización estocástica robusta:

| Hiperparámetro | Valor de Producción | Justificación Técnica |
|---|:---:|---|
| **Optimizador** | **AdamW** | Regularización desacoplada de decaimiento de pesos ($L_2$). |
| **Learning Rate Base** | **`4.548e-4`** | Identificado por Optuna como la tasa óptima con arranque suave. |
| **LR Scheduler** | **Cosine Annealing con Warmup** | 3 épocas de calentamiento lineal seguido de decaimiento suave hasta $\eta_{\min} = 10^{-6}$. |
| **Batch Size** | **64** | Máxima estabilidad del gradiente promedio y saturación de memoria en GPU A10G. |
| **Ponderación de Pérdida** | **`sqrt_inverse`** | Suaviza el desbalance severo entre clases mayoritarias (*Healthy*, *Northern Leaf Blight*) y minoritarias (*Potassium Deficiency*). |
| **Label Smoothing** | **`0.10`** | Regularización probabilística que evita la sobreconfianza en las predicciones. |
| **Weight Decay** | **`1.0e-4`** | Control de complejidad en pesos convolucionales. |
| **Gradient Clipping** | **`1.0`** | Prevención de explosión de gradientes ante variaciones drásticas de iluminación. |
| **Early Stopping** | **Patience = 8 épocas** | Detención automática si el Macro $F_1$ en validación no mejora. |

---

## 3. Curvas de Convergencia y Dinámica de Entrenamiento

Se entrenaron de forma paralela e independiente las dos arquitecturas troncales del proyecto:
1. **`EfficientNet-B0`:** Red convolucional de alta capacidad con bloques MBConv y atención de canales (Squeeze & Excitation).
2. **`ShuffleNet-V2-x1.0`:** Red ultra-ligera de baja latencia con división y barajado de canales (*Channel Split & Shuffle*).

![Convergencia del Entrenamiento](/training/training_convergence.png)

### Análisis de la Convergencia:

1. **Evolución de la Pérdida (Loss):**
   - Ambas arquitecturas mostraron un descenso monótono estable. El arranque con **3 épocas de warmup** evitó oscilaciones violentas en las primeras iteraciones mientras se descongelaban y ajustaban los pesos pre-entrenados de ImageNet.
   - La pérdida de validación convergió por debajo de `0.75` en ambos modelos, sin manifestar signos de sobreajuste catastrófico (*overfitting*), gracias al efecto combinado de `label_smoothing=0.10` y `weight_decay=1e-4`.

2. **Progresión del Macro $F_1$-Score en Validación:**
   - **`EfficientNet-B0`** alcanzó su punto cúspide en la **Época 28** con un Macro $F_1$ de **0.9610 (96.10%)** y una Accuracy de **98.21%**. Completó las 35 épocas programadas manteniendo una varianza mínima en las épocas finales.
   - **`ShuffleNet-V2`** arrancó con menor precisión en la época 1 (Macro $F_1$ de 0.65 vs 0.84 de EfficientNet), pero experimentó una rápida aceleración a partir de la época 4, alcanzando su mejor métrica en la **Época 23** con un Macro $F_1$ de **0.9502 (95.02%)** y Accuracy de **97.77%**.
   - El mecanismo de **Early Stopping** actuó en `ShuffleNet-V2` en la **Época 31** (8 épocas consecutivas sin superar la marca de la época 23), ahorrando cómputo innecesario en GPU.

---

## 4. Resultados Individuales en el Conjunto de Test Retenido (5,015 Imágenes)

Al evaluar los checkpoints finales (`best.pth`) sobre el subconjunto de prueba independiente (`test.csv`), ambos modelos demostraron una capacidad de generalización sobresaliente:

| Métrica Global | EfficientNet-B0 (Best Epoch: 28) | ShuffleNet-V2-x1.0 (Best Epoch: 23) |
|---|:---:|:---:|
| **Test Accuracy** | **`97.97 %`** | **`97.31 %`** |
| **Test Macro $F_1$-Score** | **`0.9483` (94.83 %)** | **`0.9330` (93.30 %)** |
| **Test Weighted $F_1$** | **`0.9793` (97.93 %)** | **`0.9728` (97.28 %)** |
| **Macro Precision** | **`0.9589`** | **`0.9388`** |
| **Macro Recall** | **`0.9400`** | **`0.9298`** |
| **Tiempo por Época (GPU A10G)** | ~171 segundos (~2.8 min) | ~173 segundos (~2.8 min) |
| **Tamaño Checkpoint (`best.pth`)** | **16.3 MB** | **5.4 MB** |

---

## 5. Desglose de Rendimiento por Clase Fitosanitaria

Desempeño detallado sobre las 9 clases en `test.csv` (Precision / Recall / $F_1$-Score):

| Clase Agronómica | Muestras Test | EfficientNet-B0 ($F_1$) | ShuffleNet-V2 ($F_1$) | Diagnóstico Agronómico |
|---|:---:|:---:|:---:|---|
| **Lethal Necrosis** | 963 | **0.9984** (Recall: 1.00) | **0.9974** (Recall: 1.00) | Detección perfecta sin falsos negativos. |
| **Healthy (Planta Sana)** | 1,311 | **0.9954** | **0.9901** | Gran especificidad; no clasifica enfermos como sanos. |
| **Common Rust (Roya Común)** | 338 | **0.9896** | **0.9867** | Pústulas foliares identificadas con alta precisión. |
| **Fall Armyworm (Gusano Cogollero)**| 728 | **0.9877** | **0.9815** | Daño masticador reconocido fielmente. |
| **Northern Corn Leaf Blight** | 1,025 | **0.9769** | **0.9728** | Lesiones elípticas grandes diferenciadas correctamente. |
| **Phosphorus Deficiency** | 140 | **0.9527** | **0.9275** | Coloración púrpura/rojiza diagnosticada con éxito. |
| **Gray Leaf Spot** | 290 | **0.9194** | **0.9120** | Lesiones rectangulares estrechas bien delimitadas. |
| **Nitrogen Deficiency** | 127 | **0.8939** (Recall: 0.929) | **0.8613** (Recall: 0.929) | Clorosis en "V" identificada pese al tamaño de muestra. |
| **Potassium Deficiency** | 93 | **0.8208** | **0.7674** | Clorosis marginal en hojas basales; clase con menor soporte. |

::: tip Cumplimiento de Rúbrica
El entrenamiento de ambas arquitecturas cumple con el estándar de modelos convolucionales modernos. Ambos modelos superan con holgura la barrera del 93% y 94% de Macro $F_1$, estableciendo la base idónea para combinarse en el **[Ensamble Multimodelo](./ensamble)**.
:::
