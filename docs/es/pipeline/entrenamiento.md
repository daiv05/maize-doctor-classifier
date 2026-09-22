# Entrenamiento de Producción (Pipeline Principal)

Se entrenaron tres arquitecturas sobre el corpus completo del proyecto. La materialización vigente de `seed_42` contiene **33,429 imágenes** repartidas en las 9 clases del cultivo de maíz. **`EfficientNet-Lite0` es la arquitectura desplegada**: es la que se exporta a TFLite y la que ejecuta la aplicación móvil. `EfficientNet-B0` y `ShuffleNet-V2-x1.0` se entrenan como comparación y como miembros del ensamble.

El objetivo de esta fase es converger a los checkpoints definitivos de alto rendimiento (`best.pth`) que alimentarán el **Ensamble Multimodelo**, la **Auditoría de Equidad (Fairness)** y la **Exportación a Dispositivos Móviles (TFLite/Edge)**.

Cada entrenamiento nuevo genera además un `summary.json` v1 y hashes verificables del split, la configuración y el checkpoint. El formato, la carga segura y la migración de runs históricos se describen en [Contratos versionados de runs y artefactos](/es/pipeline/contratos-runs).

---

## 1. Infraestructura y Configuración de Datos

El entrenamiento se ejecutó en la nube utilizando contenedores GPU en **Modal** con almacenamiento persistente en volúmenes (`corn-clean` y `corn-outputs`).

### Partición de Datos (Split `seed_42`)
A diferencia de los baselines (que operaron sobre un subconjunto capado a 10,020 imágenes), el pipeline principal utiliza el **100% de las imágenes** estratificadas jerárquicamente:

| Partición | Proporción | Muestras | Propósito |
|---|:---:|:---:|---|
| **Entrenamiento (`train.csv`)** | 70.0 % | **23,400** | Ajuste de gradientes mediante AdamW |
| **Validación (`val.csv`)** | 15.0 % | **5,014** | Monitoreo por época, scheduler y early stopping |
| **Prueba (`test.csv`)** | 15.0 % | **5,015** | Evaluación final retenida (no vista en entrenamiento) |
| **Total Corpus** | **100 %** | **33,429** | 9 clases patológicas y nutricionales |

Este split es estratificado por `label + environment`; las fuentes se conservan como metadato y pueden aparecer a ambos lados de la partición. Los resultados históricos de producción que aparecen más abajo corresponden a una materialización anterior de 33,433 imágenes y se mantienen para documentar el artefacto que hoy está desplegado. El primer reentrenamiento sobre el split corregido se registra por separado como [candidato `20260921_204608`](/es/resultados/run-20260921-efficientnet-lite0).

---

## 2. Hiperparámetros de Producción

Cada arquitectura usa la configuración con la que obtiene su mejor resultado en prueba, que no es la misma para las tres:

| | `EfficientNet-Lite0` (desplegada) | `EfficientNet-B0` y `ShuffleNet-V2` |
|---|:---:|:---:|
| **Learning Rate** | `1.0e-4` | `4.548e-4` |
| **Batch Size** | 32 | 64 |
| **Corrida** | `20260812_221429` | `20260910_170120` / `20260910_184521` |

`B0` y `ShuffleNet-V2` toman `learning_rate` y `batch_size` del estudio de Optuna; `Lite0` conserva los valores por defecto, porque **la configuración de Optuna la perjudica** (0.9343 frente a 0.9468 en prueba). El análisis está en [Optimización e hiperparámetros](/es/resultados/optimizacion).

Los valores de `warmup_epochs` y `weight_decay` de la tabla siguiente son los del pipeline, no los del Trial #12 —que eran 2 y 1.573e-05—, porque el wrapper de Modal no los propagaba en el momento de lanzar estas corridas. El resto es común a las tres:

| Hiperparámetro | Valor de Producción | Justificación Técnica |
|---|:---:|---|
| **Optimizador** | **AdamW** | Regularización desacoplada de decaimiento de pesos ($L_2$). |
| **Learning Rate Base** | **`4.548e-4`** (`1.0e-4` en Lite0) | Identificado por Optuna sobre `B0`; `Lite0` conserva el valor por defecto. |
| **LR Scheduler** | **Cosine Annealing con Warmup** | 3 épocas de calentamiento lineal seguido de decaimiento suave hasta $\eta_{\min} = 10^{-6}$. |
| **Batch Size** | **64** (**32** en Lite0) | Estabilidad del gradiente promedio y saturación de memoria en GPU A10G. |
| **Ponderación de Pérdida** | **`sqrt_inverse`** | Suaviza el desbalance severo entre clases mayoritarias (*Healthy*, *Northern Leaf Blight*) y minoritarias (*Potassium Deficiency*). |
| **Label Smoothing** | **`0.10`** | Regularización probabilística que evita la sobreconfianza en las predicciones. |
| **Weight Decay** | **`1.0e-4`** | Control de complejidad en pesos convolucionales. |
| **Gradient Clipping** | **`1.0`** | Prevención de explosión de gradientes ante variaciones drásticas de iluminación. |
| **Early Stopping** | **Patience = 8 épocas** | Detención automática si el Macro $F_1$ en validación no mejora. |

---

## 3. Curvas de Convergencia y Dinámica de Entrenamiento

Se entrenaron de forma independiente tres arquitecturas:
1. **`EfficientNet-B0`:** Red convolucional de alta capacidad con bloques MBConv y atención de canales (Squeeze & Excitation).
2. **`ShuffleNet-V2-x1.0`:** Red ultra-ligera de baja latencia con división y barajado de canales (*Channel Split & Shuffle*).
3. **`EfficientNet-Lite0`:** Variante sin bloques Squeeze & Excitation ni activaciones *swish*, sustituidas por operaciones cuantizables a entero de 8 bits. **Es la arquitectura desplegada.**

La figura siguiente compara las tres bajo **hiperparámetros idénticos** —los valores por defecto del pipeline— para que la curva refleje la arquitectura y no la configuración:

![Convergencia del Entrenamiento](/training/training_convergence.png)

### Análisis de la Convergencia:

1. **Evolución de la Pérdida (Loss):**
   - Ambas arquitecturas mostraron un descenso monótono estable. El arranque con **3 épocas de warmup** evitó oscilaciones violentas en las primeras iteraciones mientras se descongelaban y ajustaban los pesos pre-entrenados de ImageNet.
   - La pérdida de validación convergió por debajo de `0.75` en ambos modelos, sin manifestar signos de sobreajuste catastrófico (*overfitting*), gracias al efecto combinado de `label_smoothing=0.10` y `weight_decay=1e-4`.

2. **Progresión del Macro $F_1$-Score en Validación:**
   - **`EfficientNet-Lite0`** es la de convergencia más lenta y más alta: alcanza su pico en la **Época 35** con Macro $F_1$ de validación **0.9554**, y el early stopping la detiene en la 43. Es también la única de las tres que sigue mejorando más allá de la época 20.
   - **`EfficientNet-B0`** alcanza **0.9551** en la **Época 18** bajo la misma configuración, y se detiene en la 26. Con `learning_rate` 4.548e-4 y lote 64 sube a **0.9610** en la **Época 28** sobre 35 programadas.
   - **`ShuffleNet-V2`** arranca por debajo de las dos EfficientNet y acelera a partir de la época 4, cerrando en **0.9408** en la **Época 18** con la configuración por defecto. Con la configuración de Optuna alcanza **0.9502** en la **Época 23**.
   - El **Early Stopping** con paciencia de 8 épocas detuvo `B0` y `ShuffleNet-V2` en la 26, y `Lite0` en la 43. Esa diferencia de diecisiete épocas es la observación más relevante de la figura: las tres comparten configuración, así que refleja un comportamiento de la arquitectura.

---

## 4. Resultados Individuales en el Conjunto de Test Retenido (5,015 Imágenes)

Al evaluar los checkpoints finales (`best.pth`) sobre el subconjunto de prueba independiente (`test.csv`), ambos modelos demostraron una capacidad de generalización sobresaliente:

| Métrica Global | EfficientNet-Lite0 (desplegada) | EfficientNet-B0 | ShuffleNet-V2-x1.0 |
|---|:---:|:---:|:---:|
| **Corrida** | `20260812_221429` | `20260910_170120` | `20260910_184521` |
| **Mejor época** | 35 | 28 | 23 |
| **Test Accuracy** | **`97.91 %`** | **`97.97 %`** | **`97.31 %`** |
| **Test Macro $F_1$-Score** | **`0.9468`** | **`0.9483`** | **`0.9330`** |
| **Macro Precision** | **`0.9533`** | **`0.9589`** | **`0.9388`** |
| **Macro Recall** | **`0.9413`** | **`0.9400`** | **`0.9298`** |
| **Tamaño Checkpoint (`best.pth`)** | **13.1 MiB** | **15.6 MiB** | **5.0 MiB** |

Las tres cifras de prueba se recomputan de forma exacta desde las predicciones por imagen de cada corrida; el registro está en el [manifiesto de auditoría](/es/resultados/evidencia/).

---

## 5. Desglose de Rendimiento por Clase Fitosanitaria

Desempeño detallado sobre las 9 clases en `test.csv` (Precision / Recall / $F_1$-Score):

| Clase Agronómica | Muestras Test | **Lite0** ($F_1$) | B0 ($F_1$) | ShuffleNet-V2 ($F_1$) | Diagnóstico Agronómico |
|---|:---:|:---:|:---:|:---:|---|
| **Lethal Necrosis** | 963 | **0.9990** | 0.9984 | 0.9974 | Recall 0.998; una sola imagen no detectada. |
| **Healthy (Planta Sana)** | 1,311 | **0.9936** | 0.9954 | 0.9901 | Recall 1.000 en Lite0: ninguna planta sana clasificada como enferma. |
| **Common Rust (Roya Común)** | 338 | **0.9866** | 0.9896 | 0.9867 | Pústulas foliares identificadas con alta precisión. |
| **Fall Armyworm (Gusano Cogollero)**| 728 | **0.9814** | 0.9877 | 0.9815 | Daño masticador reconocido fielmente. |
| **Northern Corn Leaf Blight** | 1,025 | **0.9797** | 0.9769 | 0.9728 | Lesiones elípticas grandes diferenciadas correctamente. |
| **Gray Leaf Spot** | 290 | **0.9451** | 0.9194 | 0.9120 | Lite0 es la mejor de las tres en esta clase, por 2.6 puntos. |
| **Phosphorus Deficiency** | 140 | **0.9403** | 0.9527 | 0.9275 | Coloración púrpura/rojiza diagnosticada con éxito. |
| **Nitrogen Deficiency** | 127 | **0.8664** | 0.8939 | 0.8613 | Clorosis en "V"; recall de 0.843 en Lite0, el más bajo del conjunto. |
| **Potassium Deficiency** | 93 | **0.8290** | 0.8208 | 0.7674 | Clorosis marginal en hojas basales; clase con menor soporte. |

Las tres arquitecturas ordenan las clases igual: las patologías con lesión visible bien delimitada por encima de 0.97, y las tres deficiencias nutricionales por debajo de 0.95. Esa jerarquía la fija el soporte —93, 127 y 140 muestras frente a las 963-1,311 de las mayoritarias— y la similitud visual de los patrones cloróticos, no la arquitectura.

`EfficientNet-Lite0` es la mejor de las tres en `gray_leaf_spot` (0.9451 frente a 0.9194 y 0.9120), `lethal_necrosis`, `northern_corn_leaf_blight` y `potassium_deficiency`; `EfficientNet-B0` lo es en las cinco restantes.

::: tip Base para el ensamble
Que cada arquitectura sea mejor en clases distintas es la condición que hace útil el **[Ensamble Multimodelo](./ensamble)**: sus errores están parcialmente descorrelacionados.
:::
