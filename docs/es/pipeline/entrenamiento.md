# Entrenamiento del pipeline principal

Se entrenaron tres arquitecturas en la etapa histórica del pipeline principal. La materialización vigente de `seed_42` contiene **33 429 imágenes** en nueve clases. El `EfficientNet-Lite0` `20260921_204608` es el modelo oficial y el que despliega la app; el `20260812_221429` de la etapa 2 es el histórico.

El pipeline produce checkpoints `best.pth` para ensamble, auditoría y exportación. “Best” significa mejor validación dentro del run, no validación de producción.

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

Este split es estratificado por `label + environment`; las fuentes se conservan como metadato y pueden aparecer a ambos lados. Los resultados históricos que aparecen más abajo corresponden a la materialización de la etapa 2 (33 433 imágenes). El primer reentrenamiento sobre el split corregido es el [baseline `20260921_204608`](/es/resultados/run-20260921-efficientnet-lite0).

---

## 2. Hiperparámetros históricos de despliegue

`B0` y `ShuffleNet-V2` usan `learning_rate` y `batch_size` del trial 12 del estudio de Optuna sobre `B0`; `Lite0` usa los valores por defecto:

| | `EfficientNet-Lite0` (etapa 2) | `EfficientNet-B0` y `ShuffleNet-V2` |
|---|:---:|:---:|
| **Learning Rate** | `1.0e-4` | `4.548e-4` |
| **Batch Size** | 32 | 64 |
| **Corrida** | `20260812_221429` | `20260910_170120` / `20260910_184521` |
| **Techo de épocas** | 60 | 35 |

En `B0` y `ShuffleNet-V2` esa configuración dio mayor Macro-F1 de validación (`B0` 0.9610 frente a 0.9551; `ShuffleNet-V2` 0.9502 frente a 0.9408) y de prueba que la de por defecto, aunque esas corridas también cambian el techo de épocas, de 60 a 35. En `Lite0`, `lr` y lote de Optuna dieron 0.9343 en prueba frente a 0.9468 por defecto, y los tres entrenamientos con parámetros de Optuna (0.9343 a 0.9386) quedaron por debajo de la media de tres semillas de la configuración por defecto (0.9430, rango 0.9375 a 0.9468, SD 0.0049), con solapamiento de rangos: la diferencia no está establecida. El análisis está en [Optimización histórica](/es/pipeline/optimizacion) y en el [HPO formal](/es/experimentos/hpo).

Los valores de `warmup_epochs` y `weight_decay` de la tabla siguiente son los del pipeline, no los del Trial #12 —que eran 2 y 1.573e-05—: `train_main` del wrapper de Modal no expone esos dos parámetros y los `summary.json` de `20260910_170120` y `20260910_184521` registran 3 y 1e-4. El resto es común a las tres:

| Hiperparámetro | Valor histórico | Justificación técnica |
|---|:---:|---|
| **Optimizador** | **AdamW** | Decaimiento de pesos desacoplado del gradiente. |
| **Learning Rate Base** | **`4.548e-4`** (`1.0e-4` en Lite0) | Identificado por Optuna sobre `B0`; `Lite0` conserva el valor por defecto. |
| **LR Scheduler** | **Cosine Annealing con Warmup** | 3 épocas de calentamiento lineal seguidas de decaimiento coseno hasta $\eta_{\min} = 10^{-6}$ sobre el techo de épocas. |
| **Batch Size** | **64** (**32** en Lite0) | Valor del trial 12 de Optuna en B0 y ShuffleNet; por defecto en Lite0. |
| **Ponderación de Pérdida** | **`sqrt_inverse`** | Peso por clase proporcional a la raíz de la inversa de su frecuencia en train; el `WeightedRandomSampler` queda desactivado. |
| **Label Smoothing** | **`0.10`** | Suavizado de etiquetas en la entropía cruzada. |
| **Weight Decay** | **`1.0e-4`** | Coeficiente de AdamW. |
| **Gradient Clipping** | **`1.0`** | Norma máxima del gradiente por paso. |
| **Early Stopping** | **Patience = 8 épocas** | Se detiene tras 8 épocas consecutivas sin mejora del Macro $F_1$ de validación. |

---

## 3. Curvas de Convergencia y Dinámica de Entrenamiento

Se entrenaron de forma independiente tres arquitecturas:
1. **`EfficientNet-B0`:** Red convolucional de alta capacidad con bloques MBConv y atención de canales (Squeeze & Excitation).
2. **`ShuffleNet-V2-x1.0`:** Red ultra-ligera de baja latencia con división y barajado de canales (*Channel Split & Shuffle*).
3. **`EfficientNet-Lite0`:** Variante sin bloques Squeeze & Excitation ni activaciones *swish*, sustituidas por operaciones cuantizables a entero de 8 bits. **Es la arquitectura que despliega la app.**

La figura siguiente compara las tres bajo **hiperparámetros idénticos** —los valores por defecto del pipeline— para que la curva refleje la arquitectura y no la configuración:

![Convergencia del Entrenamiento](/training/training_convergence.png)

### Análisis de la Convergencia:

1. **Evolución de la Pérdida (Loss):**
   - La pérdida de validación no desciende de forma monótona: sube en 17 de las 42 transiciones de época en `Lite0`, 11 de 25 en `B0` y 9 de 25 en `ShuffleNet-V2`.
   - El mínimo de la pérdida de validación es 0.7553 (`Lite0`, época 35), 0.7528 (`B0`, época 18) y 0.7750 (`ShuffleNet-V2`, época 26). La pérdida incluye los pesos de clase y el label smoothing.

2. **Progresión del Macro $F_1$-Score en Validación:**
   - **`EfficientNet-Lite0`** es la de convergencia más lenta: alcanza su pico en la **Época 35** con Macro $F_1$ de validación **0.9554** (frente a 0.9551 de `B0`), y el early stopping la detiene en la 43. Es la única de las tres que sigue mejorando más allá de la época 20.
   - **`EfficientNet-B0`** alcanza **0.9551** en la **Época 18** bajo la misma configuración, y se detiene en la 26. Con `learning_rate` 4.548e-4 y lote 64 sube a **0.9610** en la **Época 28** sobre 35 programadas.
   - **`ShuffleNet-V2`** cierra en **0.9408** en la **Época 18** con la configuración por defecto. Con la configuración de Optuna alcanza **0.9502** en la **Época 23**.
   - El **Early Stopping** con paciencia de 8 épocas detuvo `B0` y `ShuffleNet-V2` en la 26, y `Lite0` en la 43. Las tres comparten configuración, pero la diferencia de diecisiete épocas no se atribuye a la arquitectura: `Lite0` con las semillas 1 y 2 corrió 60 y 32 épocas.

---

## 4. Resultados Individuales en el Conjunto de Test Retenido (5,015 Imágenes)

Al evaluar los checkpoints finales (`best.pth`) sobre el subconjunto de prueba independiente (`test.csv`), los tres modelos obtuvieron:

| Métrica Global | EfficientNet-Lite0 (etapa 2) | EfficientNet-B0 | ShuffleNet-V2-x1.0 |
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

Desempeño detallado sobre las 9 clases (Precision / Recall / $F_1$-Score). Los tres modelos se evaluaron sobre el `test.csv` de la etapa 2 (338 `common_rust` y 290 `gray_leaf_spot`); el `test.csv` vigente y las predicciones del run `20260921_204608` tienen 339 y 289.

| Clase Agronómica | Muestras Test | **Lite0** ($F_1$) | B0 ($F_1$) | ShuffleNet-V2 ($F_1$) | Diagnóstico Agronómico |
|---|:---:|:---:|:---:|:---:|---|
| **Lethal Necrosis** | 963 | **0.9990** | 0.9984 | 0.9974 | Recall 0.998; 2 imágenes no detectadas. |
| **Healthy (Planta Sana)** | 1,311 | **0.9936** | 0.9954 | 0.9901 | Recall 1.000 en Lite0: ninguna planta sana clasificada como enferma. |
| **Common Rust (Roya Común)** | 338 | **0.9866** | 0.9896 | 0.9867 | Pústulas foliares identificadas con alta precisión. |
| **Fall Armyworm (Gusano Cogollero)**| 728 | **0.9814** | 0.9877 | 0.9815 | Daño masticador reconocido fielmente. |
| **Northern Corn Leaf Blight** | 1,025 | **0.9797** | 0.9769 | 0.9728 | Lesiones elípticas grandes diferenciadas correctamente. |
| **Gray Leaf Spot** | 290 | **0.9451** | 0.9194 | 0.9120 | Lite0 es la mejor de las tres en esta clase, por 2.6 puntos. |
| **Phosphorus Deficiency** | 140 | **0.9403** | 0.9527 | 0.9275 | Coloración púrpura/rojiza diagnosticada con éxito. |
| **Nitrogen Deficiency** | 127 | **0.8664** | 0.8939 | 0.8613 | Clorosis en "V"; recall de 0.843 en Lite0, el más bajo del conjunto. |
| **Potassium Deficiency** | 93 | **0.8290** | 0.8208 | 0.7674 | Clorosis marginal en hojas basales; clase con menor soporte. |

Las cuatro clases más débiles son las mismas en las tres arquitecturas: `gray_leaf_spot` y las tres deficiencias nutricionales, con F1 entre 0.7674 y 0.9527. Las cinco restantes superan 0.972 en las tres. Las deficiencias son además las clases de menor soporte (93, 127 y 140 muestras frente a 963-1,311 en las mayoritarias); este análisis no separa el efecto del soporte del de la similitud visual.

`EfficientNet-Lite0` es la mejor de las tres en `gray_leaf_spot` (0.9451 frente a 0.9194 y 0.9120), `lethal_necrosis`, `northern_corn_leaf_blight` y `potassium_deficiency`; `EfficientNet-B0` lo es en las cinco restantes.

::: tip Base para el ensamble
Que cada arquitectura sea mejor en clases distintas es el punto de partida del **[Ensamble Multimodelo](./ensamble)**, que mide el solapamiento de sus errores.
:::
