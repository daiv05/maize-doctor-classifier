# Optimización de Hiperparámetros (Optuna)

Antes de comprometer el entrenamiento final de las redes convolucionales sobre las más de 31,000 imágenes del dataset, es fundamental encontrar la configuración óptima de hiperparámetros. El ajuste manual o heurístico suele producir soluciones subóptimas o incurrir en sobreajuste severo.

En esta etapa se implementó un pipeline de **Optimización Bayesiana de Hiperparámetros (HPO)** utilizando **Optuna**, aplicando el estimador de árbol de Parzen (**TPE Multivariado**) junto con algoritmos de **poda temprana (*Early Pruning*)**.

---

## Metodología y Espacio de Búsqueda

La optimización busca maximizar el **Macro $F_1$-Score** sobre el conjunto de validación, penalizando el desbalance de clases y asegurando que las patologías minoritarias no sean sacrificadas en favor de la clase mayoritaria (*Healthy*).

```
                 ┌──────────────────────────────────────┐
                 │ Espacio de Búsqueda (Hyperparameters)│
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
┌──────────────────┐     ┌─────────────────────┐     ┌────────────────────┐
│   TPE Sampler    │────▶│  Entrenamiento GPU  │────▶│   Evaluación F1    │
│  (Multivariado)  │     │   (Modal A10G)      │     │    (Validación)    │
└──────────────────┘     └──────────┬──────────┘     └─────────┬──────────┘
         ▲                          │                          │
         │                          │                          │
         │           [Poda si F1 < Mediana en Época 3]         ▼
         │           ┌──────────────────────────────────────────────┐
         └───────────┤   MedianPruner (Corta trials deficientes)    │
                     └──────────────────────────────────────────────┘
```

### Hiperparámetros Optimizados

| Hiperparámetro | Espacio / Distribución | Propósito y Justificación |
|---|---|---|
| **`learning_rate`** | Log-uniforme: $[10^{-5}, 10^{-3}]$ | Control de la velocidad de convergencia y estabilidad del gradiente en AdamW. |
| **`weight_decay`** | Log-uniforme: $[10^{-6}, 10^{-2}]$ | Regularización $L_2$ para prevenir memorización de fondos o ruido de sensor. |
| **`batch_size`** | Categórico: $\{16, 32, 64\}$ | Compromiso entre regularización estocástica y paralelismo en GPU. |
| **`class_weights`** | Categórico: $\{\text{sqrt\_inverse}, \text{inverse}, \text{none}\}$ | Estrategia de ponderación en CrossEntropyLoss para contrarrestar el desbalance. |
| **`label_smoothing`** | Uniforme: $[0.0, 0.15]$ | Suavizado de etiquetas para evitar sobreconfianza en las predicciones. |
| **`warmup_epochs`** | Entero: $[1, 5]$ | Épocas de calentamiento lineal para estabilizar capas pre-entrenadas. |
| **`clahe`** | Booleano: $\{\text{True}, \text{False}\}$ | Ecualización adaptativa de histograma para mitigar sombras en campo. |

---

## Resultados de las Corridas en GPU (Modal)

Se ejecutó un estudio de **15 trials** sobre la arquitectura base **`EfficientNet-B0`** en una GPU NVIDIA A10G (24 GB VRAM).

### Comparativa: Baseline vs. Configuración Óptima

| Métrica / Parámetro | Baseline (Valores por Defecto) | Optimizada con Optuna (Trial #12) | Impacto / Delta |
|---|:---:|:---:|:---:|
| **Macro $F_1$ (validación)** | `0.9551` (95.51 %) | **`0.9553` (95.53 %)** | **+0.02 pp** |
| **Macro $F_1$ (prueba retenida)** | `0.9426` (94.26 %) | **`0.9483` (94.83 %)** | **+0.57 pp** |
| **Learning Rate** | $1.00 \times 10^{-4}$ | **$4.55 \times 10^{-4}$** | Tasa 4.5x más eficiente con Warmup |
| **Weight Decay** | $1.00 \times 10^{-4}$ | **$1.57 \times 10^{-5}$** | Menor penalización sobre pesos |
| **Batch Size** | 32 | **64** | Mayor estabilidad y saturación GPU |
| **Pérdida Ponderada** | `sqrt_inverse` | **`sqrt_inverse`** | Confirmada como la mejor estrategia |
| **Label Smoothing** | 0.10 | **0.1001** | Regularización óptima de probabilidades |
| **Warmup Epochs** | - | **2 épocas** | Arranque suave del optimizador |
| **CLAHE** | No | **No** | Red convolucional aprende invariancia sin prefiltrado |

::: tip Resultados vs Baseline
La configuración encontrada por Optuna eleva el Macro $F_1$ de prueba de **94.26% a 94.83%**, un incremento de **+0.57 puntos porcentuales** sobre la corrida `20260910_170120`.

El baseline de referencia es la corrida archivada `efficientnet_b0/20260811_211306`, que con los valores por defecto alcanza 0.9551 de validación y 0.9426 de prueba. En validación la diferencia es de dos diezmilésimas; la ganancia real y verificable está en el conjunto de prueba.
:::

## El barrido sobre `EfficientNet-Lite0`

Se ejecutó un segundo estudio, de **25 trials**, sobre la arquitectura desplegada. Su resultado no se adoptó, y la razón es parte del resultado:

El barrido sobre `efficientnet_lite0` se ejecutó con un techo de **15 épocas** por coste de cómputo. Ese presupuesto resultó insuficiente como criterio de selección: la corrida de referencia alcanza su óptimo en la **época 35**, de modo que a las 15 ninguna configuración ha convergido y la búsqueda selecciona por **velocidad de convergencia**, no por calidad final. La configuración elegida lo confirma —0.9451 en la época 12, agotada en la 20— frente a **0.9554 en la época 35** de los valores por defecto.

En lugar de repetir el barrido, se evaluaron dos configuraciones adicionales sobre `lite0`, ambas derivadas del estudio sobre `efficientnet_b0`: la configuración completa de seis hiperparámetros (**0.9386** en prueba) y la restringida a `learning_rate` y `batch_size` (**0.9343**), que es el tratamiento exacto con el que esas mismas modificaciones **mejoran** a `efficientnet_b0` (+0.0057) y a `shufflenet_v2_x1_0` (+0.0093). Las tres quedan por debajo de los valores por defecto (**0.9468**).

| configuración sobre `lite0` | val | prueba |
|---|:---:|:---:|
| **Valores por defecto** | **0.9554** | **0.9468** |
| Optuna, techo de 15 épocas | 0.9451 | 0.9379 |
| Hiperparámetros de `b0`, los seis | 0.9548 | 0.9386 |
| Hiperparámetros de `b0`, sólo `lr` y `batch_size` | 0.9468 | 0.9343 |

El resultado identifica un **mecanismo**, no sólo una ausencia de mejora: `EfficientNet-Lite0` sustituye los bloques Squeeze & Excitation y las activaciones *swish* por operaciones cuantizables, y no tolera el `learning_rate` elevado que beneficia a las otras dos arquitecturas. Dado que el espacio de búsqueda se centra en ese rango, un barrido a presupuesto completo exploraría predominantemente la región donde se ha medido la degradación. **La configuración de producción se mantiene en los valores por defecto.**

::: warning Limitación
No se ejecutó un barrido sobre `lite0` a presupuesto completo. La conclusión se apoya en que tres configuraciones optimizadas medidas quedan por debajo de la de referencia y en el mecanismo identificado, no en haber agotado el espacio de búsqueda.
:::

![Historial de Optimización — Lite0](/tuning/lite0_optimization_history.png)

![Importancia de Hiperparámetros — Lite0](/tuning/lite0_param_importances.png)

De los 25 trials, 6 completaron y 19 fueron podados por el `MedianPruner`. El detalle y la evidencia bruta están en [Optimización e hiperparámetros](/es/resultados/optimizacion).

---

::: warning Alcance de lo que se aplicó
Las corridas de producción de `EfficientNet-B0` y `ShuffleNet-V2` tomaron de este estudio únicamente `learning_rate` y `batch_size`. Conservaron `warmup_epochs` en 3 y `weight_decay` en 1.0e-4, en lugar de los 2 y 1.573e-05 del Trial #12, porque el wrapper de Modal no propagaba esos dos parámetros.

Aplicado a `efficientnet_lite0` el mismo cambio de `learning_rate` y `batch_size`, el Macro $F_1$ de prueba baja de 0.9468 a 0.9343; con los seis parámetros del trial queda en 0.9386. La configuración mejora dos arquitecturas y perjudica a la tercera, que es la desplegada. El detalle está en [Optimización e hiperparámetros](/es/resultados/optimizacion).
:::

---

## Análisis del Historial de Optimización

A lo largo de los 15 trials evaluados por el sampler TPE, seis completaron: cinco superaron 0.93 y tres se situaron en el entorno de 0.953 (0.9529, 0.9530 y 0.9553):

![Historial de Optimización](/tuning/optimization_history.png)

### Eficiencia de la Poda Temprana (*Early Pruning*)

De los 15 trials ejecutados:
* **6 trials completaron** (valores de 0.9298 a 0.9553; cinco por encima de 0.93).
* **7 trials fueron podados (*PRUNED*)** por el `MedianPruner` al situarse por debajo de la mediana histórica.
* **1 trial falló** y **1 quedó en ejecución** al detener el estudio.

El número de épocas por trial no quedó registrado en `trials.csv`, que almacena estado, valor, duración y parámetros.

::: info Ahorro Computacional
Cada trial completo tomó de media **101 minutos** (mediana 87; rango 84 a 139), mientras que los podados requirieron de media **33 minutos** (rango 24 a 52). Suponiendo que los siete podados hubieran corrido hasta el final, la poda ahorró **unas 7.9 horas de cómputo en GPU**.
:::

---

## Importancia de los Hiperparámetros

Mediante el análisis de importancia basado en bosques aleatorios (*Random Forest Feature Importance* evaluado por Optuna), se cuantificó la influencia de cada variable en el rendimiento final:

![Importancia de Hiperparámetros](/tuning/param_importances.png)

### Hallazgos Clave:

1. **Dominancia del `batch_size` y `learning_rate` (> 75% del impacto):**
   - El tamaño de lote y la tasa de aprendizaje determinan la casi totalidad de la varianza en el rendimiento.
   - Con `batch_size: 64`, el gradiente promedio por paso fue significativamente más estable que con 16 o 32, permitiendo al modelo escapar de mínimos locales ruidosos causados por la variabilidad fotográfica de campo.
2. **Superioridad de `sqrt_inverse`:**
   - La opción `none` (sin pesos) provocó caídas de hasta 8 puntos porcentuales en clases como *Nitrogen Deficiency* y *Potassium Deficiency*.
   - Por otro lado, la opción `inverse` pura sobre-penalizó a la clase *Healthy*, reduciendo la precisión global. `sqrt_inverse` demostró ser el balance matemático perfecto para este dataset.
3. **Innecesariedad de CLAHE en el entrenamiento:**
   - Optuna descartó el uso de CLAHE (`clahe: False`). Las capas convolucionales profundas de EfficientNet-B0 son capaces de aprender representaciones invariantes a la iluminación sin necesidad de incurrir en el costo computacional adicional de la ecualización adaptativa en tiempo de inferencia móvil.

---

## Costura con el Resto del Pipeline

El estudio de optimización no opera como un módulo aislado; sus resultados se serializan en `outputs/tuning/efficientnet_b0/best_params.json`:

::: warning Sobre los campos `baseline_macro_f1` e `improvement_*`
El valor `0.9146` que aparece abajo es el que se pasó como referencia al lanzar el estudio y no corresponde a ninguna corrida archivada del proyecto; los campos `improvement_delta` e `improvement_pct` se derivan de él y por tanto tampoco. Son campos de reporte: no intervienen en la función objetivo de Optuna ni en la selección del mejor trial, así que `best_params` es válido. Las cifras correctas están en la tabla comparativa de esta página.
:::

```json
{
  "study_name": "tune_efficientnet_b0",
  "model_name": "efficientnet_b0",
  "best_trial_number": 12,
  "best_val_macro_f1": 0.9553,
  "baseline_macro_f1": 0.9146,
  "improvement_delta": 0.0407,
  "improvement_pct": 4.45,
  "best_params": {
    "learning_rate": 0.0004548,
    "weight_decay": 1.573e-05,
    "batch_size": 64,
    "class_weights": "sqrt_inverse",
    "label_smoothing": 0.1001,
    "warmup_epochs": 2,
    "clahe": false
  }
}
```

Este archivo es consumido automáticamente por:
1. **[Entrenamiento Final (`train.py`)](./entrenamiento):** Para converger al checkpoint de producción (`best.pt`).
2. **[Validación Cruzada 5-Fold (`cross_validate.py`)](./evaluacion):** Inyecta estos parámetros en cada pliegue para certificar que el $F_1 \ge 0.95$ se mantenga estable con intervalos de confianza al 95%.
3. **[Ensamble Multimodelo (`evaluate_ensemble.py`)](./evaluacion):** Combina las probabilidades de los modelos optimizados para maximizar la robustez en datos nunca vistos.
