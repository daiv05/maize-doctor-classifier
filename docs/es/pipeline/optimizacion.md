# Optimización histórica de hiperparámetros (Optuna)

**Estado: HISTÓRICO.** Esta página describe los estudios anteriores de 15/25 trials. No documenta el estudio nuevo de 60 trials, que permanece [planificado](/es/experimentos/hpo).

Ajustar a mano los hiperparámetros de una red neuronal —la tasa de aprendizaje, el tamaño del lote, el decaimiento de pesos— suele ser una pérdida de tiempo y recursos. Si se elige una tasa muy alta, el modelo diverge y no aprende; si es muy baja, el entrenamiento avanza con lentitud excesiva y puede estancarse en mínimos locales subóptimos.

Para encontrar de forma sistemática la mejor configuración sin gastar semanas de cómputo, implementamos un proceso de **Optimización Bayesiana** utilizando **Optuna**, ejecutado sobre GPUs NVIDIA A10G en la nube.

---

## El espacio de búsqueda y la poda temprana

El objetivo de la optimización fue maximizar el **Macro F1-Score** sobre el conjunto de validación, asegurando que las patologías minoritarias no fueran sacrificadas a cambio de un acierto fácil en las clases abundantes.

Exploramos siete hiperparámetros clave dentro de rangos bien delimitados:

| Hiperparámetro | Rango explorado | Propósito técnico |
|---|:---:|---|
| **`learning_rate`** | $10^{-5}$ a $10^{-3}$ (log-uniforme) | Ritmo de actualización de los pesos con AdamW. |
| **`batch_size`** | 16, 32 o 64 | Compromiso entre regularización estocástica y paralelismo en GPU. |
| **`weight_decay`** | $10^{-5}$ a $10^{-2}$ | Regularización desacoplada para prevenir sobreajuste. |
| **`class_weights`** | `none`, `inverse` o `sqrt_inverse` | Estrategia de ponderación para compensar el desbalance de clases. |
| **`label_smoothing`** | 0.0 a 0.15 | Suavizado de etiquetas para moderar la sobreconfianza de la red. |
| **`warmup_epochs`** | 1 a 5 épocas | Calentamiento inicial suave para estabilizar las capas preentrenadas. |
| **`clahe`** | Activado / Desactivado | Ecualización adaptativa de histograma para compensar sombras de campo. |

Uno de los mayores aciertos del estudio fue incorporar un mecanismo de poda temprana (**MedianPruner**). Si un ensayo mostraba un Macro F1 por debajo de la mediana histórica en las primeras épocas, Optuna lo cancelaba de inmediato en lugar de dejarlo correr hasta el final. Gracias a esta poda, los ensayos deficientes se detenían en promedio a los 30 minutos, mientras que los prometedores completaban su ciclo de hora y media, ahorrando cerca de 8 horas de cómputo en GPU.

---

## Hallazgos principales del estudio

El análisis de importancia de parámetros reveló patrones muy claros sobre el comportamiento de los modelos:

1. **El tamaño de lote y la tasa de aprendizaje mandan (>75% del impacto):**
   Con un tamaño de lote de 64 y una tasa de aprendizaje moderadamente más alta ($\approx 4.5 \times 10^{-4}$), los gradientes de AdamW ganaron mucha estabilidad frente a lotes de 16 o 32, permitiendo al modelo escapar de ruidos propios de las fotos de campo.
2. **`sqrt_inverse` confirmó ser el balance ideal:**
   Entrenar sin pesos en la pérdida (`none`) provocaba caídas severas en deficiencias nutricionales, mientras que la inversa pura (`inverse`) sobrecastigaba a la clase sana reduciendo la exactitud general. La raíz cuadrada inversa ofreció el equilibrio justo.
3. **CLAHE resultó innecesario:**
   Las capas convolucionales profundas aprenden representaciones invariantes a la luz y a las sombras por su cuenta. Aplicar ecualización adaptativa antes de entrenar no aportó mejoras y habría sumado un costo innecesario en la aplicación móvil.

---

## Diferencias entre arquitecturas

La configuración óptima encontrada sobre `EfficientNet-B0` (Trial #12) logró elevar su Macro F1 de prueba de 94.26% a **94.83%** (+0.57 pp), y al aplicarse a `ShuffleNet-V2` también mejoró su rendimiento (+0.93 pp).

Sin embargo, al probar esa misma configuración sobre la arquitectura desplegada, **`EfficientNet-Lite0`**, el resultado fue el contrario: el Macro F1 bajó a 93.43%. La razón es arquitectónica: `Lite0` prescinde de los bloques de atención Squeeze & Excitation y sustituye las funciones *swish* por operaciones cuantizables, tolerando menos las tasas de aprendizaje agresivas. 

Por esta razón, la decisión histórica fue mantener a **`EfficientNet-Lite0` con su configuración base (Macro-F1 de 94.68 % en prueba)**, mientras las otras redes adoptaron la configuración afinada para el ensamble histórico. No predetermina el resultado del HPO nuevo.
