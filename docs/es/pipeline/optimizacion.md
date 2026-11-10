# Optimización histórica de hiperparámetros (Optuna)

**Estado: HISTÓRICO.** Esta página describe los estudios anteriores de 15/25 trials. No documenta el HPO formal de 25 intentos, cerrado el 2026-09-24; ver [HPO formal](/es/experimentos/hpo).

Se buscaron hiperparámetros (tasa de aprendizaje, tamaño del lote, decaimiento de pesos, entre otros) con **Optuna** (sampler TPE y `MedianPruner` con `n_startup_trials=3` y `n_warmup_steps=3`) en GPUs NVIDIA A10G en la nube. Hubo dos estudios: uno sobre `EfficientNet-B0` de 15 trials y uno sobre `EfficientNet-Lite0` de 25 trials con techo de 15 épocas.

---

## El espacio de búsqueda y la poda temprana

El objetivo fue maximizar el **Macro F1-Score** sobre el conjunto de validación. El estudio de `B0` exploró siete hiperparámetros (el de `Lite0` seis, sin `clahe`) en estos rangos:

| Hiperparámetro | Rango explorado | Propósito técnico |
|---|:---:|---|
| **`learning_rate`** | $10^{-5}$ a $10^{-3}$ (log-uniforme) | Ritmo de actualización de los pesos con AdamW. |
| **`batch_size`** | 16, 32 o 64 | Compromiso entre regularización estocástica y paralelismo en GPU. |
| **`weight_decay`** | $10^{-5}$ a $10^{-2}$ | Regularización desacoplada para prevenir sobreajuste. |
| **`class_weights`** | `none`, `inverse` o `sqrt_inverse` | Estrategia de ponderación para compensar el desbalance de clases. |
| **`label_smoothing`** | 0.0 a 0.15 | Suavizado de etiquetas para moderar la sobreconfianza de la red. |
| **`warmup_epochs`** | 1 a 5 épocas | Calentamiento inicial suave para estabilizar las capas preentrenadas. |
| **`clahe`** | Activado / Desactivado | Ecualización adaptativa de histograma. |

Con el **MedianPruner**, un ensayo cuyo Macro F1 de validación quedaba por debajo de la mediana de los ensayos anteriores en la misma época se cancelaba. En el estudio de `B0` los 7 trials podados duraron 33 minutos en promedio y los 6 completados 101 minutos; si los podados hubieran corrido hasta el final con esa duración media, el ahorro estimado sería de unas 8 horas de GPU.

---

## Hallazgos principales del estudio

Lo que muestran los 15 trials de `B0` (`optuna_b0_presupuesto_completo_trials.csv`):

1. **Lote 64 y `sqrt_inverse`:** los tres mejores trials completados (0.9553, 0.9530 y 0.9529 de Macro F1 de validación) usan lote 64, `sqrt_inverse` y `learning_rate` entre 4.5e-4 y 4.6e-4. Los trials 1 y 6 tienen parámetros idénticos. Lote, pesos de clase y tasa de aprendizaje varían a la vez entre trials, de modo que el estudio no aísla el efecto de cada uno.
2. **Pesos de clase:** los otros tres trials completados usan `none` con lote 16 (0.9307 y 0.9454) e `inverse` con lote 64 (0.9298), todos por debajo de los tres mejores. El HPO formal de `Lite0` posterior eligió `none`: [HPO formal](/es/experimentos/hpo).
3. **CLAHE:** de los 4 trials de `B0` con CLAHE, uno se completó (0.9298, con `inverse` y otros parámetros distintos de los mejores), dos fueron podados y uno quedó sin cerrar. No hay un A/B controlado. Se retiró del espacio de búsqueda porque `tuning.py` mide que CLAHE sube el transform de 9.3 a 24.7 ms por imagen.

---

## Diferencias entre arquitecturas

Las corridas finales de `B0` y `ShuffleNet-V2` usaron solo `learning_rate` y `batch_size` del Trial #12, con techo de 35 épocas en lugar de 60. Su Macro F1 de prueba pasó de 94.26% a **94.83%** en `B0` (+0.57 pp) y de 92.37% a 93.30% en `ShuffleNet-V2` (+0.93 pp), una corrida por configuración.

En **`EfficientNet-Lite0`** los entrenamientos con parámetros de Optuna dieron 93.43% (solo `learning_rate` y lote del Trial #12), 93.86% (todos los parámetros del Trial #12) y 93.79% (el mejor trial del estudio de `Lite0`), frente a 94.68% de la configuración base. La configuración base con las semillas 1 y 2 dio 94.48% y 93.75%, de modo que el rango entre semillas (0.93 pp) es comparable a las diferencias observadas y no hay una causa demostrada para la caída en `Lite0`.

La decisión histórica fue mantener a **`EfficientNet-Lite0` con su configuración base (Macro-F1 de 94.68 % en prueba)**, mientras las otras redes adoptaron la configuración afinada para el ensamble histórico. No predetermina el resultado del HPO nuevo.
