# Síntesis de Experimentos del Pipeline Principal

**Estado: HISTÓRICO.** Resume decisiones de la etapa previa sobre la materialización de la etapa 2 (33 433 imágenes). Para el estado vigente consultar [resultados](/es/resultados/) y [protocolos](/es/metodologia/protocolos-experimentales).

A lo largo del desarrollo del pipeline principal, el foco del proyecto pasó de comparar arquitecturas aisladas a validar decisiones concretas de entrenamiento y regularización sobre el corpus de 33 433 imágenes de la etapa 2. 

Las decisiones se articularon alrededor de cinco ejes que definieron el artefacto de la etapa 2:

---

## Decisiones clave

### 1. Estrategia de balanceo de clases
El pipeline principal balancea con pérdida ponderada (`sqrt_inverse`) y deja desactivado el *WeightedRandomSampler*: `train.py` documenta que combinar ambos compensaría el mismo desbalance por dos vías. El repositorio no contiene un experimento que compare la combinación con la pérdida sola.

A la pérdida ponderada se suma aumento de datos (*data augmentation*) asimétrico en las clases cuya razón de desbalance supera `minority_ratio_threshold = 4.0`: cuatro clases en el corpus de la etapa 2 (desbalance máximo 14.1×).

### 2. Regularización y calibración de probabilidades
Se incorporaron suavizado de etiquetas (*Label Smoothing = 0.10*) y decaimiento de pesos ($L_2 = 10^{-4}$) como regularizadores. El baseline actual presenta ECE 0.1397 con confianza media de 0.844 en los aciertos y exactitud de 0.978: el modelo está subconfiado, la calibración no se considera resuelta y no se atribuye a estos mecanismos ni su efecto ni su ausencia.

### 3. Dinámica del optimizador y calentamiento
El pipeline usa coseno con calentamiento lineal de tres épocas. `src/training/optim.py` documenta la elección: a diferencia de *ReduceLROnPlateau*, la curva de tasa de aprendizaje no depende del ruido de validación, así que dos corridas con la misma semilla recorren el mismo camino. El repositorio no contiene una comparación con *StepLR* ni con *ReduceLROnPlateau*.

### 4. Cuantización y paridad en el borde
La exportación a Int8 redujo `EfficientNet-Lite0` de 13.56 MB a 3.74 MB. En el modelo oficial `20260921_204608` el int8 obtiene Macro-F1 0.9489 sobre el test completo, y en el teléfono la inferencia toma unos 23 ms.

### 5. Consenso multimodelo
La votación suave histórica combinó `EfficientNet-Lite0`, `EfficientNet-B0` y `ShuffleNet-V2` y alcanzó Macro-F1 0.9567. Es una observación sobre el test estratificado de aquella etapa, no una prueba causal sobre diversidad ni un resultado del baseline actual.
