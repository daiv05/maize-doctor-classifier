# Síntesis de Experimentos del Pipeline Principal

**Estado: HISTÓRICO.** Resume decisiones de la etapa previa sobre una materialización de 33 433 imágenes. Para el estado vigente consultar [registro de experimentos](/es/experimentos/) y [protocolos](/es/metodologia/protocolos-experimentales).

A lo largo del desarrollo del pipeline principal, el foco del proyecto pasó de comparar arquitecturas aisladas a validar decisiones concretas de entrenamiento y regularización sobre el corpus completo de 33,433 imágenes. 

Las decisiones se articularon alrededor de cinco ejes que definieron el artefacto históricamente desplegado:

---

## Decisiones clave validadas en el camino

### 1. Estrategia de balanceo de clases
Inicialmente se evaluó combinar un remuestreo agresivo en los lotes (*WeightedRandomSampler*) junto con una función de pérdida ponderada. Al probarlo sobre el desbalance real del corpus (14 a 1 entre clases mayoritarias y minoritarias), comprobamos que aplicar ambos mecanismos a la vez sobrecompensaba en exceso, degradando el rendimiento en clases comunes como *Healthy*. 

La solución adoptada fue desactivar el remuestreo de lotes y combinar una pérdida ponderada moderada (`sqrt_inverse`) con aumento de datos (*data augmentation*) focalizado exclusivamente en las cuatro clases minoritarias.

### 2. Regularización y calibración de probabilidades
Se incorporaron suavizado de etiquetas (*Label Smoothing = 0.10*) y decaimiento de pesos ($L_2 = 10^{-4}$) como regularizadores. El baseline actual todavía presenta ECE 0.1397, por lo que no se considera resuelta la calibración ni se atribuye causalmente su mejora a estos mecanismos.

### 3. Dinámica del optimizador y calentamiento
Comparamos programadores de tasa de aprendizaje por pasos (*StepLR*), por meseta (*ReduceLROnPlateau*) y por coseno con calentamiento (*Cosine Annealing with Warmup*). El arranque suave de tres épocas lineales seguido del decaimiento armónico por coseno resultó ser el más estable: evitó que los pesos preentrenados de ImageNet se descalibraran en las primeras épocas y facilitó un aterrizaje suave hacia el óptimo.

### 4. Cuantización y paridad en el borde
La exportación histórica a Int8 redujo `EfficientNet-Lite0` de 12.9 MB a 3.56 MB y se midió una latencia aproximada de 60 ms en el dispositivo documentado. La evaluación completa del modelo exportado actual sigue pendiente; esta evidencia no se transfiere automáticamente al checkpoint nuevo.

### 5. Consenso multimodelo
La votación suave histórica combinó `EfficientNet-Lite0`, `EfficientNet-B0` y `ShuffleNet-V2` y alcanzó Macro-F1 0.9567. Es una observación sobre el test estratificado de aquella etapa, no una prueba causal sobre diversidad ni un resultado del baseline actual.
