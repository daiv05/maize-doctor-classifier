# Síntesis de Experimentos del Pipeline Principal

A lo largo del desarrollo del pipeline principal, el foco del proyecto pasó de comparar arquitecturas aisladas a validar decisiones concretas de entrenamiento y regularización sobre el corpus completo de 33,433 imágenes. 

En lugar de acumular decenas de pruebas marginales, las decisiones se articularon alrededor de cinco ejes clave que definieron la versión de producción:

---

## Decisiones clave validadas en el camino

### 1. Estrategia de balanceo de clases
Inicialmente se evaluó combinar un remuestreo agresivo en los lotes (*WeightedRandomSampler*) junto con una función de pérdida ponderada. Al probarlo sobre el desbalance real del corpus (14 a 1 entre clases mayoritarias y minoritarias), comprobamos que aplicar ambos mecanismos a la vez sobrecompensaba en exceso, degradando el rendimiento en clases comunes como *Healthy*. 

La solución adoptada fue desactivar el remuestreo de lotes y combinar una pérdida ponderada moderada (`sqrt_inverse`) con aumento de datos (*data augmentation*) focalizado exclusivamente en las cuatro clases minoritarias.

### 2. Regularización y calibración de probabilidades
Para evitar que las redes memorizaran detalles circunstanciales de las imágenes y emitieran certezas absolutas injustificadas, incorporamos suavizado de etiquetas (*Label Smoothing = 0.10*) y decaimiento de pesos ($L_2 = 10^{-4}$). Esto no solo mejoró la generalización en el conjunto de prueba, sino que estabilizó las probabilidades para que el detector de anomalías (OOD) y los umbrales de confianza operaran con mayor fidelidad.

### 3. Dinámica del optimizador y calentamiento
Comparamos programadores de tasa de aprendizaje por pasos (*StepLR*), por meseta (*ReduceLROnPlateau*) y por coseno con calentamiento (*Cosine Annealing with Warmup*). El arranque suave de tres épocas lineales seguido del decaimiento armónico por coseno resultó ser el más estable: evitó que los pesos preentrenados de ImageNet se descalibraran en las primeras épocas y facilitó un aterrizaje suave hacia el óptimo.

### 4. Cuantización y paridad en el borde
La exportación a formato móvil implicó evaluar el impacto de pasar de precisión flotante de 32 bits a enteros de 8 bits (Int8). Comprobamos que para `EfficientNet-Lite0`, la cuantización redujo el tamaño de 12.9 MB a solo 3.56 MB sin pérdidas apreciables en el Macro F1 de prueba (0.9468), garantizando una inferencia de ~60 ms en procesadores móviles.

### 5. Consenso multimodelo
Finalmente, medimos la conveniencia de combinar las tres arquitecturas principales (`EfficientNet-Lite0`, `EfficientNet-B0` y `ShuffleNet-V2`). La votación suave (*Soft Voting*) demostró que la diversidad de sesgos inductivos compensa los puntos ciegos individuales, elevando el Macro F1 a 0.9567 y el Macro Recall a 95.06 %, la cifra más alta alcanzada en todo el proyecto.
