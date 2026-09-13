# Resultados del Pipeline Principal

Esta sección reúne las métricas definitivas del pipeline principal de la Etapa 2, evaluadas sobre el conjunto de prueba independiente de **5,015 imágenes** retenidas y respaldadas por predicciones auditable por imagen.

Más allá de presentar tablas numéricas, el objetivo de estos resultados es ofrecer un balance transparente: qué tanto ganamos optimizando modelos, en qué medida el ensamble mejora la bioseguridad del diagnóstico y cuál es el impacto real cuando enfrentamos los modelos a fuentes de datos que nunca vieron durante el entrenamiento.

---

## Las tres cifras de referencia del proyecto

Para entender el alcance real del sistema desarrollado, basta comparar tres magnitudes fundamentales:

| Configuración evaluada | Macro $F_1$-Score | Exactitud (Accuracy) | Entorno de aplicación |
|---|:---:|:---:|---|
| **`EfficientNet-Lite0` (desplegada)** | **0.9468** | **97.91 %** | Inferencia local offline en la aplicación móvil (3.5 MB, ~60 ms). |
| **Ensamble Soft Voting (3 modelos)** | **`0.9567`** | **`98.29 %`** | Inferencia en la nube / API cuando hay conectividad disponible. |
| **Generalización a fuente no vista** | **`0.6026 ± 0.1240`** | **76.4 %** | Comportamiento honesto frente a cámaras y parcelas desconocidas. |

La distancia entre el primer número (0.9468) y el tercero (0.6026) sintetiza la brecha de dominio en agricultura: dentro de las condiciones conocidas de los datasets el modelo es sobresaliente, pero la variabilidad de un campo completamente nuevo introduce una caída medible que solo se amortigua recolectando datos locales.

---

## La estabilidad entre semillas y el peso de las decisiones

Para saber si una mejora reportada es genuina o simple casualidad estadística, entrenamos la configuración de producción con tres semillas aleatorias distintas sobre la misma partición:

- **Semilla 42 (modelo desplegado):** Macro $F_1$ = **0.9468** (la corrida más alta).
- **Semilla 1:** Macro $F_1$ = 0.9448.
- **Semilla 2:** Macro $F_1$ = 0.9375.
- **Promedio y dispersión:** Media de **0.9430** con una desviación estándar de $\sigma = 0.0049$ (intervalo de confianza al 95 % de $\pm 0.0122$).

![Diferencias reportadas medidas en sigmas](/resultados/diferencias_en_sigmas.png)

Este análisis de dispersión nos dejó una lección metodológica muy valiosa:
- Las mejoras obtenidas ajustando hiperparámetros o combinando modelos en ensamble oscilan entre 1 y 2 desviaciones estándar ($\pm 0.005$ a $+0.009$). Son mejoras útiles y bienvenidas, pero modestas.
- En cambio, la decisión de **cómo partir los datos** (partición estratificada tradicional frente a partición agrupada por fuente) mueve la aguja en más de **66 desviaciones estándar** (una caída de 0.33).

Cómo se evalúa y cómo se estructuran los datos pesa veintisiete veces más que cualquier ajuste fino de hiperparámetros.

---

## Navegación por los resultados detallados

- **[Optimización e hiperparámetros](/es/resultados/optimizacion):** Comparativa del impacto de Optuna entre las tres arquitecturas y por qué `Lite0` conservó sus valores base.
- **[Modelos avanzados y ensamble](/es/resultados/ensamble):** Desglose del voto suave, ganancias en recall (95.06 %) y análisis por clase.
- **[Evaluación rigurosa y métricas finales](/es/resultados/evaluacion):** El experimento cruzado 2×2 entre particiones y análisis de estabilidad.
- **[Análisis de sesgos y ética](/es/resultados/equidad):** Pruebas de oclusión espacial (Clever Hans), equidad entre laboratorio y campo real, y auditoría con Grad-CAM y SHAP.
