# Evaluación Rigurosa y Métricas Finales

Evaluar un modelo de visión artificial exige distinguir entre dos preguntas muy diferentes: qué tan bien clasifica fotos similares a las que ya vio, y qué tan bien generaliza cuando se enfrenta a un campo o cámara totalmente desconocidos.

En esta sección se presentan los resultados del protocolo de validación cruzada y el análisis comparativo entre la partición estratificada tradicional y la partición agrupada por procedencia.

---

## Las dos realidades del sistema

Al evaluar el modelo desplegado (`EfficientNet-Lite0`), obtenemos dos cifras complementarias que describen dos escenarios operativos:

| Métrica | Macro $F_1$-Score | Protocolo de medición | Qué representa |
|---|:---:|---|---|
| **Rendimiento en prueba retenida** | **0.9468** | Partición estratificada estándar (5,015 fotos) | Rendimiento óptimo en condiciones y cámaras conocidas. |
| **Ensamble en prueba retenida** | **0.9567** | Partición estratificada estándar (5,015 fotos) | Consenso multimodelo de máxima fidelidad diagnóstica. |
| **Generalización fuera de fuente** | **`0.6026 ± 0.1240`** | Validación cruzada agrupada por procedencia | Comportamiento honesto frente a fuentes y parcelas nunca vistas. |

La diferencia de **0.34 puntos de Macro F1** entre la evaluación estándar y la evaluación agrupada es el coste medido de la brecha de dominio en la patología vegetal.

---

## El experimento cruzado 2×2

Para separar el impacto de los hiperparámetros del impacto de la partición de datos, diseñamos un experimento cruzado de validación en 5 pliegues:

![Experimento 2x2 de validación cruzada](/resultados/kfold_2x2.png)

| Configuración de hiperparámetros | Partición estratificada estándar | Partición agrupada por fuente | Caída de rendimiento |
|---|:---:|:---:|:---:|
| **Configuración de producción** | 0.9311 ± 0.0099 | **`0.6026 ± 0.1240`** | **−0.3286** |
| **Configuración de B0** | 0.9463 ± 0.0052 | 0.6499 ± 0.1508 | −0.2964 |
| *Presencia de fuentes en validación* | *14 de 14 (100 % solapadas)* | *0 solapadas con entrenamiento* | — |

**El impacto del protocolo (−0.33) es más de veinte veces mayor que el impacto del ajuste de hiperparámetros (+0.015).**

Este hallazgo es fundamental: demuestra que en proyectos de aprendizaje profundo aplicado, la forma en que se dividen y aíslan los datos determina las métricas con muchísima más fuerza que cualquier cambio sutil en los pesos o en el optimizador.

---

## El significado de los intervalos de confianza

En la partición estratificada tradicional, el intervalo de confianza es sumamente estrecho ($\pm 0.0099$). Sin embargo, esa aparente precisión no describe la robustez del modelo, sino la homogeneidad del protocolo: como las 14 fuentes están repartidas en todos los pliegues, cada pliegue es esencialmente el mismo experimento repetido cinco veces.

En cambio, bajo la partición agrupada por fuente, el intervalo se amplía a $\pm 0.1240$. Esa dispersión mayor refleja la variabilidad genuina del mundo real: cuando el modelo evalúa una fuente con buena iluminación y síntomas claros, el F1 ronda el 0.75; cuando evalúa una fuente con pocas fotos o tonos complejos, el rendimiento baja. 

Aceptar y reportar esa variabilidad es el estándar de honestidad que distingue a este proyecto de aproximaciones académicas que sobreestiman su capacidad de despliegue.
