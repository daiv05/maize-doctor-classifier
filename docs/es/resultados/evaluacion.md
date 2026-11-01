# Evaluación histórica: validación cruzada 2×2

Evaluar un modelo de visión artificial exige distinguir entre dos preguntas muy diferentes: qué tan bien clasifica fotos similares a las que ya vio, y qué tan bien generaliza cuando se enfrenta a un campo o cámara totalmente desconocidos.

Esta sección conserva el experimento histórico de validación cruzada de cinco pliegues. No es la evaluación final del baseline `20260921_204608`, no es LOSO y no es el holdout `seed_42_source_grouped`.

---

## Las dos realidades del sistema

Al evaluar el `EfficientNet-Lite0` de la etapa 2 (desplegado en la app hasta el 2026-11-01), obtenemos dos cifras complementarias que describen dos escenarios operativos:

| Métrica | Macro $F_1$-Score | Protocolo de medición | Qué representa |
|---|:---:|---|---|
| **Rendimiento en prueba retenida** | **0.9468** | Partición estratificada estándar (5,015 fotos) | Rendimiento óptimo en condiciones y cámaras conocidas. |
| **Ensamble en prueba retenida** | **0.9567** | Partición estratificada estándar (5,015 fotos) | Consenso multimodelo de máxima fidelidad diagnóstica. |
| **Generalización entre fuentes** | **`0.6026`** (IC95 ±0.1240) | CV de cinco pliegues agrupada por procedencia | Macro-F1 sobre las clases presentes; cada pliegue contiene 5–8 clases. El Macro-F1 convencional es 0.4359. |

La diferencia de aproximadamente **0.34 puntos** es evidencia histórica de cambio de dominio, pero no una estimación causal ni una comparación del baseline actual: las métricas proceden de protocolos y soportes distintos.

---

## El experimento cruzado 2×2

Para separar el impacto de los hiperparámetros del impacto de la partición de datos, diseñamos un experimento cruzado de validación en 5 pliegues:

![Experimento 2x2 de validación cruzada](/resultados/kfold_2x2.png)

| Configuración de hiperparámetros | Partición estratificada estándar | Partición agrupada por fuente | Caída de rendimiento |
|---|:---:|:---:|:---:|
| **Configuración de producción** | 0.9311 (IC95 ±0.0099) | **`0.6026`** (IC95 ±0.1240) | **−0.3286** |
| **Lite0 con hiperparámetros de B0** | 0.9463 (IC95 ±0.0052) | 0.6499 (IC95 ±0.1508) | −0.2964 |
| *Presencia de fuentes en validación* | *todas solapadas con entrenamiento* | *0 solapadas con entrenamiento* | — |

**El impacto del protocolo (−0.33) es más de veinte veces mayor que el impacto del ajuste de hiperparámetros (+0.015).**

Este hallazgo es fundamental: demuestra que en proyectos de aprendizaje profundo aplicado, la forma en que se dividen y aíslan los datos determina las métricas con muchísima más fuerza que cualquier cambio sutil en los pesos o en el optimizador.

---

## El significado de los intervalos de confianza

En la partición estratificada tradicional, el intervalo de confianza del 95 % es sumamente estrecho (semiamplitud $0.0099$). Sin embargo, esa aparente precisión no describe la robustez del modelo, sino la homogeneidad del protocolo: como las fuentes están repartidas en todos los pliegues, cada pliegue es esencialmente el mismo experimento repetido cinco veces.

En cambio, bajo la partición agrupada por fuente, el intervalo se amplía a una semiamplitud de $0.1240$. Esa dispersión refleja sensibilidad a qué fuentes quedan retenidas. El Macro-F1 publicado es “evaluable” porque solo 5–8 clases tienen soporte según el pliegue; contando como cero las clases ausentes, el Macro-F1 convencional de la misma validación es 0.4359.

La lectura válida es metodológica: fuente y soporte deben declararse junto a la media. Los resultados actuales y LOSO se reportan por separado en [protocolos experimentales](/es/metodologia/protocolos-experimentales).
