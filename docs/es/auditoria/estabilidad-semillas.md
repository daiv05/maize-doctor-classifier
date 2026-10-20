---
title: Estabilidad y Variabilidad Estocástica entre Semillas
description: Análisis de varianza muestral sobre el modelo baseline en tres semillas independientes (42, 123, 2026).
---

# Estabilidad y Variabilidad Estocástica entre Semillas

> **Pregunta experimental:** ¿Cuánto varía el rendimiento del modelo baseline al cambiar la inicialización de pesos y el barajado estocástico del DataLoader? ¿Son reproducibles y estadísticamente significativas las pequeñas mejoras reportadas en el proyecto?

---

## 1. Justificación Metodológica

En aprendizaje profundo, reportar el rendimiento de un modelo a partir de una única corrida (`seed=42`) introduce un riesgo de sobreinterpretación:
* Si la desviación típica ($\sigma$) entre distintas inicializaciones aleatorias ronda el $0.5 \% – 1.0 \%$, cualquier ganancia reportada por debajo de esa magnitud —como la mejora de **$+0.0083$** obtenida por el ensamble o el **$+0.0057$** atribuido a la afinación de `EfficientNet-B0`— no puede distinguirse de una fluctuación aleatoria favorable sin réplicas estocásticas.
* Para responder a esta limitación, se ejecutó un protocolo formal de estabilidad entrenando el modelo de referencia sobre **tres semillas independientes** (`42`, `123`, `2026`) con selección de checkpoint congelada antes de inferir sobre el conjunto de prueba.

---

## 2. Configuración y Protocolo de Evaluación

* **Modelo evaluado:** `EfficientNet-Lite0` (arquitectura seleccionada para el despliegue móvil).
* **Hiperparámetros de referencia:** Contrato del run canónico `20260921_204608` (AdamW, LR $10^{-4}$, weight decay $10^{-4}$, batch size 32, Cosine Annealing con 3 épocas de warmup, función de pérdida ponderada `sqrt_inverse`, label smoothing 0.1, clip norm 1.0, sin sampler y sin CLAHE).
* **Partición evaluada:** Split estratificado de desarrollo `seed_42` (23 400 train / 5 014 val / 5 015 test).
* **Criterio de parada y selección:** Máximo 60 épocas, paciencia de early stopping de 8 épocas. La mejor época se selecciona estrictamente por Macro-$F_1$ en validación; el conjunto de prueba se evalúa **una única vez tras bloquear el checkpoint** con `selection.lock.json` (`evaluation_count=1`).

---

## 3. Resultados Cuantitativos en el Conjunto de Test Retenido

| Semilla | Run ID | Mejor Época | Validación (Macro-$F_1$) | Test Final (Macro-$F_1$) | Test Final (Accuracy) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | `20260921_204608` | 39 | 0.956086 | **0.948002** | 97.81 % |
| **123** | `20261001_140756` | 39 | 0.960281 | **0.937867** | 97.31 % |
| **2026** | `20261001_145025` | 45 | 0.952856 | **0.944597** | 97.75 % |
| **Media $\pm$ SD** | — | — | **0.956408 $\pm$ 0.003723** | **0.943489 $\pm$ 0.005158** | **97.6205 % $\pm$ 0.2722 pp** |

*(La desviación estándar reportada es muestral con $ddof=1$, $n=3$).*

![Dispersión de Macro-F1 por Semilla](/auditoria/baseline_f1_by_seed.png)

---

## 4. Conclusiones Objetivas

1. **Magnitud de la dispersión ($\sigma \approx 0.0052$):**
   La variación estocástica del Macro-$F_1$ en el conjunto de prueba entre diferentes semillas tiene una desviación estándar de **$\pm 0.005158$** (un rango de 1.01 puntos porcentuales entre la semilla más favorable y la más desfavorable).
2. **Interpretación de mejoras finas:**
   Cualquier ganancia experimental por debajo de una centésima ($\Delta F_1 < 0.01$) evaluada con una sola semilla está en el mismo orden de magnitud que la variabilidad estocástica del entrenamiento. Por tanto, no debe presentarse como una ventaja categórica de una arquitectura o técnica si no se demuestra estabilidad pareada multi-semilla.
3. **Consistencia fitosanitaria:**
   A pesar de la fluctuación global de medio punto, la jerarquía de dificultad entre clases se mantuvo idéntica en las tres semillas: las patologías con lesiones bien contrastadas (`lethal_necrosis`, `healthy`, `common_rust`) superaron consistentemente el 0.98 de $F_1$, mientras que las deficiencias nutricionales (`potassium_deficiency`, `nitrogen_deficiency`) concentraron la mayor variabilidad.

---

## 5. Artefactos y Evidencia Versionada

Los datos crudos de este experimento están disponibles en el repositorio:
* Reporte formal completo: [`MULTISEED_SOURCE_STABILITY.md`](../reproducibilidad/evidencia/multiseed_source_stability/MULTISEED_SOURCE_STABILITY.md)
* Tabla de métricas por semilla: [`baseline_multiseed.csv`](../reproducibilidad/evidencia/multiseed_source_stability/baseline_multiseed.csv)
* Desglose por clase fitosanitaria en las 3 semillas: [`baseline_class_multiseed.csv`](../reproducibilidad/evidencia/multiseed_source_stability/baseline_class_multiseed.csv)
* Registro de ejecución y hashes de selección: [`evaluation_registry.csv`](../reproducibilidad/evidencia/multiseed_source_stability/evaluation_registry.csv)
