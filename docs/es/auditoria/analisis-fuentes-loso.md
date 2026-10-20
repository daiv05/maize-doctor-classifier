---
title: Procedencia, Distribución de Fuentes y Evaluación LOSO
description: Análisis de contingencia entre repositorios públicos y prueba de transferencia fuera de fuente (Leave-One-Source-Out).
---

# Procedencia, Distribución de Fuentes y Evaluación LOSO

> **Pregunta experimental:** Cuando el modelo se entrena en una partición estratificada aleatoria, ¿está aprendiendo a reconocer enfermedades o está memorizando patrones visuales de fondo y sensores propios de cada repositorio (atajos visuales)? ¿Qué sucede cuando se evalúa sobre una fuente retenida al 100 %?

---

## 1. Composición de las 11 Fuentes del Corpus Canónico

El inventario del corpus canónico ($N = 33\,429$ muestras elegibles) se compone de 11 repositorios públicos heterogéneos. Al cruzar las fuentes con las 9 clases fitosanitarias se observa una asociación estructural severa:

| Fuente (`source_id`) | Total Muestras | Train | Val | Test | Clases Aportadas | Concentración Principal |
|---|---:|---:|---:|---:|:---:|---|
| `maize-beans-tomatoes-africa` | 12 136 | 8 487 | 1 837 | 1 812 | 4 | Aporta el 61.2 % de `fall_armyworm` |
| `maize-diseases` (PlantVillage) | 6 989 | 4 881 | 1 047 | 1 061 | 3 | **67.8 % es planta sana**; aporta el 54.2 % de `healthy` |
| `multicrop-disease-maiz` | 5 816 | 4 072 | 864 | 880 | 3 | **55.6 % es necrosis letal**; aporta el 50.4 % de `lethal_necrosis` |
| `corn-leaf-roboflow` | 2 919 | 2 081 | 400 | 438 | 6 | Aporta el 50.1 % de nitrógeno y 53.2 % de fósforo |
| `cropdg-unified-multidomain` | 2 562 | 1 776 | 410 | 376 | 3 | Lab y campo (Sudáfrica/China) |
| `maize-in-field-dataset` | 852 | 593 | 123 | 136 | 3 | Tomas de campo en Sudáfrica |
| `maize-2-roboflow` | 846 | 593 | 126 | 127 | 3 | Aporta el 49.6 % de potasio |
| 4 fuentes menores restantes | 1 309 | 917 | 207 | 185 | 1–4 | Muestras complementarias |

### Cuantificación Estadística de Asociación (V de Cramér):
* En la matriz de contingencia de 11 fuentes $\times$ 9 clases, **65 de las 99 celdas posibles están vacías** (muchas fuentes solo contienen 1, 3 o 4 clases).
* El coeficiente **V de Cramér es de 0.5030** ($\chi^2 = 67\,673.21$). Esto confirma una asociación moderada-fuerte entre el origen de la foto y la patología, un factor intrínseco de los datasets públicos disponibles a nivel mundial.

![Distribución de Muestras por Fuente](/auditoria/source_sizes.png)

---

## 2. Protocolo de Evaluación Fuera de Fuente (LOSO)

Para determinar si el modelo colapsa ante fuentes no vistas, se implementó el protocolo *Leave-One-Source-Out* (LOSO):
1. Se excluye al 100 % una fuente completa durante la preparación, entrenamiento y selección de hiperparámetros.
2. El modelo se entrena sobre las fuentes restantes y se valida únicamente con ellas.
3. Se bloquea el checkpoint óptimo y se evalúa sobre la fuente excluida completa (que actúa como conjunto de prueba externo retenido).
4. El protocolo se ejecutó sobre las dos fuentes mayoritarias en **tres semillas independientes** (`42`, `123`, `2026`).

---

## 3. Resultados Cuantitativos de las Fuentes Prioritarias

| Escenario Evaluado | Imágenes en Train | Imágenes en Holdout Externo | Macro-$F_1$ en Validación (In-Dist) | Macro-$F_1$ Externo (Holdout LOSO) | Conclusión de Transferencia |
|---|:---:|:---:|:---:|:---:|---|
| **Exclusión de `maize-diseases`** | 18 519 | **6 989** | 0.9979 (ref. interna) | **0.993264 $\pm$ 0.000600** | **Excelente:** Transferencia casi perfecta a condiciones PlantVillage no vistas. |
| **Exclusión de `multicrop-disease-maiz`** | 19 328 | **5 816** | 0.9972 (ref. interna) | **0.901906 $\pm$ 0.004422** | **Caída identificable:** Sensible al cambio de dominio en `lethal_necrosis`. |

*(Resultados promedio $\pm$ desviación estándar muestral en 3 semillas, $n=3$).*

### Desglose por Semilla en `multicrop-disease-maiz`:
* **Seed 42:** 0.905490
* **Seed 123:** 0.896964
* **Seed 2026:** 0.903264
* **Media $\pm$ SD:** **$0.901906 \pm 0.004422$**

---

## 4. Conclusiones Objetivas

1. **La brecha fuera de fuente no es uniforme:**
   La hipótesis de que el clasificador se desplomaría por igual ante cualquier repositorio nuevo queda refutada: ante `maize-diseases`, el modelo generaliza con un Macro-$F_1$ de 0.993. 
2. **Causa de la caída en `multicrop`:**
   La caída a 0.902 en `multicrop-disease-maiz` se concentra en la patología `lethal_necrosis`. Dado que esta fuente aporta más de la mitad de todos los casos de necrosis letal del corpus, excluirla altera drásticamente la distribución de entrenamiento de esa patología específica.
3. **Implicación para el despliegue:**
   Para producción en parcelas de El Salvador, el clasificador cuenta con el respaldo del detector fuera de dominio (OOD Mahalanobis) para rechazar imágenes que se desvíen morfológicamente del espacio de entrenamiento.

---

## 5. Artefactos y Evidencia Versionada

* Reporte formal de análisis por fuente: [`SOURCE_ANALYSIS.md`](../reproducibilidad/evidencia/source_analysis/SOURCE_ANALYSIS.md)
* Predicciones individuales por muestra: [`baseline_test_predictions.csv`](../reproducibilidad/evidencia/source_analysis/baseline_test_predictions.csv) (1.17 MB)
* Reporte del protocolo LOSO: [`LOSO_RESULTS.md`](../reproducibilidad/evidencia/loso_baseline/LOSO_RESULTS.md)
* Evaluación multi-semilla LOSO: [`MULTISEED_SOURCE_STABILITY.md`](../reproducibilidad/evidencia/multiseed_source_stability/MULTISEED_SOURCE_STABILITY.md)
