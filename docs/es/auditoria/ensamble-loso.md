---
title: Evaluación del Ensamble Multimodelo Fuera de Dominio (LOSO)
description: Prueba de estrés del ensamble Soft Voting excluyendo una fuente completa en tres semillas independientes.
---

# Evaluación del Ensamble Multimodelo Fuera de Dominio (LOSO)

> **Pregunta experimental:** En la partición estratificada estándar, el ensamble por Soft Voting supera al mejor modelo individual por $+0.0083$ (alcanzando 0.9567 de Macro-$F_1$, cifra histórica). ¿Se mantiene esta ventaja competitiva cuando el ensamble se enfrenta a una fuente completa no vista durante el entrenamiento y la selección?

---

## 1. Diseño Experimental de la Prueba de Estrés

Para responder si el ensamble ofrece una ganancia robusta o si solo funciona bajo fuentes conocidas:

1. **Partición de exclusión completa:** Se excluyó al 100 % el repositorio `multicrop-disease-maiz` (quedando 19 328 imágenes de entrenamiento, 4 150 de validación y **5 816 imágenes en el holdout externo**).
2. **Reentrenamiento de los tres modelos desde cero:**
   * `EfficientNet-Lite0`: Configuración baseline (Batch 32, LR $10^{-4}$).
   * `EfficientNet-B0`: Configuración afinada (Batch 64, LR $4.55 \times 10^{-4}$).
   * `ShuffleNet-V2-x1.0`: Configuración afinada (Batch 64, LR $4.55 \times 10^{-4}$).
3. **Réplicas en tres semillas:** Se entrenaron y evaluaron con `seed=42`, `seed=123` y `seed=2026`.
4. **Regla de ensamble:** Promedio uniforme de las distribuciones de probabilidad softmax:
   $$P_{\text{ensamble}}(y \mid x) = \frac{1}{3} \sum_{m=1}^{3} \text{softmax}(z_m(x))$$
   No se reajustaron pesos utilizando las imágenes del holdout.

---

## 2. Comparativa Pareada Frente al Mejor Modelo Individual (`EfficientNet-B0`)

Se evalúan las 5 816 imágenes del conjunto retenido externo sobre las 3 clases presentes en esta fuente (`common_rust`, `fall_armyworm`, `lethal_necrosis`):

| Semilla | `EfficientNet-Lite0` | `EfficientNet-B0` (Referencia Individual) | `ShuffleNet-V2` | **Ensamble Soft Voting** | **$\Delta$ Pareado (Ensamble − B0)** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | 0.905490 | 0.917318 | 0.913037 | **0.930055** | **+0.012737** (Mejora) |
| **123** | 0.896964 | **0.934785** | 0.886505 | **0.931569** | **−0.003216** (Empeora) |
| **2026** | 0.903264 | 0.919831 | 0.888009 | **0.923713** | **+0.003883** (Mejora) |
| **Media $\pm$ SD** | 0.901906 $\pm$ 0.004422 | **0.923978 $\pm$ 0.009443** | 0.895850 $\pm$ 0.014896 | **0.928446 $\pm$ 0.004168** | **+0.004468 $\pm$ 0.007993** |

*(Desviaciones estándar muestrales con $ddof=1$, $n=3$).*

![Deltas Pareados del Ensamble vs B0](/auditoria/ensemble_paired_deltas.png)

---

## 3. Métricas de Calibración y Dinámica de Errores

| Métrica Evaluada | `EfficientNet-B0` (Individual) | Ensamble Soft Voting | Impacto del Ensamble |
|---|:---:|:---:|:---:|
| **Macro-$F_1$ (3 clases evaluables)** | 0.923978 $\pm$ 0.009443 | **0.928446 $\pm$ 0.004168** | $+0.004468 \pm 0.007993$ |
| **Exactitud (Accuracy)** | 83.28 % $\pm$ 2.00 pp | **84.59 % $\pm$ 0.64 pp** | $+1.31 \text{ pp} \pm 1.64 \text{ pp}$ |
| **Error de Calibración Esperado (ECE, 15 bins)** | **0.044573 $\pm$ 0.012196** | **0.089088 $\pm$ 0.006372** | **Empeora al doble (+0.0445)** |

### Análisis de Desacuerdos por Muestra:
* En la semilla 42, el ensamble corrigió 250 errores de B0 e introdujo 78 nuevos errores netos (+172 aciertos netos).
* En la **semilla 123**, el ensamble corrigió 150 errores de B0 pero **introdujo 169 nuevos errores** (−19 aciertos netos). Por esta razón, el Macro-$F_1$ del ensamble cayó por debajo del modelo individual.
* En la semilla 2026, corrigió 194 e introdujo 118 (+76 aciertos netos).

![Matriz de Confusión Normalizada del Ensamble bajo LOSO](/auditoria/ensemble_confusion_normalized.png)

---

## 4. Conclusiones Objetivas

1. **La ventaja del ensamble es mixta e inestable fuera de dominio:**
   A diferencia del test de desarrollo estratificado (donde el ensamble superaba a todos los modelos de forma consistente), ante una fuente no vista la mejora promedio se redujo a la mitad ($+0.0045$) y su desviación típica ($\pm 0.0080$) resultó mayor que la propia mejora.
2. **Degradación de la calibración:**
   Promediar probabilidades de modelos no calibrados fuera de dominio incrementa la sobreconfianza en las predicciones erróneas, duplicando el ECE (de 4.4 % a 8.9 %).
3. **Implicación para el despliegue del sistema:**
   Estos resultados respaldan la decisión de no sobrecargar la arquitectura móvil con ensambles pesados fuera de dominio: un modelo individual bien optimizado y calibrado (`EfficientNet-Lite0` o `B0`) ofrece mayor predictibilidad ante fuentes no observadas.

---

## 5. Artefactos y Evidencia Versionada

* Reporte formal del protocolo: [`ENSEMBLE_LOSO_REPORT.md`](../reproducibilidad/evidencia/ensemble_loso/ENSEMBLE_LOSO_REPORT.md)
* Desglose de desacuerdos muestra por muestra: [`ensemble_disagreements.csv`](../reproducibilidad/evidencia/ensemble_loso/ensemble_disagreements.csv) (**1.13 MB**, 5 816 filas)
* Resultados y deltas pareados: [`ensemble_loso_results.csv`](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_results.csv)
* Resumen estadístico JSON: [`ensemble_loso_summary.json`](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_summary.json)
