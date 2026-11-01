# Resultados del pipeline principal

Esta sección reúne resultados vigentes e históricos. Cada fila indica su protocolo: compartir un test de 5 015 imágenes en etapas distintas no garantiza compartir la misma materialización.

Más allá de presentar tablas numéricas, el objetivo de estos resultados es ofrecer un balance transparente: qué tanto ganamos optimizando modelos, en qué medida el ensamble mejora la bioseguridad del diagnóstico y cuál es el impacto real cuando enfrentamos los modelos a fuentes de datos que nunca vieron durante el entrenamiento.

---

## Cifras de referencia con estado

Para entender el alcance real del sistema desarrollado, basta comparar tres magnitudes fundamentales:

| Configuración evaluada | Macro $F_1$-Score | Exactitud (Accuracy) | Entorno de aplicación |
|---|:---:|:---:|---|
| **Lite0 oficial `20260921_204608`** | **0.9480** | **97.81 %** | VIGENTE: `seed_42` estratificado; modelo de la app. |
| Lite0 de la etapa 2 `20260812_221429` | 0.9468 | 97.91 % | HISTÓRICO: app hasta el 2026-11-01. |
| Ensamble Soft Voting (3 modelos) | 0.9567 | 98.29 % | HISTÓRICO: test estratificado. |
| CV agrupada por fuente, configuración base | 0.6026 (IC95 ±0.1240) | 79.69 % (IC95 ±11.95 pp) | HISTÓRICO: cinco pliegues, 5–8 clases presentes; Macro-F1 convencional 0.4359. |

La caída de la CV agrupada evidencia sensibilidad histórica al dominio de origen. No es LOSO ni el holdout `seed_42_source_grouped`, y no debe restarse directamente al baseline actual como si fuera una comparación pareada.

## Reentrenamiento sobre el split corregido

El `EfficientNet-Lite0` de la corrida `20260921_204608` es el modelo oficial sobre la materialización corregida de `seed_42` (33 429 imágenes). Alcanzó **0.9480 de Macro-F1** y **97.81 % de accuracy** en 5 015 imágenes de prueba; su exportación TFLite int8 (3.74 MB) obtiene **0.9489 de Macro-F1** y **97.87 %** sobre el mismo test. Es el modelo que despliega la app desde el 2026-11-01, con su detector fuera de distribución (umbral 106.37). Sigue pendiente la validación en fuentes no vistas y la auditoría de duplicados perceptuales.

La ficha completa incluye hashes, métricas por clase, calibración y errores dominantes: [baseline `20260921_204608`](/es/resultados/run-20260921-efficientnet-lite0).

---

## La estabilidad entre semillas y el peso de las decisiones

El siguiente análisis es **HISTÓRICO** y corresponde a la configuración de la etapa 2; no es el multi-seed del HPO ni el de estabilidad:

- **Semilla 42 (modelo de la etapa 2):** Macro $F_1$ = **0.9468** (la corrida más alta).
- **Semilla 1:** Macro $F_1$ = 0.9448.
- **Semilla 2:** Macro $F_1$ = 0.9375.
- **Promedio y dispersión:** Media de **0.9430** con una desviación estándar de $\sigma = 0.0049$ (IC95 con semiamplitud 0.0122).

![Diferencias reportadas medidas en sigmas](/resultados/diferencias_en_sigmas.png)

Este análisis de dispersión nos dejó una lección metodológica muy valiosa:
- Las mejoras obtenidas ajustando hiperparámetros o combinando modelos en ensamble oscilan entre 1 y 2 desviaciones estándar (de 0.005 a 0.009). Son mejoras útiles y bienvenidas, pero modestas.
- En cambio, la decisión de **cómo partir los datos** (partición estratificada tradicional frente a partición agrupada por fuente) produce una caída de 0.33, unas **41 veces** la desviación muestral de los pliegues estratificados (0.0080; `kfold_estratificado_actual_pliegues.csv`).

En aquel experimento, el protocolo tuvo un efecto mucho mayor que el ajuste fino. La comparación en desviaciones es descriptiva entre estudios históricos, no un test causal.

---

## Limitación: fotos de campo externas al corpus

El modelo oficial se evaluó (versión FP32 en PC, imagen completa) sobre 97 fotos de campo externas al corpus, organizadas por clase; una de ellas está en el corpus de entrenamiento. Acertó **41 de 97 (42 %)** y el detector fuera de distribución rechazó **49 de 97**. El Lite0 de la etapa 2 acierta 39 de 97 en las mismas fotos, de modo que la brecha no se debe al cambio de modelo: es la generalización fuera del corpus, coherente con el benchmark source-grouped y con el LOSO.

| Clase | Fotos | Aciertos |
|---|---:|---:|
| `gray_leaf_spot` | 10 | 10 |
| `fall_armyworm` | 10 | 10 |
| `nitrogen_deficiency` | 10 | 6 |
| `northern_corn_leaf_blight` | 10 | 5 |
| `potassium_deficiency` | 9 | 4 |
| `common_rust` | 18 | 2 |
| `phosphorus_deficiency` | 10 | 2 |
| `healthy` | 10 | 2 |
| `lethal_necrosis` | 10 | 0 |

Las fotos no se versionan. Los 41 aciertos sobre 97 no son comparables con el Macro-F1 de 0.9480 del test: aquel mide fuentes presentes en el entrenamiento y este mide imágenes de otro origen.

---

## Navegación por los resultados detallados

- **[Optimización e hiperparámetros](/es/resultados/optimizacion):** Comparativa del impacto de Optuna entre las tres arquitecturas y por qué `Lite0` conservó sus valores base.
- **[Modelos avanzados y ensamble](/es/resultados/ensamble):** Desglose del voto suave, ganancias en recall (95.06 %) y análisis por clase.
- **[Evaluación rigurosa y métricas finales](/es/resultados/evaluacion):** El experimento cruzado 2×2 entre particiones y análisis de estabilidad.
- **[Baseline EfficientNet-Lite0 `20260921_204608`](/es/resultados/run-20260921-efficientnet-lite0):** Reentrenamiento sobre el split corregido, trazabilidad y diagnóstico de errores.
- **[Benchmark source-grouped `20260921_180112`](/es/resultados/run-20260921-source-grouped):** resultado negativo útil bajo fuentes retenidas.
- **[Análisis de sesgos y ética](/es/resultados/equidad):** Pruebas de oclusión espacial (Clever Hans), equidad entre laboratorio y campo real, y auditoría con Grad-CAM y SHAP.
