# Resultados del Ensamble Multimodelo

El ensamble por votación suave (*Soft Voting*) combina las probabilidades de predicción de las tres arquitecturas principales del proyecto: **`EfficientNet-Lite0`**, **`EfficientNet-B0`** y **`ShuffleNet-V2-x1.0`**, seleccionando para cada una su mejor checkpoint entrenado.

Al promediar las distribuciones de confianza de tres modelos con sesgos inductivos dispares, el sistema alcanza los indicadores de desempeño más altos de todo el proyecto sobre las 5,015 imágenes del conjunto de prueba retenido.

---

## Comparativa cuantitativa en el conjunto de prueba

![Modelos individuales frente al ensamble](/ensemble/ensemble_comparison_bar.png)

| Configuración / Modelo | Exactitud (Accuracy) | Macro $F_1$-Score | Macro Precision | Macro Recall | Ganancia vs. Mejor Individual |
|---|:---:|:---:|:---:|:---:|:---:|
| **`ShuffleNet-V2-x1.0`** | 97.31 % | 0.9330 | 0.9388 | 0.9298 | −1.53 pp |
| **`EfficientNet-Lite0`** | 97.91 % | 0.9468 | 0.9533 | 0.9413 | −0.15 pp |
| **`EfficientNet-B0`** | 97.97 % | 0.9483 | 0.9589 | 0.9400 | (Base individual) |
| **Ensamble Soft Voting** 🏆 | **`98.29 %`** | **`0.9567`** | **`0.9642`** | **`0.9506`** | **+0.84 pp netos** 🚀 |

El ensamble supera la barrera del **95.6 % en Macro $F_1$** y alcanza casi un **98.3 % de exactitud diagnóstica**. 

La mejora más valiosa se registra en el **Macro Recall (95.06 %)**. En la práctica agronómica, un falso negativo —no detectar una plaga o deficiencia a tiempo— puede derivar en la pérdida de la cosecha. El consenso entre las tres redes reduce de manera significativa los puntos ciegos individuales, asegurando que los síntomas discretos sean detectados por al menos uno de los modelos.

---

## Diagnóstico fitosanitario por patología

![Matriz de confusión del ensamble](/ensemble/confusion_matrix_ensemble.png)

La matriz de confusión normalizada confirma la solidez del diagnóstico fitosanitario:

- **Enfermedades destructivas:** La necrosis letal del maíz (MLN) alcanzó un recall del **100 %** (963 casos correctamente clasificados de 963). En roya común y tizón foliar (NCLB), el acierto superó el 98 %.
- **Plantas sanas:** Con 99.4 % de precisión y 99.8 % de recall, el modelo prácticamente no confunde hojas enfermas con tejido sano, evitando aplicaciones fitosanitarias innecesarias.
- **Deficiencias nutricionales:** Las tres deficiencias se beneficiaron del ensamble, alcanzando $F_1$ de 0.9057 en nitrógeno y 0.9632 en fósforo. En potasio —la clase con menor cantidad de imágenes disponibles— el $F_1$ se situó en 0.8475.

---

## Dónde aporta más el ensamble

Al desglosar las predicciones por repositorio de procedencia, se observa un patrón claro: el ensamble aporta sus mayores ganancias en fuentes complejas y multi-clase donde los modelos individuales presentaban dudas (como `corn_leaf_roboflow`, con una ganancia de +1.4 pp), mientras que en fuentes homogéneas de una sola clase el rendimiento se mantiene en el techo.

Este comportamiento confirma la viabilidad de la **estrategia dual**: mantener a `EfficientNet-Lite0` como el motor ligero y rápido en el teléfono celular (~60 ms), y reservar el ensamble completo en la nube o API para análisis de confirmación en situaciones fitosanitarias complejas.
