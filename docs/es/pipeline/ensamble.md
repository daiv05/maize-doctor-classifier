# Ensamble Multimodelo por Soft Voting

Cuando tres expertos agrícolas observan una misma hoja con síntomas dudosos, es probable que se fijen en detalles diferentes: uno prestará más atención a las manchas cloróticas centrales, otro a las quemaduras del borde y el tercero a la textura de la lámina. Si combinamos sus opiniones, el veredicto final suele ser mucho más acertado que el de cualquiera de ellos por separado.

En aprendizaje profundo ocurre exactamente lo mismo. Para llevar la capacidad predictiva al máximo nivel, diseñamos un sistema de ensamble por votación suave (*Soft Voting*) que combina las tres arquitecturas entrenadas: **`EfficientNet-Lite0`**, **`EfficientNet-B0`** y **`ShuffleNet-V2-x1.0`**.

---

## El principio de diversidad

Un ensamble solo funciona de verdad si los modelos que lo componen cometen errores en cosas distintas. Si combináramos tres modelos idénticos o de la misma familia exacta, todos compartirían las mismas debilidades y el ensamble solo repetiría sus equivocaciones.

Por eso elegimos tres redes con arquitecturas y principios de diseño diferentes:
- **`EfficientNet-B0`** aporta alta capacidad de extracción visual gracias a sus bloques de atención selectiva (Squeeze & Excitation).
- **`EfficientNet-Lite0`** trabaja con convoluciones simplificadas y activaciones lineales preparadas para cuantización, desarrollando representaciones internas más directas.
- **`ShuffleNet-V2`** opera sobre división y barajado de canales, capturando correlaciones espaciales a través de otro camino computacional.

Cada red toma la imagen de 224 × 224 píxeles y calcula su propia distribución de probabilidades sobre las 9 clases. El ensamble toma esas tres distribuciones, calcula el promedio aritmético simple entre ellas y emite el diagnóstico correspondiente a la clase con mayor probabilidad conjunta.

---

## Ganancias en el conjunto de prueba

Al evaluar el ensamble sobre las 5,015 imágenes del conjunto de prueba retenido, la combinación superó a todos los modelos individuales en cada una de las métricas clave:

![Comparativa del Ensamble](/ensemble/ensemble_comparison_bar.png)

| Modelo / Ensamble | Exactitud (Accuracy) | Macro $F_1$-Score | Macro Precision | Macro Recall |
|---|:---:|:---:|:---:|:---:|
| **ShuffleNet-V2-x1.0** | 97.31 % | 0.9330 | 0.9388 | 0.9298 |
| **EfficientNet-Lite0** | 97.91 % | 0.9468 | 0.9533 | 0.9413 |
| **EfficientNet-B0** | 97.97 % | 0.9483 | 0.9589 | 0.9400 |
| **Ensamble Soft Voting** 🏆 | **`98.29 %`** | **`0.9567`** | **`0.9642`** | **`0.9506`** |

La ganancia más relevante desde el punto de vista agronómico se aprecia en el **Macro Recall (95.06 %)**. En patología vegetal, el peor error posible es un falso negativo: decirle a un productor que su planta está sana cuando en realidad tiene una enfermedad contagiosa o una deficiencia que reducirá su cosecha. Al promediar los tres modelos, la tasa de detección global sube a su punto más alto.

![Matriz de Confusión Ensamble](/ensemble/confusion_matrix_ensemble.png)

En patologías críticas como la **necrosis letal del maíz (MLN)**, el ensamble alcanzó un recall perfecto (1.0000 sobre 963 casos de prueba, sin un solo caso no detectado). En plantas sanas y roya común superó el 99 % de precisión. Las deficiencias nutricionales también se beneficiaron del consenso entre modelos, elevando el $F_1$ de nitrógeno a 0.90 y fósforo a 0.96.

---

## Estrategia de despliegue en dos niveles

La existencia de este ensamble abre una arquitectura muy práctica para el despliegue del sistema:

1. **Nivel local en el teléfono (Edge AI):** Para el trabajo diario en la parcela sin cobertura de red, la app ejecuta **`EfficientNet-Lite0` cuantizado a Int8** (3.5 MB), entregando diagnósticos en 60 ms con un Macro F1 de 0.9468.
2. **Nivel en la nube (API para técnicos y extensión):** Cuando el productor cuenta con conexión o cuando técnicos agrícolas del CENTA/MAG consultan el sistema desde una computadora, el backend puede ejecutar el **ensamble completo**, exprimiendo el 95.67 % de Macro F1 y el 98.3 % de exactitud para corroborar casos especialmente complejos.
