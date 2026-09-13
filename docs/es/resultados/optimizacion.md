# Resultados de Optimización de Hiperparámetros

Uno de los resultados más ilustrativos de esta etapa fue descubrir que una configuración de hiperparámetros que resulta fantástica para un modelo puede perjudicar a otro, incluso si ambos pertenecen a la misma familia conceptual.

En esta sección se presentan los resultados comparativos entre las configuraciones por defecto del pipeline y las descubiertas mediante la optimización bayesiana con Optuna sobre las tres arquitecturas evaluadas: **`EfficientNet-Lite0`**, **`EfficientNet-B0`** y **`ShuffleNet-V2-x1.0`**.

---

## Resultados en el conjunto de prueba retenido

Al evaluar las configuraciones en el conjunto de prueba independiente (5,015 imágenes), el impacto de los hiperparámetros dependió estrechamente de la estructura interna de cada red:

![Las tres arquitecturas bajo la configuración de Optuna](/resultados/optimizacion_tres_arquitecturas.png)

| Arquitectura | Configuración por defecto (Test $F_1$) | Configuración optimizada Optuna (Test $F_1$) | Impacto neto | Decisión adoptada |
|---|:---:|:---:|:---:|---|
| **`EfficientNet-B0`** | 0.9426 | **`0.9483`** | **+0.57 pp** | Adoptada para el ensamble |
| **`ShuffleNet-V2-x1.0`** | 0.9237 | **`0.9330`** | **+0.93 pp** | Adoptada para el ensamble |
| **`EfficientNet-Lite0`** | **`0.9468`** | 0.9343 | **−1.25 pp** | **Conserva valores base** |

Para `EfficientNet-B0` y `ShuffleNet-V2`, aplicar una tasa de aprendizaje más ágil ($4.55 \times 10^{-4}$) junto con un tamaño de lote de 64 estabilizó los gradientes y mejoró la capacidad de generalización en el conjunto de prueba.

En cambio, para **`EfficientNet-Lite0`**, la misma intervención redujo el Macro F1 en más de un punto porcentual.

---

## Por qué `Lite0` prefirió su configuración base

La discrepancia entre `B0` y `Lite0` tiene una explicación arquitectónica directa:

- **Estructura de las capas:** `EfficientNet-Lite0` sustituye los bloques de atención Squeeze & Excitation y las activaciones suaves *swish* por operaciones lineales y ReLU6, con el fin de permitir una cuantización entera limpia a 8 bits.
- **Sensibilidad al ritmo de aprendizaje:** Al carecer de mecanismos de recalibración de canales, la red ligera es mucho más sensible a tasas de aprendizaje elevadas, sufriendo oscilaciones en los gradientes que terminan degradando el aprendizaje en las clases minoritarias.
- **Dinámica de convergencia:** Mientras que los modelos con atención alcanzan su pico antes de la época 25, `Lite0` converge de forma más pausada y continua, alcanzando su mejor rendimiento en la **época 35** bajo los hiperparámetros base (tasa de $10^{-4}$ y lote de 32).

Este contraste confirmó la decisión de producción: **`EfficientNet-Lite0` se mantiene en sus valores por defecto (Macro F1 de 0.9468)**, garantizando la máxima fidelidad en el modelo que se exporta a la aplicación móvil, mientras que `B0` y `ShuffleNet` adoptan la configuración optimizada para sumar potencia al ensamble.
