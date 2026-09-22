# Fase 2 — Evaluación de Intervenciones y Límite de Datos

Una vez demostrada la caída de rendimiento al evaluar sobre fuentes desconocidas, el siguiente paso fue investigar si esa brecha podía cerrarse mediante técnicas de intervención sobre los datos respaldadas por la literatura científica.

Exploramos dos estrategias fundamentales bajo el mismo protocolo de validación fuera de fuente: el **balanceo de grupos de procedencia** y la **sustitución sintética de fondos (*BackMix*)**.

---

## Las dos intervenciones evaluadas

1. **Balanceo de grupos de procedencia:**
   En lugar de permitir que las fuentes dominantes acaparen el entrenamiento de una clase (como ocurría con *Healthy*, donde el 80 % provenía de tres repositorios monocromáticos), forzamos una distribución equilibrada entre todas las fuentes disponibles para cada patología. El objetivo era evitar que la red aprendiera a asociar una etiqueta con la firma visual de una única cámara.
2. **Sustitución de fondos (*BackMix*):**
   Para romper la correlación entre la patología y el fondo del laboratorio o del campo, sustituimos el fondo de las imágenes por entornos tomados de otras fotografías del dataset. Esta técnica buscaba obligar a la red a concentrarse en la lámina foliar, desacoplándola del entorno exterior.

---

## Resultados medidos

Ambas intervenciones fueron evaluadas sobre los 11 pliegues de validación fuera de fuente:

| Estrategia evaluada | Macro $F_1$-Score | Exactitud (Accuracy) | Impacto frente a la línea base |
|---|:---:|:---:|:---:|
| **Línea base (fuera de fuente)** | **0.5573** | **68.84 %** | — |
| **Con balanceo de grupos** | **0.5594** | **70.09 %** | +0.0021 (dentro del ruido) |
| **Con sustitución de fondo (BackMix)** | **0.5609** | **69.81 %** | +0.0036 (dentro del ruido) |

Ambas mejoras se ubicaron por debajo del umbral de desviación estadística entre semillas ($\sigma \approx 0.007$). Ninguna de las dos intervenciones logró una recuperación apreciable en las clases más afectadas por la brecha de dominio.

---

## Interpretación y límite de esta campaña

El resultado observado en esta fase fue:

- **El balanceo no crea diversidad:** Repartir cupos entre repositorios no genera variedad donde no la hay; si una fuente contiene fondos uniformes y artificiales, submuestrearla no altera la firma visual de fondo.
- **Los atajos no son solo el fondo:** El atajo visual que utiliza la red no se limita a si el fondo es negro o verde; abarca la resolución del sensor, la relación de compresión JPEG, el ángulo de captura y la variedad botánica del maíz.

Estas dos intervenciones no cerraron la brecha dentro del presupuesto y protocolo probados. El resultado respalda priorizar recolección genuina en campo salvadoreño, pero no demuestra que toda intervención algorítmica futura sea inútil. Las cifras son históricas y deben leerse con sus artefactos LOSO.
