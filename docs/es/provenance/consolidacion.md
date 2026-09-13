# Consolidación y Límites de los Datos

La investigación sobre procedencia y fuga de información permitió trazar una frontera muy clara entre dos tipos de límites en el proyecto: los defectos de código, que se corrigen con ingeniería en unas pocas horas, y los techos de datos, que no se resuelven con más líneas de código ni con modelos más grandes.

Esta página resume los hallazgos consolidados de la auditoría de procedencia y cómo transformaron las decisiones de diseño del sistema.

---

## Lo que quedó demostrado

A través de experimentos cruzados sobre más de 33,000 imágenes, establecimos cuatro conclusiones fundamentales:

1. **El atajo de procedencia es real y medible:**
   El experimento del anillo perimetral demostró que un modelo puede clasificar correctamente el 78.3 % de las imágenes de prueba viendo únicamente el 10 % del borde exterior de la foto, sin observar jamás la lámina foliar ni la lesión. La correlación entre ciertas fuentes y ciertas clases (por ejemplo, tres fuentes que aportan exclusivamente fotos de plantas sanas) permite que la red identifique la sesión fotográfica en lugar de la enfermedad.
2. **Las particiones aleatorias sobreestiman la capacidad del modelo:**
   Cuando se divide el dataset al azar, las mismas cámaras, fondos y condiciones de iluminación se reparten por igual entre entrenamiento y prueba. El modelo obtiene más de 94 % de Macro F1 en ese escenario porque se le evalúa sobre el mismo entorno que ya memorizó.
3. **La evaluación fuera de fuente revela la brecha agrícola:**
   Al evaluar el sistema bajo un protocolo honesto dejando fuentes completas fuera (*Leave-One-Source-Out*), el Macro F1 real se sitúa en torno a **0.56 – 0.60**. Las clases con abundancia de fotos y orígenes variados (como hojas sanas o necrosis letal) resisten bien el cambio de dominio, mientras que las clases con pocas muestras y dominadas por una sola fuente sufren el impacto de la novedad.
4. **Las intervenciones algorítmicas tienen un techo:**
   Probamos equilibrar las fuentes dentro de cada lote, sustituir fondos artificialmente y aplicar aumentaciones severas de color y recorte. Aunque algunas técnicas ofrecieron mejoras marginales en patologías puntuales, ninguna logró cerrar la brecha de fondo. La causa no es que el algoritmo sea malo, sino que los datos de partida no contienen la variabilidad suficiente.

---

## Cómo influyó este hallazgo en el producto final

En lugar de esconder estos resultados o insistir en métricas infladas, utilizamos esta evidencia para blindar la aplicación móvil frente a la realidad del campo:

- **Diseño del visor de cámara con marco guía:** Sabiendo que el modelo es sensible al suelo y al entorno, la app le exige al agricultor llenar un marco central con la hoja y descarta automáticamente el 75 % del área exterior de la foto antes de enviarla a clasificar.
- **Detector de anomalías fuera de dominio (OOD):** Implementamos el filtro de distancia de Mahalanobis para que el sistema aprenda a abstenerse y marcar "Imagen no reconocida" cuando una foto se aleja demasiado de lo que el modelo aprendió con seguridad.
- **Módulo de aporte para agricultores:** La aplicación incluye una pantalla para subir fotografías capturadas en campo local. La única manera de llevar el clasificador al 95 % de generalización real en El Salvador es incorporar gradualmente la variabilidad de nuestras propias parcelas.
