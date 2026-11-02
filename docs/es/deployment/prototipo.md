# Prototipo en dispositivo

El prototipo de DoctorMaiz se probó en un Motorola edge 40 neo (MediaTek Dimensity 7030, Android 15) con el modelo `efficientnet_lite0` cuantizado a int8 (3.74 MB), sin conexión.

## Uso de la app

![Pantalla de inicio y cámara guiada](/app/home.png)

La pantalla de inicio da acceso directo al escáner y al historial de análisis.

![Cámara con el marco guía](/app/camara-marco-guia.png)

En el campo la foto suele tomarse desde un metro de distancia y la hoja ocupa una parte pequeña del encuadre. La cámara muestra un marco guía con silueta de hoja: el productor se acerca a unos 20 o 30 centímetros hasta que la lámina llene el marco, y la app recorta esa región central antes de procesarla.

![Pantalla de resultado con diagnóstico](/app/resultado-mancha-gris.png)

La pantalla de resultado muestra el diagnóstico, la confianza, la foto procesada y recomendaciones de manejo.

## Tiempos de respuesta

Medidos en el teléfono con 143 imágenes (95 fotos de campo externas y 48 del corpus). La meta es que preprocesado e inferencia sumen menos de 300 ms.

| Etapa | Mediana | Qué incluye |
|---|:---:|---|
| Preprocesado | 98 ms (p95 279 ms) | Decodificación, corrección de orientación, reescalado con antialiasing y armado del tensor |
| Inferencia | 23 ms | Ejecución del modelo cuantizado en la CPU del teléfono |

La inferencia no es el cuello de botella. El resto del tiempo hasta el resultado corresponde a la interfaz, la escritura en SQLite local y el renderizado.

### Coincidencia con el servidor

La primera medición coincidía con el computador en el 84 % de las predicciones. Las imágenes de entre 224 y 896 px por lado se escalaban en un solo paso bilineal sin prefiltro, mientras que el entrenamiento usa el redimensionado con antialias de Pillow. Al pasar el tensor del teléfono por el int8 del servidor se obtuvieron las mismas predicciones, de modo que el intérprete no era la causa. La app incorpora ahora el remuestreo de Pillow (`src/ml/resample.ts`, comprobado byte a byte contra su salida) y el teléfono coincide con el int8 del servidor en las 143 imágenes.

| Lote | Aciertos en el teléfono | Rechazo OOD en el teléfono |
|---|---:|---:|
| Corpus (48) | 45 (93.8 %) | 8.3 % |
| Fotos de campo externas (95) | 38 (40.0 %) | 52.6 % |

Las cifras de campo coinciden con las del computador y reflejan la limitación descrita en [Resultados](../resultados/index.md). La paridad de muestra de 30 imágenes del int8 en el servidor no se aprobó por una imagen límite (máx. |Δprob| 0.198 frente al umbral 0.15).

## Detector fuera de dominio

![Resultados y rechazo de imágenes fuera de dominio](/app/resultado-potasio.png)

Cuando la imagen no corresponde a maíz o es poco clara, la app no fuerza una etiqueta: el detector OOD marca el resultado como no reconocido y sugiere repetir la captura con mejor iluminación o enfoque.

![Imagen rechazada por el detector](/app/resultado-no-reconocida.png)

El detector del modelo oficial usa un umbral de 106.37 (percentil 95 de validación): detecta 5 de 5 imágenes sintéticas (negro, blanco, gris, ruido y azul) y rechaza el 3.8 % de 500 imágenes de prueba. La evidencia de la exportación está en [`export_movil_20260921_204608/`](../reproducibilidad/evidencia/export_movil_20260921_204608/export_summary_int8.json).

## Contribución al dataset

De las 11 fuentes del corpus (`source_id`), casi todas provienen de África, India o laboratorios de Estados Unidos; ninguna se capturó en El Salvador.

![Pantalla para contribuir al dataset](/app/contribuir.png)

Cuando hay conexión, la app permite compartir fotos con su diagnóstico para formar un banco de datos nacional. Esas imágenes son la vía para cerrar la brecha entre las fotos de datasets públicos y las de parcelas locales.
