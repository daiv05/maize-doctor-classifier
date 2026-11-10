# Prototipo en dispositivo

El prototipo de DoctorMaiz se probó en un Motorola edge 40 neo (MediaTek Dimensity 7030, Android 15) con el modelo `efficientnet_lite0` cuantizado a int8 (3.74 MB), sin conexión.

## Uso de la app

![Pantalla de inicio y cámara guiada](/app/home.png)

La pantalla de inicio da acceso directo al escáner y al historial de análisis.

![Cámara con el marco guía](/app/camara-marco-guia.png)

La cámara muestra un marco guía con silueta de hoja (260 × 320 dp): el productor se acerca hasta que la lámina llene el marco, y la app recorta esa región central, con un 15 % de margen, antes de procesarla. Las fotos elegidas de la galería no pasan por ese recorte.

![Pantalla de resultado con diagnóstico](/app/resultado-mancha-gris.png)

La pantalla de resultado muestra el diagnóstico, la confianza, la foto procesada y recomendaciones de manejo.

## Tiempos de respuesta

Medidos en el teléfono con 143 imágenes (95 fotos de campo externas y 48 del corpus). La meta es que preprocesado e inferencia sumen menos de 300 ms.

| Etapa | Mediana | Qué incluye |
|---|:---:|---|
| Preprocesado | 98 ms (p95 279 ms) | Decodificación, corrección de orientación, reescalado con antialiasing y armado del tensor |
| Inferencia | 23 ms | Ejecución del modelo cuantizado en la CPU del teléfono |

La inferencia no es el cuello de botella: el preprocesado tarda unas cuatro veces más.

### Coincidencia con el servidor

La primera medición coincidía con el computador en el 84 % de las predicciones. Las fotos con algún lado menor de 448 px se escalaban a 224 × 224 en un solo paso bilineal sin prefiltro, mientras que el entrenamiento usa el redimensionado con antialias de Pillow. Al pasar el tensor del teléfono por el int8 del servidor se obtuvieron las mismas predicciones, de modo que el intérprete no era la causa. La app incorpora ahora el remuestreo de Pillow para el paso final (`src/ml/resample.ts`, comprobado byte a byte contra su salida en las pruebas; los lados de 896 px o más se reducen antes a la mitad con Skia) y el teléfono coincide con el int8 del servidor en las 143 imágenes.

| Lote | Aciertos en el teléfono | Rechazo OOD en el teléfono |
|---|---:|---:|
| Corpus (48) | 45 (93.8 %) | 8.3 % |
| Fotos de campo externas (95) | 38 (40.0 %) | 52.6 % |

En las mismas 95 fotos de campo el modelo FP32 en el computador acierta 40 y su detector OOD marca 49; las cifras del teléfono reflejan la limitación descrita en [Resultados](../resultados/index.md). La paridad de muestra de 30 imágenes del int8 en el servidor no se aprobó por una imagen límite (máx. |Δprob| 0.198 frente al umbral 0.15).

## Detector fuera de dominio

![Resultados y rechazo de imágenes fuera de dominio](/app/resultado-potasio.png)

Cuando la imagen no corresponde a maíz o es poco clara, la app no fuerza una etiqueta: marca el resultado como no reconocido si la confianza es menor que 0.6, si el margen entre las dos clases más probables es menor que 0.15 o si el detector OOD supera su umbral, y sugiere repetir la captura enfocando una sola hoja que llene el marco. La imagen rechazada no se guarda en el historial.

![Imagen rechazada por el detector](/app/resultado-no-reconocida.png)

El detector del modelo oficial usa un umbral de 106.37 (percentil 95 de validación): detecta 5 de 5 imágenes sintéticas (negro, blanco, gris, ruido y azul) y rechaza el 3.8 % de 500 imágenes de prueba ([validación](../reproducibilidad/evidencia/export_movil_20260921_204608/ood_validation.json)). La evidencia de la exportación está en [`export_movil_20260921_204608/`](../reproducibilidad/evidencia/export_movil_20260921_204608/export_summary_int8.json).

## Contribución al dataset

De las 11 fuentes del corpus (`source_id`), casi todas provienen de África, India o laboratorios de Estados Unidos; ninguna se capturó en El Salvador.

![Pantalla para contribuir al dataset](/app/contribuir.png)

Con una sesión iniciada y conexión, la app sube las fotos que el usuario aporta, con la etiqueta que él elige, para formar un banco de datos nacional. Esas imágenes son la vía para cerrar la brecha entre las fotos de datasets públicos y las de parcelas locales.
