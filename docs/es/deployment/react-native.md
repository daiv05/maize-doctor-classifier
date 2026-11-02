# Aplicación móvil con React Native y Expo

DoctorMaiz es una aplicación en React Native con Expo que ejecuta el modelo en el teléfono, sin conexión. Esta página describe los artefactos que se exportan desde PyTorch y cómo la app los usa.

## Artefactos exportados

El pipeline principal convierte los checkpoints (`best.pth`) a TensorFlow Lite (TFLite) y ONNX. Además del modelo en FP32, genera versiones cuantizadas a enteros de 8 bits (int8), que ocupan cerca de una cuarta parte del espacio:

| Arquitectura | Tamaño FP32 | Tamaño int8 | Reducción |
|---|:---:|:---:|:---:|
| `shufflenet_v2_x1_0` | 4.93 MB | 1.42 MB | 71 % |
| `efficientnet_lite0` (oficial) | 13.56 MB | 3.74 MB | 72 % |
| `efficientnet_b0` | 15.48 MB | 4.44 MB | 71 % |

La app despliega `efficientnet_lite0` en int8 (3.74 MB), la corrida oficial `20260921_204608`. Junto al `.tflite` se exportan dos archivos:

1. `labels.json`: el orden de las 9 clases. La app lo lee al iniciar, de modo que el índice de salida siempre corresponde al nombre de la clase.
2. `ood_stats.json`: medias, covarianzas y umbral del detector de imágenes fuera de dominio (OOD).

## Preprocesado

El modelo espera los píxeles preparados igual que en el entrenamiento; si la app los prepara distinto, las predicciones se degradan sin error visible. La secuencia es:

1. Corrección de orientación EXIF, porque los teléfonos guardan la rotación en los metadatos y no en los píxeles.
2. Escalado directo a 224 × 224 con antialiasing.
3. Conversión a RGB, escala a $[0, 1]$ y normalización con las medias y desviaciones de ImageNet:
   $$\text{canal} = \frac{(\text{pixel} / 255) - \mu}{\sigma}$$
   con $\mu = [0.485, 0.456, 0.406]$ y $\sigma = [0.229, 0.224, 0.225]$.
4. Reorganización del buffer de canales entrelazados (alto, ancho, canal) a canales separados (canal, alto, ancho), que es el formato NCHW de PyTorch.

## Inferencia local

La app usa `react-native-fast-tflite`, que ejecuta el modelo en C++ sobre el procesador del teléfono. Al requerir código nativo, el desarrollo usa Expo Development Builds (EAS Build o prebuild local) y no Expo Go.

La inferencia está detrás de una interfaz `InferenceBackend`, de modo que la interfaz gráfica y la base local no dependen de si el motor es TFLite u ONNX. El modelo devuelve dos salidas:

- Los logits de las 9 clases, que con softmax dan el diagnóstico y su confianza.
- El vector de características de la penúltima capa, que el módulo OOD usa para calcular la distancia de Mahalanobis.

Si esa distancia supera el umbral, la app marca el resultado como "No reconocida" y sugiere repetir la toma con un mejor encuadre. Las mediciones en el teléfono están en [Prototipo en dispositivo](./prototipo.md).
