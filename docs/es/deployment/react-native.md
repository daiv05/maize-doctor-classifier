# Aplicación móvil con React Native y Expo

DoctorMaiz es una aplicación en React Native con Expo que ejecuta el modelo en el teléfono, sin necesidad de conexión. Esta página describe los artefactos que se exportan desde PyTorch y cómo la app los usa.

## Artefactos exportados

El pipeline principal convierte los checkpoints (`best.pth`) a TensorFlow Lite (TFLite) y ONNX. Además del modelo en FP32, genera versiones cuantizadas a enteros de 8 bits (int8), que ocupan cerca de una cuarta parte del espacio del TFLite FP32:

| Arquitectura | Tamaño FP32 | Tamaño int8 | Reducción |
|---|:---:|:---:|:---:|
| `shufflenet_v2_x1_0` | 5.17 MB | 1.49 MB | 71 % |
| `efficientnet_lite0` (oficial) | 13.56 MB | 3.74 MB | 72 % |
| `efficientnet_b0` | 16.24 MB | 4.66 MB | 71 % |

Tamaños de los TFLite FP32 e int8, en MB decimales.

La app despliega `efficientnet_lite0` en int8 (3.74 MB), la corrida oficial `20260921_204608`. Junto al `.tflite` se exportan dos archivos:

1. `labels.json`: el orden de las 9 clases. La app lo lee al iniciar, de modo que el índice de salida siempre corresponde al nombre de la clase.
2. `ood_stats.json`: PCA, centroides de clase, covarianzas, gaussiana de fondo y umbral del detector de imágenes fuera de dominio (OOD).

## Preprocesado

El modelo espera los píxeles preparados igual que en el entrenamiento; si la app los prepara distinto, las predicciones se degradan sin error visible. La secuencia es:

1. Corrección de orientación EXIF, porque los teléfonos guardan la rotación en los metadatos y no en los píxeles.
2. Escalado directo a 224 × 224 con remuestreo bilineal con antialias equivalente al de Pillow, sin conservar la proporción de la imagen (igual que el entrenamiento).
3. Conversión a RGB, escala a $[0, 1]$ y normalización con las medias y desviaciones de ImageNet:
   $$\text{canal} = \frac{(\text{pixel} / 255) - \mu}{\sigma}$$
   con $\mu = [0.485, 0.456, 0.406]$ y $\sigma = [0.229, 0.224, 0.225]$.
4. Reorganización del buffer de canales entrelazados (alto, ancho, canal) a canales separados (canal, alto, ancho), que es el formato NCHW de PyTorch.

## Inferencia local

La app usa `react-native-fast-tflite`, que ejecuta el modelo en C++ sobre el procesador del teléfono. Al requerir código nativo, el desarrollo usa Expo Development Builds (EAS Build o prebuild local) y no Expo Go.

La inferencia está detrás de la interfaz `InferenceEngine` (`src/ml/InferenceEngine.ts`), con una implementación TFLite y una simulada para desarrollo. El modelo devuelve dos salidas:

- Los logits de las 9 clases, que con softmax dan el diagnóstico y su confianza.
- El vector de características de la penúltima capa (1280 dimensiones), que el módulo OOD normaliza, proyecta con PCA y usa para calcular la distancia de Mahalanobis relativa.

Si esa distancia supera el umbral (106.37), si la confianza de la clase más probable es menor que 0.6 o si el margen con la segunda clase es menor que 0.15, la app marca el resultado como "No reconocida" y sugiere repetir la toma con un mejor encuadre. Las mediciones en el teléfono están en [Prototipo en dispositivo](./prototipo.md).
