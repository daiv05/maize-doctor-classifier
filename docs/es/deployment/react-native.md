# Aplicación Móvil con React Native y Expo

Un modelo de visión por computadora entrenado en PyTorch dentro de un servidor con GPU no sirve de mucho por sí solo si el agricultor tiene que usarlo en una parcela remota sin cobertura telefónica. La verdadera prueba de fuego del proyecto está en lograr que ese modelo corra en el procesador de un teléfono celular común, de forma completamente autónoma y en una fracción de segundo.

Para construir esta experiencia desarrollamos **DoctorMaiz**, una aplicación móvil en React Native con Expo orientada a la facilidad de uso en campo.

---

## De PyTorch al teléfono: los artefactos exportados

El pipeline principal toma los checkpoints entrenados (`best.pth`) y los convierte a formatos optimizados para inferencia en el borde (*Edge AI*), principalmente **TensorFlow Lite (TFLite)** y **ONNX**.

Además del modelo estándar en precisión flotante (FP32), generamos versiones cuantizadas a enteros de 8 bits (**Int8**). La cuantización reduce drásticamente el espacio que ocupa el modelo en el almacenamiento del teléfono sin apenas sacrificar capacidad diagnóstica:

| Arquitectura | Tamaño FP32 | Tamaño cuantizado (Int8) | Reducción de espacio |
|---|:---:|:---:|:---:|
| **`shufflenet_v2_x1_0`** | 4.93 MB | **1.42 MB** | 71 % más ligero |
| **`efficientnet_lite0`** (desplegada) | 12.92 MB | **3.56 MB** | 72 % más ligero |
| **`efficientnet_b0`** | 15.48 MB | **4.44 MB** | 71 % más ligero |

El modelo elegido para la versión final es **`efficientnet_lite0` cuantizado a Int8** (alrededor de 3.5 MB). Su arquitectura prescinde de activaciones complejas para facilitar la aceleración por hardware en CPUs móviles modestas.

Junto al archivo `.tflite`, el pipeline exporta dos acompañantes fundamentales:
1. **`labels.json`**: El mapa oficial con el orden exacto de las 9 clases. La aplicación lo lee dinámicamente al iniciar, asegurando que nunca haya desincronizaciones entre el índice de salida y el nombre de la patología.
2. **`ood_stats.json`**: Las medias, covarianzas y el umbral calibrado del detector de imágenes fuera de dominio (OOD).

---

## El contrato de preprocesamiento

Un error frecuente al llevar modelos a dispositivos móviles es suponer que el archivo del modelo es el clasificador completo. En realidad, el modelo es solo el núcleo matemático. Si la aplicación móvil prepara los píxeles de una manera ligeramente distinta a como los vio el modelo durante el entrenamiento, las predicciones empezarán a fallar de forma silenciosa.

Para garantizar paridad total con el servidor, la app implementa estrictamente esta secuencia:

1. **Corrección de orientación EXIF:** Los teléfonos suelen guardar la rotación de la cámara en los metadatos de la foto en lugar de girar los píxeles reales. Corregir esto primero evita que el modelo reciba hojas volteadas 90 grados.
2. **Escalado a 224 × 224 píxeles:** Redimensionamos directamente la imagen al tamaño cuadrado de entrada mediante un filtrado cuidadoso con antialiasing para evitar artefactos visuales.
3. **Conversión y normalización:** Los píxeles en formato RGB se escalan al rango $[0, 1]$ dividiendo entre 255 y luego se normalizan con los promedios y desviaciones estándar de ImageNet:
   $$\text{canal} = \frac{(\text{pixel} / 255) - \mu}{\sigma}$$
   donde $\mu = [0.485, 0.456, 0.406]$ y $\sigma = [0.229, 0.224, 0.225]$.
4. **Organización del tensor (NCHW):** Los motores gráficos de los teléfonos entregan los píxeles entrelazados por canal (alto, ancho, canales). El modelo de PyTorch espera los canales agrupados por separado (canal, alto, ancho), por lo que reorganizamos el buffer en memoria antes de la inferencia.

---

## Inferencia local y arquitectura desacoplada

La aplicación utiliza un motor nativo de alto rendimiento (`react-native-fast-tflite`) que ejecuta el modelo directamente sobre el procesador del dispositivo móvil mediante C++. Dado que las bibliotecas nativas de inferencia requieren compilar código C++/Java, el desarrollo se realiza mediante *Expo Development Builds* (EAS Build o prebuild local) en lugar del cliente genérico Expo Go.

Para mantener el código limpio y flexible, la lógica de inferencia se aísla detrás de una interfaz genérica (`InferenceBackend`). El resto de la interfaz gráfica y la base de datos local no necesitan saber si el motor que corre por debajo es TFLite u ONNX; solo solicitan una predicción entregando el tensor procesado.

El modelo devuelve dos salidas simultáneas:
- **Logits de clasificación:** 9 valores numéricos que, pasados por softmax, entregan el diagnóstico más probable y su nivel de confianza.
- **Vector de características (embeddings):** Un vector de la penúltima capa que el módulo OOD utiliza para calcular la distancia de Mahalanobis.

Si la distancia de Mahalanobis supera el umbral de seguridad, la app marca el resultado como "No reconocida", protegiendo al agricultor de diagnósticos inventados y sugiriéndole repetir la toma con un mejor encuadre.
