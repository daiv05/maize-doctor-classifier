---
title: Pipeline de Inferencia Edge AI
description: Del entrenamiento en PyTorch a la ejecución nativa en C++ mediante TFLite Int8 y detección OOD en el dispositivo.
---

# Pipeline de Inferencia Edge AI en el Dispositivo

Llevar una arquitectura de aprendizaje profundo moderna a un teléfono móvil que opera sin conexión requiere una cuidadosa ingeniería de optimización, empaquetado y procesamiento de tensores.

---

## De PyTorch a TensorFlow Lite Int8

El modelo exportado a la app es **`efficientnet_lite0`** (run oficial `20260921_204608`), una variante concebida para prescindir de activaciones complejas (como *swish* o *hard-swish*) y maximizar la compatibilidad con unidades de procesamiento de punto fijo en chips móviles.

| Modelo / Formato | Precisión Numérica | Tamaño en Disco | Macro-F1 en test (5 015) |
|---|:---:|:---:|:---:|
| **PyTorch (`best.pth`)** | Float32 | 13.79 MB | 0.9480 |
| **ONNX** | Float32 | 13.67 MB | paridad aprobada (max \|Δprob\| 3e-6) |
| **TFLite** | Float32 | 13.56 MB | paridad aprobada (max \|Δprob\| 4e-7) |
| **TFLite Cuantizado (Int8)** | **Entero 8-bit** | **3.74 MB** | **0.9489** (acuerdo con PyTorch 99.74 %) |

En el Motorola edge 40 neo (Dimensity 7030) el preprocesado toma 98 ms (p95 279 ms) y la inferencia 23 ms (medianas). El preprocesado replica byte a byte el redimensionado bilineal de Pillow del entrenamiento, y el teléfono reproduce las predicciones del int8 del servidor en 143 de 143 imágenes.

::: tip CUANTIZACIÓN POST-ENTRENAMIENTO (PTQ)
La cuantización reduce en un **72 %** el peso del archivo respecto al TFLite Float32 y mantiene el Macro-F1 del test completo (0.9489 frente a 0.9480). La paridad de una muestra de 30 imágenes no se aprobó por una sola imagen límite (max |Δprob| 0.198 frente al umbral 0.15).
:::

---

## Contrato de Preprocesamiento de Píxeles

Un error habitual en aplicaciones de visión artificial móvil es procesar la imagen de manera distinta a como fue entrenada la red neuronal. DoctorMaiz implementa un pipeline estricto de preprocesamiento:

```mermaid
flowchart TD
    A["Foto de Cámara<br/>4000 × 3000"] --> B["Corrección de Rotación<br/>mediante EXIF"]
    B --> C["Recorte al Marco<br/>Guía 1:1"]
    C --> D["Redimensionado 224 × 224<br/>con Antialiasing"]
    D --> E["Conversión a<br/>Rango 0 – 1"]
    E --> F["Normalización ImageNet<br/>Media y Desviación"]
    F --> G["Reorganización Memoria<br/>HWC a NCHW Planar"]
    G --> H["Tensor de Entrada<br/>TFLite Int8"]
```

### 1. Corrección EXIF
Los sensores de cámara de Android no rotan los píxeles físicamente al tomar fotos verticales: almacenan una etiqueta numérica de orientación EXIF (`Orientation: 6` para 90° CW). DoctorMaiz recompone los píxeles antes de cualquier cálculo para evitar que la red analice hojas giradas.

### 2. Recorte Central Asistido (*Crop to Overlay*)
La app toma las dimensiones de la pantalla y recorta automáticamente la región encerrada por el marco guía foliar, descartando tierra, cielo y el cuerpo del productor.

### 3. Normalización ImageNet
Los canales RGB escalados entre $[0, 1]$ se normalizan según la ecuación:
$$\text{Canal}_{\text{norm}} = \frac{(\text{Pixel} / 255.0) - \mu_c}{\sigma_c}$$
con constantes de referencia:
$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 4. Transposición Planar (NCHW)
Los motores gráficos móviles entregan un arreglo entrelazado $[H, W, C]$ (R, G, B, R, G, B...). Para satisfacer la entrada del modelo convolucional, el buffer se reordena en memoria a formato planar $[1, C, H, W]$ en una sola pasada de microsegundos.

---

## Motor C++ con Nitro Modules (`react-native-fast-tflite`)

Para eliminar la penalización de rendimiento del puente JavaScript tradicional de React Native:
- DoctorMaiz utiliza **Nitro Modules**, una tecnología moderna de interoperabilidad directa en C++ (JSI - *JavaScript Interface*).
- La llamada a `predict(tensor)` invoca directamente las bibliotecas nativas de TensorFlow Lite sin serialización JSON intermedia.
- Esto reduce el consumo de memoria y mantiene la inferencia en torno a **23 milisegundos** (mediana).

---

## Salvaguarda Mahalanobis OOD (Out-of-Distribution)

El modelo exporta una salida secundaria con los **embeddings de la capa pre-clasificación** (un vector de 1280 dimensiones). 

El clasificador OOD calcula la distancia mínima de Mahalanobis $D_M$ respecto a las 9 clases conocidas:
$$D_M(x) = \min_{c} \sqrt{(x - \mu_c)^T \Sigma^{-1} (x - \mu_c)}$$

Si $D_M(x)$ es mayor que el umbral de seguridad $\tau = 42.6$, la muestra se cataloga como **"No reconocida"**, protegiendo al productor contra falsos positivos.
