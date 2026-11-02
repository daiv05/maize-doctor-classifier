# Maize Diseases

## Descripción

Dataset para clasificación de enfermedades en hojas de maíz. Contiene imágenes de alta calidad pensadas para entrenar y evaluar modelos de aprendizaje profundo. Posee dos versiones

Las enfermedades incluidas son:

- Tizón de la hoja del maíz del norte (Northern Leaf Blight)
- Roya común (Common Rust)
- Mancha gris de la hoja (Gray Leaf Spot)
- Sano (Healthy)

## Estructura

Para facilitar la mejora continua del modelo, el dataset se publicó en dos versiones:

- Versión base v1.0
- Versión actualizada v1.1 (única actualización hasta la fecha (mayo 2026))

Cada versión se organiza en carpetas separadas para cada clase, con imágenes en formato .jpg y .JPG. La estructura de carpetas es la siguiente:

### v1.0

- Common Rust (fondo negro): 1192 archivos
- Gray Leaf Spot (fondo gris tipo cemento, con sombras y variaciones de luz): 513 archivos
- Healthy (hoja aislada, ambiente de mañana, poca variación de luz o temperatura, algunas con sombras): 1162 archivos
- Northern Leaf Blight (fondo gris tipo cemento, con sombras y variaciones de luz): 985 archivos

### v1.1

- Common Rust (fondo negro): 1192 archivos
- Gray Leaf Spot (fondo gris tipo cemento, con sombras y variaciones de luz): 513 archivos
- Healthy (ambiente real, luz de mañana, sombras y variaciones de luz): 5326 archivos
- Northern Leaf Blight (ambiente real, luz de mañana, sombras y variaciones de luz): 5279 archivos

## Aspectos importantes

- Tal como se detalló en la sección anterior, existen muchas imágenes con fondo totalmente negro, fondo gris en condiciones controladas, sin fondo y otras en entornos reales de campo.
- El tamaño de cada imagen varía, desde 1024x768 hasta 3016x4032 px (20kb - 200kb - 5mb)

::: warning Common Rust y Gray Leaf Spot no crecieron entre versiones
Las imágenes de **Common Rust (1 192) y Gray Leaf Spot (513) son idénticas en v1.0 y v1.1**; el crecimiento entre versiones afectó únicamente a Healthy y Northern Leaf Blight (que pasaron de ~1 162/985 a ~5 326/5 279). Esto significa que la cobertura de laboratorio de Roya y GLS está **fija en ~1 192 y ~513 originales** y no puede ampliarse con este dataset.
:::

<ImageCarousel :images="[
	{ src: '/maize-diseases/img (10).png', alt: 'Fondo negro' }, { src: '/maize-diseases/img (1008).png', alt: 'Entorno real' }, { src: '/maize-diseases/img (1138).png', alt: 'Hoja sana, sin fondo' } ]" />

## Tamaño y distribución

- 8.72 GB

Para v1.0

| Clase | Número de archivos |
|-------|-------------------|
| Common Rust | 1192 |
| Gray Leaf Spot | 513 |
| Healthy | 1162 |
| Northern Leaf Blight | 985 |

Para v1.1

| Clase | Número de archivos |
|-------|-------------------|
| Common Rust | 1192 |
| Gray Leaf Spot | 513 |
| Healthy | 5326 |
| Northern Leaf Blight | 5279 |

Entre ambas versiones se tiene:

| Clase | Total de archivos |
|-------|-------------------|
| Common Rust | 2384 |
| Gray Leaf Spot | 1026 |
| Healthy | 6488 |
| Northern Leaf Blight | 6264 |

Se tiene una mayor cantidad de hojas sanas y con tizón foliar del norte.

## Formato

- Imágenes en formato .jpg y .JPG

## Origen

1. [Dataset en Kaggle](https://www.kaggle.com/datasets/kaustavbiswal/maize-diseases/data)
2. Algunas de ellas tomadas con teléfonos móviles, otras con cámaras profesionales.

## Citación

No se provee información sobre la citación del dataset. En Kaggle solo mencionan que es una recopilación de imágenes de diferentes fuentes públicas, las que están en condiciones controladas probablemente sean de PlantVillage o PlantDoc.

## Licencia

- CC BY-NC-SA 4.0