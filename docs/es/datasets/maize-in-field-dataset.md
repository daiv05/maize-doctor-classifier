# Maize_in_Field_Dataset

## Descripción

Imágenes de hojas de maíz enfermas tomadas en condiciones reales (pero un poco contaminadas): 2355 imágenes que fueron tomadas en diferentes fechas y lugares de **Sudáfrica**.

Las enfermedades etiquetadas son:

- Mancha gris de la hoja (GLS)
- Tizón de la hoja del maíz del norte (NCLB)
- Roya común (CR)
- Roya del sur (SR)
- Mancha de la hoja por Phaeosphaeria (PLS).

Contiene una representación realista de las condiciones de campo. Hay imágenes de hojas dañadas por insectos, deficiencias de proteínas, hojas cubiertas por manos, diferentes condiciones de iluminación, algunas hojas mojadas, fondos muy variados, excrementos de aves, varias enfermedades simultáneas e incluso superpuestas.

## Estructura

Los datos del dataset original se encuentran en leaf_images/\*. Son imágenes originales de alta resolución.

Junto con estos archivos de imagen, se incluye un archivo CSV titulado "Database.csv" que contiene las etiquetas de clase para el conjunto de datos.

El CSV tiene el siguiente formato: imgID_id, filePath, y etiquetas codificadas para cada clase: GLS, NCLB, PLS, CR, SR, NoFoliarSymptoms, Other, UnidentifiedDisease (seguramente creadas mediante one-hot encoding).

Este conjunto de datos incluye las siguientes clases:

1. Mancha gris de la hoja
2. Tizón de la hoja del maíz del norte
3. Mancha de la hoja por Phaeosphaeria
4. Roya común
5. Roya del sur
6. Sano
7. Otro
8. Síntoma de enfermedad no identificado

"Otros" se refiere a cualquier característica notable (para un ser humano) que no figure en las demás categorías. Esto puede incluir el desgarro excesivo de las hojas, la presencia de insectos y los daños causados ​​por ellos. 

"Síntoma de enfermedad no identificada" se refiere al caso en que se cree que una hoja de maíz padece una enfermedad que no se encuentra entre las enumeradas.

Para este proyecto no se tomarán en cuenta las clases "Otros" y "Síntoma de enfermedad no identificado", ya que no aportan información útil para el modelo.

## Aspectos importantes

- La mayoría de las imágenes se tomaron en condiciones de campo, PERO, la mayoría tienen presente una tarjeta verde (que desconocemos su función) que puede significar un sesgo para el modelo. Podría valorarse recortar las imágenes para eliminar esta tarjeta.
- Las imágenes tienen una resolución muy alta (3600 x 2700 px, aprox +1.6 MB por imagen), lo que puede requerir un preprocesamiento para reducir su tamaño antes de ser utilizadas.

::: warning Roya común - clase crítica de campo real
El repositorio tiene **300 imágenes de campo real de Roya común**, de las que el corpus vigente conserva **106**: son las únicas imágenes de roya de campo del corpus, frente a 2 150 de laboratorio. De esta fuente el corpus integra además 606 de `gray_leaf_spot` y 140 de `northern_corn_leaf_blight`.
:::

<ImageCarousel :images="[
	{ src: '/maize-in-field-dataset/117_cimg3853_1-121-2.jpg', alt: 'Mancha gris de la hoja, a luz clara y con dedos presentes en la parte inferior izquierda' }, { src: '/maize-in-field-dataset/128_cimg3864_1-7-2.jpg', alt: 'Mancha gris de la hoja, fondo negro con tarjeta verde y manos presentes' }, { src: '/maize-in-field-dataset/157_cimg3893_1-29-1.jpg', alt: 'Luz y brillo altos, con  tarjeta verde en el fondo' }, { src: '/maize-in-field-dataset/173_cimg3909_1-220-2.jpg', alt: 'Fondo negro con tarjeta verde y manos presentes' }, { src: '/maize-in-field-dataset/175_cimg3911_1-190-1.jpg', alt: 'Fondo negro con tarjeta verde y manos presentes' } ]" />

## Tamaño y distribución

- 2355 imágenes
- 10.79 GB

| GLS | NCLB | PLS |CR |SR |NoFoliarSymptoms |Other |UnidentifiedDisease|
| --- | --- | --- |--- |--- |--- |--- |--- |
| 1084 | 554 | 493 |300 |39 |285 |324 |332|

Presenta un desbalance de clases, con una mayoría de imágenes de "Mancha gris de la hoja" y una cantidad significativamente menor de imágenes de "Roya del sur".

## Formato

Imágenes en formato JPG (en mayúsculas) y un archivo CSV con las etiquetas.

## Origen

1. [Dataset en Kaggle](https://www.kaggle.com/datasets/hamishcrazeai/maize-in-field-dataset)
2. Las imágenes se tomaron utilizando una cámara réflex Nikon D90, así como diversos teléfonos inteligentes.

## Citación

- Hamish Craze, and Dave Berger. (2022). Maize_in_Field_Dataset [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3603983

## Licencia

- CC BY-NC-SA 4.0