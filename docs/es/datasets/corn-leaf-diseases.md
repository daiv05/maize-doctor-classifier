# Corn Leaf Diseases Plant Village Augmented Data

## Descripción

Es una subcolección del conocido PlantVillage dataset pero en su versión aumentada (PlantVillage-AD), y para nuestro caso, limitada solo a las imágenes de hojas de maíz. Contiene imágenes de hojas sanas y con tres enfermedades comunes: mancha gris, tizón común y tizón foliar del norte. Las imágenes fueron aumentadas mediante técnicas como ajuste de brillo, contraste, recorte, volteo, ruido gaussiano, rotación, entre otras.

## Estructura

Posee una sola carpeta Corn dividida en 4 subcarpetas, una por cada clase:

- Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot
- Corn_(maize)__Common_rust
- Corn_(maize)__Northern_Leaf_Blight
- Corn_(maize)__healthy

Cada carpeta posee 17 subcarpetas, cada una con una técnica de data augmentation aplicada a las imágenes originales:
- brightness_adjusted
- contrast_adjusted
- cropped
- flipped_horizontal
- flipped_vertical
- gaussian_noise
- high_pass
- hist_equalized
- jittered
- laplacian
- poisson_noise
- rotated
- salt_pepper_noise
- saturation_adjusted
- sobel
- translated
- unsharp_mask


## Aspectos importantes

- Todas las imágenes provienen del datasets público ya conocido de [PlantVillage dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) y su versión [PlantVillage-AD (Augmented Dataset)](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)

- Las imágenes tienen una resolución bastante correcta para el entrenamiento (256 x 256 px, aprox +20kb por imagen).

<ImageCarousel :images="[
    { src: '/corn-leaf-desease/102_contrast_adjusted.png', alt: 'Imagen con ajuste de contraste (Gray_leaf_spot)' }, { src: '/corn-leaf-desease/102_cropped.png', alt: 'Imagen recortada (Gray_leaf_spot)' }, { src: '/corn-leaf-desease/126_laplacian.png', alt: 'Imagen con filtro de Laplaciano (Gray_leaf_spot)' } ]" />

## Tamaño y distribución

- 52360 imágenes
- 1.51 GB

410 de Cercospora_leaf_spot Gray_leaf_spot

953 de Common_rust

788 de Northern_Leaf_Blight

929 de healthy

De cada clase se dispone de 17 carpetas, en cada carpeta se le ha aplicado una técnica de data augmentation. En "total":

- Cercospora leaf spot / Gray leaf spot: 6970
- Common Rust: 16201
- Northern Leaf Blight: 13396
- Healthy: 15793

Posee un buen balance entre clases, con excepción de la clase "Cercospora leaf spot / Gray leaf spot" que tiene menos de la mitad que las otras clases.

## Formato

Imágenes en formato .jpg

## Origen

1. [Dataset en Kaggle](https://www.kaggle.com/datasets/shuvokumarbasak2030/corn-leaf-diseases-plant-village-augmented-data)

## Citación

- Shuvo Kumar Basak. (2025). Plant Village Augmented Dataset New and Update Tec [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/11077233

Dataset e investigación original de PlantVillage:

- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

## Licencia

- MIT