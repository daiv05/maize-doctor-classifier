# Maize, Beans and Tomatoes image dataset for Africa

## Descripción

Este dataset comprende imágenes de 10 dataset individuales recopilados en varios países del África subsahariana. Tiene un total de 68.519 imágenes en 32 clases. Comprende varios datasets de frijol, tomate y maíz.

Los dataset de maíz provienen: uno de Burkina Faso [3], Uganda [4] y Namibia [5] y dos de Tanzania [5] y [6]

Esta colección busca resolver el problema de la diversidad geográfica de los conjuntos de datos agrícolas en la región y, por lo tanto, conducir al desarrollo de aplicaciones de visión artificial robustas para la detección y clasificación de enfermedades de los cultivos.

## Estructura

Es un dataset recopilatorio bastante grande, por lo que se acortará a mostrar solo las carpetas relacionadas al maíz, que es el foco de este proyecto.

El dataset se organiza en dos versiones, cada una con una estructura de carpetas similar. Cada versión tiene una carpeta principal que contiene dos subcarpetas: `train_data` y `test_data`. Dentro de cada una de estas subcarpetas, hay carpetas específicas para cada clase de enfermedad o condición del maíz.

```
Super_Image_Dataset_of_Maize_Beans_and_Tomatoes/
├─ Super_Image_Dataset_of_Maize_Beans_and_Tomatoes/
│  ├─ test_data/
│  │  ├─ Maize Abiotic Disease/
│  │  ├─ Maize Aphids Pest/
│  │  ├─ Maize Curvularia Disease/
│  │  ├─ Maize Fall Army Worm Activity/
│  │  ├─ Maize Fall Army Worm Pest/
│  │  ├─ Maize Healthy Crop/
│  │  ├─ Maize Helminthosporiosis Disease/
│  │  ├─ Maize Leaf Blight Disease/
│  │  ├─ Maize Lethal Necrosis Disease/
│  │  ├─ Maize Rust Disease/
│  │  ├─ Maize Streak Virus Disease/
│  │  └─ Maize Stripe Virus Disease/
│  └─ train_data/
│     ├─ Maize Abiotic Disease/
│     ├─ Maize Aphids Pest/
│     ├─ Maize Curvularia Disease/
│     ├─ Maize Fall Army Worm Activity/
│     ├─ Maize Fall Army Worm Pest/
│     ├─ Maize Healthy Crop/
│     ├─ Maize Helminthosporiosis Disease/
│     ├─ Maize Leaf Blight Disease/
│     ├─ Maize Lethal Necrosis Disease/
│     ├─ Maize Rust Disease/
│     ├─ Maize Streak Virus Disease/
│     └─ Maize Stripe Virus Disease/

Super_Image_Dataset_of_Maize_Beans_and_Tomatoes_v2/
├─ Super_Image_Dataset_of_Maize_Beans_and_Tomatoes/
│  ├─ test_data/
│  │  ├─ Maize Abiotic Disease/
│  │  ├─ Maize Aphids Pest/
│  │  ├─ Maize Curvularia Disease/
│  │  ├─ Maize Fall Army Worm Activity/
│  │  ├─ Maize Fall Army Worm Pest/
│  │  ├─ Maize Healthy Crop/
│  │  ├─ Maize Helminthosporiosis Disease/
│  │  ├─ Maize Leaf Blight Disease/
│  │  ├─ Maize Lethal Necrosis Disease/
│  │  ├─ Maize Rust Disease/
│  │  ├─ Maize Streak Virus Disease/
│  │  └─ Maize Stripe Virus Disease/
│  └─ train_data/
│     ├─ Maize Abiotic Disease/
│     ├─ Maize Aphids Pest/
│     ├─ Maize Curvularia Disease/
│     ├─ Maize Fall Army Worm Activity/
│     ├─ Maize Fall Army Worm Pest/
│     ├─ Maize Healthy Crop/
│     ├─ Maize Helminthosporiosis Disease/
│     ├─ Maize Leaf Blight Disease/
│     ├─ Maize Lethal Necrosis Disease/
│     ├─ Maize Rust Disease/
│     ├─ Maize Streak Virus Disease/
│     └─ Maize Stripe Virus Disease/
```

## Aspectos importantes

- Es un dataset recopilatorio con imágenes en entornos reales.
- Existen clases que no están presentes en otros datasets de este proyecto.
- Los tamaños de las imágenes varían, desde 256x256 píxeles hasta 4000x4000 píxeles.

::: warning Maize Rust Disease - aporte marginal a Roya común
**"Maize Rust Disease" tiene solo 99 imágenes**, siendo la clase más pequeña de maíz en este dataset y la única fuente africana de Roya de campo real para el proyecto. Representa aproximadamente el **25% del total de imágenes de campo real de Roya común** (~399 en todo el proyecto). No es suficiente por sí sola para equilibrar esta clase; debe combinarse con data augmentation sobre estas y las ~300 del dataset de Sudáfrica.
:::

## Tamaño y distribución

El dataset tiene un total de 68.519 imágenes. Para el caso del maíz, las clases y su distribución son las siguientes:

| Clase                            | Cantidad Version 1   | Cantidad Version 2   |
| -------------------------------- | -------------------- | -------------------- |
| Maize Abiotic Disease            | 15 + 62 --- 77       | 15 + 62 --- 77       |  
| Maize Aphids Pest                | 15 + 62 --- 77       | 15 + 62 --- 77       |
| Maize Curvularia Disease         | 52 + 207 --- 259     | 52 + 207 --- 259     |
| Maize Fall Army Worm Activity    | 370 + 1478 --- 1848  | 370 + 1478 --- 1848  |
| Maize Fall Army Worm Pest        | 120 + 480 --- 600    | 120 + 480 --- 600    |
| Maize Healthy Crop               | 468 + 1872 --- 2340  | 468 + 1872 --- 2340  |
| Maize Helminthosporiosis Disease | 32 + 127 --- 159     | 32 + 127 --- 159     |
| Maize Leaf Blight Disease        | 1056 + 4223 --- 5279 | 1056 + 4223 --- 5279 |
| Maize Lethal Necrosis Disease    | 796 + 3184 --- 3980  | 796 + 3184 --- 3980  |
| Maize Rust Disease               | 20 + 79 --- 99       | 20 + 79 --- 99       |
| Maize Streak Virus Disease       | 1275 + 5103 --- 6378 | 1275 + 5103 --- 6378 |
| Maize Stripe Virus Disease       | 438 + 1752 --- 2190  | 438 + 1752 --- 2190  |

Basicamente no hay diferencias entre la version 1 y la version 2 del dataset, por lo que se pueden usar indistintamente. En total hay 23,286 imágenes de maíz, distribuidas en 12 clases. La clase con más imágenes es Maize Streak Virus Disease, con 6378 imágenes, y la clase con menos imágenes es Maize Rust Disease, con 99 imágenes.


## Formato

Imágenes en formato .jpg

## Origen

1. [Maize, Beans and Tomatoes image dataset for Africa](https://www.kaggle.com/datasets/osutokaggle/maize-beans-and-tomatoes-image-dataset-for-africa)

## Citación

Citas originales de la publicación del dataset:

[3] O. Appiah, H. O. Kwame , S. Diakalia, A. K. D. Codjia, M. Bêbê, V. Ouedraogo, B. A. A. Diallo, K. Gandji, D. Abdoul-Karim, K. O. Ogunjobi, G. Dabire, C. L. Sanou, D. Anaafo and E. Ramde, "TOM2024," 2024, https://doi.org/10.17632/3d4yg89rtr.1.

[4] C. Babirye, J. N. Nakatumba , G. Namanya, C. Mutebi, M. Ebellu, J. Murungi, S. Tobius, J. Ssemwogerere, A. Nakayima, D. Nabagereka, J. Asasira and R. Kanyesigye, "Makerere University Maize Image Dataset," 2022, https://doi.org/10.7910/DVN/LPGHKK.

[5] Blessing, Sibanda, A. M. Gamundani, G. E. Iyawa, L. S. Matsa, A. K. Koruhama, J. T. Pasipanodya, B. Kasaona, A. Kadhikwa and H. N. Amadhilla, "Namibia University of Science and Technology Maize Dataset," 2022, https://doi.org/10.7910/DVN/6R78HR.

[6] K. Fue, D. Massawe, A. Barakabitze, A. Geofrey, B. Lebalwa, N. Lyimo, F. Mwaipaja, J. Jonathan, S. Mbacho, C. Sanga and G. Rains, "The YEESI Lab Dataset," Zenodo, 2022, https://doi.org/10.5281/zenodo.7729284.

## Licencia

- Apache 2.0

[3] CC BY 4.0

[4] CC0 1.0

[5] CC0 1.0

[6] CC BY 4.0