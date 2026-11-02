# Corn Leaf Diseases Classification - Roboflow (corn-leaf-diseases-classifcation-nwmtk)

## Identificador

`corn_leaf_diseases_classification_roboflow`

> El slug original contiene una errata (`classifcation`) que se conserva tal cual para poder resolver el dataset en Roboflow.

## Descripción

Dataset de segmentación de enfermedades foliares en maíz, publicado en Roboflow Universe por el workspace `david-deras`. Anotado en formato YOLOv8 de **polígono**, sobre imágenes de campo real.

De este dataset solo interesa la clase `gray_leaf_spot`; es la fuente individual que más aporta a esa clase entre los cuatro datasets nuevos.

Las clases incluidas son:

| ID YOLO | Clase | Descripción |
|---|---|---|
| 0 | gray_leaf_spot | Mancha gris de la hoja (*Cercospora zeae-maydis*) |
| 1 | leaf | **Contenedor genérico** - contorno de la hoja, sin valor diagnóstico |
| 2 | northern_leaf_blight | Tizón foliar del norte (NCLB) |

## Estructura

El export solo trae el split `train`; `data.yaml` referencia `valid/` y `test/` pero esos directorios no existen en el paquete descargado.

```
corn-leaf-diseases-classification-roboflow/
├── train/
│   ├── images/   (1 003 imágenes)
│   └── labels/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
```

Todas las anotaciones son polígonos de segmentación (14 416 líneas, 0 bbox).

## Aspectos importantes

::: warning La clase `leaf` es un contenedor, no un diagnóstico
`leaf` aparece en 1 001 de las 1 003 imágenes (99.8 %): marca el contorno de la hoja, no una condición. Tratada como clase diagnóstica, **toda** imagen del dataset resultaría "multiclase" y por tanto ambigua, descartando el dataset completo. Debe ignorarse explícitamente antes de decidir la clase de la imagen.
:::

- **Estructura de anotación en dos niveles.** Cada imagen lleva un polígono `leaf` (la hoja) más N polígonos de lesión. Con 11 302 instancias de `gray_leaf_spot` sobre ~501 imágenes, el promedio ronda las 22 lesiones anotadas por imagen: es el dataset con anotación más densa de los cuatro.
- **Separación limpia entre las dos enfermedades.** Ignorando `leaf`, solo 1 imagen combina `gray_leaf_spot` con `northern_leaf_blight`. El dataset está partido casi por mitades: ~501 imágenes de GLS y ~500 de NCLB.
- **Resolución muy heterogénea.** 36 resoluciones distintas, desde 900 x 600 hasta 4096 x 3072 px. La más frecuente es 3024 x 3024 px (411 imágenes). Sugiere agregación de material de varias cámaras/fuentes.
- Sin augmentation aplicada.

## Tamaño y distribución

- **Total de imágenes:** 1 003
- **Resolución:** muy variable (36 distintas); predomina 3024 x 3024 px
- **Formato:** JPEG / JPG

| Clase | Instancias | Imágenes |
|---|---|---|
| gray_leaf_spot | 11 302 | 501 |
| northern_leaf_blight | 2 091 | 500 |
| leaf | 1 023 | 1 001 |

## Formato

- Anotaciones en **YOLOv8** (segmentación de polígono)
- Imágenes en formato JPEG

## Origen

1. [Dataset en Roboflow Universe](https://universe.roboflow.com/david-deras/corn-leaf-diseases-classifcation-nwmtk/dataset/1)

## Citación

@misc{ corn-leaf-diseases-classifcation-nwmtk_dataset, title = { corn leaf diseases classifcation Dataset }, type = { Open Source Dataset }, author = { David Deras }, howpublished = { \url{ https://universe.roboflow.com/david-deras/corn-leaf-diseases-classifcation-nwmtk } }, url = { https://universe.roboflow.com/david-deras/corn-leaf-diseases-classifcation-nwmtk }, journal = { Roboflow Universe }, publisher = { Roboflow }, year = { 2026 }, month = { aug }, note = { visited on 2026-08-11 }, }

## Licencia

- **CC BY 4.0**
