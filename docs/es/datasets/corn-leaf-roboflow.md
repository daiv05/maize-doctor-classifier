# Corn Leaf - Roboflow (corn-leaf-hd9iy)

## Identificador

`corn_leaf_roboflow`

## Descripción

Dataset de detección de enfermedades y deficiencias nutricionales en hojas de maíz, publicado en Roboflow Universe por el workspace `labonis-workspace`. Las imágenes están anotadas en formato YOLO (bounding boxes y segmentación), capturadas en condiciones de campo real.

Las clases incluidas son:

| ID YOLO | Clase | Descripción |
|---|---|---|
| 0 | Fall Armyworm Damage | Daño foliar por gusano cogollero (*Spodoptera frugiperda*) |
| 1 | Healthy Leaf | Hoja sana sin síntomas visibles |
| 2 | Leaf Spot | Mancha foliar (mixta: GLS + otras) |
| 3 | Magnesium Deficiency | Deficiencia de magnesio |
| 4 | Nitrogen Deficiency | Deficiencia de nitrógeno |
| 5 | Northern Leaf Blight | Tizón foliar del norte (NCLB) |
| 6 | Phosphorus Deficiency | Deficiencia de fósforo |
| 7 | Potassium Deficiency | Deficiencia de potasio |

## Estructura

El dataset sigue el formato estándar YOLO con división en splits:

```
corn-leaf-roboflow/
├── train/
│   ├── images/   (2 888 imágenes)
│   └── labels/
├── valid/
│   ├── images/   (702 imágenes)
│   └── labels/
├── test/
│   ├── images/   (353 imágenes)
│   └── labels/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
```

Las anotaciones mezclan formato YOLO bbox (5 campos por línea) y segmentación de polígono (N > 5 campos). Ambos son válidos; el `class_id` es siempre el primer token.

## Aspectos importantes

- La clase dominante en anotaciones es `Leaf Spot` (ID 2), con ~3 946 instancias frente a ~55 de Nitrogen Deficiency y ~46 de Phosphorus Deficiency. El desbalance es significativo.
- Las imágenes fueron preprocesadas: auto-orientación EXIF y redimensionado a **640 x 640 px** (stretch). No se aplicaron técnicas de augmentation.
- Las imágenes son de campo real; no existen contrapartes de laboratorio en este dataset.

::: warning Desbalance severo en deficiencias nutricionales
Los conteos de instancias de esta ficha corresponden a la versión del repositorio revisada al catalogarlo. En el corpus vigente esta fuente aporta 807 imágenes de `fall_armyworm`, 456 de `healthy`, 424 de `nitrogen_deficiency`, 523 de `northern_corn_leaf_blight`, 499 de `phosphorus_deficiency` y 210 de `potassium_deficiency`.
:::

## Tamaño y distribución

- **Total de imágenes:** 3 943
- **Resolución:** 640 x 640 px
- **Formato:** JPEG / JPG

Distribución aproximada de instancias por clase (conteo de etiquetas en todos los splits):

| Clase | Instancias (aprox.) |
|---|---|
| Fall Armyworm Damage | ~503 |
| Healthy Leaf | ~619 |
| Leaf Spot | ~3 946 |
| Magnesium Deficiency | ~779 |
| Nitrogen Deficiency | ~55 |
| Northern Leaf Blight | ~292 |
| Phosphorus Deficiency | ~46 |
| Potassium Deficiency | ~52 |

## Formato

- Anotaciones en **YOLOv8** (bbox + segmentación de polígono)
- Imágenes en formato JPEG

## Origen

1. [Dataset en Roboflow Universe](https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy/dataset/4)

## Citación

@misc{ corn-leaf-hd9iy_dataset, title = { Corn Leaf Dataset }, type = { Open Source Dataset }, author = { Labonis Workspace }, howpublished = { \url{ https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy } }, url = { https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy }, journal = { Roboflow Universe }, publisher = { Roboflow }, year = { 2026 }, month = { may }, note = { visited on 2026-06-11 }, }

## Licencia

- **CC BY 4.0**
