# Maize Leaf - Roboflow (maize-leaf-7zwoi-he5vd)

## Identificador

`maize_leaf_roboflow`

## Descripción

Dataset de detección de enfermedades foliares en maíz, publicado en Roboflow Universe por el workspace `david-deras`. Anotado en formato YOLOv8 (bounding boxes) sobre imágenes de campo real de muy alta resolución.

De este dataset solo interesa la clase `gls` (mancha gris de la hoja); `nlb` y `nls` quedan fuera del alcance de la incorporación.

Las clases incluidas son:

| ID YOLO | Clase | Descripción |
|---|---|---|
| 0 | gls | Gray Leaf Spot - mancha gris de la hoja (*Cercospora zeae-maydis*) |
| 1 | nlb | Northern Leaf Blight - tizón foliar del norte (NCLB) |
| 2 | nls | Northern Leaf Spot - mancha foliar del norte (*Bipolaris zeicola*) |

## Estructura

El export solo trae el split `train`; `data.yaml` referencia `valid/` y `test/` pero esos directorios no existen en el paquete descargado.

```
maize-leaf-roboflow/
├── train/
│   ├── images/   (1 087 imágenes)
│   └── labels/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
```

Todas las anotaciones son bbox YOLO de 5 campos (9 336 líneas, 0 polígonos).

## Aspectos importantes

- **Anotación densa por lesión.** Con 9 336 instancias sobre 1 087 imágenes, el promedio es de ~8.6 lesiones anotadas por imagen: cada mancha se marca por separado en lugar de encuadrar la hoja completa.
- **Co-ocurrencia de clases relevante.** 38 imágenes (3.5 %) mezclan más de una enfermedad: 32 con `gls` + `nlb`, 4 con `gls` + `nls`, 1 con las tres y 1 con `nlb` + `nls`. Es la tasa de ambigüedad más alta de los cuatro datasets nuevos, y era esperable: GLS y NLS son confundibles entre sí y pueden coexistir en la misma planta.
- **Resolución muy alta y uniforme.** 1 078 de 1 087 imágenes están a 3024 x 3024 px. Sin redimensionado por parte de Roboflow.
- La clase `nls` (Northern Leaf Spot) **no es una clase objetivo** del proyecto y no debe confundirse con `nlb` (NCLB), que sí lo es. En esta incorporación no se tomó ninguna de las dos.
- Sin augmentation aplicada.

## Tamaño y distribución

- **Total de imágenes:** 1 087
- **Resolución:** 3024 x 3024 px (1 078 imágenes); 4 resoluciones distintas en total
- **Formato:** JPEG / JPG

| Clase | Instancias | Imágenes |
|---|---|---|
| gls | 4 446 | 373 |
| nls | 3 577 | 367 |
| nlb | 1 313 | 386 |

> El conteo de instancias no refleja el de imágenes: `gls` tiene más del triple de instancias que `nlb` pero aparece en menos imágenes, por la anotación lesión a lesión.

## Formato

- Anotaciones en **YOLOv8** (bbox)
- Imágenes en formato JPEG

## Origen

1. [Dataset en Roboflow Universe](https://universe.roboflow.com/david-deras/maize-leaf-7zwoi-he5vd/dataset/1)

## Citación

@misc{ maize-leaf-7zwoi-he5vd_dataset, title = { maize-leaf Dataset }, type = { Open Source Dataset }, author = { David Deras }, howpublished = { \url{ https://universe.roboflow.com/david-deras/maize-leaf-7zwoi-he5vd } }, url = { https://universe.roboflow.com/david-deras/maize-leaf-7zwoi-he5vd }, journal = { Roboflow Universe }, publisher = { Roboflow }, year = { 2026 }, month = { aug }, note = { visited on 2026-08-11 }, }

## Licencia

- **CC BY 4.0**
