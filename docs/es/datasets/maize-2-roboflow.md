# Maize 2 - Roboflow (maize-2-qag1o-76kth)

## Identificador

`maize_2_roboflow`

## Descripción

Dataset de detección de deficiencias nutricionales en hojas de maíz, publicado en Roboflow Universe por el workspace `david-deras`. Las imágenes están anotadas en formato YOLOv8 (bounding boxes) y fueron capturadas en condiciones de campo real.

Es la fuente más productiva de las cuatro incorporadas en agosto 2026 para las tres clases de deficiencia nutricional, que eran las de menor representación del corpus.

Las clases incluidas son:

| ID YOLO | Clase | Descripción |
|---|---|---|
| 0 | K_Deficiency | Deficiencia de potasio |
| 1 | N_Deficiency | Deficiencia de nitrógeno |
| 2 | Nutrient_Sufficiency | Planta sin deficiencia (suficiencia nutricional) |
| 3 | P_Deficiency | Deficiencia de fósforo |

## Estructura

```
maize-2-roboflow/
├── train/
│   ├── images/   (923 imágenes)
│   └── labels/
├── valid/
│   ├── images/   (198 imágenes)
│   └── labels/
├── test/
│   ├── images/   (198 imágenes)
│   └── labels/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
```

Todas las anotaciones son bbox YOLO de 5 campos (1 371 líneas, 0 polígonos). El `class_id` es siempre el primer token.

## Aspectos importantes

- **Casi todas las imágenes son de clase única.** De 1 319 imágenes, solo 4 combinan dos clases distintas (3 con `Nutrient_Sufficiency` + `P_Deficiency`, 1 con `N_Deficiency` + `P_Deficiency`). Esto permite tratar el dataset como clasificación de imagen completa con pérdida mínima.
- **Resolución heterogénea.** A diferencia de `corn-leaf-roboflow`, este export **no** viene redimensionado de forma uniforme: hay 10 resoluciones distintas, con 557 imágenes a 640 x 640 px y el resto en resoluciones nativas de cámara (3000 x 4000, 4000 x 3000, 3000 x 3000, etc.). Se conservan tal cual; el redimensionado a `target_size` ocurre en el pipeline de carga.
- Sin augmentation aplicada por Roboflow.
- `Nutrient_Sufficiency` **no se integró.** Aunque conceptualmente se acerca a `healthy`, denota ausencia de deficiencia nutricional y no ausencia de enfermedad; una hoja puede ser nutricionalmente suficiente y aun así presentar roya o tizón. Mezclarla con `healthy` habría contaminado esa clase.

## Tamaño y distribución

- **Total de imágenes:** 1 319
- **Resolución:** variable (10 distintas; 640 x 640 px en 557 imágenes)
- **Formato:** JPEG / JPG

Distribución por clase (instancias en etiquetas e imágenes que contienen la clase):

| Clase | Instancias | Imágenes |
|---|---|---|
| Nutrient_Sufficiency | 456 | 436 |
| K_Deficiency | 329 | 321 |
| P_Deficiency | 299 | 296 |
| N_Deficiency | 287 | 270 |

## Formato

- Anotaciones en **YOLOv8** (bbox)
- Imágenes en formato JPEG

## Origen

1. [Dataset en Roboflow Universe](https://universe.roboflow.com/david-deras/maize-2-qag1o-76kth/dataset/1)

## Citación

@misc{ maize-2-qag1o-76kth_dataset, title = { Maize 2 Dataset }, type = { Open Source Dataset }, author = { David Deras }, howpublished = { \url{ https://universe.roboflow.com/david-deras/maize-2-qag1o-76kth } }, url = { https://universe.roboflow.com/david-deras/maize-2-qag1o-76kth }, journal = { Roboflow Universe }, publisher = { Roboflow }, year = { 2026 }, month = { aug }, note = { visited on 2026-08-11 }, }

## Licencia

- **CC BY 4.0**
