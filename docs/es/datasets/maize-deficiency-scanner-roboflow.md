# Maize Deficiency Scanner - Roboflow (maize-deficiency-scanner-z2whz)

## Identificador

`maize_deficiency_scanner_roboflow`

## Descripción

Dataset de segmentación de deficiencias nutricionales en hojas de maíz, publicado en Roboflow Universe por el workspace `david-deras`. Anotado en formato YOLOv8 de **polígono** (segmentación), sobre fotografías de campo real tomadas a corta distancia, con la hoja sostenida a mano en muchos casos.

Es el más pequeño de los cuatro datasets nuevos, pero aporta las tres clases de deficiencia nutricional.

Las clases incluidas son:

| ID YOLO | Clase | Descripción |
|---|---|---|
| 0 | Healthy | Hoja sana sin síntomas de deficiencia |
| 1 | Nitrogen-deficient | Deficiencia de nitrógeno |
| 2 | Phosphorus-deficient | Deficiencia de fósforo |
| 3 | Potassium-deficient | Deficiencia de potasio |

## Estructura

```
maize-deficiency-scanner-roboflow/
├── train/
│   ├── images/   (196 imágenes)
│   └── labels/
├── valid/
│   ├── images/   (41 imágenes)
│   └── labels/
├── test/
│   ├── images/   (39 imágenes)
│   └── labels/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
```

Todas las anotaciones son polígonos de segmentación (312 líneas, 0 bbox), con contornos muy densos - algunas superan los 200 pares de coordenadas por instancia. El `class_id` sigue siendo el primer token.

## Aspectos importantes

- **Cero ambigüedad de clase.** Es el único de los cuatro datasets nuevos en el que ninguna imagen mezcla clases: las 276 imágenes tienen una sola clase. La segmentación es hoja a hoja, no lesión a lesión.
- **Volumen bajo.** 276 imágenes en total, de las cuales solo 158 corresponden a clases de deficiencia. Es un aporte modesto comparado con `maize_2_roboflow`.
- **Nomenclatura de archivo con prefijo de clase.** Los nombres originales codifican la clase (`H-0001`, etc.), pero la fuente de verdad usada para clasificar fue siempre la etiqueta YOLO, no el nombre.
- **Resolución alta y vertical.** Predomina 3024 x 4032 px (194 imágenes), típico de cámara de teléfono en orientación retrato. Sin redimensionado por parte de Roboflow.
- La clase `Healthy` **no se integró** en esta incorporación: `healthy` ya es la clase mayoritaria del corpus (8 740 elegibles en la materialización vigente; 8 744 en la instantánea de agosto) y no necesita refuerzo.
- Sin augmentation aplicada.

## Tamaño y distribución

- **Total de imágenes:** 276
- **Resolución:** variable (5 distintas); predomina 3024 x 4032 px
- **Formato:** JPEG / JPG

| Clase | Instancias | Imágenes |
|---|---|---|
| Healthy | 149 | 118 |
| Nitrogen-deficient | 79 | 76 |
| Potassium-deficient | 47 | 47 |
| Phosphorus-deficient | 37 | 35 |

## Formato

- Anotaciones en **YOLOv8** (segmentación de polígono)
- Imágenes en formato JPEG

## Origen

1. [Dataset en Roboflow Universe](https://universe.roboflow.com/david-deras/maize-deficiency-scanner-z2whz/dataset/1)

## Citación

@misc{ maize-deficiency-scanner-z2whz_dataset,
  title = { Maize Deficiency Scanner Dataset },
  type = { Open Source Dataset },
  author = { David Deras },
  howpublished = { \url{ https://universe.roboflow.com/david-deras/maize-deficiency-scanner-z2whz } },
  url = { https://universe.roboflow.com/david-deras/maize-deficiency-scanner-z2whz },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2026 },
  month = { aug },
  note = { visited on 2026-08-11 },
}

## Licencia

- **CC BY 4.0**
