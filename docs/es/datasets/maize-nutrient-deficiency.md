# Maize Nutrient Deficiency

## Descripción

Dataset de imágenes de hojas de maíz con deficiencias nutricionales. Incluye fotografías de hojas afectadas por carencia de cuatro macronutrientes esenciales y hojas sanas, pensado para clasificación de síntomas visuales de nutrición en campo.

Las clases incluidas son:

- Sano (Healthy)
- Deficiencia de Magnesio (Magnesium)
- Deficiencia de Nitrógeno (Nitrogen)
- Deficiencia de Fósforo (Phosphorous)
- Deficiencia de Potasio (Potassium)

## Estructura

Cinco subcarpetas, una por clase:

```
maize-nutrient-deficiency/
├── Helathy/        # sic - typo en el nombre original
├── Magnessium/     # sic - typo en el nombre original
├── Nitrogen/
├── Phosphorous/
└── Pottasium/      # sic - typo en el nombre original
```

> Los nombres de tres carpetas contienen errores ortográficos del dataset original (`Helathy`, `Magnessium`, `Pottasium`). Se conservan tal cual en `raw/` para respetar la inmutabilidad de la fuente.

## Aspectos importantes

- El dataset cubre deficiencias nutricionales. Sus clases de N, P y K sí se alinean con las 9 clases objetivo del proyecto (common_rust, fall_armyworm, gray_leaf_spot, healthy, lethal_necrosis, nitrogen_deficiency, northern_corn_leaf_blight, phosphorus_deficiency, potassium_deficiency); la clase Magnesio no forma parte del alcance actual. Las imágenes de N, P, K y Sano se integran al corpus como fuente secundaria de campo real.
- La clase Sano (`Helathy`) sí es compatible y podría aprovecharse como datos adicionales de hojas sanas si la calidad lo permite.
- El volumen total es reducido (463 imágenes, 29 MB), con un desbalance notable entre clases: Potasio tiene menos de la mitad de imágenes que Magnesio.
- Las imágenes fueron tomadas en campo por expertos del dominio durante relevamientos en las aldeas de Mangsuli y Bedag (frontera Maharashtra–Karnataka, India), con un smartphone Redmi Note 13s. Resolución: 4080 x 3060 px. Reflejan variabilidad ambiental real de cultivos de maíz.

## Tamaño y distribución

- 463 imágenes
- 29 MB

| Clase | Carpeta | Imágenes |
|---|---|---|
| Sano | `Helathy` | 72 |
| Magnesio | `Magnessium` | 123 |
| Nitrógeno | `Nitrogen` | 99 |
| Fósforo | `Phosphorous` | 113 |
| Potasio | `Pottasium` | 56 |

Presenta desbalance de clases, con Magnesio (123) como la clase más representada y Potasio (56) como la menos representada.

## Formato

- Imágenes en formato JPEG

## Origen

1. [Dataset en Mendeley Data](https://data.mendeley.com/datasets/34gb2gr7p2/1)

## Citación

- Khade, Vishnu (2025). Maize Nutrient Deficiency Dataset [Dataset]. Mendeley Data, V1. https://doi.org/10.17632/34gb2gr7p2.1

## Licencia

- CC BY 4.0
