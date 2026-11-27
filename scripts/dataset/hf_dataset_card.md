---
license: other
license_name: licencias-por-fuente
language:
- es
task_categories:
- image-classification
pretty_name: Hojas de maíz, enfermedades, plagas y deficiencias nutricionales
size_categories:
- 10K<n<100K
tags:
- agriculture
- maize
- plant-disease
---

# Hojas de maíz: enfermedades, plagas y deficiencias nutricionales

Corpus de 33 429 imágenes de nueve condiciones de la hoja de maíz, consolidado por Josias Abner
Rivas Fuentes, David Alejandro Deras Cerros y Elmer Edenilson Rosales Molina en la Escuela de
Ingeniería de Sistemas Informáticos de la Universidad de El Salvador.

| Clase | Imágenes |
|---|---:|
| `healthy` | 8 740 |
| `northern_corn_leaf_blight` | 6 830 |
| `lethal_necrosis` | 6 415 |
| `fall_armyworm` | 4 853 |
| `common_rust` | 2 256 |
| `gray_leaf_spot` | 1 930 |
| `phosphorus_deficiency` | 938 |
| `nitrogen_deficiency` | 846 |
| `potassium_deficiency` | 621 |

## Licencia y procedencia

El corpus no es una obra original: reúne 11 fuentes públicas y **cada imagen conserva la licencia
de su fuente**. No hay una licencia única para el conjunto. Las columnas `source_id` y `license`
de `metadata.csv` indican la que corresponde a cada archivo.

| `source_id` | Fuente | Autor o publicador | Licencia | Imágenes |
|---|---|---|---|---:|
| `maize-beans-tomatoes-africa` | [Maize, Beans and Tomatoes Image Dataset for Africa](https://www.kaggle.com/datasets/osutokaggle/maize-beans-and-tomatoes-image-dataset-for-africa) (compila TOM2024, Makerere, NUST y YEESI Lab) | osutokaggle y los autores de cada subconjunto | Apache 2.0; subconjuntos CC BY 4.0 y CC0 1.0 | 12 136 |
| `maize-diseases` | [Maize Diseases](https://www.kaggle.com/datasets/kaustavbiswal/maize-diseases/data) | Kaustav Biswal | CC BY-NC-SA 4.0 | 6 989 |
| `multicrop-disease-maiz` | [Maize Crop Disease (Leaf)](https://doi.org/10.17632/6w6gsvghfw.1) | Abebe Adege (2026) | CC BY 4.0 | 5 816 |
| `corn-leaf-roboflow` | [Corn Leaf](https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy) | Labonis Workspace | CC BY 4.0 | 2 919 |
| `cropdg-unified-multidomain` | [CropDG Unified Multi-Domain](https://doi.org/10.34740/KAGGLE/DSV/12009964) (PlantVillage, CCMT y PlantDoc) | Saniya Verma (2024) | CC BY-NC-SA 4.0 | 2 562 |
| `maize-in-field-dataset` | [Maize in Field Dataset](https://doi.org/10.34740/KAGGLE/DSV/3603983) | Hamish Craze y Dave Berger (2022) | CC BY-NC-SA 4.0 | 852 |
| `maize-2-roboflow` | [Maize 2](https://universe.roboflow.com/david-deras/maize-2-qag1o-76kth) | Workspace david-deras | CC BY 4.0 | 846 |
| `corn-leaf-diseases-classification-roboflow` | [Corn Leaf Diseases Classification](https://universe.roboflow.com/david-deras/corn-leaf-diseases-classifcation-nwmtk) | Workspace david-deras | CC BY 4.0 | 480 |
| `maize-nutrient-deficiency` | [Maize Nutrient Deficiency](https://doi.org/10.17632/34gb2gr7p2.1) | Vishnu Khade (2025) | CC BY 4.0 | 340 |
| `maize-leaf-roboflow` | [Maize Leaf](https://universe.roboflow.com/david-deras/maize-leaf-7zwoi-he5vd) | Workspace david-deras | CC BY 4.0 | 331 |
| `maize-deficiency-scanner-roboflow` | [Maize Deficiency Scanner](https://universe.roboflow.com/david-deras/maize-deficiency-scanner-z2whz) | Workspace david-deras | CC BY 4.0 | 158 |

Textos: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/),
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### Qué implica

- Las 10 403 imágenes bajo CC BY-NC-SA 4.0 **no pueden usarse con fines comerciales**, y sus
  adaptaciones, si se comparten, deben usar la misma licencia.
- Las imágenes CC BY 4.0, CC0 1.0 y Apache 2.0 mantienen sus propios términos; las anteriores
  restricciones no se les extienden.
- Toda redistribución debe conservar la atribución y la licencia de cada fuente.

Ninguno de los autores originales respalda este trabajo.

## Cambios realizados sobre las fuentes

Selección de nueve clases, armonización de etiquetas, conversión de anotaciones por región en una
etiqueta por imagen, eliminación de duplicados y de imágenes con etiquetas contradictorias, y
renombrado y reorganización en `<clase>/<entorno>/<archivo>`.

## Código

El código que construye y consume el corpus es propiedad de la Universidad de El Salvador, bajo
licencia propietaria. Para adquirirlo se debe presentar una solicitud formal a la Escuela de
Ingeniería de Sistemas Informáticos. Esa licencia no se aplica a las imágenes.
