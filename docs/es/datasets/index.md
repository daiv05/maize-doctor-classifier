# Fuentes de datos

El corpus reúne imágenes de hojas de maíz de repositorios públicos de Kaggle, Mendeley Data, Dataverse, Zenodo y Roboflow Universe. Cada imagen conserva el identificador de su fuente (`source_id`) en el nombre de archivo y en `master_manifest.csv`, que resuelve 11 fuentes y 33 429 muestras elegibles de nueve clases. La integración está descrita en [Limpieza y ordenado](/es/cleanup-and-ordered/) y la identidad de las muestras en [Pipeline de datos](/es/metodologia/pipeline-datos).

## Fuentes integradas

| `source_id` | Repositorio | Licencia | Imágenes en el corpus | Clases en el corpus | Entorno | Identificador |
|---|---|---|---:|---:|---|---|
| [`maize-beans-tomatoes-africa`](/es/datasets/maize-beans-tomatoes-africa) | [Maize, Beans and Tomatoes Image Dataset for Africa](https://www.kaggle.com/datasets/osutokaggle/maize-beans-and-tomatoes-image-dataset-for-africa) (Kaggle) | Apache 2.0 y CC según el subconjunto | 12 136 | 4 | Campo | Compila TOM2024 `10.17632/3d4yg89rtr.1`, Makerere `10.7910/DVN/LPGHKK`, NUST `10.7910/DVN/6R78HR`, YEESI `10.5281/zenodo.7729284` |
| [`maize-diseases`](/es/datasets/maize-diseases) | [Maize Diseases](https://www.kaggle.com/datasets/kaustavbiswal/maize-diseases/data) (Kaggle, v1.0 y v1.1) | CC BY-NC-SA 4.0 | 6 989 | 3 | Laboratorio y campo | Sin DOI propio |
| [`multicrop-disease-maiz`](/es/datasets/multicrop-disease-maiz-disease-pests-and-disease) | [Maize Crop Disease (Leaf)](https://data.mendeley.com/datasets/6w6gsvghfw/1) (Mendeley, Adege 2026) | CC BY 4.0 | 5 816 | 3 | Laboratorio y campo | `10.17632/6w6gsvghfw.1`; la necrosis letal proviene de [Maize Imagery Dataset - Tanzania](https://data.mendeley.com/datasets/fkw49mz3xs/1) |
| [`corn-leaf-roboflow`](/es/datasets/corn-leaf-roboflow) | [Corn Leaf](https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy) (Roboflow, v4) | CC BY 4.0 | 2 919 | 6 | Campo | Solo URL |
| [`cropdg-unified-multidomain`](/es/datasets/cropdg-unified-multidomain) | [CropDG Unified Multi-Domain](https://www.kaggle.com/datasets/saniyaverma914/cropdg-unified-multidomain) (Kaggle) | CC BY-NC-SA 4.0 | 2 562 | 3 | Laboratorio y campo | `10.34740/KAGGLE/DSV/12009964`; reúne PlantVillage `10.3389/fpls.2016.01419` y PlantDoc `10.34740/KAGGLE/DSV/9411594` |
| [`maize-in-field-dataset`](/es/datasets/maize-in-field-dataset) | [Maize in Field Dataset](https://www.kaggle.com/datasets/hamishcrazeai/maize-in-field-dataset) (Kaggle, Sudáfrica) | CC BY-NC-SA 4.0 | 852 | 3 | Campo | `10.34740/KAGGLE/DSV/3603983` |
| [`maize-2-roboflow`](/es/datasets/maize-2-roboflow) | [Maize 2](https://universe.roboflow.com/david-deras/maize-2-qag1o-76kth) (Roboflow, v1) | CC BY 4.0 | 846 | 3 | Campo | Solo URL |
| [`corn-leaf-diseases-classification-roboflow`](/es/datasets/corn-leaf-diseases-classification-roboflow) | [Corn Leaf Diseases Classification](https://universe.roboflow.com/david-deras/corn-leaf-diseases-classifcation-nwmtk) (Roboflow, v1) | CC BY 4.0 | 480 | 1 | Campo | Solo URL |
| [`maize-nutrient-deficiency`](/es/datasets/maize-nutrient-deficiency) | [Maize Nutrient Deficiency](https://data.mendeley.com/datasets/34gb2gr7p2/1) (Mendeley, India) | CC BY 4.0 | 340 | 4 | Campo | `10.17632/34gb2gr7p2.1` |
| [`maize-leaf-roboflow`](/es/datasets/maize-leaf-roboflow) | [Maize Leaf](https://universe.roboflow.com/david-deras/maize-leaf-7zwoi-he5vd) (Roboflow, v1) | CC BY 4.0 | 331 | 1 | Campo | Solo URL |
| [`maize-deficiency-scanner-roboflow`](/es/datasets/maize-deficiency-scanner-roboflow) | [Maize Deficiency Scanner](https://universe.roboflow.com/david-deras/maize-deficiency-scanner-z2whz) (Roboflow, v1) | CC BY 4.0 | 158 | 3 | Campo | Solo URL |

Los conteos son muestras elegibles del `seed_42` (train, validación y prueba). Las imágenes del repositorio original que no corresponden a las nueve clases objetivo, o que se descartaron por duplicado o por calidad, no se cuentan.

## Fuente evaluada y no integrada

| Repositorio | Licencia | Imágenes | Motivo |
|---|---|---:|---|
| [Corn Leaf Diseases (PlantVillage aumentado)](/es/datasets/corn-leaf-diseases) · [Kaggle](https://www.kaggle.com/datasets/shuvokumarbasak2030/corn-leaf-diseases-plant-village-augmented-data) (`10.34740/KAGGLE/DSV/11077233`) | MIT | 52 360 | Contiene solo imágenes ya aumentadas (17 transformaciones) y no incluye los originales; la mayoría están en otras fuentes. |

## Notas por fuente

- **`maize-diseases`**: Common Rust (1 192) y Gray Leaf Spot (513) son idénticas en v1.0 y v1.1; solo crecieron Healthy y Northern Leaf Blight. PHash detectó las 513 imágenes de Gray Leaf Spot como clones de las de `cropdg-unified-multidomain` y no se conservaron, por lo que el corpus no contiene `gray_leaf_spot` de esta fuente. Mezcla fondos negros, grises de laboratorio y escenas de campo. Es 67.8 % `healthy` y aporta el 54.2 % de todas las imágenes sanas.
- **`maize-in-field-dataset`**: 300 imágenes con etiqueta de roya común en el origen, de las cuales 107 tienen solo esa etiqueta; el corpus conserva 106, las únicas imágenes de campo de `common_rust` (frente a 2 150 de laboratorio). Muchas imágenes incluyen una tarjeta verde que puede actuar como sesgo. En el corpus hay 19 resoluciones, de 1024 a 5120 px por lado; la más frecuente es 3648 × 2736 (181 imágenes).
- **`maize-beans-tomatoes-africa`**: "Maize Rust Disease" tiene 99 imágenes que corresponden a roya del sur (*Puccinia polysora*) y no se integraron: el corpus no contiene `common_rust` de esta fuente. En el corpus hay 490 resoluciones, de 24 a 5184 px por lado; las más frecuentes son 640 × 480 (3 733 imágenes) y 1024 × 768 (2 057).
- **`multicrop-disease-maiz`**: el 55.6 % de la fuente es `lethal_necrosis` y aporta el 50.4 % de esa clase en el corpus.
- **`cropdg-unified-multidomain`**: combina dominios PlantVillage (laboratorio), CCMT (campo) y PlantDoc (campo). Su espacio de etiquetas unificado agrupa `Leaf_spot`, `Leaf_blight` y `Healthy` para maíz; PlantDoc no tiene maíz sano.
- **`maize-nutrient-deficiency`**: 463 imágenes originales tomadas en Mangsuli y Bedag (India) con un Redmi Note 13s; el corpus conserva 340, con 334 resoluciones distintas de 179 a 960 px por lado. La clase Magnesio no es una clase objetivo.
- **`corn-leaf-roboflow`**: 3 943 imágenes de 640 × 640 px (estiradas por Roboflow). El corpus conserva 2 919: 807 `fall_armyworm`, 456 `healthy`, 424 `nitrogen_deficiency`, 523 `northern_corn_leaf_blight`, 499 `phosphorus_deficiency` y 210 `potassium_deficiency`; no conserva `gray_leaf_spot` de esta fuente.
- **`maize-2-roboflow`**: 1 319 imágenes, 4 con dos clases. `Nutrient_Sufficiency` no se integró porque no equivale a `healthy`: una hoja suficiente en nutrientes puede tener roya o tizón.
- **`maize-leaf-roboflow`**: 1 087 imágenes de 3024 × 3024 px con anotación por lesión; 38 mezclan más de una enfermedad. `nls` (Northern Leaf Spot) no es una clase objetivo.
- **`maize-deficiency-scanner-roboflow`**: 276 imágenes de una sola clase cada una, de las cuales 158 son deficiencias. No se integró `Healthy`. De las 158 del corpus, 121 miden 3024 × 4032 px, 19 miden 1536 × 2048, 14 miden 4032 × 3024 y 4 tienen otra resolución.
- **`corn-leaf-diseases-classification-roboflow`**: 1 003 imágenes; la clase `leaf` aparece en el 99.8 % y marca el contorno de la hoja, no una condición, por lo que se ignora al asignar la clase. Mitad GLS y mitad NCLB.

## Composición y sesgo de fuente

Dos fuentes aportan una sola clase, seis aportan tres, dos aportan cuatro y una aporta seis, y la contingencia fuente × clase deja 65 de 99 celdas vacías (V de Cramér 0.5030, `chi²` = 67 673.21 sobre n = 33 429). Esto permite que un modelo identifique el repositorio en lugar de la enfermedad; el efecto se evalúa en [Generalización entre fuentes](/es/resultados/generalizacion). Ninguna fuente fue capturada en El Salvador ni en Centroamérica.

Las referencias teóricas están en [Interpretabilidad](/es/deep-learning/interpretability), [Detección OOD](/es/deep-learning/ood-detection) y [Baselines](/es/baselines/); los trabajos previos en [Trabajos previos](/es/previous-work/).
