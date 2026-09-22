# Limpieza y ordenado de datasets

Aquí se resume el proceso y hallazgos encontrados al limpiar, clasificar y estandarizar las imágenes antes de incorporarlas al dataset final. La prioridad es preservar la trazabilidad (origen del dataset) y evitar modificar el material crudo (raw).

## Objetivo

- Consolidar imágenes útiles por clase (enfermedad) y contexto (lab/real).
- Eliminar duplicados, outliers y material procesado (recortes, filtros, etc.).
- Estandarizar nombres para facilitar auditorías y entrenamiento.

### Descartado

**`corn-leaf-diseases-plant-village-augmented-data`**

Debido a que contiene imágenes procesadas (recortes, filtros, augmentaciones) y no se dispone del material original, se decidió omitir este dataset para evitar confusiones. Se mantiene documentado en esta etapa de limpieza para referencia futura.

### A procesar

Se exploran los siguientes datasets, y se definen identificadores para cada uno, que se incluirán en los nombres de las imágenes para mantener la trazabilidad:

**DATASET                                         ---> IDENTIFICADOR**

`corn-leaf-roboflow                                ---> corn_leaf_roboflow`

`cropdg-unified-multidomain                        ---> cropdg`

`maize-beans-and-tomatoes-image-dataset-for-africa ---> maize_africa`

`maize-diseases                                    ---> maize_desease`

`maize-in-field-dataset                            ---> maize_field`

`maize-nutrient-deficiency                         ---> maize_nutrient`

`multicrop-disease-maiz-disease-pests-and-disease  ---> multi_desease`

Ampliación de agosto 2026 (cuatro datasets Roboflow adicionales):

`maize-2-roboflow                                  ---> maize_2_roboflow`

`maize-leaf-roboflow                               ---> maize_leaf_roboflow`

`maize-deficiency-scanner-roboflow                 ---> maize_deficiency_scanner_roboflow`

`corn-leaf-diseases-classification-roboflow        ---> corn_leaf_diseases_classification_roboflow`

## Clases consideradas (junio 2026)

### Enfermedades foliares

1. `Roya común` | `Common Rust` | `Puccinia sorghi`

2. `Tizón foliar del norte (NCLB)` | `Northern Corn Leaf Blight` | `Exserohilum turcicum`

3. `Mancha gris de la hoja (GLS)` | `Gray Leaf Spot` | `Cercospora zeae-maydis`

4. `Hoja sana` | `Healthy` | `-`

5. `Gusano cogollero` | `Fall Armyworm` | `Spodoptera frugiperda`

6. `Necrosis letal del maíz` | `Lethal Necrosis` | `MLN (MCMV + SCMV)`

### Deficiencias nutricionales

7. `Deficiencia de nitrógeno` | `Nitrogen Deficiency`

8. `Deficiencia de fósforo` | `Phosphorus Deficiency`

9. `Deficiencia de potasio` | `Potassium Deficiency`

> **Nota:** `aphids_pest` (áfidos del maíz) fue evaluada pero descartada definitivamente. Solo se hallaron ~77 imágenes en los datasets disponibles y no se encontraron fuentes adicionales. En su lugar se incorporó `lethal_necrosis`, que cuenta con ~6 415 imágenes de campo real de dos datasets distintos.

## Flujo de trabajo implementado

1. Clasificar por entorno en `/data/clean`:
    1. `lab`: fotos en entornos controlados (fondo negro/blanco/gris, estudio).
    2. `real`: fotos tomadas en campo.

    - Agregar el identificador del dataset en un paso intermedio para conservar el origen.
    - Excluir imágenes procesadas (recortes, filtros, augmentaciones). Si un dataset tiene este tipo de imágenes, documentarlo y omitir.

2. Eliminar duplicados con `imagededup` y el script disponible en el repositorio. Asegurarse de no afectar las imágenes originales ni de otras carpetas.
3. Estandarizar nombres:
    - Anteponer el nombre de la clase, en inglés:
        - common_rust
        - northern_corn_leaf_blight
        - gray_leaf_spot
        - healthy
    - Agregar el identificador del dataset (ver sección de [Por procesar](./#por-procesar))
    - Incluir el ambiente: `real` o `lab`.
    - Terminar con número random de 8 dígitos.
    - Ejemplo final: `common_rust_maize_africa_lab_1`.

    Este orden garantiza trazabilidad y evita colisiones entre datasets.

4. Revisar y eliminar outliers: imágenes que no correspondan a la enfermedad, que posean carteles o marcas, tomas aéreas o detalles inconsistentes.

## Documentación

- Para cada dataset se registran decisiones y hallazgos (dataset descartado, criterios, problemas de calidad), además de anotar cualquier ajuste al proceso para mantener la trazabilidad del dataset final.

## Resultados obtenidos

> **Instantánea histórica.** Esta sección documenta la ampliación de agosto (33 438). La preparación contractual vigente descubre 33 437 archivos y, tras ocho exclusiones cross-label, utiliza 33 429 muestras. No se reescriben las cifras siguientes porque explican la evolución del corpus.

Tras aplicar las rutinas automatizadas de deduplicación y filtros de exclusión por calidad, el volumen neto de imágenes útiles integradas en `data/clean/` por clase es el siguiente:

| Clase                        | Lab    | Real   | Total  |
|------------------------------|-------:|-------:|-------:|
| `common_rust`                |  2 150 |    106 |  2 256 |
| `fall_armyworm`              |      0 |  4 858 |  4 858 |
| `gray_leaf_spot`             |    513 |  1 417 |  1 930 |
| `healthy`                    |      0 |  8 744 |  8 744 |
| `lethal_necrosis`            |      0 |  6 415 |  6 415 |
| `nitrogen_deficiency`        |      0 |    846 |    846 |
| `northern_corn_leaf_blight`  |    888 |  5 942 |  6 830 |
| `phosphorus_deficiency`      |      0 |    938 |    938 |
| `potassium_deficiency`       |      0 |    621 |    621 |
| **TOTAL**                    |  **3 551** | **29 887** | **33 438** |

> Cifras finales tras la ampliación de agosto 2026, **ya deduplicadas**. El total previo (primera etapa) era de 31 622 imágenes.

### Efecto de la ampliación de agosto 2026

| Clase | Antes | Integradas | Duplicados | Después | Factor |
|---|---:|---:|---:|---:|---:|
| `gray_leaf_spot` | 1 119 | +836 | -25 | 1 930 | x1.72 |
| `nitrogen_deficiency` | 523 | +345 | -22 | 846 | x1.62 |
| `phosphorus_deficiency` | 612 | +327 | -1 | 938 | x1.53 |
| `potassium_deficiency` | 266 | +368 | -13 | 621 | x2.33 |
| **TOTAL** | **31 622** | **+1 876** | **-61** | **33 438** | |

El desbalance máximo frente a `healthy` bajó de **32.9x** (potasio, 266 img) a **14.1x** (potasio, 621 img), sin que ninguna otra clase se moviera.

## Observaciones

> Junio 2026 - Las clases con menor representación son las deficiencias nutricionales (nitrógeno 523, fósforo 612, potasio 266). Se evalúa la posibilidad de incluir datasets adicionales para estas clases. Está pendiente definir un techo de imágenes por clase para evitar sesgos o aplicar técnicas avanzadas para balancear el dataset (oversampling, SMOTE, etc.).

> Agosto 2026 - Se atendió la observación anterior incorporando cuatro datasets Roboflow dirigidos a esas clases. Las deficiencias siguen siendo las clases minoritarias (potasio 621, nitrógeno 846, fósforo 938) pero el desbalance se redujo a menos de la mitad. Las cuatro clases reforzadas **siguen superando** el umbral `max_count/count > 4.0` que activa el pipeline extendido de augmentación y el `WeightedRandomSampler`, así que la composición de clases minoritarias no cambia: potasio 14.08x, nitrógeno 10.34x, fósforo 9.32x y GLS 4.53x. GLS es la que quedó más cerca del límite - si se refuerza un poco más, saldría del grupo minoritario y **cambiaría el comportamiento del entrenamiento**. `common_rust` sigue fuera con 3.88x.

> Agosto 2026 (fuentes) - La ampliación resolvió además el sesgo de **fuente única** que arrastraban las deficiencias: cada una pasó de depender solo de `maize_nutrient` a combinar cuatro fuentes, y `gray_leaf_spot` pasó de una a tres en su porción de campo. Ver el detalle en [EDA - distribución por fuente](/es/exploratory-data-analysis/#distribucion-por-fuente-de-origen).

## corn-leaf-diseases

Se descartó su uso, debido a que el dataset incluye imágenes ya augmentadas, no incluye las imágenes originales (además de que la mayoría de ellas están presentes en otros datasets).

## corn-leaf-roboflow

### Identificador

`corn_leaf_roboflow`

### Metodología

El dataset original usa formato YOLO (bounding boxes + polígonos de segmentación) con splits `train/valid/test`. Se creó un script para extraer y reordenar las imágenes según las clases objetivo del proyecto, integrándolas en la estructura de `data/clean/` para su uso en el pipeline de entrenamiento.

### Mapeo de clases YOLO -> carpetas clean

| Clase YOLO | Carpeta destino | Clases objetivo |
|---|---|---|
| `fall_armyworm_damage` (0) | `fall_armyworm/real/` | Sí  |

> **Contenido visual:** la clase `fall_armyworm_damage` de Roboflow contiene imágenes de **hoja con daño** (perforaciones, mordeduras características) sin presencia visible del insecto. No hay imágenes del gusano en sí.
| `healthy` (1) | `healthy/real/` | Sí  |
| `leaf_spot` (2) | `gray_leaf_spot/real/` | Parcial  |
| `magnesium_deficiency` (3) | `magnesium_deficiency/real/` | No (referencia) |
| `nitrogen_deficiency` (4) | `nitrogen_deficiency/real/` | Sí  |
| `northern_corn_leaf_blight` (5) | `northern_corn_leaf_blight/real/` | Sí  |
| `phosphorus_deficiency` (6) | `phosphorus_deficiency/real/` | Sí  |
| `potassium_deficiency` (7) | `potassium_deficiency/real/` | Sí  |

### Resultados de extracción

Imágenes integradas:

| Clase | Imágenes copiadas |
|---|---|
| `fall_armyworm` | 503 |
| `healthy` | 619 |
| `gray_leaf_spot` | 3 946 |
| `nitrogen_deficiency` | 55 |
| `northern_corn_leaf_blight` | 292 |
| `phosphorus_deficiency` | 46 |
| `potassium_deficiency` | 52 |

### Decisiones y hallazgos

- Las imágenes ya vienen redimensionadas a **640 x 640 px**, lo que puede introducir distorsión (stretch) respecto a las proporciones originales.
- La clase `nitrogen_deficiency` (~55 instancias en etiquetas) y `phosphorus_deficiency` (~46) tienen muy baja representación.

## cropdg-unified-multidomain

### Criterio de selección específico

El proceso de extracción para este dataset se limitó estrictamente a la carpeta **`PV`** (PlantVillage). La carpeta `CCMT` fue descartada por completo tras identificar que las imágenes en su interior habían sido tratadas y alteradas previamente.

### Common Rust (CR) | Puccinia sorghi

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

Se recopilaron un total de **513 imágenes** en un entorno de laboratorio (`lab`), **no se encontraron imágenes duplicadas** en este lote.

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se recopiló un total de **888 imágenes** en un entorno de laboratorio (`lab`)

### Healthy | Sana

Muestras complementarias evaluadas bajo el mismo estándar de entorno controlado.

## maize-beans-tomatoes-africa

### Identificador

`maize_africa`

### Common Rust (CR) | Puccinia sorghi

Se detectó que la enfermedad presente en este dataset **NO corresponde a la roya común** del maíz (*Puccinia sorghi*), sino que muestra características de la roya del sur (*Puccinia polysora*).

No se incluye en la versión final del dataset, pero se mantiene en esta etapa de limpieza y ordenado para evitar confusiones.

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

No se encontraron imagenes correspondientes a esta enfermedad en este dataset, por lo que no se incluye en la versión final del dataset.

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se reunieron un total de **4,223 imágenes** tomadas en un entorno de campo abierto (`real`)

### Lethal Necrosis (MLN) | Maize Lethal Necrosis

Se recopilaron imágenes de las carpetas `Maize Lethal Necrosis Disease` presentes en las versiones v1 y v2 del dataset. Ambas versiones contienen exactamente las mismas imágenes, por lo que se procesó únicamente la v1. Las imágenes fueron renombradas con el prefijo `lethal_necrosis_maize_africa_real_` e integradas en `clean/lethal_necrosis/real/`. No se reportaron duplicados.

### Fall Armyworm | Spodoptera frugiperda

Las carpetas `Maize Fall Army Worm Pest` y `Maize Fall Army Worm Activity` contienen imágenes de cogollero tomadas en campo. Fueron integradas en `clean/fall_armyworm/real/`.

> **Contenido visual por carpeta:**
> - `Maize Fall Army Worm Activity` - imágenes de **hoja con daño** (sin insecto visible).
> - `Maize Fall Army Worm Pest` - imágenes de **hoja con daño y gusano** visible sobre la planta.

### Healthy | Sana

Muestras adicionales evaluadas según consistencia con el entorno real de campo.

## maize-diseases

### Identificador

`maize_desease`

### Common Rust (CR) | Puccinia sorghi

Todas las imágenes disponibles fueron tomadas en entornos controlados.

Se detectó que las imágenes presentes en la v1 y v1.1 del dataset son exactamente las mismas, por lo que se decidió eliminar la v1.1 del dataset limpio.

Las imágenes se añadieron a /clean/lab y renombraron para identificarlas.

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

No se incluyeron imágenes de esta enfermedad en este bloque debido a que el algoritmo PHash detectó que las 513 imágenes disponibles eran clones exactos de las imágenes ya integradas mediante el dataset `cropdg-unified-multidomain`.

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se reunieron alrededor de 1056 imagenes en un entorno de campo abierto. El lugar donde fueron rescatadas es en /maize-diseases/v1.1.
Adicionalmente, se detectaron imágenes duplicadas con una suma de 4223 imágenes, por lo que se decidió depurar.

### Healthy | Sana

Muestras originales validadas e integradas en el repositorio consolidado. -5,326 imágenes netas post-deduplicación.

## maize-in-field-dataset

### Identificador

`maize_field`

### Reorganización por enfermedad

#### Estado original

El dataset descargado desde Kaggle contenía una carpeta plana `leaf_images/` con **2355 imágenes** sin ninguna subdivisión, acompañada del archivo `Database.csv` con las etiquetas por imagen.

Estructura del CSV:

| Campo | Descripción |
|---|---|
| `imgID_id` | Identificador numérico de la imagen |
| `filePath` | Nombre del archivo |
| `GLS` | Gray Leaf Spot |
| `NCLB` | Northern Corn Leaf Blight |
| `PLS` | Phaeosphaeria Leaf Spot |
| `CR` | Common Rust |
| `SR` | Southern Rust |
| `NoFoliarSymptoms` | Sin síntomas foliares (sana) |
| `Other` | Otra condición |
| `UnidentifiedDisease` | Enfermedad no identificada |

Cada fila puede tener **una o más etiquetas activas** (valor `1`).

#### Criterio de organización

- Imagen con **una sola etiqueta** - carpeta con el nombre de la enfermedad.
- Imagen con **más de una etiqueta** - carpeta `multi_label/`.

#### Resultado

Las imágenes fueron movidas desde `leaf_images/` a subcarpetas dentro de `maize-in-field-dataset/`:

| Carpeta | Imágenes |
|---|---|
| `GLS/` | 630 |
| `multi_label/` | 891 |
| `NoFoliarSymptoms/` | 232 |
| `UnidentifiedDisease/` | 150 |
| `PLS/` | 149 |
| `NCLB/` | 140 |
| `CR/` | 107 |
| `Other/` | 49 |
| `SR/` | 7 |
| **Total** | **2355** |

### Common Rust (CR) | Puccinia sorghi

Todas las imágenes presentes fueron tomadas en entornos reales. Se movieron al dataset limpio.

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

Las imagenes con esta enfermedad fueron tomadas 607 en entornos reales. Se movieron al dataset limpio. De 629 imágenes, se encontraron 22 duplicados que fueron eliminados.

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se recopilaron alrededor de 140 imágenes con esta enfermedad. Todas fueron tomadas en entornos reales y se movieron al dataset limpio. No se encontraron imágenes duplicadas.

### Healthy | Sana

Las 232 imágenes de la carpeta `NoFoliarSymptoms` fueron validadas y enviadas al dataset limpio bajo la clase `healthy`. No se encontraron imágenes duplicadas.

## maize-nutrient-deficiency

### Identificador

`maize_nutrient`

### Hallazgos

No se presentan duplicados ni necesidad de limpieza específica para este dataset.

Fueron copiadas en su totalidad todas las imágenes referentes a:

| Clase | Carpeta | Imágenes |
|---|---|---|
| Sano | `Helathy` | 72 |
| Nitrógeno | `Nitrogen` | 99 |
| Fósforo | `Phosphorous` | 113 |
| Potasio | `Pottasium` | 56 |

Solamente `MagnesiumDeficiency` no se incluyó en el dataset limpio, ya que no es una clase objetivo para este proyecto, debido a los pocos ejemplos disponibles y su relevancia menor en comparación con las otras deficiencias.

## multicrop-disease-maiz-disease-pests-and-disease

### Identificador

`multi_desease`

### Common Rust (CR) | Puccinia sorghi

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

1. **Entropía y Desorden de Datos:** Las imágenes dentro de este repositorio se encuentran completamente desorganizadas y sin una estructura jerárquica clara, lo que impide una segmentación programática eficiente y automatizada para el entrenamiento de los modelos de Machine Learning.
2. **Mezcla Destructiva de Entornos:** Coexisten de manera aleatoria imágenes preprocesadas artificialmente (con filtros aplicados y recortes forzados), imágenes tomadas en campo abierto (`real`) y capturas controladas de laboratorio (`lab`) sin un orden específico ni metadatos de etiquetado que permitan su separación.
3. **Compromiso de Calidad:** La falta de homogeneidad en el origen e integridad de las imágenes introduce ruido masivo y disminuye significativamente el rendimiento y la capacidad de generalización del modelo de aprendizaje automático.

Por lo tanto, para salvaguardar la robustez del set consolidado, se decidió omitir este dataset del flujo final de limpieza.

### Lethal Necrosis (MLN) | Maize Lethal Necrosis

A diferencia de las otras clases de este dataset, las imágenes de Lethal Necrosis se encuentran correctamente aisladas en su propia carpeta y son identificables sin ambigüedad. Se integraron en su totalidad en `clean/lethal_necrosis/real/`. No se reportaron duplicados. Las imágenes son todas de entorno real.

### Fall Armyworm | Spodoptera frugiperda

> **Contenido visual:** las imágenes de cogollero de este dataset son una mezcla de **hoja con daño** y **hoja con daño + gusano** visible, sin separación entre ambos tipos dentro de la carpeta.

### Healthy | Sana

---

# Ampliación agosto 2026 - datasets Roboflow adicionales

Cuatro datasets nuevos incorporados para reforzar exclusivamente las clases más escasas del corpus: las tres deficiencias nutricionales y GLS. Aportaron **1 815 imágenes netas** de campo real (1 876 integradas menos 61 duplicados), llevando el corpus de **31 622** a **33 438**.

> Esta etapa es **posterior a la primera entrega** del proyecto. El material de esa entrega (paper y notebook de EDA) se conserva congelado en `reports/firts-phase/` y **no refleja estas cifras** a propósito.

## Metodología común

Los cuatro datasets vienen en formato **YOLO de detección/segmentación**, no en carpeta-por-clase. Esto obliga a derivar la clase de cada imagen a partir de sus etiquetas, no de su ruta.

### Regla de asignación de clase

Se leyó el `class_id` (primer token de cada línea del `.txt`, válido tanto para bbox de 5 campos como para polígonos de N campos) y se aplicó:

1. Recolectar el conjunto de clases distintas presentes en la imagen.
2. **Descartar la clase contenedora genérica `leaf`** (ver el caso de `corn_leaf_diseases_classification_roboflow` más abajo).
3. Si queda **exactamente una** clase y esa clase es objetivo -> se integra la imagen.
4. Si quedan **cero o más de una** -> se descarta por ambigua.

El criterio es deliberadamente conservador: en detección una imagen puede contener varias enfermedades, y una imagen con GLS + NCLB no es material de entrenamiento válido para un clasificador de etiqueta única. Se prefirió perder 46 imágenes antes que introducir etiquetas incorrectas.

### Entorno: `real`

Se inspeccionaron muestras de cada clase de cada dataset. Las cuatro fuentes son fotografía de campo (suelo, hierba, manos sosteniendo la hoja, fondo vegetal natural), sin ninguna captura de laboratorio. Todo se integró como `real`, consistente con las demás fuentes Roboflow ya presentes en `clean/`.

### Subcarpeta por origen (temporal)

Durante la ingesta cada dataset escribió en **su propia subcarpeta**, para poder auditar, revertir o deduplicar el aporte de una fuente concreta sin tocar el material anterior:

```
clean/<clase>/real/<identificador_dataset>/<archivo>.jpg
```

Una vez verificada la ingesta, esas subcarpetas se aplanaron hacia `clean/<clase>/real/` (ver [Aplanado de subcarpetas](#aplanado-de-subcarpetas)), dejando la estructura homogénea con el resto del corpus.

### Nomenclatura

`<clase>_<identificador_dataset>_real_<hash8>.jpg`

Ejemplo: `potassium_deficiency_maize_2_roboflow_real_00038ceb.jpg`

El sufijo es un **hash de contenido** (primeros 8 caracteres del SHA-256 del archivo) en lugar del número aleatorio usado en incorporaciones anteriores. Es determinista y reproducible entre máquinas - el mismo criterio que ya usa `create_splits.py` para deduplicar -, así que reejecutar la ingesta produce exactamente los mismos nombres. Los 1 876 nombres generados resultaron únicos, sin colisiones.

### Validación de integridad

Cada imagen se verificó con PIL (`verify()` + `convert("RGB")`, el mismo criterio de `create_splits.py`) antes de copiarse. **0 imágenes corruptas** en los cuatro datasets.

### Garantías de no contaminación

- `raw/` quedó **intacto byte a byte** (checksum global antes/después de la ingesta).
- **0** archivos preexistentes de `clean/` eliminados, renombrados o modificados.
- Las 1 876 adiciones están **todas** dentro de las subcarpetas nuevas.

## Resultados de extracción

| Dataset | Clase destino | Imágenes |
|---|---|---:|
| `maize_2_roboflow` | `potassium_deficiency` | 321 |
| `maize_2_roboflow` | `phosphorus_deficiency` | 292 |
| `maize_2_roboflow` | `nitrogen_deficiency` | 269 |
| `maize_leaf_roboflow` | `gray_leaf_spot` | 336 |
| `maize_deficiency_scanner_roboflow` | `nitrogen_deficiency` | 76 |
| `maize_deficiency_scanner_roboflow` | `potassium_deficiency` | 47 |
| `maize_deficiency_scanner_roboflow` | `phosphorus_deficiency` | 35 |
| `corn_leaf_diseases_classification_roboflow` | `gray_leaf_spot` | 500 |
| **TOTAL** | | **1 876** |

Imágenes descartadas por dataset:

| Dataset | Ambiguas | Otras clases (no objetivo) | Corruptas |
|---|---:|---:|---:|
| `maize_2_roboflow` | 4 | 433 | 0 |
| `maize_leaf_roboflow` | 38 | 713 | 0 |
| `maize_deficiency_scanner_roboflow` | 0 | 118 | 0 |
| `corn_leaf_diseases_classification_roboflow` | 4 | 499 | 0 |

## maize-2-roboflow

### Identificador

`maize_2_roboflow`

### Mapeo de clases YOLO -> carpetas clean

| Clase YOLO | Carpeta destino | ¿Clase objetivo? |
|---|---|---|
| `K_Deficiency` (0) | `potassium_deficiency/real/maize_2_roboflow/` | Sí |
| `N_Deficiency` (1) | `nitrogen_deficiency/real/maize_2_roboflow/` | Sí |
| `Nutrient_Sufficiency` (2) | - | No (descartada) |
| `P_Deficiency` (3) | `phosphorus_deficiency/real/maize_2_roboflow/` | Sí |

### Decisiones y hallazgos

- **Es el mayor aporte de la ampliación**: 882 de las 1 876 imágenes (47 %), y la fuente principal para las tres deficiencias.
- **`Nutrient_Sufficiency` se descartó deliberadamente** (433 imágenes). Aunque es tentador mapearla a `healthy`, denota *ausencia de deficiencia nutricional*, no *ausencia de enfermedad*: una hoja nutricionalmente suficiente puede presentar roya o tizón. Integrarla habría contaminado `healthy` con posibles hojas enfermas. Además `healthy` ya es la clase mayoritaria (8 744) y no requiere refuerzo.
- **Solo 4 imágenes ambiguas** de 1 319 (0.3 %): 3 con `Nutrient_Sufficiency` + `P_Deficiency` y 1 con `N_Deficiency` + `P_Deficiency`. Prácticamente es un dataset de clase única.
- **Resolución no uniforme**: contrario a lo observado en `corn-leaf-roboflow` (que venía todo a 640 x 640), aquí conviven 10 resoluciones - 557 imágenes a 640 x 640 y el resto en resolución nativa de cámara (hasta 4000 x 3000). No se redimensionó nada al integrar; eso lo resuelve el pipeline de carga.

## maize-leaf-roboflow

### Identificador

`maize_leaf_roboflow`

### Mapeo de clases YOLO -> carpetas clean

| Clase YOLO | Carpeta destino | ¿Clase objetivo? |
|---|---|---|
| `gls` (0) | `gray_leaf_spot/real/maize_leaf_roboflow/` | Sí |
| `nlb` (1) | - | No tomada en esta incorporación |
| `nls` (2) | - | No (Northern Leaf Spot no es clase objetivo) |

### Decisiones y hallazgos

- **La tasa de ambigüedad más alta de los cuatro**: 38 imágenes descartadas (3.5 %), de las cuales 32 combinan `gls` + `nlb`. Es un hallazgo esperable y clínicamente coherente - GLS, NLS y NCLB son confundibles entre sí y pueden coexistir en la misma planta -, pero confirma que la regla de clase única era necesaria aquí.
- **Ojo con `nls` vs `nlb`.** `nls` es *Northern Leaf Spot* (*Bipolaris zeicola*), que **no** es clase objetivo, y no debe confundirse con `nlb` (*Northern Leaf Blight* / NCLB), que sí lo es. La similitud de los identificadores es una fuente de error real al mapear.
- **`nlb` no se tomó** pese a ser clase objetivo: el alcance definido para esta ampliación era reforzar solo deficiencias y GLS, y `northern_corn_leaf_blight` ya cuenta con 6 830 imágenes. Quedan ~386 imágenes disponibles si en el futuro se decide ampliarla.
- **Anotación lesión a lesión**: 4 446 instancias de `gls` sobre 373 imágenes (~12 por imagen). El conteo de instancias no debe leerse como conteo de imágenes.
- **Solo trae split `train`**, aunque `data.yaml` referencia `valid/` y `test/`. Irrelevante para la ingesta, porque los splits del proyecto se regeneran con `make splits`.
- Resolución muy alta y homogénea: 1 078 de 1 087 imágenes a 3024 x 3024 px.

## maize-deficiency-scanner-roboflow

### Identificador

`maize_deficiency_scanner_roboflow`

### Mapeo de clases YOLO -> carpetas clean

| Clase YOLO | Carpeta destino | ¿Clase objetivo? |
|---|---|---|
| `Healthy` (0) | - | No tomada (clase ya mayoritaria) |
| `Nitrogen-deficient` (1) | `nitrogen_deficiency/real/maize_deficiency_scanner_roboflow/` | Sí |
| `Phosphorus-deficient` (2) | `phosphorus_deficiency/real/maize_deficiency_scanner_roboflow/` | Sí |
| `Potassium-deficient` (3) | `potassium_deficiency/real/maize_deficiency_scanner_roboflow/` | Sí |

### Decisiones y hallazgos

- **Cero imágenes ambiguas**: el único de los cuatro sin ninguna co-ocurrencia de clases. La anotación es hoja a hoja (una hoja segmentada por imagen), no lesión a lesión.
- **Anotación por polígono exclusivamente** (312 líneas, 0 bbox), con contornos muy densos - algunos superan los 200 pares de coordenadas. Irrelevante para clasificación, pero confirma que el parser debe leer el `class_id` como primer token sin asumir 5 campos por línea.
- **Aporte modesto**: 158 imágenes útiles de 276. Es el dataset más pequeño de la ampliación.
- **Los nombres de archivo codifican la clase** (`H-0001`, `N-...`, etc.). Se ignoró esa señal a propósito: la fuente de verdad para clasificar fue siempre la etiqueta YOLO, no el nombre del archivo.
- Predomina 3024 x 4032 px (orientación retrato, cámara de teléfono).

## corn-leaf-diseases-classification-roboflow

### Identificador

`corn_leaf_diseases_classification_roboflow`

> El slug de Roboflow contiene una errata (`classifcation`); se conserva tal cual para poder resolver el dataset. El identificador local sí usa la grafía correcta.

### Mapeo de clases YOLO -> carpetas clean

| Clase YOLO | Carpeta destino | ¿Clase objetivo? |
|---|---|---|
| `gray_leaf_spot` (0) | `gray_leaf_spot/real/corn_leaf_diseases_classification_roboflow/` | Sí |
| `leaf` (1) | - | No - contenedor genérico, se ignora |
| `northern_leaf_blight` (2) | - | No tomada en esta incorporación |

### Decisiones y hallazgos

::: warning Hallazgo principal: la clase `leaf` es un contenedor, no un diagnóstico
`leaf` aparece en **1 001 de las 1 003 imágenes (99.8 %)**: marca el contorno de la hoja, no una condición patológica. Aplicando la regla de clase única sin excepciones, *toda* imagen del dataset tendría 2 clases y sería descartada por ambigua - **el dataset entero se habría perdido**. Por eso la regla incluye el paso explícito de ignorar `leaf` antes de contar clases distintas. Es el hallazgo que más impacto tuvo en el resultado de la ampliación: rescató las 500 imágenes de GLS, el 27 % del total incorporado.
:::

- **Estructura de anotación en dos niveles**: un polígono `leaf` por hoja más N polígonos de lesión. Con 11 302 instancias de `gray_leaf_spot` sobre ~501 imágenes, el promedio ronda las **22 lesiones anotadas por imagen** - la anotación más densa de los cuatro datasets.
- **Separación limpia entre enfermedades** una vez ignorada `leaf`: solo 1 imagen combina `gray_leaf_spot` con `northern_leaf_blight`. El dataset está partido casi por mitades (~501 GLS / ~500 NCLB).
- **`northern_leaf_blight` no se tomó**, por el mismo criterio de alcance aplicado en `maize_leaf_roboflow`. Quedan ~500 imágenes disponibles a futuro.
- **Resolución extremadamente heterogénea**: 36 resoluciones distintas, de 900 x 600 a 4096 x 3072 px. Sugiere que el dataset agrega material de varias cámaras o fuentes; conviene tenerlo presente si aparecen sesgos de subgrupo en el análisis por entorno.

## Aplanado de subcarpetas

Las subcarpetas temporales por dataset cumplieron su función (auditar el aporte de cada fuente de forma aislada) y se **eliminaron tras la verificación**. Las 1 876 imágenes se movieron a un listado único en `clean/<clase>/real/`, dejando la estructura homogénea con el resto del corpus:

```
clean/<clase>/{lab,real}/<archivo>.jpg     # sin subcarpetas intermedias
```

La trazabilidad no se pierde: el identificador del dataset de origen ya va **en el nombre del archivo**, que es de donde lo leen tanto el EDA como el análisis de duplicados. Verificado tras el movimiento: mismo conteo de archivos antes y después, conjunto de nombres idéntico, cero colisiones y cero subcarpetas restantes.

## Deduplicación

Ejecutada con `find_duplicates.py` (PHash, `threshold = 0`) sobre las cuatro clases afectadas. Los CSV quedaron registrados en `src/cleanup/results/` con fecha 2026-08-11.

| Clase | Analizadas | Grupos | Eliminadas |
|---|---:|---:|---:|
| `gray_leaf_spot` | 1 955 | 25 | 25 |
| `nitrogen_deficiency` | 868 | 21 | 22 |
| `phosphorus_deficiency` | 939 | 1 | 1 |
| `potassium_deficiency` | 634 | 13 | 13 |
| **TOTAL** | | **60** | **61** |

**61 imágenes eliminadas de 1 876 incorporadas (3.3 %).**

### Hallazgo: cero contaminación contra el material preexistente

**Todos los grupos de duplicados quedaron contenidos dentro de los datasets nuevos.** No se detectó ni un solo duplicado contra las imágenes que ya estaban en `clean/`, lo que descarta uno de los dos riesgos anticipados: las 606 imágenes de GLS previas no se solapan con las fuentes nuevas.

| Tipo de grupo | Grupos | Detalle |
|---|---:|---|
| Interno de `maize_2_roboflow` | 35 | Duplicados dentro del propio dataset |
| Interno de `corn_leaf_diseases_classification_roboflow` | 20 | Duplicados dentro del propio dataset |
| Cruzado entre las dos fuentes de GLS | 5 | `maize_leaf_roboflow` ↔ `corn_leaf_diseases_classification_roboflow` |

Sobre los otros dos hallazgos:

- **`maize_2_roboflow` y `maize_deficiency_scanner_roboflow` no se solapan entre sí**, pese a cubrir las mismas tres deficiencias. El riesgo anticipado no se materializó.
- **Los 5 grupos cruzados de GLS son byte a byte idénticos** (mismo SHA-256): los dos datasets de Roboflow redistribuyen material común.

::: tip Por qué hizo falta PHash y no bastaba el SHA-256
Solo esos **5** grupos son copias exactas; los **56** restantes son *near-duplicates* - mismo contenido visual pero bytes distintos (reencodings, recortes, recompresión). `create_splits.py` deduplica por SHA-256 al generar los splits, así que habría atrapado los 5 exactos pero **no** los 56 restantes, que habrían provocado *data leakage* entre train y validación.
:::
