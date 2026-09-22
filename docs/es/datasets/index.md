# Recopilación de Datasets

En esta sección se documentan los datasets evaluados para el proyecto. Se busca priorizar fuentes con imágenes en entornos de campo real.

> Esta es una bitácora de repositorios evaluados, no el conteo canónico de `source_id`. El `master_manifest.csv` vigente resuelve 11 fuentes; algunas páginas describen repositorios descartados, solapados o incorporados parcialmente.

> **Alcance actualizado (junio 2026):** El proyecto pasó de 4 a **9 clases objetivo** incorporando plagas (gusano cogollero, necrosis letal), enfermedades foliares (roya común, NCLB, GLS) y deficiencias nutricionales (nitrógeno, fósforo, potasio). `aphids_pest` fue descartada por insuficiencia de datos (~77 imágenes, augmentation no viable); en su lugar se incorporó `lethal_necrosis` (~6 415 imágenes de campo real).

> **Ampliación (agosto 2026, posterior a la primera entrega):** Se incorporaron **cuatro datasets Roboflow adicionales** dirigidos exclusivamente a las clases más escasas del corpus - las tres deficiencias nutricionales y GLS. Aportaron **1 815 imágenes netas** de campo real (1 876 integradas menos 61 duplicados), llevando el total de `clean/` de 31 622 a **33 438**. El impacto se concentra donde más falta hacía: potasio pasó de 266 a 621 imágenes (x2.3) y GLS de 1 119 a 1 930 (x1.7), y el desbalance máximo bajó de 32.9x a **14.1x**. La ampliación además resolvió el sesgo de **fuente única** en las deficiencias, que pasaron de 1 a 4 fuentes cada una. Detalle del procesamiento en [Limpieza y ordenado](/es/cleanup-and-ordered/).

## Criterios de Evaluación

Para cada dataset se registra:

- **Licencia y permisos** - si permite uso académico y redistribución
- **Dominio de captura** - laboratorio controlado, campo real, o mixto
- **Clases disponibles** - enfermedades etiquetadas y alineación con las 9 clases objetivo (ver tabla en cleanup-and-ordered)
- **Distribución por clase** - balance y posibles sesgos
- **Calidad y particularidades** - resolución, condiciones de iluminación, fondos, artefactos

## Resumen de Datasets

| Dataset | Dominio | Imágenes (maíz) | Clases objetivo disponibles | Licencia |
|---|---|---|---|---|
| [Maize in Field Dataset](/es/datasets/maize-in-field-dataset) | Campo real  | ~2 223 útiles (2 355 total) | Roya, NCLB, GLS, Sano | CC BY-NC-SA 4.0 |
| [Maize Diseases](/es/datasets/maize-diseases) | Laboratorio + campo  | ~16 162 (v1.0 + v1.1) | Roya, NCLB, GLS, Sano | CC BY-NC-SA 4.0 |
| [Corn Leaf Diseases](/es/datasets/corn-leaf-diseases) | Laboratorio augmentado  | 52 360 (aug x17) | Roya, NCLB, GLS, Sano | MIT |
| [CropDG Unified Multi-Domain](/es/datasets/cropdg-unified-multidomain) | Multi-dominio  | ~13 275 (maíz + tomate) | NCLB, GLS, Sano (sin Roya) | CC BY-NC-SA 4.0 |
| [Maize, Beans & Tomatoes - África](/es/datasets/maize-beans-tomatoes-africa) | Campo real  | 23 286 (12 clases) | Roya (~99), Sano (2 340), Cogollero, Necrosis Letal | Apache 2.0 + CC |
| [Maize Crop Disease (Leaf)](/es/datasets/multicrop-disease-maiz-disease-pests-and-disease) | Mixto  | 30 120 | Roya, NCLB, GLS, Sano, Cogollero, Necrosis Letal | CC BY 4.0 |
| [Maize Nutrient Deficiency](/es/datasets/maize-nutrient-deficiency) | Campo real  | 463 | N, P, K, Mg, Sano | CC BY 4.0 |
| [Corn Leaf - Roboflow](/es/datasets/corn-leaf-roboflow) | Campo real  | 3 943 | Cogollero, NCLB, GLS, Sano, N, P, K, Mg | CC BY 4.0 |
| [Maize 2 - Roboflow](/es/datasets/maize-2-roboflow) | Campo real  | 1 319 | N, P, K (+ suficiencia nutricional) | CC BY 4.0 |
| [Maize Leaf - Roboflow](/es/datasets/maize-leaf-roboflow) | Campo real  | 1 087 | GLS (+ NLB, NLS) | CC BY 4.0 |
| [Maize Deficiency Scanner - Roboflow](/es/datasets/maize-deficiency-scanner-roboflow) | Campo real  | 276 | N, P, K (+ Sano) | CC BY 4.0 |
| [Corn Leaf Diseases Classification - Roboflow](/es/datasets/corn-leaf-diseases-classification-roboflow) | Campo real  | 1 003 | GLS (+ NCLB) | CC BY 4.0 |

> Las últimas cuatro filas corresponden a la ampliación de agosto 2026. De cada una se tomó únicamente el subconjunto de clases objetivo indicado en negrita en [Limpieza y ordenado](/es/cleanup-and-ordered/); las clases entre paréntesis están presentes en el dataset pero no se integraron.

<!-- 
## Inventario por Clase

Conteo consolidado de imágenes **originales únicas** (sin augmentación sintética), deduplicando fuentes compartidas (PlantVillage aparece en Maize Diseases, CropDG-PV y Corn Leaf Diseases).

### Clases de enfermedad foliar

| Clase | Lab original (únicos) | Campo real | **Total original** |
|---|---|---|---|
| Roya común | ~1 192 (PV) | ~300 (ZA) + ~99 (África) | **~1 591** |
| NCLB | ~985 (PV) + ~192 (PlantDoc) | ~554 (ZA) + ~5 029 (CCMT) + ~292 (RF) | **~7 052** |
| GLS | ~513 (PV) + ~68 (PlantDoc) | ~1 084 (ZA) + ~4 285 (CCMT) + ~3 946 (RF) | **~9 896** |
| Sano | ~1 162 (PV) + ~1 041 (CCMT) | ~285 (ZA) + ~2 340 (África) + ~619 (RF) | **~5 447** |
| Cogollero *(Fall Armyworm)* | - | ~2 448 (África) + ~503 (RF) + ~2 560 (MCD) | **~5 511** |
| Necrosis Letal | - | ~3 980 (África) + ~3 231 (MCD) | **~7 211** |

### Clases de deficiencia nutricional

| Clase | Lab original | Campo real | **Total original** |
|---|---|---|---|
| Nitrógeno | - | ~99 (MND) + ~523 (RF) | **~622** |
| Fósforo | - | ~113 (MND) + ~612 (RF) | **~725** |
| Potasio | - | ~56 (MND) + ~266 (RF) | **~322** |

> ZA = Maize in Field Dataset (Sudáfrica). CCMT = dominio campo de CropDG. PlantDoc = dominio PD de CropDG. RF = Corn Leaf Roboflow. MND = Maize Nutrient Deficiency. África = Maize, Beans & Tomatoes Africa. MCD = Maize Crop Disease (Leaf).

## Material de Campo Real

Las imágenes de campo real son el activo más valioso para la robustez del modelo. Su distribución actual por fuente:

### Enfermedades foliares

| Clase | ZA (Maize in Field) | CCMT (CropDG) | PlantDoc (CropDG) | África | RF (Roboflow) | **Total campo real** |
|---|---|---|---|---|---|---|
| Roya común | 300 | - | - | ~99 | - | **~399**  |
| NCLB | 554 | 5 029 | 192 | - | ~292 | **~6 067** |
| GLS | 1 084 | 4 285 | 68 | - | ~3 946 | **~9 383** |
| Sano | 285 | 1 041 | - | 2 340 | ~619 | **~4 285** |
| Cogollero | - | - | - | ~2 448 | ~503 | **~2 951** |
| Necrosis Letal | - | - | - | ~3 980 | - | **~3 980** |

### Deficiencias nutricionales

| Clase | MND (Nutrient Deficiency) | RF (Roboflow) | **Total campo real** |
|---|---|---|---|
| Nitrógeno | ~99 | ~523 | **~622**  |
| Fósforo | ~113 | ~612 | **~725**  |
| Potasio | ~56 | ~266 | **~322**  |

::: warning Desbalance crítico - Roya común y deficiencias nutricionales
**Roya común** sigue siendo la clase más escasa con solo **~399 imágenes de campo real**. Las clases de deficiencia nutricional también presentan volúmenes bajos (322–725 imágenes). Todas estas clases requieren data augmentation antes de entrenamiento.
:::

## Potencial de Data Augmentation

El dataset **Corn Leaf Diseases** ya aplica 17 técnicas de augmentation sobre los originales de PlantVillage, con un ratio efectivo de x17:

| Clase | Originales PV | Factor | Imágenes aug disponibles |
|---|---|---|---|
| Roya común | 953 | x17 | 16 201 |
| NCLB | 788 | x17 | 13 396 |
| GLS | 410 | x17 | 6 970 |
| Sano | 929 | x17 | 15 793 |
| **Total** | **3 080** | **x17** | **52 360** |

Aplicando el mismo ratio x17 sobre las **~399 imágenes de campo real de Roya común** se obtendrían **~6 783 imágenes adicionales de campo**, equiparando su cobertura con NCLB y GLS. Técnicas adicionales recomendables para imágenes de campo: distorsión elástica, CutMix, perspectiva aleatoria y simulación de variaciones de iluminación natural. -->

<!-- ## Estrategia de Uso

El objetivo de corpus es alcanzar **≥ 2 000 imágenes de campo real por clase** antes de la etapa de adaptación de dominio.

| Clase | Campo real disponible | Objetivo | Estado |
|---|---|---|---|
| Roya común | ~399 | ≥ 2 000 | Requiere augmentation  |
| NCLB | ~6 067 | ≥ 2 000 | Cubierto ✓ |
| GLS | ~9 383 | ≥ 2 000 | Cubierto ✓ |
| Sano | ~4 285 | ≥ 2 000 | Cubierto ✓ |
| Cogollero | ~5 511 | ≥ 2 000 | Cubierto ✓ |
| Necrosis Letal | ~7 211 | ≥ 2 000 | Cubierto ✓ |
| Nitrógeno | ~622 | ≥ 2 000 | Requiere augmentation  |
| Fósforo | ~725 | ≥ 2 000 | Requiere augmentation  |
| Potasio | ~322 | ≥ 2 000 | Requiere augmentation  |

**Restricciones activas:** Roya común (~399), Potasio (~322), Nitrógeno (~622) y Fósforo (~725) no alcanzan el umbral mínimo. Antes de iniciar la adaptación de dominio se deben generar imágenes adicionales mediante augmentation para estas clases.

La estrategia de entrenamiento tiene tres etapas:

1. **Pre-entrenamiento de dominio**: ajuste fino inicial con imágenes de laboratorio (PlantVillage y derivados), que son más abundantes y uniformes.
2. **Adaptación de dominio**: segunda ronda de fine-tuning con imágenes de campo real, una vez que Roya común alcance el umbral de ≥ 2 000 mediante augmentation.
3. **Evaluación final**: conjunto de prueba completamente independiente, compuesto mayoritariamente por imágenes reales de campo.

::: warning Limitación conocida
Los modelos entrenados exclusivamente con imágenes de laboratorio suelen fallar en condiciones de campo (fondos variados, iluminación no controlada, oclusión parcial de la hoja). Por eso el conjunto de prueba prioriza imágenes reales aunque sean menos numerosas.
::: -->
