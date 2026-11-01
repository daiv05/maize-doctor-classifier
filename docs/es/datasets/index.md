# Recopilación de Datasets

En esta sección se documentan los datasets evaluados para el proyecto. Se busca priorizar fuentes con imágenes en entornos de campo real.

> Esta es una bitácora de repositorios evaluados, no el conteo canónico de `source_id`. El `master_manifest.csv` vigente resuelve 11 fuentes; algunas páginas describen repositorios descartados, solapados o incorporados parcialmente.

> **Alcance actualizado (junio 2026):** El proyecto pasó de 4 a **9 clases objetivo** incorporando plagas (gusano cogollero, necrosis letal), enfermedades foliares (roya común, NCLB, GLS) y deficiencias nutricionales (nitrógeno, fósforo, potasio). `aphids_pest` fue descartada por insuficiencia de datos (~77 imágenes, augmentation no viable); en su lugar se incorporó `lethal_necrosis` (~6 415 imágenes de campo real).

> **Ampliación (agosto 2026, posterior a la primera entrega):** Se incorporaron **cuatro datasets Roboflow adicionales** dirigidos exclusivamente a las clases más escasas del corpus - las tres deficiencias nutricionales y GLS. Aportaron **1 815 imágenes netas** de campo real (1 876 integradas menos 61 duplicados), llevando el total de `clean/` de 31 623 a **33 438**. El impacto se concentra donde más falta hacía: potasio pasó de 266 a 621 imágenes (x2.3) y GLS de 1 119 a 1 930 (x1.7), y el desbalance máximo bajó de 32.9x a **14.1x**. La ampliación además resolvió el sesgo de **fuente única** en las deficiencias, que pasaron de 1 a 4 fuentes cada una. Detalle del procesamiento en [Limpieza y ordenado](/es/cleanup-and-ordered/).

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

