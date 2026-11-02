# CropDG Unified Multi-Domain Dataset

## Identificador

`cropdg`

## Descripción

CropDG es un dataset unificado multi-dominio para enfermedades de cultivos, construido a partir de tres conjuntos publicos: PlantVillage (PV), CCMT y PlantDoc (PD). Integra y armoniza etiquetas para Tomato y Corn (Maize), con el objetivo de habilitar experimentos consistentes de aprendizaje centralizado, federado y de generalizacion de dominio.

El dataset organiza las imagenes por dominio de captura (laboratorio vs campo) y mantiene separacion estricta entre fuentes para evaluar robustez y rendimiento en escenarios con cambio de dominio.

## Estructura

Estructura de carpetas del dataset:

```
CropDG/
	PV/
		Tomato/
			Leaf_spot/
			Blight/
			Viral/
			Healthy/
		Corn/
			Leaf_spot/
			Leaf_blight/
			Healthy/
	CCMT/
		Tomato/
		Corn/
	PD/
		Tomato/
		Corn/
```

Cada dominio (PV, CCMT, PD) se mantiene separado para habilitar entrenamiento federado, adaptacion de dominio y evaluacion en dominios no vistos.

## Aspectos importantes
- **Dominios**:
	- PV: imagenes controladas estilo laboratorio.
	- CCMT: imagenes de campo con fondos mixtos.
	- PD: imagenes de campo reales, usado como dominio de prueba no visto.
- **Espacio de etiquetas unificado**:
	- Tomato (4 clases):
		- Leaf_spot (Bacterial spot + Septoria leaf spot)
		- Blight (Early blight + Late blight)
		- Viral (Yellow curl virus + Mosaic virus)
		- Healthy
	- Corn (3 clases):
		- Leaf_spot
		- Leaf_blight
		- Healthy
	- Nota: PlantDoc no contiene muestras de Corn Healthy; esa clase se incluye solo en dominios de entrenamiento.
- **Motivacion del autor**: reduce inconsistencias entre datasets (condiciones de captura, iluminacion, fondos, nombres de clases y sesgos de dominio).
- **Uso recomendado por el autor**: entrenamiento en PV + CCMT y prueba en PD (dominio no visto).

::: warning Este dataset no aporta ninguna imagen de campo real de Roya común
Ni CCMT ni PlantDoc incluyen imágenes de **Roya común (Common Rust) para maíz**. Toda la cobertura de campo de este dataset se concentra en NCLB (~5 029 CCMT + 192 PD) y GLS (~4 285 CCMT + 68 PD). Además, **PlantDoc no tiene clase Healthy para maíz**, por lo que el dominio de prueba no visto (PD) solo cubre dos de las clases objetivo del proyecto.
:::

## Tamaño y distribución

- Numero total de imagenes: 13275
- Distribucion por clases, solo para el maíz:
    - CCMT:
        - Leaf_spot: 4285
        - Leaf_blight: 5029
        - Healthy: 1041
    - PD:
        - Leaf_spot: 68
        - Leaf_blight: 192
    - PV:
        - Leaf_spot: 513
        - Leaf_blight: 985
        - Healthy: 1162
- En total:
    - Leaf_spot: 4866
    - Leaf_blight: 6206
    - Healthy: 2203

## Formato

Imágenes en formato .JPG, .jpeg, .png

## Origen

1. [Dataset en Kaggle](https://www.kaggle.com/datasets/saniyaverma914/cropdg-unified-multidomain)

## Citación

Dataset e investigación original de PlantVillage:
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

Dataset de Crop Pest and Disease Dataset:
- Saniya Verma. (2024). CropDG Unified Multi-Domain Dataset [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/12009964. https://data.mendeley.com/datasets/bwh3zbpkpv/1

Dataset de PlantDoc:
- Naman Jain, and Pratik Kayal. (2024). PlantDoc Classification dataset [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/9411594

## Licencia

- CC BY-NC-SA 4.0
