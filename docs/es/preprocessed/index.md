# Preprocesado

## Normalizado

Antes de que una imagen llegue al modelo tiene que pasar por un único punto de entrada que la deje en un formato consistente, sin importar si viene de una cámara de laboratorio o de un smartphone en el campo. Esto evita que se cuele ruido en nuestro dataset y que el modelo aprenda artefactos irrelevantes.

Toda imagen se carga en caliente, mediante dos pasos en orden:

1. **Corrección EXIF**: corrige la orientación física de fotos tomadas con smartphones antes de cualquier transformación.
2. **Conversión a RGB estricta**: elimina el canal alfa (RGBA) y expande imágenes monocromáticas, garantizando 3 canales en todos los tensores.

Ya como parte de los pipelines de transformación, cada tensor se normaliza con las medias y desviaciones estándar de ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Son las estadísticas de ImageNet, el corpus de preentrenamiento de los backbones.

## División

El siguiente paso es decidir qué imágenes ve el modelo en cada etapa (entrenamiento, validación, evaluación) y garantizar que esa división sea reproducible (que es uno de los puntos más importantes de este proyecto académico).

La división separa el dataset en:
- **70% train**
- **15% val**
- **15% test**

Con seed 42 declarada en `config/dataset.yaml`. La reproducibilidad material se fija además con `manifest.lock.json`; la seed por sí sola no basta si cambia el corpus o la configuración.

El protocolo principal vigente estratifica por `label + environment`. `source_id` es metadato y puede aparecer en varios splits. La generalización entre fuentes se evalúa por separado con `seed_42_source_grouped` y LOSO; ver [protocolos experimentales](/es/metodologia/protocolos-experimentales).

La división no crea copias físicas de las imágenes, sino que genera tres CSV (`train.csv`, `val.csv` y `test.csv`) con el `sample_id`, la ruta relativa, la etiqueta y el entorno de cada imagen, más las columnas de fuente y grupo. Esto permite que el dataset sea reproducible y que los CSV puedan ser versionados en Git si se desea.

## Balanceo

El dataset completo que se tiene a la fecha está lejos de ser uniforme: las clases van de 621 a 8 740 imágenes.

### Desbalance del corpus

| Clase | Muestras elegibles | Razón frente a `healthy` |
|---|---:|---:|
| `potassium_deficiency` | 621 | 14.1x |
| `nitrogen_deficiency` | 846 | 10.3x |
| `phosphorus_deficiency` | 938 | 9.3x |
| `gray_leaf_spot` | 1 930 | 4.5x |
| `common_rust` | 2 256 | 3.9x |
| `fall_armyworm` | 4 853 | 1.8x |
| `lethal_necrosis` | 6 415 | 1.4x |
| `northern_corn_leaf_blight` | 6 830 | 1.3x |
| `healthy` | 8 740 | 1.0x |

### Estrategias de balanceo

- **`WeightedRandomSampler`:** cada muestra recibe un peso `1 / count_of_its_class`; repite muestras minoritarias dentro de cada época sin cambiar su tamaño y se combina con la augmentation en caliente. Los baselines lo usan con una `CrossEntropyLoss` sin ponderar.
- **`CrossEntropyLoss` ponderada:** peso por clase `w_i = total / (num_clases x count_i)`, o su variante `sqrt_inverse`. Es la opción del modelo oficial, que mantiene el sampler desactivado para no compensar dos veces el mismo desbalance.

Por defecto el pipeline principal no submuestrea ni duplica archivos en disco; los perfiles baseline sí limitan cada clase a `max_images_per_class` (1 500).

## Data Augmentation

Además de balancear cuántas veces ve el modelo cada clase, hace falta variar cómo las ve para que no memorice detalles irrelevantes del fondo o del encuadre. La augmentation se aplica en caliente, en cada carga de imagen, y con más intensidad en las clases que más lo necesitan.

Esto se basa en el valor de la configuración:
```
augmentation:
  minority_ratio_threshold: 4.0
```

Lo que se traduce en que las clases cuya frecuencia es superada en **más de 4x** por la clase más numerosa del split reciben un pipeline de augmentación más agresivo, mientras que las que quedan por debajo de ese umbral reciben el pipeline estándar. El umbral se evalúa sobre la distribución real del split de train, no sobre el dataset completo.

Esto se aplica únicamente en train, en cambio val y test usan transformaciones deterministas para garantizar una evaluación justa.

### Pipeline estándar de augmentación

```
Resize(224x224)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(±15°, BILINEAR)
ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0)
ToTensor()
Normalize(ImageNet)
```

El ColorJitter es conservador (sin saturación ni hue) porque las deficiencias nutricionales se diagnostican por color. Alteraciones agresivas de tono destruirían esa señal.

### Pipeline extendido de augmentación para clases minoritarias

En el train vigente (`healthy` = 6 118 como techo de referencia), las clases que cruzan el umbral de 4x son cuatro: `potassium_deficiency` (14.1x), `nitrogen_deficiency` (10.3x), `phosphorus_deficiency` (9.3x) y `gray_leaf_spot` (4.5x). `common_rust` (3.9x) queda justo por debajo y recibe el pipeline estándar. A esas cuatro clases se les aplica en caliente el pipeline extendido:

> Antes de la ampliación de agosto 2026 los ratios eran 32.9x, 16.8x, 14.3x y 7.9x; las cuatro clases son las mismas. `gray_leaf_spot` (4.5x) está cerca del umbral.

```
RandomResizedCrop(224x224, scale=(0.7, 1.0))   <--- recortes aleatorios
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(±30°, BILINEAR)                 <--- más agresivo que estándar
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
GaussianBlur(kernel=3, sigma=(0.1, 1.5))       <--- desenfoque gaussiano leve
ToTensor()
Normalize(ImageNet)
```

Se mantiene `hue=0.05` (bajo) para no destruir la señal de clorosis en deficiencias nutricionales, a pesar de ser clases minoritarias.

### Augmentación mínima para Val / Test

```
Resize(224x224)
ToTensor()
Normalize(ImageNet)
```

Sin ninguna augmentation aleatoria.
