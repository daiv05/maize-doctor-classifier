# Preprocesado

## Normalizado

Antes de que una imagen llegue al modelo tiene que pasar por un único punto de entrada que la deje en un formato consistente, sin importar si viene de una cámara de laboratorio o de un smartphone en el campo. Esto evita que se cuele ruido en nuestro dataset y que el modelo aprenda artefactos irrelevantes.

Toda imagen se carga en caliente, mediante dos pasos en orden:

1. **Corrección EXIF**: corrige la orientación física de fotos tomadas con smartphones antes de cualquier transformación.
2. **Conversión a RGB estricta**: elimina el canal alfa (RGBA) y expande imágenes monocromáticas, garantizando 3 canales en todos los tensores.

Ya como parte de los pipelines de transformación, cada tensor se normaliza con las medias y desviaciones estándar de ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Se eligieron porque todos los backbones preentrenados esperan esta normalización.

## División

El siguiente paso es decidir qué imágenes ve el modelo en cada etapa (entrenamiento, validación, evaluación) y garantizar que esa división sea reproducible (que es uno de los puntos más importantes de este proyecto académico).

La división separa el dataset en:
- **70% train**
- **15% val**
- **15% test**

Con seed 42 declarada en `config/dataset.yaml`. La reproducibilidad material se fija además con `manifest.lock.json`; la seed por sí sola no basta si cambia el corpus o la configuración.

El protocolo principal vigente estratifica por `label + environment`. `source_id` es metadato y puede aparecer en varios splits. La generalización entre fuentes se evalúa por separado con `seed_42_source_grouped` y LOSO; ver [protocolos experimentales](/es/metodologia/protocolos-experimentales).

La división no crea copias físicas de las imágenes, sino que genera tres CSV (`train.csv`, `val.csv` y `test.csv`) con la ruta relativa de cada imagen y su etiqueta. Esto permite que el dataset sea reproducible y que los CSV puedan ser versionados en Git si se desea.

## Balanceo

El dataset completo que se tiene a la fecha está lejos de ser uniforme, algunas clases tienen miles de imágenes y otras apenas un par de cientos. Sin corrección, un modelo entrenado directamente sobre esa distribución tendería a ignorar las clases minoritarias.

### Diagnóstico del desbalance (para train)

| Clase | N | Ratio vs healthy (sana) |
|---|---|---|
| potassium_deficiency | 186 | 32.9x |
| nitrogen_deficiency | 364 | 16.8x |
| phosphorus_deficiency | 428 | 14.3x |
| gray_leaf_spot | 778 | 7.9x |
| common_rust | 1575 | 3.9x |
| fall_armyworm | 3223 | 1.9x |
| lethal_necrosis | 4491 | 1.4x |
| northern_corn_leaf_blight | 4774 | 1.3x |
| healthy | 6118 | 1.0x |

::: warning Cifras de la primera etapa (31 622 imágenes)
Esta tabla corresponde al split **anterior** a la ampliación de agosto. En la materialización vigente de 33 429 elegibles los ratios aproximados son `potassium_deficiency` **14.1x**, `nitrogen_deficiency` **10.3x**, `phosphorus_deficiency` **9.3x** y `gray_leaf_spot` **4.5x**.

**La conclusión operativa no cambia:** son exactamente las mismas cuatro clases las que cruzan el umbral de 4x, y `common_rust` (3.9x) sigue justo por debajo. Al regenerar los splits con `make splits`, la composición de clases minoritarias es idéntica - solo se atenúa la magnitud de la corrección.
:::

### Técnicas discutidas

Antes de llegar a la estrategia actual se evaluaron otras alternativas más simples, y se descartaron por razones concretas:

- **Undersampling agresivo:** se descartó porque elimina datos reales y escasos. Bajar la enorme variedad de imágenes de clases como `healthy` o `lethal_necrosis` a 500 o 1 000 imágenes por clase sería un desperdicio de información valiosa. Además, el modelo perdería la oportunidad de aprender patrones de fondo y variaciones de iluminación que solo aparecen en clases mayoritarias.
- **Oversampling físico (crear más copias físicas en disco):** se descartó (de momento) porque con técnicas como `WeightedRandomSampler` se logra el mismo efecto en memoria, es reversible y se combina con la augmentation en caliente.
- **Focal Loss:** se contempla como opción para el pipeline principal si los resultados no mejoran en las clases minoritarias.

### Estrategia histórica considerada: dos capas

El pipeline principal vigente mantiene el `WeightedRandomSampler` **desactivado** cuando usa pérdida ponderada, para no sobrecompensar el mismo desbalance dos veces. El sampler sigue integrado y puede usarse en otros perfiles; la sección siguiente describe el diseño histórico.

Para el pipeline principal se plantea una estrategia de balanceo de **dos capas**:

**Capa 1: `WeightedRandomSampler`.** Cada muestra recibe un peso `1 / count_of_its_class`. El sampler repite muestras minoritarias dentro de cada epoch sin inflar su tamaño (`num_samples` = tamaño original). Combinado con augmentation en caliente, cada repetición recibe transformaciones distintas.

**Capa 2: `CrossEntropyLoss` ponderada.** Peso por clase `w_i = total / (num_clases x count_i)`. Reforzaría el gradiente de clases minoritarias incluso cuando aparecen en menor proporción dentro de un batch, complementando al sampler que actúa sobre la frecuencia de aparición.

> El **pipeline de baselines** implementa por ahora solo la Capa 1 (sampler) con una `CrossEntropyLoss` estándar sin ponderar, para mantener las corridas simples y comparables entre arquitecturas.

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

Sobre el dataset completo (train, con `healthy` = 6 118 como techo de referencia), las clases que cruzan el umbral de 4x son cuatro: `potassium_deficiency` (32.9x), `nitrogen_deficiency` (16.8x), `phosphorus_deficiency` (14.3x) y `gray_leaf_spot` (7.9x). `common_rust` (3.9x) queda justo por debajo y recibe el pipeline estándar. A esas cuatro clases se les aplica en caliente el pipeline extendido:

> Ratios de la primera etapa; tras la ampliación de agosto 2026 son 14.1x / 10.3x / 9.3x / 4.5x respectivamente. **Las cuatro clases seleccionadas siguen siendo las mismas**, así que este pipeline no cambia. Conviene vigilar `gray_leaf_spot`: con 4.5x quedó cerca del umbral, y un refuerzo adicional lo sacaría de este grupo.

```
RandomResizedCrop(224x224, scale=(0.7, 1.0))   <--- recortes aleatorios
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(±30°, BILINEAR)                 <--- más agresivo que estándar
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
GaussianBlur(kernel=3, sigma=(0.1, 1.5))       <--- ruido gaussiano suave
ToTensor()
Normalize(ImageNet)
```

Se mantiene `hue=0.05` (mínimo) para no destruir la señal de clorosis en deficiencias nutricionales, a pesar de ser clases minoritarias.

### Augmentación mínima para Val / Test

```
Resize(224x224)
ToTensor()
Normalize(ImageNet)
```

Sin ninguna augmentation aleatoria.
