# Pipeline de datos, identidad e integridad

Especificación de la preparación de datos: identidad de las muestras, manifiestos y protocolos de partición.

## Flujo canónico

```text
dataset clean/
  → descubrimiento y validación PIL
  → sample_id de la ruta lógica
  → SHA-256 de los bytes
  → exclusiones verificadas por ruta + hash
  → deduplicación exacta
  → master_manifest.csv
  → split reproducible
  → CornDataset / DataLoader
  → entrenamiento y selección de best epoch
  → best.pth + summary.json
  → evaluación + predictions.csv
```

La implementación está en `scripts/pipeline/create_splits.py`, `src/data/identity.py`, `src/data/preparation.py`, `src/data/splitter.py` y `src/data/dataset.py`. `clean/` es la fuente de verdad; los CSV bajo `outputs/splits/` son derivados y no se editan a mano.

## Identificadores

| Campo | Define | Cálculo/Origen | Uso |
|---|---|---|---|
| `sample_id` | quién es la muestra lógica | `SHA256(ruta relativa normalizada en UTF-8)` | unión estable de splits, batches y predicciones |
| `sha256` / `file_sha256` | qué bytes contiene el archivo | SHA-256 del archivo; la auditoría llama `file_sha256` al mismo valor | integridad y duplicados byte a byte |
| `pixel_sha256` | qué píxeles decodifica la imagen | SHA-256 de la representación RGB canónica, definida abajo | duplicados exactos tras recodificación |
| `source_id` / `group_id` | con qué dominio o muestras está relacionada | `source_id`: dataset de origen derivado del token de fuente del nombre de archivo (`src/data/provenance.py`); `group_id`: grupo explícito del manifiesto de grupos, si no `source_id`, si no `sample_id` | auditoría o agrupación experimental |

Dos rutas distintas pueden contener los mismos bytes: tendrán `sample_id` distintos y `sha256` igual. Un archivo reemplazado conservando la ruta mantiene su `sample_id`, pero cambia su `sha256`. Varias muestras pueden compartir `source_id` sin ser duplicados.

La columna sha256 del manifiesto y file_sha256 del inventario de auditoría son el mismo digest de los bytes originales. Un file_sha256 diferente no garantiza una imagen distinta: dos codificaciones pueden producir exactamente el mismo raster RGB.

## Representación canónica de píxeles

La auditoría independiente de preparación ejecuta, para cada archivo verificado:

1. Decodifica la imagen con Pillow.
2. Aplica la orientación EXIF con ImageOps.exif_transpose.
3. Convierte a RGB de tres canales. Registra el modo posterior a EXIF como image_mode y el formato del archivo, pero esos metadatos no entran al digest.
4. Conserva el ancho y el alto tras orientar; no redimensiona.
5. Serializa los bytes RGB en orden de filas. Al convertir a RGB se descarta alpha y no se aplica el perfil ICC, igual que en el cargador vigente.
6. Antecede los bytes por el separador de versión doctormaiz-pixel-sha256-v1, el modo canónico RGB y ancho/alto como enteros de 32 bits big endian.
7. Calcula SHA-256 de esa secuencia:

~~~text
SHA256(UTF8("doctormaiz-pixel-sha256-v1\0RGB\0")
       || uint32be(width) || uint32be(height) || RGB_row_major_bytes)
~~~

Así, pixel_sha256 puede detectar archivos recodificados que mantienen exactamente los mismos píxeles según esta representación. No calcula similitud perceptual. No aplica transforms ni augmentations de entrenamiento.

## Contrato de `sample_id`

La definición exacta es:

```text
sample_id = SHA256(normalize_sample_path(image_path).encode("utf-8"))
```

`normalize_sample_path` exige una ruta lógica relativa, reemplaza `\` por `/`, elimina componentes `.` y rechaza rutas absolutas, unidades de Windows, rutas vacías y cualquier `..`. No abre la imagen ni usa sus bytes. Esto conserva compatibilidad entre Windows/POSIX y permite incorporar `sample_id` a manifests históricos que solo tenían `image_path`.

`CornDataset.__getitem__` devuelve exactamente `(image, label, sample_id)`. El `DataLoader`, incluso con `shuffle`, sampler o varios workers, transporta el identificador dentro del batch. `write_predictions_csv` lo conserva junto con la predicción; por eso la asociación no depende del número de fila ni del orden de inferencia.

## `master_manifest.csv`

Se escribe antes de partir los datos. Sus columnas actuales, en orden contractual, son:

```text
sample_id,image_path,label,environment,sha256,
source_id,explicit_group_id,group_id,effective_group_id,group_origin
```

- `image_path` es relativa a `DATASET_ROOT`.
- `environment` solo admite `lab` o `real` en la preparación actual.
- `explicit_group_id` procede de un manifiesto opcional.
- `group_id` es el grupo explícito si existe, si no `source_id` (solo cuando el reparto es agrupado), si no `sample_id`; `effective_group_id` repite ese valor como texto.
- `group_origin` indica `explicit`, `inferred` o `individual`. En `seed_42` las 33 429 filas son `individual`: cada muestra es su propio grupo y `effective_group_id` es su `sample_id`.

No existe una columna genérica llamada `source`; la columna real es `source_id`.

## Conflictos, exclusiones y artefactos

La preparación lee y hashea cada imagen una sola vez, en paralelo, pero aplica la decisión de deduplicación sobre el escaneo ordenado para que el resultado sea determinista.

- Un `sha256` repetido con la misma etiqueta es un duplicado exacto: se conserva el primer registro ordenado.
- El mismo `sha256` con etiquetas distintas es un conflicto *cross-label*: el pipeline aborta.
- `config/dataset_exclusions.csv` permite resolver explícitamente un conflicto. Cada exclusión fija `image_path`, `sha256` y razón; si el contenido cambia, la regeneración falla.
- Una imagen ilegible se registra en la auditoría y no llega al entrenamiento.

Cada materialización produce:

| Artefacto | Función |
|---|---|
| `master_manifest.csv` | inventario elegible canónico |
| `train.csv`, `val.csv`, `test.csv` | vistas del manifiesto según el protocolo |
| `manifest.lock.json` | SHA-256 de los parámetros efectivos del split (`config_sha256`), del archivo de exclusiones, del manifiesto y de cada CSV; los propios parámetros (`split_parameters`) |
| `preparation_audit.json` | conteos, distribuciones, conflictos, exclusiones y solapamientos |
| `split_audit_report.csv` | desglose por split, clase, ambiente, fuente y grupo |

La materialización vigente descubrió 33 437 archivos válidos, excluyó ocho archivos pertenecientes a cuatro hashes con etiquetas contradictorias y dejó 33 429 muestras elegibles. Las cuatro copias `fall_armyworm` y las cuatro `healthy` de esos hashes están en `clean/`; la materialización de la etapa 2 (33 433 muestras) conserva las cuatro rutas `fall_armyworm` y no contiene las `healthy`. El corpus de agosto de 2026 analizado en el EDA tenía 33 438 imágenes (una `fall_armyworm` más que `clean/` actual) y el de la primera entrega, 31 623. La evidencia resumida y el lock exacto están en [`split_protocols_summary.json`](../reproducibilidad/evidencia/split_protocols_summary.json) y [`seed_42_manifest_lock.json`](../reproducibilidad/evidencia/seed_42_manifest_lock.json).

## Protocolos de split

| Protocolo | Pregunta | Estrategia | Train / Val / Test | Uso |
|---|---|---|---:|---|
| `seed_42` | selección y desarrollo dentro de fuentes conocidas | estratificación `label + environment` | 23 400 / 5 014 / 5 015 | Principal |
| `seed_42_source_grouped` | generalización a fuentes completas no vistas | 11 `source_id` indivisibles | 16 554 / 7 809 / 9 066 | Benchmark cross-source |

En `seed_42`, `source_id` es metadato: puede aparecer en varias particiones. Las nueve clases están presentes en train, validación y test. La auditoría reporta cero solapamientos por `sample_id`, `sha256` y `effective_group_id`. En este protocolo esas tres comprobaciones garantizan que ninguna muestra ni ningún contenido byte a byte idéntico cruza particiones: `effective_group_id` es el `sample_id` y el `sha256` es único en el manifiesto tras la deduplicación exacta, por lo que no miden ninguna relación entre fuentes ni entre imágenes parecidas.

En `seed_42_source_grouped`, las fuentes permanecen completas: 5 fuentes en train, 3 en validación y 3 en test. Sus proporciones reales son 49.52 / 23.36 / 27.12 % porque los 11 grupos indivisibles (de 158 a 12 136 muestras) no permiten aproximar 70 / 15 / 15 sin romperlos. El split se generó con `--allow-incomplete-splits`. Validación no contiene `fall_armyworm` ni `lethal_necrosis`. Esa ausencia es una propiedad del protocolo.

## Duplicados exactos en seed_42

La [auditoría de píxeles](../reproducibilidad/evidencia/pixel_duplicate_audit/pixel_duplicate_audit.md) cubrió la materialización canónica corn-clean:/clean y los manifests de corn-outputs:/splits/seed_42. Verificó 33 437 rutas y SHA-256 de archivo: 33 429 muestras elegibles más ocho exclusiones contractuales, sin faltantes, discrepancias ni errores de decodificación. El master manifest tiene SHA-256 64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163 y el lock 0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8.

Entre las muestras elegibles hubo cero grupos duplicados por file_sha256 y cero por pixel_sha256; tampoco hubo grupos entre train/validation/test, casos con distinto SHA de archivo y mismos píxeles, ni conflictos de etiqueta nuevos. Test contiene 5 015 imágenes: cero afectadas, 0 %. Los únicos cuatro grupos idénticos aparecen al incluir las ocho imágenes con etiquetas contradictorias ya excluidas mediante [dataset_exclusions.csv](../../../config/dataset_exclusions.csv). Su source_id se resolvió como maize-beans-tomatoes-africa mediante una familia de nombre con correspondencia unívoca en el master manifest; la lista de exclusiones no almacena ese campo.

Un análisis anterior reportó unos 15 grupos y 0.68 % del test con gemelos; no se reproduce sobre la materialización vigente y puede corresponder a una versión anterior del corpus.

`seed_42` se conserva sin cambios. La verificación cubre la integridad de la partición, no la generalización entre fuentes ni los casi duplicados (hojas fotografiadas en secuencia, recortes, resize, rotaciones, cambios fotométricos).

El `config_sha256` del lock es el SHA-256 de los parámetros efectivos del split (`split_parameters`: semilla, clases, proporciones, estratificación y deduplicación), no del archivo `config/dataset.yaml`. Con la configuración actual, `create_splits.py` regenera `manifest.lock.json` byte a byte (`0db3ff3e…`), con los mismos `sample_id` en train, validación y prueba. `exclusions_sha256` sí es el SHA-256 de los bytes de `config/dataset_exclusions.csv` (`4ca25fcb…`); con finales de línea CRLF cambia y el lock regenerado deja de coincidir.

## Deduplicación perceptual

La implementación de PHash existe y se conserva. Sin embargo, ambas materializaciones anteriores registran `deduplicate_perceptual=false` y `perceptual_overlap_count=null`. Por tanto:

- Hay cero solapamiento por bytes y por raster RGB canónico idéntico entre splits.
- No se mide la presencia de imágenes reescaladas, recortadas, volteadas, recomprimidas con pérdida de píxeles o casi duplicadas.

`create_splits.py` no ejecutó PHash en esta materialización, por lo que los casi duplicados entre particiones no están medidos. La limpieza previa de `clean/` sí aplicó PHash con umbral 0 (ver [Limpieza y ordenado](../cleanup-and-ordered/index.md)); esa distancia solo agrupa imágenes con el mismo hash perceptual.

## Garantías durante la carga

`CornDataset` conserva `ImageCache`, `max_per_class`, el cálculo de clases minoritarias antes del tope, los transforms normal/minoritario, el `class_to_idx` canónico y la integración con `WeightedRandomSampler`/`DataLoader`. Un fallo de lectura en el índice solicitado levanta un error con `idx`, `sample_id`, ruta y etiqueta. No devuelve `None`, no altera el DataFrame y no sustituye la muestra por `idx+1`.

Los detalles de checkpoints y consumidores están en [Contratos de runs](../pipeline/contratos-runs.md); la semántica experimental está en [Protocolos experimentales](./protocolos-experimentales.md).
