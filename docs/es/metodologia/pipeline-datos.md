# Pipeline de datos, identidad e integridad

**Estado: VIGENTE.** Esta es la especificación documental principal de preparación de datos. Las demás páginas deben enlazarla en lugar de redefinir `sample_id`, los manifiestos o los protocolos de partición.

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

## Tres conceptos que no son intercambiables

| Campo | Define | Cálculo/Origen | Uso |
|---|---|---|---|
| `sample_id` | quién es la muestra lógica | `SHA256(ruta relativa normalizada en UTF-8)` | unión estable de splits, batches y predicciones |
| `sha256` | qué bytes contiene el archivo | SHA-256 del contenido leído durante preparación | integridad y duplicados exactos |
| `source_id` / `group_id` | con qué dominio o muestras está relacionada | procedencia inferida o manifiesto de grupos | auditoría o agrupación experimental |

Dos rutas distintas pueden contener los mismos bytes: tendrán `sample_id` distintos y `sha256` igual. Un archivo reemplazado conservando la ruta mantiene su `sample_id`, pero cambia su `sha256`. Varias muestras pueden compartir `source_id` sin ser duplicados.

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
- `group_id` y `effective_group_id` materializan la relación usada para auditar o agrupar.
- `group_origin` indica `explicit`, `inferred` o `individual`.

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
| `manifest.lock.json` | hashes de configuración, exclusiones, manifiesto y cada CSV; parámetros efectivos |
| `preparation_audit.json` | conteos, distribuciones, conflictos, exclusiones y solapamientos |
| `split_audit_report.csv` | desglose por split, clase, ambiente, fuente y grupo |

La materialización vigente descubrió 33 437 archivos válidos, excluyó ocho archivos pertenecientes a cuatro hashes con etiquetas contradictorias y dejó 33 429 muestras elegibles. La evidencia resumida y el lock exacto están en [`split_protocols_summary.json`](../reproducibilidad/evidencia/split_protocols_summary.json) y [`seed_42_manifest_lock.json`](../reproducibilidad/evidencia/seed_42_manifest_lock.json).

## Protocolos de split

| Protocolo | Pregunta | Estrategia | Train / Val / Test | Estado |
|---|---|---|---:|---|
| `seed_42` | selección y desarrollo dentro de fuentes conocidas | estratificación `label + environment` | 23 400 / 5 014 / 5 015 | **VIGENTE, principal** |
| `seed_42_source_grouped` | generalización a fuentes completas no vistas | 11 `source_id` indivisibles | 16 554 / 7 809 / 9 066 | **VIGENTE como benchmark cross-source** |

En `seed_42`, `source_id` es metadato: puede aparecer en varias particiones. Las nueve clases están presentes en train, validación y test. La auditoría reporta cero solapamientos por `sample_id`, `sha256` y `effective_group_id`.

En `seed_42_source_grouped`, las fuentes permanecen completas. Sus proporciones reales son 49.52 / 23.36 / 27.12 % porque once grupos grandes no permiten aproximar 70 / 15 / 15 sin romperlos. Validación no contiene `fall_armyworm` ni `lethal_necrosis`. Esta incompletitud es una propiedad del protocolo y no autoriza sustituirlo por el split de desarrollo.

## Deduplicación perceptual

La implementación de PHash existe y se conserva. Sin embargo, ambas materializaciones anteriores registran `deduplicate_perceptual=false` y `perceptual_overlap_count=null`. Por tanto:

- sí puede afirmarse cero fuga por bytes idénticos entre splits;
- no puede afirmarse cero fuga por imágenes reescaladas, recomprimidas, volteadas o casi duplicadas.

No se encontró una decisión versionada que justifique científicamente omitir PHash en esta materialización. La omisión se documenta como estado observado y la auditoría perceptual queda pendiente.

## Garantías durante la carga

`CornDataset` conserva `ImageCache`, `max_per_class`, el cálculo de clases minoritarias antes del tope, los transforms normal/minoritario, el `class_to_idx` canónico y la integración con `WeightedRandomSampler`/`DataLoader`. Un fallo de lectura en el índice solicitado levanta un error con `idx`, `sample_id`, ruta y etiqueta. No devuelve `None`, no altera el DataFrame y no sustituye la muestra por `idx+1`.

Los detalles de checkpoints y consumidores están en [Contratos de runs](../pipeline/contratos-runs.md); la semántica experimental está en [Protocolos experimentales](./protocolos-experimentales.md).
