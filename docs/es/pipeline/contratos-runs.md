# Contratos versionados de runs y artefactos

Desde el esquema v1, un `best.pth` no se considera válido solo porque PyTorch pueda deserializarlo. Cada run relaciona de forma verificable la configuración, el preprocessing, las clases, el split y los bytes exactos del checkpoint. Si un consumidor encuentra una incompatibilidad, aborta antes de ejecutar inferencia.

## Ciclo de vida

```text
best.pth
  → SHA-256 del checkpoint
  → summary.json (schema_version = 1)
  → latest.json atómico
  → validación del contrato
  → construcción del modelo
  → torch.load + load_state_dict
```

La implementación autoritativa vive en `src/training/runs.py`. La escritura JSON reutiliza `atomic_write_json()` de `src/data/preparation.py`: escribe un temporal, ejecuta `flush`/`fsync` y lo publica con `os.replace`.

## `summary.json` v1

Todo run nuevo conserva los campos históricos y agrega, como mínimo:

| Campo | Garantía |
|---|---|
| `schema_version` | Permite rechazar esquemas desconocidos. |
| `run_id` | Debe coincidir con el nombre del directorio del run. |
| `architecture` | Fija `model_name`, `input_size` y `num_classes`. |
| `seed` y `hyperparameters` | Registran los valores efectivos del entrenamiento. |
| `preprocessing` | Describe la entrada determinista utilizada en inferencia. |
| `training_preprocessing` | Declara aparte la política de augmentations aleatorias. |
| `class_to_idx` | Fija por igualdad exacta nombres, índices y orden de salida. |
| `config_sha256` | Huella canónica de arquitectura, seed, hiperparámetros, preprocessing y clases. |
| `split_manifest_sha256` | SHA-256 de `manifest.lock.json`, o fingerprint derivado de los CSV legacy. |
| `split_identifier` | Identificador portable del split, independiente de una ruta de máquina. |
| `best_epoch` y `metrics` | Vinculan el checkpoint con la mejor época de validación. |
| `checkpoint_sha256` | Detecta sustitución o modificación de `best.pth`. |

`config_sha256` se calcula sobre JSON canónico con claves ordenadas. No incluye timestamps, directorios absolutos ni métricas, por lo que dos configuraciones lógicamente iguales producen la misma huella.

## Contrato de preprocessing

`CornTransformFactory.to_contract()` serializa únicamente operaciones deterministas requeridas por evaluación e inferencia:

- orientación EXIF y conversión RGB;
- redimensionamiento, tamaño, interpolación y política de aspecto;
- normalización y sus vectores `mean`/`std`;
- estado y parámetros de CLAHE.

`CornTransformFactory.from_contract()` reconstruye la factory y rechaza valores o esquemas no soportados. Las augmentations aleatorias de `train` y `minority` se documentan en `training_preprocessing`, pero no se aplican durante inferencia.

## Resolución y carga segura

El orden de resolución es:

1. run indicado explícitamente con `--run`;
2. `latest.json` del modelo.

No existe selección contractual por fecha de modificación. `latest.json` usa el formato:

```json
{
  "schema_version": 1,
  "run_id": "20260921_204608"
}
```

`validate_run_contract()` comprueba metadata, configuración solicitada y hash del checkpoint. `load_validated_run()` solo construye el modelo y llama a `torch.load` después de superar esas comprobaciones. Los errores distinguen contrato inválido, incompatibilidad, checkpoint modificado, run legacy y ensemble incompatible.

Los consumidores de predicción, evaluación, exportación, OOD, equidad, explicabilidad y ensembles reutilizan esta carga central. La exportación también registra `run_id`, `checkpoint_sha256`, `class_to_idx`, preprocessing y SHA-256 de cada ONNX/TFLite; `evaluate_export.py` valida esa metadata antes de ejecutar el artefacto.

## Ensembles

Un ensemble normal exige que todos sus miembros compartan:

- `class_to_idx` exacto;
- número de clases;
- preprocessing completo, porque reciben un único tensor de entrada;
- fingerprint del mismo split.

Las políticas explícitas `cross_validation` y `loso` permiten fingerprints de splits distintos, pero mantienen las demás compatibilidades y verifican individualmente cada checkpoint. Si falla un miembro, se rechaza el ensemble completo antes de inferencia.

## Runs legacy

Los runs sin `schema_version` no se cargan en modo estricto. La migración es explícita:

```bash
python scripts/pipeline/migrate_run.py \
  --run-dir outputs/main/<modelo>/<run_id> \
  --preprocessing-contract preprocessing.json \
  --hyperparameters hyperparameters.json \
  --seed 42 \
  --splits-dir outputs/splits/seed_42
```

El migrador reutiliza los campos históricos que puede verificar y exige por argumento cualquier metadata crítica ausente. No inventa `class_to_idx`, preprocessing, hiperparámetros ni split. Una migración válida queda marcada mediante `migration.from_schema`, `method` y `verified`.

## Verificación mínima

```bash
python -m pytest -q \
  tests/training/test_run_contracts.py \
  tests/training/test_best_epoch_integrity.py \
  tests/export/test_common.py \
  tests/models/test_ensemble.py
```

Estas pruebas cubren schema, hashes deterministas, prioridad de run explícito, escritura atómica, mejor época distinta de la última, migración, incompatibilidades y rechazo de checkpoints o exports alterados.
