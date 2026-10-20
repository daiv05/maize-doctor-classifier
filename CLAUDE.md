# CLAUDE.md - Corn Leaf Disease Project

## Reglas de datos

- **Nunca modificar `raw/`.** Es inmutable, solo fuente original.
- `clean/` es la única fuente de verdad para entrenamiento. Estructura: `clean/<clase>/{lab,real}/`.
- Los CSV de `splits/` son derivados reproducibles (`make splits` / `make splits-baseline`). No editarlos a mano. Viven en `outputs/splits/` (ver más abajo), no bajo `DATASET_ROOT`.

### Contratos que no deben romperse

- `sample_id = SHA256(ruta relativa normalizada UTF-8)`: identifica la muestra lógica y no depende de sus bytes. El SHA-256 de contenido se calcula durante preparación, nunca dentro de `CornDataset.__getitem__`.
- `master_manifest.csv`, `manifest.lock.json`, `preparation_audit.json` y `split_audit_report.csv` son el contrato de una materialización; no inferir identidad por número de fila.
- `CornDataset` devuelve `(image, label, sample_id)` y conserva `ImageCache`, `max_per_class`, selección de `minority_classes`, transforms y `class_to_idx`. Un fallo carga **esa misma muestra o levanta un error trazable**; nunca cambia a `idx+1`, devuelve `None`, filtra en `collate_fn` ni muta el DataFrame.
- `seed_42` es desarrollo estratificado por `label + environment`; `source_id` es metadato. `seed_42_source_grouped` es un benchmark separado con fuentes indivisibles. No sustituir uno por otro ni llamar LOSO a ninguno.
- La materialización actual no ejecutó PHash (`deduplicate_perceptual=false`): solo está demostrado el cero solapamiento exacto.
- Especificaciones canónicas: `docs/es/metodologia/pipeline-datos.md`, `docs/es/metodologia/protocolos-experimentales.md` y `docs/es/pipeline/contratos-runs.md`. Estado y evidencia: `docs/es/experimentos/current-status.md` y `docs/es/tesis/EVIDENCE_REGISTRY.md`.

## Arquitectura (`src/`)

- **Único punto de entrada a imagen:** `load_and_normalize_image()` (`src/data/loader.py`).
- **Rutas:** dataset fuente vía `get_dataset_root()`, artefactos generados vía `get_output_root()` (ambas en `src/config.py`) - nunca hardcodear paths ni usar la constante `DATASET_ROOT` directo.
- **Config centralizada:** `config/dataset.yaml` (clases, `target_size`, seed, perfil `baseline`). Nunca hardcodear constantes de dominio.
- **Sin `sys.path.append`.** Paquete editable (`pip install -e .`); los imports `src.*` resuelven directo.
- Convenciones detalladas de carga/rutas/`target_size`/clases minoritarias → skill `corn-data-pipeline`. Sampler de balanceo, utilidades de entrenamiento y versionado de runs → skill `corn-training-internals`.
- Para ubicar símbolos, llamadas o impacto de cambios en `src/`, usa CodeGraph (si está disponible) en vez de grep/lectura manual.

## Pipelines

- **Datos:** `clean/<clase>/{lab,real}/` → `create_splits.py` (valida integridad PIL, aplica `config/dataset_exclusions.csv`, deduplica por SHA-256 con escaneo `sorted()` - determinista entre máquinas -, estratifica por `label+environment`) → `outputs/splits/seed_42/` (9 clases) o `outputs/splits/seed_42_baseline/` (`--baseline`, subset de `config/dataset.yaml -> baseline:`). La materialización vigente de `seed_42` tiene 33 429 muestras (23 400 / 5 014 / 5 015), sin solapamientos de `sample_id`, SHA-256 ni `effective_group_id`; se excluyen de forma explícita 8 archivos con contenido idéntico y etiquetas contradictorias. El benchmark opt-in `--group-by-source` conserva fuentes completas y se guarda aparte como `seed_42_source_grouped`, para no reemplazar los splits de desarrollo usados al entrenar. La ejecución vigente no activó deduplicación perceptual, por lo que no prueba ausencia de casi duplicados.
- **Baselines (funcional, PyTorch):** `CornDataset` → `WeightedRandomSampler` → `DataLoader` → `MODEL_REGISTRY.build(<efficientnet_b0|efficientnet_lite0|mobilenet_v3_large|fastvit_t8|ghostnetv2_100|shufflenet_v2_x1_0>)` vía `train_baselines.py`. Pese al nombre, no es un pipeline sklearn - es DL completo, pensado para comparar arquitecturas rápido y barato. Cada run también escribe `predictions.csv` (predicción + confianza por imagen de test), usado por los subcomandos `fidelity`/`errors` de `scripts/pipeline/explain.py` para el análisis de errores.
- **Principal (`train.py`):** comparte toda la infraestructura de datos/modelos con baselines. Entrena una arquitectura (default `shufflenet_v2_x1_0`) sobre el dataset completo (`outputs/splits/seed_42`, 33 429 muestras elegibles en la materialización vigente; 31 623 antes de la ampliación) con pérdida ponderada (`sqrt_inverse`) + label smoothing, scheduler cosine con warmup, early stopping y gradient clipping. Ojo: ese default de `shufflenet_v2_x1_0` es el de `train.py --models` cuando se omite el flag; `make train`/`make train-main` siempre pasan `--models $(MAIN_MODELS)` explícitamente, y `MAIN_MODELS` por defecto son **tres** modelos (`efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0`, ver `Makefile`). Para entrenar solo `shufflenet_v2_x1_0` vía `make`, pasar `MAIN_MODELS=shufflenet_v2_x1_0` explícitamente. El `WeightedRandomSampler` va **desactivado**: con el desbalance del dataset (32.9x en la primera etapa, 14.1x tras la ampliación), sampler + pérdida ponderada sobre-compensaría el mismo desbalance por dos vías, y augmenta solo las clases minoritarias. Además de las métricas estándar, escribe `test_calibration.json` (incluye `brier_binary_hit`, un Brier **binario** de acierto - el multiclase no es calculable porque `predictions.csv` guarda `pred_prob` escalar), `test_by_environment.csv` (formato largo: fila agregada `class == "__all__"` con accuracy/macro-F1, más una fila por clase con su `f1` y su `n`) y `test_grouped_metrics.json`. CLAHE es opt-in vía `--clahe` (CLI) / `CLAHE=1` (Makefile). Su explicabilidad incluye SHAP (subcomandos `compare`/`global`), exclusivo de este pipeline.
- **Explicabilidad (post-hoc, no acoplada al entrenamiento):** `scripts/pipeline/explain.py` unifica cinco subcomandos - `visual` (LIME + Grad-CAM por imagen), `fidelity` (agregado por clase), `errors` (dirigido a `label != pred_label`), `compare` (LIME | SHAP | Grad-CAM + acuerdo) y `global` (perfil global por clase con SHAP) -, más `scripts/checks/lime_stability.py` (auditoría manual de estabilidad de LIME). `compare` y `global` son exclusivos del pipeline principal. Ver sección "Explicabilidad" más abajo.
- **Exportación (ONNX/TFLite, solo pipeline principal):** `scripts/pipeline/export.py` convierte checkpoints ya entrenados (`outputs/main/<modelo>/<run_id>/best.pth`) a `export/model.<fmt>`, validando paridad numérica contra una muestra del split de test (`ParityResult.passed`, ver `export/export_summary.json`). Acepta **varios** modelos (`--models`, los 3 de `MAIN_MODELS` por defecto). `train.py --export onnx,tflite` exporta automáticamente al terminar un run (reutiliza el modelo y el `test_loader` ya en memoria, no bloquea el entrenamiento si falla). Lógica compartida en `src/export/` (`export_model()` es el punto de entrada único; además de `export_summary.json`/`export_summary_int8.json`, escribe `labels.json` - el orden de clases del modelo, para que la app móvil u otro consumidor no tenga que re-derivarlo ni hardcodearlo).
  - **TFLite va por `litert-torch`**, no por `ai-edge-torch`: ese paquete quedó deprecado y sus versiones >=0.7 son un stub que solo emite un `DeprecationWarning` y **no expone `convert()`**, así que un `import` exitoso no implica que se pueda exportar. Solo soporta Linux (de ahí el marcador `sys_platform` en el extra `export`).
  - **Cuantización opt-in:** `--quantize int8` (CLI) / `QUANTIZE=int8` (Makefile) escribe `model_int8.<fmt>` **junto** al FP32 sin pisarlo, con su propio `export_summary_int8.json`. Los umbrales de paridad se relajan solos para el modo cuantizado (`_PARITY_DEFAULTS`): exigirle la tolerancia de FP32 lo reprobaría siempre.
  - Instalar el extra `export` degrada `typing-extensions` a 4.12.2 (pin viejo de `xdsl`, dependencia transitiva de `litert-torch`). No afecta al proyecto, pero rompe `jupyter_client`; se recupera con `pip install "typing-extensions>=4.14"`, y el aviso de pip sobre `xdsl` que queda es inocuo.
  - **ONNX sale en un solo archivo:** el exportador de torch saca los pesos a un `model.onnx.data` aparte, y un `.onnx` sin su sidecar carga pero falla al inferir - trampa fea dentro de un bundle móvil. `_consolidate_single_file()` los une. Para cuantizar hay que borrar antes el `value_info` del grafo: el exportador dynamo deja anotaciones de shape inconsistentes que abortan la inferencia de shapes de onnxruntime.
- **Evaluación de lo exportado:** `scripts/pipeline/evaluate_export.py` (`make eval-export-main`) corre el **archivo exportado** sobre el split de test completo y reporta accuracy/macro-F1 reales, desglose por entorno y delta contra PyTorch (`export/eval_<fmt>[_<quant>].json` + CSV por imagen). Es distinto de la paridad que valida `export.py`, que solo compara ~30 muestras numéricamente: una int8 puede pasar la paridad y aun así perder macro-F1 en las clases minoritarias. Falla si la caída supera `--max-macro-f1-drop` (0.01 por defecto).
- **Contratos de runs:** `src/training/runs.py` es la fuente autoritativa para crear, resolver, validar y migrar runs. Los nuevos `summary.json` usan `schema_version=1` y fijan arquitectura, hiperparámetros efectivos, preprocessing, `class_to_idx`, fingerprint del split, mejor época y SHA-256 de `best.pth`. Los consumidores deben usar `load_validated_run()`; un run legacy requiere `scripts/pipeline/migrate_run.py` y no se carga silenciosamente. Detalle en `docs/es/pipeline/contratos-runs.md`.

## Clases del dataset

Definidas en `config/dataset.yaml -> dataset.classes` (orden canónico para `class_to_idx`). Ratios de desbalance vs. `healthy` (corpus actual, tras la ampliación de agosto 2026):
`common_rust` (3.9x), `gray_leaf_spot` (4.5x), `phosphorus_deficiency` (9.3x), `nitrogen_deficiency` (10.3x), `potassium_deficiency` (14.1x).
El pipeline extendido de augmentación / el `WeightedRandomSampler` se activan con umbral estricto `max_count/count > 4.0`, así que sobre el dataset completo califican `gray_leaf_spot`, `nitrogen_deficiency`, `phosphorus_deficiency` y `potassium_deficiency` (no `common_rust`, que queda en 3.9x).
Los ratios previos a la ampliación eran 7.9x / 14.3x / 16.8x / 32.9x respectivamente: **la selección de clases minoritarias no cambió**, solo su magnitud. Ojo con `gray_leaf_spot`: con 4.5x quedó cerca del umbral, y reforzarlo más lo sacaría del grupo, cambiando el comportamiento del entrenamiento.
El perfil `baseline` (`config/dataset.yaml -> baseline:`) usa las 9 clases con un tope de 1500 img/clase (`max_images_per_class`).

## Explicabilidad

Post-hoc, no acoplada al entrenamiento: `scripts/pipeline/explain.py` (subcomandos `visual`, `fidelity`, `errors`, `compare`, `global`), `scripts/checks/lime_stability.py` (auditoría manual). LIME ya no corre automáticamente al entrenar (usar flag `--lime` puntual en `train_baselines.py`).

- Flujo XAI (`make explain-visual`/`fidelity`/`errors`/`compare`/`global`, `lime_stability.py`) → skill `corn-xai`.
- Grad-CAM (`GRADCAM_TARGET_LAYERS`, requisito al añadir modelos nuevos) → skill `corn-gradcam`.

## Dataset: hosting y descarga

`clean/` (~33k imágenes, ~19 GB) vive en Hugging Face Datasets Hub (fuente primaria) con Google Drive de respaldo;
`download_dataset.py --source auto` resuelve cuál usar. `scripts/download_datasets.sh` es un flujo distinto:
ingesta de fuentes crudas nuevas (Kaggle/Mendeley/Roboflow) hacia `raw/`, no toca `clean/`.

En HF el dataset **no** se publica como archivos sueltos sino en shards `clean-<NNNNN>.tar` de ~800 MB
que contienen el árbol `<clase>/<entorno>/<archivo>`: 33k blobs individuales implicarían 33k requests
HTTP por descarga, prohibitivo sobre el Volume remoto de Modal. `download_dataset.py` los extrae y borra
tras bajarlos, reconstruyendo `clean/<clase>/{lab,real}/` — el resto del pipeline no se entera del formato.
Se acompañan de metadata ligera (`metadata.csv` con `file_name`/`label`/`environment`, `dataset_infos.json`)
para que HF reconozca el repo como dataset de clasificación de imágenes; se excluye al descargar. No se
suben Parquet con imágenes embebidas: duplicarían el tamaño del repo a cambio del Dataset Viewer.
Subida: `make upload-dataset STAGE_DIR=<dir con ~19 GB libres>` (`DRY_RUN=1` para ver el plan de shards).

Para propagar una actualización a Modal: `make modal-seed FORCE=1` y luego `make modal-splits`.
`FORCE=1` vacía `/data/clean` antes de descargar - sin eso `seed_dataset` es un no-op (el Volume
ya tiene contenido), y forzar solo la descarga tampoco basta: los shards se extraen sobre el árbol
existente y `snapshot_download` no borra lo que desapareció del repo, así que un archivo renombrado
aguas arriba sobreviviría con sus dos nombres. Los splits del Volume `corn-outputs` quedan obsoletos
tras cambiar el dataset y hay que regenerarlos.


## Comandos frecuentes

Convención de nombres: prefijo `modal-` = GPU en la nube, sin prefijo = local; sufijo
`-baselines` = runs de `outputs/baselines` (var `MODELS`), sufijo `-main` = runs de
`outputs/main` (var `MAIN_MODELS`). Los targets sin sufijo son los genéricos y apuntan a
baselines salvo que se pase `OUTPUT_DIR` (local) o `PIPELINE=main` (Modal). `make help`
lista todo agrupado.

```bash
make install                          # pip install -e ".[dev,analysis,xai,cloud]"
                                       # + ",export" si necesitas exportar a ONNX/TFLite
make download-dataset                 # clean/ (HF Hub, fallback Google Drive)
make upload-dataset STAGE_DIR=<dir>   # empaqueta clean/ en shards .tar y sube a HF [DRY_RUN=1]
make splits / make splits-baseline    # regenera splits CSV
make train-baselines [MODELS=<nombre> NO_CAP=1|MAX_PER_CLASS=<n>]
make train-main [MAIN_MODELS=<nombre> MAIN_EPOCHS=<n> CLAHE=1 CLASS_WEIGHTS=<estrategia>]  # alias: make train
make modal-train-baselines [MODELS=<nombre> EPOCHS=<n>]        # baselines en GPU
make modal-train-main [MAIN_MODELS=<nombre> MAIN_EPOCHS=<n>]   # principal en GPU (alias: modal-train)
make explain-visual-baselines [MODELS=<nombre>]   # reporte visual LIME+Grad-CAM post-hoc
make explain-fidelity-baselines [MODELS=<nombre> SAMPLE_SIZE=<n>]  # fidelidad agregada
make explain-errors-baselines [MODELS=<nombre>] # LIME dirigido a falsos positivos/negativos
make explain-visual-main / explain-fidelity-main / explain-errors-main [MAIN_MODELS=<nombre>]
make explain-compare-main [MAIN_MODELS=<nombre>]   # panel LIME | SHAP | Grad-CAM + acuerdo (solo main)
make explain-global-main [MAIN_MODELS=<nombre>]    # perfil global por clase con SHAP (solo main)
make modal-explain-visual-baselines / modal-explain-fidelity-baselines / modal-explain-errors-baselines
make modal-explain-visual-main / modal-explain-fidelity-main / modal-explain-errors-main
make modal-explain-compare-main / modal-explain-global-main
make export-main [EXPORT_FORMATS=onnx,tflite QUANTIZE=int8 MAIN_MODELS=<nombre> RUN=<run_id>]  # solo main
make eval-export-main [EXPORT_FORMATS=onnx,tflite QUANTIZE=int8]  # mide el exportado sobre el test completo
make modal-export-main / modal-eval-export-main [MAIN_MODELS=<nombre> EXPORT_FORMATS=onnx,tflite QUANTIZE=int8]
make summary                          # conteo de imágenes por clase/entorno
make test-loader                      # smoke check del pipeline de carga
make lint / make fmt                  # ruff check / ruff format
```

## Setup local

Ver [LOCAL.md](LOCAL.md) para levantar el proyecto (venv, `.env`, descarga del dataset).
