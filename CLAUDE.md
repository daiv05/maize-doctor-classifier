# CLAUDE.md - Corn Leaf Disease Project

Estado técnico revisado: 11–12 septiembre 2026. Ver [registro de reparación](docs/reviews/2026-09-11-reparacion-integral.md), [corridas recibidas](docs/reviews/2026-09-11-corridas-recibidas.md) y [reproducción/migración](docs/reviews/2026-09-11-reproducibilidad.md). Ninguna prueba local acredita producción Android.

## Reglas de datos

- **Nunca modificar `raw/`.** Es inmutable, solo fuente original.
- `clean/` es la única fuente de verdad para entrenamiento. Estructura: `clean/<clase>/{lab,real}/`.
- Los CSV de `splits/` son derivados reproducibles (`make splits` / `make splits-baseline`). No editarlos a mano. Viven en `outputs/splits/` (ver más abajo), no bajo `DATASET_ROOT`.

## Arquitectura (`src/`)

- **Único punto de entrada a imagen:** `load_and_normalize_image()` (`src/data/loader.py`).
- **Rutas:** dataset fuente vía `get_dataset_root()`, artefactos generados vía `get_output_root()` (ambas en `src/config.py`) - nunca hardcodear paths ni usar la constante `DATASET_ROOT` directo.
- **Config centralizada:** `config/dataset.yaml` (clases, `target_size`, seed, perfil `baseline`). Nunca hardcodear constantes de dominio.
- **Sin `sys.path.append`.** Paquete editable (`pip install -e .`); los imports `src.*` resuelven directo.
- Convenciones detalladas de carga/rutas/`target_size`/clases minoritarias → skill `corn-data-pipeline`. Sampler de balanceo, utilidades de entrenamiento y versionado de runs → skill `corn-training-internals`.
- Para ubicar símbolos, llamadas o impacto de cambios en `src/`, usa CodeGraph (si está disponible) en vez de grep/lectura manual.

## Pipelines

- **Datos:** `clean/<clase>/{lab,real}/` → `create_splits.py` (valida integridad PIL, deduplica por SHA-256 con escaneo `sorted()` - determinista entre máquinas -, estratifica por `label+environment`) → `outputs/splits/seed_42/` (9 clases) o `outputs/splits/seed_42_baseline/` (`--baseline`, subset de `config/dataset.yaml -> baseline:`).
- **Baselines (funcional, PyTorch):** `CornDataset` → `WeightedRandomSampler` → `DataLoader` → `MODEL_REGISTRY.build(<efficientnet_b0|efficientnet_lite0|mobilenet_v3_large|fastvit_t8|ghostnetv2_100|shufflenet_v2_x1_0>)` vía `train_baselines.py`. Pese al nombre, no es un pipeline sklearn - es DL completo, pensado para comparar arquitecturas rápido y barato. Solo con `--evaluate-test`, cada run escribe `predictions.csv` (predicción + confianza por imagen de test), usado por los subcomandos `fidelity`/`errors` de `scripts/pipeline/explain.py` para el análisis de errores.
- **Principal (`train.py`):** comparte toda la infraestructura de datos/modelos con baselines. Entrena una arquitectura (default `shufflenet_v2_x1_0`) sobre el dataset completo (`outputs/splits/seed_42`, 33 433 filas en los splits recibidos el 11 de septiembre; la revisión HF anuncia 33 437 antes de deduplicar; los conteos históricos dependen de versión) con pérdida ponderada (`sqrt_inverse`) + label smoothing, scheduler cosine con warmup, early stopping y gradient clipping. Ojo: ese default de `shufflenet_v2_x1_0` es el de `train.py --models` cuando se omite el flag; `make train`/`make train-main` siempre pasan `--models $(MAIN_MODELS)` explícitamente, y `MAIN_MODELS` por defecto son **tres** modelos (`efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0`, ver `Makefile`). Para entrenar solo `shufflenet_v2_x1_0` vía `make`, pasar `MAIN_MODELS=shufflenet_v2_x1_0` explícitamente. El `WeightedRandomSampler` va **desactivado**: con el desbalance del dataset (32.9x en la primera etapa, 14.1x tras la ampliación), sampler + pérdida ponderada sobre-compensaría el mismo desbalance por dos vías, y augmenta solo las clases minoritarias. Test es opt-in (`--evaluate-test`). Solo al evaluarlo, además de las métricas estándar, escribe `test_calibration.json` (incluye `brier_binary_hit`, un Brier **binario** de acierto - el multiclase no es calculable porque `predictions.csv` guarda `pred_prob` escalar), `test_by_environment.csv` (formato largo: fila agregada `class == "__all__"` con accuracy/macro-F1, más una fila por clase con su `f1` y su `n`) y `test_grouped_metrics.json`. CLAHE es opt-in vía `--clahe` (CLI) / `CLAHE=1` (Makefile). Su explicabilidad incluye SHAP (subcomandos `compare`/`global`), exclusivo de este pipeline.
- **Explicabilidad (post-hoc, no acoplada al entrenamiento):** `scripts/pipeline/explain.py` unifica cinco subcomandos - `visual` (LIME + Grad-CAM por imagen), `fidelity` (agregado por clase), `errors` (dirigido a `label != pred_label`), `compare` (LIME | SHAP | Grad-CAM + acuerdo) y `global` (perfil global por clase con SHAP) -, más `scripts/checks/lime_stability.py` (auditoría manual de estabilidad de LIME). `compare` y `global` son exclusivos del pipeline principal. Ver sección "Explicabilidad" más abajo.
- **Exportación (ONNX/TFLite, solo pipeline principal):** `scripts/pipeline/export.py` convierte checkpoints ya entrenados (`outputs/main/<modelo>/<run_id>/best.pth`) a `export/model.<fmt>`, validando paridad numérica contra una muestra de validación (`ParityResult.passed`, ver `export/export_summary.json`). Acepta **varios** modelos (`--models`, los 3 de `MAIN_MODELS` por defecto). `train.py --export onnx,tflite` exporta automáticamente al terminar un run (reutiliza el modelo y el loader de validación, no bloquea el entrenamiento si falla). Lógica compartida en `src/export/` (`export_model()` es el punto de entrada único; además de `export_summary.json`/`export_summary_int8.json`, escribe `labels.json` - el orden de clases del modelo, para que la app móvil u otro consumidor no tenga que re-derivarlo ni hardcodearlo).
  - **TFLite va por `litert-torch`**, no por `ai-edge-torch`: ese paquete quedó deprecado y sus versiones >=0.7 son un stub que solo emite un `DeprecationWarning` y **no expone `convert()`**, así que un `import` exitoso no implica que se pueda exportar. Solo soporta Linux (de ahí el marcador `sys_platform` en el extra `export`).
  - **Cuantización opt-in:** `--quantize int8` (CLI) / `QUANTIZE=int8` (Makefile) escribe `model_int8.<fmt>` **junto** al FP32 sin pisarlo, con su propio `export_summary_int8.json`. Los umbrales de paridad se relajan solos para el modo cuantizado (`_PARITY_DEFAULTS`): exigirle la tolerancia de FP32 lo reprobaría siempre.
  - Separar entornos ONNX y LiteRT (`.[onnx]` / `.[tflite]`). Ejecutar `pip check` y resolver conflictos reales; no forzar dependencias transitivas ni declarar inocuos los avisos del resolutor. Las resoluciones Linux/Python 3.12 comprobadas están en `constraints/`. LiteRT FP32 pasó paridad real para ambos checkpoints recibidos; B0 INT8 falló y no debe entregarse. Esto no acredita Android.
  - **ONNX sale en un solo archivo:** el exportador de torch saca los pesos a un `model.onnx.data` aparte, y un `.onnx` sin su sidecar carga pero falla al inferir - trampa fea dentro de un bundle móvil. `_consolidate_single_file()` los une. Para cuantizar hay que borrar antes el `value_info` del grafo: el exportador dynamo deja anotaciones de shape inconsistentes que abortan la inferencia de shapes de onnxruntime.
- **Evaluación de lo exportado:** `scripts/pipeline/evaluate_export.py` (`make eval-export-main`) corre el **archivo exportado** sobre el split de test completo y reporta accuracy/macro-F1 reales, desglose por entorno y delta contra PyTorch (`export/eval_<fmt>[_<quant>].json` + CSV por imagen). Es distinto de la paridad que valida `export.py`, que solo compara ~30 muestras numéricamente: una int8 puede pasar la paridad y aun así perder macro-F1 en las clases minoritarias. Falla si la caída supera `--max-macro-f1-drop` (0.01 por defecto).

## Clases del dataset

Definidas en `config/dataset.yaml -> dataset.classes` (orden canónico para `class_to_idx`). Ratios históricos de desbalance vs. `healthy` (ampliación de agosto 2026; recomputar para cada versión):
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

Para actualizar Modal: `make modal-seed FORCE=1` descarga y valida en staging, conserva
`clean` anterior como backup y activa solo una versión completa. Los splits son inmutables:
crear una versión nueva con `create_splits.py --output-dir`, no volver a escribir la anterior.
El flujo segmentado hereda IDs y particiones originales; no las vuelve a sortear.

Revisión 12/09: cuatro pares de bytes idénticos tienen etiquetas healthy/fall_armyworm
contradictorias. `create_splits --exclusions CSV` verifica hashes y excluye provisionalmente
las ocho entradas sin tocar `clean`. `--group-manifest CSV` acepta sample_id o image_path
y group_id; protege únicamente las relaciones suministradas. Nueva versión local:
`outputs/repair-20260912/splits-reviewed-quarantine-v3` (33 429 imágenes), todavía con
revisión de duplicados cercanos y etiquetas pendiente. Etapa 2 puede heredar esos splits
con `prepare --source-splits`, sin volver a sortearlos.

## Contratos de integridad

- `sample_id` identifica el original; `sha256` verifica sus bytes. Nunca sustituir una imagen ausente/corrupta por la fila siguiente.
- `training/runs.py` centraliza resolución y carga estricta. Run/checkpoint explícito inexistente falla; no hay fallback a latest. Clases y preproceso vienen del contrato, no del subconjunto evaluado.
- Los ensembles requieren un manifest con miembros, pesos y hashes. Un CSV de probabilidad máxima no permite reconstruir soft voting.
- HPO es opt-in con `--best-params`: CLI > JSON validado > defaults. SGD de Etapa 2 no es un esquema PyTorch compatible.
- `fit` restaura la mejor época aun sin run_dir; CV persiste checkpoint/summary/predicciones por fold. Test es opt-in, y CV posterior a HPO no es CV anidada.
- Cada caché se vincula a IDs/orden/labels/bytes/backbone/preproceso. Features y locks históricos incompletos se rechazan; usar nueva versión.
- Para inferir imágenes originales con un modelo segmentado, proporcionar el checkpoint exacto y respetar perfil/runtime/fallback. Máscaras inciertas no validan transferencia de etiqueta: revisión humana pendiente.
- Archivo exportado creado != paridad aprobada != entrega móvil. Sync exige evaluación completa, caída macro-F1 ≤0.01, clases/preproceso/OOD coherentes, staging y rollback. No declarar validación Android sin dispositivo.

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
make splits / make splits-baseline    # crea splits si el destino está vacío; nueva versión para repetir
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
