# Reproducción y migración después de la reparación

Fecha de referencia: 11–12 septiembre 2026. Ejecutar desde la raíz del repositorio.
Los ejemplos no autorizan gasto en Modal ni evaluación repetida del holdout.

## Entornos separados

La suite local se ejecutó con Python 3.12.3/Linux x86_64, PyTorch 2.12.1,
torchvision 0.27.1, NumPy 2.5.1, pandas 2.3.3, sklearn 1.9.0, pytest 9.1.1,
Optuna 4.9.0, ONNX 1.22.0 y ONNX Runtime 1.28.0. Los pesos de prueba se ejecutaron
en CPU; no se acredita CUDA ni Android por esas pruebas. Posteriormente se instalaron
dos venv limpios separados y se ejecutaron tanto ONNX como LiteRT con runtimes reales.

`constraints/cpu-2026-09-11.txt` es un inventario observado: pip freeze más versiones
de módulos importados. El target temporal retuvo metadatos antiguos de Optuna/ORT,
por eso se contrastó el número con `__version__`; no se presentó ese target mezclado
como un lock limpio. El entorno vecino también contiene dos distribuciones OpenCV;
no se propone reproducir esa superposición.

La segunda resolución **sí** proviene de `pip install --dry-run --ignore-installed
--report`, terminó correctamente con 99 distribuciones y produjo
`constraints/resolved-linux-py312-2026-09-12.txt`. Sus hashes y plataforma están en
[la evidencia](evidence/2026-09-12-dependency-resolution.json). Los wheels PyPI de torch
incluyen dependencias CUDA aunque las pruebas sean CPU. No es un lock universal para
otra plataforma ni una instalación CPU-only del índice de PyTorch.

Se instaló y ejecutó realmente esa resolución en `/tmp/corn-repair-clean-Rq8ucd`;
`pip check` no encontró dependencias rotas. Para la prueba opcional de CLAHE sobre una
imagen real se usó `CORN_TEST_REAL_IMAGE` apuntando a
`clean/gray_leaf_spot/real/gray_leaf_spot_maize_field_real_13749986.jpg` del dataset HF.
Sin esa variable/archivo se omite ese ancla real, no los tests sintéticos de CLAHE.

```bash
python3.12 -m venv .venv-repair
source .venv-repair/bin/activate
python -m pip install -c constraints/resolved-linux-py312-2026-09-12.txt -e '.[test,xai,onnx]'
python -m pip check
MPLCONFIGDIR=/tmp/doctor-maiz-mpl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

El extra `test` evita instalar Jupyter y herramientas de análisis masivo para una regresión.
`onnx` separa exportación ONNX de `tflite`; `export` mantiene ambos por compatibilidad.
LiteRT se comprobó en `/tmp/corn-repair-litert-QnPicA`, con resolución real de pip,
instalación y `pip check` satisfactorios. Pruebas reales: 2 aprobadas, 5 avisos,
13,52 s; [XML](evidence/2026-09-12-pytest-litert.xml). Incluyen logits y features.
Para reproducir en otro venv Linux/Python 3.12:

```bash
python3.12 -m venv .venv-litert
source .venv-litert/bin/activate
python -m pip install -c constraints/litert-linux-py312-2026-09-12.txt \
  -c constraints/litert-test-linux-py312-2026-09-12.txt \
  -e '.[tflite]' pytest==9.1.1 matplotlib==3.11.1
python -m pip check
MPLCONFIGDIR=/tmp/doctor-maiz-mpl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/export/test_tflite_export.py -q
```

No forzar versiones transitivas después de resolver ni considerar un conflicto «inocuo».

Para segmentación: venv separado con `pip install -e '.[segmentation]'`. Ultralytics
8.4.104 queda declarado, pero sus dependencias deben comprobarse con `pip check`.
No mezclar ad hoc OpenCV GUI/headless o el entorno LiteRT con el de entrenamiento.
Las imágenes de Modal tienen su propia resolución y no fueron ejecutadas en la nube.

## Dataset y splits nuevos

Crear una raíz nueva con espacio suficiente (al menos 40 GB durante descarga/extracción):

```bash
mkdir -p /ruta/datos/maiz-hf-e515ab2
export DATASET_ROOT=/ruta/datos/maiz-hf-e515ab2
export OUTPUT_ROOT=/ruta/salidas/doctor-maiz-repair
python -m pip install -e '.[cloud]'
HF_DOWNLOAD_WORKERS=2 HF_HUB_DOWNLOAD_TIMEOUT=60 python -m scripts.dataset.download_dataset \
  --source hf --hf-repo daiv05/corn-leaf-diseases-pests-and-deficiencies \
  --revision e515ab2f1e4c5729f8447520f1a630cf14c532dc
```

**La revisión fijada contiene cuatro conflictos healthy/fall_armyworm.** Preparar sin
exclusiones aborta intencionalmente; no copiar automáticamente la primera etiqueta.
Para reproducir la exclusión provisional autorizada y el único par cercano confirmado:

```bash
python -m scripts.checks.build_reviewed_groups \
  --splits-dir outputs/outputs-11092026/splits/seed_42 \
  --reviewed-pairs docs/reviews/evidence/2026-09-12-reviewed-near-pairs.csv \
  --output-dir "$OUTPUT_ROOT/groups/reviewed-v1"
python -m scripts.pipeline.create_splits \
  --output-dir "$OUTPUT_ROOT/splits/hf-e515ab2-reviewed-v1" \
  --group-manifest "$OUTPUT_ROOT/groups/reviewed-v1/groups.csv" \
  --exclusions docs/reviews/evidence/2026-09-12-provisional-exclusions.csv
```

La generación de grupos necesita los CSV históricos recibidos (están bajo outputs,
no se descargan con Git). Esta agrupación es parcial: quedan 372 pares candidatos sin
resolver. Para la ejecución realizada ver [dataset y piloto](2026-09-12-piloto-y-dataset.md).

Una raíz de datos explícita inexistente falla; OUTPUT_ROOT puede crearse. `clean/` solo
se activa después de validar imágenes, clases y hashes. `--force` conserva el árbol anterior
en un backup; no mezcla shards nuevos con archivos viejos. Si HF falla, conservar el
staging y reanudar con `--source hf --revision <mismo SHA> --resume-staging <ruta impresa>`.
Un staging con otro origen o revisión se rechaza. No usar staging incompleto para entrenar.

Los splits no se sobrescriben: otra preparación requiere otro `--output-dir`. Si hay
metadatos de planta/sesión/derivado, proporcionar `--group-manifest` con
`sample_id,group_id` o `image_path,group_id`.
No inventar grupos de planta a partir del nombre. La agrupación tiene prioridad sobre
proporciones exactas y falla si no se cubren las clases. Hash exacto no descarta duplicados
cercanos; su revisión y la procedencia de campo siguen siendo necesarias.

## Corridas anteriores: copia reconstruida, nunca modificación de originales

El cargador estricto rechaza summaries sin preprocesamiento/hash. Para los dos runs
originales revisados (CLAHE false y 224×224), crear una copia nueva:

```bash
python -m scripts.checks.migrate_legacy_run \
  --source-run outputs/outputs-11092026/main/efficientnet_b0/20260910_170120 \
  --splits-dir outputs/outputs-11092026/splits/seed_42 \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/reconstructed/efficientnet_b0/r1" --acknowledge-legacy
python -m scripts.checks.migrate_legacy_run \
  --source-run outputs/outputs-11092026/main/shufflenet_v2_x1_0/20260910_184521 \
  --splits-dir outputs/outputs-11092026/splits/seed_42 \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/reconstructed/shufflenet_v2_x1_0/r1" --acknowledge-legacy
```

Se verifica cada ruta y contenido actual, se heredan las particiones, se copia `best.pth`
y el summary antiguo se conserva como `legacy_summary.json`. Las métricas antiguas no
se promueven a resultados del contrato nuevo. No se puede demostrar retrospectivamente
qué bytes cargó el entrenamiento ni la equivalencia exacta de su preprocesamiento.
La migración falla para segmentación antigua desconocida o metadatos insuficientes.

Etapa 2 requiere regenerar features en una versión nueva, no «firmar» cachés antiguas:

```bash
python -m scripts.etapa_2.stage2_experiments --output-dir "$OUTPUT_ROOT/etapa2-v2" \
  prepare --dataset-root "$DATASET_ROOT" \
  --source-splits "$OUTPUT_ROOT/splits/hf-e515ab2-reviewed-v1"
python -m scripts.etapa_2.stage2_experiments --output-dir "$OUTPUT_ROOT/etapa2-v2" \
  extract --dataset-root "$DATASET_ROOT" --models efficientnet_b0 shufflenet_v2_x1_0
```

`--source-splits` valida el lock, los bytes actuales, IDs y grupos; hereda exactamente
train/val/test (test se llama holdout), sin reintroducir exclusiones ni volver a dividir.
Preparación real completada: `outputs/repair-20260912/etapa2-inherited-v1`, 33 429 filas,
0 corruptas. La extracción completa de features de esta nueva versión no se ejecutó:
antes del nuevo entrenamiento científico falta completar la revisión de grupos/etiquetas.

Antes de `final`, congelar selección y parámetros. Su recibo permite recuperar una
interrupción con exactamente la misma configuración; no permite cambiar la selección
para consultar otra vez el holdout. CV posterior a HPO sobre desarrollo no es CV anidada
y sus intervalos descriptivos no eliminan el sesgo de selección ni dependencia entre folds.

## Piloto de desarrollo ejecutable

Ya se ejecutó con 288 imágenes y tres semillas; resultados, límites y artefactos en
[el informe del piloto](2026-09-12-piloto-y-dataset.md). La nueva versión del script
distingue delta observado/media bootstrap y estratifica previews también por split;
el protocolo v1 retiene el hash exacto del script con el que se ejecutó.

```bash
python -m scripts.checks.paired_segmentation_pilot \
  --dataset-root "$DATASET_ROOT" \
  --splits-dir outputs/outputs-11092026/splits/seed_42 \
  --checkpoints "$OUTPUT_ROOT/reconstructed/efficientnet_b0/r1/best.pth" \
    "$OUTPUT_ROOT/reconstructed/shufflenet_v2_x1_0/r1/best.pth" \
  --segmenter-checkpoint /ruta/doctor_maiz_leaf_segmenter_best.pt \
  --output-dir "$OUTPUT_ROOT/pilots/original-segmented-v1" \
  --train-per-stratum 12 --val-per-stratum 12 --training-seeds 42 43 44 --epochs 20
```

No carga imágenes de test. Congela el protocolo antes de inferir, conserva todos los IDs
seleccionados y contabiliza incertidumbre/rechazo como fallback original. Genera matriz,
macro-F1, métricas por clase/entorno, soportes, bootstrap pareado y artefactos con hashes.
El bootstrap por imagen no modela dependencia entre plantas sin metadatos.

Dos análisis distintos: (1) sensibilidad de los pesos originales a entradas segmentadas;
(2) heads de igual presupuesto sobre ImageNet congelado, original vs segmentado, tres
semillas. Un resultado del segundo no equivale a fine-tuning completo. Revisar
`human_review.csv` (N/P/K, campo, rechazos y ambigüedades) antes de calibrar umbrales.
No introducir variación de fondo ni cambiar umbrales observando test. El subconjunto
estratificado no estima automáticamente el rendimiento de las 33 mil imágenes.

## Entrenamiento, evaluación y entrega

`train` y `cross_validate` consumen HPO solo con `--best-params`; precedencia:
CLI explícita > JSON validado > defaults. Main usa pérdida ponderada sin sampler;
baselines puede usar sampler con pérdida no ponderada. `--no-clahe` desactiva un valor
heredado. Ningún entrenamiento evalúa automáticamente test: se requiere `--evaluate-test`.

`predict --model ensemble` requiere `--ensemble-manifest`; un checkpoint o run explícito
inexistente falla sin seleccionar latest. Para modelos segmentados, inferencia sobre una
imagen original exige `--segmenter-checkpoint` exacto, mismo runtime y fallback. No enviar
una imagen original a un modelo segmentado omitiendo ese contrato.

Para exportar usar validación para paridad y después evaluar el archivo exportado completo
con el protocolo congelado. Sync exige paridad aprobada, caída macro-F1 ≤0.01, hashes,
clases, preprocesamiento y OOD coherentes (incluye paridad de features). `--without-ood`
es una capacidad explícita: crea paquete sin OOD; no retiene estadísticas viejas.
Esto no sustituye medir p50/p95 frío/caliente, memoria y tamaño en Android. Los objetivos
≤20 MB/≤300 ms no son resultados verificados y los paquetes segmentados requieren soporte
de segmentación móvil antes de habilitar su entrega.

Los dos checkpoints recibidos convertidos a TFLite FP32 pasaron paridad a 224×224
con 30 imágenes val y features con 8. ShuffleNet INT8 pasó; **B0 INT8 falló** por diferencia
máxima de probabilidad 0,264658 > 0,15 y permanece no entregable, sin relajar tolerancias.
Comandos realmente ejecutados en el venv LiteRT sobre las copias reconstruidas:

```bash
python -m scripts.pipeline.export --models efficientnet_b0 shufflenet_v2_x1_0 \
  --run r1 --output-dir outputs/repair-20260912/reconstructed \
  --formats tflite --parity-sample-size 30 --batch-size 8
# Misma invocación con --quantize int8: retorno 1 por paridad fallida de B0.
```

La evaluación posterior de conversión usa el test histórico ya expuesto y un
[protocolo de variantes congelado](evidence/2026-09-12-export-evaluation-protocol.json),
sin ajustar modelos por sus resultados. No confundirla con un holdout independiente nuevo.
Las tres evaluaciones completas terminaron; [resultados, tamaños y recálculo de entornos](2026-09-12-exportacion-real.md).

## Documentación y bloqueos de entorno

`npm ci && npm run docs:build` requiere Node ≥22.12; el build ya no ejecuta git fetch.
Con Node 18 falló por `node:util.styleText`; la instalación temporal de Node nuevo fue
rechazada por límite de cuenta y no se eludió. El aviso de npm de 2 vulnerabilidades
queda pendiente de resolución específica, no se ocultó con una actualización masiva.

El informe LaTeX conserva su PDF histórico: no hay latexmk/pdflatex disponibles.
Las fuentes corrigen porcentajes e interpretaciones. La regeneración de tablas/figuras
no es una nueva evaluación ni una recompilación del PDF; ver `docs/etapa_2/informe/MANIFEST.md`.
