# Comparación multi-seed: baseline frente a HPO trial 0

Estado al 24 de septiembre de 2026: implementación validada localmente,
con 0 de 10 entrenamientos iniciados. La ejecución queda pendiente hasta el
**1 de octubre de 2026**, fecha prevista para retomar el experimento.
La reanudación será manual, desde la rama `dev-abner2`, tras comprobar el protocolo
y los artefactos. No hay un lanzamiento automático programado.

## Objetivo e hipótesis

Determinar si la pequeña diferencia en validation entre el baseline principal
`20260921_204608` y el ganador del study `efficientnet_lite0_seed42_hpo_v1`
persiste al variar la semilla. Hipótesis: parte de esa diferencia podría ser
variabilidad del entrenamiento. No se realiza otra búsqueda de hiperparámetros.

Cinco seeds emparejadas: **42, 123, 2026, 3407, 7777**. Dos configuraciones,
**diez entrenamientos nuevos como máximo**, secuenciales, alternando baseline/HPO
por seed. Las runs históricas no se mezclan: el summary baseline no acredita
entorno/workers ni todos los diagnósticos validation necesarios, y el trial antiguo
se ejecutó mediante otro orquestador. No se da por probada equivalencia exacta.

## Auditoría previa y fuentes canónicas

La revisión del pipeline abarcó carga de datos, generación de lotes, entrenamiento
y escritura de contratos: `train.main → CornDataset/DataLoader → fit/run_epoch
→ build_run_contract`. También se contrastó el entrenamiento principal con
`TuningObjective._run_trial`, prestando atención a semillas, selección de
checkpoint e identidad de las muestras.

El baseline aquí es **el pipeline principal completo**, no el perfil reducido de
`train_baselines.py` con `WeightedRandomSampler` y tope por clase.

Fuentes: `hpo_baseline_summary.json`, `best_hyperparameters.json`,
`trials/trial_000/summary.json`, `preflight.json` y `split_hashes_after.json`,
bajo `docs/es/reproducibilidad/evidencia/`. El runner comprueba los valores
del protocolo contra el JSON canónico y obtiene de los summaries los parámetros
complementarios.

| Parámetro | Baseline | HPO trial 0 |
|---|---:|---:|
| Batch | 32 | 16 |
| Learning rate | 0.0001 | 0.00008468008575248323 |
| Weight decay | 0.0001 | 0.005669849511478858 |
| Label smoothing | 0.1 | 0.075 |
| Class weights | sqrt_inverse | none |
| Warmup | 3 | 1 |

Compartidos: EfficientNet-Lite0, nueve clases, 224×224, pretrained, AdamW,
cosine, máximo 60 épocas, patience 8, clipping 1.0, min_lr 1e-6,
CLAHE desactivado, sampler balanceado desactivado y sin tope por clase.
Mismo pipeline de transforms y mismo número de workers (**32**).

## Contratos conservados

`CornDataset` no fue modificado ni portado desde otra rama:

- `ImageCache` conserva acceso opcional a imágenes decodificadas; main no lo
  inyectaba antes y sigue sin inyectarlo. No se cambia su estrategia de rendimiento.
- `max_per_class` sigue aplicándose después de determinar minorías; aquí vale 0.
- Minorías derivadas de distribución real, umbral estricto `max_count/count > 4`:
  gray leaf spot, N, P y K. Se conserva su augmentation asimétrica.
- Mismo loader EXIF/RGB, transforms train/minority/val y normalización.
- `class_to_idx` canónico desde YAML, propagado desde train a validation.
- `sample_id` sigue siendo SHA-256 de la ruta relativa normalizada, no del contenido.
- Mismo batch `(image, label, sample_id)` y propagación por `run_epoch`.
- Mismo shuffle train, DataLoader, workers y pinning CUDA; sin sampler balanceado.
- Fallo de lectura: error trazable de esa misma muestra; sin saltos ni filtrado.

`fit` permanece intacto: mejor checkpoint por mejora estricta de validation
Macro-F1, scheduler y early stopping existentes. Ninguna métrica N/P/K interviene
en la selección. El contrato de run versión 1 continúa siendo validado.

## Qué cambia realmente al cambiar seed

| Componente | Efecto |
|---|---|
| Inicialización | `random`, NumPy y PyTorch; incluye la cabeza aleatoria sobre pesos pretrained |
| DataLoader/shuffle | RNG global PyTorch; orden de train y semilla base de workers por época |
| Augmentations | RNG de PyTorch; `worker_init_fn` propaga a Python/NumPy |
| Sampler | Sin `WeightedRandomSampler`; barajado normal de train |
| CUDA/cuDNN | `manual_seed_all`, deterministic=True, benchmark=False; no garantía bit a bit entre entornos |
| Split | **No cambia**: se leen CSV materializados, nunca se invoca `create_splits` |
| Submuestreo | Desactivado; no modifica corpus por seed |

## Integridad y exclusión de test

Hash del master: `64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163`.
Hash del lock: `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8`.
El plan guarda los cinco hashes completos en `PROTOCOL.json` y `PREFLIGHT.json`.

Verificados 23 400 train / 5 014 validation / 5 015 restantes reservadas,
unicidad y correspondencia de sample_id, labels/rutas frente al master y cero
solapamiento por identidad, SHA-256 de contenido registrado y grupo efectivo.
No se demuestra ausencia de casi duplicados: PHash sigue sin ejecutarse.
Es una auditoría de manifiestos congelados, no un nuevo escaneo de bytes de imágenes.

Durante el experimento `test.csv` se lee **solo como bytes para SHA-256**;
no se parsea para entrenar/evaluar. El master contiene metadatos del corpus y se
usa para comprobar el complemento reservado. No se cargan imágenes ni se crea
dataset, DataLoader o inferencia de test. `--skip-test` es explícito y rechaza
`--export`, cuya paridad usaría test. El flujo general sin el flag conserva test.

## Ejecución, artefactos y recuperación

Se reutiliza `scripts/pipeline/train.py` en subprocesos independientes para aislar
RNG/modelo. El runner está en `src/training/multiseed.py`; CLI/reportes en
`scripts/pipeline/multiseed.py`; entrada Modal en `scripts/modal/multiseed.py`.
Sin nuevas dependencias. Mismos Volumes `corn-clean`/`corn-outputs` e imagen Modal.

Directorio: `outputs/multiseed/efficientnet_lite0_baseline_vs_hpo/`.
Cada configuración contiene `seed_<seed>/efficientnet_lite0/<run_id>/` con
`summary.json`, `best.pth`, `last.pth`, `train_history.csv`, predicciones validation
con sample_id, diagnósticos por clase/NPK/ECE y metadata del experimento.
El protocolo captura configuración completa, fuentes y hashes, git SHA/dirty,
dataset fingerprint y clases. Cada run registra timestamp, entorno, GPU/CUDA,
Python/Torch/torchvision/timm y SHA-256 de checkpoint y demás artefactos.

`COMPLETE.json` permite saltar una run terminada solo después de verificar
contrato, hashes y métricas. Cambiar código/configuración/entorno bloquea la
continuación. Un solo lanzador Modal, un contenedor, un input y sin retries;
no lanzar dos apps independientes en paralelo. Hay bloqueo local del escritor.

Limitación real conservada: `fit` no guarda optimizador/scheduler/RNG completos.
Un `STARTED.json` sin recibo de finalización o una run fallida **requiere diagnóstico**;
no se reinicia automáticamente desde cero ni se exceden diez intentos. Los pesos
e historial permanecen intactos. La reanudación automática es entre runs completas,
no desde una época interrumpida.

## Reportes y estadística

`MULTISEED_SUMMARY.json`, `MULTISEED_RESULTS.csv` y `MULTISEED_REPORT.md`
incluyen valores por seed, media, mediana, SD **muestral**, min/max e IC t 95%
exploratorio cuando n≥2. Pendientes son null, nunca ceros. Métricas: validation
Macro-F1, accuracy, F1 individual N/P/K y ECE existente (15 bins).
Se informan deltas emparejados HPO−baseline, media/SD, victorias/derrotas/empates.

Matplotlib existente genera cinco figuras al haber resultados: Macro-F1 por
seed, distribución, deltas, N/P/K y ECE. No se crean curvas "representativas"
seleccionando retrospectivamente el mejor seed. Historiales completos disponibles.
La comprobación con datos sintéticos genera figuras únicamente en directorios pytest.

Cinco seeds no son cross-validation (cambia los pliegues), LOSO (retiene fuentes)
ni entrenamiento formal posterior. No prueban generalización entre dominios.
La selección HPO utilizó esta misma validation: multi-seed no elimina ese sesgo.
IC exploratorio frágil con n=5 y supuestos de independencia/normalidad entre seeds;
no se declara significancia ni se promueve un ganador automáticamente.

## Calendario y procedimiento de reanudación

La ejecución se retomará a partir del 1 de octubre de 2026. Durante la pausa se
conservan el protocolo, los manifiestos, los parámetros y los artefactos de origen.
La pausa no modifica las cinco semillas, las dos configuraciones ni las reglas
de evaluación.

Antes del lanzamiento:

1. Comprobar la rama y los cambios locales. Contrastar el código efectivo con
   `PROTOCOL.json` y verificar los archivos de `REPORT_HASHES.json`.
2. Repetir la validación local del plan con los mismos manifiestos. No regenerar
   splits ni actualizar el dataset durante esta fase.
3. Revisar los resultados existentes y confirmar que no haya otra ejecución
   activa. Una run completa se reutiliza solo si pasa las comprobaciones de
   contrato y hashes; un intento interrumpido requiere diagnóstico.
4. Lanzar un único proceso secuencial. Al punto de corte hay diez runs pendientes.
5. Descargar los artefactos, verificar su integridad y consolidar el reporte.
   El cierre requiere las diez runs válidas y los cinco pares completos.

El entorno previsto es A10G, 32 cores físicos, 32 GiB y 32 workers.
Como referencia temporal, el baseline previo completó 47 épocas en 2524.47 s
y el trial 0 del HPO completó 52 épocas en 2882.99 s. Cinco pares con duraciones
similares representarían unas 7.51 horas, sin contar inicialización ni validación
adicional. La duración real dependerá del early stopping de cada ejecución.

## Comandos

Desde la raíz, con Python y el CLI de Modal disponibles en sus respectivos
entornos. Las variables pueden apuntar a los ejecutables de cada instalación:

```bash
PY=python
MODAL=modal
EXP=outputs/multiseed/efficientnet_lite0_baseline_vs_hpo

# Plan e integridad locales; cero GPU, splits ya descargados como metadatos.
"$PY" -m scripts.pipeline.multiseed plan \
  --splits-dir outputs/multiseed-inputs/frozen/seed_42 --output-dir "$EXP"

# Comprobar que no haya otra ejecución activa.
"$MODAL" app list --json

# A partir del 2026-10-01, después de las comprobaciones anteriores.
# Hasta diez runs pendientes, secuenciales, exclusivamente validation.
"$MODAL" run --detach scripts/modal/multiseed.py \
  --protocol "$EXP/PROTOCOL.json" --execute --max-new-runs 10

# Recuperar artefactos en un destino nuevo/existente, sin --force.
mkdir -p outputs/multiseed-receipts
"$MODAL" volume get corn-outputs \
  /multiseed/efficientnet_lite0_baseline_vs_hpo outputs/multiseed-receipts

"$PY" -m scripts.pipeline.multiseed report \
  --output-dir outputs/multiseed-receipts/efficientnet_lite0_baseline_vs_hpo
```

El plan rechaza cambios de fuentes posteriores a su congelación. Si se hace un
commit o una corrección antes del primer lanzamiento, revisar la procedencia y
generar un plan nuevo explícitamente; no editar el protocolo a mano.

## Estado de resultados y verificación técnica

Entrenamientos del experimento completados/fallidos/iniciados: **0/0/0**. Ambas columnas
para las cinco semillas están pendientes. Media±SD, N/P/K, ECE y deltas no están
disponibles todavía. No hay conclusión experimental ni recomendación de promover
una configuración hasta completar la comparación.

Pruebas: smoke CPU sintético con/sin `--skip-test`, export incompatible, fuentes
canónicas, hashes/leakage, estadísticas, diez slots simulados sin duplicados,
interrupciones, recibos alterados, bloqueo y reporte incompleto. Regresiones de
dataset/cache/max_per_class/identidad, contratos, loop y HPO. Evidencia JUnit:
`outputs/multiseed-checks/tests.xml`.

Resultado: **146 tests aprobados**. Los 29 warnings corresponden a Optuna
multivariante experimental y pruebas mock de scheduler existentes; no son fallos
de runs. Ruff aprobado y módulo Modal importado localmente sin llamadas remotas.

Archivos nuevos: `src/training/multiseed.py`, `scripts/pipeline/multiseed.py`,
`scripts/modal/multiseed.py`, `tests/training/test_multiseed.py`.
Cambios funcionales: `scripts/pipeline/train.py` y `src/training/artifacts.py`.
Documentación actualizada: esta página, estado actual, HPO, decision log, backlog
y registro de evidencia. Se preservan los cambios previos y los artefactos HPO.

Para repetir la suite en el entorno del proyecto, con las dependencias de
desarrollo y HPO disponibles:

```bash
"$PY" -m pytest \
  tests/training tests/data/test_dataset_fail_fast.py tests/data/test_image_cache.py \
  tests/data/test_max_per_class.py tests/data/test_identity.py \
  tests/pipeline/test_best_params.py tests/pipeline/test_hpo_reporting.py \
  tests/pipeline/test_hpo_budget_amendment.py -q \
  --junitxml=outputs/multiseed-checks/tests.xml
```

Compilación VitePress aprobada con Node 24.16.0; `git diff --check` limpio.
El grafo de código está actualizado. Al último control no había aplicaciones
de entrenamiento activas.

Hashes SHA-256 de archivos entregados:

| Archivo | SHA-256 |
|---|---|
| PROTOCOL.json | `b654609f33bdbda65099c26d3b8475edd061d3e93a49fc2316c6fe4639ddd081` |
| PREFLIGHT.json | `347a19c5ada41cf99d615d8919303184659b34a15d1d347d4bc463688df58b64` |
| source_snapshot.zip | `0df770f8adb4b30f3f19b77c765805e8224e4641eadfdc8ba3db94aba103b1bc` |

Estos son hashes de archivos; el campo `protocol_sha256` de los recibos usa la
serialización JSON canónica del proyecto. Los hashes de reportes figuran en
`REPORT_HASHES.json` y cambiarán cuando haya resultados legítimos.
