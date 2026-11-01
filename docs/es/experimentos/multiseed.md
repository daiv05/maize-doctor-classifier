# Comparación multi-seed: baseline frente a HPO trial 0

Hay dos estudios multi-seed distintos. El de [estabilidad baseline/LOSO](../reproducibilidad/evidencia/multiseed_source_stability/MULTISEED_SOURCE_STABILITY.md) está cerrado con tres semillas (42, 123 y 2026). Esta página describe el segundo: la comparación pareada del baseline frente al ganador del HPO, que tiene **2 de 10 entrenamientos completados**, todos de validación, ejecutados en Modal el 2026-10-19.

## Estado

| Configuración | Seed | Run | Validation Macro-F1 | ECE | Mejor época |
|---|---:|---|---:|---:|---:|
| baseline | 42 | `20261019_203033` | 0.956086 | 0.1398 | 39 |
| HPO trial 0 | 42 | `20261019_211317` | 0.957292 | 0.0579 | 44 |

El `best.pth` del baseline (SHA-256 `860180f3…`) es idéntico al del run oficial `20260921_204608`, de modo que el entrenamiento se reproduce bit a bit. La diferencia de +0.0012 en validación entre ambas configuraciones no es interpretable con una semilla (McNemar p = 0.83). Las ocho corridas restantes (seeds 123, 2026, 3407 y 7777 para cada configuración) no se ejecutaron, por lo que el estudio no permite concluir nada sobre la variabilidad del HPO. Ninguna corrida usó test.

Los artefactos versionados están en [`multiseed_baseline_vs_hpo/`](../reproducibilidad/evidencia/multiseed_baseline_vs_hpo/MULTISEED_SUMMARY.json): resultados, protocolo, preflight, hashes de reportes, los dos recibos `COMPLETE` y las sumas SHA-256 de los paquetes de cada run.

## Objetivo e hipótesis

Determinar si la pequeña diferencia en validación entre el baseline `20260921_204608` y el ganador del study `efficientnet_lite0_seed42_hpo_v1` persiste al variar la semilla. Hipótesis: parte de esa diferencia podría ser variabilidad del entrenamiento. No se realiza otra búsqueda de hiperparámetros.

Cinco seeds emparejadas (42, 123, 2026, 3407, 7777), dos configuraciones y diez entrenamientos nuevos como máximo, secuenciales y alternando baseline/HPO por seed. Las corridas históricas no se mezclan.

| Parámetro | Baseline | HPO trial 0 |
|---|---:|---:|
| Batch | 32 | 16 |
| Learning rate | 0.0001 | 0.00008468008575248323 |
| Weight decay | 0.0001 | 0.005669849511478858 |
| Label smoothing | 0.1 | 0.075 |
| Class weights | sqrt_inverse | none |
| Warmup | 3 | 1 |

Compartidos: EfficientNet-Lite0, nueve clases, 224×224, preentrenado, AdamW, cosine, máximo 60 épocas, patience 8, clipping 1.0, min_lr 1e-6, CLAHE desactivado, sampler balanceado desactivado y sin tope por clase. El baseline es el pipeline principal completo, no el perfil reducido de `train_baselines.py`. Fuentes canónicas: `hpo_baseline_summary.json`, `best_hyperparameters.json`, `trials/trial_000/summary.json`, `preflight.json` y `split_hashes_after.json`, bajo `docs/es/reproducibilidad/evidencia/`.

## Qué cambia al cambiar la seed

| Componente | Efecto |
|---|---|
| Inicialización | `random`, NumPy y PyTorch; incluye la cabeza aleatoria sobre pesos preentrenados |
| DataLoader y shuffle | RNG global de PyTorch; orden de train y semilla base de workers por época |
| Augmentations | RNG de PyTorch; `worker_init_fn` propaga a Python y NumPy |
| CUDA/cuDNN | `manual_seed_all`, deterministic=True, benchmark=False; sin garantía bit a bit entre entornos |
| Split | No cambia: se leen los CSV materializados |

## Integridad y exclusión de test

Hash del master `64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163` y del lock `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8`. Se verificaron 23 400 train y 5 014 validation, unicidad y correspondencia de `sample_id` y cero solapamiento por identidad, SHA-256 de contenido y grupo efectivo. PHash no se ejecutó.

`test.csv` se lee solo como bytes para su SHA-256; no se parsea ni se hace inferencia. `--skip-test` es explícito y rechaza `--export`.

## Ejecución y recuperación

Se reutiliza `scripts/pipeline/train.py` en subprocesos independientes. El runner está en `src/training/multiseed.py`, la CLI en `scripts/pipeline/multiseed.py` y la entrada Modal en `scripts/modal/multiseed.py`. Cada run escribe `summary.json`, `best.pth`, `last.pth`, `train_history.csv`, predicciones de validación con `sample_id`, diagnósticos por clase, N/P/K y ECE, y metadatos del experimento. `COMPLETE.json` permite saltar una run terminada solo tras verificar contrato, hashes y métricas. `fit` no guarda optimizador, scheduler ni RNG completos: una run interrumpida requiere diagnóstico y no se reanuda desde una época intermedia.

## Reportes y estadística

`MULTISEED_SUMMARY.json`, `MULTISEED_RESULTS.csv` y `MULTISEED_REPORT.md` incluyen valores por seed, media, mediana, desviación estándar muestral, mínimo y máximo, e IC t del 95 % exploratorio cuando n ≥ 2; los pendientes son `null`. Cinco seeds no son validación cruzada ni LOSO, no prueban generalización entre dominios, y la selección del HPO usó esta misma validación, sesgo que multi-seed no elimina.

## Comandos

```bash
PY=python
EXP=outputs/multiseed/efficientnet_lite0_baseline_vs_hpo

"$PY" -m scripts.pipeline.multiseed plan --splits-dir outputs/multiseed-inputs/frozen/seed_42 --output-dir "$EXP"

modal run --detach scripts/modal/multiseed.py --protocol "$EXP/PROTOCOL.json" --execute --max-new-runs 10

mkdir -p outputs/multiseed-receipts
modal volume get corn-outputs /multiseed/efficientnet_lite0_baseline_vs_hpo outputs/multiseed-receipts

"$PY" -m scripts.pipeline.multiseed report --output-dir outputs/multiseed-receipts/efficientnet_lite0_baseline_vs_hpo
```

El plan rechaza cambios de fuentes posteriores a su congelación; ante un cambio, generar un plan nuevo y no editar el protocolo a mano.

## Pruebas

La suite de este módulo cubre un smoke CPU sintético con y sin `--skip-test`, fuentes canónicas, hashes y fuga, estadísticas, diez slots sin duplicados, interrupciones, recibos alterados, bloqueo y reporte incompleto, además de las regresiones de dataset, contratos, loop y HPO.

```bash
python -m pytest tests/training tests/data/test_dataset_fail_fast.py tests/data/test_image_cache.py tests/data/test_max_per_class.py tests/data/test_identity.py tests/pipeline/test_best_params.py tests/pipeline/test_hpo_reporting.py tests/pipeline/test_hpo_budget_amendment.py -q
```

| Archivo | SHA-256 |
|---|---|
| PROTOCOL.json | `b654609f33bdbda65099c26d3b8475edd061d3e93a49fc2316c6fe4639ddd081` |
| PREFLIGHT.json | `347a19c5ada41cf99d615d8919303184659b34a15d1d347d4bc463688df58b64` |
| REPORT_HASHES.json | `1319445679b55ffc3a0d3544bf4c9a75980508b87190882f684faade31263473` |
