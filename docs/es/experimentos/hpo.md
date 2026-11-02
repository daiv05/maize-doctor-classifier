# HPO — EfficientNet-Lite0

Estado al 2026-09-24: **completado**, con 25 intentos y test único del ganador verificados. Los estudios históricos de 15 y 25 trials con techo de 15 épocas son otro experimento y no cuentan como evidencia de esta fase.

El presupuesto original de 60 intentos se redujo a 25 durante la ejecución. `BUDGET_AMENDMENT.json` registra protocolos anterior y nuevo, hashes, trials al corte y RNG, y `budget_revisions/60-to-25/` conserva la base, el código y las evidencias anteriores. Los archivos del dataset, el modelo y la lógica del objetivo no cambiaron. [Enmienda](../reproducibilidad/evidencia/hpo_lite0_seed42/BUDGET_AMENDMENT.json).

Se reutilizan `TuningObjective`, `CornDataset`, transforms, `run_epoch`, pérdidas, scheduler y contratos de runs existentes. `tuning_study.py` administra la persistencia y el cierre del mismo Optuna. No se modificó `CornDataset`.

## Protocolo

| Campo | Valor formal |
|---|---|
| Study | `efficientnet_lite0_seed42_hpo_v1` |
| Modelo / input | `efficientnet_lite0` / 224×224 |
| Split / training seed / sampler seed | `seed_42` / 42 / 42 |
| Datos | train 23 400; validation 5 014; test 5 015 |
| Objetivo | mejor Macro-F1 de validación de cada trial COMPLETE |
| Presupuesto | 25 intentos entre COMPLETE, PRUNED y FAIL |
| Sampler | `TPESampler(seed=42, multivariate=True)`; estado RNG persistido |
| Pruner | `MedianPruner(n_startup_trials=5, n_warmup_steps=8, interval_steps=1)` |
| Épocas | máximo 60; `EarlyStopping(patience=8, min_delta=0)` |
| Fijos | AdamW, cosine, min_lr=1e-6, clip_grad_norm=1, pretrained=true |
| Balanceo | sampler desactivado; pérdida según `class_weights` muestreado |
| Persistencia | `<OUTPUT_ROOT>/hpo/efficientnet_lite0/<study>/study.db` |
| Modal | una GPU A10G, 32 CPU, 32 workers; un trial a la vez |
| Test durante la búsqueda | no se construye Dataset ni DataLoader; solo se verifica el hash y el inventario del CSV |

La comparación durante la búsqueda usa exclusivamente la validación Macro-F1 del baseline `20260921_204608`, 0.9560862657056215, cuyo `best.pth` se fija en el preflight.

### Espacio congelado antes de entrenar

| Parámetro | Espacio |
|---|---|
| learning_rate | `[1e-5, 3e-3]`, log |
| weight_decay | `[1e-7, 1e-2]`, log |
| label_smoothing | `{0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15}` |
| batch_size | `{16, 32, 64}` |
| class_weights | `{sqrt_inverse, inverse, none}` |
| warmup_epochs | enteros 1–5 |

CLAHE permanece desactivado. Dropout conserva el valor de fábrica porque el cargador contractual no reconstruye un dropout personalizado. Los pasos reportados al pruner son épocas de base 1; la primera poda posible es la época 8, tras cinco trials COMPLETE, y usa solo validación.

### Recuperación y ejecución

El presupuesto es total: una reanudación con 10 intentos registrados ejecuta como máximo los 15 restantes. Un trial interrumpido se conserva como FAIL con su ID y su traza, y no es candidato ganador. Tres fallos consecutivos detienen la ejecución para diagnóstico y un cambio de splits la aborta. El espacio no se reduce ante un OOM con batch 64.

Cada trial es `trial_000` … `trial_024`. El RNG de TPE se guarda en el mismo SQLite antes y después de cada trial. Solo deben cargarse bases propias: el estado del sampler usa pickle y requiere la misma versión de Optuna. La [documentación de Optuna](https://optuna.readthedocs.io/en/v3.5.1/tutorial/20_recipes/001_rdb.html) explica por qué el almacenamiento RDB no basta para restaurar el sampler.

Modal ejecuta dos trials por invocación y encadena la siguiente tras confirmar el Volume. El dataset se reutiliza desde `corn-clean` y los artefactos persisten en `corn-outputs`. No debe lanzarse otra app escritora contra el mismo estudio: SQLite sobre un Volume no es un diseño multiworker.

```bash
make modal-hpo-preflight MODAL=modal
make modal-hpo-smoke MODAL=modal
make modal-hpo MODAL=modal
```

El smoke usa un study separado con sufijo `_smoke_25`: dos trials de una época, cerrando y reabriendo SQLite entre ambos. No selecciona parámetros ni cuenta en los 25, y el estudio formal exige su marcador `SMOKE_COMPLETE.json`. Un `FINAL_TEST_STARTED.json` sin `FINAL_TEST_COMPLETE.json` exige auditoría y no dispara otra evaluación de test.

### Evidencia y política de selección

`preflight.json` fija los hashes de código, configuración y los cinco artefactos de splits, y `source_code.zip` conserva el código efectivo. Los hashes de referencia son:

| Archivo | SHA-256 |
|---|---|
| master_manifest.csv | `64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163` |
| train.csv | `231048178f3450bf84925f8d19eb5a672669ee9f2658a23fe87d88d6e9949434` |
| val.csv | `6f37710ebd797e470bc918fec6bf9502f0635e8220542336e88fe5c13b5ae6a5` |
| test.csv | `08c81aeec5a57e04416edf2c94f997c422bfcada357c6a3434e73aa9f771f724` |
| manifest.lock.json | `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` |

Cada trial conserva su historial, resumen, el checkpoint de su mejor época, predicciones y diagnósticos de validación; `failure.json` o `interruption.json` preservan los errores. Se exportan `trials.csv`, `top_10_trials`, `convergence.json`, `hyperparameter_importance.csv`, `comparison_vs_baseline.csv`, `study_summary.json` y figuras. Los gaps train–validación se miden con augmentation y en modo entrenamiento, por lo que son diagnósticos y no una estimación independiente del sobreajuste.

Al completar los 25, `best_hyperparameters.json` se valida con `load_best_params`. `HPO_SELECTION_LOCK.json` congela ganador, parámetros, split y checkpoint antes de crear el test, que se evalúa una sola vez sobre el checkpoint exacto de la mejor época. El ganador es el máximo de validación Macro-F1 entre los COMPLETE (empates: primer trial). Se reportan los mejores hiperparámetros dentro del espacio y presupuesto evaluados, sin interpretación causal.

## Ejecución

El smoke completó sus dos trials y verificó la reanudación ([marcador](../reproducibilidad/evidencia/hpo_smoke_complete.json)). Tras la enmienda de presupuesto y su [smoke de reapertura](../reproducibilidad/evidencia/hpo_smoke_complete_25.json), el estudio se reanudó con el protocolo revisado sobre el mismo study. El código original tiene SHA `d768e2ebd591cae0bd2260e086492f30572e687b29f86f0a9f27feb5ec1dbeac` y el de la revisión de presupuesto `1abbfe41c86d2f793da1ad4b569a845d23918df782b6435daef2f93bc0431e09`. Entorno: Python 3.11.12, torch 2.12.1+cu126, torchvision 0.27.1+cu126, timm 1.0.29, Optuna 4.9.0 y GPU NVIDIA A10G con CUDA 12.6.

`scripts/pipeline/report_hpo.py` valida el cierre: 25 IDs únicos y la enmienda, concordancia SQLite/CSV, ganador por validación, hashes de checkpoint y artefactos y orden temporal lock → test. Solo entonces genera el informe.

## Resultados

Study `efficientnet_lite0_seed42_hpo_v1`: **8 COMPLETE, 15 PRUNED y 2 FAIL** por interrupción. Ganador trial 0, época 44, con validación Macro-F1 0.957292 frente a 0.956086 del baseline (+0.12 pp). Test único posterior al lock: Macro-F1 **0.943125**, accuracy **0.975274** y ECE **0.060042**, frente a 0.948002 del baseline: el HPO no lo superó en test. No se infiere mejora de generalización ni se cambia la selección utilizando el test. [Informe completo, Top 10 y hashes](../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md) y.

| Métrica | Baseline `20260921_204608` | HPO trial 0 | Diferencia (pp) |
|---|---:|---:|---:|
| Macro-F1 de validación | 0.956086 | 0.957292 | +0.1206 |
| Macro-F1 de prueba | 0.948002 | 0.943125 | −0.4877 |
| Accuracy de prueba | 0.978066 | 0.975274 | −0.2792 |

En prueba, el ganador tiene F1 de 0.8949 en nitrógeno, 0.9363 en fósforo y 0.8041 en potasio (la clase más débil), y un ECE de 0.0600 sin temperature scaling. El ganador tiene una sola semilla, por lo que la diferencia en validación no demuestra mejor generalización y el checkpoint no reemplaza al oficial.

El estudio pareado de semillas ([multi-seed](./multiseed.md)) tiene 2 de 10 entrenamientos de validación.
