# Optimización de hiperparámetros (HPO)

Búsqueda de hiperparámetros con Optuna para EfficientNet-Lite0 sobre el split `seed_42`. Ejecutó 25 intentos y evaluó una sola vez el test del ganador. Los estudios anteriores de 15 y 25 trials con techo de 15 épocas son otro experimento.

El presupuesto original era de 60 intentos y se redujo a 25 durante la ejecución. [`budget_amendment.json`](../reproducibilidad/evidencia/hpo_lite0_seed42/budget_amendment.json) registra ambos protocolos, sus hashes, los trials al corte y el estado del RNG; `budget_revisions/60-to-25/` conserva la base, el código y la evidencia anteriores. El dataset, el modelo y el objetivo no cambiaron.

El estudio reutiliza `TuningObjective`, `CornDataset`, los transforms, `run_epoch`, las pérdidas, el scheduler y los contratos de runs. `tuning_study.py` administra la persistencia y el cierre del estudio.

## Protocolo

| Campo | Valor |
|---|---|
| Estudio | `efficientnet_lite0_seed42_hpo_v1` |
| Modelo / entrada | `efficientnet_lite0` / 224×224 |
| Split / semilla de entrenamiento / semilla del sampler | `seed_42` / 42 / 42 |
| Datos | train 23 400, validación 5 014, test 5 015 |
| Objetivo | Macro-F1 de validación de cada trial COMPLETE |
| Presupuesto | 25 intentos entre COMPLETE, PRUNED y FAIL |
| Sampler | `TPESampler(seed=42, multivariate=True)`, con el estado del RNG guardado |
| Pruner | `MedianPruner(n_startup_trials=5, n_warmup_steps=8, interval_steps=1)` |
| Épocas | máximo 60; `EarlyStopping(patience=8, min_delta=0)` |
| Fijos | AdamW, cosine, min_lr=1e-6, clip_grad_norm=1, pretrained=true |
| Balanceo | sampler desactivado; pérdida según el `class_weights` muestreado |
| Persistencia | `<OUTPUT_ROOT>/hpo/efficientnet_lite0/<study>/study.db` |
| Cómputo | una GPU A10G, 32 CPU, 32 workers; un trial a la vez |
| Test durante la búsqueda | no se construye Dataset ni DataLoader; solo se verifica el hash y el inventario del CSV |

La comparación durante la búsqueda usa la Macro-F1 de validación del baseline `20260921_204608` (0.9560862657056215).

### Espacio de búsqueda

| Parámetro | Espacio |
|---|---|
| learning_rate | `[1e-5, 3e-3]`, log |
| weight_decay | `[1e-7, 1e-2]`, log |
| label_smoothing | `{0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15}` |
| batch_size | `{16, 32, 64}` |
| class_weights | `{sqrt_inverse, inverse, none}` |
| warmup_epochs | enteros 1–5 |

CLAHE permanece desactivado y el dropout conserva el valor de fábrica. Los pasos que se reportan al pruner son épocas de base 1; la primera poda posible es la época 8, tras cinco trials COMPLETE, y usa solo validación.

### Ejecución y recuperación

El presupuesto es total: una reanudación con 10 intentos registrados ejecuta como máximo los 15 restantes. Un trial interrumpido queda como FAIL con su ID y su traza y no puede ser el ganador. Tres fallos consecutivos detienen la ejecución y un cambio de splits la aborta.

Cada trial se guarda como `trial_000` … `trial_024`. El estado del RNG de TPE se guarda en el mismo SQLite antes y después de cada trial. Ese estado usa pickle: solo deben cargarse bases propias y con la misma versión de Optuna. La [documentación de Optuna](https://optuna.readthedocs.io/en/v3.5.1/tutorial/20_recipes/001_rdb.html) explica por qué el almacenamiento RDB no basta para restaurar el sampler.

En Modal cada invocación ejecuta dos trials y encadena la siguiente tras confirmar el volumen. No debe lanzarse otra app contra el mismo estudio: SQLite sobre un volumen no admite varios escritores.

```bash
make modal-hpo-preflight MODAL=modal
make modal-hpo-smoke MODAL=modal
make modal-hpo MODAL=modal
```

El smoke usa un estudio aparte (sufijo `_smoke_25`) con dos trials de una época, cerrando y reabriendo SQLite entre ambos. No selecciona parámetros ni cuenta entre los 25, y el estudio formal exige su marcador `SMOKE_COMPLETE.json`. Un `final_test_started.json` sin `final_test_complete.json` exige una auditoría manual y no dispara otra evaluación de test.

### Evidencia y selección

`preflight.json` fija los hashes del código, la configuración y los cinco artefactos de splits; `source_code.zip` conserva el código efectivo.

| Archivo | SHA-256 |
|---|---|
| master_manifest.csv | `64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163` |
| train.csv | `231048178f3450bf84925f8d19eb5a672669ee9f2658a23fe87d88d6e9949434` |
| val.csv | `6f37710ebd797e470bc918fec6bf9502f0635e8220542336e88fe5c13b5ae6a5` |
| test.csv | `08c81aeec5a57e04416edf2c94f997c422bfcada357c6a3434e73aa9f771f724` |
| manifest.lock.json | `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` |

Cada trial conserva su historial, su resumen, el checkpoint de su mejor época, las predicciones y los diagnósticos de validación; `failure.json` o `interruption.json` guardan los errores. El estudio exporta `trials.csv`, `top_10_trials`, `convergence.json`, `hyperparameter_importance.csv`, `comparison_vs_baseline.csv`, `study_summary.json` y figuras. Los gaps entre train y validación se miden con augmentation y en modo entrenamiento, por lo que son diagnósticos y no una estimación independiente del sobreajuste.

Al completar los 25 intentos, `best_hyperparameters.json` se valida con `load_best_params`. `hpo_selection_lock.json` congela el ganador, los parámetros, el split y el checkpoint antes de crear el test, que se evalúa una sola vez sobre el checkpoint de la mejor época. El ganador es el máximo de Macro-F1 de validación entre los COMPLETE (empates: el primer trial).

`scripts/pipeline/report_hpo.py` valida el cierre (25 IDs únicos y la enmienda, concordancia SQLite/CSV, ganador por validación, hashes de checkpoint y artefactos, orden lock → test) y entonces genera el informe.

Entorno: Python 3.11.12, torch 2.12.1+cu126, torchvision 0.27.1+cu126, timm 1.0.29, Optuna 4.9.0 y una GPU NVIDIA A10G con CUDA 12.6. El código original tiene SHA-256 `d768e2ebd591cae0bd2260e086492f30572e687b29f86f0a9f27feb5ec1dbeac` y el de la revisión de presupuesto `1abbfe41c86d2f793da1ad4b569a845d23918df782b6435daef2f93bc0431e09`. Los smokes están en [`hpo_smoke_complete.json`](../reproducibilidad/evidencia/hpo_smoke_complete.json) y [`hpo_smoke_complete_25.json`](../reproducibilidad/evidencia/hpo_smoke_complete_25.json).

## Resultados

El estudio terminó con 8 trials COMPLETE, 15 PRUNED y 2 FAIL por interrupción. El ganador es el trial 0 (época 44), con Macro-F1 de validación 0.957292 frente a 0.956086 del baseline (+0.12 pp). En el test único posterior al lock obtuvo Macro-F1 0.943125, accuracy 0.975274 y ECE 0.060042, frente a 0.948002 del baseline: no lo supera. Informe completo, Top 10 y hashes en [hpo_report.md](../reproducibilidad/evidencia/hpo_lite0_seed42/hpo_report.md).

| Métrica | Baseline `20260921_204608` | HPO trial 0 | Diferencia (pp) |
|---|---:|---:|---:|
| Macro-F1 de validación | 0.956086 | 0.957292 | +0.1206 |
| Macro-F1 de prueba | 0.948002 | 0.943125 | −0.4877 |
| Accuracy de prueba | 0.978066 | 0.975274 | −0.2792 |

En prueba el ganador tiene F1 de 0.8949 en nitrógeno, 0.9363 en fósforo y 0.8041 en potasio (la clase más débil), y un ECE de 0.0600 sin temperature scaling. Al ser una sola semilla, la diferencia en validación no indica mejor generalización y el checkpoint no reemplaza al oficial. El estudio pareado de semillas está en [multi-seed](./multiseed.md).
