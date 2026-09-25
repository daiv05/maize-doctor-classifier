# HPO — EfficientNet-Lite0

## Protocolo formal implementado (2026-09-22)

**Estado al 2026-09-24: COMPLETADO; 25 intentos y test único del ganador verificados.**
Los estudios históricos de 15/25 trials no cuentan como evidencia de esta fase.

### Enmienda autorizada del presupuesto — 2026-09-23

El número máximo de intentos se redujo de 60 a **25 en total**.
No son 25 adicionales. Se conservan study, trials, estado RNG de TPE,
splits, espacio de búsqueda, objetivo, 60 épocas máximas y evaluación única del ganador.
La decisión se tomó con resultados parciales de validation ya observados; debe
presentarse como una enmienda posterior al inicio, no como un presupuesto prefijado.

Se detuvo la app anterior y se respaldó SQLite: trials 0 y 1 completos; trial 2
interrumpido, recuperado como FAIL sin reutilizar su ID. El siguiente es trial 3.
`BUDGET_AMENDMENT.json` registra autorización, protocolos anterior/nuevo, hashes,
trials al corte y RNG. `budget_revisions/60-to-25/` conserva DB, código y evidencias
anteriores. Los archivos del dataset, modelo y lógica del objetivo no cambiaron.
[Enmienda](../reproducibilidad/evidencia/hpo_budget_amendment_25.json).

La app antigua `ap-g5ZJ44pDZDVKYXzeGUtHVw` quedó detenida; no debe relanzarse con su
código de 60 intentos. Para mover el estudio a otro workspace autorizado, consultar
[continuidad y respaldo](../reproducibilidad/hpo-continuidad.md).

Se reutilizan `TuningObjective`, `CornDataset`, transforms, `run_epoch`, pérdidas,
scheduler y contratos de runs existentes. `tuning_study.py` administra persistencia
y cierre del mismo Optuna. No se modificó `CornDataset`.

| Campo | Valor formal |
|---|---|
| Study | `efficientnet_lite0_seed42_hpo_v1` |
| Modelo / input | `efficientnet_lite0` / 224×224 |
| Split / training seed / sampler seed | `seed_42` / 42 / 42 |
| Datos | train 23 400; validation 5 014; test 5 015 |
| Objetivo | **mejor Macro-F1 de validation** de cada trial COMPLETE |
| Presupuesto | exactamente 25 intentos COMPLETE + PRUNED + FAIL |
| Sampler | `TPESampler(seed=42, multivariate=True)`; estado RNG persistido |
| Pruner | `MedianPruner(n_startup_trials=5, n_warmup_steps=8, interval_steps=1)` |
| Épocas | máximo 60; `EarlyStopping(patience=8, min_delta=0)` |
| Fijos | AdamW, cosine, min_lr=1e-6, clip_grad_norm=1, pretrained=true |
| Balanceo | sampler desactivado; pérdida según `class_weights` muestreado |
| Augmentation / clases | implementación vigente; sin modificación |
| Persistencia | `<OUTPUT_ROOT>/hpo/efficientnet_lite0/<study>/study.db` |
| Modal | una A10, 32 CPU, 32 DataLoader workers; un trial a la vez |
| Test durante búsqueda | no se construye Dataset/DataLoader; solo se verifica hash e inventario del CSV |

Se preserva `main/efficientnet_lite0/20260921_204608/best.pth`, cuyo SHA se fija en
preflight y se comprueba al terminar cada invocación. La comparación durante búsqueda
usa exclusivamente su **validation Macro-F1 = 0.9560862657056215**.

### Espacio congelado antes de entrenamiento

| Parámetro | Espacio |
|---|---|
| learning_rate | `[1e-5, 3e-3]`, log |
| weight_decay | `[1e-7, 1e-2]`, log |
| label_smoothing | `{0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15}` |
| batch_size | `{16, 32, 64}` |
| class_weights | `{sqrt_inverse, inverse, none}` |
| warmup_epochs | enteros 1–5 |

CLAHE permanece desactivado. Dropout conserva el valor de fábrica: aunque timm admite
argumentos adicionales, el cargador contractual actual no reconstruye un dropout
personalizado; se excluye de esta fase. Cosine es el scheduler del baseline. Se
conservan las dos dimensiones existentes `class_weights` y `warmup_epochs`.

Los steps reportados al pruner son épocas **1-based**. La primera comprobación de
poda posible es la época 8, después de reunir cinco trials COMPLETE; la decisión
usa únicamente validation. Esto da margen tras los warmups de hasta cinco épocas.

### Recuperación y ejecución

El presupuesto es total: una reanudación con 10 intentos registrados ejecuta como
máximo los 15 restantes. Un trial interrumpido se conserva como FAIL con su ID y
traza de recuperación. Los fallos no son candidatos ganadores. Tres fallos
consecutivos detienen la ejecución para diagnóstico; un cambio de splits aborta
inmediatamente. El espacio no se reduce silenciosamente ante OOM de batch 64.

Cada trial tiene `trial_000` … `trial_024`. Se guarda el RNG de TPE en el mismo SQLite
antes de entrenar y después de terminar el trial. Solo deben cargarse bases propias:
el estado del sampler usa pickle y requiere la misma versión de Optuna. La
[documentación de Optuna](https://optuna.readthedocs.io/en/v3.5.1/tutorial/20_recipes/001_rdb.html)
explica por qué el almacenamiento RDB no basta para restaurar el sampler.

Modal ejecuta dos trials por invocación y encadena la siguiente después de confirmar
el Volume, hasta 25. El límite por invocación es 24 h, conforme a los
[límites de Modal](https://modal.com/docs/guide/functions). El dataset se reutiliza
desde `corn-clean`; los artefactos persisten en `corn-outputs`. No lanzar otra app
escritora simultánea contra el mismo estudio: SQLite sobre Volume no es un diseño
multiworker. La función limita sus contenedores a uno y la CLI usa bloqueo de proceso.

```bash
MODAL=modal  # CLI disponible en el entorno de ejecución
make modal-hpo-preflight MODAL="$MODAL"
make modal-hpo-smoke MODAL="$MODAL"
make modal-hpo MODAL="$MODAL"
```

El smoke revisado tiene study separado con sufijo `_smoke_25`: dos trials de una época, cerrando
y reabriendo SQLite entre ambos. No selecciona parámetros ni cuenta en los 25. El
formal exige su marcador `SMOKE_COMPLETE.json` del mismo código/split.

El último comando también reanuda el estudio existente. Un `FINAL_TEST_STARTED.json`
sin `FINAL_TEST_COMPLETE.json` exige auditoría; no dispara otra evaluación de test.

### Evidencia y política de selección

`preflight.json` fija hashes de código, configuración y los cinco artefactos de
splits. `source_code.zip` conserva el código efectivo sin necesidad de commit.
`source_archive.json` identifica el ZIP por SHA-256. Los hashes de referencia son:

| Archivo | SHA-256 |
|---|---|
| master_manifest.csv | `64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163` |
| train.csv | `231048178f3450bf84925f8d19eb5a672669ee9f2658a23fe87d88d6e9949434` |
| val.csv | `6f37710ebd797e470bc918fec6bf9502f0635e8220542336e88fe5c13b5ae6a5` |
| test.csv | `08c81aeec5a57e04416edf2c94f997c422bfcada357c6a3434e73aa9f771f724` |
| manifest.lock.json | `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` |

Cada trial conserva `training_history.csv`, `trial_summary.json`, el checkpoint de
su mejor época, `summary.json` contractual cuando hay checkpoint, predicciones y
diagnósticos de validation. `failure.json` o `interruption.json` preservan errores.

Se exportan `trials.csv`, `top_10_trials.csv/json`, `convergence.json`,
`hyperparameter_importance.csv`, `comparison_vs_baseline.csv`, `study_summary.json`
y figuras en `figures/`. Top 10 incluye calibración y F1 N/P/K de validation. El gap
`train_val_gap` usa train en la época de mejor validation; `best_train_val_gap`
usa el máximo train observado. Train se mide con augmentation y en modo training,
por lo que esos gaps son diagnósticos y no una estimación independiente del sobreajuste.

Al completar 25, `best_hyperparameters.json` se valida con `load_best_params` y es
compatible con `--best-params`. `HPO_SELECTION_LOCK.json` congela ganador, parámetros,
split y checkpoint **antes de crear test**. Se evalúa una sola vez el checkpoint
exacto de la mejor época, sin reentrenarlo. `final_test/` contiene clasificación,
confusión, predicciones, calibration, source/environment y N/P/K. Los marcadores
de inicio/cierre identifican la evaluación y hashes de sus artefactos.

El ganador es el máximo validation Macro-F1 entre COMPLETE (empates: primer trial).
Los gráficos de importancia describen este estudio, sin interpretación causal.
Se reportarán los **mejores hiperparámetros encontrados dentro del espacio y
presupuesto evaluados**. El HPO winner no equivale al futuro formal tuned model.

### Validación técnica

133 pruebas pasaron: training, regresiones de caché/tope/identidad/fail-fast de
CornDataset, contratos y mejores parámetros. Incluyen secuencia TPE tras resume,
presupuesto total, IDs, fallos, objetivo de mejor validation y test único tras lock.
También cubren el cierre documental idempotente, rechazo de evidencia incompleta
o alterada y observación sin relanzar entrenamiento.
El preflight inicial detectó que Modal no copiaba `pyproject.toml`; se añadió el
archivo para registrar el código. El fallo ocurrió antes de crear trials.

### Resultados formales

El smoke completó sus dos trials y verificó resume: [marcador](../reproducibilidad/evidencia/hpo_smoke_complete.json).
El formal empezó el 2026-09-23 a las 13:25 UTC en
[Modal](https://modal.com/apps/abner-rivas/main/ap-g5ZJ44pDZDVKYXzeGUtHVw).
Tras la enmienda y el [smoke de reapertura](../reproducibilidad/evidencia/hpo_smoke_complete_25.json),
se reanudó a las 20:16 UTC en la
[app de 25 intentos](https://modal.com/apps/abner-rivas/main/ap-bqlA0QXYYvcLGAFCCjIK8W).
El preflight remoto aceptó el protocolo revisado y abrió el mismo study existente.
El código original tenía SHA `d768e2ebd591cae0bd2260e086492f30572e687b29f86f0a9f27feb5ec1dbeac`.
La revisión de presupuesto tiene SHA `1abbfe41c86d2f793da1ad4b569a845d23918df782b6435daef2f93bc0431e09`.
Versiones reales: Python 3.11.12, torch 2.12.1+cu126, torchvision 0.27.1+cu126,
timm 1.0.29 y Optuna 4.9.0. GPU NVIDIA A10, CUDA 12.6.

`scripts/pipeline/watch_hpo.py` observa la app sin modificar Modal ni relanzar trials.
Cuando encuentra `FINAL_TEST_COMPLETE.json`, descarga el estudio a un directorio
único bajo `outputs/hpo-receipts/` y ejecuta `report_hpo.py --update-docs`. El
generador exige 25 IDs únicos y la enmienda, concordancia SQLite/CSV, ganador por validation,
hashes de checkpoint y artefactos, orden temporal lock → test y código archivado.
Solo entonces añade resultados, figuras y evidencia a los documentos de tesis.

El observador local requiere el equipo activo y acceso a Modal; si se suspende,
la GPU remota continúa y la recogida espera. Su estado está en
`outputs/hpo-monitor/efficientnet_lite0_seed42_hpo_v1/status.json`. Si la app se
detiene sin cierre final, registra `needs_audit` y termina. No cambia el protocolo.
Tras la interrupción se reanudó en `ap-uA4LWIV3qLV2Wclaa1M9zv`, con el
observador `doctor-maiz-hpo-watch-free.service`. Este registró `state: complete`
el 2026-09-24 a las 02:13:25 UTC, después de descargar y validar la evidencia.
Ya no hay una run que reanudar: búsqueda y test final están cerrados.

El presupuesto revisado de 25 intentos y el test único ya están verificados; véase «Resultados completados» al final de esta página. Multi-seed, entrenamiento formal, CV, LOSO,
ensemble y temperature scaling continúan pendientes.

## Protocolo anterior conservado como historial

<details>
<summary>Plan previo a implementar el protocolo formal (no usar estos comandos)</summary>

## Protocolo acordado

| Campo | Valor |
|---|---|
| Framework | Optuna |
| Modelo | `efficientnet_lite0` |
| Objetivo | maximizar Macro-F1 de validación |
| Split | `seed_42` |
| Seed del estudio | 42 (`TPESampler`) |
| Trials planificados | 60 |
| Test | cerrado; no se usa para búsqueda ni poda |
| Persistencia | SQLite, `<OUTPUT_ROOT>/tuning/optuna_study.db` |
| Nombre | prefijo configurable + `_efficientnet_lite0` |
| Pruner inicial | `MedianPruner`, salvo decisión registrada antes del run |

## Espacio implementado hoy

Los rangos siguientes provienen de `HyperparameterSpace` y `TuningObjective`; no son una propuesta ficticia.

| Parámetro | Espacio actual | Muestreo |
|---|---|---|
| `learning_rate` | `[1e-5, 1e-3]` | log-uniforme |
| `weight_decay` | `[1e-5, 1e-2]` | log-uniforme |
| `batch_size` | `{16, 32, 64}` | categórico |
| `class_weights` | `{sqrt_inverse, inverse, none}` | categórico |
| `label_smoothing` | `[0.0, 0.15]` | continuo |
| `warmup_epochs` | enteros `[1, 5]` | discreto |
| `clahe` | `false` por defecto; opt-in con `--search-clahe` | categórico solo si se habilita |

El scheduler es coseno fijo y no forma parte del espacio. `dropout` tampoco está expuesto por la implementación de tuning; no debe aparecer como parámetro buscado hasta que el código lo soporte. Cada trial usa `seed_dataset + trial.number`, early stopping y reporte por época al pruner.

## Comando planificado

Validado contra `make help` y el parser actual; no ejecutado:

```bash
make tune-main MAIN_MODELS=efficientnet_lite0 N_TRIALS=60 \
  MAIN_EPOCHS=30 PRUNER=median BASELINE_F1=0.9560862657056215
```

Antes de lanzarlo se debe fijar y registrar: SHA del código, hash del lock `seed_42`, versión de dependencias, GPU, timeout, nombre de estudio, decisión sobre CLAHE y techo de épocas.

## Salidas obligatorias

- base SQLite del estudio;
- CSV de trials con estados;
- `best_params.json` con estudio, best trial y baseline;
- historial e importancia de parámetros;
- conteos completos/podados/fallidos;
- comparación del mejor trial solo en validación.

El test seguirá sin abrirse. La evaluación de test ocurre en el [entrenamiento formal](./formal-training.md).

</details>

<!-- hpo-lite0-seed42-completed -->

## Resultados completados — 2026-09-24

Study `efficientnet_lite0_seed42_hpo_v1`: 25 intentos, ganador trial 0, validation Macro-F1 0.957292225; baseline 0.956086266; delta +0.120596 pp.

Test único posterior al lock. [Entrega completa, Top 10 y hashes](../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md).

**8 COMPLETE, 15 PRUNED y 2 FAIL por interrupción.** Ganador trial 0, época 44.
Test: Macro-F1 **0.943125073**, accuracy **0.975274177**, ECE **0.060041622**.
El baseline obtuvo Macro-F1 test 0.948002144: **el HPO no lo superó en test**.
No se infiere mejora de generalización ni se cambia la selección utilizando test.
[Comparación completa, N/P/K y auditoría](../tesis/HPO_BASELINE_COMPARISON.md).

Siguiente fase implementada: [comparación multi-seed baseline/HPO](./multiseed.md),
sin nuevo HPO ni inferencia de test. Diez runs pendientes; reanudación manual
prevista para el 1 de octubre de 2026.
