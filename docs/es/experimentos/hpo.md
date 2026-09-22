# Plan HPO — EfficientNet-Lite0

**Estado: PENDIENTE. No se han ejecutado los 60 trials.** Los archivos Optuna existentes corresponden a estudios históricos de 15 y 25 trials y no son el resultado de esta fase.

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
