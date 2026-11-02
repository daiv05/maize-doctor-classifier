# Entrenamiento del pipeline principal

El pipeline principal entrena sobre el 100 % de las muestras elegibles de la materialización vigente y produce, por cada corrida, un `best.pth` con la mejor época de validación y un `summary.json` v1 con hashes verificables del split, la configuración y el checkpoint. El formato, la carga segura y la migración de runs históricos están en [Contratos versionados de runs y artefactos](/es/pipeline/contratos-runs). Los resultados del modelo oficial están en [Resultados](/es/resultados/).

## Datos

Split `seed_42`, estratificado por `label + environment`; las fuentes se conservan como metadato y pueden aparecer en las tres particiones.

| Partición | Archivo | Muestras | Uso |
|---|---|---:|---|
| Entrenamiento | `train.csv` | 23 400 | Ajuste de pesos |
| Validación | `val.csv` | 5 014 | Scheduler, early stopping y selección del checkpoint |
| Prueba | `test.csv` | 5 015 | Evaluación única tras bloquear el checkpoint |
| Total | | 33 429 | Nueve clases |

El preprocesado y el balanceo están en [Preprocesado](/es/preprocessed/).

## Configuración del modelo oficial

EfficientNet-Lite0 (`timm`), pesos ImageNet, 224 × 224, corrida `20260921_204608`.

| Hiperparámetro | Valor |
|---|---|
| Optimizador | AdamW, lr 1e-4, weight decay 1e-4 |
| Scheduler | Coseno con 3 épocas de warmup, mínimo 1e-6 |
| Lote | 32 |
| Pérdida | Entropía cruzada con pesos de clase `sqrt_inverse` y label smoothing 0.1 |
| Gradient clipping | 1.0 |
| Early stopping | Paciencia 8, máximo 60 épocas, criterio Macro-F1 de validación |
| Sampler ponderado y CLAHE | Desactivados |

La corrida se detuvo en la época 47 con la mejor en la 39. La configuración de B0 y ShuffleNet-V2 de la etapa 2 usó lr 4.548e-4 y lote 64; transferirla a Lite0 empeoró su resultado (ver [Resultados](/es/resultados/)).

## Ejecución

```bash
make train-main MAIN_MODELS=efficientnet_lite0 MAIN_EPOCHS=60
make modal-train-main MAIN_MODELS=efficientnet_lite0
make tune-main N_TRIALS=20 MAIN_EPOCHS=30
make hpo-main
make cross-validate MODEL=efficientnet_lite0 K_FOLDS=5
make evaluate-ensemble
```

El entrenamiento puede correr en local o en una GPU de [Modal](/es/deployment/modal) con la misma CLI. La búsqueda de hiperparámetros formal está descrita en [HPO](/es/experimentos/hpo) y la comparación entre semillas en [multi-semilla](/es/experimentos/multiseed).

## Arquitecturas

| Modelo | Característica |
|---|---|
| `efficientnet_b0` | MBConv con Squeeze-and-Excitation; mayor capacidad |
| `shufflenet_v2_x1_0` | División y barajado de canales; el más pequeño |
| `efficientnet_lite0` | Sin Squeeze-and-Excitation ni swish, para cuantización a 8 bits; es la que despliega la app |

La descripción de cada una está en [Baselines](/es/baselines/). El [ensamble](/es/pipeline/ensamble) combina las tres por voto suave.
