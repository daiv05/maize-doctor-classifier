# Plantilla de entrenamiento formal

**Estado: PENDIENTE.** El HPO formal de 25 intentos ya terminó; no repetir
`make tune-main` como requisito de esta fase. Antes de entrenar, registrar en el
decision log la configuración elegida usando la
[regla de selección fijada](./seleccion-modelo-final.md) y ejecutar
[`make preflight-data`](../../../LOCAL.md) sobre la
materialización congelada. Completar esta ficha únicamente desde artefactos
del nuevo run.

## Identidad

| Campo | Valor |
|---|---|
| Configuración elegida y justificación | `PENDIENTE — baseline o HPO, según decisión registrada` |
| Modelo | EfficientNet-Lite0 para la comparación baseline/HPO; run formal pendiente |
| Commit | `PENDIENTE — git rev-parse HEAD` |
| Split lock SHA-256 | `PENDIENTE — manifest.lock.json` |
| Config efectiva | `PENDIENTE — summary.json` |
| Run ID | `PENDIENTE — summary.json` |
| Checkpoint / hash | `PENDIENTE — best.pth + summary.json` |

## Comandos tras cerrar la selección

Para conservar los valores por defecto del pipeline principal:

```bash
make train-main MAIN_MODELS=efficientnet_lite0
```

Solo si la configuración elegida usa los parámetros del HPO, indicar el JSON
verificado del estudio formal. `BEST_PARAMS` se transmite a `--best-params` tanto
en `train-main` como en `modal-train-main`; las opciones explícitas de la CLI
prevalecen sobre ese archivo.

```bash
make train-main MAIN_MODELS=efficientnet_lite0 BEST_PARAMS=/ruta/al/best_hyperparameters.json
```

La ruta del JSON debe existir en el entorno que ejecuta el entrenamiento (local
o contenedor Modal). El resultado del test histórico del HPO no selecciona por
sí solo la configuración formal. Ver [comparación baseline/HPO](../tesis/HPO_BASELINE_COMPARISON.md).

## Resultado observado

| Métrica | Validación | Test |
|---|---:|---:|
| Macro-F1 | `PENDIENTE` | `PENDIENTE` |
| Accuracy | `PENDIENTE` | `PENDIENTE` |
| Loss | `PENDIENTE` | `PENDIENTE` |

Registrar épocas ejecutadas, mejor época, métricas por clase, matriz de confusión, calibración, ambiente y fuente. No copiar automáticamente la métrica del best trial como resultado formal: debe proceder del reentrenamiento contractual.

## Interpretación

`PENDIENTE — redactar después de comparar con 20260921_204608 bajo el mismo split.`

## Limitaciones

`PENDIENTE — incluir fuentes compartidas, auditoría perceptual y alcance de exportación/dispositivo.`
