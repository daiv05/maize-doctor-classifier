# Registro de experimentos

La fuente estructurada histórica es [`manifiesto_corridas.csv`](../resultados/evidencia/manifiesto_corridas.csv). Los runs contractuales recientes se complementan con sus `summary.json`; no se copian valores futuros ni se promueven resultados por estar cronológicamente más recientes.

## Punto de corte actual

| Fecha/run | Modelo | Protocolo/seed | Objetivo | Val Macro-F1 | Test Macro-F1 | Accuracy | Estado |
|---|---|---|---|---:|---:|---:|---|
| 2026-09-21 / `20260921_204608` | EfficientNet-Lite0 | `seed_42` estratificado / 42 | baseline de desarrollo tras corregir el uso de fuentes | 0.956086 | 0.948002 | 0.978066 | **VIGENTE — baseline de desarrollo** |
| 2026-09-21 / `20260921_180112` | EfficientNet-Lite0 | source-grouped / 42 | medir cambio de dominio con fuentes retenidas | 0.266927 | 0.400665 | 0.585705 | **HISTÓRICO — benchmark cross-source** |
| 2026-08-12 / `20260812_221429` | EfficientNet-Lite0 | `seed_42` histórico / 42 | modelo ligero para exportación móvil | 0.955370 | 0.946789 | 0.979063 | **HISTÓRICO — desplegado** |

Las tres filas no forman una clasificación directa: las dos primeras usan splits diferentes y la tercera pertenece a una materialización histórica.

## Contratos verificables

| Campo | `20260921_204608` | `20260921_180112` |
|---|---|---|
| Seed | 42 | 42 |
| Épocas ejecutadas / solicitadas | 47 / 60 | 12 / 60 |
| Mejor época | 39 | 4 |
| Config SHA-256 | `53cc091e…714e8b` | `53cc091e…714e8b` |
| Split lock SHA-256 | `0db3ff3e…8afe8` | `ac9c441f…1b95b` |
| Checkpoint SHA-256 | `860180f3…7859` | `fc2b57f3…d713` |
| Artefacto | `corn-outputs:/main/efficientnet_lite0/20260921_204608/best.pth` | `corn-outputs:/main/efficientnet_lite0/20260921_180112/best.pth` |

Los hashes completos viven en los resúmenes versionados. Las fichas son [baseline de desarrollo](../resultados/run-20260921-efficientnet-lite0.md) y [benchmark source-grouped](../resultados/run-20260921-source-grouped.md).

## Hallazgo de auditoría

`run_afinada_b0_summary.json` tiene un nombre que sugiere B0, pero su campo interno `model` es `efficientnet_lite0`. Se conserva como evidencia histórica y no debe citarse como corrida B0 sin recuperar el artefacto original y resolver la discrepancia.

## Cómo añadir una corrida

1. Conservar `summary.json`, `train_history.csv`, `predictions.csv`, reporte por clase y matriz de confusión.
2. Registrar protocolo y hash del split, no solo `seed_42` como texto.
3. Verificar `best.pth` contra `checkpoint_sha256`.
4. Añadir una fila al CSV de manifiesto solo después de recomputar métricas desde predicciones.
5. Etiquetar el propósito y el estado: `VIGENTE`, `HISTÓRICO`, `EXPERIMENTAL`, `REEMPLAZADO` o `PENDIENTE`.
