# Primer par baseline/HPO: seed 42

Corte: **19 de octubre de 2026**. La app Modal
`ap-mTUc6PbR24XsRPUsQ8oevu` terminó con **2/10 runs completas y 1/5 pares**
del estudio `efficientnet_lite0_baseline_vs_hpo`. Los otros ocho slots siguen
pendientes. El [intento baseline interrumpido](../../../experimentos/multiseed-incidente-2026-10-19.md)
se archivó antes de la reejecución y no entra en estas cifras.

| Métrica de validación | Baseline | HPO trial 0 | HPO − baseline |
|---|---:|---:|---:|
| Macro-F1, nueve clases | 0.956086 | 0.957292 | +0.001206 (+0.120596 pp) |
| Accuracy | 0.980654 | 0.980056 | −0.000598 (−0.059832 pp) |
| F1 nitrógeno | 0.898438 | 0.904000 | +0.005563 |
| F1 fósforo | 0.967273 | 0.967033 | −0.000240 |
| F1 potasio | 0.857143 | 0.858586 | +0.001443 |
| ECE, 15 bins; menor es mejor | 0.139844 | 0.057925 | −0.081919 |

En las mismas 5 014 imágenes, HPO corrigió **40** errores del baseline e
introdujo **43** errores nuevos: tuvo 4 914 aciertos frente a 4 917 del
baseline. Por eso su accuracy bajó ligeramente aunque el Macro-F1 subió.

El baseline ejecutó 47 épocas (mejor: 39); HPO ejecutó 52 (mejor: 44).
El baseline volvió a producir el mismo SHA-256 de `best.pth` que el run
histórico `20260921_204608`: `860180f33bf1749ffee9f3cccb9569a47afd9ca0fcef52713e18f99f82a57859`.
El nuevo HPO tiene SHA-256
`00ff9aa5670dec814096ff3755fdd9c5e81ef29587297ad5acf4a94d7a715aa0`;
coincide en Macro-F1 de validación y mejor época con el trial histórico,
pero no se declara identidad de checkpoint entre orquestadores.

## Verificación independiente

- Los cinco hashes de `REPORT_HASHES.json` y los 16 hashes de artefactos
  enumerados en ambos recibos `COMPLETE.json` coinciden con los archivos
  descargados: **21/21 correctos**. Cada `best.pth` coincide además con el
  `checkpoint_sha256` de su summary y recibo.
- El hash del `val.csv` congelado coincide con `PROTOCOL.json`. Cada CSV de
  predicciones tiene **5 014 filas y 5 014 `sample_id` únicos**, exactamente
  los mismos IDs del manifiesto de validación.
- Macro-F1 de nueve clases y accuracy se recalcularon desde `true_label`
  y `pred_label` de cada CSV; reproducen los recibos. ECE y F1 de N/P/K
  coinciden con `validation_diagnostics.json` y los recibos.
- Ambos contratos indican `test_used=false`; este estudio no ejecutó
  inferencia de test. El protocolo canónico conserva SHA-256
  `bb7a509bd1418056614519ecdd45d9646278ba0881ee3fcea85ae64c01cdd7a9`.

Los archivos pequeños de evidencia están en esta carpeta; sus hashes constan
en `MANIFEST.sha256` de esta carpeta. Los checkpoints, predicciones e
historiales completos permanecen en
`corn-outputs:/multiseed/efficientnet_lite0_baseline_vs_hpo/` y en la copia
local ignorada `outputs/multiseed-receipts/efficientnet_lite0_baseline_vs_hpo/`.

## Interpretación

HPO gana este primer par en Macro-F1 por **0.120596 puntos porcentuales** y
presenta menor ECE, pero pierde ligeramente en accuracy. Se reutilizó la
misma seed 42 y la misma validación utilizada durante la selección HPO;
este par comprueba reproducción del pipeline, **no aporta una validación
independiente de generalización**. Con n=1 no hay SD muestral ni intervalo
informativo. La [regla de selección](../../../experimentos/seleccion-modelo-final.md)
exige los cinco pares completos: todavía no se elige configuración para el
entrenamiento formal.
