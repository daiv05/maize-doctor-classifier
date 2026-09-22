# Benchmark source-grouped — corrida `20260921_180112`

**Estado: HISTÓRICO, VIGENTE COMO EVIDENCIA CROSS-SOURCE.** No es competidor directo del baseline estratificado.

## Resultado observado

| Campo | Valor |
|---|---:|
| Modelo | EfficientNet-Lite0 |
| Protocolo | fuentes completas retenidas |
| Split | 16 554 train / 7 809 val / 9 066 test |
| Épocas ejecutadas / solicitadas | 12 / 60 |
| Mejor época | 4 |
| Mejor validación Macro-F1 | **0.266927** |
| Train Macro-F1 al cierre | **0.966368** |
| Test Macro-F1 | **0.400665** |
| Test accuracy | **0.585705** |
| Checkpoint SHA-256 | `fc2b57f388b2546846fd43039b483e3dbc1c51a2629a5b0b965d48f08de8d713` |

La validación no contenía `fall_armyworm` ni `lethal_necrosis`; el test estaba formado por tres fuentes completas. Las proporciones 49.52/23.36/27.12 % se apartan de 70/15/15 porque las once fuentes son grupos indivisibles.

## Interpretación

La divergencia entre entrenamiento y validación reveló dependencia fuerte de procedencia. El resultado no demuestra que EfficientNet-Lite0 “empeoró” respecto del run estratificado: cambió la pregunta experimental. Aquí el modelo debe transferir a dominios que no vio; en `seed_42` las mismas fuentes pueden aparecer en todos los splits.

Este resultado motivó separar dos protocolos: `seed_42` para desarrollo balanceado y `seed_42_source_grouped` para generalización entre fuentes. El benchmark conserva valor científico precisamente porque expone el *domain shift*.

## Limitaciones

- Pocas fuentes y tamaños muy desiguales producen splits alejados del objetivo.
- Faltan clases en validación y la composición de clases cambia entre particiones.
- La corrida usó el directorio lógico `seed_42`; su hash de lock (`ac9c441f…`) es la evidencia que identifica la materialización source-grouped histórica.
- No es comparable punto a punto con `20260921_204608`.

Fuentes: [`summary` versionado](./evidencia/run_20260921_180112_summary.json), [`train_history.csv`](./evidencia/run_20260921_180112_train_history.csv) y artefactos originales en `corn-outputs:/main/efficientnet_lite0/20260921_180112/`.
