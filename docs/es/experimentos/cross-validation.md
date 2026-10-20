# Plan de validación cruzada y generalización

**Estado: PENDIENTE para la configuración posterior al HPO.** Existen resultados históricos, pero no se trasladan automáticamente al corpus ni al checkpoint vigentes.

## Protocolos separados

| Protocolo | Unidad que no puede cruzar pliegues | Pregunta | Salida mínima |
|---|---|---|---|
| CV estratificada | muestra | variación dentro de fuentes conocidas | métricas por pliegue, media y SD |
| CV source-grouped | `source_id` | sensibilidad a conjuntos de fuentes retenidas | fuentes/clases por pliegue + Macro-F1 evaluable |
| LOSO | una fuente completa | transferencia específica a cada fuente | una fila por fuente/clase y seed |

`stratified CV != source-grouped CV != LOSO`. El holdout `seed_42` tampoco es un pliegue de CV, y `seed_42_source_grouped` es una asignación única, no un promedio.

## Protocolo previo a ejecutar

Antes de correr se debe fijar en el decision log:

1. checkpoint/configuración de partida y commit;
2. manifest y hash del corpus;
3. número de pliegues y seeds;
4. unidad de agrupación y tratamiento de clases ausentes;
5. presupuesto por pliegue y regla de selección;
6. definición de Macro-F1 global frente a Macro-F1 evaluable;
7. agregación e intervalo reportado.

El test de `seed_42` no se reutiliza para seleccionar hiperparámetros durante CV. Cada ejecución debe conservar manifest de pliegue, summary, historial, predicciones y checkpoint/hash cuando corresponda.

## Resultado observado histórico

La campaña 2×2 previa reportó, para la configuración base, CV estratificada `0.931145 ± 0.009872` y CV source-grouped `0.602575 ± 0.123966` de Macro-F1 evaluable. Los pliegues agrupados tuvieron entre cinco y ocho clases con soporte. La evidencia está en [`kfold_2x2_comparacion.csv`](../resultados/evidencia/kfold_2x2_comparacion.csv).

## Interpretación

En esa campaña, variar la partición tuvo un efecto mayor que cambiar la configuración. Es una observación histórica que motiva repetir ambos ejes con el modelo elegido, no una predicción de sus valores futuros.

## Limitaciones

Los valores anteriores pertenecen a una materialización y configuración históricas. LOSO tiene artefactos diferentes en `docs/es/provenance/evidencia/`. La validación final permanece pendiente hasta que sus nuevas corridas aparezcan en el registro de evidencias.
