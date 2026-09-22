# Protocolos experimentales

**Estado: VIGENTE.** Una métrica solo es comparable con otra si comparten pregunta, corpus, partición, clases evaluables y presupuesto.

## Matriz de protocolos

| Protocolo | Unidad aislada | Pregunta | Uso del test | Estado |
|---|---|---|---|---|
| Holdout `seed_42` | imagen, estratificada por clase+ambiente | ¿qué modelo conviene dentro de fuentes conocidas? | una vez tras selección | **VIGENTE para desarrollo** |
| Holdout `seed_42_source_grouped` | `source_id` completo | ¿qué ocurre ante tres fuentes retenidas? | benchmark separado | **VIGENTE cross-source** |
| CV estratificada | imágenes en K pliegues | ¿cuánta variación hay dentro del mismo dominio mezclado? | no usa el test de `seed_42` como tuning | **HISTÓRICA/EXPERIMENTAL** |
| CV agrupada por fuente | fuentes completas por pliegue | ¿cómo varía el modelo al retener conjuntos de fuentes? | evaluación cruzada | **HISTÓRICA/EXPERIMENTAL** |
| LOSO | una fuente retenida por vez | ¿qué fuentes/clases sostienen señal fuera de dominio? | la fuente retenida actúa como prueba | **HISTÓRICA exploratoria; validación final pendiente** |

`stratified CV != source-grouped CV != LOSO`. La CV agrupada reparte las fuentes en K pliegues; LOSO ejecuta una evaluación por cada fuente retenida. El holdout source-grouped actual es una sola asignación de cinco/tres/tres fuentes.

## Lectura correcta de métricas

- Macro-F1 de nueve clases: media de las nueve clases del clasificador.
- Macro-F1 evaluable: media solo sobre clases con soporte en ese pliegue o subgrupo; debe indicarse cuántas clases entraron.
- Macro-F1 N/P/K agrupado: transforma las tres deficiencias en una categoría y responde otra pregunta. No reemplaza la métrica de nueve clases.
- Macro-F1 por ambiente/fuente: puede promediar conjuntos diferentes de clases. Nunca se interpreta como comparación causal sin revisar soporte.

El resultado histórico `0.6026 ± 0.1240` procede de CV agrupada por fuente de cinco pliegues (`kfold_2x2_comparacion.csv`), no del holdout `seed_42_source_grouped` ni de un promedio LOSO. La ejecución LOSO tiene artefactos propios bajo `docs/es/provenance/evidencia/` y se usa para análisis por clase/fuente.

## Regla de selección

1. HPO usa train/validación de `seed_42`; el test queda cerrado.
2. La configuración elegida se reentrena formalmente y se evalúa una vez en test.
3. Varias semillas cuantifican sensibilidad a inicialización/barajado.
4. CV estratificada estima variación dentro de fuentes conocidas.
5. El benchmark source-grouped y LOSO miden cambio de dominio; no se usan para proclamar una mejora in-distribution.

Cada resultado debe presentar tres bloques: **Resultado observado**, **Interpretación** y **Limitaciones**.
