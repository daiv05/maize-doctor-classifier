# Selección de configuración para el entrenamiento formal

**Regla fijada el 19 de octubre de 2026, antes de ejecutar la cohorte pareada.**
Estado de la decisión: **PENDIENTE**. La regla se fijó con 0/10 runs;
al 19 de octubre de 2026 el [primer par seed 42](../reproducibilidad/evidencia/multiseed_baseline_hpo_seed42/REVIEW.md)
dejó 2/10 runs completas y 1/5 pares. Este documento no designa todavía
una configuración ganadora.

## Alcance y candidatos

La decisión compara únicamente las dos configuraciones **EfficientNet-Lite0**
congeladas en el [protocolo multi-seed](./multiseed.md): `baseline` del run
`20260921_204608` y `hpo_trial_0` del study
`efficientnet_lite0_seed42_hpo_v1`. Se usan los mismos cinco seeds
42/123/2026/3407/7777 y el split `seed_42` congelado. Se trata de elegir
**hiperparámetros para una nueva run formal**, no de declarar ganador a uno de
los checkpoints históricos ni de elegir entre arquitecturas o ensambles.

El test del baseline y del HPO ya fue observado en la fase anterior. Por ello,
esta regla es prospectiva respecto de los **diez entrenamientos nuevos**, no
respecto de toda la investigación. El resultado de test conocido permanece en
la [comparación histórica](../tesis/HPO_BASELINE_COMPARISON.md) como dato
descriptivo; no se usa para cambiar la regla, los candidatos o la decisión.

## Puerta de evidencia

La decisión solo se calcula después de:

1. Obtener diez recibos `COMPLETE.json` válidos, uno por configuración y seed,
   verificados por `python -m scripts.pipeline.multiseed report`. El reporte
   debe tener `complete=true`, `completed_runs=10`, cinco pares y `test_used=false`.
2. Verificar `PROTOCOL.json`, `PREFLIGHT.json`, `REPORT_HASHES.json` y los hashes
   de los diez checkpoints/contratos. Los cinco pares deben compartir protocolo,
   split y entorno; una run histórica, fallida o interrumpida no reemplaza un slot.
3. Registrar las cinco diferencias pareadas de **validation Macro-F1 de nueve
   clases**: `hpo_trial_0 − baseline`. Los valores faltantes nunca se imputan.

Si falta alguna condición, el estado sigue **PENDIENTE**. No se elige el
baseline por omisión mientras la cohorte esté incompleta. Los problemas de
integridad o de entorno se resuelven y documentan antes de aplicar la regla;
no se cambian seeds, splits, candidatos ni umbrales para acomodar resultados.

## Regla de elección

Se elige `hpo_trial_0` **solo si se cumplen ambas condiciones**:

- La media de las cinco diferencias pareadas de validation Macro-F1 es al
  menos **+0,005** en escala 0–1 (**+0,5 puntos porcentuales**).
- HPO supera al baseline en **al menos cuatro de las cinco seeds**; un empate
  exacto no cuenta como victoria.

Con los diez resultados válidos, si falla cualquiera de las dos condiciones,
se elige `baseline` para el entrenamiento formal. El margen de 0,5 puntos es
un requisito práctico de mejora repetible fijado antes de la nueva cohorte,
considerando el coste de cambiar la configuración y que HPO fue elegido con
esta misma validación. **No es un umbral de significancia ni de relevancia
clínica**. No se hará una prueba de hipótesis con cinco seeds ni se afirmará
superioridad de generalización a partir de esta comparación.

Accuracy, F1 de N/P/K, ECE, dispersión e intervalos exploratorios se reportan
por seed y como deltas pareados para interpretar la decisión y sus riesgos.
No alteran esta regla a posteriori. Una degradación preocupante se documenta
como limitación y puede motivar un **experimento posterior con protocolo nuevo**;
no autoriza a cambiar el resultado de este contraste tras mirar test/holdout.

## Cierre y siguiente fase

Tras aplicar la regla, crear un registro de decisión fechado que incluya:
configuración elegida, cinco deltas, media, número de victorias, hashes del
protocolo y reporte, rutas y hashes de los diez checkpoints, estado de
`test_used`, y límites de validación y procedencia. Enlazar ese registro y sus
artefactos en el [registro de evidencias](../tesis/EVIDENCE_REGISTRY.md) y
actualizar el [registro de decisiones](../tesis/DECISION_LOG.md).

El entrenamiento formal posterior debe tener un protocolo y un run contract
propios. La elección de configuración no autoriza por sí sola una nueva
evaluación de test, una selección de seed por su resultado ni la promoción a
producción. CV, LOSO y validación por fuente mantienen preguntas y conjuntos
de evaluación separados.
