# Incidente y enmienda técnica del par seed 42 — 2026-10-19

## Hecho observado

La app Modal `ap-hvpdYklqpc5XPXr6gsomWz` inició el primer slot
(`baseline`, seed 42, run `20261019_202150`). La invocación usaba
`run_batch.remote(...)`, que espera sincrónicamente el resultado. Al
interrumpir el cliente local, Modal registró a las **20:24:35 UTC**
`Received a cancellation signal while processing input`; luego apareció
`Runner terminated` y la app se detuvo. La traza llegó aproximadamente al
82 % del entrenamiento de la primera época. **No hubo época completa,
validación, checkpoint, `COMPLETE.json` ni inferencia de test.** El slot HPO
no se inició. La causa fue la cancelación del cliente, no un error de datos
ni una métrica del modelo.

El `STARTED.json` original y `INTERRUPTED.json` quedaron preservados en
`corn-outputs:/multiseed/efficientnet_lite0_baseline_vs_hpo/interrupted/baseline_seed42_20261019_202150/`.
El SHA-256 del recibo `STARTED.json`, comprobado después de archivarlo, es
`bf78ec8eab088c18123bd23cab70101d66b0206a9e844a191f264af37f80ca42`.
El slot original contenía únicamente ese recibo y una carpeta de run vacía;
se liberó después de verificar la copia. El protocolo y el snapshot de
código congelados se mantienen intactos.

## Enmienda de recuperación

Se autoriza **una reejecución manual del slot baseline seed 42**, seguida
del slot HPO seed 42 ya previsto. Es una excepción técnica explícita al
límite original de diez intentos: habrá como máximo **once intentos físicos
si se completan los diez slots**, con diez resultados válidos como máximo.
El intento interrumpido nunca entrará en los agregados ni se presentará
como entrenamiento completo. No se habilitan reintentos automáticos; otro
fallo requiere diagnóstico y una nueva decisión documentada.

La reejecución conserva las configuraciones, las cinco seeds, los hashes del
split, la función remota y `--skip-test`. Solo cambia la forma de **invocar**
esa función: el lanzador temporal, fuera del snapshot de fuentes del
protocolo, usa `run_batch.spawn(...)` y `modal run --detach`. Según la
[documentación de Modal](https://modal.com/docs/guide/function-invocation-methods),
una invocación asíncrona continúa si termina el proceso llamador; una
invocación sincrónica puede cancelarse al perder ese proceso. Se guarda el
ID de la nueva FunctionCall para seguimiento.

La reejecución quedó lanzada en la app Modal `ap-mTUc6PbR24XsRPUsQ8oevu`,
FunctionCall `fc-01M5AXQ2B0949KMTKKXJTWBAG2`, con
`max_new_runs=2`. El cliente local terminó normalmente; la tarea remota
seguía activa y comenzó la segunda época del baseline tras validar la
primera. En el [cierre posterior del primer par](../reproducibilidad/evidencia/multiseed_baseline_hpo_seed42/REVIEW.md),
baseline y HPO terminaron con recibos válidos; el intento archivado no se
incluyó en los agregados.

Esta enmienda se registró después del intento parcial y **antes de observar
métricas de validation del nuevo par**. No cambia la regla de selección
fijada en [selección de configuración](./seleccion-modelo-final.md) ni usa
resultados de test/holdout para decidir la recuperación.
