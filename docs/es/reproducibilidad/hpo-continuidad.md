# Continuidad del HPO y cambio de workspace

El presupuesto autorizado desde el 2026-09-23 es **25 intentos totales**. Reanudar
no reinicia ese contador. No cambiar el dataset, regenerar splits ni crear otro
study para simular una continuación.

El estudio se cerró el 24 de septiembre de 2026. Los procedimientos siguientes
describen recuperación y transferencia de artefactos; no habilitan nuevos trials
ni una segunda evaluación de test. La fase siguiente tiene su propio
[protocolo multi-seed](../experimentos/multiseed.md).

## Qué puede recuperarse

- Los trials completos conservan sus resultados y checkpoints.
- `study.db` conserva el historial y, en este proyecto, el estado RNG serializado
  del sampler TPE. Solo cargar bases propias/confiables: ese estado usa pickle.
- Un trial RUNNING al interrumpir el proceso pasa a FAIL al reanudar y cuenta
  dentro de los 25. **No se retoma su última época**: `best.pth` no es un checkpoint
  completo de optimizer/scheduler/RNG para continuar ese entrenamiento.
- Los últimos datos no persistidos pueden perderse. Respaldar con el escritor
  detenido y verificar SQLite; no copiar una DB mientras está cambiando.
- `FINAL_TEST_STARTED.json` sin `FINAL_TEST_COMPLETE.json` exige auditoría: no
  repetir automáticamente el test. Con selección/test cerrados no añadir trials.

## Transferencia de artefactos entre entornos

La transferencia requiere restaurar explícitamente los artefactos y comprobar
su integridad. Cada workspace mantiene recursos separados; el entorno de destino
debe disponer de los permisos y las dependencias del protocolo original.
[Workspaces](https://modal.com/docs/guide/workspaces).

1. Detener el observador local y la app escritora del origen. Confirmar que no
   quedan invocaciones encadenadas. Registrar el último estado y app ID.
2. Descargar desde `corn-outputs` el directorio completo
   `hpo/efficientnet_lite0/efficientnet_lite0_seed42_hpo_v1/`, incluido SQLite,
   checkpoints, `BUDGET_AMENDMENT.json`, `budget_revisions/`, código y evidencias.
   Descargar además el smoke `_smoke_25`, `splits/seed_42/` y el run baseline
   `main/efficientnet_lite0/20260921_204608/`.
3. Conservar la misma copia de `corn-clean:/clean/` y sus bytes. Una descarga futura
   de Hugging Face podría no ser el mismo corpus; no basta mantener nombres.
4. Respaldar el repositorio efectivo, no solo el último commit: hay cambios sin
   commit. `source_code.zip` contiene el código experimental y configuración;
   guardar también los scripts locales de reporte/observación/migración.
5. Verificar hashes de archivos y `PRAGMA integrity_check` de SQLite. Guardar un
   inventario SHA-256 de la transferencia para comprobarlo en destino.
6. Seleccionar el perfil autorizado del destino, crear volúmenes con los mismos
   nombres y restaurar las rutas exactas. Configurar el secret `hf` por separado;
   nunca incluir tokens ni archivos de credenciales en el respaldo.
7. Ejecutar `make modal-hpo-preflight MODAL="$MODAL"` sobre el código archivado.
   Deben coincidir versiones (incluido Optuna 4.9.0), hashes, splits, baseline y
   protocolo. Si falla, detenerse: no editar hashes para forzar aceptación.
8. En un estudio todavía abierto, la continuación solo puede ejecutar los intentos
   restantes después de validar el estado. Para este estudio, ya cerrado, la
   transferencia termina con la verificación: no ejecutar `make modal-hpo`.
   No mantener escritores activos simultáneamente en origen y destino.

Herramientas oficiales: [`modal volume get/put`](https://modal.com/docs/cli/latest/volume)
y [`modal profile activate`](https://modal.com/docs/cli/latest/profile).
Al descargar una carpeta a un destino existente, el CLI usado aquí conserva el
nombre de la carpeta remota: comprobar la ruta antes de restaurar. Esta guía no
realiza la transferencia ni modifica la configuración del entorno automáticamente.

## Enmienda local de presupuesto ya aplicada

`scripts/pipeline/amend_hpo_budget.py` trabaja sobre una copia local y exige
`--confirm-writer-stopped` y un motivo. Solo admite 60 → 25, antes de selección/test,
verifica que la lógica del objetivo conserva su AST y que trials/RNG no cambian.
Rechaza reutilizar un destino existente. El respaldo previo local está en
`outputs/hpo-backups/before-budget25-uQm2Sw/efficientnet_lite0_seed42_hpo_v1/`.

Esta enmienda no es un procedimiento para reducir nuevamente el presupuesto tras
observar test ni para extenderlo después de elegir al ganador.
