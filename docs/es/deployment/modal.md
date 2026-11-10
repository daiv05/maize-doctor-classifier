# Entrenamiento en la nube con Modal

El proyecto utiliza Modal para ejecutar entrenamientos y análisis de explicabilidad en contenedores con GPU. Los scripts remotos invocan los mismos pipelines que la ejecución local; la configuración de la imagen y los volúmenes se comparte en `scripts/modal/_common.py`.

---

## Funcionamiento

Cada función remota declara sus recursos (GPU A10 y CPU en las de entrenamiento) y su tiempo máximo. Al ejecutarse, el contenedor recibe el código y monta los volúmenes necesarios. Los resultados que deban conservarse se escriben en almacenamiento persistente y se confirman antes de finalizar la invocación.

El almacenamiento se organiza en volúmenes persistentes de Modal:

- `corn-clean`: almacena el corpus consolidado de imágenes limpias, descargado desde Hugging Face con `make modal-seed` (idempotente; `FORCE=1` lo vacía y lo vuelve a descargar).
- `corn-clean-segmented`: corpus pre-segmentado, usado por las corridas con `SEGMENTED=1`.
- `corn-outputs`: guarda todos los artefactos generados: particiones de datos (*splits*), checkpoints de los modelos (`best.pth`), historiales de entrenamiento y los reportes de explicabilidad visual.

---

## Ejecución de experimentos

Los comandos de la nube están en el `Makefile` del proyecto con el prefijo `modal-`:

```bash
# Entrenar una arquitectura del pipeline principal en GPU remota
make modal-train-main MAIN_MODELS=efficientnet_lite0 MAIN_EPOCHS=60

# Panel comparado LIME | SHAP | Grad-CAM en la nube
make modal-explain-compare-main MAIN_MODELS=efficientnet_lite0 SAMPLE_SIZE=20

# Descargar el volumen corn-outputs completo a ./outputs-remote
make modal-pull
```

Los scripts ejecutan la lógica en la GPU remota y guardan los registros en el volumen. Al terminar la corrida, `make modal-pull` ejecuta `modal volume get --force corn-outputs / ./outputs-remote`: copia todo el volumen a `./outputs-remote` sobrescribiendo los archivos existentes.
