# Entrenamiento en la nube con Modal

El proyecto utiliza Modal para ejecutar entrenamientos y análisis de explicabilidad en contenedores con GPU. Los scripts remotos invocan los mismos pipelines que la ejecución local; la configuración de la imagen y los volúmenes se comparte en `scripts/modal/_common.py`.

---

## Funcionamiento

Cada función remota declara sus recursos de GPU, CPU y memoria. Al ejecutarse, el contenedor recibe el código y monta los volúmenes necesarios. Los resultados que deban conservarse se escriben en almacenamiento persistente y se confirman antes de finalizar la invocación.

Todo el almacenamiento se organiza alrededor de dos volúmenes persistentes en la nube que actúan como discos duros compartidos:

- `corn-clean`: almacena el corpus consolidado de imágenes limpias. Se descarga una sola vez desde Hugging Face y queda listo para que cualquier contenedor lo lea a máxima velocidad.
- `corn-outputs`: guarda todos los artefactos generados: particiones de datos (*splits*), checkpoints de los modelos (`best.pth`), historiales de entrenamiento y los reportes de explicabilidad visual.

---

## Ejecución de experimentos

Los comandos de la nube están en el `Makefile` del proyecto con el prefijo `modal-`:

```bash
# Entrenar una arquitectura del pipeline principal en GPU remota
make modal-train-main MAIN_MODELS=efficientnet_lite0 MAIN_EPOCHS=60

# Correr análisis de explicabilidad multimodal en la nube
make modal-explain-compare-main MAIN_MODELS=efficientnet_lite0 SAMPLE_SIZE=20

# Descargar los resultados y checkpoints a la máquina local
make modal-pull
```

Los scripts ejecutan la lógica en la GPU remota y guardan los registros en el volumen. Al terminar la corrida, `make modal-pull` sincroniza la carpeta remota con el entorno local para evaluar y exportar los modelos.
