# Entrenamiento en la nube con Modal

El proyecto utiliza Modal para ejecutar entrenamientos y análisis de explicabilidad
en contenedores con GPU. Los scripts remotos invocan los mismos pipelines que la
ejecución local; la configuración de la imagen y los volúmenes se comparte en
`scripts/modal/_common.py`.

---

## Cómo encaja en nuestro flujo de trabajo

Cada función remota declara sus recursos de GPU, CPU y memoria. Al ejecutarse,
el contenedor recibe el código y monta los volúmenes necesarios. Los resultados
que deban conservarse se escriben en almacenamiento persistente y se confirman
antes de finalizar la invocación.

Todo el almacenamiento se organiza alrededor de dos volúmenes persistentes en la nube que actúan como discos duros compartidos:

- **`corn-clean`**: Almacena el corpus consolidado de imágenes limpias. Se descarga una sola vez desde Hugging Face y queda listo para que cualquier contenedor lo lea a máxima velocidad.
- **`corn-outputs`**: Guarda todos los artefactos generados: particiones de datos (*splits*), checkpoints de los modelos (`best.pth`), historiales de entrenamiento y los reportes de explicabilidad visual.

---

## Ejecución de experimentos

Para mantener la experiencia de desarrollo simple, los comandos de la nube se integraron en el mismo `Makefile` del proyecto, usando el prefijo `modal-` para distinguirlos de las corridas locales:

```bash
# Entrenar una arquitectura del pipeline principal en GPU remota
make modal-train-main MAIN_MODELS=efficientnet_lite0 MAIN_EPOCHS=60

# Correr análisis de explicabilidad multimodal en la nube
make modal-explain-compare-main MAIN_MODELS=efficientnet_lite0 SAMPLE_SIZE=20

# Descargar los resultados y checkpoints a la máquina local
make modal-pull
```

Por debajo, los scripts empaquetan los parámetros necesarios, ejecutan la lógica en la GPU remota y guardan los registros en el volumen. Una vez finalizada la corrida, `make modal-pull` sincroniza la carpeta remota con el entorno local, dejando los modelos listos para su posterior evaluación y exportación a dispositivos móviles.

Este enfoque nos permitió iterar con rapidez, comparar múltiples arquitecturas en paralelo y ejecutar auditorías de interpretabilidad pesadas sin depender de la infraestructura física del equipo de desarrollo.
