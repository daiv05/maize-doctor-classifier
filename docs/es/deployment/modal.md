# Entrenamiento en la nube con Modal

Entrenar redes convolucionales sobre más de 33,000 imágenes en alta resolución y correr análisis de explicabilidad con miles de perturbaciones no es algo que se deba hacer en una computadora portátil. Intentarlo en una máquina local sin GPU dedicada significaría esperar días enteros por cada experimento, con el riesgo constante de sobrecalentamientos o interrupciones.

La alternativa tradicional en la nube —alquilar una máquina virtual con GPU en AWS o GCP— tiene su propia trampa: hay que configurar controladores de Nvidia, instalar dependencias a mano y, sobre todo, acordarse de apagar la instancia. Un olvido de fin de semana con una GPU potente encendida puede agotar el presupuesto del proyecto sin haber entrenado nada.

Para resolver esto utilizamos **Modal**, una plataforma de cómputo en la nube basada en contenedores bajo demanda.

---

## Cómo encaja en nuestro flujo de trabajo

La gran ventaja de Modal es su modelo de servidor efímero (*serverless*). Escribimos el código de entrenamiento en Python exactamente igual que para local, pero le indicamos a Modal qué recursos de hardware necesita cada función (por ejemplo, una GPU NVIDIA A10G de 24 GB de memoria y 8 núcleos de procesador). 

Cuando lanzamos un comando, Modal realiza el aprovisionamiento en segundos: levanta el contenedor, monta el código, ejecuta el entrenamiento y se destruye automáticamente en cuanto termina. La facturación se calcula por segundo exacto de uso, garantizando que nunca se pague por recursos ociosos.

Todo el almacenamiento se organiza alrededor de dos volúmenes persistentes en la nube que actúan como discos duros compartidos:

- **`corn-clean`**: Almacena el corpus consolidado de imágenes limpias. Se descarga una sola vez desde Hugging Face y queda listo para que cualquier contenedor lo lea a máxima velocidad.
- **`corn-outputs`**: Guarda todos los artefactos generados: particiones de datos (*splits*), checkpoints de los modelos (`best.pth`), historiales de entrenamiento y los reportes de explicabilidad visual.

---

## Ejecución de experimentos

Para mantener la experiencia de desarrollo simple, los comandos de la nube se integraron en el mismo `Makefile` del proyecto, usando el prefijo `modal-` para distinguirlos de las corridas locales:

```bash
# Entrenar la arquitectura de producción en GPU remota
make modal-train-main MAIN_MODELS=efficientnet_lite0 MAIN_EPOCHS=60

# Correr análisis de explicabilidad multimodal en la nube
make modal-explain-compare-main MAIN_MODELS=efficientnet_lite0 SAMPLE_SIZE=20

# Descargar los resultados y checkpoints a la máquina local
make modal-pull
```

Por debajo, los scripts empaquetan los parámetros necesarios, ejecutan la lógica en la GPU remota y guardan los registros en el volumen. Una vez finalizada la corrida, `make modal-pull` sincroniza la carpeta remota con el entorno local, dejando los modelos listos para su posterior evaluación y exportación a dispositivos móviles.

Este enfoque nos permitió iterar con rapidez, comparar múltiples arquitecturas en paralelo y ejecutar auditorías de interpretabilidad pesadas sin depender de la infraestructura física del equipo de desarrollo.
