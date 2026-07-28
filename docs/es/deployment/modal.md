# Entrenamiento en Modal

[Modal](https://modal.com/docs/guide) es la plataforma de GPU en la nube que usa el proyecto para entrenar los **baselines**, el **pipeline principal** y correr **explicabilidad** sobre cualquiera de los dos, cuando conviene más potencia o más tiempo del que da la máquina local. La idea es simple: defines código Python normal, lo decoras para que Modal sepa qué correr en la nube, y esa ejecución se factura por segundo mientras dura. Cuando termina, la instancia se destruye sola (auto-teardown), así que no hay que acordarse de apagar nada ni pagar por GPU ociosa. Otra ventaja práctica es que los scripts de Modal exponen la misma CLI que sus equivalentes locales (`train_baselines.py`, `explain.py`): cualquier combinación de flags que funcione en local funciona igual en Modal.

## Cómo está montado

Todo el código que corre en Modal comparte una única imagen definida en `scripts/modal/_common.py`. Esa factorización es deliberada: `train.py` y `explain.py` necesitan exactamente la misma versión de torch y los mismos extras instalados para que los checkpoints generados por uno se puedan leer desde el otro. La imagen instala `torch==2.12.1` y `torchvision==0.27.1` desde el índice de PyTorch, más los extras `cloud` y `xai` del proyecto, y monta `src/` y `scripts/` en caliente para no tener que reconstruir la imagen en cada cambio de código.

El almacenamiento persistente se resuelve con dos Volumes de Modal, que se crean solos la primera vez que se usan:

- `corn-clean`, montado en `/data`, contiene el dataset limpio.
- `corn-outputs`, montado en `/outputs`, contiene los artefactos generados: splits, pesos de modelos, métricas y reportes de explicabilidad (LIME, Grad-CAM, SHAP).

El dataset no se sube en cada corrida: se sube una única vez al volumen con `make modal-seed`, que es idempotente (si ya existe, no lo vuelve a descargar). Antes de esa primera vez hace falta autenticar la cuenta y darle a Modal acceso al dataset de Hugging Face:

```bash
pip install -e ".[cloud]"        # incluye el cliente modal
modal setup                      # autentica tu cuenta de Modal en el navegador
modal secret create hf HF_TOKEN=hf_xxxxxxxx   # token de Hugging Face para el dataset
```

## Cómo se corre el flujo

Con el dataset ya sembrado, entrenar un baseline en Modal se ve casi igual que entrenarlo en local, solo que con el prefijo `modal-`. Por ejemplo, para entrenar `efficientnet_b0` en la GPU A10, que es la que está configurada por defecto:

```bash
make modal-train-baselines MODELS=efficientnet_b0 EPOCHS=30
```

Las mismas banderas que acepta `train_baselines.py` en local están disponibles aquí, por ejemplo `NO_CAP=1` para entrenar sin tope de imágenes por clase, o `LIME=1` para que el run termine generando también los reportes LIME:

```bash
make modal-train-baselines MODELS=efficientnet_b0 NO_CAP=1
make modal-train-baselines MODELS=efficientnet_b0 LIME=1
```

Por debajo, `scripts/modal/train.py` traduce estas variables de `make` a los flags reales del script (`--models`, `--epochs`, `--no-cap`/`--max-per-class`, `--batch-size`, `--image-size`, `--learning-rate`, `--weight-decay`, `--num-workers`, `--no-pretrained`, `--lime`, `--regenerate-splits`), así que cualquier combinación que funcione en local funciona igual aquí.

El pipeline principal se entrena con su propio target, que usa `MAIN_MODELS`/`MAIN_EPOCHS` en vez de `MODELS`/`EPOCHS` para no confundir la configuración de un pipeline con la del otro:

```bash
make modal-train-main MAIN_MODELS=shufflenet_v2_x1_0 MAIN_EPOCHS=60
```

Una vez que hay un run entrenado, la explicabilidad post-hoc se corre por separado con el CLI unificado `scripts/pipeline/explain.py` (cinco subcomandos: `visual`, `fidelity`, `errors`, `compare`, `global`). Aquí también los targets están diferenciados por pipeline, porque cada uno lee de un directorio de runs distinto dentro del Volume:

```bash
# baselines -> /outputs/baselines
make modal-explain-visual-baselines MODELS=efficientnet_b0
make modal-explain-fidelity-baselines MODELS=efficientnet_b0 SAMPLE_SIZE=50
make modal-explain-errors-baselines MODELS=efficientnet_b0

# pipeline principal -> /outputs/main
make modal-explain-visual-main MAIN_MODELS=shufflenet_v2_x1_0
make modal-explain-fidelity-main MAIN_MODELS=shufflenet_v2_x1_0 SAMPLE_SIZE=50
make modal-explain-errors-main MAIN_MODELS=shufflenet_v2_x1_0
```

Los targets sin sufijo (`modal-explain-visual`, `modal-explain-fidelity`, `modal-explain-errors`) siguen existiendo como genéricos: apuntan a baselines salvo que se les pase `PIPELINE=main`.

Por debajo, esa elección viaja como `--pipeline baselines|main` y es el contenedor el que la traduce a `--output-dir /outputs/<pipeline>`. Se hace por nombre y no pasando la ruta directamente porque, al invocar `make` desde Git Bash en Windows, MSYS reescribe cualquier argumento que empiece con `/` a una ruta de Windows (`/outputs/main` terminaría como `C:/Program Files/Git/outputs/main`).

Igual que con el entrenamiento, `scripts/modal/explain.py` espeja los flags de `scripts/pipeline/explain.py` (`--run`, `--baseline`, `--sample-size`, `--num-samples`, `--errors-only`). La única diferencia a tener en cuenta es que `--image`/`--output` de `explain-visual` deben ser rutas dentro del contenedor (relativas a `/data` u `/outputs`), no del filesystem local.

Los subcomandos `compare` y `global` traen SHAP y son exclusivos del pipeline principal (no tienen variante `-baselines`), porque el spec que introdujo SHAP lo acotó a `outputs/main`:

```bash
make modal-explain-compare-main MAIN_MODELS=shufflenet_v2_x1_0 SAMPLE_SIZE=5    # panel LIME | SHAP | Grad-CAM + acuerdo
make modal-explain-global-main MAIN_MODELS=shufflenet_v2_x1_0 SAMPLE_SIZE=30    # perfil global por clase (mapas + ratio hoja/fondo)
```

Si en algún momento hace falta empezar de cero (splits, runs o reportes viejos que ya no aplican), `make modal-clean-outputs` vacía el volumen de outputs sin tocar el dataset.

Quien prefiera no pasar por `make` puede invocar los mismos comandos de Modal directamente:

```bash
modal run scripts/modal/train.py::seed_dataset
modal run scripts/modal/train.py --models "efficientnet_b0" --epochs 30
modal run scripts/modal/train.py::train_main --models "shufflenet_v2_x1_0" --epochs 60
modal run scripts/modal/explain.py::explain_fidelity --models "efficientnet_b0" --sample-size 50
modal run scripts/modal/explain.py::explain_fidelity --models "shufflenet_v2_x1_0" --pipeline main
modal run scripts/modal/train.py::clean_outputs
```

## Cómo se traen los resultados

Los resultados de cada corrida se versionan igual que en local, en `/outputs/baselines/<modelo>/<run_id>/` (o `/outputs/main/<modelo>/<run_id>/` para el pipeline principal, donde `run_id` es un timestamp), así que un mismo modelo puede acumular varios runs sin pisarse entre sí. Para bajarlos a la máquina local:

```bash
make modal-pull            # copia el volumen corn-outputs -> ./outputs-remote
```

Por debajo esto es `modal volume get --force corn-outputs / ./outputs-remote`; el `--force` sobreescribe la carpeta local si ya existía de una corrida anterior.

## Otras notas útiles

Los splits del baseline se generan la primera vez que se necesitan y se reutilizan (lazy) en corridas siguientes, igual que hace `train_baselines.py` en local. Si ya existen con otro tope de imágenes por clase, no se regeneran solos: hay que correr `make modal-clean-outputs` primero, o pasar `REGEN_SPLITS=1` para forzar la regeneración del split baseline en la misma corrida.

Para cambiar de GPU basta con editar `gpu="A10"` en `scripts/modal/train.py`/`explain.py`; las opciones disponibles incluyen T4, L4, A10, L40S, A100 y H100.

A diferencia de `train_baselines.py`, el pipeline principal no genera los splits de forma lazy: requiere `outputs/splits/seed_42` y falla si no existe, así que la primera corrida de `modal-train-main` necesita un `make modal-splits` previo.

### Problema conocido: `splits_dir` absoluto de contenedor en runs de Modal

Un run entrenado en Modal persiste `splits_dir` en su `summary.json` como ruta absoluta del contenedor (p.ej. `/outputs/main/shufflenet_v2_x1_0/<run_id>/../../splits/seed_42`). `load_run_metadata` (`src/training/common.py`) lee ese valor tal cual (ver docstring), sin traducirlo al filesystem local. Si ese run se explica luego en Windows (por ejemplo tras un `make modal-pull` y `make explain-visual-main` local), el subcomando `visual` de `scripts/pipeline/explain.py` falla al resolver `test.csv` en su modo de muestreo por lotes, porque intenta leer esa ruta absoluta de contenedor tal cual en vez de resolverla contra `outputs/splits/` local. Este bug es preexistente a la migración a SHAP y queda fuera de alcance de este cambio; si te encuentras con un `FileNotFoundError` de `test.csv` al explicar un run entrenado en Modal, sospecha primero de esto antes de asumir que faltan splits.
