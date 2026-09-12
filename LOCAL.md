# LOCAL.md - Levantar el proyecto en local

Guía paso a paso para dejar el proyecto corriendo en tu máquina: entorno virtual, variables de entorno, dependencias y descarga del dataset.

Para reproducir la reparación comprobada en Linux/Python 3.12, seguir primero la
[guía de entornos separados, constraints y migración](docs/reviews/2026-09-11-reproducibilidad.md).
El dataset HF actual requiere tratar conflictos de etiquetas antes de generar splits;
no sobrescribir las particiones históricas ni asumir que `make splits` los resolverá solo.

## Qué compartir con el equipo

Git conserva código, pruebas, configuraciones, `package-lock.json`, `constraints/`,
instrucciones del proyecto y documentación, incluidas las evidencias pequeñas y los CSV
de exclusiones/revisión de duplicados. No excluir esas carpetas para reducir el número
de archivos pendientes: forman parte de la reparación y su trazabilidad.

`.gitignore` excluye entornos virtuales, secretos `.env`, cachés, builds, temporales,
datos, `outputs/` y paquetes locales. Las plantillas `.env.example` sí se comparten.
Ignorar no borra archivos ni retira los que Git ya tiene versionados.

Para continuar una corrida concreta, compartir por almacenamiento externo los checkpoints,
summaries, manifests, locks y splits exactos que referencia la documentación, con hashes
y permisos para el equipo. No basta con clonar el repo: `outputs/` no se incluye y usar
la misma semilla no garantiza reconstruir sus particiones históricas. El dataset HF se
descarga fijando la revisión indicada en la guía; `/tmp/doctor-maiz-hf-20260911-JjlLG8`
es una ruta temporal de esta máquina, no una ubicación compartida ni permanente.

## Requisitos previos

- Python >= 3.11
- `make` disponible en el PATH (en Windows: Git Bash, WSL, o `choco install make`)

## 1. Clonar el repo

```bash
git clone https://github.com/daiv05/maize-doctor-classifier
cd maize-doctor-classifier
```

## 2. Crear el entorno virtual

El `Makefile` asume que el venv vive en `venv/` en la raíz del proyecto (detecta Windows vs. Linux/macOS automáticamente para elegir `venv/Scripts` o `venv/bin`).

```bash
python -m venv venv
```

Actívalo antes de correr cualquier comando fuera de `make` (los targets de `make` ya invocan el Python del venv directamente, sin necesidad de activarlo):

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

## 3. Configurar `.env`

Copia la plantilla y ajusta las variables:

```bash
cp .env.example .env
```

Variables relevantes:

| Variable | Descripción |
|---|---|
| `DATASET_ROOT` | Ruta local donde vivirá el dataset fuente (debe contener `raw/`, `clean/`). Elige cualquier carpeta de tu máquina, p.ej. `C:/Users/tu_usuario/datasets/corn-leaf-diseases` en Windows o `/Users/tu_usuario/datasets/corn-leaf-diseases` en macOS/Linux. Los artefactos generados (`splits/`, resultados de entrenamiento, reports) viven aparte, en `outputs/` dentro del propio repo - ver `get_output_root()` en `src/config.py`. |
| `HF_DATASET_REPO` | Repo de tipo *dataset* en Hugging Face Hub que contiene `clean/` (fuente primaria de descarga). |
| `HF_TOKEN` | Solo necesario si el repo de HF es privado o no hiciste `huggingface-cli login`. |
| `GDRIVE_DATASET_ID` | ID de carpeta pública de Google Drive, usada como respaldo si falla la descarga desde HF. |

`DATASET_ROOT` no tiene que ser `data/` dentro del repo - puede apuntar a cualquier ruta. El directorio `data/` en la raíz del proyecto es opcionalmente un symlink de conveniencia hacia `DATASET_ROOT`; el código nunca depende de ese symlink, siempre resuelve rutas leyendo la variable de entorno `DATASET_ROOT` (ver `src/config.py`).

Si quieres ese symlink para navegar el dataset más fácilmente desde el editor:

```bash
# Windows (PowerShell, como administrador o con Developer Mode activado)
New-Item -ItemType SymbolicLink -Path data -Target "C:\ruta\a\tu\dataset"

# Linux/macOS
ln -s /ruta/a/tu/dataset data
```

## 4. Instalar dependencias

```bash
make install
```

Esto corre `pip install -e ".[dev,analysis,xai,cloud]"` dentro del venv (instala el paquete `src/` en modo editable + todos los extras necesarios para el flujo local, incluida la descarga del dataset). Extras disponibles en `pyproject.toml`:

- `dev`: ipykernel, jupyterlab, matplotlib, seaborn, ruff, pyright
- `analysis`: imagededup, fiftyone, imageio, mongoengine, motor (necesario para deduplicación y
  exploración visual)
- `xai`: lime, shap, scikit-image, matplotlib (necesario para `make explain-visual`/`fidelity`/`errors`/`compare`/`global`)
- `cloud`: huggingface_hub, gdown (necesario para descargar/subir el dataset)
- `export`: onnx, onnxruntime (necesario para `make export-main`/`train --export`; no incluido
  en `make install` por defecto - instalar con `pip install -e ".[export]"`). TFLite requiere
  ademas `litert-torch` (renombrado desde `ai-edge-torch`, deprecado) y `ai-edge-litert`, que
  solo soportan Linux - ya declarados como dependencia condicional (`sys_platform == 'linux'`)
  dentro del extra `export`, asi que `pip install -e ".[export]"` en un entorno Linux (WSL
  incluido) los instala automaticamente; no hace falta instalarlos aparte.

Si solo necesitas descargar el dataset sin las herramientas de desarrollo/análisis:

```bash
venv\Scripts\pip install -e ".[cloud]"   # Windows
venv/bin/pip install -e ".[cloud]"        # Linux/macOS
```

## 5. Descargar el dataset (`clean/`)

Con `DATASET_ROOT` ya configurado en `.env` (las dependencias de `cloud` ya vienen con `make install`):

```bash
make download-dataset
```

Esto ejecuta `scripts/dataset/download_dataset.py`, que descarga `clean/` hacia
`$DATASET_ROOT/clean/`, intentando primero Hugging Face Hub (`HF_DATASET_REPO`) y usando Google Drive (`GDRIVE_DATASET_ID`) como respaldo si falla. Si `$DATASET_ROOT/clean/` ya tiene contenido, el script no vuelve a descargar (usa `--force` para forzarlo).

Desde HF el dataset llega en shards `clean-<NNNNN>.tar` (~800 MB cada uno, ~19 GB en total); el script
los extrae y elimina automáticamente, dejando el árbol `clean/<clase>/{lab,real}/`. Un `.tar` sin extraer
se considera descarga incompleta y dispara el reintento.

Para **publicar** una versión nueva del dataset (empaqueta los shards y los sube):

```bash
make upload-dataset STAGE_DIR=/ruta/con/espacio DRY_RUN=1   # plan de shards, sin escribir nada
make upload-dataset STAGE_DIR=/ruta/con/espacio             # empaqueta y sube
```

`STAGE_DIR` es obligatorio y necesita ~19 GB libres: ahí se materializan los `.tar` antes de subirlos
(se borra al terminar, salvo con `KEEP_STAGE=1`). Requiere `HF_TOKEN` con permiso de escritura.

**Nunca coloques ni modifiques nada manualmente en `raw/`** - esa carpeta es inmutable y no forma parte de este flujo de descarga; `clean/` es la única fuente de verdad para el pipeline.

## 6. Generar los splits

Con `clean/` ya poblado:

```bash
make splits              # Splits completos (9 clases) -> outputs/splits/seed_42/
make splits-baseline      # Splits del perfil baseline (subset + límite por clase) -> outputs/splits/seed_42_baseline/
```

No edites los CSV de `outputs/splits/` a mano - son derivados reproducibles.

## 7. Verificar que todo funciona

```bash
make summary   # Conteo de imágenes por clase/entorno (valida que clean/ esté bien poblado)
make lint      # ruff check
make fmt       # ruff format
```

## 8. Entrenar

```bash
make train-baselines                          # Entrena baselines sobre el perfil baseline (9 clases, cap 1500 img/clase)
make train-baselines MODELS=efficientnet_b0   # Solo un modelo
make train-baselines NO_CAP=1                 # Mismas 9 clases, sin tope de imágenes
make train-baselines MAX_PER_CLASS=1000       # Mismas 9 clases, tope custom
make train                                    # Principal: desarrollo train/val, sin abrir test
```

## Resumen rápido (happy path)

```bash
git clone https://github.com/daiv05/maize-doctor-classifier
cd maize-doctor-classifier
python -m venv venv
source venv/bin/activate   # o venv\Scripts\Activate.ps1 en Windows
cp .env.example .env       # editar DATASET_ROOT / HF_DATASET_REPO / GDRIVE_DATASET_ID
make install
make download-dataset
make splits
make splits-baseline
make summary
```
