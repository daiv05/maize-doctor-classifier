# Pipeline Principal de Entrenamiento — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar el loop de entrenamiento de `scripts/pipeline/train.py` con loss ponderada, cosine+warmup, early stopping y métricas de calibración/environment/agrupadas, extrayendo el loop compartido a `src/training/` sin alterar el comportamiento de los baselines.

**Architecture:** Se extrae el loop de `train_baselines.py` a módulos en `src/training/` con scheduler y early stopping opcionales (`None` = comportamiento baseline actual). `train.py` pasa de andamiaje con `TODO` a ensamblador delgado que activa esos mecanismos. Las métricas nuevas se calculan post-hoc sobre `predictions.csv`, sin GPU.

**Tech Stack:** PyTorch 2.x, torchvision, timm, scikit-learn, pandas, pytest (nuevo), opencv-python (promovido a dependencia principal).

## Global Constraints

- **Python** `>=3.11`. Ruff `line-length = 100`, `target-version = "py311"`, lint `select = ["E","F","W","I"]`.
- **Comentarios de código:** exclusivamente DocBlocks estilo JSDoc/PHPDoc (`@param`, `@returns`). Prohibidos los comentarios narrativos dentro de la lógica. Si una función es autoexplicativa, no lleva comentario. Regla de `CLAUDE.md` global — no negociable.
- **Nunca hardcodear rutas ni constantes de dominio.** Dataset vía `get_dataset_root()`, artefactos vía `get_output_root()` (ambas en `src/config.py`). Clases, `target_size` y seed vienen de `config/dataset.yaml`.
- **Sin `sys.path.append`.** El paquete es editable; los imports `src.*` resuelven directo.
- **`train_baselines.py` no cambia de comportamiento.** Mismos defaults, mismos números. Protegido por el test de regresión de la Tarea 2.
- **Commits:** conventional commits, sin trailer `Co-Authored-By` ni atribución de modelo.
- **Idioma:** docstrings, mensajes de log y de commit en español, consistente con el código existente.
- **Punto único de entrada a imagen:** `load_and_normalize_image()` (`src/data/loader.py`).
- **Balanceo del pipeline principal:** loss ponderada `sqrt_inverse` + augmentation de minoritarias. El `WeightedRandomSampler` va **desactivado** (dos capas, no tres).

---

### Task 1: Infraestructura de tests

No existe `tests/` ni pytest en el proyecto. Todo el plan es TDD, así que esto va primero.

**Files:**
- Modify: `pyproject.toml:18` (extra `dev`), `pyproject.toml:35` (config pytest nueva)
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nada.
- Produces: fixture `tmp_splits_dir(tmp_path) -> Path` que genera `train.csv`/`val.csv`/`test.csv` sintéticos con columnas `image_path,label,environment`; fixture `fake_image_root(tmp_path) -> Path` con imágenes RGB reales de 32×32 escritas en disco.

- [ ] **Step 1: Añadir pytest a las dependencias de dev**

En `pyproject.toml`, línea 18, reemplazar la línea `dev` y añadir la config de pytest después del bloque `[tool.ruff.lint]`:

```toml
dev      = ["ipykernel", "jupyterlab", "matplotlib", "seaborn", "ruff", "pyright", "pytest>=8.0,<9.0"]
```

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Instalar**

Run: `pip install -e ".[dev]"`
Expected: instala pytest sin conflictos.

- [ ] **Step 3: Crear las fixtures**

`tests/__init__.py` queda vacío. `tests/conftest.py`:

```python
"""Fixtures compartidas por la suite de tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

_CLASS_DISTRIBUTION = {
    "healthy": 12,
    "common_rust": 6,
    "potassium_deficiency": 2,
}


@pytest.fixture
def fake_image_root(tmp_path: Path) -> Path:
    """
    Crea un árbol clean/<clase>/<entorno>/ con imágenes RGB de 32x32.

    @param {Path} tmp_path Directorio temporal provisto por pytest.
    @returns {Path} Raíz del dataset sintético.
    """
    rng = np.random.default_rng(42)
    root = tmp_path / "dataset"
    for class_name, count in _CLASS_DISTRIBUTION.items():
        for index in range(count):
            environment = "lab" if index % 2 == 0 else "real"
            directory = root / "clean" / class_name / environment
            directory.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(directory / f"{class_name}_{index}.png")
    return root


@pytest.fixture
def tmp_splits_dir(tmp_path: Path, fake_image_root: Path) -> Path:
    """
    Genera train/val/test.csv apuntando a las imágenes de `fake_image_root`.

    @param {Path} tmp_path Directorio temporal provisto por pytest.
    @param {Path} fake_image_root Raíz del dataset sintético.
    @returns {Path} Directorio con los tres manifiestos.
    """
    rows = []
    for image_path in sorted((fake_image_root / "clean").rglob("*.png")):
        relative = image_path.relative_to(fake_image_root)
        rows.append(
            {
                "image_path": relative.as_posix(),
                "label": image_path.parent.parent.name,
                "environment": image_path.parent.name,
            }
        )
    data_frame = pd.DataFrame(rows)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for split_name in ("train", "val", "test"):
        data_frame.to_csv(splits_dir / f"{split_name}.csv", index=False)
    return splits_dir
```

- [ ] **Step 4: Test que valida las fixtures**

`tests/test_smoke.py`:

```python
import pandas as pd


def test_tmp_splits_dir_tiene_las_columnas_esperadas(tmp_splits_dir):
    data_frame = pd.read_csv(tmp_splits_dir / "train.csv")
    assert list(data_frame.columns) == ["image_path", "label", "environment"]
    assert len(data_frame) == 20
    assert set(data_frame["environment"]) == {"lab", "real"}


def test_distribucion_desbalanceada(tmp_splits_dir):
    counts = pd.read_csv(tmp_splits_dir / "train.csv")["label"].value_counts()
    assert counts["healthy"] == 12
    assert counts["potassium_deficiency"] == 2
```

- [ ] **Step 5: Ejecutar**

Run: `pytest tests/test_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/
git commit -m "test: infraestructura de pytest con fixtures de dataset sintetico"
```

---

### Task 2: Test de regresión de baselines (bloqueante)

Captura el comportamiento actual **antes** de refactorizar. Si este test cambia de resultado en cualquier tarea posterior, el refactor rompió los baselines publicados.

**Files:**
- Create: `tests/training/__init__.py`, `tests/training/test_baseline_regression.py`

**Interfaces:**
- Consumes: `tmp_splits_dir`, `fake_image_root` (Tarea 1).
- Produces: `run_one_epoch_baseline(splits_dir, dataset_root, seed) -> dict[str, float]` — helper que entrena 1 época de shufflenet sin pretrained y devuelve las métricas; reutilizado como oráculo tras cada refactor.

- [ ] **Step 1: Escribir el test de referencia**

`tests/training/__init__.py` vacío. `tests/training/test_baseline_regression.py`:

```python
"""Congela el comportamiento del loop de baselines para detectar regresiones del refactor.

Los resultados de la Tabla 6.2 del reporte de primera fase ya estan publicados: el
loop de baselines debe producir numeros identicos antes y despues de extraer el
codigo compartido a src/training/.
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def run_one_epoch_baseline(splits_dir: Path, dataset_root: Path, seed: int = 42) -> dict:
    """
    Entrena una epoca de shufflenet sobre el dataset sintetico y devuelve las metricas.

    @param {Path} splits_dir Directorio con train/val/test.csv.
    @param {Path} dataset_root Raiz del dataset apuntada por los manifiestos.
    @param {int} seed Semilla global.
    @returns {dict} Metricas de train y val de la unica epoca.
    """
    os.environ["DATASET_ROOT"] = str(dataset_root)

    from src.config import set_global_seed
    from src.data.dataset import CornDataset
    from src.data.transforms import CornTransformFactory
    from src.models import build_model
    from src.training.loop import fit

    set_global_seed(seed)
    factory = CornTransformFactory(target_size=(32, 32))
    train_dataset = CornDataset(
        csv_path=str(splits_dir / "train.csv"),
        transform=factory.get_pipeline("train"),
        minority_transform=factory.get_pipeline("minority"),
    )
    val_dataset = CornDataset(
        csv_path=str(splits_dir / "val.csv"),
        transform=factory.get_pipeline("val"),
        class_to_idx=train_dataset.class_to_idx,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = build_model(
        "shufflenet_v2_x1_0", num_classes=len(train_dataset.class_to_idx), pretrained=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
        model_name="shufflenet_v2_x1_0",
    )
    return history[-1]


def test_una_epoca_es_reproducible(tmp_splits_dir, fake_image_root):
    """Dos corridas con la misma semilla producen exactamente las mismas metricas."""
    first = run_one_epoch_baseline(tmp_splits_dir, fake_image_root)
    second = run_one_epoch_baseline(tmp_splits_dir, fake_image_root)

    assert first["train_loss"] == pytest.approx(second["train_loss"], abs=1e-6)
    assert first["val_macro_f1"] == pytest.approx(second["val_macro_f1"], abs=1e-6)


def test_historial_tiene_las_claves_del_csv_publicado(tmp_splits_dir, fake_image_root):
    """El esquema de train_history.csv no cambia: explain_report.py y los notebooks lo leen."""
    row = run_one_epoch_baseline(tmp_splits_dir, fake_image_root)

    for key in (
        "model",
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "epoch_seconds",
    ):
        assert key in row, f"falta la clave {key} en el historial"
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/training/test_baseline_regression.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.training.loop'`. Es el fallo correcto — el módulo se crea en la Tarea 3.

- [ ] **Step 3: Commit**

```bash
git add tests/training/
git commit -m "test: congela el comportamiento del loop de baselines antes del refactor"
```

---

### Task 3: Extraer el loop a `src/training/loop.py`

**Files:**
- Create: `src/training/loop.py`
- Modify: `scripts/pipeline/train_baselines.py:140-183` (borrar `_run_epoch`), `:291-337` (usar `fit`)
- Test: `tests/training/test_baseline_regression.py` (ya escrito, debe pasar)

**Interfaces:**
- Consumes: `tmp_splits_dir`, `fake_image_root`.
- Produces:
  - `run_epoch(model, loader, criterion, device, optimizer=None, desc="", clip_grad_norm=None) -> tuple[dict[str,float], list[int], list[int], list[float]]`
  - `fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_name, run_dir=None, scheduler=None, early_stopping=None, clip_grad_norm=None, on_epoch_end=None) -> list[dict]`

`fit` con `scheduler=None`, `early_stopping=None` y `clip_grad_norm=None` debe ser idéntico en comportamiento al loop actual de baselines.

- [ ] **Step 1: Crear `src/training/loop.py`**

```python
"""Loop de entrenamiento compartido por el pipeline de baselines y el principal.

Los mecanismos que distinguen al pipeline principal (scheduler, early stopping,
gradient clipping) son opcionales: con los tres en None, `fit` se reduce
exactamente al loop de baselines de la Tabla 6.2 del reporte.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _metrics_from_predictions(
    labels: list[int], predictions: list[int], loss: float
) -> dict[str, float]:
    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
    clip_grad_norm: float | None = None,
) -> tuple[dict[str, float], list[int], list[int], list[float]]:
    """
    Ejecuta una pasada completa sobre el loader, entrenando si se pasa optimizer.

    @param {torch.nn.Module} model Modelo a ejecutar.
    @param {DataLoader} loader Origen de los batches.
    @param {torch.nn.Module} criterion Funcion de perdida.
    @param {torch.device} device Dispositivo de computo.
    @param {torch.optim.Optimizer|None} optimizer Si se pasa, la pasada es de entrenamiento.
    @param {str} desc Etiqueta de la barra de progreso.
    @param {float|None} clip_grad_norm Norma maxima de gradiente; None desactiva el recorte.
    @returns {tuple} Metricas, etiquetas reales, predicciones y confianzas.
    """
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    seen = 0
    labels_all: list[int] = []
    preds_all: list[int] = []
    probs_all: list[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            labels_all.extend(labels.detach().cpu().tolist())
            probs = logits.detach().softmax(dim=1)
            preds_all.extend(probs.argmax(dim=1).cpu().tolist())
            probs_all.extend(probs.max(dim=1).values.cpu().tolist())

    avg_loss = running_loss / max(seen, 1)
    return _metrics_from_predictions(labels_all, preds_all, avg_loss), labels_all, preds_all, probs_all


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_name: str,
    run_dir: Path | None = None,
    scheduler: object | None = None,
    early_stopping: object | None = None,
    clip_grad_norm: float | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """
    Entrena `epochs` epocas, guardando el mejor checkpoint por val_macro_f1.

    @param {torch.nn.Module} model Modelo a entrenar.
    @param {DataLoader} train_loader Batches de entrenamiento.
    @param {DataLoader} val_loader Batches de validacion.
    @param {torch.nn.Module} criterion Funcion de perdida.
    @param {torch.optim.Optimizer} optimizer Optimizador.
    @param {torch.device} device Dispositivo de computo.
    @param {int} epochs Numero maximo de epocas.
    @param {str} model_name Nombre del modelo, usado en logs e historial.
    @param {Path|None} run_dir Destino de best.pth/last.pth; None omite la escritura.
    @param {object|None} scheduler Scheduler con `.step()` por epoca; None deja el lr fijo.
    @param {object|None} early_stopping Objeto con `.step(metric) -> bool`; None desactiva la parada.
    @param {float|None} clip_grad_norm Norma maxima de gradiente.
    @param {Callable|None} on_epoch_end Callback invocado con (epoca, fila del historial).
    @returns {list[dict]} Historial, una fila por epoca ejecutada.
    """
    history: list[dict] = []
    best_val_macro_f1 = -1.0

    for epoch in range(1, epochs + 1):
        started = perf_counter()
        train_metrics, _, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"{model_name} train {epoch}/{epochs}",
            clip_grad_norm=clip_grad_norm,
        )
        val_metrics, _, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            desc=f"{model_name} val {epoch}/{epochs}",
        )

        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": perf_counter() - started,
        }
        if scheduler is not None:
            row["learning_rate"] = optimizer.param_groups[0]["lr"]
            scheduler.step()
        history.append(row)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            row["is_best"] = True
            if run_dir is not None:
                torch.save(model.state_dict(), run_dir / "best.pth")
        if run_dir is not None:
            torch.save(model.state_dict(), run_dir / "last.pth")

        logger.info(
            "[%s] epoch %s/%s train_f1=%.4f val_f1=%.4f",
            model_name,
            epoch,
            epochs,
            train_metrics["macro_f1"],
            val_metrics["macro_f1"],
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)

        if early_stopping is not None and early_stopping.step(val_metrics["macro_f1"]):
            logger.info("[%s] Early stopping en la epoca %s", model_name, epoch)
            break

    return history
```

- [ ] **Step 2: Ejecutar el test de regresión**

Run: `pytest tests/training/test_baseline_regression.py -v`
Expected: 2 passed.

- [ ] **Step 3: Migrar `train_baselines.py` a `fit`**

Borrar `_run_epoch` (líneas 140-183) y `_metrics_from_predictions` (128-137). Añadir el import y reemplazar el bucle de épocas (líneas 291-337) por:

```python
from src.training.loop import fit, run_epoch
```

```python
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_name=model_name,
        run_dir=run_dir,
        on_epoch_end=lambda _epoch, _row: pd.DataFrame(history).to_csv(
            run_dir / "train_history.csv", index=False
        ),
    )
    best_row = max(history, key=lambda item: item["val_macro_f1"])
    best_epoch = best_row["epoch"]
    best_val_macro_f1 = best_row["val_macro_f1"]
```

En la línea 343, `_run_epoch(...)` para test pasa a `run_epoch(...)`.

- [ ] **Step 4: Verificar que baselines sigue corriendo igual**

Run: `pytest tests/training/ -v && python scripts/pipeline/train_baselines.py --models shufflenet_v2_x1_0 --baseline --epochs 1 --max-per-class 20`
Expected: tests passed; el run escribe `train_history.csv` con las 9 columnas y `summary.json`.

- [ ] **Step 5: Commit**

```bash
git add src/training/loop.py scripts/pipeline/train_baselines.py
git commit -m "refactor: extrae el loop de entrenamiento a src/training/loop.py"
```

---

### Task 4: Loss ponderada y label smoothing

**Files:**
- Create: `src/training/losses.py`
- Test: `tests/training/test_losses.py`

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces:
  - `compute_class_weights(labels: pd.Series | list[str], class_to_idx: dict[str,int], strategy: str) -> torch.Tensor | None` — devuelve `None` si `strategy == "none"`; tensor ordenado por índice de clase en caso contrario.
  - `build_criterion(labels, class_to_idx, strategy="none", label_smoothing=0.0, device=None) -> torch.nn.Module`
  - Estrategias válidas: `"none"`, `"inverse"`, `"sqrt_inverse"`.

- [ ] **Step 1: Escribir los tests**

`tests/training/test_losses.py`:

```python
import math

import pytest
import torch

from src.training.losses import build_criterion, compute_class_weights

CLASS_TO_IDX = {"healthy": 0, "common_rust": 1, "potassium_deficiency": 2}
LABELS = ["healthy"] * 12 + ["common_rust"] * 6 + ["potassium_deficiency"] * 2


def test_strategy_none_no_devuelve_pesos():
    assert compute_class_weights(LABELS, CLASS_TO_IDX, "none") is None


def test_inverse_da_ratio_igual_al_desbalance():
    weights = compute_class_weights(LABELS, CLASS_TO_IDX, "inverse")
    assert weights[2] / weights[0] == pytest.approx(6.0, abs=1e-5)


def test_sqrt_inverse_comprime_el_ratio():
    weights = compute_class_weights(LABELS, CLASS_TO_IDX, "sqrt_inverse")
    assert weights[2] / weights[0] == pytest.approx(math.sqrt(6.0), abs=1e-5)


def test_los_pesos_respetan_el_orden_de_class_to_idx():
    weights = compute_class_weights(LABELS, CLASS_TO_IDX, "inverse")
    assert weights[0] < weights[1] < weights[2]
    assert weights.shape == (3,)


def test_estrategia_desconocida_falla():
    with pytest.raises(ValueError, match="Estrategia de pesos desconocida"):
        compute_class_weights(LABELS, CLASS_TO_IDX, "magica")


def test_build_criterion_propaga_label_smoothing():
    criterion = build_criterion(LABELS, CLASS_TO_IDX, "sqrt_inverse", label_smoothing=0.1)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.1
    assert criterion.weight is not None


def test_build_criterion_sin_pesos_equivale_a_crossentropy_plana():
    criterion = build_criterion(LABELS, CLASS_TO_IDX, "none")
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.8, 0.3]])
    targets = torch.tensor([0, 1])
    expected = torch.nn.CrossEntropyLoss()(logits, targets)
    assert criterion(logits, targets) == pytest.approx(expected.item(), abs=1e-6)
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/training/test_losses.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.training.losses'`.

- [ ] **Step 3: Implementar**

`src/training/losses.py`:

```python
"""Construccion de la funcion de perdida del pipeline principal.

El pipeline principal balancea con perdida ponderada en vez de WeightedRandomSampler:
sobre el dataset completo el desbalance llega a 32.9x, donde combinar sampler y pesos
sobre-compensaria el mismo desbalance por dos vias. `sqrt_inverse` es el default porque
comprime el rango de pesos de 32.9x a 5.7x, evitando que las 186 imagenes de train de
potassium_deficiency dominen el gradiente.
"""

from __future__ import annotations

from collections import Counter

import torch

_STRATEGIES = ("none", "inverse", "sqrt_inverse")


def compute_class_weights(
    labels: list[str],
    class_to_idx: dict[str, int],
    strategy: str,
) -> torch.Tensor | None:
    """
    Calcula el peso por clase a partir de la distribucion del split de entrenamiento.

    @param {list[str]} labels Etiquetas de texto de cada muestra de entrenamiento.
    @param {dict[str,int]} class_to_idx Mapeo canonico clase->indice.
    @param {str} strategy Una de "none", "inverse" o "sqrt_inverse".
    @returns {torch.Tensor|None} Pesos ordenados por indice de clase, o None si strategy es "none".
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Estrategia de pesos desconocida: '{strategy}'. Validas: {', '.join(_STRATEGIES)}"
        )
    if strategy == "none":
        return None

    counts = Counter(list(labels))
    missing = set(class_to_idx) - set(counts)
    if missing:
        raise ValueError(f"Clases sin muestras en el split de train: {sorted(missing)}")

    total = sum(counts.values())
    num_classes = len(class_to_idx)
    weights = torch.zeros(num_classes, dtype=torch.float)
    for class_name, class_idx in class_to_idx.items():
        weight = total / (num_classes * counts[class_name])
        weights[class_idx] = weight**0.5 if strategy == "sqrt_inverse" else weight
    return weights


def build_criterion(
    labels: list[str],
    class_to_idx: dict[str, int],
    strategy: str = "none",
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Construye la CrossEntropy del run, con pesos de clase y label smoothing opcionales.

    @param {list[str]} labels Etiquetas del split de entrenamiento.
    @param {dict[str,int]} class_to_idx Mapeo canonico clase->indice.
    @param {str} strategy Estrategia de pesos.
    @param {float} label_smoothing Suavizado de etiquetas; 0.0 lo desactiva.
    @param {torch.device|None} device Dispositivo al que mover los pesos.
    @returns {torch.nn.Module} Criterio listo para el loop.
    """
    weights = compute_class_weights(labels, class_to_idx, strategy)
    if weights is not None and device is not None:
        weights = weights.to(device)
    return torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
```

- [ ] **Step 4: Ejecutar**

Run: `pytest tests/training/test_losses.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/training/losses.py tests/training/test_losses.py
git commit -m "feat: perdida ponderada por clase con label smoothing"
```

---

### Task 5: Scheduler cosine con warmup y early stopping

**Files:**
- Create: `src/training/optim.py`
- Test: `tests/training/test_optim.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `build_scheduler(optimizer, kind, total_epochs, warmup_epochs=0, min_lr=1e-6) -> torch.optim.lr_scheduler.LRScheduler | None` — `kind` en `{"cosine","none"}`.
  - `EarlyStopping(patience: int, min_delta: float = 0.0)` con `.step(metric: float) -> bool` (True = parar) y atributos `.best`, `.num_bad_epochs`.

- [ ] **Step 1: Escribir los tests**

`tests/training/test_optim.py`:

```python
import pytest
import torch

from src.training.optim import EarlyStopping, build_scheduler


def _optimizer(lr: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_kind_none_no_devuelve_scheduler():
    assert build_scheduler(_optimizer(), "none", total_epochs=10) is None


def test_warmup_arranca_por_debajo_del_lr_base():
    optimizer = _optimizer(lr=1e-4)
    build_scheduler(optimizer, "cosine", total_epochs=10, warmup_epochs=3)
    assert optimizer.param_groups[0]["lr"] < 1e-4


def test_warmup_alcanza_el_lr_base_al_terminar():
    optimizer = _optimizer(lr=1e-4)
    scheduler = build_scheduler(optimizer, "cosine", total_epochs=10, warmup_epochs=3)
    for _ in range(3):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-3)


def test_cosine_desciende_tras_el_warmup():
    optimizer = _optimizer(lr=1e-4)
    scheduler = build_scheduler(optimizer, "cosine", total_epochs=10, warmup_epochs=3)
    for _ in range(3):
        scheduler.step()
    peak = optimizer.param_groups[0]["lr"]
    for _ in range(5):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] < peak


def test_early_stopping_dispara_tras_patience_epocas_sin_mejora():
    stopper = EarlyStopping(patience=3)
    assert stopper.step(0.80) is False
    assert stopper.step(0.79) is False
    assert stopper.step(0.78) is False
    assert stopper.step(0.77) is True


def test_early_stopping_se_reinicia_con_una_mejora():
    stopper = EarlyStopping(patience=2)
    stopper.step(0.80)
    stopper.step(0.79)
    assert stopper.step(0.85) is False
    assert stopper.num_bad_epochs == 0
    assert stopper.best == pytest.approx(0.85)


def test_min_delta_ignora_mejoras_insignificantes():
    stopper = EarlyStopping(patience=1, min_delta=0.01)
    stopper.step(0.80)
    assert stopper.step(0.8005) is True
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/training/test_optim.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.training.optim'`.

- [ ] **Step 3: Implementar**

`src/training/optim.py`:

```python
"""Scheduler y parada temprana del pipeline principal.

Cosine con warmup lineal es la eleccion por determinismo: a diferencia de
ReduceLROnPlateau, la curva de lr no depende del ruido de validacion, asi que dos
corridas con la misma semilla recorren exactamente el mismo camino.
"""

from __future__ import annotations

import math

import torch

_SCHEDULERS = ("none", "cosine")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Construye el scheduler por epoca: warmup lineal seguido de cosine annealing.

    @param {torch.optim.Optimizer} optimizer Optimizador cuyo lr se modula.
    @param {str} kind Una de "none" o "cosine".
    @param {int} total_epochs Total de epocas previstas.
    @param {int} warmup_epochs Epocas de subida lineal desde ~0 hasta el lr base.
    @param {float} min_lr Piso del descenso coseno.
    @returns {LRScheduler|None} Scheduler listo para `.step()` por epoca, o None si kind es "none".
    """
    if kind not in _SCHEDULERS:
        raise ValueError(
            f"Scheduler desconocido: '{kind}'. Validos: {', '.join(_SCHEDULERS)}"
        )
    if kind == "none":
        return None

    base_lr = optimizer.param_groups[0]["lr"]
    decay_epochs = max(total_epochs - warmup_epochs, 1)
    floor_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = min((epoch - warmup_epochs) / decay_epochs, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Detiene el entrenamiento tras `patience` epocas sin mejora de la metrica vigilada."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """
        @param {int} patience Epocas toleradas sin mejora antes de parar.
        @param {float} min_delta Mejora minima para considerarse tal.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.num_bad_epochs = 0

    def step(self, metric: float) -> bool:
        """
        Registra la metrica de la epoca y decide si hay que parar.

        @param {float} metric Valor a maximizar (val_macro_f1).
        @returns {bool} True si se agoto la paciencia.
        """
        if metric > self.best + self.min_delta:
            self.best = metric
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience
```

- [ ] **Step 4: Ejecutar**

Run: `pytest tests/training/test_optim.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/training/optim.py tests/training/test_optim.py
git commit -m "feat: scheduler cosine con warmup y early stopping"
```

---

### Task 6: Métricas de evaluación (calibración, environment, N/P/K)

**Files:**
- Create: `src/training/evaluation.py`
- Test: `tests/training/test_evaluation.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `expected_calibration_error(confidences: list[float], correct: list[bool], num_bins: int = 15) -> float`
  - `compute_calibration_metrics(predictions_df: pd.DataFrame, class_to_idx: dict[str,int]) -> dict`
  - `compute_environment_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame`
  - `compute_grouped_metrics(predictions_df: pd.DataFrame, groups: dict[str,str]) -> dict`

`predictions_df` tiene las columnas `image_path,label,pred_label,pred_prob` (más `environment` para el desglose), tal como las escribe `train_baselines.py:352-361`.

- [ ] **Step 1: Escribir los tests**

`tests/training/test_evaluation.py`:

```python
import pandas as pd
import pytest

from src.training.evaluation import (
    compute_environment_metrics,
    compute_grouped_metrics,
    expected_calibration_error,
)

NPK_GROUPS = {
    "nitrogen_deficiency": "nutrient_deficiency",
    "phosphorus_deficiency": "nutrient_deficiency",
    "potassium_deficiency": "nutrient_deficiency",
}


def test_ece_es_cero_con_confianza_perfectamente_calibrada():
    confidences = [1.0, 1.0, 1.0, 1.0]
    correct = [True, True, True, True]
    assert expected_calibration_error(confidences, correct, num_bins=15) == pytest.approx(0.0)


def test_ece_es_uno_con_sobreconfianza_total():
    confidences = [1.0, 1.0]
    correct = [False, False]
    assert expected_calibration_error(confidences, correct, num_bins=15) == pytest.approx(1.0)


def test_ece_calculado_a_mano():
    """Dos bins: uno con confianza 0.9 y 50% de acierto (gap 0.4), otro perfecto."""
    confidences = [0.9, 0.9, 1.0, 1.0]
    correct = [True, False, True, True]
    assert expected_calibration_error(confidences, correct, num_bins=10) == pytest.approx(0.2)


def test_metricas_por_environment_separan_lab_y_real():
    frame = pd.DataFrame(
        {
            "label": ["common_rust"] * 4,
            "pred_label": ["common_rust", "common_rust", "healthy", "healthy"],
            "pred_prob": [0.9, 0.9, 0.8, 0.8],
            "environment": ["lab", "lab", "real", "real"],
        }
    )
    result = compute_environment_metrics(frame).set_index("environment")

    assert result.loc["lab", "accuracy"] == pytest.approx(1.0)
    assert result.loc["real", "accuracy"] == pytest.approx(0.0)
    assert result.loc["lab", "n"] == 2


def test_agrupado_npk_convierte_confusion_interna_en_acierto():
    frame = pd.DataFrame(
        {
            "label": ["potassium_deficiency", "nitrogen_deficiency", "healthy"],
            "pred_label": ["nitrogen_deficiency", "phosphorus_deficiency", "healthy"],
            "pred_prob": [0.99, 0.95, 0.99],
        }
    )
    result = compute_grouped_metrics(frame, NPK_GROUPS)

    assert result["grouped_accuracy"] == pytest.approx(1.0)
    assert result["ungrouped_accuracy"] == pytest.approx(1 / 3)


def test_agrupado_no_oculta_errores_fuera_del_bloque():
    frame = pd.DataFrame(
        {
            "label": ["potassium_deficiency", "healthy"],
            "pred_label": ["healthy", "healthy"],
            "pred_prob": [0.9, 0.9],
        }
    )
    result = compute_grouped_metrics(frame, NPK_GROUPS)
    assert result["grouped_accuracy"] == pytest.approx(0.5)
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/training/test_evaluation.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.training.evaluation'`.

- [ ] **Step 3: Implementar**

`src/training/evaluation.py`:

```python
"""Metricas post-hoc calculadas sobre predictions.csv, sin GPU.

Cubren los tres huecos que dejo la primera fase: calibracion (el reporte documenta que
el modelo se equivoca con 0.914 de confianza media), desglose lab/real (common_rust es
95% lab, lo que abre la sospecha de shortcut learning) y la metrica agrupada N/P/K
(el 97% de los errores de deficiencia se quedan dentro del propio bloque).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def expected_calibration_error(
    confidences: list[float],
    correct: list[bool],
    num_bins: int = 15,
) -> float:
    """
    Calcula el Expected Calibration Error con bins uniformes.

    @param {list[float]} confidences Confianza de la clase predicha por muestra.
    @param {list[bool]} correct True si la prediccion fue correcta.
    @param {int} num_bins Numero de bins uniformes sobre [0, 1].
    @returns {float} Promedio ponderado de |accuracy - confianza| por bin.
    """
    confidence_array = np.asarray(confidences, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    if confidence_array.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence_array > lower) & (confidence_array <= upper)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        error += weight * abs(correct_array[in_bin].mean() - confidence_array[in_bin].mean())
    return float(error)


def compute_calibration_metrics(
    predictions_df: pd.DataFrame,
    class_to_idx: dict[str, int],
) -> dict:
    """
    Resume calibracion: ECE, Brier binario de acierto y confianza media de aciertos vs. fallos.

    @param {pd.DataFrame} predictions_df Columnas label, pred_label y pred_prob.
    @param {dict[str,int]} class_to_idx Mapeo canonico clase->indice.
    @returns {dict} Metricas de calibracion del run.
    """
    correct = (predictions_df["label"] == predictions_df["pred_label"]).tolist()
    confidences = predictions_df["pred_prob"].tolist()

    hits = predictions_df.loc[predictions_df["label"] == predictions_df["pred_label"], "pred_prob"]
    misses = predictions_df.loc[predictions_df["label"] != predictions_df["pred_label"], "pred_prob"]

    confidence_array = np.asarray(confidences, dtype=float)
    correct_array = np.asarray(correct, dtype=float)
    brier = float(np.mean((confidence_array - correct_array) ** 2))

    return {
        "ece": expected_calibration_error(confidences, correct),
        "brier_binary_hit": brier,
        "mean_confidence_hits": float(hits.mean()) if len(hits) else 0.0,
        "mean_confidence_misses": float(misses.mean()) if len(misses) else 0.0,
        "n_hits": int(len(hits)),
        "n_misses": int(len(misses)),
        "num_classes": len(class_to_idx),
    }


def compute_environment_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Desglosa accuracy y macro-F1 por entorno de captura.

    @param {pd.DataFrame} predictions_df Debe incluir la columna environment.
    @returns {pd.DataFrame} Una fila por entorno, con la n visible para leer su fiabilidad.
    """
    rows = []
    for environment, group in predictions_df.groupby("environment"):
        rows.append(
            {
                "environment": environment,
                "n": len(group),
                "accuracy": accuracy_score(group["label"], group["pred_label"]),
                "macro_f1": f1_score(
                    group["label"], group["pred_label"], average="macro", zero_division=0
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_grouped_metrics(predictions_df: pd.DataFrame, groups: dict[str, str]) -> dict:
    """
    Recalcula las metricas colapsando las clases indicadas en una sola categoria.

    @param {pd.DataFrame} predictions_df Columnas label y pred_label.
    @param {dict[str,str]} groups Mapeo clase original -> nombre de la clase agrupada.
    @returns {dict} Metricas antes y despues de agrupar.
    """
    grouped_labels = predictions_df["label"].replace(groups)
    grouped_predictions = predictions_df["pred_label"].replace(groups)

    return {
        "groups": groups,
        "ungrouped_accuracy": accuracy_score(
            predictions_df["label"], predictions_df["pred_label"]
        ),
        "ungrouped_macro_f1": f1_score(
            predictions_df["label"],
            predictions_df["pred_label"],
            average="macro",
            zero_division=0,
        ),
        "grouped_accuracy": accuracy_score(grouped_labels, grouped_predictions),
        "grouped_macro_f1": f1_score(
            grouped_labels, grouped_predictions, average="macro", zero_division=0
        ),
    }
```

- [ ] **Step 4: Ejecutar**

Run: `pytest tests/training/test_evaluation.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/training/evaluation.py tests/training/test_evaluation.py
git commit -m "feat: metricas de calibracion, environment y agrupado N/P/K"
```

---

### Task 7: CLAHE como preprocesamiento opt-in

**Files:**
- Modify: `pyproject.toml:5-15` (mover `opencv-python` a dependencias principales), `config/dataset.yaml` (bloque `clahe`)
- Modify: `src/data/transforms.py:107-138` (`CornTransformFactory`)
- Test: `tests/data/__init__.py`, `tests/data/test_clahe.py`

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces: `CornCLAHETransform(clip_limit: float = 2.0, tile_grid: int = 8)` — callable `PIL.Image -> PIL.Image`; `CornTransformFactory(config_path, target_size, clahe=False)` inyecta la transformada **antes** del `Resize` en los cuatro pipelines.

CLAHE es preprocesamiento determinista, **no** augmentation: se aplica igual en train, val, test e inference. Aplicarlo solo en train produciría desajuste train/test garantizado.

- [ ] **Step 1: Escribir los tests**

`tests/data/__init__.py` vacío. `tests/data/test_clahe.py`:

```python
import numpy as np
import pytest
from PIL import Image

from src.data.transforms import CornCLAHETransform, CornTransformFactory


@pytest.fixture
def leaf_image() -> Image.Image:
    """Imagen con iluminacion irregular: mitad sobreexpuesta, mitad en sombra."""
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 60, size=(64, 64, 3), dtype=np.uint8)
    pixels[:32] = np.clip(pixels[:32].astype(int) + 180, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def test_clahe_preserva_el_tono(leaf_image):
    """El hue es senal diagnostica en las deficiencias: no puede desplazarse."""
    import cv2

    original = np.array(leaf_image)
    processed = np.array(CornCLAHETransform()(leaf_image))

    hue_original = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)[..., 0].astype(float)
    hue_processed = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)[..., 0].astype(float)
    shift = np.abs(((hue_processed - hue_original + 90) % 180) - 90).mean()

    assert shift < 2.0, f"desplazamiento de hue demasiado alto: {shift:.2f} grados"


def test_clahe_aumenta_el_contraste_local(leaf_image):
    import cv2

    original = cv2.cvtColor(np.array(leaf_image), cv2.COLOR_RGB2LAB)[..., 0]
    processed = cv2.cvtColor(np.array(CornCLAHETransform()(leaf_image)), cv2.COLOR_RGB2LAB)[..., 0]
    assert processed.std() > original.std()


def test_clahe_es_determinista(leaf_image):
    transform = CornCLAHETransform()
    first = np.array(transform(leaf_image))
    second = np.array(transform(leaf_image))
    assert np.array_equal(first, second)


def test_clahe_devuelve_rgb_del_mismo_tamano(leaf_image):
    processed = CornCLAHETransform()(leaf_image)
    assert processed.mode == "RGB"
    assert processed.size == leaf_image.size


def test_factory_sin_clahe_no_lo_inyecta():
    factory = CornTransformFactory(target_size=(32, 32), clahe=False)
    for stage in ("train", "minority", "val", "test"):
        names = [type(t).__name__ for t in factory.get_pipeline(stage).transforms]
        assert "CornCLAHETransform" not in names


def test_factory_con_clahe_lo_inyecta_en_los_cuatro_pipelines():
    """CLAHE es preprocesamiento, no augmentation: va tambien en val y test."""
    factory = CornTransformFactory(target_size=(32, 32), clahe=True)
    for stage in ("train", "minority", "val", "test"):
        names = [type(t).__name__ for t in factory.get_pipeline(stage).transforms]
        assert names[0] == "CornCLAHETransform", f"CLAHE no va primero en el pipeline {stage}"
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/data/test_clahe.py -v`
Expected: FAIL con `ImportError: cannot import name 'CornCLAHETransform'`.

- [ ] **Step 3: Promover opencv a dependencia principal**

En `pyproject.toml`, añadir a `dependencies` (después de la línea 14, `tqdm`):

```toml
    "opencv-python>=4.8,<5.0",
```

Run: `pip install -e ".[dev]"`

- [ ] **Step 4: Añadir el bloque de config**

En `config/dataset.yaml`, después del bloque `augmentation`:

```yaml
clahe:
  clip_limit: 2.0
  tile_grid: 8
```

- [ ] **Step 5: Implementar la transformada**

En `src/data/transforms.py`, añadir tras los imports (`import cv2`, `import numpy as np`, `from PIL import Image`):

```python
class CornCLAHETransform:
    """
    Ecualizacion adaptativa de histograma sobre el canal L de LAB.

    Solo toca la luminancia: ecualizar los tres canales RGB por separado desplaza el
    tono, y el color es la senal diagnostica de las deficiencias nutricionales, donde
    la clorosis amarillenta es justamente lo que distingue la clase. Es preprocesamiento
    determinista, no augmentation, asi que se aplica igual en train, val, test e inferencia.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: int = 8):
        """
        @param {float} clip_limit Umbral de recorte; valores altos amplifican ruido.
        @param {int} tile_grid Lado de la grilla de tiles.
        """
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        @param {Image.Image} image Imagen RGB de entrada.
        @returns {Image.Image} Imagen RGB con la luminancia ecualizada.
        """
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        lightness, green_red, blue_yellow = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=(self.tile_grid, self.tile_grid)
        )
        merged = cv2.merge((clahe.apply(lightness), green_red, blue_yellow))
        return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))
```

- [ ] **Step 6: Inyectarla en la factory**

Reemplazar `CornTransformFactory.__init__` y `get_pipeline` (líneas 112-138):

```python
    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        target_size: tuple[int, int] | None = None,
        clahe: bool = False,
    ):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # target_size es [alto, ancho] - convención (h, w) de torchvision (ver CLAUDE.md)
        if target_size is None:
            height, width = config["dataset"]["target_size"]
            target_size = (height, width)
        self.target_size = target_size

        clahe_config = config.get("clahe", {})
        self.clahe_transform = (
            CornCLAHETransform(
                clip_limit=float(clahe_config.get("clip_limit", 2.0)),
                tile_grid=int(clahe_config.get("tile_grid", 8)),
            )
            if clahe
            else None
        )

    def get_pipeline(self, stage: str) -> T.Compose:
        """Retorna el pipeline de transformación correspondiente a la etapa."""
        if stage.lower() == "train":
            pipeline = CornTrainingTransforms(self.target_size).create_transforms()
        elif stage.lower() == "minority":
            pipeline = CornMinorityTransforms(self.target_size).create_transforms()
        elif stage.lower() in ["val", "test", "inference"]:
            pipeline = CornValidationTransforms(self.target_size).create_transforms()
        else:
            raise ValueError(
                f"Etapa de pipeline desconocida: '{stage}'. "
                "Use 'train', 'minority', 'val', 'test' o 'inference'."
            )

        if self.clahe_transform is None:
            return pipeline
        return T.Compose([self.clahe_transform, *pipeline.transforms])
```

- [ ] **Step 7: Ejecutar**

Run: `pytest tests/data/test_clahe.py -v`
Expected: 6 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml config/dataset.yaml src/data/transforms.py tests/data/
git commit -m "feat: CLAHE opt-in sobre el canal L de LAB como preprocesamiento"
```

---

### Task 8: Escritura de artefactos

**Files:**
- Create: `src/training/artifacts.py`
- Modify: `scripts/pipeline/train_baselines.py:186-241` (borrar `_write_test_outputs` y `_write_summary`, importar de `artifacts`)
- Test: `tests/training/test_artifacts.py`

**Interfaces:**
- Consumes: `compute_calibration_metrics`, `compute_environment_metrics`, `compute_grouped_metrics` (Tarea 6).
- Produces:
  - `write_test_outputs(run_dir, idx_to_class, labels, predictions) -> None`
  - `write_predictions_csv(run_dir, test_dataset, idx_to_class, predictions, probs) -> pd.DataFrame`
  - `write_extended_metrics(run_dir, predictions_df, class_to_idx, npk_groups) -> None` — escribe `test_calibration.json`, `test_by_environment.csv` y `test_grouped_metrics.json`
  - `write_summary(run_dir, payload: dict) -> None`
  - Constante `NPK_GROUPS: dict[str,str]`

- [ ] **Step 1: Escribir los tests**

`tests/training/test_artifacts.py`:

```python
import json

import pandas as pd

from src.training.artifacts import NPK_GROUPS, write_extended_metrics, write_summary


def _predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": [f"img_{i}.png" for i in range(4)],
            "label": ["healthy", "potassium_deficiency", "common_rust", "healthy"],
            "pred_label": ["healthy", "nitrogen_deficiency", "common_rust", "healthy"],
            "pred_prob": [0.99, 0.95, 0.88, 0.97],
            "environment": ["real", "real", "lab", "lab"],
        }
    )


def test_escribe_los_tres_artefactos(tmp_path):
    write_extended_metrics(
        tmp_path, _predictions_frame(), {"healthy": 0, "common_rust": 1}, NPK_GROUPS
    )

    assert (tmp_path / "test_calibration.json").exists()
    assert (tmp_path / "test_by_environment.csv").exists()
    assert (tmp_path / "test_grouped_metrics.json").exists()


def test_calibracion_separa_confianza_de_aciertos_y_fallos(tmp_path):
    write_extended_metrics(
        tmp_path, _predictions_frame(), {"healthy": 0, "common_rust": 1}, NPK_GROUPS
    )
    payload = json.loads((tmp_path / "test_calibration.json").read_text())

    assert payload["n_hits"] == 3
    assert payload["n_misses"] == 1
    assert payload["mean_confidence_misses"] > 0.9


def test_environment_incluye_la_n_por_fila(tmp_path):
    write_extended_metrics(
        tmp_path, _predictions_frame(), {"healthy": 0, "common_rust": 1}, NPK_GROUPS
    )
    frame = pd.read_csv(tmp_path / "test_by_environment.csv")

    assert set(frame["environment"]) == {"lab", "real"}
    assert frame["n"].sum() == 4


def test_agrupado_npk_mejora_sobre_el_desagrupado(tmp_path):
    write_extended_metrics(
        tmp_path, _predictions_frame(), {"healthy": 0, "common_rust": 1}, NPK_GROUPS
    )
    payload = json.loads((tmp_path / "test_grouped_metrics.json").read_text())

    assert payload["grouped_accuracy"] > payload["ungrouped_accuracy"]


def test_summary_es_json_valido(tmp_path):
    write_summary(tmp_path, {"model": "shufflenet_v2_x1_0", "best_epoch": 7})
    assert json.loads((tmp_path / "summary.json").read_text())["best_epoch"] == 7
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/training/test_artifacts.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.training.artifacts'`.

- [ ] **Step 3: Implementar**

`src/training/artifacts.py`:

```python
"""Escritura de los artefactos de un run de entrenamiento.

Compartido por baselines y pipeline principal para que ambos produzcan el mismo
esquema de salida y explain_report.py pueda leer los dos indistintamente.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.training.evaluation import (
    compute_calibration_metrics,
    compute_environment_metrics,
    compute_grouped_metrics,
)

NPK_GROUPS: dict[str, str] = {
    "nitrogen_deficiency": "nutrient_deficiency",
    "phosphorus_deficiency": "nutrient_deficiency",
    "potassium_deficiency": "nutrient_deficiency",
}


def write_test_outputs(
    run_dir: Path,
    idx_to_class: dict[int, str],
    labels: list[int],
    predictions: list[int],
) -> None:
    """
    Escribe el classification report y la matriz de confusion del split de test.

    @param {Path} run_dir Directorio del run.
    @param {dict[int,str]} idx_to_class Mapeo indice->clase.
    @param {list[int]} labels Etiquetas reales codificadas.
    @param {list[int]} predictions Predicciones codificadas.
    """
    target_ids = sorted(idx_to_class)
    target_names = [idx_to_class[idx] for idx in target_ids]

    report = classification_report(
        labels,
        predictions,
        labels=target_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(run_dir / "test_classification_report.csv")

    matrix = confusion_matrix(labels, predictions, labels=target_ids)
    pd.DataFrame(matrix, index=target_names, columns=target_names).to_csv(
        run_dir / "test_confusion_matrix.csv"
    )


def write_predictions_csv(
    run_dir: Path,
    test_dataset,
    idx_to_class: dict[int, str],
    predictions: list[int],
    probs: list[float],
) -> pd.DataFrame:
    """
    Escribe predictions.csv con una fila por imagen de test.

    @param {Path} run_dir Directorio del run.
    @param {CornDataset} test_dataset Dataset de test, fuente de rutas y etiquetas.
    @param {dict[int,str]} idx_to_class Mapeo indice->clase.
    @param {list[int]} predictions Predicciones codificadas.
    @param {list[float]} probs Confianza de la clase predicha.
    @returns {pd.DataFrame} El propio dataframe escrito.
    """
    frame = pd.DataFrame(
        {
            "image_path": test_dataset.data_frame["image_path"].tolist(),
            "label": test_dataset.data_frame["label"].tolist(),
            "pred_label": [idx_to_class[p] for p in predictions],
            "pred_prob": probs,
        }
    )
    if "environment" in test_dataset.data_frame.columns:
        frame["environment"] = test_dataset.data_frame["environment"].tolist()
    frame.to_csv(run_dir / "predictions.csv", index=False)
    return frame


def write_extended_metrics(
    run_dir: Path,
    predictions_df: pd.DataFrame,
    class_to_idx: dict[str, int],
    npk_groups: dict[str, str] = NPK_GROUPS,
) -> None:
    """
    Escribe calibracion, desglose por environment y metricas agrupadas N/P/K.

    @param {Path} run_dir Directorio del run.
    @param {pd.DataFrame} predictions_df Salida de write_predictions_csv.
    @param {dict[str,int]} class_to_idx Mapeo canonico clase->indice.
    @param {dict[str,str]} npk_groups Mapeo de clases a agrupar.
    """
    calibration = compute_calibration_metrics(predictions_df, class_to_idx)
    (run_dir / "test_calibration.json").write_text(json.dumps(calibration, indent=2))

    if "environment" in predictions_df.columns:
        compute_environment_metrics(predictions_df).to_csv(
            run_dir / "test_by_environment.csv", index=False
        )

    grouped = compute_grouped_metrics(predictions_df, npk_groups)
    (run_dir / "test_grouped_metrics.json").write_text(json.dumps(grouped, indent=2))


def write_summary(run_dir: Path, payload: dict) -> None:
    """
    Persiste summary.json, fuente de verdad del run para los scripts de explicabilidad.

    @param {Path} run_dir Directorio del run.
    @param {dict} payload Configuracion y metricas del run.
    """
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
```

- [ ] **Step 4: Ejecutar**

Run: `pytest tests/training/test_artifacts.py -v`
Expected: 5 passed.

- [ ] **Step 5: Migrar `train_baselines.py`**

Borrar `_write_test_outputs` (líneas 186-208) y `_write_summary` (211-241). Añadir el import y sustituir las llamadas: `_write_test_outputs(...)` → `write_test_outputs(...)`; el bloque de `predictions_df` (352-361) → `write_predictions_csv(...)`; `_write_summary(...)` → `write_summary(run_dir, {...})` con el mismo payload que construía antes. Añadir tras `write_predictions_csv`:

```python
    write_extended_metrics(run_dir, predictions_df, class_to_idx, NPK_GROUPS)
```

- [ ] **Step 6: Verificar que baselines sigue igual**

Run: `pytest tests/ -v && python scripts/pipeline/train_baselines.py --models shufflenet_v2_x1_0 --baseline --epochs 1 --max-per-class 20`
Expected: todos los tests pasan; el run escribe además los 3 artefactos nuevos.

- [ ] **Step 7: Commit**

```bash
git add src/training/artifacts.py tests/training/test_artifacts.py scripts/pipeline/train_baselines.py
git commit -m "refactor: centraliza la escritura de artefactos en src/training/artifacts.py"
```

---

### Task 9: Ensamblar `train.py`

**Files:**
- Modify: `scripts/pipeline/train.py` (reescritura completa: hoy termina en un `TODO`)
- Modify: `Makefile:49-50` (target `train`)

**Interfaces:**
- Consumes: `fit`, `run_epoch` (T3); `build_criterion` (T4); `build_scheduler`, `EarlyStopping` (T5); `write_test_outputs`, `write_predictions_csv`, `write_extended_metrics`, `write_summary`, `NPK_GROUPS` (T8); `CornTransformFactory(clahe=...)` (T7).
- Produces: CLI del pipeline principal con `--models`, `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--scheduler`, `--warmup-epochs`, `--min-lr`, `--patience`, `--class-weights`, `--label-smoothing`, `--clip-grad-norm`, `--clahe`, `--no-pretrained`, `--num-workers`, `--splits-dir`, `--output-dir`, `--config`.

- [ ] **Step 1: Reescribir `scripts/pipeline/train.py`**

```python
"""Pipeline principal de entrenamiento.

Comparte toda la infraestructura de datos y modelos con train_baselines.py; lo que
cambia es cuanto se afina el loop. Sobre el dataset completo el desbalance llega a
32.9x, asi que el balanceo se hace con perdida ponderada y el WeightedRandomSampler
queda desactivado: combinarlos sobre-compensaria el mismo desbalance por dos vias, y
sin `replacement=True` cada epoca ve el 100% de las imagenes unicas en vez del ~63%.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, get_output_root, set_global_seed
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.models import build_model, list_models, resolve_input_size
from src.training.artifacts import (
    NPK_GROUPS,
    write_extended_metrics,
    write_predictions_csv,
    write_summary,
    write_test_outputs,
)
from src.training.common import (
    build_run_dir,
    generate_run_id,
    resolve_model_names,
    select_device,
    update_latest_pointer,
    worker_init_fn,
)
from src.training.loop import fit, run_epoch
from src.training.losses import build_criterion
from src.training.optim import EarlyStopping, build_scheduler
from src.models.registry import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el pipeline principal.")
    parser.add_argument("--models", nargs="+", default=["shufflenet_v2_x1_0"],
                        help=f"Modelos a entrenar. Disponibles: {list_models()}")
    parser.add_argument("--splits-dir", default=None, dest="splits_dir")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="learning_rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=3, dest="warmup_epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6, dest="min_lr")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--class-weights", choices=["sqrt_inverse", "inverse", "none"],
                        default="sqrt_inverse", dest="class_weights")
    parser.add_argument("--label-smoothing", type=float, default=0.1, dest="label_smoothing")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, dest="clip_grad_norm")
    parser.add_argument("--clahe", action="store_true",
                        help="Aplica CLAHE como preprocesamiento en los cuatro pipelines.")
    parser.add_argument("--no-pretrained", action="store_true", dest="no_pretrained")
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs debe ser mayor o igual a 1.")

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["dataset"]["seed"]
    set_global_seed(seed)

    model_names = resolve_model_names(args.models, MODEL_REGISTRY)
    output_root = get_output_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else output_root / "splits" / "seed_42"
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "main"

    if not splits_dir.exists():
        raise SystemExit(
            f"El directorio de splits no existe: {splits_dir}\n"
            "Genera los splits primero con: make splits"
        )

    device = select_device()
    base_target_size = tuple(cfg["dataset"]["target_size"])
    logger.info("Modelos a entrenar: %s", model_names)
    logger.info("Balanceo: perdida '%s' (sampler desactivado)", args.class_weights)
    if args.clahe:
        logger.info("CLAHE activo en los cuatro pipelines de transformacion")

    for model_name in model_names:
        target_size = resolve_input_size(model_name, base_target_size)
        factory = CornTransformFactory(
            config_path=str(config_path), target_size=target_size, clahe=args.clahe
        )

        train_dataset = CornDataset(
            csv_path=str(splits_dir / "train.csv"),
            config_path=str(config_path),
            transform=factory.get_pipeline("train"),
            minority_transform=factory.get_pipeline("minority"),
        )
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = train_dataset.idx_to_class
        val_dataset = CornDataset(
            csv_path=str(splits_dir / "val.csv"),
            config_path=str(config_path),
            transform=factory.get_pipeline("val"),
            class_to_idx=class_to_idx,
        )
        test_dataset = CornDataset(
            csv_path=str(splits_dir / "test.csv"),
            config_path=str(config_path),
            transform=factory.get_pipeline("test"),
            class_to_idx=class_to_idx,
        )

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=pin_memory)

        run_id = generate_run_id()
        run_dir = build_run_dir(output_dir, model_name, run_id)
        logger.info("[%s] Checkpoints en %s", model_name, run_dir)

        model = build_model(
            model_name, num_classes=len(class_to_idx), pretrained=not args.no_pretrained
        ).to(device)
        criterion = build_criterion(
            labels=train_dataset.data_frame["label"].tolist(),
            class_to_idx=class_to_idx,
            strategy=args.class_weights,
            label_smoothing=args.label_smoothing,
            device=device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = build_scheduler(
            optimizer,
            kind=args.scheduler,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )
        early_stopping = EarlyStopping(patience=args.patience)

        history: list[dict] = []

        def _persist_history(_epoch: int, _row: dict) -> None:
            pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            model_name=model_name,
            run_dir=run_dir,
            scheduler=scheduler,
            early_stopping=early_stopping,
            clip_grad_norm=args.clip_grad_norm,
            on_epoch_end=_persist_history,
        )
        pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

        best_row = max(history, key=lambda item: item["val_macro_f1"])
        best_path = run_dir / "best.pth"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))

        test_metrics, labels, predictions, probs = run_epoch(
            model, test_loader, criterion, device, desc=f"{model_name} test"
        )
        write_test_outputs(run_dir, idx_to_class, labels, predictions)
        predictions_df = write_predictions_csv(
            run_dir, test_dataset, idx_to_class, predictions, probs
        )
        write_extended_metrics(run_dir, predictions_df, class_to_idx, NPK_GROUPS)
        write_summary(
            run_dir,
            {
                "pipeline": "main",
                "model": model_name,
                "run_id": run_id,
                "num_classes": len(class_to_idx),
                "class_to_idx": class_to_idx,
                "image_size": list(target_size),
                "splits_dir": str(splits_dir),
                "epochs_requested": args.epochs,
                "epochs_run": len(history),
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "warmup_epochs": args.warmup_epochs,
                "min_lr": args.min_lr,
                "patience": args.patience,
                "class_weights": args.class_weights,
                "label_smoothing": args.label_smoothing,
                "clip_grad_norm": args.clip_grad_norm,
                "clahe": args.clahe,
                "sampler": None,
                "pretrained": not args.no_pretrained,
                "best_epoch": best_row["epoch"],
                "best_val_macro_f1": best_row["val_macro_f1"],
                "test": test_metrics,
            },
        )
        update_latest_pointer(output_dir, model_name, run_id)
        logger.info("[%s] Test macro_f1=%.4f", model_name, test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Actualizar el Makefile**

Reemplazar el target `train` (líneas 49-50):

```makefile
train:
	$(PYTHON) scripts/pipeline/train.py --models $(MODELS) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(CLAHE),--clahe,) \
		$(if $(CLASS_WEIGHTS),--class-weights $(CLASS_WEIGHTS),)
```

Verificar que `MODELS` tiene default en el Makefile; si no, añadir `MODELS ?= shufflenet_v2_x1_0` junto a las demás variables.

- [ ] **Step 3: Smoke run**

Run: `python scripts/pipeline/train.py --models shufflenet_v2_x1_0 --epochs 2 --splits-dir outputs/splits/seed_42_baseline --no-pretrained --num-workers 0`
Expected: escribe en `outputs/main/shufflenet_v2_x1_0/<run_id>/` los artefactos `best.pth`, `last.pth`, `train_history.csv` (con columna `learning_rate`), `predictions.csv`, `test_calibration.json`, `test_by_environment.csv`, `test_grouped_metrics.json`, `summary.json`, y `latest.json` en el directorio del modelo.

- [ ] **Step 4: Verificar los artefactos**

Run:
```bash
python -c "
import json, glob, pandas as pd
run = sorted(glob.glob('outputs/main/shufflenet_v2_x1_0/*/'))[-1]
print('summary:', json.load(open(run+'summary.json'))['class_weights'])
print('calibration:', json.load(open(run+'test_calibration.json'))['ece'])
print(pd.read_csv(run+'test_by_environment.csv'))
print('grouped:', json.load(open(run+'test_grouped_metrics.json'))['grouped_accuracy'])
"
```
Expected: `class_weights` es `sqrt_inverse`, ECE es un float en [0,1], la tabla de environment tiene filas `lab` y `real` con su `n`, y `grouped_accuracy` es un float.

- [ ] **Step 5: Lint**

Run: `ruff check scripts/pipeline/train.py src/training/ src/data/transforms.py && ruff format --check scripts/pipeline/train.py src/training/`
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/train.py Makefile
git commit -m "feat: implementa el loop del pipeline principal de entrenamiento"
```

---

### Task 10: Ejecución en Modal y documentación

**Files:**
- Modify: `scripts/modal/train.py` (añadir `train_main`)
- Modify: `Makefile` (target `modal-train`)
- Modify: `CLAUDE.md:34` (línea del pipeline principal), `CLAUDE.md` sección "Comandos frecuentes"

**Interfaces:**
- Consumes: la CLI de `train.py` (T9).
- Produces: `train_main(models, epochs, clahe, class_weights, ...)` como función Modal; target `make modal-train`.

- [ ] **Step 1: Añadir la función Modal**

En `scripts/modal/train.py`, tras `train_baselines`, replicando su patrón de Volumes y secrets:

```python
@app.function(
    gpu="A10",
    cpu=4.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    # El pipeline principal corre sobre las 31 623 imagenes (3.2x el perfil capado) con
    # hasta 60 epocas: ~2-4 h por corrida. El techo de 8 h deja margen para la variante
    # con CLAHE sin arriesgar una muerte por timeout a mitad de entrenamiento.
    timeout=8 * 3600,
)
def train_main(
    models: str = "shufflenet_v2_x1_0",
    epochs: int = 60,
    batch_size: int = 0,
    learning_rate: float = 0.0,
    class_weights: str = "",
    label_smoothing: float = -1.0,
    patience: int = 0,
    clahe: bool = False,
    no_pretrained: bool = False,
    num_workers: int = 0,
) -> None:
    """
    Entrena el pipeline principal en GPU, persistiendo en el Volume corn-outputs.

    @param {str} models Modelos separados por espacio.
    @param {int} epochs Techo de epocas; el early stopping puede cortar antes.
    @param {int} batch_size 0 usa el default del script.
    @param {float} learning_rate 0.0 usa el default del script.
    @param {str} class_weights Estrategia de pesos; "" usa el default (sqrt_inverse).
    @param {float} label_smoothing Negativo usa el default del script.
    @param {int} patience 0 usa el default del script.
    @param {bool} clahe Activa CLAHE como preprocesamiento.
    @param {bool} no_pretrained Entrena desde cero.
    @param {int} num_workers 0 usa el default del script.
    """
    dataset_vol.reload()
    command = [
        sys.executable,
        "scripts/pipeline/train.py",
        "--models",
        *models.split(),
        "--epochs",
        str(epochs),
    ]
    if batch_size:
        command += ["--batch-size", str(batch_size)]
    if learning_rate:
        command += ["--learning-rate", str(learning_rate)]
    if class_weights:
        command += ["--class-weights", class_weights]
    if label_smoothing >= 0:
        command += ["--label-smoothing", str(label_smoothing)]
    if patience:
        command += ["--patience", str(patience)]
    if num_workers:
        command += ["--num-workers", str(num_workers)]
    if clahe:
        command.append("--clahe")
    if no_pretrained:
        command.append("--no-pretrained")

    subprocess.run(command, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()
```

- [ ] **Step 2: Añadir el target al Makefile**

Junto a `modal-train-baselines`, y añadir `modal-train` a la lista `.PHONY` de la línea 35:

```makefile
modal-train:
	$(MODAL) run scripts/modal/train.py::train_main --models "$(MODELS)" --epochs "$(EPOCHS)" \
		$(if $(CLAHE),--clahe,) \
		$(if $(CLASS_WEIGHTS),--class-weights "$(CLASS_WEIGHTS)",)
```

- [ ] **Step 3: Actualizar `CLAUDE.md`**

Reemplazar la línea del pipeline principal (línea 34):

```markdown
- **Principal (`train.py`):** comparte la infraestructura de datos/modelos con baselines. Entrena una arquitectura (default `shufflenet_v2_x1_0`) sobre el dataset completo con pérdida ponderada (`sqrt_inverse`) + label smoothing, cosine con warmup, early stopping y gradient clipping. El `WeightedRandomSampler` va **desactivado**: con desbalance de 32.9x, sampler + pérdida ponderada sobre-compensaría por dos vías. Escribe además `test_calibration.json`, `test_by_environment.csv` y `test_grouped_metrics.json`. CLAHE disponible con `--clahe` (opt-in, ver spec).
```

En "Comandos frecuentes", reemplazar la línea de `make train`:

```bash
make train [MODELS=<nombre> EPOCHS=<n> CLAHE=1 CLASS_WEIGHTS=<estrategia>]   # pipeline principal
make modal-train [MODELS=<nombre> EPOCHS=<n> CLAHE=1]                        # el mismo, en GPU
```

- [ ] **Step 4: Verificar la suite completa**

Run: `pytest tests/ -v && ruff check src/ scripts/pipeline/train.py`
Expected: todos los tests pasan, lint limpio.

- [ ] **Step 5: Commit**

```bash
git add scripts/modal/train.py Makefile CLAUDE.md
git commit -m "feat: ejecucion del pipeline principal en Modal y documentacion"
```

---

### Task 11: Corrida A/B de CLAHE

Cierra la validación que el spec §7 dejó planteada. Requiere GPU y las dos corridas completas — es la única tarea que no es de código.

**Files:**
- Create: `experiments/clahe/RESULTS.md`

**Interfaces:**
- Consumes: `make modal-train` (T10).
- Produces: `experiments/clahe/RESULTS.md` con el veredicto y la evidencia.

- [ ] **Step 1: Generar los splits completos**

Run: `make splits`
Expected: `outputs/splits/seed_42/` con 31 623 imágenes repartidas 70/15/15.

- [ ] **Step 2: Corrida base (sin CLAHE)**

Run: `make modal-train MODELS=shufflenet_v2_x1_0 EPOCHS=60`
Expected: ~2-4 h. Anotar el `run_id`.

- [ ] **Step 3: Corrida con CLAHE**

Run: `make modal-train MODELS=shufflenet_v2_x1_0 EPOCHS=60 CLAHE=1`
Expected: ~2-4 h. Anotar el `run_id`.

- [ ] **Step 4: Descargar y comparar**

Run: `make modal-pull`, luego:

```bash
python -c "
import json, glob, pandas as pd
for run in sorted(glob.glob('outputs/main/shufflenet_v2_x1_0/*/')):
    summary = json.load(open(run+'summary.json'))
    calibration = json.load(open(run+'test_calibration.json'))
    grouped = json.load(open(run+'test_grouped_metrics.json'))
    print(f\"clahe={summary['clahe']} epocas={summary['epochs_run']} \"
          f\"macro_f1={summary['test']['macro_f1']:.4f} ece={calibration['ece']:.4f} \"
          f\"agrupado={grouped['grouped_macro_f1']:.4f}\")
    print(pd.read_csv(run+'test_by_environment.csv').to_string(index=False))
"
```

- [ ] **Step 5: Escribir el veredicto**

Crear `experiments/clahe/RESULTS.md` con: tabla comparativa (macro-F1, ECE, macro-F1 agrupado, épocas hasta early stopping), el desglose lab/real de ambas corridas, y el veredicto contra la **predicción falsable del spec**: *si CLAHE ayuda, la mejora debe concentrarse en `real` y no en `lab`*. Si la mejora aparece en `lab`, o es uniforme, CLAHE no está haciendo lo que la hipótesis predice y no debe adoptarse por defecto.

Incluir explícitamente el F1 de `common_rust` en `real` (con su `n`), por el riesgo de que CLAHE amplifique el fondo tanto como la lesión.

- [ ] **Step 6: Commit**

```bash
git add experiments/clahe/RESULTS.md
git commit -m "docs: resultados de la corrida A/B de CLAHE"
```

---

## Self-Review

**Cobertura del spec:**

| Sección del spec | Tarea |
|---|---|
| §4 `loop.py` | T3 |
| §4 `optim.py` | T5 |
| §4 `losses.py` | T4 |
| §4 `evaluation.py` | T6 |
| §4 `artifacts.py` | T8 |
| §5 config del loop | T9 |
| §6 métricas nuevas | T6 + T8 |
| §7 CLAHE | T7 (código) + T11 (validación) |
| §8.1 regresión baselines | T2 |
| §8.2 unitarios sin GPU | T4, T5, T6, T7 |
| §8.3 smoke run | T9 step 3 |
| §9 Modal | T10 |

Gap encontrado y cubierto: el spec asume infraestructura de tests que no existía (no hay `tests/` ni pytest) — añadida como T1.

**Consistencia de tipos:** `fit()` y `run_epoch()` tienen la misma firma en T3, T9 y el test de T2. `write_extended_metrics(run_dir, predictions_df, class_to_idx, npk_groups)` mantiene el orden posicional en T8 y T9. `EarlyStopping.step()` devuelve `bool` en T5 y se consume como tal en `fit()`. `compute_class_weights` devuelve `None` con `strategy="none"`, y `build_criterion` lo propaga a `weight=None`.
