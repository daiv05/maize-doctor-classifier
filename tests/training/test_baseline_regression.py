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

    from src.training.loop import fit

    from src.config import set_global_seed
    from src.data.dataset import CornDataset
    from src.data.transforms import CornTransformFactory
    from src.models import build_model

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
