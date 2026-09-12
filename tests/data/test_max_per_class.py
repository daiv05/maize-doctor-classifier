import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.data.dataset import CornDataset, compute_minority_classes

_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")


@pytest.fixture
def split_csv(tmp_path):
    """Split desbalanceado donde el tope comprimiria los ratios por debajo del umbral."""
    counts = {
        "healthy": 6000,
        "northern_corn_leaf_blight": 4800,
        "lethal_necrosis": 4500,
        "fall_armyworm": 3400,
        "common_rust": 1600,
        "gray_leaf_spot": 1350,
        "phosphorus_deficiency": 650,
        "nitrogen_deficiency": 590,
        "potassium_deficiency": 430,
    }
    rows = [
        {"image_path": f"clean/{label}/real/{label}_src_real_{index}.jpg",
         "label": label, "environment": "real"}
        for label, total in counts.items()
        for index in range(total)
    ]
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_el_tope_reduce_volumen(split_csv):
    """El cupo se aplica por clase, no sobre el total."""
    sin_tope = CornDataset(csv_path=str(split_csv), config_path=_CONFIG)
    con_tope = CornDataset(csv_path=str(split_csv), config_path=_CONFIG, max_per_class=1500)

    assert len(con_tope) < len(sin_tope)
    assert con_tope.data_frame.label.value_counts().max() == 1500


def test_el_tope_no_desactiva_el_augmentation_de_minoritarias(split_csv):
    """Recortar antes de derivar las minoritarias vaciaria el conjunto en silencio.

    Con tope 1500 sobre este split ninguna clase supera el umbral de 4.0, asi que si el
    submuestreo se aplicara primero nadie recibiria el pipeline agresivo.
    """
    sin_tope = CornDataset(csv_path=str(split_csv), config_path=_CONFIG)
    con_tope = CornDataset(csv_path=str(split_csv), config_path=_CONFIG, max_per_class=1500)

    assert con_tope.minority_classes == sin_tope.minority_classes
    assert con_tope.minority_classes

    capado = con_tope.data_frame.label.value_counts()
    assert not compute_minority_classes(con_tope.data_frame, 4.0), (
        "el split capado no tiene minoritarias por si mismo: la politica debe venir del "
        f"split original, no de este reparto {capado.to_dict()}"
    )


def test_sin_tope_el_dataset_queda_intacto(split_csv):
    """El valor por defecto no altera el comportamiento vigente."""
    base = CornDataset(csv_path=str(split_csv), config_path=_CONFIG)
    cero = CornDataset(csv_path=str(split_csv), config_path=_CONFIG, max_per_class=0)
    nulo = CornDataset(csv_path=str(split_csv), config_path=_CONFIG, max_per_class=None)

    assert len(cero) == len(base)
    assert len(nulo) == len(base)


def test_el_tope_es_reproducible(split_csv):
    """Misma semilla, mismo subconjunto: las corridas deben ser comparables entre si."""
    primero = CornDataset(csv_path=str(split_csv), config_path=_CONFIG,
                          max_per_class=1500, seed=7)
    segundo = CornDataset(csv_path=str(split_csv), config_path=_CONFIG,
                          max_per_class=1500, seed=7)
    distinto = CornDataset(csv_path=str(split_csv), config_path=_CONFIG,
                           max_per_class=1500, seed=8)

    assert primero.data_frame.image_path.tolist() == segundo.data_frame.image_path.tolist()
    assert primero.data_frame.image_path.tolist() != distinto.data_frame.image_path.tolist()
