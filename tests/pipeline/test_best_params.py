import importlib.util
import json
import sys

import pytest

from src.config import PROJECT_ROOT

_TRAIN = PROJECT_ROOT / "scripts" / "pipeline" / "train.py"


@pytest.fixture
def modulo():
    """Carga el CLI de entrenamiento sin ejecutarlo."""
    spec = importlib.util.spec_from_file_location("train_cli", _TRAIN)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture
def fichero_parametros(tmp_path):
    """best_params.json con los seis hiperparámetros que afina Optuna."""
    destino = tmp_path / "best_params.json"
    destino.write_text(
        json.dumps(
            {
                "best_params": {
                    "learning_rate": 0.000462,
                    "weight_decay": 4.3e-05,
                    "batch_size": 64,
                    "class_weights": "sqrt_inverse",
                    "label_smoothing": 0.0917,
                    "warmup_epochs": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    return destino


def test_aplica_los_seis_hiperparametros(modulo, fichero_parametros, monkeypatch):
    """Enumerar los parámetros en cada llamador perdía weight_decay y warmup_epochs."""
    monkeypatch.setattr(sys, "argv", ["train.py", "--best-params", str(fichero_parametros)])

    args = modulo._parse_args()

    assert args.learning_rate == 0.000462
    assert args.weight_decay == 4.3e-05
    assert args.batch_size == 64
    assert args.class_weights == "sqrt_inverse"
    assert args.label_smoothing == 0.0917
    assert args.warmup_epochs == 1


def test_una_bandera_explicita_gana_sobre_el_json(modulo, fichero_parametros, monkeypatch):
    """El JSON fija defaults, así que la línea de comandos sigue mandando."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--best-params", str(fichero_parametros), "--learning-rate", "0.005"],
    )

    args = modulo._parse_args()

    assert args.learning_rate == 0.005
    assert args.weight_decay == 4.3e-05


def test_sin_el_fichero_los_defaults_no_cambian(modulo, monkeypatch):
    """La bandera es opcional y no altera el comportamiento por defecto."""
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = modulo._parse_args()

    assert args.best_params == ""
    assert args.learning_rate == pytest.approx(1e-4)


def test_una_clave_desconocida_se_reporta_y_no_se_aplica(modulo, tmp_path, monkeypatch, capsys):
    """Una clave sin equivalente en el CLI no puede desaparecer en silencio."""
    destino = tmp_path / "raro.json"
    destino.write_text(json.dumps({"best_params": {"parametro_inventado": 3}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train.py", "--best-params", str(destino)])

    modulo._parse_args()

    assert "parametro_inventado" in capsys.readouterr().out
