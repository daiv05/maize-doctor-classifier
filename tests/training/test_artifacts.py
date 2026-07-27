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
