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
