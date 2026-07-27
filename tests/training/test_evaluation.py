import numpy as np
import pandas as pd
import pytest

from src.training.evaluation import (
    AGGREGATE_ROW_LABEL,
    compute_calibration_metrics,
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


def test_ece_no_pierde_muestras_con_confianza_cero():
    """
    Regresion: confianza 0.0 caia fuera de todo bin porque el primer bin era
    exclusivo por la izquierda ``(0.0, 1/15]``. Con num_bins=15 el primer bin es
    ``[0.0, 1/15]``, que agrupa las dos muestras de confianza 0.0 (accuracy 0.5,
    gap 0.5, peso 0.5 -> 0.25) y dejo el ultimo bin ``(14/15, 1.0]`` con las dos
    muestras de confianza 1.0 (accuracy 1.0, gap 0, peso 0.5 -> 0). Total 0.25.
    """
    confidences = [0.0, 0.0, 1.0, 1.0]
    correct = [True, False, True, True]
    assert expected_calibration_error(confidences, correct, num_bins=15) == pytest.approx(0.25)


def test_ece_pesos_de_bins_suman_uno_sin_perder_muestras():
    """
    Reimplementa el binning de forma independiente (mismo criterio ``[lower, upper]``
    inclusivo solo en el primer bin, ``(lower, upper]`` en el resto) para confirmar
    que ninguna muestra queda fuera de todos los bins: la suma de los pesos
    (fraccion de muestras por bin no vacio) debe dar 1.0.
    """
    confidences = np.array([0.0, 0.05, 0.5, 0.9, 1.0])
    num_bins = 10
    edges = np.linspace(0.0, 1.0, num_bins + 1)

    total_weight = 0.0
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        total_weight += in_bin.mean()

    assert total_weight == pytest.approx(1.0)


def test_compute_calibration_metrics_valores_calculados_a_mano():
    """
    label=[a,a,a,b], pred_label=[a,a,b,b], pred_prob=[0.9,0.7,0.6,0.8].
    Correctos: [True, True, False, True] -> n_hits=3, n_misses=1.

    mean_confidence_hits = mean(0.9, 0.7, 0.8) = 0.8
    mean_confidence_misses = mean(0.6) = 0.6
    brier_binary_hit = mean((0.9-1)^2, (0.7-1)^2, (0.6-0)^2, (0.8-1)^2)
                     = mean(0.01, 0.09, 0.36, 0.04) = 0.125

    Con num_bins=15 (default) cada confianza cae en un bin propio (ancho ~0.0667):
    0.9 -> gap 0.1, 0.7 -> gap 0.3, 0.6 -> gap 0.6, 0.8 -> gap 0.2; cada uno con
    peso 0.25 -> ece = 0.25*(0.1+0.3+0.6+0.2) = 0.3.
    """
    frame = pd.DataFrame(
        {
            "label": ["a", "a", "a", "b"],
            "pred_label": ["a", "a", "b", "b"],
            "pred_prob": [0.9, 0.7, 0.6, 0.8],
        }
    )
    result = compute_calibration_metrics(frame, {"a": 0, "b": 1})

    assert result["ece"] == pytest.approx(0.3)
    assert result["brier_binary_hit"] == pytest.approx(0.125)
    assert result["mean_confidence_hits"] == pytest.approx(0.8)
    assert result["mean_confidence_misses"] == pytest.approx(0.6)
    assert result["n_hits"] == 3
    assert result["n_misses"] == 1


def _environment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["common_rust"] * 4,
            "pred_label": ["common_rust", "common_rust", "healthy", "healthy"],
            "pred_prob": [0.9, 0.9, 0.8, 0.8],
            "environment": ["lab", "lab", "real", "real"],
        }
    )


def test_metricas_por_environment_separan_lab_y_real():
    """Las filas agregadas (class == __all__) conservan accuracy, macro_f1 y n por entorno."""
    result = compute_environment_metrics(_environment_frame())
    aggregate = result[result["class"] == AGGREGATE_ROW_LABEL].set_index("environment")

    assert aggregate.loc["lab", "accuracy"] == pytest.approx(1.0)
    assert aggregate.loc["real", "accuracy"] == pytest.approx(0.0)
    assert aggregate.loc["lab", "n"] == 2


def test_environment_expone_f1_por_clase_con_su_n():
    """
    Aislar una clase en un entorno concreto es la razon de existir del artefacto.

    `common_rust` acierta las 2 de lab (F1=1.0) y falla las 2 de real (F1=0.0);
    la n de cada combinacion debe verse al lado para juzgar su fiabilidad.
    """
    result = compute_environment_metrics(_environment_frame())
    per_class = result[result["class"] != AGGREGATE_ROW_LABEL].set_index(["environment", "class"])

    assert per_class.loc[("lab", "common_rust"), "f1"] == pytest.approx(1.0)
    assert per_class.loc[("lab", "common_rust"), "n"] == 2
    assert per_class.loc[("real", "common_rust"), "f1"] == pytest.approx(0.0)
    assert per_class.loc[("real", "common_rust"), "n"] == 2


def test_environment_incluye_clases_solo_predichas_con_n_cero():
    """`healthy` solo aparece como prediccion en real: F1 definido y n de soporte 0."""
    result = compute_environment_metrics(_environment_frame())
    per_class = result[result["class"] != AGGREGATE_ROW_LABEL].set_index(["environment", "class"])

    assert per_class.loc[("real", "healthy"), "n"] == 0
    assert per_class.loc[("real", "healthy"), "f1"] == pytest.approx(0.0)


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
