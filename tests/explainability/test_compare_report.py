import json
import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
from lime import lime_image
from PIL import Image

from src.explainability.compare_report import render_comparison
from src.explainability.segmentation import build_segments
from src.explainability.visual_report import prepare_lime_image

_TARGET_SIZE = (32, 32)
_IDX_TO_CLASS = {0: "healthy", 1: "common_rust"}
_LIME_CFG = {"num_samples": 20, "num_features": 3, "seed": 42}
_SHAP_CFG = {
    "segmentation": "slic",
    "n_segments": 4,
    "compactness": 10.0,
    "nsamples": 32,
    "batch_size": 8,
    "background": "black",
    "seed": 42,
}


class _FakeExplanation:
    """Sustituto minimo de la explicacion de LIME: `render_comparison` solo lee
    `local_exp`, indexado por clase."""

    def __init__(self, local_exp: dict[int, list[tuple[int, float]]]):
        self.local_exp = local_exp


@pytest.fixture
def captured_kwargs(monkeypatch) -> dict:
    """Intercepta `explain_instance` para capturar el `segmentation_fn` que recibe LIME,
    sin pagar el costo del muestreo real."""
    captured: dict = {}

    def fake_explain_instance(self, image, classifier_fn, **kwargs):
        captured.update(kwargs)
        segments = kwargs["segmentation_fn"](image)
        local_exp = {
            class_idx: [(segment_id, 1.0) for segment_id in range(int(segments.max()) + 1)]
            for class_idx in range(len(_IDX_TO_CLASS))
        }
        return _FakeExplanation(local_exp)

    monkeypatch.setattr(lime_image.LimeImageExplainer, "explain_instance", fake_explain_instance)
    return captured


def _dummy_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 2)).eval()


def _dummy_image() -> Image.Image:
    """Imagen 64x64 con cuatro cuadrantes de color solido y contiguo.

    SLIC segmenta por coherencia de color y espacio: el ruido uniforme (usado
    originalmente) no tiene regiones coherentes, asi que `enforce_connectivity` termina
    fusionando todo en un unico superpixel. Cuatro bloques de color bien diferenciados
    son justo lo que SLIC esta disenado para encontrar, y con ellos el fixture produce
    un mapa de 4 superpixeles reproducible (ver `test_fixture_segmentation_has_multiple_segments`).
    """
    size = 64
    half = size // 2
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:half, :half] = (220, 40, 40)
    image[:half, half:] = (40, 200, 60)
    image[half:, :half] = (40, 80, 220)
    image[half:, half:] = (230, 200, 40)
    return Image.fromarray(image)


def _render(tmp_path):
    return render_comparison(
        image=_dummy_image(),
        model=_dummy_model(),
        model_name=None,
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=tmp_path / "compare.png",
        lime_cfg=_LIME_CFG,
        shap_cfg=_SHAP_CFG,
        device=torch.device("cpu"),
    )


def test_fixture_segmentation_has_multiple_segments():
    """Guarda que el fixture compartido no colapse a un unico superpixel.

    Un mapa de un solo segmento es siempre un array de ceros (las etiquetas de
    `build_segments` son consecutivas desde 0), y `assert_array_equal` no distingue dos
    segmentaciones distintas que colapsen igual: la revision detecto que eso volvia vacua
    la comparacion de identidad en `test_lime_and_shap_receive_the_identical_segmentation_array`,
    y con ella las demas pruebas del modulo, que validaban vectores de atribucion de
    longitud 1. Si una edicion futura del fixture (imagen, target_size o n_segments)
    vuelve a colapsar el mapa, este test debe fallar primero y en voz alta.
    """
    image_np = prepare_lime_image(_dummy_image(), _TARGET_SIZE)
    segments = build_segments(
        image_np,
        algorithm=_SHAP_CFG["segmentation"],
        n_segments=_SHAP_CFG["n_segments"],
        compactness=_SHAP_CFG["compactness"],
    )

    assert int(segments.max()) + 1 > 1


def test_writes_png_and_sidecars(tmp_path):
    _render(tmp_path)

    assert (tmp_path / "compare.png").exists()
    assert (tmp_path / "compare.json").exists()
    assert (tmp_path / "compare.npy").exists()


def test_result_reports_prediction_and_agreement(tmp_path):
    result = _render(tmp_path)

    assert result["predicted_label"] in _IDX_TO_CLASS.values()
    assert 0.0 <= result["predicted_prob"] <= 1.0
    assert set(result["agreement"]) == {"iou_topk", "spearman", "sign_agreement"}


def test_sidecar_holds_both_attribution_vectors(tmp_path):
    _render(tmp_path)

    metadata = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))
    segments = np.load(tmp_path / "compare.npy")

    n_segments = int(segments.max()) + 1
    assert len(metadata["lime_weights"]) == n_segments
    assert len(metadata["shap_values"]) == n_segments


def test_lime_and_shap_share_the_segmentation(tmp_path):
    _render(tmp_path)

    metadata = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))

    assert len(metadata["lime_weights"]) == len(metadata["shap_values"])
    assert metadata["n_segments"] == len(metadata["shap_values"])


def test_black_background_is_reliable_and_does_not_warn(tmp_path, caplog):
    """La linea base por defecto (D7) preserva la paridad de nocion de ausencia con
    LIME: la comparacion es confiable y no dispara ninguna advertencia."""
    with caplog.at_level(logging.WARNING):
        result = _render(tmp_path)

    assert result["agreement_reliable"] is True
    metadata = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))
    assert metadata["agreement_reliable"] is True
    assert not caplog.records


def test_non_black_background_flips_the_flag_and_warns(tmp_path, caplog):
    """`shap.background != "black"` (ej. "mean") rompe la paridad de nocion de ausencia
    con `hide_color=0` de LIME: las metricas de acuerdo dejan de ser comparables y eso
    debe quedar marcado tanto en el log como en el sidecar."""
    shap_cfg = dict(_SHAP_CFG, background="mean")

    with caplog.at_level(logging.WARNING):
        result = render_comparison(
            image=_dummy_image(),
            model=_dummy_model(),
            model_name=None,
            idx_to_class=_IDX_TO_CLASS,
            target_size=_TARGET_SIZE,
            output_path=tmp_path / "compare.png",
            lime_cfg=_LIME_CFG,
            shap_cfg=shap_cfg,
            device=torch.device("cpu"),
        )

    assert result["agreement_reliable"] is False
    metadata = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))
    assert metadata["agreement_reliable"] is False
    assert "no son comparables" in caplog.text


def test_lime_and_shap_receive_the_identical_segmentation_array(captured_kwargs, tmp_path):
    """Prueba de identidad, no solo de forma: el `segmentation_fn` capturado en LIME debe
    devolver el mismo array de superpixeles que quedo persistido en el sidecar `.npy`, el
    mismo que `explain_with_kernel_shap` recibio para calcular sus valores de Shapley.
    Dos segmentaciones distintas con igual cantidad de superpixeles no pasarian esto."""
    _render(tmp_path)

    segments_from_sidecar = np.load(tmp_path / "compare.npy")
    forwarded = captured_kwargs["segmentation_fn"](np.zeros((*_TARGET_SIZE, 3), dtype=np.uint8))

    np.testing.assert_array_equal(forwarded, segments_from_sidecar)
