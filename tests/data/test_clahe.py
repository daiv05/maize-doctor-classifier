import numpy as np
import pytest
from PIL import Image

from src.config import PROJECT_ROOT
from src.data.transforms import CornCLAHETransform, CornTransformFactory

_REAL_LEAF_IMAGE = (
    PROJECT_ROOT
    / "experiments"
    / "clahe"
    / "input"
    / "gray_leaf_spot_maize_field_real_13749986.jpg"
)


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


def test_clahe_aumenta_el_contraste_dentro_de_cada_region(leaf_image):
    """
    CLAHE ecualiza por tiles, no globalmente: sube el contraste dentro de cada
    region local pero puede comprimir la diferencia de luminancia entre regiones
    separadas (p. ej. una mitad sobreexpuesta y otra en sombra). Medir el std sobre
    la imagen completa mezcla ambos efectos y puede bajar aunque el contraste local
    -que es la propiedad que la clase promete- haya subido. Por eso se mide el std
    por separado dentro de cada mitad del fixture, no sobre el total.
    """
    import cv2

    original = cv2.cvtColor(np.array(leaf_image), cv2.COLOR_RGB2LAB)[..., 0]
    processed = cv2.cvtColor(np.array(CornCLAHETransform()(leaf_image)), cv2.COLOR_RGB2LAB)[..., 0]

    mitad = original.shape[0] // 2
    assert processed[:mitad].std() > original[:mitad].std()
    assert processed[mitad:].std() > original[mitad:].std()


@pytest.mark.skipif(
    not _REAL_LEAF_IMAGE.exists(), reason=f"no se encontro la imagen real: {_REAL_LEAF_IMAGE}"
)
def test_clahe_preserva_el_tono_en_imagen_real():
    """Ancla en datos reales la propiedad de diseno de la clase: CLAHE sobre el
    canal L de LAB no debe desplazar el hue de forma perceptible, porque el color
    es la senal diagnostica de las deficiencias nutricionales."""
    import cv2

    original_image = Image.open(_REAL_LEAF_IMAGE).convert("RGB").resize((224, 224))
    original = np.array(original_image)
    processed = np.array(CornCLAHETransform()(original_image))

    hue_original = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)[..., 0].astype(float)
    hue_processed = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)[..., 0].astype(float)
    shift = np.abs(((hue_processed - hue_original + 90) % 180) - 90).mean()

    assert shift < 2.0, f"desplazamiento de hue demasiado alto: {shift:.2f} grados"


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
