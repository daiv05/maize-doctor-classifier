import yaml

from src.config import PROJECT_ROOT

_REQUIRED_KEYS = {
    "segmentation",
    "n_segments",
    "compactness",
    "nsamples",
    "batch_size",
    "background",
    "images_per_class",
    "global_sample_size",
    "seed",
}


def _load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "dataset.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_shap_block_has_every_required_key():
    shap_cfg = _load_config()["shap"]

    assert _REQUIRED_KEYS <= set(shap_cfg)


def test_shap_block_values_are_supported():
    shap_cfg = _load_config()["shap"]

    assert shap_cfg["segmentation"] in {"slic", "quickshift"}
    assert shap_cfg["background"] in {"black", "mean", "blur"}
    assert shap_cfg["nsamples"] > 2 * shap_cfg["n_segments"]


def test_shap_library_is_importable():
    import shap

    assert hasattr(shap, "KernelExplainer")
