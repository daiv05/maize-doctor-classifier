import numpy as np
from PIL import Image

from src.segmentation.leaf_processor import LeafInstance, SegmentedLeafProcessor


def instance(array, index=0, bbox=(0, 0, 100, 100)):
    return LeafInstance(Image.fromarray(array), 0.9, bbox, index)


def test_empty_and_total_masks_are_rejected():
    image = Image.new("RGB", (100, 100))
    for fill in (0, 255):
        result = SegmentedLeafProcessor().process(
            image, [instance(np.full((100, 100), fill, dtype=np.uint8))]
        )
        assert result.status == "rejected"
        assert result.fallback_used


def test_near_total_mask_is_uncertain():
    array = np.full((100, 100), 255, dtype=np.uint8)
    array[0] = 0
    result = SegmentedLeafProcessor().process(Image.new("RGB", (100, 100)), [instance(array)])
    assert result.status == "uncertain"
    assert "near_full_mask" in result.warnings


def test_tied_candidates_multiple_components_and_crop_loss_are_recorded():
    array = np.zeros((100, 100), dtype=np.uint8)
    array[10:40, 10:40] = 255
    array[60:90, 60:90] = 255
    instances = [instance(array, i, (10, 10, 50, 50)) for i in range(2)]
    result = SegmentedLeafProcessor().process(Image.new("RGB", (100, 100)), instances)
    assert result.status == "uncertain"
    assert {"ambiguous_candidates", "multiple_components", "crop_discards_mask_pixels"} <= set(
        result.warnings
    )
    assert result.quality["components"] == 2
    assert result.quality["crop_retained_fraction"] == 0.5


def test_segmenter_cache_is_keyed_by_checkpoint_content(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from src.data.segmented import _segmenter
    from src.provenance import sha256_file

    monkeypatch.setitem(
        sys.modules,
        "src.segmentation.detector",
        SimpleNamespace(MaizeLeafSegmenter=lambda **kwargs: object()),
    )
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"old weights")
    _segmenter.cache_clear()
    try:
        old = _segmenter(str(checkpoint), None, sha256_file(checkpoint))
        checkpoint.write_bytes(b"new weights at same path")
        new = _segmenter(str(checkpoint), None, sha256_file(checkpoint))
        assert new is not old
        assert _segmenter(str(checkpoint), None, sha256_file(checkpoint)) is new
    finally:
        _segmenter.cache_clear()
