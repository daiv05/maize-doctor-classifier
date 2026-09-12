import argparse
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from scripts.pipeline import segment_dataset
from src.data.loader import load_and_normalize_image
from src.data.segmented import bind_segmented_splits
from src.data.transforms import CornTransformFactory
from src.provenance import atomic_json
from src.training.hyperparameters import parse_with_best_params


def test_cli_overrides_hpo_including_false(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-params")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=False)
    path = tmp_path / "best.json"
    atomic_json(path, {"schema": "pytorch_hpo_v1", "learning_rate": 0.02, "clahe": True})
    args = parse_with_best_params(
        parser, ["--best-params", str(path), "--no-clahe", "--learning-rate", ".03"]
    )
    assert args.learning_rate == 0.03
    assert args.clahe is False


def test_exif_clahe_tensor_agrees_between_xai_and_training(tmp_path):
    pytest.importorskip("lime")
    from src.explainability.visual_report import build_validation_transform, prepare_lime_image

    raw = Image.fromarray(np.random.default_rng(42).integers(0, 255, (40, 60, 3), dtype=np.uint8))
    exif = raw.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    raw.save(path, exif=exif)
    image = load_and_normalize_image(path)
    assert image.size == (40, 60)
    factory = CornTransformFactory(target_size=(16, 24), clahe=True)
    expected = factory.get_pipeline("test")(image)
    actual = build_validation_transform((16, 24), factory.to_contract())(image)
    pixels = prepare_lime_image(image, (16, 24), factory.to_contract())
    perturbed_baseline = build_validation_transform((16, 24))(Image.fromarray(pixels))
    assert torch.equal(expected, actual)
    assert torch.equal(expected, perturbed_baseline)


def test_segmented_resume_no_collision_and_original_splits(tmp_path, monkeypatch):
    root = tmp_path / "data"
    source = root / "clean/healthy/lab"
    source.mkdir(parents=True)
    frames = []
    for index, name in enumerate(("leaf.jpg", "leaf.png", "other.png")):
        Image.new("RGB", (32, 32), (index * 70, 30, 50)).save(source / name)
        frames.append(
            {"image_path": f"clean/healthy/lab/{name}", "label": "healthy", "environment": "lab"}
        )
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, row in zip(("train", "val", "test"), frames):
        pd.DataFrame([row]).to_csv(splits / f"{split}.csv", index=False)
    checkpoint = tmp_path / "segmenter.pt"
    checkpoint.write_bytes(b"unit-test-segmenter")
    calls = []

    class EmptySegmenter:
        def __init__(self, **kwargs):
            pass

        def segment(self, image):
            calls.append(1)
            return []

    monkeypatch.setattr(segment_dataset, "MaizeLeafSegmenter", EmptySegmenter)
    args = argparse.Namespace(
        dataset_dir=root,
        output_dir=tmp_path / "derived/clean",
        splits_dir=splits,
        checkpoint=checkpoint,
        profile="crop_mask_letterbox",
        target_size=[16, 16],
        max_images=0,
        max_previews=0,
        preview_dir=None,
        device="cpu",
        uncertain_policy="original",
    )
    assert segment_dataset.run_segmentation(args)["complete"]
    assert len(calls) == 3
    segment_dataset.run_segmentation(args)
    assert len(calls) == 3
    audit = args.output_dir / ".segmentation"
    manifest = json.loads((audit / "manifest.json").read_text())
    assert len({r["image_path"] for r in manifest}) == 3
    assert all(r["fallback_original"] for r in manifest)
    for split, row in zip(("train", "val", "test"), frames):
        derived = pd.read_csv(audit / "splits" / f"{split}.csv")
        assert derived.original_image_path.tolist() == [row["image_path"]]
    factory = CornTransformFactory()
    bind_segmented_splits(factory, audit / "splits")
    assert factory.to_contract()["segmentation"]["checkpoint_sha256"]
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible"):
        segment_dataset.run_segmentation(args)


def test_empty_class_directories_do_not_mean_complete_download(tmp_path):
    from scripts.dataset.download_dataset import _clean_dir_has_content

    (tmp_path / "healthy/real").mkdir(parents=True)
    assert not _clean_dir_has_content(tmp_path)
