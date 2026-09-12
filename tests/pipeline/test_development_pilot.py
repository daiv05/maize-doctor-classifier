import argparse
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from scripts.checks import paired_segmentation_pilot as pilot
from src.data.transforms import CornTransformFactory


def test_structural_paired_pilot_never_loads_test_images(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    splits = tmp_path / "splits"
    splits.mkdir()
    rows = []
    for i in range(12):
        label = "healthy" if i % 2 else "common_rust"
        path = root / f"clean/{label}/real/{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (i * 17, i * 9, 200)).save(path)
        rows.append(
            {"image_path": path.relative_to(root).as_posix(), "label": label, "environment": "real"}
        )
    for j, split in enumerate(("train", "val", "test")):
        pd.DataFrame(rows[j * 4 : (j + 1) * 4]).to_csv(splits / f"{split}.csv", index=False)
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"model": "tiny"}))
    checkpoint = run / "best.pth"
    checkpoint.write_bytes(b"structural fixture")
    factory = CornTransformFactory(target_size=(16, 16))

    def model(*args, **kwargs):
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2)
        )

    def loaded(*args, **kwargs):
        return SimpleNamespace(
            model=model(),
            class_to_idx={"common_rust": 0, "healthy": 1},
            factory=factory,
            summary={"model": "tiny", "image_size": [16, 16]},
            manifest_entry=lambda: {"fixture": True},
        )

    class Exposed(torch.nn.Module):
        def __init__(self, model, name):
            super().__init__()
            self.model = model

        def forward(self, tensor):
            return self.model(tensor), tensor.mean((2, 3))

    class EmptySegmenter:
        def __init__(self, *args, **kwargs):
            pass

        def segment(self, image):
            return []

    read_paths = []
    original_loader = pilot.load_and_normalize_image

    def tracked_load(path):
        read_paths.append(str(path))
        return original_loader(path)

    monkeypatch.setattr(pilot, "load_run", loaded)
    monkeypatch.setattr(pilot, "build_model", model)
    monkeypatch.setattr(pilot, "FeatureExposedModel", Exposed)
    monkeypatch.setattr(pilot, "MaizeLeafSegmenter", EmptySegmenter)
    monkeypatch.setattr(pilot, "load_and_normalize_image", tracked_load)
    args = argparse.Namespace(
        dataset_root=root,
        splits_dir=splits,
        checkpoints=[checkpoint],
        segmenter_checkpoint=checkpoint,
        output_dir=tmp_path / "pilot",
        train_per_stratum=2,
        val_per_stratum=2,
        seed=42,
        training_seeds=[42, 43],
        epochs=2,
        batch_size=4,
        threads=1,
    )
    pilot.run_pilot(args)
    assert not any(str(root / row["image_path"]) in read_paths for row in rows[8:])
    result = json.loads((args.output_dir / "results.json").read_text())
    assert result["fallback_count"] == 8
    assert len(result["frozen_imagenet_head_pilot"]) == 2
    for item in result["frozen_imagenet_head_pilot"]:
        assert item["original"]["confusion_matrix"] == item["segmented"]["confusion_matrix"]
    with pytest.raises(FileExistsError):
        pilot.run_pilot(args)
