import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.etapa_2 import stage2_experiments as stage
from src.provenance import sha256_file


def test_embedding_same_alone_with_other_model_and_when_other_cached(tmp_path, monkeypatch):
    image = tmp_path / "clean/healthy/real/example.png"
    image.parent.mkdir(parents=True)
    Image.fromarray(np.random.default_rng(42).integers(0, 255, (19, 31, 3), dtype=np.uint8)).save(
        image
    )
    manifest = pd.DataFrame(
        [
            {
                "image_path": image.relative_to(tmp_path).as_posix(),
                "label": "healthy",
                "label_idx": 0,
                "environment": "real",
                "split": "train",
                "sha256": sha256_file(image),
            }
        ]
    )
    monkeypatch.setattr(
        stage,
        "resolve_input_size",
        lambda name, fallback: (16, 24) if name == "small" else (32, 40),
    )
    monkeypatch.setattr(
        stage,
        "_as_feature_extractor",
        lambda *args, **kwargs: torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
        ),
    )
    for scenario in ("alone", "together", "other_cached"):
        folder = tmp_path / scenario
        folder.mkdir()
        manifest.to_csv(folder / "master_manifest.csv", index=False)
        if scenario == "other_cached":
            stage.extract_features(
                dataset_root=tmp_path, output_dir=folder, models=["large"], batch_size=1, workers=0
            )
        models = ["small"] if scenario == "alone" else ["large", "small"]
        stage.extract_features(
            dataset_root=tmp_path, output_dir=folder, models=models, batch_size=1, workers=0
        )
    expected = np.load(tmp_path / "alone/features/small.npy")
    for scenario in ("together", "other_cached"):
        np.testing.assert_array_equal(expected, np.load(tmp_path / scenario / "features/small.npy"))
