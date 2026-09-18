from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import CornDataset, build_weighted_sampler
from src.data.identity import sample_id_for_path


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"classes": ["healthy", "minor"]},
                "augmentation": {"minority_ratio_threshold": 2.0},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_error_de_carga_identifica_la_misma_muestra_y_no_avanza(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    config_path = _config(tmp_path)
    rows = [
        {
            "image_path": "clean/healthy/real/missing.png",
            "label": "healthy",
            "environment": "real",
        },
        {
            "image_path": "clean/healthy/real/valid.png",
            "label": "healthy",
            "environment": "real",
        },
    ]
    calls: list[Path] = []

    def controlled_load(path):
        resolved = Path(path)
        calls.append(resolved)
        if resolved.name == "missing.png":
            raise FileNotFoundError("fallo controlado")
        return Image.new("RGB", (4, 4), (10, 20, 30))

    monkeypatch.setattr("src.data.dataset.get_dataset_root", lambda: dataset_root)
    monkeypatch.setattr("src.data.dataset.load_and_normalize_image", controlled_load)
    dataset = CornDataset(pd.DataFrame(rows), config_path=str(config_path))
    original_frame = dataset.data_frame.copy(deep=True)
    missing_id = sample_id_for_path(rows[0]["image_path"])

    with pytest.raises(RuntimeError) as caught:
        dataset[0]

    message = str(caught.value)
    assert "idx=0" in message
    assert f"sample_id={missing_id}" in message
    assert f"image_path={rows[0]['image_path']}" in message
    assert "label=healthy" in message
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert calls == [dataset_root / rows[0]["image_path"]]
    pd.testing.assert_frame_equal(dataset.data_frame, original_frame)

    calls.clear()
    with pytest.raises(RuntimeError, match=missing_id):
        next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    assert calls == [dataset_root / rows[0]["image_path"]]


class _MemoryCache:
    def __init__(self, paths: list[str]) -> None:
        self.images = {path: Image.new("RGB", (4, 4), (30, 60, 90)) for path in paths}

    def __contains__(self, image_path: str) -> bool:
        return image_path in self.images

    def get(self, image_path: str) -> Image.Image:
        return self.images[image_path].copy()


def test_contrato_actual_de_dataset_cache_transforms_ids_batch_y_sampler(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    config_path = _config(tmp_path)
    rows = [
        {
            "image_path": f"clean/{label}/real/{label}_{index}.png",
            "label": label,
            "environment": "real",
        }
        for label in ("healthy", "minor")
        for index in range(3)
    ]
    paths = [row["image_path"] for row in rows]

    def disk_access_is_a_regression(path):
        raise AssertionError(f"se intentó leer del disco: {path}")

    def standard_transform(_image):
        return torch.full((3, 2, 2), 1.0)

    def minority_transform(_image):
        return torch.full((3, 2, 2), 2.0)

    monkeypatch.setattr("src.data.dataset.get_dataset_root", lambda: dataset_root)
    monkeypatch.setattr(
        "src.data.dataset.load_and_normalize_image",
        disk_access_is_a_regression,
    )
    class_to_idx = {"healthy": 0, "minor": 1}
    dataset = CornDataset(
        pd.DataFrame(rows),
        config_path=str(config_path),
        transform=standard_transform,
        minority_transform=minority_transform,
        class_to_idx=class_to_idx,
        minority_classes={"minor"},
        max_per_class=1,
        seed=7,
        image_cache=_MemoryCache(paths),
    )

    assert dataset.class_to_idx == class_to_idx
    assert dataset.idx_to_class == {0: "healthy", 1: "minor"}
    assert dataset.minority_classes == {"minor"}
    assert dataset.data_frame["label"].value_counts().to_dict() == {"healthy": 1, "minor": 1}

    observed_ids = []
    for index, row in dataset.data_frame.iterrows():
        image, label, sample_id = dataset[index]
        expected_value = 2.0 if row["label"] == "minor" else 1.0
        assert torch.all(image.eq(expected_value))
        assert label == class_to_idx[row["label"]]
        assert sample_id == row["sample_id"]
        observed_ids.append(sample_id)

    images, labels, batch_ids = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))
    )
    assert images.shape == (2, 3, 2, 2)
    assert labels.tolist() == [class_to_idx[label] for label in dataset.data_frame["label"]]
    assert list(batch_ids) == observed_ids

    sampler = build_weighted_sampler(dataset, seed=42)
    assert sampler is not None
    assert sampler.num_samples == len(dataset)
