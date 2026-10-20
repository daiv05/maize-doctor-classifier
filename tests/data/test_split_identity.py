import pandas as pd
import yaml
from PIL import Image

from scripts.pipeline import create_splits
from src.data.identity import validate_sample_ids


def test_splits_nuevos_persisten_sample_id(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "outputs"
    for environment, offset in (("lab", 0), ("real", 100)):
        directory = dataset_root / "clean" / "healthy" / environment
        directory.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (8, 8), (offset + index, index, 255 - index)).save(
                directory / f"img_{index}.png"
            )

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"classes": ["healthy"], "seed": 42},
                "paths": {"raw_dir": "clean", "split_output_dir": "splits/seed_42"},
            }
        )
    )
    monkeypatch.setattr(create_splits, "get_dataset_root", lambda: dataset_root)
    monkeypatch.setattr(create_splits, "get_output_root", lambda: output_root)
    monkeypatch.setattr(create_splits, "_resolve_index_workers", lambda: 1)

    create_splits.run_data_preparation_pipeline(str(config_path))

    for split_name in ("train", "val", "test"):
        split_path = output_root / "splits" / "seed_42" / f"{split_name}.csv"
        frame = pd.read_csv(split_path)
        assert "sample_id" in frame.columns
        validate_sample_ids(frame)
