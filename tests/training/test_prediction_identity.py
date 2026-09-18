from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Sampler, Subset

from src.data.dataset import CornDataset
from src.data.identity import sample_id_for_path
from src.training.artifacts import write_predictions_csv
from src.training.loop import run_epoch


class _FixedSampler(Sampler[int]):
    def __iter__(self):
        return iter((2, 0, 1))

    def __len__(self):
        return 3


class _SignalModel(torch.nn.Module):
    def forward(self, images):
        signal = images[:, 0, 0, 0].long()
        logits = torch.full((len(images), 2), -10.0)
        logits[torch.arange(len(images)), signal.remainder(2)] = 10.0
        return logits


def test_predicciones_con_sampler_conservan_identidad(tmp_path, monkeypatch):
    rows = [
        {"image_path": f"clean/healthy/real/img_{index}.jpg", "label": "healthy"}
        if index % 2 == 0
        else {
            "image_path": f"clean/common_rust/real/img_{index}.jpg",
            "label": "common_rust",
        }
        for index in range(4)
    ]
    split_path = tmp_path / "legacy_test.csv"
    pd.DataFrame(rows).to_csv(split_path, index=False)
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump({"dataset": {"classes": ["healthy", "common_rust"]}}))

    monkeypatch.setattr("src.data.dataset.get_dataset_root", lambda: Path("/unused"))

    def fake_load(path):
        index = int(path.stem.removeprefix("img_"))
        return torch.full((3, 1, 1), float(index))

    monkeypatch.setattr("src.data.dataset.load_and_normalize_image", fake_load)

    dataset = CornDataset(csv_path=str(split_path), config_path=str(config_path))
    assert "sample_id" not in pd.read_csv(split_path).columns
    subset = Subset(dataset, [3, 1, 2])
    loader = DataLoader(subset, batch_size=2, sampler=_FixedSampler())
    _, labels, predictions, probabilities = run_epoch(
        _SignalModel(),
        loader,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )

    frame = write_predictions_csv(
        tmp_path,
        subset,
        dataset.idx_to_class,
        labels,
        predictions,
        probabilities,
    )

    expected_paths = [rows[index]["image_path"] for index in (2, 3, 1)]
    assert expected_paths != [rows[index]["image_path"] for index in (3, 1, 2)]
    assert frame["image_path"].tolist() == expected_paths
    assert frame["sample_id"].tolist() == [sample_id_for_path(path) for path in expected_paths]
    assert frame["true_label"].tolist() == [rows[index]["label"] for index in (2, 3, 1)]
    assert frame["pred_label"].tolist() == frame["true_label"].tolist()

    persisted = pd.read_csv(tmp_path / "predictions.csv")
    assert persisted[["sample_id", "image_path", "true_label", "pred_label"]].equals(
        frame[["sample_id", "image_path", "true_label", "pred_label"]]
    )
