import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import CornDataset
from src.data.identity import ensure_sample_ids
from src.data.transforms import CornTransformFactory
from src.training.artifacts import write_predictions_csv
from src.training.loop import run_epoch


def dataset(monkeypatch, root, frame):
    monkeypatch.setattr("src.data.dataset.get_dataset_root", lambda: root)
    return CornDataset(
        frame, transform=CornTransformFactory(target_size=(16, 16)).get_pipeline("test")
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_bad_image_never_substituted(monkeypatch, fake_image_root, tmp_splits_dir, corrupt):
    frame = pd.read_csv(tmp_splits_dir / "test.csv")
    bad = fake_image_root / frame.iloc[0].image_path
    if corrupt:
        bad.write_bytes(b"not an image")
    else:
        bad.unlink()
    ds = dataset(monkeypatch, fake_image_root, frame)
    with pytest.raises(RuntimeError, match="sample_id=.*no sample substitution"):
        ds[0]
    assert ds[1][0].shape == (3, 16, 16)


def test_identity_survives_filter_reorder_and_shuffled_inference(
    monkeypatch, fake_image_root, tmp_splits_dir, tmp_path
):
    source = ensure_sample_ids(pd.read_csv(tmp_splits_dir / "test.csv"))
    selected = source.iloc[[5, 1, 14, 2]]
    ds = dataset(monkeypatch, fake_image_root, selected)
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, len(ds.class_to_idx))
    )
    loader = DataLoader(Subset(ds, [3, 1, 0]), batch_size=2, shuffle=True)
    _, labels, predictions, probs = run_epoch(
        model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    frame = write_predictions_csv(tmp_path, ds, ds.idx_to_class, predictions, probs)
    assert set(frame.sample_id) == set(selected.iloc[[3, 1, 0]].sample_id)
    expected = source.set_index("sample_id").loc[frame.sample_id]
    assert frame.image_path.tolist() == expected.image_path.tolist()
    assert frame.label.tolist() == [ds.idx_to_class[label] for label in labels]


def test_unidentified_predictions_rejected(monkeypatch, fake_image_root, tmp_splits_dir, tmp_path):
    ds = dataset(monkeypatch, fake_image_root, pd.read_csv(tmp_splits_dir / "test.csv"))
    with pytest.raises(ValueError, match="sample_ids"):
        write_predictions_csv(tmp_path, ds, ds.idx_to_class, [0] * len(ds), [0.5] * len(ds))


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate sample_id"):
        ensure_sample_ids(pd.DataFrame({"image_path": ["a.png", "a.png"], "label": ["x", "x"]}))
