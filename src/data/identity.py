"""Stable sample identities and explicit association of inference results."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset, default_collate


def sample_id_for_path(path: str) -> str:
    """Identity of an original relative path; content revisions have separate hashes."""
    normalized = PurePosixPath(str(path).replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"image_path must be relative to the dataset root: {path}")
    return hashlib.sha256(normalized.as_posix().encode("utf-8")).hexdigest()


def ensure_sample_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    if frame[["image_path", "label"]].isna().any().any():
        raise ValueError("Manifest contains missing paths or labels")
    path_ids = frame["image_path"].map(sample_id_for_path)
    if "sample_id" not in frame:
        frame["sample_id"] = path_ids
    if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Manifest contains empty sample_id")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id")
    return frame


def dataset_frame(dataset) -> pd.DataFrame | None:
    if isinstance(dataset, Subset):
        frame = dataset_frame(dataset.dataset)
        return None if frame is None else frame.iloc[list(dataset.indices)].reset_index(drop=True)
    frame = getattr(dataset, "data_frame", None)
    return None if frame is None else ensure_sample_ids(frame)


class IdentifiedValues(list):
    """List compatible with metrics, carrying IDs in the actual inference order."""

    def __init__(self):
        super().__init__()
        self.sample_ids: list[str] = []


class _IdentifiedDataset(Dataset):
    def __init__(self, dataset, frame):
        self.dataset = dataset
        self.ids = frame["sample_id"].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, self.ids[index]


def identified_batches(loader: DataLoader):
    """Carry IDs through sampling/workers without changing CornDataset's pair API.

    Synthetic TensorDataset callers have no manifest; their identity is explicitly None.
    Batch sampler is reused, so shuffled/restricted inference is associated correctly.
    """
    frame = dataset_frame(loader.dataset)
    if frame is None:
        for images, labels in loader:
            yield images, labels, None
        return
    if loader.collate_fn is not default_collate:
        raise ValueError("Identified inference requires the default collate function")
    wrapped = DataLoader(
        _IdentifiedDataset(loader.dataset, frame),
        batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
    )
    yield from wrapped


def align_predictions(frame: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    frame = ensure_sample_ids(frame)
    ids = list(sample_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in predictions")
    indexed = frame.set_index("sample_id", drop=False)
    unknown = set(ids) - set(indexed.index)
    if unknown:
        raise ValueError(f"Predictions reference unknown sample IDs: {sorted(unknown)[:3]}")
    return indexed.loc[ids].reset_index(drop=True)
