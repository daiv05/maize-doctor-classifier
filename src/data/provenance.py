"""Ordered manifests and immutable feature cache contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.identity import ensure_sample_ids
from src.provenance import contract_hash, sha256_file


def ordered_manifest_contract(frame, root=None):
    frame = ensure_sample_ids(frame)
    rows = []
    for row in frame.to_dict("records"):
        digest = row.get("sha256")
        if root is not None:
            actual = sha256_file(Path(root) / row["image_path"])
            if digest and actual != digest:
                raise ValueError(f"Source content changed: sample_id={row['sample_id']}")
            digest = actual
        if not digest:
            raise ValueError("A source content hash is required for every cache row")
        rows.append(
            {
                key: row[key]
                for key in ("sample_id", "image_path", "label", "label_idx", "environment", "split")
                if key in row
            }
            | {"sha256": digest}
        )
    return {"schema_version": 1, "ordered_rows": rows}


def model_state_hash(model):
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_feature_cache(
    output_dir, model_name, *, expected_preprocessing=None, expected_backbone=None
):
    output_dir = Path(output_dir)
    artifact = output_dir / "features" / f"{model_name}.npy"
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    contract = metadata.get("contract")
    if not contract:
        raise ValueError("Legacy feature cache has no provenance; use a new experiment directory")
    if contract_hash(contract) != metadata["contract_sha256"]:
        raise ValueError("Feature contract hash mismatch")
    frame = pd.read_csv(output_dir / "master_manifest.csv")
    actual = ordered_manifest_contract(frame, contract["dataset_root"])
    if actual != contract["manifest"]:
        raise ValueError("Feature cache sample order, labels, partitions or content changed")
    if expected_preprocessing is not None and contract["preprocessing"] != expected_preprocessing:
        raise ValueError("Feature preprocessing changed")
    if expected_backbone is not None and contract["backbone_sha256"] != expected_backbone:
        raise ValueError("Feature backbone weights changed")
    if sha256_file(artifact) != metadata["artifact_sha256"]:
        raise ValueError("Feature artifact hash mismatch")
    matrix = np.load(artifact, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(frame):
        raise ValueError("Feature rows differ from manifest")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature artifact contains nonfinite values")
    return matrix


def validate_holdout_lock(output_dir):
    output_dir = Path(output_dir)
    lock = json.loads((output_dir / "holdout.lock.json").read_text())
    if lock.get("master_manifest_sha256") != sha256_file(output_dir / "master_manifest.csv"):
        raise ValueError("Master manifest differs from frozen holdout contract (or legacy lock)")
    if lock.get("holdout_manifest_sha256") != sha256_file(output_dir / "holdout.csv"):
        raise ValueError("Holdout manifest differs from lock")
    master = ensure_sample_ids(pd.read_csv(output_dir / "master_manifest.csv"))
    held = ensure_sample_ids(pd.read_csv(output_dir / "holdout.csv"))
    expected = master.loc[master.split.eq("holdout")].reset_index(drop=True)
    columns = [c for c in expected if c in held]
    if set(held.columns) != set(expected.columns) or not held[columns].equals(expected[columns]):
        raise ValueError("Loaded holdout does not equal the frozen master subset")
    return lock
