"""Copy a reviewed legacy run into a new, explicitly reconstructed provenance contract.

This does not certify the original dataset bytes or reproduce historical predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from scripts.pipeline.segment_dataset import source_manifest
from src.config import PROJECT_ROOT
from src.data.transforms import CornTransformFactory
from src.provenance import atomic_json, sha256_file
from src.training.runs import load_run


def migrate(source_run, splits_dir, dataset_root, output_dir, config_path):
    source_run, splits_dir = Path(source_run), Path(splits_dir)
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Migration requires a new output directory")
    legacy = json.loads((source_run / "summary.json").read_text())
    if "segmented" in str(legacy.get("config", "")) or legacy.get("segmented"):
        raise ValueError("Legacy segmented preprocessing requires a separate reviewed migration")
    if type(legacy.get("clahe")) is not bool:
        raise ValueError("Legacy CLAHE policy is unknown; do not infer a default")
    size = legacy.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Legacy input size is unknown")
    # Verifies original split identity and all current bytes, without image inference.
    frame = source_manifest(splits_dir, dataset_root)
    frame["sha256"] = frame["source_sha256"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".migration-", dir=output_dir.parent))
    derived_splits = staging / "splits"
    derived_splits.mkdir()
    for split in ("train", "val", "test"):
        frame.loc[frame.split.eq(split)].to_csv(derived_splits / f"{split}.csv", index=False)
    checkpoint = source_run / "best.pth"
    shutil.copy2(checkpoint, staging / "best.pth")
    shutil.copy2(source_run / "summary.json", staging / "legacy_summary.json")
    factory = CornTransformFactory(
        config_path=str(config_path), target_size=tuple(size), clahe=legacy["clahe"]
    )
    download = dataset_root / "clean/download_manifest.json"
    dataset = json.loads(download.read_text()) if download.exists() else {}
    summary = {
        "schema_version": 2,
        "model": legacy["model"],
        "class_to_idx": legacy["class_to_idx"],
        "image_size": size,
        "preprocessing": factory.to_contract(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "splits_dir": str((output_dir / "splits").resolve()),
        "split_sha256": {
            s: sha256_file(derived_splits / f"{s}.csv") for s in ("train", "val", "test")
        },
        "dataset_root": str(dataset_root.resolve()),
        "migration": {
            "status": "reconstructed_for_reevaluation_not_historical_certification",
            "utc": pd.Timestamp.utcnow().isoformat(),
            "source_run": str(source_run.resolve()),
            "legacy_summary_sha256": sha256_file(source_run / "summary.json"),
            "source_split_sha256": {
                s: sha256_file(splits_dir / f"{s}.csv") for s in ("train", "val", "test")
            },
            "dataset_revision_observed": dataset.get("revision"),
            "dataset_manifest_sha256": sha256_file(download) if download.exists() else None,
            "preprocessing_basis": "explicit legacy size/CLAHE and reviewed canonical EXIF/RGB",
            "historical_image_bytes": "not_recorded_in_source_run",
            "historical_preprocessing_equivalence": "unproven",
            "missing_group_metadata": "group_id" not in frame,
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    atomic_json(staging / "summary.json", summary)
    load_run(staging / "best.pth", legacy["model"], config_path=config_path)
    os.replace(staging, output_dir)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "status": summary["migration"]["status"],
                "split_counts": frame.split.value_counts().to_dict(),
            },
            indent=2,
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/dataset.yaml")
    parser.add_argument(
        "--acknowledge-legacy",
        action="store_true",
        required=True,
        help="Acknowledge that this cannot certify historical preprocessing/data",
    )
    args = parser.parse_args()
    migrate(args.source_run, args.splits_dir, args.dataset_root, args.output_dir, args.config)


if __name__ == "__main__":
    main()
