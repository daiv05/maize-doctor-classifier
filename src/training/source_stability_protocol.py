"""Protocolo congelado de estabilidad por fuente, sin dependencia de Modal."""

import json
from pathlib import Path

from src.data.preparation import sha256_file

LOCK_SHA256 = {
    "baseline": "0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8",
    "maize-diseases": "6486473932ae2937ff5a97881953d6ffbfac3725aa81170ef2b20a6bb49cbc07",
    "multicrop-disease-maiz": "d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b",
}


def verify_splits(scenario: str, splits: Path) -> dict:
    """Check byte-identical manifests; the LOSO holdout is hashed, never opened."""
    lock_path = splits / "manifest.lock.json"
    if sha256_file(lock_path) != LOCK_SHA256[scenario]:
        raise ValueError(f"Manifest lock distinto: {scenario}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if scenario == "baseline":
        expected = {
            "master_manifest.csv": lock["master_manifest_sha256"],
            "train.csv": lock["train_sha256"],
            "val.csv": lock["val_sha256"],
            "test.csv": lock["test_sha256"],
        }
    else:
        if lock["protocol"] != "loso_baseline_seed42_v1" or lock["held_out_source"] != scenario:
            raise ValueError(f"Protocolo LOSO distinto: {scenario}")
        expected = lock["derived_sha256"]
        frozen = Path("/outputs/splits/seed_42")
        verify_splits("baseline", frozen)
        for name, digest in lock["frozen_sha256"].items():
            if sha256_file(frozen / name) != digest:
                raise ValueError(f"Input congelado distinto: {name}")
    for name, digest in expected.items():
        if sha256_file(splits / name) != digest:
            raise ValueError(f"CSV distinto: {scenario}/{name}")
    return lock
