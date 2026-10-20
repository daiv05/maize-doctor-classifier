"""Read-only split/identity preflight for the fixed 3-scenario stability study."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.preparation import sha256_file

FROZEN = PROJECT_ROOT / "outputs/multiseed-inputs/frozen/seed_42"
LOSO = PROJECT_ROOT / "outputs/loso/efficientnet_lite0_baseline"
INVENTORY = (
    PROJECT_ROOT
    / "docs/es/reproducibilidad/evidencia"
    / "pixel_duplicate_audit/pixel_hash_inventory.csv"
)
SOURCES = ("maize-diseases", "multicrop-disease-maiz")


def audit(splits: Path, final_name: str, inventory: pd.DataFrame) -> dict:
    """Compare all three partitions by sample, file bytes, and decoded pixels."""
    frames = {
        name: pd.read_csv(splits / f"{name}.csv", dtype={"sample_id": str})
        for name in ("train", "val", final_name)
    }
    index = inventory.set_index("sample_id", verify_integrity=True)
    identity = {}
    for name, frame in frames.items():
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"sample_id inválido: {name}")
        if not set(frame["sample_id"]) <= set(index.index):
            raise ValueError(f"Muestras fuera del inventario: {name}")
        aligned = index.loc[frame["sample_id"]]
        if (
            aligned["relative_path"].tolist() != frame["image_path"].tolist()
            or aligned["label"].tolist() != frame["label"].tolist()
        ):
            raise ValueError(f"Identidad/label distinta del inventario: {name}")
        identity[name] = {
            "sample_id": set(frame["sample_id"]),
            "file_sha256": set(aligned["file_sha256"]),
            "pixel_sha256": set(aligned["pixel_sha256"]),
        }
    overlap = {}
    names = list(identity)
    for key in ("sample_id", "file_sha256", "pixel_sha256"):
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                count = len(identity[left][key] & identity[right][key])
                overlap[f"{left}/{right}/{key}"] = count
                if count:
                    raise ValueError(f"Solapamiento {left}/{right}/{key}: {count}")
    return {
        "split_lock_sha256": sha256_file(splits / "manifest.lock.json"),
        "csv_sha256": {name: sha256_file(splits / f"{name}.csv") for name in names},
        "samples": {name: len(frame) for name, frame in frames.items()},
        "overlap": overlap,
    }


def main() -> None:
    inventory = pd.read_csv(INVENTORY, dtype={"sample_id": str})
    if (
        inventory["sample_id"].isna().any()
        or inventory["sample_id"].duplicated().any()
        or inventory[["file_sha256", "pixel_sha256"]].isna().any().any()
    ):
        raise ValueError("Inventario de identidad incompleto")
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "pixel_inventory_sha256": sha256_file(INVENTORY),
        "baseline": audit(FROZEN, "test", inventory),
    }
    for source in SOURCES:
        splits = LOSO / source / "seed_42/splits"
        result[source] = audit(splits, "holdout", inventory)
        holdout = pd.read_csv(splits / "holdout.csv", usecols=["source_id"])
        if set(holdout["source_id"]) != {source}:
            raise ValueError(f"Holdout mezcla fuentes: {source}")
    output = (
        PROJECT_ROOT
        / "docs/es/reproducibilidad/evidencia"
        / "multiseed_source_stability/integrity_preflight.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key]["overlap"] for key in ("baseline", *SOURCES)}, indent=2))


if __name__ == "__main__":
    main()
