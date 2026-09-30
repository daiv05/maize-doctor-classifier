"""Derive the two pre-registered LOSO manifests from the frozen seed_42 inputs.

This command only reads manifests and writes derived CSV/JSON. It never loads images,
trains a model, or evaluates the held-out source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

SOURCES = ("maize-diseases", "multicrop-disease-maiz")
FROZEN = PROJECT_ROOT / "outputs/multiseed-inputs/frozen/seed_42"
INVENTORY = (
    PROJECT_ROOT
    / "docs/es/reproducibilidad/evidencia/pixel_duplicate_audit/pixel_hash_inventory.csv"
)
OUTPUT = PROJECT_ROOT / "outputs/loso/efficientnet_lite0_baseline"
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(frame: pd.DataFrame) -> set[str]:
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id ausente o duplicado")
    return set(frame["sample_id"])


def load_frozen(frozen: Path, inventory_path: Path) -> tuple[dict, dict, pd.DataFrame]:
    lock_path = frozen / "manifest.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    hashes = {"manifest.lock.json": sha256(lock_path)}
    for name, key in (("master_manifest.csv", "master_manifest_sha256"),
                      ("train.csv", "train_sha256"),
                      ("val.csv", "val_sha256"),
                      ("test.csv", "test_sha256")):
        actual = sha256(frozen / name)
        if actual != lock[key]:
            raise ValueError(f"Hash congelado diferente para {name}: {actual}")
        hashes[name] = actual
    master = pd.read_csv(frozen / "master_manifest.csv", dtype={"sample_id": str})
    frames = {name: pd.read_csv(frozen / f"{name}.csv", dtype={"sample_id": str})
              for name in SPLITS}
    master_ids = _ids(master)
    split_ids = {name: _ids(frame) for name, frame in frames.items()}
    if set.union(*split_ids.values()) != master_ids:
        raise ValueError("Los tres splits no cubren exactamente el master")
    if any(split_ids[a] & split_ids[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Solapamiento de sample_id en splits congelados")
    lookup = master.set_index("sample_id")
    for name, frame in frames.items():
        expected = lookup.loc[frame["sample_id"]]
        for column in ("image_path", "label", "source_id"):
            if not (frame[column].to_numpy() == expected[column].to_numpy()).all():
                raise ValueError(f"{name}: {column} difiere del master")

    inventory = pd.read_csv(inventory_path, dtype={"sample_id": str})
    if _ids(inventory) != master_ids or inventory["error"].notna().any():
        raise ValueError("Inventario de píxeles incompleto o con errores")
    aligned = inventory.set_index("sample_id").loc[master["sample_id"]]
    if not (aligned["file_sha256"].to_numpy() == master["sha256"].to_numpy()).all():
        raise ValueError("SHA-256 de archivo difiere entre master e inventario")
    for column, reference in (("source_id", "source_id"), ("relative_path", "image_path")):
        if not (aligned[column].to_numpy() == master[reference].to_numpy()).all():
            raise ValueError(f"Inventario: {column} difiere del master")
    if aligned["pixel_sha256"].isna().any():
        raise ValueError("Inventario sin pixel_sha256 completo")
    return frames, hashes, inventory


def derive(source: str, frames: dict[str, pd.DataFrame], inventory: pd.DataFrame
           ) -> tuple[dict[str, pd.DataFrame], dict]:
    if source not in SOURCES:
        raise ValueError(f"Fuente no predefinida: {source}")
    train = frames["train"].loc[frames["train"]["source_id"] != source].copy()
    val = frames["val"].loc[frames["val"]["source_id"] != source].copy()
    pieces = []
    for name in SPLITS:
        piece = frames[name].loc[frames[name]["source_id"] == source].copy()
        piece["historical_split"] = name
        pieces.append(piece)
    holdout = pd.concat(pieces, ignore_index=True)
    result = {"train": train, "val": val, "holdout": holdout}
    sets = {name: _ids(frame) for name, frame in result.items()}
    if any(sets[a] & sets[b] for a, b in
           (("train", "val"), ("train", "holdout"), ("val", "holdout"))):
        raise ValueError("Solapamiento de sample_id en LOSO")
    all_source_ids = set.union(*(
        set(frame.loc[frame["source_id"] == source, "sample_id"])
        for frame in frames.values()
    ))
    if sets["holdout"] != all_source_ids:
        raise ValueError("El holdout no cubre toda la fuente")
    train_classes = set(train["label"])
    holdout_classes = set(holdout["label"])
    if train_classes != set(frames["train"]["label"]):
        raise ValueError("Alguna clase global desapareció del train")
    if not holdout_classes <= train_classes:
        raise ValueError("Clase holdout ausente del train")
    if val.empty or train.empty or holdout.empty:
        raise ValueError("Split LOSO vacío")
    audit = inventory.set_index("sample_id")
    for column in ("file_sha256", "pixel_sha256"):
        values = {name: set(audit.loc[frame["sample_id"], column])
                  for name, frame in result.items()}
        for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
            if values[a] & values[b]:
                raise ValueError(f"Solapamiento de {column} entre {a} y {b}")
    counts = {name: len(frame) for name, frame in result.items()}
    audit_summary = {
        "source_id": source,
        "counts": counts,
        "historical_holdout_counts": {name: int((holdout["historical_split"] == name).sum())
                                      for name in SPLITS},
        "holdout_class_counts": {name: int(value) for name, value in
                                 holdout["label"].value_counts().sort_index().items()},
        "remaining_train_class_counts": {name: int(value) for name, value in
                                         train["label"].value_counts().sort_index().items()},
        "original_train": len(frames["train"]),
        "train_delta": len(train) - len(frames["train"]),
        "train_delta_percent": 100 * (len(train) / len(frames["train"]) - 1),
        "checks": {
            "sample_id_disjoint": True,
            "file_sha256_disjoint": True,
            "pixel_sha256_disjoint": True,
            "holdout_complete": True,
            "all_holdout_classes_in_train": True,
        },
    }
    return result, audit_summary


def materialize(source: str, frozen: Path, inventory_path: Path, output: Path) -> dict:
    frames, frozen_hashes, inventory = load_frozen(frozen, inventory_path)
    derived, report = derive(source, frames, inventory)
    target = output / source / "seed_42" / "splits"
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in derived.items():
        path = target / f"{name}.csv"
        contents = frame.to_csv(index=False, lineterminator="\n")
        if path.exists() and path.read_text(encoding="utf-8") != contents:
            raise ValueError(f"Manifiesto derivado existente distinto: {path}")
        path.write_text(contents, encoding="utf-8")
    lock = {
        "schema_version": 1,
        "protocol": "loso_baseline_seed42_v1",
        "held_out_source": source,
        "seed": 42,
        "historical_split_policy": "train/val filtered; non-holdout test unused",
        "holdout_policy": "all eligible source rows from historical train/val/test",
        "frozen_sha256": frozen_hashes,
        "pixel_inventory_sha256": sha256(inventory_path),
        "derived_sha256": {f"{name}.csv": sha256(target / f"{name}.csv")
                           for name in derived},
        "preflight": report,
    }
    lock_text = json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    lock_path = target / "manifest.lock.json"
    if lock_path.exists() and lock_path.read_text(encoding="utf-8") != lock_text:
        raise ValueError(f"Lock derivado existente distinto: {lock_path}")
    lock_path.write_text(lock_text, encoding="utf-8")
    return {"lock_path": str(lock_path), "lock_sha256": sha256(lock_path), **lock}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    results = [materialize(source, args.frozen, args.inventory, args.output)
               for source in SOURCES]
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
