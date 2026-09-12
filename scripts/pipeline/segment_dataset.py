"""Segmentación auditable: mismos IDs/splits, recibos por imagen y reanudación verificada."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from PIL import __version__ as pillow_version

from src.config import get_dataset_root
from src.data.identity import ensure_sample_ids
from src.data.loader import load_and_normalize_image as load_image
from src.provenance import atomic_json, contract_hash, sha256_file
from src.segmentation.detector import MaizeLeafSegmenter, segmentation_runtime_contract
from src.segmentation.leaf_processor import (
    CROP_MASK_LETTERBOX,
    SUPPORTED_MASK_PROFILES,
    LeafMaskProcessorConfig,
    SegmentedLeafProcessor,
    build_comparison_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Splits originales congelados; nunca se vuelven a sortear.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=sorted(SUPPORTED_MASK_PROFILES), default=CROP_MASK_LETTERBOX
    )
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-previews", type=int, default=50)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--uncertain-policy", choices=["original", "reject", "accept"], default="original"
    )
    return parser.parse_args()


def source_manifest(splits_dir: Path, dataset_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        frame = ensure_sample_ids(pd.read_csv(splits_dir / f"{split}.csv"))
        frame["split"] = split
        frames.append(frame)
    frame = ensure_sample_ids(pd.concat(frames, ignore_index=True))
    if "group_id" in frame:
        if frame.group_id.isna().any() or (frame.groupby("group_id").split.nunique() > 1).any():
            raise ValueError("Procedencia de grupo incompleta o compartida entre splits")
    hashes = []
    for row in frame.itertuples():
        path = (dataset_dir / row.image_path).resolve()
        if not path.is_relative_to(dataset_dir.resolve()):
            raise ValueError(f"Ruta fuera del dataset: {row.image_path}")
        digest = sha256_file(path)
        if hasattr(row, "sha256") and row.sha256 != digest:
            raise ValueError(f"Fuente modificada: {row.sample_id}")
        hashes.append(digest)
    frame["source_sha256"] = hashes
    # Un mismo contenido no puede cruzar particiones ni tener etiquetas distintas.
    for digest, group in frame.groupby("source_sha256"):
        if group["label"].nunique() > 1 or group["split"].nunique() > 1:
            raise ValueError(f"Duplicado conflictivo/entre splits: {digest}")
    return frame


def atomic_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def run_segmentation(args: argparse.Namespace) -> dict:
    dataset_dir = args.dataset_dir or get_dataset_root()
    output_dir = (
        args.output_dir
        or Path(os.environ.get("SEGMENTED_DATASET_ROOT", str(get_dataset_root() / "segmented")))
        / "clean"
    )
    if dataset_dir.resolve() == output_dir.resolve():
        raise ValueError("El original es inmutable: entrada y salida deben diferir.")
    frame = source_manifest(args.splits_dir, dataset_dir)
    config = LeafMaskProcessorConfig(
        processing_profile=args.profile, target_size=tuple(args.target_size)
    )
    contract = {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "processor": asdict(config),
        "uncertain_policy": args.uncertain_policy,
        "source_rows": frame.fillna("").to_dict("records"),
        "exif": True,
        "mode": "RGB",
        "pillow_version": pillow_version,
        "detector": segmentation_runtime_contract(),
        "processor_source_sha256": sha256_file(
            Path(__file__).parents[2] / "src/segmentation/leaf_processor.py"
        ),
    }
    # Normalize tuples as serialized JSON for exact equality on resume.
    contract = json.loads(json.dumps(contract))
    contract_id = contract_hash(contract)
    audit = output_dir / ".segmentation"
    if audit.joinpath("contract.json").exists():
        if json.loads(audit.joinpath("contract.json").read_text()) != contract:
            raise ValueError("Salida incompatible; use un output-dir nuevo.")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Salida preexistente sin contrato; no se puede reusar.")
    audit.mkdir(parents=True, exist_ok=True)
    atomic_json(audit / "contract.json", contract)
    records_dir = audit / "records"
    records_dir.mkdir(exist_ok=True)
    processor = SegmentedLeafProcessor(config=config)
    segmenter = MaizeLeafSegmenter(checkpoint_path=args.checkpoint, device=args.device)
    records, errors = [], []
    selected = frame.iloc[: args.max_images] if args.max_images > 0 else frame
    preview_dir = args.preview_dir or audit / "previews"
    preview_counts = {}
    for source in selected.to_dict("records"):
        sid = source["sample_id"]
        filename = contract_hash({"sample_id": sid}) + ".png"
        destination = (
            Path(str(source["label"])) / str(source.get("environment", "unknown")) / filename
        )
        if not (output_dir / destination).resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("Etiqueta/ambiente contiene una ruta insegura.")
        receipt = records_dir / (Path(filename).stem + ".json")
        if receipt.exists():
            cached = json.loads(receipt.read_text())
            if (
                cached.get("contract_id") == contract_id
                and cached.get("source_sha256") == source["source_sha256"]
                and (output_dir / cached["output_relative_path"]).is_file()
                and sha256_file(output_dir / cached["output_relative_path"]) == cached["sha256"]
            ):
                records.append(cached)
                key = (cached["label"], cached.get("environment"), cached["segmentation_status"])
                preview_counts[key] = preview_counts.get(key, 0) + 1
                continue
            raise ValueError(f"Recibo/salida alterado: {sid}; use una versión nueva.")
        try:
            original = load_image(dataset_dir / source["image_path"])
            result = processor.process(
                original, segmenter.segment(original), source_image=source["image_path"]
            )
            use_original = result.status == "rejected" or (
                result.status == "uncertain" and args.uncertain_policy == "original"
            )
            if result.status == "uncertain" and args.uncertain_policy == "reject":
                raise ValueError("Máscara incierta rechazada por política explícita.")
            output = original if use_original else result.processed_image
            if output is None:
                raise ValueError("No hay imagen de salida.")
            atomic_image(output, output_dir / destination)
            record = {
                **source,
                "original_image_path": source["image_path"],
                "image_path": (Path(output_dir.name) / destination).as_posix(),
                "output_relative_path": destination.as_posix(),
                "contract_id": contract_id,
                "sha256": sha256_file(output_dir / destination),
                "segmentation_status": result.status,
                "fallback_original": use_original,
                "quality": result.quality,
                "warnings": result.warnings,
            }
            atomic_json(receipt, record)
            records.append(record)
            key = (record["label"], record.get("environment"), record["segmentation_status"])
            preview_counts[key] = preview_counts.get(key, 0) + 1
            if args.max_previews > 0 and preview_counts[key] <= 5:
                atomic_image(
                    build_comparison_panel(result), audit / "preview_candidates" / filename
                )
        except Exception as error:
            errors.append(
                {
                    "sample_id": sid,
                    "image_path": source["image_path"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    complete = not errors and len(records) == len(frame)
    summary = {
        "contract_id": contract_id,
        "total": len(frame),
        "processed": len(records),
        "complete": complete,
        "errors": errors,
        "status_counts": pd.Series([r["segmentation_status"] for r in records], dtype="str")
        .value_counts()
        .to_dict(),
        "fallback_original": sum(r["fallback_original"] for r in records),
        "threshold_calibration": "not_performed",
        "human_review": "pending",
    }
    atomic_json(audit / "summary.json", summary)
    atomic_json(audit / "manifest.json", records)
    if records:
        review = pd.DataFrame(records)
        keys = [k for k in ("label", "environment", "segmentation_status") if k in review]
        review = review.groupby(keys, dropna=False, sort=True).head(5).copy()
        review["stratum_rank"] = review.groupby(keys, dropna=False).cumcount()
        review = review.sort_values(["stratum_rank", *keys, "sample_id"])
        review["preview_path"] = ""
        for index, row in review.head(args.max_previews).iterrows():
            filename = Path(row["output_relative_path"]).name
            candidate = audit / "preview_candidates" / filename
            if candidate.is_file():
                atomic_image(load_image(candidate), preview_dir / filename)
                review.loc[index, "preview_path"] = str(preview_dir / filename)
        review["human_accept"] = ""
        review["lesion_preserved"] = ""
        review["review_notes"] = ""
        review_path = audit / "human_review.csv"
        if not review_path.exists():  # Never erase manual annotations on resume.
            review.to_csv(review_path, index=False)
    if complete:
        splits = audit / "splits"
        splits.mkdir(exist_ok=True)
        derived = pd.DataFrame(records).drop(columns=["quality", "warnings"])
        for split in ("train", "val", "test"):
            temporary = splits / f"{split}.partial"
            derived[derived["split"] == split].to_csv(temporary, index=False)
            os.replace(temporary, splits / f"{split}.csv")
        atomic_json(
            splits / "segmentation.json",
            {
                "contract_id": contract_id,
                "contract": contract,
                "complete": True,
                "split_hashes": {
                    s: sha256_file(splits / f"{s}.csv") for s in ("train", "val", "test")
                },
            },
        )
    if errors:
        raise RuntimeError(f"{len(errors)} imágenes fallaron; ver {audit / 'summary.json'}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_segmentation(parse_args())
