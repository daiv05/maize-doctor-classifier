"""Bind pre-segmented datasets and raw-image inference to the same segmenter contract."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.provenance import contract_hash, sha256_file


def bind_segmented_splits(factory, splits_dir: Path) -> None:
    marker = Path(splits_dir) / "segmentation.json"
    if not marker.is_file():
        if factory.segmentation_contract:
            raise ValueError("Segmented configuration requires segmentation.json")
        return
    payload = json.loads(marker.read_text())
    contract = payload["contract"]
    if not payload.get("complete") or contract_hash(contract) != payload["contract_id"]:
        raise ValueError("Incomplete or altered segmentation contract")
    for split, digest in payload["split_hashes"].items():
        if sha256_file(Path(splits_dir) / f"{split}.csv") != digest:
            raise ValueError(f"Altered segmented split: {split}")
    factory.segmentation_contract = {
        "contract_id": payload["contract_id"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "processor": contract["processor"],
        "uncertain_policy": contract["uncertain_policy"],
        "processor_source_sha256": contract["processor_source_sha256"],
        "detector": contract["detector"],
        "pillow_version": contract["pillow_version"],
    }


@lru_cache(maxsize=2)
def _segmenter(checkpoint: str, device: str | None, checkpoint_sha256: str):
    from src.segmentation.detector import MaizeLeafSegmenter

    if sha256_file(Path(checkpoint)) != checkpoint_sha256:
        raise ValueError("Segmenter checkpoint changed before loading")
    return MaizeLeafSegmenter(checkpoint_path=checkpoint, device=device)


def prepare_segmented_inference(image, contract, checkpoint=None, device=None):
    if not contract:
        return image, {"segmentation_status": "not_applicable"}
    if checkpoint is None or sha256_file(Path(checkpoint)) != contract["checkpoint_sha256"]:
        raise ValueError("Segmented run requires its exact --segmenter-checkpoint")
    from PIL import __version__ as pillow_version

    import src.segmentation.leaf_processor as processor_module
    from src.segmentation.detector import segmentation_runtime_contract
    from src.segmentation.leaf_processor import LeafMaskProcessorConfig, SegmentedLeafProcessor

    if sha256_file(Path(processor_module.__file__)) != contract["processor_source_sha256"]:
        raise ValueError("Segmenter processing code differs from training contract")
    if (
        contract.get("detector") != segmentation_runtime_contract()
        or contract.get("pillow_version") != pillow_version
    ):
        raise ValueError("Segmenter runtime/configuration differs from training contract")
    config = LeafMaskProcessorConfig(**contract["processor"])
    result = SegmentedLeafProcessor(config).process(
        image,
        _segmenter(
            str(Path(checkpoint).resolve()),
            str(device) if device else None,
            contract["checkpoint_sha256"],
        ).segment(image),
    )
    policy = contract["uncertain_policy"]
    if result.status == "uncertain" and policy == "reject":
        raise ValueError("Uncertain segmentation rejected by run policy")
    fallback = result.status == "rejected" or (
        result.status == "uncertain" and policy == "original"
    )
    return (image if fallback else result.processed_image), {
        "segmentation_status": result.status,
        "fallback_original": fallback,
        "quality": result.quality,
        "warnings": result.warnings,
    }
