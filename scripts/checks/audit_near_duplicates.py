"""Candidate audit using perceptual hashes; no automatic deletion or leakage claim."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.pipeline.segment_dataset import source_manifest
from src.data.loader import load_and_normalize_image
from src.provenance import atomic_json, sha256_file


def perceptual_hash(path):
    image = load_and_normalize_image(path).convert("L").resize((32, 32))
    coefficients = cv2.dct(np.asarray(image, dtype=np.float32))[:8, :8].flatten()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return sum(int(bit) << index for index, bit in enumerate(bits))


def candidate_pairs(hashes, radius=4):
    """Exact Hamming search within the radius via r+1 pigeonhole partitions."""
    if not 0 <= radius <= 8:
        raise ValueError("Radius must be in [0, 8]")
    width = 64 // (radius + 1)
    buckets = [{} for _ in range(radius + 1)]
    for index, value in enumerate(hashes):
        candidates = set()
        keys = []
        for part, bucket in enumerate(buckets):
            bits = width if part < radius else 64 - width * part
            key = (value >> (part * width)) & ((1 << bits) - 1)
            keys.append(key)
            candidates.update(bucket.get(key, []))
        for other in sorted(candidates):
            distance = (value ^ hashes[other]).bit_count()
            if distance <= radius:
                yield other, index, distance
        for bucket, key in zip(buckets, keys):
            bucket.setdefault(key, []).append(index)


def audit(dataset_root, splits_dir, output_dir, radius=4, workers=4):
    if output_dir.exists():
        raise FileExistsError("Choose a new audit directory")
    frame = source_manifest(splits_dir, dataset_root)
    cv2.setNumThreads(1)
    paths = [dataset_root / path for path in frame.image_path]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(perceptual_hash, paths))
    output_dir.mkdir(parents=True)
    frame["phash64"] = [f"{value:016x}" for value in hashes]
    frame.to_csv(output_dir / "perceptual_hashes.csv", index=False)
    rows = []
    for first, second, distance in candidate_pairs(hashes, radius):
        a, b = frame.iloc[first], frame.iloc[second]
        rows.append(
            {
                "sample_id_a": a.sample_id,
                "sample_id_b": b.sample_id,
                "image_path_a": a.image_path,
                "image_path_b": b.image_path,
                "label_a": a.label,
                "label_b": b.label,
                "split_a": a.split,
                "split_b": b.split,
                "hamming_distance": distance,
                "cross_split": bool(a.split != b.split),
                "label_conflict": bool(a.label != b.label),
                "exact_bytes": bool(a.source_sha256 == b.source_sha256),
                "human_same_source": "",
                "human_notes": "",
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "candidates.csv", index=False)
    summary = {
        "status": "candidate_audit_requires_human_review",
        "images": len(frame),
        "radius": radius,
        "pairs": len(rows),
        "cross_split_candidate_pairs": sum(row["cross_split"] for row in rows),
        "cross_label_candidate_pairs": sum(row["label_conflict"] for row in rows),
        "exact_bytes_pairs": sum(row["exact_bytes"] for row in rows),
        "limitations": "pHash similarity is not proof of common plant/session or leakage",
        "source_split_sha256": {
            s: sha256_file(splits_dir / f"{s}.csv") for s in ("train", "val", "test")
        },
        "artifact_sha256": {p.name: sha256_file(p) for p in output_dir.iterdir()},
        "script_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    audit(args.dataset_root, args.splits_dir, args.output_dir, args.radius, args.workers)


if __name__ == "__main__":
    main()
