"""Confirm identical canonical RGB pixels and prepare grouping without altering old splits."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.data.identity import ensure_sample_ids
from src.data.loader import load_and_normalize_image
from src.provenance import atomic_json, sha256_file


def verify(dataset_root, candidates, splits_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError("Choose a new verification directory")
    pairs = pd.read_csv(candidates)
    hashes = {}
    for relative in sorted(set(pairs.image_path_a) | set(pairs.image_path_b)):
        image = load_and_normalize_image(dataset_root / relative)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        hashes[relative] = digest
    pairs["same_canonical_rgb"] = [
        hashes[a] == hashes[b] for a, b in zip(pairs.image_path_a, pairs.image_path_b)
    ]
    frame = ensure_sample_ids(
        pd.concat(
            [pd.read_csv(splits_dir / f"{s}.csv") for s in ("train", "val", "test")],
            ignore_index=True,
        )
    )
    frame["group_id"] = [
        hashes.get(path, sid) for path, sid in zip(frame.image_path, frame.sample_id)
    ]
    output_dir.mkdir(parents=True)
    pairs.to_csv(output_dir / "verified_candidates.csv", index=False)
    frame[["image_path", "group_id"]].to_csv(output_dir / "canonical_rgb_groups.csv", index=False)
    summary = {
        "status": "pixel_equivalence_verified_near_similarity_still_requires_review",
        "candidate_pairs": len(pairs),
        "same_canonical_rgb_pairs": int(pairs.same_canonical_rgb.sum()),
        "same_rgb_cross_split_pairs": int((pairs.same_canonical_rgb & pairs.cross_split).sum()),
        "same_rgb_cross_label_pairs": int((pairs.same_canonical_rgb & pairs.label_conflict).sum()),
        "images_compared": len(hashes),
        "all_images_grouped": len(frame),
        "source_candidates_sha256": sha256_file(candidates),
        "script_sha256": sha256_file(Path(__file__)),
        "artifacts": {p.name: sha256_file(p) for p in output_dir.iterdir()},
        "next_step": "new splits with complete canonical_rgb_groups.csv; review other candidates",
        "interpretation": (
            "Identical decoded RGB across splits confirms duplicate leakage; "
            "inflation not measured."
            if (pairs.same_canonical_rgb & pairs.cross_split).any()
            else "No exact canonical RGB duplicates confirmed; near candidates require review."
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.dataset_root, args.candidates, args.splits_dir, args.output_dir)
