"""Build complete sample groups from explicitly reviewed same-source relationships."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.identity import ensure_sample_ids
from src.provenance import atomic_json, sha256_file


def build_groups(splits_dir, reviewed_pairs, output_dir):
    if output_dir.exists():
        raise FileExistsError("Choose a new group version")
    frame = ensure_sample_ids(
        pd.concat(
            [pd.read_csv(splits_dir / f"{s}.csv") for s in ("train", "val", "test")],
            ignore_index=True,
        )
    )
    parents = {sid: sid for sid in frame.sample_id}

    def representative(sid):
        while parents[sid] != sid:
            parents[sid] = parents[parents[sid]]
            sid = parents[sid]
        return sid

    reviewed = pd.read_csv(reviewed_pairs, dtype=str)
    accepted = 0
    for row in reviewed.itertuples():
        if row.sample_id_a not in parents or row.sample_id_b not in parents:
            raise ValueError("Reviewed pair is not in the original manifests")
        if any(
            pd.isna(value) or not str(value).strip()
            for value in (row.same_source, row.reviewer, row.reason)
        ) or str(row.same_source).lower() not in {"true", "false"}:
            raise ValueError("Every reviewed relationship needs an explicit verdict and reviewer")
        if row.same_source.lower() == "true":
            a, b = sorted([representative(row.sample_id_a), representative(row.sample_id_b)])
            parents[b] = a
            accepted += 1
    frame["group_id"] = frame.sample_id.map(representative)
    output_dir.mkdir(parents=True)
    frame[["sample_id", "image_path", "group_id"]].to_csv(output_dir / "groups.csv", index=False)
    summary = {
        "status": "partial_review_grouping_not_full_dataset_clearance",
        "rows": len(frame),
        "groups": int(frame.group_id.nunique()),
        "reviewed_same_source_edges": accepted,
        "review_sha256": sha256_file(reviewed_pairs),
        "groups_sha256": sha256_file(output_dir / "groups.csv"),
        "limitations": "Unreviewed pHash candidates and missing plant/session IDs remain",
    }
    atomic_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--reviewed-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_groups(args.splits_dir, args.reviewed_pairs, args.output_dir)
