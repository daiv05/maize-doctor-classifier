import json

import numpy as np
import pandas as pd
import pytest

from src.data.provenance import (
    ordered_manifest_contract,
    validate_feature_cache,
    validate_holdout_lock,
)
from src.provenance import contract_hash, sha256_file


def cache(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.png").write_bytes(b"a")
    (root / "b.png").write_bytes(b"b")
    frame = pd.DataFrame(
        {
            "image_path": ["a.png", "b.png"],
            "label": ["a", "b"],
            "sha256": [sha256_file(root / "a.png"), sha256_file(root / "b.png")],
            "split": ["train", "holdout"],
        }
    )
    frame.to_csv(tmp_path / "master_manifest.csv", index=False)
    (tmp_path / "features").mkdir()
    path = tmp_path / "features" / "tiny.npy"
    np.save(path, np.ones((2, 3)))
    contract = {
        "dataset_root": str(root),
        "manifest": ordered_manifest_contract(frame, root),
        "preprocessing": {"size": 16},
        "backbone_sha256": "weights",
    }
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "contract": contract,
                "contract_sha256": contract_hash(contract),
                "artifact_sha256": sha256_file(path),
            }
        )
    )
    return root, frame


@pytest.mark.parametrize("change", ["order", "file", "preprocessing", "weights"])
def test_cache_invalidates_relevant_changes(tmp_path, change):
    root, frame = cache(tmp_path)
    assert validate_feature_cache(tmp_path, "tiny").shape == (2, 3)
    kwargs = {}
    if change == "order":
        frame.iloc[::-1].to_csv(tmp_path / "master_manifest.csv", index=False)
    elif change == "file":
        (root / "a.png").write_bytes(b"different")
    elif change == "preprocessing":
        kwargs["expected_preprocessing"] = {"size": 32}
    else:
        kwargs["expected_backbone"] = "other weights"
    with pytest.raises(ValueError):
        validate_feature_cache(tmp_path, "tiny", **kwargs)


def test_altered_master_blocks_holdout_even_if_holdout_file_unchanged(tmp_path):
    _, frame = cache(tmp_path)
    frame[frame.split.eq("holdout")].to_csv(tmp_path / "holdout.csv", index=False)
    lock = {
        "holdout_manifest_sha256": sha256_file(tmp_path / "holdout.csv"),
        "master_manifest_sha256": sha256_file(tmp_path / "master_manifest.csv"),
    }
    (tmp_path / "holdout.lock.json").write_text(json.dumps(lock))
    validate_holdout_lock(tmp_path)
    frame.loc[0, "label"] = "different"
    frame.to_csv(tmp_path / "master_manifest.csv", index=False)
    with pytest.raises(ValueError, match="Master manifest"):
        validate_holdout_lock(tmp_path)
