"""Integrity tests for the two frozen, pre-registered LOSO partitions."""

import json

import pytest

from scripts.experiments.prepare_loso_baseline import (
    FROZEN,
    INVENTORY,
    SOURCES,
    derive,
    load_frozen,
    materialize,
    sha256,
)


@pytest.fixture(scope="module")
def frozen_data():
    frames, _, inventory = load_frozen(FROZEN, INVENTORY)
    return frames, inventory


@pytest.mark.parametrize("source", SOURCES)
def test_complete_exclusion_and_coverage(source, frozen_data):
    frames, inventory = frozen_data
    derived, report = derive(source, frames, inventory)
    assert source not in set(derived["train"]["source_id"])
    assert source not in set(derived["val"]["source_id"])
    assert set(derived["holdout"]["source_id"]) == {source}
    expected = set(inventory.loc[inventory["source_id"] == source, "sample_id"])
    assert set(derived["holdout"]["sample_id"]) == expected
    assert set(derived["holdout"]["label"]) <= set(derived["train"]["label"])
    assert all(report["checks"].values())
    for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
        assert set(derived[a]["sample_id"]).isdisjoint(derived[b]["sample_id"])
    assert report["counts"]["holdout"] == len(expected)
    assert sum(report["historical_holdout_counts"].values()) == len(expected)


@pytest.mark.parametrize("source", SOURCES)
def test_reproducible_hashes_and_immutable_lock(source, frozen_data, tmp_path):
    first = materialize(source, FROZEN, INVENTORY, tmp_path)
    second = materialize(source, FROZEN, INVENTORY, tmp_path)
    assert first == second
    assert sha256(FROZEN / "manifest.lock.json") == first["frozen_sha256"][
        "manifest.lock.json"
    ]
    lock_path = tmp_path / source / "seed_42" / "splits" / "manifest.lock.json"
    assert sha256(lock_path) == first["lock_sha256"]
    assert json.loads(lock_path.read_text())["held_out_source"] == source
    with (lock_path.parent / "holdout.csv").open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    with pytest.raises(ValueError, match="Manifiesto derivado existente distinto"):
        materialize(source, FROZEN, INVENTORY, tmp_path)


def test_pixel_overlap_is_rejected(frozen_data):
    frames, inventory = frozen_data
    damaged = inventory.copy()
    source = SOURCES[0]
    train_id = frames["train"].loc[frames["train"]["source_id"] != source,
                                   "sample_id"].iloc[0]
    holdout_pixel = damaged.loc[damaged["source_id"] == source,
                                "pixel_sha256"].iloc[0]
    damaged.loc[damaged["sample_id"] == train_id, "pixel_sha256"] = holdout_pixel
    with pytest.raises(ValueError, match="Solapamiento de pixel_sha256"):
        derive(source, frames, damaged)
