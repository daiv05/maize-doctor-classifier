import csv
from pathlib import Path

from scripts.dataset.upload_to_hf import (
    SOURCE_LICENSES,
    pending_uploads,
    write_metadata_csv,
)
from src.data.provenance import PROVENANCE_GROUPS


def test_every_provenance_group_declares_a_license():
    assert sorted(SOURCE_LICENSES) == PROVENANCE_GROUPS


def test_metadata_csv_records_source_and_license(tmp_path: Path):
    images = [
        (Path("healthy_maize_desease_lab_1a2b.jpg"), "healthy", "lab"),
        (Path("healthy_corn_leaf_roboflow_real_3c4d.jpg"), "healthy", "real"),
    ]
    destination = tmp_path / "metadata.csv"

    write_metadata_csv(images, destination)

    with open(destination, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["source_id"], r["license"]) for r in rows] == [
        ("maize-diseases", "CC-BY-NC-SA-4.0"),
        ("corn-leaf-roboflow", "CC-BY-4.0"),
    ]


def test_metadata_files_are_uploaded_again_and_existing_shards_are_skipped():
    written = [Path("clean-00000.tar"), Path("clean-00001.tar"), Path("metadata.csv")]
    remote = {"clean-00000.tar", "metadata.csv"}

    assert pending_uploads(written, remote) == [Path("clean-00001.tar"), Path("metadata.csv")]
