import pandas as pd
import pytest
import yaml
from PIL import Image

from scripts.dataset import download_dataset
from src.data.cross_validation import HierarchicalKFoldSplitter
from src.data.splitter import HierarchicalStratifiedSplitter
from src.provenance import sha256_file


def test_download_failure_keeps_original_and_separates_attempts(tmp_path, monkeypatch):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "keep.txt").write_text("previous version")
    monkeypatch.setattr(download_dataset, "get_dataset_root", lambda: tmp_path)
    paths = []

    def failing_hf(repo, directory, **kwargs):
        paths.append(directory)
        (directory / "incomplete").write_text("partial HF")
        raise OSError("network fixture")

    def failing_drive(repo, directory):
        paths.append(directory)
        assert not (directory / "incomplete").exists()
        raise OSError("other fixture")

    monkeypatch.setattr(download_dataset, "_download_from_hf", failing_hf)
    monkeypatch.setattr(download_dataset, "_download_from_gdrive", failing_drive)
    with pytest.raises(RuntimeError, match="Ninguna descarga"):
        download_dataset.download_clean_dataset(force=True, hf_repo="fixture", gdrive_id="fixture")
    assert paths[0] != paths[1]
    assert (clean / "keep.txt").read_text() == "previous version"


def test_related_samples_stay_together_in_split_and_cv():
    frame = pd.DataFrame(
        [
            {"group_id": str(g), "label": label, "environment": "real"}
            for g in range(40)
            for label in ("healthy", "rust")
        ]
    )
    parts = HierarchicalStratifiedSplitter().split(frame, 0.7, 0.15, 0.15)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert set(parts[a].group_id).isdisjoint(parts[b].group_id)
    for fold in HierarchicalKFoldSplitter(3).split(frame):
        assert set(fold.train_df.group_id).isdisjoint(fold.val_df.group_id)


def test_explicit_quarantine_and_path_groups_preserve_originals(tmp_path, monkeypatch):
    from scripts.pipeline import create_splits

    root = tmp_path / "data"
    rows = []
    for c, label in enumerate(("healthy", "fall_armyworm")):
        directory = root / "clean" / label / "real"
        directory.mkdir(parents=True)
        for i in range(20):
            path = directory / f"{i}.png"
            Image.new("RGB", (8, 8), (c * 100 + i, 30, 50)).save(path)
            rows.append(
                {
                    "image_path": path.relative_to(root).as_posix(),
                    "group_id": f"{label}-{i if i > 1 else 0}",
                }
            )
    other = root / "clean/fall_armyworm/real/0.png"
    duplicate = root / "clean/healthy/real/conflict.png"
    duplicate.write_bytes(other.read_bytes())
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"classes": ["healthy", "fall_armyworm"], "seed": 42},
                "paths": {"raw_dir": "clean", "split_output_dir": "splits"},
            }
        )
    )
    monkeypatch.setattr(create_splits, "get_dataset_root", lambda: root)
    monkeypatch.setattr(create_splits, "get_output_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="etiquetas conflictivas"):
        create_splits.run_data_preparation_pipeline(str(config), output_dir=tmp_path / "conflict")
    exclusions = tmp_path / "exclusions.csv"
    excluded = pd.DataFrame(
        [
            {
                "image_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "reason": "Conflicting labels; keep source unchanged pending expert review",
            }
            for path in (duplicate, other)
        ]
    )
    excluded.to_csv(exclusions, index=False)
    groups = tmp_path / "groups.csv"
    pd.DataFrame(rows).to_csv(groups, index=False)
    output = tmp_path / "quarantined"
    create_splits.run_data_preparation_pipeline(
        str(config), output_dir=output, exclusions=exclusions, group_manifest=groups
    )
    master = pd.read_csv(output / "master_manifest.csv")
    assert len(master) == 39
    assert set(master.image_path).isdisjoint(excluded.image_path)
    parts = [pd.read_csv(output / f"{s}.csv") for s in ("train", "val", "test")]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert set(parts[a].group_id).isdisjoint(parts[b].group_id)
    for row in excluded.itertuples():
        assert sha256_file(root / row.image_path) == row.sha256
    from scripts.etapa_2.stage2_experiments import prepare

    stage2 = tmp_path / "stage2"
    prepare(root, stage2, source_splits=output)
    holdout = pd.read_csv(stage2 / "holdout.csv")
    assert set(holdout.sample_id) == set(parts[2].sample_id)
    assert set(holdout.group_id) == set(parts[2].group_id)
    (output / "val.csv").write_text("altered source")
    with pytest.raises(ValueError, match="Source split differs"):
        prepare(root, tmp_path / "altered-stage2", source_splits=output)
    excluded.loc[0, "sha256"] = "0" * 64
    excluded.to_csv(exclusions, index=False)
    with pytest.raises(ValueError, match="Contenido de exclusión cambió"):
        create_splits.run_data_preparation_pipeline(
            str(config), output_dir=tmp_path / "changed", exclusions=exclusions
        )


@pytest.mark.parametrize("reviewer,reason", [(None, "same capture"), ("person", " ")])
def test_reviewed_groups_require_nonempty_review_metadata(tmp_path, reviewer, reason):
    from scripts.checks.build_reviewed_groups import build_groups
    from src.data.identity import ensure_sample_ids

    splits = tmp_path / "splits"
    splits.mkdir()
    frame = ensure_sample_ids(
        pd.DataFrame(
            [{"image_path": f"clean/healthy/real/{i}.png", "label": "healthy"} for i in range(3)]
        )
    )
    for i, split in enumerate(("train", "val", "test")):
        frame.iloc[[i]].to_csv(splits / f"{split}.csv", index=False)
    review = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {
                "sample_id_a": frame.sample_id[0],
                "sample_id_b": frame.sample_id[1],
                "same_source": "true",
                "reviewer": reviewer,
                "reason": reason,
            }
        ]
    ).to_csv(review, index=False)
    with pytest.raises(ValueError, match="explicit verdict and reviewer"):
        build_groups(splits, review, tmp_path / "groups")
