"""Guards for the frozen historical ensemble LOSO protocol."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from scripts.experiments import ensemble_loso_finalize as finalize
from scripts.experiments import ensemble_loso_report as report
from scripts.modal import ensemble_loso_finalize as modal_finalize
from scripts.modal import ensemble_loso_train as train
from src.models.ensemble import SoftVotingEnsemble


class ConstantModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, batch):
        return self.logits.expand(batch.shape[0], -1)


def test_only_six_missing_slots_and_historical_training_flags():
    assert len(train.SEEDS) * len(train.MODELS) == 6
    assert train.MODELS == ("efficientnet_b0", "shufflenet_v2_x1_0")
    for seed in train.SEEDS:
        for model in train.MODELS:
            cmd = train.command(model, seed, train.ROOT / f"seed_{seed}" / model)
            assert cmd[cmd.index("--models") + 1] == model
            assert cmd[cmd.index("--seed") + 1] == str(seed)
            assert cmd[cmd.index("--epochs") + 1] == "35"
            assert cmd[cmd.index("--batch-size") + 1] == "64"
            assert cmd[cmd.index("--learning-rate") + 1] == "0.0004548"
            assert "--skip-test" in cmd
            assert "--no-clahe" in cmd
            assert "--best-params" not in cmd
            assert "holdout.csv" not in " ".join(cmd)
    with pytest.raises(ValueError):
        train.command("efficientnet_lite0", 42, train.ROOT)


def test_fusion_matches_historical_soft_voting():
    models = [ConstantModel([3, 1, -1]), ConstantModel([1, 3, -1]), ConstantModel([2, 2, 0])]
    images = torch.zeros((2, 1))
    individual = [model(images).softmax(dim=1) for model in models]
    historical = SoftVotingEnsemble(models)
    assert torch.equal(finalize.fuse_softmax(individual), historical.predict_probabilities(images))
    assert finalize.WEIGHTS == (1 / 3, 1 / 3, 1 / 3)


def test_predictions_align_by_sample_id_not_position():
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": ["x", "y"],
            "source_id": [train.SOURCE] * 2,
            "historical_split": ["train", "test"],
            "label": ["common_rust", "fall_armyworm"],
        }
    )
    inferred = pd.DataFrame({"sample_id": ["b", "a"], "pred_label": ["fall_armyworm", "x"]})
    aligned = finalize.aligned_predictions(manifest, inferred)
    assert aligned["sample_id"].tolist() == ["a", "b"]
    assert aligned["pred_label"].tolist() == ["x", "fall_armyworm"]
    with pytest.raises(ValueError, match="exactamente"):
        finalize.aligned_predictions(manifest, inferred.iloc[:1])
    with pytest.raises(ValueError, match="duplicado"):
        finalize.aligned_predictions(manifest, pd.concat([inferred.iloc[:1]] * 2))


def test_macro_f1_uses_three_present_classes():
    frame = pd.DataFrame(
        {
            "label": ["common_rust", "fall_armyworm", "lethal_necrosis"],
            "pred_label": ["common_rust", "fall_armyworm", "common_rust"],
            "confidence": [0.8, 0.7, 0.6],
        }
    )
    metrics, classes, matrix = finalize.metrics_for(frame)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert classes["class"].tolist() == ["common_rust", "fall_armyworm", "lethal_necrosis"]
    assert matrix.values.sum() == 3


def test_started_incomplete_evaluation_is_never_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_finalize, "ROOT", tmp_path)
    out = tmp_path / "seed_42" / "evaluation"
    out.mkdir(parents=True)
    (out / "final_evaluation.started.json").write_text(
        json.dumps({"seed": 42, "evaluation_count": 1}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="no repetir"):
        modal_finalize.final_status(42)


def test_all_component_locks_precede_any_holdout_read(tmp_path, monkeypatch):
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "manifest.lock.json").write_text("{}", encoding="utf-8")
    for name in ("train.csv", "val.csv"):
        (splits / name).write_text("source_id\nother-source\n", encoding="utf-8")
    monkeypatch.setattr(finalize, "SPLITS", splits)
    monkeypatch.setattr(finalize, "LOCK_SHA256", train.sha256_file(splits / "manifest.lock.json"))
    monkeypatch.setattr(finalize, "ROOT", tmp_path / "ensemble")
    manifest_hashes = {name: "h" * 64 for name in ("train.csv", "val.csv", "holdout.csv")}
    monkeypatch.setattr(finalize, "verify_splits", lambda *_: {"derived_sha256": manifest_hashes})
    runs = {}
    summaries = {}
    for seed in finalize.SEEDS:
        for model in finalize.MEMBERS:
            run = tmp_path / f"seed_{seed}" / model / "runs" / model / "run"
            run.mkdir(parents=True)
            (run / "summary.json").write_text("{}", encoding="utf-8")
            (run / "best.pth").write_bytes(b"best")
            summary = {
                "run_id": "run",
                "checkpoint_sha256": train.sha256_file(run / "best.pth"),
                "class_to_idx": {"a": 0},
                "preprocessing": {"resize": 224},
                "best_epoch": 2,
                "best_val_macro_f1": 0.8,
                "config_sha256": "c" * 64,
            }
            runs[seed, model] = run
            summaries[seed, model] = summary
            if model == "efficientnet_lite0":
                (run / "selection.lock.json").write_text("{}", encoding="utf-8")
            else:
                train.write_json(
                    run.parents[2] / "experiment_metadata.json",
                    {
                        "status": "validation_complete",
                        "model": model,
                        "seed": seed,
                        "run_id": "run",
                        "checkpoint_sha256": summary["checkpoint_sha256"],
                        "split_lock_sha256": finalize.LOCK_SHA256,
                        "finished_at_utc": "2026-10-01T00:00:00+00:00",
                        "git_commit": "commit",
                        "git_dirty": True,
                    },
                )
    monkeypatch.setattr(finalize, "member_run", lambda seed, model: runs[seed, model])
    monkeypatch.setattr(
        finalize, "validated_member", lambda seed, model, historical: summaries[seed, model]
    )
    written = finalize.lock_all({model: {} for model in finalize.MODELS})
    assert len(written) == 9
    assert not (splits / "holdout.csv").exists()
    for seed in finalize.SEEDS:
        group = finalize.ROOT / f"seed_{seed}" / "ensemble.selection.lock.json"
        assert group.exists()
        for model in finalize.MODELS:
            assert (runs[seed, model] / "selection.lock.json").exists()
    assert finalize.lock_all({model: {} for model in finalize.MODELS}) == []


def test_paired_deltas_match_seed_even_when_input_is_shuffled():
    rows = []
    for seed, b0, ensemble in ((42, 0.7, 0.8), (123, 0.5, 0.4), (2026, 0.6, 0.6)):
        for model, f1 in (
            ("efficientnet_b0", b0),
            ("efficientnet_lite0", 0.65),
            ("shufflenet_v2_x1_0", 0.55),
            ("ensemble", ensemble),
        ):
            rows.append({"seed": seed, "model": model, "macro_f1": f1, "accuracy": f1})
    shuffled = pd.DataFrame(rows).sample(frac=1, random_state=9)
    paired = report.paired_results(shuffled)
    assert paired["delta_vs_b0"].tolist() == pytest.approx([0.1, -0.1, 0])
    assert report.stats(paired["delta_vs_b0"].tolist())["sample_sd"] == pytest.approx(0.1)
    assert report.stats(paired["delta_vs_b0"].tolist())["ddof"] == 1
    with pytest.raises(ValueError, match="Faltan modelos"):
        report.paired_results(shuffled.iloc[:-1])


def test_disagreement_and_corrected_vs_introduced_errors_use_sample_id():
    ids = ["s1", "s2", "s3", "s4"]
    labels = ["common_rust", "fall_armyworm", "lethal_necrosis", "common_rust"]

    def frame(predictions):
        return pd.DataFrame(
            {"label": labels, "pred_label": predictions, "confidence": [0.8] * 4},
            index=ids,
        )

    frames = {
        "efficientnet_lite0": frame(
            ["common_rust", "common_rust", "fall_armyworm", "common_rust"]
        ).iloc[::-1],
        "efficientnet_b0": frame(
            ["fall_armyworm", "fall_armyworm", "fall_armyworm", "common_rust"]
        ),
        "shufflenet_v2_x1_0": frame(["common_rust", "common_rust", "common_rust", "common_rust"]),
        "ensemble": frame(["common_rust", "common_rust", "fall_armyworm", "common_rust"]),
    }
    errors, diversity = report.analyze_errors(frames, 42)
    assert errors["sample_id"].tolist() == ["s1", "s2", "s3"]
    assert diversity["corrected_count"] == 1
    assert diversity["corrected_fraction"] == 0.25
    assert diversity["introduced_count"] == 1
    assert diversity["introduced_fraction"] == 0.25
    assert diversity["all_members_wrong_count"] == 1
    assert diversity["member_disagreement_count"] == 3
    assert diversity["exclusive_errors"]["efficientnet_b0"] == 1
    assert diversity["corrected_by_class"]["common_rust"] == 1
    assert diversity["introduced_by_class"]["fall_armyworm"] == 1
    with pytest.raises(ValueError, match="no alineado"):
        report.analyze_errors({**frames, "efficientnet_b0": frames["efficientnet_b0"].iloc[:3]}, 42)


def test_final_inference_is_guarded_once_and_uses_same_holdout(tmp_path, monkeypatch):
    from types import SimpleNamespace

    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "manifest.lock.json").write_text("{}", encoding="utf-8")
    (splits / "holdout.csv").write_text("frozen-holdout", encoding="utf-8")
    monkeypatch.setattr(finalize, "SPLITS", splits)
    monkeypatch.setattr(finalize, "ROOT", tmp_path / "ensemble")
    monkeypatch.setattr(finalize, "LOCK_SHA256", train.sha256_file(splits / "manifest.lock.json"))
    monkeypatch.setattr(finalize, "HOLDOUT_SHA256", train.sha256_file(splits / "holdout.csv"))
    # Use the canonical class names/indices expected by the project's nine-class contract.
    mapping = {
        "common_rust": 0,
        "fall_armyworm": 1,
        "gray_leaf_spot": 2,
        "healthy": 3,
        "lethal_necrosis": 4,
        "nitrogen_deficiency": 5,
        "northern_corn_leaf_blight": 6,
        "phosphorus_deficiency": 7,
        "potassium_deficiency": 8,
    }
    group_root = finalize.ROOT / "seed_42"
    group_root.mkdir(parents=True)
    members = []
    for model in finalize.MEMBERS:
        run = tmp_path / model
        run.mkdir()
        (run / "best.pth").write_bytes(model.encode())
        (run / "summary.json").write_text("{}", encoding="utf-8")
        (run / "selection.lock.json").write_text("{}", encoding="utf-8")
        members.append(
            {
                "model": model,
                "run_dir": str(run),
                "checkpoint_sha256": train.sha256_file(run / "best.pth"),
                "summary_sha256": train.sha256_file(run / "summary.json"),
                "selection_lock_sha256": train.sha256_file(run / "selection.lock.json"),
            }
        )
    train.write_json(
        group_root / "ensemble.selection.lock.json",
        {
            "seed": 42,
            "held_out_source": finalize.SOURCE,
            "split_lock_sha256": finalize.LOCK_SHA256,
            "manifest_hashes": {name: "h" * 64 for name in ("train.csv", "val.csv", "holdout.csv")},
            "holdout_sha256": finalize.HOLDOUT_SHA256,
            "historical_ensemble_evidence_sha256": finalize.HISTORICAL_ENSEMBLE_SHA256,
            "locked_at_utc": "2026-10-01T00:00:00+00:00",
            "members": members,
            "weights": list(finalize.WEIGHTS),
            "fusion": "mean_of_softmax_probabilities",
        },
    )
    guard = group_root / "evaluation/final_evaluation.started.json"
    committed = []

    class FakeDataset:
        def __init__(self, csv_path, transform, class_to_idx):
            assert csv_path == str(splits / "holdout.csv")
            assert guard.exists() and committed == [True]
            assert class_to_idx == mapping
            self.data_frame = pd.DataFrame(
                {
                    "sample_id": [str(i) for i in range(5816)],
                    "image_path": [f"image_{i}.jpg" for i in range(5816)],
                    "source_id": [finalize.SOURCE] * 5816,
                    "historical_split": ["train"] * 5816,
                    "label": ["common_rust", "fall_armyworm", "lethal_necrosis"] * 1938
                    + ["common_rust", "fall_armyworm"],
                }
            )

        def __len__(self):
            return len(self.data_frame)

        def __getitem__(self, index):
            row = self.data_frame.iloc[index]
            return torch.zeros(1), mapping[row["label"]], row["sample_id"]

    def fake_load(*args, expected_model, **kwargs):
        class_index = {"efficientnet_lite0": 0, "efficientnet_b0": 1, "shufflenet_v2_x1_0": 4}[
            expected_model
        ]
        logits = [0.0] * 9
        logits[class_index] = 3.0
        return SimpleNamespace(
            model=ConstantModel(logits),
            summary={"class_to_idx": mapping, "preprocessing": {"resize": 224}},
            factory=SimpleNamespace(get_pipeline=lambda _purpose: None),
        )

    original_loader = torch.utils.data.DataLoader
    monkeypatch.setattr(finalize, "CornDataset", FakeDataset)
    monkeypatch.setattr(finalize, "load_validated_run", fake_load)
    monkeypatch.setattr(
        finalize,
        "DataLoader",
        lambda dataset, **kwargs: original_loader(dataset, batch_size=128, num_workers=0),
    )
    result = finalize.evaluate_seed(42, commit_guard=lambda: committed.append(guard.exists()))
    assert set(result) == set(finalize.MEMBERS) | {"ensemble"}
    assert all(metrics["samples"] == 5816 for metrics in result.values())
    assert all(metrics["evaluation_count"] == 1 for metrics in result.values())
    legacy = tmp_path / "legacy/seed_42/runs/efficientnet_lite0/run"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text("{}", encoding="utf-8")
    pd.read_csv(group_root / "evaluation/efficientnet_lite0_predictions.csv")[
        ["sample_id", "pred_label"]
    ].to_csv(legacy / "holdout_predictions.csv", index=False)
    monkeypatch.setattr(report, "OUTPUT", finalize.ROOT)
    monkeypatch.setattr(report, "LITE", tmp_path / "legacy")
    rows, classes, errors, diversity, registry = report.one_seed(42)
    assert len(rows) == 4 and len(classes) == 12
    assert not errors.empty
    assert registry["evaluation_count"] == 1
    assert diversity["member_disagreement_count"] > 0
    monkeypatch.setattr(modal_finalize, "ROOT", finalize.ROOT)
    assert modal_finalize.final_status(42) is True
    with pytest.raises(FileExistsError):
        finalize.evaluate_seed(42)
