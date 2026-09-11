import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.cross_validation import (
    HierarchicalKFoldSplitter,
    compute_aggregate_statistics,
    plot_kfold_boxplot,
)
from scripts.pipeline.cross_validate import (
    _load_best_params_if_available,
    plot_test_confusion_matrix,
)


def test_hierarchical_kfold_splitter_disjoint_and_stratified():
    # Dataset sintético con 2 clases y 2 entornos
    data = {
        "image_path": [f"img_{i}.jpg" for i in range(100)],
        "label": ["healthy"] * 60 + ["common_rust"] * 40,
        "environment": (["lab"] * 30 + ["real"] * 30) + (["lab"] * 20 + ["real"] * 20),
    }
    df = pd.DataFrame(data)

    splitter = HierarchicalKFoldSplitter(n_splits=5, seed=42)
    splits = splitter.split(df)

    assert len(splits) == 5

    # Verificar que las particiones de validación son disjuntas y cubren todo el dataset
    val_indices_seen = []
    for split in splits:
        assert len(split.train_df) == 80
        assert len(split.val_df) == 20
        # Verificar estratificación en val
        counts = split.val_df["label"].value_counts()
        assert counts["healthy"] == 12  # 60 / 5
        assert counts["common_rust"] == 8   # 40 / 5
        val_indices_seen.extend(split.val_df["image_path"].tolist())

    assert len(val_indices_seen) == 100
    assert len(set(val_indices_seen)) == 100


def test_compute_aggregate_statistics():
    metrics = [
        {"macro_f1": 0.9100, "accuracy": 0.9300},
        {"macro_f1": 0.9150, "accuracy": 0.9350},
        {"macro_f1": 0.9200, "accuracy": 0.9400},
        {"macro_f1": 0.9120, "accuracy": 0.9320},
        {"macro_f1": 0.9180, "accuracy": 0.9380},
    ]

    stats = compute_aggregate_statistics(metrics)

    assert "macro_f1" in stats
    assert "accuracy" in stats
    assert stats["macro_f1"]["mean"] == pytest.approx(0.9150, abs=1e-4)
    assert stats["macro_f1"]["std"] > 0
    assert stats["macro_f1"]["ci_95_lower"] < stats["macro_f1"]["mean"] < stats["macro_f1"]["ci_95_upper"]
    assert stats["macro_f1"]["min"] == 0.9100
    assert stats["macro_f1"]["max"] == 0.9200


def test_plot_kfold_boxplot(tmp_path: Path):
    metrics = [
        {"macro_f1": 0.9100, "accuracy": 0.9300, "macro_precision": 0.9000, "macro_recall": 0.9200},
        {"macro_f1": 0.9150, "accuracy": 0.9350, "macro_precision": 0.9050, "macro_recall": 0.9250},
    ]
    out_file = tmp_path / "test_boxplot.png"
    plot_kfold_boxplot(metrics, out_file, model_name="TestModel")
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_test_confusion_matrix(tmp_path: Path):
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 1, 0]
    class_names = ["healthy", "common_rust"]
    out_file = tmp_path / "confusion_matrix_test.png"
    plot_test_confusion_matrix(y_true, y_pred, class_names, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_load_best_params_if_available(tmp_path: Path):
    optuna_dir = tmp_path / "tuning" / "efficientnet_b0"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    best_json = optuna_dir / "best_params.json"
    best_json.write_text(json.dumps({"best_params": {"learning_rate": 0.0003, "batch_size": 16}}))

    params = _load_best_params_if_available("efficientnet_b0", str(best_json), tmp_path)
    assert params is not None
    assert params["learning_rate"] == 0.0003
    assert params["batch_size"] == 16


def test_corn_dataset_accepts_dataframe(tmp_path: Path):
    from src.data.dataset import CornDataset
    import yaml

    cfg_file = tmp_path / "dataset.yaml"
    cfg_file.write_text(yaml.dump({"dataset": {"classes": ["healthy", "common_rust"]}}))

    df = pd.DataFrame({
        "image_path": ["dummy1.jpg", "dummy2.jpg"],
        "label": ["healthy", "common_rust"],
    })

    ds = CornDataset(csv_path=df, config_path=str(cfg_file))
    assert len(ds) == 2
    assert "healthy" in ds.class_to_idx
    assert "common_rust" in ds.class_to_idx

