import pandas as pd


def test_tmp_splits_dir_tiene_las_columnas_esperadas(tmp_splits_dir):
    data_frame = pd.read_csv(tmp_splits_dir / "train.csv")
    assert list(data_frame.columns) == ["image_path", "label", "environment"]
    assert len(data_frame) == 20
    assert set(data_frame["environment"]) == {"lab", "real"}


def test_distribucion_desbalanceada(tmp_splits_dir):
    counts = pd.read_csv(tmp_splits_dir / "train.csv")["label"].value_counts()
    assert counts["healthy"] == 12
    assert counts["potassium_deficiency"] == 2
