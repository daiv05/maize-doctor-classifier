import sys

import pandas as pd
import pytest

from scripts.experiments.comparar_kfold import main


def test_compara_dos_protocolos_desde_predicciones(tmp_path, monkeypatch):
    stratified = tmp_path / "estratificada"
    grouped = tmp_path / "agrupada"
    for root, predictions in [
        (stratified, [0, 1, 0, 1]),
        (grouped, [0, 0, 0, 0]),
    ]:
        for fold in (1, 2):
            fold_dir = root / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            pd.DataFrame(
                {"y_true": [0, 1, 0, 1], "y_pred": predictions}
            ).to_csv(fold_dir / "predictions.csv", index=False)

    output = tmp_path / "comparacion.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comparar_kfold.py",
            "--estratificada-dir", str(stratified),
            "--agrupada-dir", str(grouped),
            "--salida", str(output),
        ],
    )
    main()

    result = pd.read_csv(output).set_index("protocolo")
    assert set(result.index) == {"estratificada", "agrupada"}
    assert result.loc["estratificada", "macro_f1_evaluable"] == pytest.approx(1.0)
    assert result.loc["agrupada", "macro_f1_evaluable"] < 1.0
