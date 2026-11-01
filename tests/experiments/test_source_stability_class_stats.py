"""Fixed-seed class tables retain per-seed values and sample SD."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.experiments.source_stability_report import summarize_classes


def test_per_class_three_seed_summary() -> None:
    frame = pd.DataFrame(
        {
            "scenario": ["multicrop-disease-maiz"] * 3,
            "stage": ["holdout"] * 3,
            "class": ["lethal_necrosis"] * 3,
            "seed": [42, 123, 2026],
            "f1": [0.8, 0.9, 1.0],
            "support": [120, 120, 120],
        }
    )
    row = summarize_classes(frame).iloc[0]
    assert row["seed_42"] == 0.8
    assert row["seed_123"] == 0.9
    assert row["seed_2026"] == 1.0
    assert math.isclose(row["mean"], 0.9)
    assert math.isclose(row["sample_sd"], 0.1)
    assert row["support"] == 120
    with pytest.raises(ValueError, match="Faltan seeds"):
        summarize_classes(frame.iloc[:2])
    changed = frame.copy()
    changed.loc[changed.seed == 2026, "support"] = 119
    with pytest.raises(ValueError, match="Soporte distinto"):
        summarize_classes(changed)
