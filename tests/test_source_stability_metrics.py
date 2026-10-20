"""LOSO Macro-F1 must use classes present in the held-out source's truth."""

from __future__ import annotations

import pandas as pd

from scripts.experiments.source_stability_report import _metrics


def test_loso_macro_f1_ignores_absent_truth_class_as_separate_term() -> None:
    frame = pd.DataFrame(
        {
            "label": ["healthy", "healthy", "rust"],
            "pred_label": ["healthy", "other", "rust"],
            "pred_prob": [0.9, 0.6, 0.8],
        }
    )
    result = _metrics(frame)
    # healthy F1 = 2/3; rust F1 = 1; mean over ground-truth classes = 5/6.
    assert abs(result["macro_f1"] - 5 / 6) < 1e-12
