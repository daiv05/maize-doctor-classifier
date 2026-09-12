"""Recalculate only export subgroup metrics from hash-verified saved predictions."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.export.evaluate import _environment_breakdown
from src.provenance import atomic_json, sha256_file


def recalculate(report: Path, output: Path):
    if output.exists():
        raise FileExistsError("Choose a new recalculation report")
    saved = json.loads(report.read_text())
    predictions = report.with_name(report.stem + "_predictions.csv")
    if sha256_file(predictions) != saved["predictions_sha256"]:
        raise ValueError("Saved predictions differ from evaluated report")
    training = json.loads((report.parent.parent / "summary.json").read_text())
    names = {int(i): name for name, i in training["class_to_idx"].items()}
    frame = pd.read_csv(predictions)
    if len(frame) != saved["n_samples"] or frame.sample_id.duplicated().any():
        raise ValueError("Predictions have missing or duplicate identities")
    corrected = {**saved, "by_environment": _environment_breakdown(frame, names)}
    corrected["recalculation"] = {
        "source_report": str(report),
        "source_report_sha256": sha256_file(report),
        "script_sha256": sha256_file(Path(__file__)),
        "changed_fields": ["by_environment"],
        "purpose": "Supported-class subgroup policy; no new inference or model selection",
    }
    atomic_json(output, corrected)
    print(json.dumps(corrected["by_environment"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recalculate(args.report, args.output)
