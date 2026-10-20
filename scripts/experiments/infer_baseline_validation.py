"""Regenera solo las predicciones de validation de un run contractual ya entrenado.

No entrena, no modifica el run original y nunca accede al split de test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CornDataset
from src.data.identity import unpack_batch
from src.training.artifacts import write_predictions_csv
from src.training.runs import load_validated_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_validated_run(
        args.checkpoint,
        expected_model="efficientnet_lite0",
        splits_dir=args.splits_dir,
        device=device,
    )
    if loaded.summary["run_id"] != "20260921_204608":
        raise ValueError("Este script solo admite el baseline contractual 20260921_204608")
    mapping = loaded.summary["class_to_idx"]
    dataset = CornDataset(
        str(args.splits_dir / "val.csv"),
        transform=loaded.factory.get_pipeline("val"),
        class_to_idx=mapping,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            images, targets, ids = unpack_batch(batch)
            if ids is None:
                raise ValueError("El DataLoader no propagó sample_id")
            scores = loaded.model(images.to(device)).softmax(dim=1)
            confidence, predicted = scores.max(dim=1)
            labels.extend(targets.tolist())
            predictions.extend(predicted.cpu().tolist())
            probabilities.extend(confidence.cpu().tolist())
            sample_ids.extend(ids)

    # El escritor contractual alinea por sample_id; nunca por posición de CSV.
    from src.data.identity import IdentifiedValues

    indexed_labels = IdentifiedValues(labels, sample_ids=sample_ids)
    indexed_predictions = IdentifiedValues(predictions, sample_ids=sample_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = write_predictions_csv(
        args.output_dir,
        dataset,
        {index: name for name, index in mapping.items()},
        indexed_labels,
        indexed_predictions,
        probabilities,
        filename="baseline_validation_predictions.csv",
    )
    from sklearn.metrics import f1_score

    macro_f1 = float(f1_score(frame["label"], frame["pred_label"], average="macro"))
    result = {
        "run_id": loaded.summary["run_id"],
        "checkpoint_sha256": loaded.summary["checkpoint_sha256"],
        "split_manifest_sha256": loaded.summary["split_manifest_sha256"],
        "device": str(device),
        "samples": len(frame),
        "macro_f1_recomputed": macro_f1,
        "macro_f1_recorded": loaded.summary["best_val_macro_f1"],
    }
    (args.output_dir / "baseline_validation_inference.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
