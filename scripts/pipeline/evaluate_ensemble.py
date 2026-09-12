"""Compare trained ensemble members, aligned by identity and selected on development."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, get_output_root
from src.data.dataset import CornDataset
from src.data.identity import align_predictions, identified_batches
from src.export.data import resolve_split_csv
from src.models.ensemble import SoftVotingEnsemble
from src.provenance import atomic_json, sha256_file
from src.training.common import generate_run_id, select_device
from src.training.loop import _metrics_from_predictions
from src.training.runs import load_run, resolve_checkpoint, validate_ensemble_runs


def _find_checkpoint(model_name, output_root):
    return resolve_checkpoint(model_name, output_root)


def evaluate_predictions(y_true, y_pred, labels_names):
    return _metrics_from_predictions(y_true, y_pred, 0.0, list(range(len(labels_names))))


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
    title: str = "Matriz de Confusión — Soft Voting Ensemble",
) -> None:
    """Genera y guarda el mapa de calor de la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Etiqueta Real",
        xlabel="Predicción del Ensamble",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Anotar valores numéricos
    fmt = ".2f"
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:{fmt}}\n({cm[i, j]})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--run")
    parser.add_argument("--pipeline", choices=["main", "baselines"], default="main")
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--splits-dir")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/dataset.yaml"))
    args = parser.parse_args()
    if len(args.models) != len(set(args.models)):
        parser.error("Duplicate model names; distinguish runs explicitly in an ensemble manifest")
    if args.checkpoints and len(args.checkpoints) != len(args.models):
        parser.error("--checkpoints must match --models")
    device = select_device()
    paths = [
        resolve_checkpoint(
            name,
            get_output_root(),
            args.checkpoints[i] if args.checkpoints else None,
            args.run,
            args.pipeline,
        )
        for i, name in enumerate(args.models)
    ]
    runs = [
        load_run(path, name, device, config_path=args.config)
        for path, name in zip(paths, args.models)
    ]
    validate_ensemble_runs(runs)
    ensemble = SoftVotingEnsemble([r.model for r in runs], args.weights, args.models)
    weights = ensemble.weights.cpu().tolist()
    output = (
        Path(args.output_dir)
        if args.output_dir
        else get_output_root() / "ensemble" / generate_run_id()
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference = None
    probabilities = []
    metrics = {}
    class_names = sorted(runs[0].class_to_idx, key=runs[0].class_to_idx.get)
    split_hashes = []
    for run in runs:
        split_csv = resolve_split_csv(run.checkpoint.parent, args.splits_dir, args.split)
        split_hashes.append(sha256_file(split_csv))
        dataset = CornDataset(
            str(split_csv),
            args.config,
            transform=run.factory.get_pipeline("test"),
            class_to_idx=run.class_to_idx,
        )
        ids, labels, values = [], [], []
        with torch.no_grad():
            for images, targets, sample_ids in identified_batches(
                DataLoader(dataset, batch_size=args.batch_size)
            ):
                ids.extend(sample_ids)
                labels.extend(targets.tolist())
                values.extend(run.model(images.to(device)).softmax(-1).cpu().tolist())
        frame = align_predictions(dataset.data_frame, ids)
        if reference is None:
            reference = frame
        elif set(frame.sample_id) != set(reference.sample_id):
            raise ValueError("Ensemble members do not evaluate the same samples")
        order = frame.set_index("sample_id").loc[reference.sample_id]
        if (
            order.label.tolist() != reference.label.tolist()
            or order.image_path.tolist() != reference.image_path.tolist()
        ):
            raise ValueError("Ensemble members disagree on sample metadata")
        indexed = pd.DataFrame(values, index=frame.sample_id).loc[reference.sample_id].to_numpy()
        probabilities.append(indexed)
        y_true = [run.class_to_idx[label] for label in reference.label]
        metrics[run.summary["model"]] = evaluate_predictions(
            y_true, indexed.argmax(1).tolist(), class_names
        )
    combined = np.einsum("m,mnc->nc", weights, np.array(probabilities))
    predictions = combined.argmax(1)
    metrics["ensemble"] = evaluate_predictions(y_true, predictions.tolist(), class_names)
    reference["pred_label"] = [class_names[i] for i in predictions]
    reference["pred_prob"] = combined.max(1)
    reference.to_csv(output / "predictions.csv", index=False)
    manifest = {
        "schema_version": 1,
        "members": [r.manifest_entry(w) for r, w in zip(runs, weights)],
    }
    atomic_json(output / "ensemble_manifest.json", manifest)
    atomic_json(
        output / "ensemble_summary.json",
        {
            "schema_version": 2,
            "split": args.split,
            "split_hashes": split_hashes,
            "metrics_per_model": metrics,
            "samples": len(reference),
            "manifest": manifest,
        },
    )
    pd.DataFrame(metrics).T.to_csv(output / "ensemble_comparison.csv")
    plot_confusion_matrix(
        y_true, predictions.tolist(), class_names, output / "confusion_matrix_ensemble.png"
    )
    print(f"Ensemble evaluation: {output}")


if __name__ == "__main__":
    main()
