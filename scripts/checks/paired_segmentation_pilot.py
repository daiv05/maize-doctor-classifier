"""Small, paired development-only pilot; never loads test images or chooses on test.

Two distinct questions: input sensitivity of existing classifiers, and training paired
linear heads on frozen ImageNet features. Neither is a full segmented-model benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from scripts.pipeline.segment_dataset import atomic_image, source_manifest
from src.analysis.fairness import compute_disparity_metrics, compute_subgroup_metrics
from src.config import PROJECT_ROOT, set_global_seed
from src.data.loader import load_and_normalize_image
from src.data.provenance import model_state_hash
from src.data.transforms import CornTransformFactory
from src.models import build_model
from src.models.feature_exposed import FeatureExposedModel
from src.provenance import atomic_json, sha256_file
from src.segmentation.detector import MaizeLeafSegmenter, segmentation_runtime_contract
from src.segmentation.leaf_processor import (
    LeafMaskProcessorConfig,
    SegmentedLeafProcessor,
    build_comparison_panel,
)
from src.training.loop import fit
from src.training.runs import load_run, validate_ensemble_runs


def select_development(frame, train_per_stratum, val_per_stratum, seed):
    parts = []
    for split, count in (("train", train_per_stratum), ("val", val_per_stratum)):
        for _, group in frame.loc[frame.split.eq(split)].groupby(["label", "environment"]):
            parts.append(
                group.sort_values("sample_id").sample(min(count, len(group)), random_state=seed)
            )
    selected = pd.concat(parts).sort_values(["split", "sample_id"]).reset_index(drop=True)
    if not selected.sample_id.is_unique or selected.split.eq("test").any():
        raise ValueError("Invalid pilot population")
    return selected


def metrics(frame, probabilities, names):
    mapping = {name: idx for idx, name in enumerate(names)}
    true = frame.label.map(mapping).to_numpy()
    pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=range(len(names)), zero_division=0
    )
    groups = compute_subgroup_metrics(
        true.tolist(), pred.tolist(), frame.environment.tolist(), names
    )
    return {
        "n": len(frame),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion_matrix(true, pred, labels=range(len(names))).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(names)
        },
        "subgroups": groups,
        "disparity": compute_disparity_metrics(groups),
    }


def save_predictions(path, frame, probabilities, names):
    frame = frame.copy()
    frame["pred_label"] = [names[i] for i in probabilities.argmax(axis=1)]
    for i, name in enumerate(names):
        frame[f"prob_{name}"] = probabilities[:, i]
    frame.to_csv(path, index=False)


def feature_batches(model, frame, root, factory, variant, output, batch_size):
    arrays = []
    transform = factory.get_pipeline("val")
    for start in range(0, len(frame), batch_size):
        tensors = []
        for row in frame.iloc[start : start + batch_size].itertuples():
            path = root / row.image_path if variant == "original" else output / row.derived_path
            expected = row.source_sha256 if variant == "original" else row.derived_sha256
            if sha256_file(path) != expected:
                raise ValueError(f"Pilot image changed: {row.sample_id}")
            tensors.append(transform(load_and_normalize_image(path)))
        with torch.inference_mode():
            result = model(torch.stack(tensors))
        arrays.append(result.cpu().numpy())
    return np.concatenate(arrays)


def paired_bootstrap(frame, original, segmented, names, iterations=500, seed=42):
    mapping = {name: i for i, name in enumerate(names)}
    true = frame.label.map(mapping).to_numpy()
    first, second = original.argmax(1), segmented.argmax(1)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(true == i) for i in range(len(names))]
    deltas = []
    for _ in range(iterations):
        indices = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in strata if len(group)]
        )
        scores = [
            precision_recall_fscore_support(
                true[indices],
                pred[indices],
                labels=range(len(names)),
                average="macro",
                zero_division=0,
            )[2]
            for pred in (first, second)
        ]
        deltas.append(scores[1] - scores[0])
    point_scores = [
        precision_recall_fscore_support(
            true,
            pred,
            labels=range(len(names)),
            average="macro",
            zero_division=0,
        )[2]
        for pred in (first, second)
    ]
    return {
        "delta_macro_f1_segmented_minus_original": float(point_scores[1] - point_scores[0]),
        "bootstrap_mean_delta": float(np.mean(deltas)),
        "percentile_95_ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "bootstrap_iterations": iterations,
        "limitation": "paired class-stratified image bootstrap; no plant/session metadata",
    }


def run_pilot(args):
    if args.output_dir.exists():
        raise FileExistsError("Pilot output must be a new experiment directory")
    if min(args.train_per_stratum, args.val_per_stratum, args.epochs, args.batch_size) < 1:
        raise ValueError("Pilot counts and budget must be positive")
    torch.set_num_threads(args.threads)
    os.environ["DATASET_ROOT"] = str(args.dataset_root.resolve())
    runs = [
        load_run(p, json.loads((p.parent / "summary.json").read_text())["model"])
        for p in args.checkpoints
    ]
    validate_ensemble_runs(runs)
    names = sorted(runs[0].class_to_idx, key=runs[0].class_to_idx.get)
    frame = source_manifest(args.splits_dir, args.dataset_root)
    selected = select_development(frame, args.train_per_stratum, args.val_per_stratum, args.seed)
    args.output_dir.mkdir(parents=True)
    config = LeafMaskProcessorConfig()
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_pilot",
        "utc": pd.Timestamp.utcnow().isoformat(),
        "selection_seed": args.seed,
        "training_seeds": args.training_seeds,
        "head_epochs": args.epochs,
        "train_per_stratum": args.train_per_stratum,
        "val_per_stratum": args.val_per_stratum,
        "split_counts": selected.split.value_counts().to_dict(),
        "members": [r.manifest_entry() for r in runs],
        "source_splits_sha256": {
            s: sha256_file(args.splits_dir / f"{s}.csv") for s in ("train", "val", "test")
        },
        "segmenter_sha256": sha256_file(args.segmenter_checkpoint),
        "detector": segmentation_runtime_contract(),
        "processor": asdict(config),
        "processor_sha256": sha256_file(PROJECT_ROOT / "src/segmentation/leaf_processor.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "fallback": "original for rejected and uncertain; all selected images retained",
        "test_image_inference": False,
        "background_augmentation": "not_enabled",
        "scope": "stratified small development subset; not a population estimate",
        "human_review": "pending",
        "threshold_calibration": "not_performed",
    }
    selected.to_csv(args.output_dir / "selected.csv", index=False)
    atomic_json(args.output_dir / "protocol.json", protocol)
    processor = SegmentedLeafProcessor(config)
    detector = MaizeLeafSegmenter(args.segmenter_checkpoint, device="cpu")
    records, preview_counts = [], {}
    for index, row in enumerate(selected.to_dict("records")):
        image = load_and_normalize_image(args.dataset_root / row["image_path"])
        result = processor.process(image, detector.segment(image))
        fallback = result.status != "accepted"
        relative = Path("derived") / f"{row['sample_id']}.png"
        atomic_image(image if fallback else result.processed_image, args.output_dir / relative)
        record = {
            **row,
            "derived_path": relative.as_posix(),
            "derived_sha256": sha256_file(args.output_dir / relative),
            "segmentation_status": result.status,
            "fallback_original": fallback,
            "quality": result.quality,
            "warnings": result.warnings,
        }
        records.append(record)
        key = (row["split"], row["label"], row["environment"], result.status)
        preview_counts[key] = preview_counts.get(key, 0) + 1
        if preview_counts[key] <= 2:
            preview = args.output_dir / "previews" / relative.name
            atomic_image(build_comparison_panel(result), preview)
            record["preview_path"] = str(preview)
        if index % 20 == 0:
            print(f"Segmented {index + 1}/{len(selected)}", flush=True)
    atomic_json(args.output_dir / "segmentation_manifest.json", records)
    frame = pd.DataFrame(records)
    review = frame.loc[frame.preview_path.notna()].copy()
    for column in ("human_accept", "lesion_preserved", "review_notes"):
        review[column] = ""
    review.to_csv(args.output_dir / "human_review.csv", index=False)
    validation = frame.loc[frame.split.eq("val")].reset_index(drop=True)
    training = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    results = {
        "existing_checkpoint_sensitivity": [],
        "frozen_imagenet_head_pilot": [],
        "segmentation_status_counts": frame.segmentation_status.value_counts().to_dict(),
        "segmentation_by_class_environment": frame.groupby(
            ["label", "environment", "segmentation_status"]
        )
        .size()
        .rename("n")
        .reset_index()
        .to_dict("records"),
        "fallback_count": int(frame.fallback_original.sum()),
        "population": "all selected validation images, including fallbacks",
    }
    for run in runs:
        model_name = run.summary["model"]
        print(f"Existing checkpoint input sensitivity: {model_name}", flush=True)
        pair = {}
        for variant in ("original", "segmented"):
            logits = feature_batches(
                run.model,
                validation,
                args.dataset_root,
                run.factory,
                variant,
                args.output_dir,
                args.batch_size,
            )
            probabilities = torch.from_numpy(logits).softmax(1).numpy()
            pair[variant] = probabilities
            save_predictions(
                args.output_dir / f"existing_{model_name}_{variant}.csv",
                validation,
                probabilities,
                names,
            )
        results["existing_checkpoint_sensitivity"].append(
            {
                "model": model_name,
                **{
                    variant: metrics(validation, probabilities, names)
                    for variant, probabilities in pair.items()
                },
                "paired_uncertainty": paired_bootstrap(
                    validation, pair["original"], pair["segmented"], names
                ),
                "interpretation": "input shift on original-trained weights, not segmented training",
            }
        )
        del run.model
        print(f"Frozen ImageNet paired head pilot: {model_name}", flush=True)
        set_global_seed(args.seed)
        model = build_model(model_name, num_classes=len(names), pretrained=True).eval()
        backbone_hash = model_state_hash(model)
        exposed = FeatureExposedModel(model, model_name).eval()

        class Features(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x):
                return self.wrapped(x)[1]

        factory = CornTransformFactory(target_size=tuple(run.summary["image_size"]))
        train_mask = frame.split.eq("train").to_numpy()
        labels = torch.tensor(training.label.map(run.class_to_idx).tolist())
        val_labels = torch.tensor(validation.label.map(run.class_to_idx).tolist())
        counts = torch.bincount(labels, minlength=len(names)).float()
        weights = counts.max().div(counts.clamp(min=1)).sqrt()
        variants = {
            variant: feature_batches(
                Features(exposed),
                frame,
                args.dataset_root,
                factory,
                variant,
                args.output_dir,
                args.batch_size,
            )
            for variant in ("original", "segmented")
        }
        for seed in args.training_seeds:
            pair, entries = {}, {}
            for variant, features in variants.items():
                set_global_seed(seed)
                train_x = torch.tensor(features[train_mask])
                val_x = torch.tensor(features[~train_mask])
                center, scale = train_x.mean(0), train_x.std(0).clamp(min=1e-5)
                train_x, val_x = (train_x - center) / scale, (val_x - center) / scale
                head = torch.nn.Linear(train_x.shape[1], len(names))
                train_loader = DataLoader(
                    TensorDataset(train_x, labels), batch_size=args.batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(val_x, val_labels), batch_size=args.batch_size
                )
                folder = args.output_dir / f"heads/{model_name}/{seed}/{variant}"
                history = fit(
                    head,
                    train_loader,
                    val_loader,
                    torch.nn.CrossEntropyLoss(weight=weights),
                    torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001),
                    torch.device("cpu"),
                    args.epochs,
                    f"pilot_{model_name}",
                    run_dir=folder,
                )
                with torch.inference_mode():
                    probabilities = head(val_x).softmax(1).numpy()
                pair[variant] = probabilities
                entries[variant] = metrics(validation, probabilities, names)
                save_predictions(folder / "val_predictions.csv", validation, probabilities, names)
                np.savez(folder / "standardization.npz", mean=center.numpy(), scale=scale.numpy())
                pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
                atomic_json(
                    folder / "contract.json",
                    {
                        "backbone_sha256": backbone_hash,
                        "preprocessing": factory.to_contract(),
                        "head_sha256": sha256_file(folder / "best.pth"),
                        "seed": seed,
                        "class_to_idx": run.class_to_idx,
                        "criterion": "sqrt_inverse_weighted_ce",
                        "sampler": "shuffle_no_weighted_sampler",
                        "protocol": "../../../../protocol.json",
                    },
                )
            results["frozen_imagenet_head_pilot"].append(
                {
                    "model": model_name,
                    "seed": seed,
                    **entries,
                    "paired_uncertainty": paired_bootstrap(
                        validation, pair["original"], pair["segmented"], names
                    ),
                }
            )
        del model, exposed, variants
        atomic_json(args.output_dir / "results.json", results)
    results["limitations"] = [
        "small stratified development subset",
        "thresholds uncalibrated",
        "human label-transfer review pending",
        "no plant/session grouping",
        "best head epoch selected on the same validation subset",
        "not end-to-end retraining",
        "no holdout or mobile validation",
    ]
    atomic_json(args.output_dir / "results.json", results)
    atomic_json(
        args.output_dir / "receipt.json",
        {
            "status": "complete",
            "artifact_sha256": {
                p.relative_to(args.output_dir).as_posix(): sha256_file(p)
                for p in args.output_dir.rglob("*")
                if p.is_file() and "derived" not in p.parts
            },
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "fallback_count": results["fallback_count"],
                "status": "complete_development_pilot",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--segmenter-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--train-per-stratum", type=int, default=12)
    parser.add_argument("--val-per-stratum", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    run_pilot(parser.parse_args())


if __name__ == "__main__":
    main()
