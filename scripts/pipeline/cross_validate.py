"""CV del pipeline principal sobre train+val; HPO explícito y test opt-in.

No es el CV independiente de Etapa 2 ni una estimación anidada tras seleccionar HPO.
Cada fold guarda su mejor checkpoint, historia, configuración y predicciones.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, get_output_root, set_global_seed
from src.data.cross_validation import (
    HierarchicalKFoldSplitter,
    compute_aggregate_statistics,
    plot_kfold_boxplot,
)
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.models import build_model, list_models, resolve_input_size
from src.training.common import select_device, worker_init_fn
from src.training.loop import fit, run_epoch
from src.training.losses import build_criterion
from src.training.optim import EarlyStopping, build_scheduler


def _calc_cm_np(y_t, y_p, num_classes=4, normalize=False):
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1.0
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("kfold")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CV del pipeline principal con evaluación final de test opt-in."
    )
    parser.add_argument(
        "--model",
        default="efficientnet_b0",
        help=f"Modelo a evaluar en K-Fold. Disponibles: {list_models()}",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        dest="k_folds",
        help="Número de particiones (folds) a evaluar (default: 5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Techo máximo de épocas por fold (default: 20).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciencia de early stopping por fold (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Tamaño de lote (sobrescrito si se provee best_params.json).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        dest="learning_rate",
        help="Tasa de aprendizaje base (sobrescrito si se provee best_params.json).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        dest="weight_decay",
        help="Decaimiento de pesos AdamW (sobrescrito si se provee best_params.json).",
    )
    parser.add_argument(
        "--class-weights",
        choices=["sqrt_inverse", "inverse", "none"],
        default="sqrt_inverse",
        dest="class_weights",
        help="Ponderación de función de pérdida.",
    )
    parser.add_argument(
        "--best-params",
        default=None,
        dest="best_params_path",
        help="JSON PyTorch explícito; precedencia CLI > archivo > defaults.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train/val/test.csv (default: <outputs>/splits/seed_42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Destino de reportes y métricas de K-Fold (default: <outputs>/kfold)",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
        help="Ruta al archivo dataset.yaml.",
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    from src.training.hyperparameters import parse_with_best_params

    return parse_with_best_params(parser)


def _load_best_params_if_available(model_name, explicit_path, output_root):
    from src.training.hyperparameters import load_best_params

    return load_best_params(explicit_path) if explicit_path else None


def plot_test_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
    title: str = "Matriz de Confusión — Test Set Final (5-Fold Averaging)",
) -> None:
    """Genera y guarda el mapa de calor de la matriz de confusión sobre el test set."""
    num_classes = len(class_names)
    cm = _calc_cm_np(y_true, y_pred, num_classes=num_classes, normalize=False)
    cm_norm = _calc_cm_np(y_true, y_pred, num_classes=num_classes, normalize=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Greens")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Etiqueta Real",
        xlabel="Predicción del Modelo",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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
    import gc

    from src.data.identity import ensure_sample_ids, identified_batches
    from src.provenance import atomic_json
    from src.training.artifacts import write_predictions_csv, write_summary
    from src.training.common import generate_run_id
    from src.training.loop import _metrics_from_predictions

    args = _parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["dataset"]["seed"]
    set_global_seed(seed)
    root = get_output_root()
    split_dir = (
        Path(args.splits_dir) if args.splits_dir else root / cfg["paths"]["split_output_dir"]
    )
    output = (
        Path(args.output_dir)
        if args.output_dir
        else root / "kfold" / args.model / generate_run_id()
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty CV output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dev = ensure_sample_ids(
        pd.concat(
            [pd.read_csv(split_dir / "train.csv"), pd.read_csv(split_dir / "val.csv")],
            ignore_index=True,
        )
    )
    mapping = {name: i for i, name in enumerate(cfg["dataset"]["classes"])}
    class_names = list(mapping)
    params = _load_best_params_if_available(args.model, args.best_params_path, root) or {}
    lr, wd = args.learning_rate, args.weight_decay
    bs = args.batch_size
    cw, smoothing = args.class_weights, args.label_smoothing
    size = resolve_input_size(args.model, tuple(cfg["dataset"]["target_size"]))
    factory = CornTransformFactory(args.config, size, clahe=args.clahe)
    from src.data.segmented import bind_segmented_splits

    bind_segmented_splits(factory, split_dir)
    device = select_device()
    rows, checkpoints = [], []
    for fold in HierarchicalKFoldSplitter(args.k_folds, seed).split(dev):
        set_global_seed(seed + fold.fold_index)
        fold_dir = output / f"fold_{fold.fold_index}"
        fold_dir.mkdir()
        fold.train_df.to_csv(fold_dir / "train.csv", index=False)
        fold.val_df.to_csv(fold_dir / "val.csv", index=False)
        train_ds = CornDataset(
            fold.train_df,
            args.config,
            transform=factory.get_pipeline("train"),
            minority_transform=factory.get_pipeline("minority"),
            class_to_idx=mapping,
        )
        val_ds = CornDataset(
            fold.val_df, args.config, transform=factory.get_pipeline("val"), class_to_idx=mapping
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(val_ds, batch_size=bs, num_workers=args.num_workers)
        model = build_model(
            args.model, num_classes=len(mapping), pretrained=not args.no_pretrained
        ).to(device)
        criterion = build_criterion(
            train_ds.data_frame.label.tolist(),
            mapping,
            strategy=cw,
            label_smoothing=smoothing,
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = build_scheduler(
            optimizer,
            kind="cosine",
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr=1e-6,
        )
        history = fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            args.epochs,
            args.model,
            run_dir=fold_dir,
            scheduler=scheduler,
            early_stopping=EarlyStopping(patience=args.patience),
            clip_grad_norm=1.0,
        )
        metrics, labels, predictions, probs = run_epoch(model, val_loader, criterion, device)
        write_predictions_csv(fold_dir, val_ds, dict(enumerate(class_names)), predictions, probs)
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        best = max(history, key=lambda row: row["val_macro_f1"])
        write_summary(
            fold_dir,
            {
                "model": args.model,
                "class_to_idx": mapping,
                "image_size": list(size),
                "preprocessing": factory.to_contract(),
                "config_path": args.config,
                "splits_dir": str(fold_dir),
                "best_epoch": best["epoch"],
                "best_val_macro_f1": best["val_macro_f1"],
                "effective_args": vars(args),
                "hpo_parameters": params,
                "validation": metrics,
            },
        )
        rows.append(
            {
                "fold": fold.fold_index,
                **{
                    k: metrics[k]
                    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
                },
            }
        )
        checkpoints.append(fold_dir / "best.pth")
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stats = compute_aggregate_statistics(
        [{k: v for k, v in row.items() if k != "fold"} for row in rows]
    )
    pd.DataFrame(rows).to_csv(output / "kfold_metrics.csv", index=False)
    plot_kfold_boxplot(rows, output / "kfold_boxplot.png", model_name=args.model)
    test_metrics = None
    if args.evaluate_test:
        test = ensure_sample_ids(pd.read_csv(split_dir / "test.csv"))
        if set(test.sample_id) & set(dev.sample_id):
            raise ValueError("Development and holdout share sample IDs")
        dataset = CornDataset(
            test, args.config, transform=factory.get_pipeline("test"), class_to_idx=mapping
        )
        loader = DataLoader(dataset, batch_size=bs, num_workers=args.num_workers)
        total_probs = None
        for checkpoint in checkpoints:
            model = build_model(args.model, num_classes=len(mapping), pretrained=False).to(device)
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True), strict=True
            )
            model.eval()
            ids, labels, values = [], [], []
            with torch.no_grad():
                for images, targets, sample_ids in identified_batches(loader):
                    ids.extend(sample_ids)
                    labels.extend(targets.tolist())
                    values.extend(model(images.to(device)).softmax(1).cpu().tolist())
            values = np.array(values)
            total_probs = values if total_probs is None else total_probs + values
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        probs = total_probs / len(checkpoints)
        predictions = probs.argmax(1).tolist()
        test_metrics = _metrics_from_predictions(
            labels, predictions, 0.0, list(range(len(mapping)))
        )
        write_predictions_csv(
            output,
            dataset,
            dict(enumerate(class_names)),
            predictions,
            probs.max(1).tolist(),
            sample_ids=ids,
        )
        plot_test_confusion_matrix(
            labels, predictions, class_names, output / "confusion_matrix_test.png"
        )
    atomic_json(
        output / "kfold_summary.json",
        {
            "model": args.model,
            "class_policy": "canonical classes, zero division=0",
            "kfold_validation_statistics": stats,
            "fold_checkpoints": [str(p) for p in checkpoints],
            "hold_out_test_metrics": test_metrics,
            "effective_args": vars(args),
            "hpo_parameters": params,
            "limitation": "CV after prior HPO is not independent nested CV.",
        },
    )
    print(f"Cross-validation artifacts: {output}")


if __name__ == "__main__":
    main()
