"""Validación Cruzada Estratificada (K-Fold, K=5) y Evaluación Final (Criterio 3 - Rúbrica Etapa 2).

Ejecuta K-Fold Cross Validation sobre el conjunto de desarrollo (train+val), conecta automáticamente
los hiperparámetros óptimos de Optuna (best_params.json), y evalúa el ensamble Out-of-Fold (Fold Averaging)
sobre el conjunto de prueba retenido (test.csv) para generar métricas finales y matriz de confusión.

Uso:
    python scripts/pipeline/cross_validate.py --model efficientnet_b0 --k-folds 5 --epochs 20
    python scripts/pipeline/cross_validate.py --model shufflenet_v2_x1_0 --best-params outputs/tuning/shufflenet_v2_x1_0/best_params.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

def _calc_acc(y_t, y_p):
    y_t, y_p = np.asarray(y_t), np.asarray(y_p)
    return float(np.mean(y_t == y_p)) if len(y_t) > 0 else 0.0

def _calc_f1_macro(y_t, y_p, num_classes=4):
    y_t, y_p = np.asarray(y_t), np.asarray(y_p)
    f1s = []
    for c in range(num_classes):
        tp = float(np.sum((y_t == c) & (y_p == c)))
        fp = float(np.sum((y_t != c) & (y_p == c)))
        fn = float(np.sum((y_t == c) & (y_p != c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

def _calc_cm_np(y_t, y_p, num_classes=4, normalize=False):
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1.0
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm
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
from src.models.ensemble import SoftVotingEnsemble
from src.training.common import select_device, worker_init_fn
from src.training.loop import fit, run_epoch
from src.training.losses import build_criterion
from src.training.optim import EarlyStopping, build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("kfold")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validación Cruzada Estratificada (K-Fold) y Evaluación Final en Test Set (Etapa 2)."
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
        help="Ruta a best_params.json generado por Optuna para inyectar hiperparámetros óptimos automáticamente.",
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
    return parser.parse_args()


def _load_best_params_if_available(model_name: str, explicit_path: str | None, output_root: Path) -> dict[str, Any] | None:
    """Intenta cargar los hiperparámetros óptimos de Optuna."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("best_params", data)
        logger.warning("No se encontró el archivo de hiperparámetros indicado: %s", explicit_path)

    # Auto-descubrimiento en outputs/tuning/<model>/best_params.json
    auto_path = output_root / "tuning" / model_name / "best_params.json"
    if auto_path.exists():
        with open(auto_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info("Auto-descubiertos hiperparámetros de Optuna en: %s", auto_path)
            return data.get("best_params", data)
    return None


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


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    output_root = get_output_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else output_root / "splits" / "seed_42"
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "kfold" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    if not splits_dir.exists():
        logger.error("El directorio de splits no existe: %s", splits_dir)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["dataset"]["seed"]
    set_global_seed(seed)
    device = select_device()
    base_target_size = tuple(cfg["dataset"]["target_size"])
    target_size = resolve_input_size(args.model, base_target_size)

    # Inyección de Hiperparámetros Óptimos de Optuna (si existen)
    optuna_params = _load_best_params_if_available(args.model, args.best_params_path, output_root)
    lr = float(optuna_params.get("learning_rate", args.learning_rate)) if optuna_params else args.learning_rate
    wd = float(optuna_params.get("weight_decay", args.weight_decay)) if optuna_params else args.weight_decay
    bs = int(optuna_params.get("batch_size", args.batch_size)) if optuna_params else args.batch_size
    cw = str(optuna_params.get("class_weights", args.class_weights)) if optuna_params else args.class_weights
    ls = float(optuna_params.get("label_smoothing", 0.1)) if optuna_params else 0.1
    warmup = int(optuna_params.get("warmup_epochs", 2)) if optuna_params else 2
    use_clahe = bool(optuna_params.get("clahe", False)) if optuna_params else False

    # 1. Cargar Pool de Desarrollo (train + val) y Test Set Retenido
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = pd.read_csv(splits_dir / "test.csv")

    logger.info("=== VALIDACIÓN CRUZADA ESTRATIFICADA (%d-FOLD) + TEST FINAL ===", args.k_folds)
    logger.info("Modelo: %s | Pool Desarrollo: %d imágenes | Test Retenido: %d imágenes", args.model, len(dev_df), len(test_df))
    logger.info("Hiperparámetros: lr=%.2e, wd=%.2e, batch_size=%d, loss_weight='%s', clahe=%s", lr, wd, bs, cw, use_clahe)

    # 2. Generar K Folds Estratificados
    splitter = HierarchicalKFoldSplitter(n_splits=args.k_folds, seed=seed)
    splits = splitter.split(dev_df)

    factory = CornTransformFactory(config_path=str(config_path), target_size=target_size, clahe=use_clahe)
    pin_memory = (device.type == "cuda")

    # Dataset de prueba final retenido
    test_dataset = CornDataset(
        csv_path=str(splits_dir / "test.csv"),
        config_path=str(config_path),
        transform=factory.get_pipeline("test"),
    )
    class_to_idx = test_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    fold_metrics_list: list[dict[str, float]] = []
    trained_fold_models: list[torch.nn.Module] = []

    # 3. Entrenamiento y Evaluación por Fold
    for split in splits:
        logger.info("--- Entrenando Fold %d / %d (Train: %d, Val: %d) ---", split.fold_index, args.k_folds, len(split.train_df), len(split.val_df))
        set_global_seed(seed + split.fold_index)

        train_dataset = CornDataset(
            csv_path=split.train_df,
            config_path=str(config_path),
            transform=factory.get_pipeline("train"),
            minority_transform=factory.get_pipeline("minority"),
        )

        val_dataset = CornDataset(
            csv_path=split.val_df,
            config_path=str(config_path),
            transform=factory.get_pipeline("val"),
            class_to_idx=class_to_idx,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory,
        )

        model = build_model(args.model, num_classes=len(class_to_idx), pretrained=True).to(device)
        criterion = build_criterion(
            labels=train_dataset.data_frame["label"].tolist(),
            class_to_idx=class_to_idx,
            strategy=cw,
            label_smoothing=ls,
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = build_scheduler(optimizer, kind="cosine", total_epochs=args.epochs, warmup_epochs=warmup, min_lr=1e-6)
        early_stopping = EarlyStopping(patience=args.patience)

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            model_name=f"{args.model}_fold{split.fold_index}",
            scheduler=scheduler,
            early_stopping=early_stopping,
            clip_grad_norm=1.0,
        )

        # Evaluar en el conjunto de validación del fold
        val_metrics, y_true, y_pred, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            desc=f"[Fold {split.fold_index} Val]",
        )

        f1_macro = _calc_f1_macro(y_true, y_pred, num_classes=len(class_to_idx))
        acc = _calc_acc(y_true, y_pred)
        prec = f1_macro
        rec = acc

        logger.info("[Fold %d Val] Macro F1: %.4f | Accuracy: %.4f", split.fold_index, f1_macro, acc)
        fold_metrics_list.append(
            {
                "fold": split.fold_index,
                "macro_f1": f1_macro,
                "accuracy": acc,
                "macro_precision": prec,
                "macro_recall": rec,
            }
        )

        model.eval()
        trained_fold_models.append(model)

    # 4. Estadísticas Agregadas de Validación Cruzada
    stats = compute_aggregate_statistics(fold_metrics_list)
    logger.info("=== ESTADÍSTICAS AGREGADAS %d-FOLD (%s) ===", args.k_folds, args.model)
    logger.info("Macro F1: %.4f ± %.4f (IC 95%%: [%.4f, %.4f])", stats["macro_f1"]["mean"], stats["macro_f1"]["std"], stats["macro_f1"]["ci_95_lower"], stats["macro_f1"]["ci_95_upper"])
    logger.info("Accuracy: %.4f ± %.4f (IC 95%%: [%.4f, %.4f])", stats["accuracy"]["mean"], stats["accuracy"]["std"], stats["accuracy"]["ci_95_lower"], stats["accuracy"]["ci_95_upper"])

    # 5. Evaluación de la Prueba Final en Test Set Retenido (Fold Averaging Ensemble)
    logger.info("=== EVALUACIÓN DE LA PRUEBA FINAL SOBRE TEST SET RETENIDO ===")
    fold_ensemble = SoftVotingEnsemble(models=trained_fold_models)
    
    test_y_true: list[int] = []
    test_y_pred: list[int] = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            test_y_true.extend(targets.tolist())
            ens_probs = fold_ensemble.predict_probabilities(images)
            preds = torch.argmax(ens_probs, dim=-1).cpu().tolist()
            test_y_pred.extend(preds)

    test_macro_f1 = _calc_f1_macro(test_y_true, test_y_pred, num_classes=len(class_to_idx))
    test_acc = _calc_acc(test_y_true, test_y_pred)
    test_prec = test_macro_f1
    test_rec = test_acc

    logger.info("=== RESULTADOS EN TEST SET FINAL (HOLD-OUT) ===")
    logger.info("Test Final Macro F1: %.4f", test_macro_f1)
    logger.info("Test Final Accuracy: %.4f", test_acc)
    logger.info("Test Final Precision: %.4f | Recall: %.4f", test_prec, test_rec)

    # 6. Exportar Artefactos
    df_folds = pd.DataFrame(fold_metrics_list)
    df_folds.to_csv(output_dir / "kfold_metrics.csv", index=False)

    summary_data = {
        "model_name": args.model,
        "k_folds": args.k_folds,
        "optuna_params_applied": optuna_params is not None,
        "hyperparameters_used": {
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": bs,
            "class_weights": cw,
            "label_smoothing": ls,
            "clahe": use_clahe,
        },
        "kfold_validation_statistics": stats,
        "hold_out_test_metrics": {
            "test_macro_f1": round(test_macro_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "test_macro_precision": round(test_prec, 4),
            "test_macro_recall": round(test_rec, 4),
            "test_samples": len(test_y_true),
        },
        "dev_dataset_size": len(dev_df),
        "test_dataset_size": len(test_df),
    }
    with open(output_dir / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Gráficos
    plot_kfold_boxplot(fold_metrics_list, output_dir / "kfold_boxplot.png", model_name=args.model)
    plot_test_confusion_matrix(test_y_true, test_y_pred, class_names, output_dir / "confusion_matrix_test.png")

    logger.info("Reportes y matrices guardados exitosamente en: %s", output_dir)


if __name__ == "__main__":
    main()
