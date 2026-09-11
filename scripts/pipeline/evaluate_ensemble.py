"""Evaluación y Comparativa de Ensamble por Soft Voting (Criterio 2 - Rúbrica Etapa 2).

Evalúa las arquitecturas canónicas de forma individual y conjunta sobre el split de prueba (test.csv)
para calcular el techo de rendimiento teórico del ensamble frente a los modelos individuales.

Uso:
    python scripts/pipeline/evaluate_ensemble.py
    python scripts/pipeline/evaluate_ensemble.py --models efficientnet_b0 shufflenet_v2_x1_0 efficientnet_lite0
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, get_output_root, set_global_seed
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.models import build_model
from src.models.ensemble import SoftVotingEnsemble
from src.training.common import select_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ensemble")

DEFAULT_CANONICAL_MODELS = ["efficientnet_b0", "shufflenet_v2_x1_0", "efficientnet_lite0"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación comparativa de Ensamble por Soft Voting (Etapa 2)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_CANONICAL_MODELS,
        help="Modelos a incluir en el ensamble.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Rutas explícitas a los checkpoints .pt de cada modelo (mismo orden que --models).",
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
        help="Destino de reportes y matrices del ensamble (default: <outputs>/ensemble)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Tamaño de lote para evaluación.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Pesos de votación para cada modelo en el ensamble (ej: 0.4 0.3 0.3).",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
        help="Ruta al archivo dataset.yaml.",
    )
    return parser.parse_args()


def _find_checkpoint(model_name: str, output_root: Path) -> Path | None:
    """Busca el checkpoint más reciente en outputs/main/ o outputs/baselines/."""
    for pipeline_dir in ["main", "baselines"]:
        parent = output_root / pipeline_dir / model_name
        if not parent.exists():
            continue
        latest_json = parent / "latest.json"
        if latest_json.exists():
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                run_id = meta.get("run_id") or meta.get("run")
                if run_id:
                    for name in ["best.pth", "best.pt"]:
                        p = parent / run_id / name
                        if p.exists():
                            return p
            except Exception:
                pass

        for name in ["best.pth", "best.pt"]:
            for cand in [
                parent / "latest" / "checkpoints" / name,
                parent / "latest" / name,
                parent / name,
            ]:
                if cand.exists():
                    return cand

        pts = list(parent.rglob("best.pth")) + list(parent.rglob("best.pt"))
        if pts:
            return sorted(pts, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    labels_names: list[str],
) -> dict[str, float]:
    """Calcula métricas multiclase estándar."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
    title: str = "Matriz de Confusión — Soft Voting Ensemble",
) -> None:
    """Genera y guarda el mapa de calor de la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
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


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    output_root = get_output_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else output_root / "splits" / "seed_42"
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not splits_dir.exists():
        logger.error("El directorio de splits no existe: %s\nGenera los splits primero.", splits_dir)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_global_seed(cfg["dataset"]["seed"])
    device = select_device()

    # 1. Resolver Checkpoints
    model_names = list(args.models)
    checkpoint_paths: list[Path] = []
    if args.checkpoints:
        if len(args.checkpoints) != len(model_names):
            logger.error("La cantidad de checkpoints no coincide con la cantidad de modelos.")
            sys.exit(1)
        checkpoint_paths = [Path(p) for p in args.checkpoints]
        for p in checkpoint_paths:
            if not p.exists():
                logger.error("Checkpoint explícito no encontrado: %s", p)
                sys.exit(1)
    else:
        for m in model_names:
            found = _find_checkpoint(m, output_root)
            if not found:
                logger.error(
                    "No se encontró checkpoint entrenado para '%s'. "
                    "Entrena primero o pasa --checkpoints con rutas explícitas.",
                    m,
                )
                sys.exit(1)
            checkpoint_paths.append(found)

    # 2. Validar consistencia de class_to_idx entre miembros del ensamble
    reference_class_to_idx: dict[str, int] | None = None
    for name, ckpt in zip(model_names, checkpoint_paths):
        summary_path = ckpt.parent / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            if "class_to_idx" in summary:
                member_mapping = {str(k): int(v) for k, v in summary["class_to_idx"].items()}
                if reference_class_to_idx is None:
                    reference_class_to_idx = member_mapping
                    logger.info(
                        "Mapeo de clases obtenido de summary.json de '%s': %s",
                        name, list(member_mapping.keys()),
                    )
                elif member_mapping != reference_class_to_idx:
                    logger.error(
                        "Inconsistencia en class_to_idx entre miembros del ensamble.\n"
                        "  Referencia: %s\n  %s tiene: %s",
                        reference_class_to_idx, name, member_mapping,
                    )
                    sys.exit(1)
        else:
            logger.warning("No se encontró summary.json para '%s' en %s", name, ckpt.parent)

    # 3. Preparar Dataset de Test
    factory = CornTransformFactory(config_path=str(config_path), target_size=(224, 224), clahe=False)
    test_dataset = CornDataset(
        csv_path=str(splits_dir / "test.csv"),
        config_path=str(config_path),
        transform=factory.get_pipeline("test"),
        **(dict(class_to_idx=reference_class_to_idx) if reference_class_to_idx else {}),
    )
    if reference_class_to_idx is None:
        reference_class_to_idx = test_dataset.class_to_idx
        logger.warning(
            "No se encontró summary.json en ningún miembro; usando mapeo del dataset de test."
        )
    class_to_idx = reference_class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Cargar Modelos Individuales y Construir Ensamble
    loaded_models: list[torch.nn.Module] = []
    for name, ckpt in zip(model_names, checkpoint_paths):
        model = build_model(name, num_classes=len(class_to_idx), pretrained=False)
        checkpoint_data = torch.load(ckpt, map_location=device)
        state_dict = (
            checkpoint_data["model_state_dict"]
            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data
            else checkpoint_data
        )
        model.load_state_dict(state_dict, strict=True)
        logger.info("Cargado checkpoint para %s desde %s", name, ckpt)
        model.eval()
        model.to(device)
        loaded_models.append(model)

    ensemble = SoftVotingEnsemble(models=loaded_models, weights=args.weights, model_names=model_names)

    # 4. Evaluación Individual y del Ensamble
    logger.info("=== EVALUANDO MODELOS INDIVIDUALES Y ENSAMBLE SOBRE TEST SET ===")
    results_per_model: dict[str, dict[str, Any]] = {}
    all_targets: list[int] = []
    individual_predictions: dict[str, list[int]] = {m: [] for m in model_names}
    ensemble_predictions: list[int] = []
    ensemble_probs_list: list[list[float]] = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            all_targets.extend(targets.tolist())

            # Predicciones individuales
            for model_name, model in zip(model_names, loaded_models):
                logits = model(images)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                individual_predictions[model_name].extend(preds)

            # Predicción del ensamble
            ens_probs = ensemble.predict_probabilities(images)
            ens_preds = torch.argmax(ens_probs, dim=-1).cpu().tolist()
            ensemble_predictions.extend(ens_preds)
            ensemble_probs_list.extend(ens_probs.cpu().tolist())

    # 5. Métricas Individuales
    comparison_table: list[dict[str, Any]] = []
    for model_name in model_names:
        m_metrics = evaluate_predictions(all_targets, individual_predictions[model_name], class_names)
        results_per_model[model_name] = m_metrics
        comparison_table.append(
            {
                "Tipo": "Modelo Individual",
                "Arquitectura": model_name,
                "Macro F1": round(m_metrics["macro_f1"], 4),
                "Accuracy": round(m_metrics["accuracy"], 4),
                "Macro Precision": round(m_metrics["macro_precision"], 4),
                "Macro Recall": round(m_metrics["macro_recall"], 4),
            }
        )

    # 6. Métricas del Ensamble
    ens_metrics = evaluate_predictions(all_targets, ensemble_predictions, class_names)
    results_per_model["soft_voting_ensemble"] = ens_metrics

    # Calcular mejor modelo individual para comparar delta
    best_single_f1 = max(results_per_model[m]["macro_f1"] for m in model_names)
    delta_vs_best = ens_metrics["macro_f1"] - best_single_f1

    comparison_table.append(
        {
            "Tipo": "Ensamble (Soft Voting)",
            "Arquitectura": " + ".join(model_names),
            "Macro F1": round(ens_metrics["macro_f1"], 4),
            "Accuracy": round(ens_metrics["accuracy"], 4),
            "Macro Precision": round(ens_metrics["macro_precision"], 4),
            "Macro Recall": round(ens_metrics["macro_recall"], 4),
        }
    )

    # 7. Guardar Artefactos
    df_comparison = pd.DataFrame(comparison_table)
    df_comparison.to_csv(output_dir / "ensemble_comparison.csv", index=False)

    summary = {
        "models_included": model_names,
        "weights": [float(w) for w in ensemble.weights.cpu().tolist()],
        "metrics_per_model": results_per_model,
        "best_individual_macro_f1": round(best_single_f1, 4),
        "ensemble_macro_f1": round(ens_metrics["macro_f1"], 4),
        "ensemble_gain_delta": round(delta_vs_best, 4),
        "total_test_samples": len(all_targets),
    }
    with open(output_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Guardar matriz de confusión
    cm_path = output_dir / "confusion_matrix_ensemble.png"
    plot_confusion_matrix(all_targets, ensemble_predictions, class_names, cm_path)

    logger.info("=== RESULTADOS DEL ENSAMBLE ===")
    logger.info("Mejor Modelo Individual Macro F1: %.4f", best_single_f1)
    logger.info("Soft Voting Ensemble Macro F1:    %.4f (Delta: %+.4f)", ens_metrics["macro_f1"], delta_vs_best)
    logger.info("Accuracy del Ensamble:            %.4f", ens_metrics["accuracy"])
    logger.info("Artefactos guardados en: %s", output_dir)


if __name__ == "__main__":
    main()
