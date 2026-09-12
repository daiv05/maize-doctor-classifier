"""Domain metrics and spatial sensitivity from an identified, trained checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.analysis.fairness import (
    compute_disparity_metrics,
    compute_subgroup_metrics,
    evaluate_dual_shortcut_audit,
    plot_disaggregated_confusion_matrices,
    plot_subgroup_disparity_bars,
)
from src.config import PROJECT_ROOT, get_dataset_root, get_output_root, set_global_seed
from src.data.dataset import CornDataset
from src.data.identity import align_predictions, identified_batches
from src.data.loader import load_and_normalize_image
from src.data.transforms import CornTransformFactory
from src.explainability.gradcam import GradCAM, build_gradcam_overlay, get_target_layer
from src.export.data import resolve_split_csv
from src.provenance import atomic_json, sha256_file
from src.training.common import generate_run_id, select_device
from src.training.runs import load_run, resolve_checkpoint

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--checkpoint", dest="checkpoint_path")
    parser.add_argument("--run")
    parser.add_argument("--pipeline", choices=["main", "baselines"], default="main")
    parser.add_argument("--splits-dir")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/dataset.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-gradcam", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-shortcut-test", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_checkpoint(model_name, explicit_path):
    return resolve_checkpoint(model_name, get_output_root(), explicit_path)


def _generate_gradcam_panel(
    model: nn.Module,
    model_name: str,
    test_df: pd.DataFrame,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    device: torch.device,
    output_path: Path,
    input_size: tuple[int, int],
    factory: CornTransformFactory,
) -> None:
    """Genera un panel comparativo de Grad-CAM auditando casos en campo real y laboratorio."""
    try:
        target_layer = get_target_layer(model, model_name)
    except Exception as e:
        logger.warning("No se pudo resolver la capa Grad-CAM para %s: %s", model_name, e)
        return

    dataset_root = get_dataset_root()
    tf_eval = factory.get_pipeline("test")

    # Seleccionar muestras representativas: 2 de campo real y 2 de lab
    real_samples = test_df[test_df["environment"] == "real"].head(2)
    lab_samples = test_df[test_df["environment"] == "lab"].head(2)
    selected_samples = pd.concat([real_samples, lab_samples]).reset_index(drop=True)

    if selected_samples.empty:
        logger.warning("No se encontraron muestras suficientes para Grad-CAM.")
        return

    fig, axes = plt.subplots(
        len(selected_samples), 3, figsize=(12, 3.5 * len(selected_samples)), dpi=150
    )
    if len(selected_samples) == 1:
        axes = np.expand_dims(axes, 0)

    model.eval()

    for row_idx, row in selected_samples.iterrows():
        img_rel_path = row["image_path"]
        img_path = dataset_root / img_rel_path
        true_label = row["label"]
        env = row["environment"]

        if not img_path.exists():
            continue

        pil_img = load_and_normalize_image(img_path)
        img_tensor = tf_eval(pil_img).unsqueeze(0).to(device)

        # Inferencia
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs, dim=-1).item())
            pred_conf = float(probs[0, pred_idx].item())
            pred_label = idx_to_class.get(pred_idx, f"Clase_{pred_idx}")

        # Grad-CAM heatmap
        with GradCAM(model, target_layer) as cam_extractor:
            cam_map = cam_extractor(img_tensor, pred_idx)

        img_np01 = np.array(pil_img.resize((input_size[1], input_size[0]))) / 255.0
        overlay = build_gradcam_overlay(img_np01, cam_map, input_size)

        # Columna 1: Imagen Original
        axes[row_idx, 0].imshow(img_np01)
        axes[row_idx, 0].set_title(
            f"Original ({env})\nReal: {true_label}", fontsize=9, fontweight="bold"
        )
        axes[row_idx, 0].axis("off")

        # Columna 2: Mapa de Calor Grad-CAM
        upsampled_cam = (
            torch.nn.functional.interpolate(
                cam_map[None, None, :, :],
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        axes[row_idx, 1].imshow(upsampled_cam, cmap="jet")
        axes[row_idx, 1].set_title(
            f"Mapa Grad-CAM (Atención)\nPred: {pred_label} ({pred_conf * 100:.1f}%)",
            fontsize=9,
            fontweight="bold",
        )
        axes[row_idx, 1].axis("off")

        # Columna 3: Superposición (Overlay)
        axes[row_idx, 2].imshow(overlay)
        status = "CORRECTO" if true_label == pred_label else "ERROR"
        axes[row_idx, 2].set_title(
            f"Superposición [{status}]\nActivación cualitativa",
            fontsize=9,
            fontweight="bold",
            color="green" if status == "CORRECTO" else "red",
        )
        axes[row_idx, 2].axis("off")

    plt.suptitle(
        f"Auditoría Visual Grad-CAM: Explicabilidad y Atajos Visuales ({model_name})",
        fontsize=13,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Panel Grad-CAM guardado en: %s", output_path)


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    set_global_seed(args.seed)
    device = select_device()
    checkpoint = resolve_checkpoint(
        args.model, get_output_root(), args.checkpoint_path, args.run, args.pipeline
    )
    run = load_run(checkpoint, args.model, device, config_path=args.config)
    split_csv = resolve_split_csv(checkpoint.parent, args.splits_dir, args.split)
    dataset = CornDataset(
        str(split_csv),
        args.config,
        transform=run.factory.get_pipeline("test"),
        class_to_idx=run.class_to_idx,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    output = (
        Path(args.output_dir)
        if args.output_dir
        else get_output_root() / "fairness" / generate_run_id()
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    class_names = sorted(run.class_to_idx, key=run.class_to_idx.get)
    ids, y_true, y_pred, confidence = [], [], [], []
    with torch.no_grad():
        for images, targets, sample_ids in identified_batches(loader):
            probabilities = run.model(images.to(device)).softmax(-1).cpu()
            ids.extend(sample_ids)
            y_true.extend(targets.tolist())
            y_pred.extend(probabilities.argmax(-1).tolist())
            confidence.extend(probabilities.max(-1).values.tolist())
    frame = align_predictions(dataset.data_frame, ids)
    if "environment" not in frame:
        frame["environment"] = "unknown"
    frame["pred_label"] = [class_names[i] for i in y_pred]
    frame["pred_prob"] = confidence
    frame.to_csv(output / "predictions.csv", index=False)
    groups = compute_subgroup_metrics(y_true, y_pred, frame.environment.tolist(), class_names)
    disparity = compute_disparity_metrics(groups)
    sensitivity = (
        evaluate_dual_shortcut_audit(run.model, loader, device) if args.run_shortcut_test else None
    )
    payload = {
        "schema_version": 2,
        "run": run.manifest_entry(),
        "split": args.split,
        "split_sha256": sha256_file(split_csv),
        "subgroup_metrics": groups,
        "disparity_analysis": disparity,
        "shortcut_learning_audit": sensitivity,
    }
    atomic_json(output / "fairness_metrics.json", payload)
    rows = [
        {
            **{key: value for key, value in values.items() if not isinstance(value, (dict, list))},
            "subgroup": name,
        }
        for name, values in groups["subgroups"].items()
    ]
    pd.DataFrame(rows).to_csv(output / "fairness_disparity.csv", index=False)
    plot_subgroup_disparity_bars(groups, output / "fairness_disparity.png")
    lab, real = frame.environment.eq("lab").to_numpy(), frame.environment.eq("real").to_numpy()
    if lab.any() and real.any():
        plot_disaggregated_confusion_matrices(
            np.array(y_true)[lab],
            np.array(y_pred)[lab],
            np.array(y_true)[real],
            np.array(y_pred)[real],
            class_names,
            output / "disaggregated_confusion_matrices.png",
        )
    if args.run_gradcam:
        _generate_gradcam_panel(
            model=run.model,
            model_name=args.model,
            test_df=frame,
            class_to_idx=run.class_to_idx,
            idx_to_class=dict(enumerate(class_names)),
            device=device,
            output_path=output / "gradcam_samples.png",
            input_size=run.factory.target_size,
            factory=run.factory,
        )
    logger.info("Evaluation written to %s; comparison status=%s", output, disparity["status"])


if __name__ == "__main__":
    main()
