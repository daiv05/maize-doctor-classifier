"""Script CLI de Evaluación de Sesgos, Equidad y Control de Atajos Visuales (Criterio 4 - Rúbrica Etapa 2).

Ejecuta la auditoría de equidad algorítmica:
1. Evaluación desagregada por entorno (lab vs real).
2. Cálculo de métricas de disparidad (Δ_F1, Disparate Impact Ratio DIR, tasas de falsos negativos FNR).
3. Test de control negativo contra atajos visuales (Shortcut Learning / Efecto Clever Hans).
4. Generación de mapas de activación visual con Grad-CAM.
5. Exportación de artefactos estructurados (CSV, JSON, PNG) y reporte en Markdown.

Uso:
    python scripts/pipeline/evaluate_fairness.py --model efficientnet_b0 --run-gradcam --run-shortcut-test
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
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.analysis.fairness import (
    compute_disparity_metrics,
    compute_subgroup_metrics,
    evaluate_background_shortcut,
    evaluate_dual_shortcut_audit,
    plot_disaggregated_confusion_matrices,
    plot_subgroup_disparity_bars,
)
from src.config import PROJECT_ROOT, get_dataset_root, get_output_root, set_global_seed
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.explainability.gradcam import GradCAM, build_gradcam_overlay, get_target_layer
from src.models import build_model, list_models, resolve_input_size
from src.training.common import select_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fairness_audit")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditoría de Equidad, Sesgos y Control de Atajos Visuales (Criterio 4 - Etapa 2)."
    )
    parser.add_argument(
        "--model",
        default="efficientnet_b0",
        help=f"Arquitectura a auditar. Opciones: {list_models()}",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        dest="checkpoint_path",
        help="Ruta al checkpoint .pt del modelo entrenado. Si no se especifica, busca en outputs/main/ o outputs/baselines/.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio que contiene test.csv (default: <outputs>/splits/seed_42).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio de salida para los artefactos de equidad (default: <outputs>/fairness).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Tamaño de lote para la inferencia de evaluación.",
    )
    parser.add_argument(
        "--run-gradcam",
        action="store_true",
        default=True,
        dest="run_gradcam",
        help="Generar visualizaciones comparativas de mapas de calor con Grad-CAM.",
    )
    parser.add_argument(
        "--run-shortcut-test",
        action="store_true",
        default=True,
        dest="run_shortcut_test",
        help="Ejecutar el test de control negativo de oclusión foliar (Clever Hans check).",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
        help="Ruta al archivo dataset.yaml de configuración.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    return parser.parse_args()


def _resolve_checkpoint(model_name: str, explicit_path: str | None) -> Path | None:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        logger.warning("Checkpoint especificado no encontrado: %s", explicit_path)

    output_root = get_output_root()
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
                            logger.info("Checkpoint auto-descubierto: %s", p)
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
                    logger.info("Checkpoint auto-descubierto: %s", cand)
                    return cand

        pts = list(parent.rglob("best.pth")) + list(parent.rglob("best.pt"))
        if pts:
            chosen = sorted(pts, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            logger.info("Checkpoint auto-descubierto: %s", chosen)
            return chosen

    return None


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

    fig, axes = plt.subplots(len(selected_samples), 3, figsize=(12, 3.5 * len(selected_samples)), dpi=150)
    if len(selected_samples) == 1:
        axes = np.expand_dims(axes, 0)

    model.eval()

    for row_idx, row in selected_samples.iterrows():
        img_rel_path = row["image_path"]
        img_path = dataset_root / img_rel_path
        true_label = row["label"]
        env = row["environment"]
        true_idx = class_to_idx.get(true_label, 0)

        if not img_path.exists():
            continue

        pil_img = Image.open(img_path).convert("RGB")
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
        axes[row_idx, 0].set_title(f"Original ({env})\nReal: {true_label}", fontsize=9, fontweight="bold")
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
        axes[row_idx, 1].set_title(f"Mapa Grad-CAM (Atención)\nPred: {pred_label} ({pred_conf*100:.1f}%)", fontsize=9, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Columna 3: Superposición (Overlay)
        axes[row_idx, 2].imshow(overlay)
        status = "CORRECTO" if true_label == pred_label else "ERROR"
        axes[row_idx, 2].set_title(f"Superposición [{status}]\nFoco en Lesión Foliar", fontsize=9, fontweight="bold", color="green" if status == "CORRECTO" else "red")
        axes[row_idx, 2].axis("off")

    plt.suptitle(f"Auditoría Visual Grad-CAM: Explicabilidad y Atajos Visuales ({model_name})", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Panel Grad-CAM guardado en: %s", output_path)


def main() -> None:
    args = _parse_args()
    set_global_seed(args.seed)

    splits_dir = Path(args.splits_dir) if args.splits_dir else (get_output_root() / "splits" / f"seed_{args.seed}")
    test_csv_path = splits_dir / "test.csv"
    if not test_csv_path.exists():
        logger.error("No se encontró test.csv en %s. Ejecuta primero 'make splits'.", splits_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else (get_output_root() / "fairness")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_csv_path)
    if "environment" not in test_df.columns:
        logger.warning("Columna 'environment' no presente en test.csv. Asignando 'unknown'.")
        test_df["environment"] = "unknown"

    class_names = sorted(test_df["label"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    logger.info("Iniciando Auditoría de Equidad (Fairness Report) para: %s", args.model)
    logger.info("Clases a auditar (%d): %s", len(class_names), class_names)
    logger.info("Total muestras en Test Set: %d (Lab: %d, Real: %d)", len(test_df), (test_df['environment'] == 'lab').sum(), (test_df['environment'] == 'real').sum())

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_target_size = tuple(cfg["dataset"]["target_size"])

    device = select_device()
    input_size = resolve_input_size(args.model, fallback=base_target_size)
    factory = CornTransformFactory(config_path=str(config_path), target_size=input_size)
    eval_transform = factory.get_pipeline("test")

    # Instanciar y cargar modelo
    model = build_model(args.model, num_classes=len(class_names), pretrained=True)
    ckpt_path = _resolve_checkpoint(args.model, args.checkpoint_path)
    if ckpt_path and ckpt_path.exists():
        logger.info("Cargando pesos entrenados desde: %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning("No se encontró checkpoint entrenado; auditando con pesos pre-entrenados/iniciales.")

    model = model.to(device)
    model.eval()

    # DataLoader de prueba
    test_dataset = CornDataset(
        csv_path=str(test_csv_path),
        config_path=str(config_path),
        transform=eval_transform,
        class_to_idx=class_to_idx,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Inferencia completa
    y_true: list[int] = []
    y_pred: list[int] = []
    subgroups: list[str] = test_df["environment"].tolist()

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            y_true.extend(targets.tolist())
            y_pred.extend(preds)

    # 1. Calcular métricas desagregadas por subgrupo
    subgroup_metrics = compute_subgroup_metrics(y_true, y_pred, subgroups, class_names)
    disparity_metrics = compute_disparity_metrics(subgroup_metrics)

    logger.info("=== RESULTADOS DESAGREGADOS POR ENTORNO ===")
    for grp, m in subgroup_metrics["subgroups"].items():
        logger.info("Subgrupo [%s] (N=%d) -> Macro F1: %.4f | Accuracy: %.4f | Precision: %.4f | Recall: %.4f", grp, m["sample_count"], m["macro_f1"], m["accuracy"], m["macro_precision"], m["macro_recall"])

    logger.info("=== MÉTRICAS DE DISPARIDAD Y EQUIDAD ===")
    logger.info("Delta Macro F1 (|Real - Lab|): %.4f", disparity_metrics["delta_macro_f1"])
    logger.info("Delta Accuracy (|Real - Lab|): %.4f", disparity_metrics["delta_accuracy"])
    logger.info("Disparate Impact Ratio (DIR): %.4f (Regla 80%% cumplida: %s)", disparity_metrics["disparate_impact_ratio"], disparity_metrics["four_fifths_rule_passed"])

    # 2. Control Negativo y Control Inverso contra Atajos Visuales (Clever Hans Audit)
    dual_shortcut_results: dict[str, Any] = {}
    if args.run_shortcut_test:
        logger.info("Ejecutando Auditoría Dual de Atajos Visuales (Clever Hans check)...")
        dual_shortcut_results = evaluate_dual_shortcut_audit(
            model=model,
            loader=test_loader,
            device=device,
        )
        c_res = dual_shortcut_results["center_occlusion"]
        p_res = dual_shortcut_results["peripheral_occlusion"]

        logger.info("=== RESULTADOS TEST DE CONTROL NEGATIVO (OCLUSIÓN CENTRAL 60%%) ===")
        logger.info("Confianza Original Media: %.4f -> Confianza sin Centro: %.4f (Caída: %.4f)", c_res["mean_original_confidence"], c_res["mean_masked_confidence"], c_res["confidence_drop"])
        logger.info("Ratio de Retención de Certeza en Fondo: %.4f (Atajo detectado: %s, Riesgo: %s)", c_res["shortcut_vulnerability_ratio"], c_res["shortcut_detected"], c_res["risk_level"])
        logger.info("Exactitud Original: %.4f -> Exactitud sin Centro: %.4f (Caída Acc: %.4f, Flips: %.4f)", c_res["accuracy_original"], c_res["accuracy_masked"], c_res["accuracy_drop"], c_res["flip_rate"])

        logger.info("=== RESULTADOS CONTROL INVERSO (OCLUSIÓN PERIFÉRICA 40%% - SOLO CENTRO) ===")
        logger.info("Confianza Original Media: %.4f -> Confianza solo Centro: %.4f (Caída: %.4f)", p_res["mean_original_confidence"], p_res["mean_masked_confidence"], p_res["confidence_drop"])
        logger.info("Exactitud Original: %.4f -> Exactitud solo Centro: %.4f (Caída Acc: %.4f, Flips: %.4f)", p_res["accuracy_original"], p_res["accuracy_masked"], p_res["accuracy_drop"], p_res["flip_rate"])
        logger.info("Diagnóstico Final de Atajos: %s", dual_shortcut_results["diagnostic_summary"])

    # 3. Visualizaciones
    plot_subgroup_disparity_bars(
        subgroup_metrics=subgroup_metrics,
        output_path=output_dir / "fairness_disparity.png",
    )

    mask_lab = np.array(subgroups) == "lab"
    mask_real = np.array(subgroups) == "real"
    if np.any(mask_lab) and np.any(mask_real):
        plot_disaggregated_confusion_matrices(
            y_true_lab=np.array(y_true)[mask_lab],
            y_pred_lab=np.array(y_pred)[mask_lab],
            y_true_real=np.array(y_true)[mask_real],
            y_pred_real=np.array(y_pred)[mask_real],
            class_names=class_names,
            output_path=output_dir / "disaggregated_confusion_matrices.png",
        )

    # 4. Grad-CAM visualizaciones
    if args.run_gradcam:
        _generate_gradcam_panel(
            model=model,
            model_name=args.model,
            test_df=test_df,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            device=device,
            output_path=output_dir / "gradcam_samples.png",
            input_size=input_size,
            factory=factory,
        )

    # 5. Exportar CSV y JSON
    disparity_rows = []
    for grp, m in subgroup_metrics["subgroups"].items():
        row = {
            "subgroup": grp,
            "sample_count": m["sample_count"],
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
        }
        for cls_name, fnr_val in m["class_fnr"].items():
            row[f"fnr_{cls_name}"] = fnr_val
        disparity_rows.append(row)

    df_disparity = pd.DataFrame(disparity_rows)
    df_disparity.to_csv(output_dir / "fairness_disparity.csv", index=False)

    full_report_data = {
        "model_name": args.model,
        "checkpoint_used": str(ckpt_path) if ckpt_path else "pretrained_initial",
        "subgroup_metrics": subgroup_metrics,
        "disparity_analysis": disparity_metrics,
        "shortcut_learning_test": dual_shortcut_results.get("center_occlusion", {}),
        "shortcut_learning_audit": dual_shortcut_results,
    }

    with open(output_dir / "fairness_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2, ensure_ascii=False)

    logger.info("Auditoría de Equidad finalizada exitosamente. Artefactos exportados a: %s", output_dir)


if __name__ == "__main__":
    main()
