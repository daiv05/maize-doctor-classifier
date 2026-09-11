import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from src.config import PROJECT_ROOT, get_output_root
from src.data.dataset import resolve_class_mapping
from src.data.loader import load_and_normalize_image
from src.data.transforms import CornTransformFactory
from src.models import build_model, list_models
from src.models.ensemble import SoftVotingEnsemble
from src.training.common import resolve_run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_SIZE_BY_MODEL = {
    "mobilenet_v3_large": 224,
    "mobilenet_v3_small": 224,
    "efficientnet_b4": 380,
}

_DEFAULT_ENSEMBLE_MODELS = ["efficientnet_b0", "shufflenet_v2_x1_0", "efficientnet_lite0"]


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_splits_dir(output_root: Path, baseline: bool, full: bool) -> Path:
    if baseline and full:
        raise SystemExit("Usa solo uno de --baseline o --full.")
    if baseline:
        return output_root / "splits" / "seed_42_baseline"
    if full:
        return output_root / "splits" / "seed_42"

    main_dir = output_root / "splits" / "seed_42"
    if main_dir.exists():
        return main_dir
    baseline_dir = output_root / "splits" / "seed_42_baseline"
    return baseline_dir if baseline_dir.exists() else main_dir


def _load_summary(checkpoint_path: Path) -> dict[str, Any]:
    summary_path = checkpoint_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_class_mapping(
    summary: dict[str, Any],
    splits_dir: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, int], dict[int, str]]:
    if "class_to_idx" in summary:
        class_to_idx = {str(name): int(idx) for name, idx in summary["class_to_idx"].items()}
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        return class_to_idx, idx_to_class

    train_csv = splits_dir / "train.csv"
    if not train_csv.exists():
        raise SystemExit(
            f"No existe {train_csv}. Pasa --splits-dir o conserva summary.json junto al checkpoint."
        )
    return resolve_class_mapping(str(train_csv), cfg["dataset"]["classes"])


def _resolve_target_size(
    model_name: str,
    explicit_size: int | None,
    summary: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[int, int]:
    if explicit_size is not None:
        return (explicit_size, explicit_size)

    image_size = summary.get("image_size")
    if isinstance(image_size, list) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))

    default_size = _DEFAULT_IMAGE_SIZE_BY_MODEL.get(model_name)
    if default_size is not None:
        return (default_size, default_size)

    height, width = cfg["dataset"]["target_size"]
    return (height, width)


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Checkpoint inválido: {checkpoint_path}")
    return checkpoint


def _find_model_checkpoint(model_name: str, output_root: Path, explicit_checkpoint: str | None = None, run_id: str | None = None) -> Path:
    """Auto-descubre el mejor checkpoint en outputs/main/ y outputs/baselines/.

    Si *run_id* fue proporcionado explícitamente y no se encuentra su checkpoint,
    la función falla de inmediato en lugar de caer a un fallback por ``rglob``.
    """
    if explicit_checkpoint:
        p = Path(explicit_checkpoint)
        if p.exists():
            return p
        raise SystemExit(f"No existe el checkpoint especificado: {explicit_checkpoint}")

    # 1. Buscar en main
    for pipeline_dir in ["main", "baselines"]:
        model_dir = output_root / pipeline_dir / model_name
        if model_dir.exists():
            latest_json = model_dir / "latest.json"
            if latest_json.exists():
                try:
                    with open(latest_json, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    rid = run_id or meta.get("run_id") or meta.get("run")
                    if rid:
                        for name in ["best.pth", "best.pt"]:
                            cp = model_dir / rid / name
                            if cp.exists():
                                return cp
                        # Si el run fue solicitado explícitamente y no tiene checkpoint, fallar
                        if run_id:
                            raise SystemExit(
                                f"No se encontró checkpoint para el run '{run_id}' del modelo "
                                f"'{model_name}' en {model_dir / run_id}."
                            )
                except SystemExit:
                    raise
                except Exception:
                    pass

            # Fallback rglob solo si no se pidió un run específico
            if not run_id:
                pts = list(model_dir.rglob("best.pth")) + list(model_dir.rglob("best.pt"))
                if pts:
                    return sorted(pts, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise SystemExit(f"No se encontró checkpoint para el modelo '{model_name}' en {output_root}")


def _predict_single_image(
    image_path: Path,
    model: torch.nn.Module,
    factory: CornTransformFactory,
    idx_to_class: dict[int, str],
    device: torch.device,
    top_k: int,
    model_name: str,
    individual_models: dict[str, torch.nn.Module] | None = None,
) -> None:
    image = load_and_normalize_image(str(image_path))
    tensor = factory.get_pipeline("inference")(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if isinstance(model, SoftVotingEnsemble):
            probabilities = model.predict_probabilities(tensor).squeeze(0).cpu()
        else:
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    k = min(top_k, len(idx_to_class))
    values, indices = torch.topk(probabilities, k=k)
    prediction = idx_to_class[int(indices[0])]
    confidence = float(values[0]) * 100

    print(f"\n{'='*60}")
    print(f" Imagen: {image_path.name}")
    print(f" Modelo: {model_name.upper()}")
    print(f" Diagnostico: {prediction} ({confidence:.2f}%)")
    print(f"{'-'*60}")
    print(" Top-k Probabilidades:")
    for prob, idx in zip(values.tolist(), indices.tolist(), strict=True):
        bar_len = int(prob * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"   * {idx_to_class[int(idx)]:<26} [{bar}] {prob*100:6.2f}%")

    # Si es ensamble, mostrar votos de cada miembro
    if individual_models:
        print(f"\n   Desglose por Miembro del Ensamble:")
        with torch.no_grad():
            for name, sub_model in individual_models.items():
                sub_logits = sub_model(tensor)
                sub_probs = torch.softmax(sub_logits, dim=1).squeeze(0).cpu()
                sub_top_val, sub_top_idx = torch.topk(sub_probs, k=1)
                sub_pred = idx_to_class[int(sub_top_idx[0])]
                sub_conf = float(sub_top_val[0]) * 100
                print(f"     - {name:<20}: {sub_pred} ({sub_conf:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice la clase de una imagen de hoja de maíz.")
    parser.add_argument(
        "--model",
        default="ensemble",
        help="Modelo a usar: 'ensemble', 'efficientnet_b0', 'shufflenet_v2_x1_0', etc.",
    )
    parser.add_argument("--image", required=True, help="Ruta a una imagen o a un directorio con imágenes.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta explícita al checkpoint .pth (opcional; auto-descubre si se omite).",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="run_id específico a usar. Por defecto usa latest.json.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train.csv para reconstruir class_to_idx.",
    )
    parser.add_argument("--baseline", action="store_true", help="Usa splits/seed_42_baseline.")
    parser.add_argument("--full", action="store_true", help="Usa splits/seed_42.")
    parser.add_argument("--image-size", type=int, default=None, dest="image_size")
    parser.add_argument("--top-k", type=int, default=4, dest="top_k")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
    )
    args = parser.parse_args()

    output_root = get_output_root()
    config_path = Path(args.config)
    cfg = _load_config(config_path)
    device = select_device()
    splits_dir = (
        Path(args.splits_dir)
        if args.splits_dir
        else _default_splits_dir(output_root, baseline=args.baseline, full=args.full)
    )

    # 1. Configuración de Modelo (Individual o Ensamble)
    is_ensemble = (args.model.lower() == "ensemble")
    individual_models: dict[str, torch.nn.Module] | None = None

    if is_ensemble:
        # Cargar composición desde manifiesto del ensamble evaluado o usar modelos canónicos
        manifest_path = output_root / "ensemble" / "ensemble_summary.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            canonical = manifest.get("models_included", _DEFAULT_ENSEMBLE_MODELS)
            ensemble_weights = manifest.get("weights", None)
            logger.info("Ensamble cargado desde manifiesto: %s (pesos: %s)", canonical, ensemble_weights)
        else:
            canonical = list(_DEFAULT_ENSEMBLE_MODELS)
            ensemble_weights = None
            logger.info("Sin manifiesto de ensamble; usando modelos canónicos: %s", canonical)

        models_list = []
        individual_models = {}
        first_summary: dict[str, Any] = {}
        reference_class_to_idx: dict[str, int] | None = None

        for m_name in canonical:
            ckpt = _find_model_checkpoint(m_name, output_root)
            summary = _load_summary(ckpt)
            if not first_summary and summary:
                first_summary = summary
            class_to_idx, idx_to_class = _resolve_class_mapping(summary, splits_dir, cfg)

            # Validar consistencia de class_to_idx entre miembros
            if reference_class_to_idx is None:
                reference_class_to_idx = class_to_idx
            elif class_to_idx != reference_class_to_idx:
                raise SystemExit(
                    f"Inconsistencia en class_to_idx entre miembros del ensamble.\n"
                    f"  Referencia: {reference_class_to_idx}\n"
                    f"  {m_name}: {class_to_idx}"
                )

            m = build_model(m_name, num_classes=len(class_to_idx), pretrained=False).to(device)
            m.load_state_dict(_load_state_dict(ckpt, device))
            m.eval()
            models_list.append(m)
            individual_models[m_name] = m

        model = SoftVotingEnsemble(models_list, weights=ensemble_weights, model_names=canonical)
        target_size = _resolve_target_size(canonical[0], None, first_summary, cfg)
        active_model_name = f"Soft Voting Ensemble ({' + '.join(canonical)})"
    else:
        checkpoint_path = _find_model_checkpoint(args.model, output_root, args.checkpoint, args.run)
        summary = _load_summary(checkpoint_path)
        class_to_idx, idx_to_class = _resolve_class_mapping(summary, splits_dir, cfg)
        target_size = _resolve_target_size(args.model, args.image_size, summary, cfg)

        model = build_model(args.model, num_classes=len(class_to_idx), pretrained=False).to(device)
        model.load_state_dict(_load_state_dict(checkpoint_path, device))
        model.eval()
        active_model_name = args.model

    factory = CornTransformFactory(config_path=str(config_path), target_size=target_size)

    # 2. Recolectar imágenes
    image_input = Path(args.image)
    if not image_input.exists():
        raise SystemExit(f"No existe la ruta de imagen: {image_input}")

    if image_input.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = sorted([f for f in image_input.iterdir() if f.suffix.lower() in valid_exts])
        if not image_files:
            raise SystemExit(f"No se encontraron imágenes válidas en el directorio: {image_input}")
    else:
        image_files = [image_input]

    # 3. Inferencia
    for img_path in image_files:
        _predict_single_image(
            image_path=img_path,
            model=model,
            factory=factory,
            idx_to_class=idx_to_class,
            device=device,
            top_k=args.top_k,
            model_name=active_model_name,
            individual_models=individual_models,
        )
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
