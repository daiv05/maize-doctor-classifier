import argparse
import logging
from pathlib import Path

import torch

from src.config import PROJECT_ROOT, get_output_root
from src.data.loader import load_and_normalize_image
from src.data.transforms import CornTransformFactory
from src.models.ensemble import SoftVotingEnsemble
from src.training.common import select_device
from src.training.runs import (
    load_ensemble_manifest,
    load_validated_run,
    resolve_checkpoint,
    validate_ensemble_runs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_ENSEMBLE_MODELS = ["efficientnet_b0", "shufflenet_v2_x1_0", "efficientnet_lite0"]


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


def _find_model_checkpoint(
    model_name: str,
    output_root: Path,
    explicit_checkpoint: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Auto-descubre el mejor checkpoint en outputs/main/ y outputs/baselines/.

    Si *run_id* fue proporcionado explícitamente y no se encuentra su checkpoint,
    la función falla de inmediato en lugar de caer a un fallback por ``rglob``.
    """
    return resolve_checkpoint(
        model_name,
        output_root,
        explicit_checkpoint=explicit_checkpoint,
        run_id=run_id,
    )


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

    print(f"\n{'=' * 60}")
    print(f" Imagen: {image_path.name}")
    print(f" Modelo: {model_name.upper()}")
    print(f" Diagnostico: {prediction} ({confidence:.2f}%)")
    print(f"{'-' * 60}")
    print(" Top-k Probabilidades:")
    for prob, idx in zip(values.tolist(), indices.tolist(), strict=True):
        bar_len = int(prob * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"   * {idx_to_class[int(idx)]:<26} [{bar}] {prob * 100:6.2f}%")

    # Si es ensamble, mostrar votos de cada miembro
    if individual_models:
        print("\n   Desglose por Miembro del Ensamble:")
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
    parser.add_argument(
        "--image", required=True, help="Ruta a una imagen o a un directorio con imágenes."
    )
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
    device = select_device()
    requested_splits_dir = (
        Path(args.splits_dir)
        if args.splits_dir
        else (
            _default_splits_dir(output_root, baseline=args.baseline, full=args.full)
            if args.baseline or args.full
            else None
        )
    )

    # 1. Configuración de Modelo (Individual o Ensamble)
    is_ensemble = args.model.lower() == "ensemble"
    individual_models: dict[str, torch.nn.Module] | None = None

    if is_ensemble:
        # Cargar composición desde manifiesto del ensamble evaluado o usar modelos canónicos
        manifest_path = output_root / "ensemble" / "ensemble_summary.json"
        if manifest_path.exists():
            loaded_runs, ensemble_weights, _ = load_ensemble_manifest(
                manifest_path,
                device=device,
                config_path=str(config_path),
                splits_dir=requested_splits_dir,
            )
            canonical = [run.summary["model"] for run in loaded_runs]
            logger.info(
                "Ensamble contractual cargado desde manifiesto: %s (pesos: %s)",
                canonical,
                ensemble_weights,
            )
        else:
            canonical = list(_DEFAULT_ENSEMBLE_MODELS)
            ensemble_weights = None
            loaded_runs = [
                load_validated_run(
                    _find_model_checkpoint(name, output_root),
                    expected_model=name,
                    device=device,
                    config_path=str(config_path),
                    splits_dir=requested_splits_dir,
                )
                for name in canonical
            ]
            validate_ensemble_runs(loaded_runs, policy="normal", shared_input=True)
            logger.info("Sin manifiesto; latest.json resolvió miembros canónicos: %s", canonical)

        class_to_idx = loaded_runs[0].class_to_idx
        idx_to_class = {index: name for name, index in class_to_idx.items()}
        models_list = [run.model for run in loaded_runs]
        individual_models = {name: run.model for name, run in zip(canonical, loaded_runs)}

        model = SoftVotingEnsemble(models_list, weights=ensemble_weights, model_names=canonical)
        factory = loaded_runs[0].factory
        active_model_name = f"Soft Voting Ensemble ({' + '.join(canonical)})"
    else:
        checkpoint_path = _find_model_checkpoint(args.model, output_root, args.checkpoint, args.run)
        expected_size = (args.image_size, args.image_size) if args.image_size is not None else None
        loaded = load_validated_run(
            checkpoint_path,
            expected_model=args.model,
            expected_input_size=expected_size,
            device=device,
            config_path=str(config_path),
            splits_dir=requested_splits_dir,
        )
        class_to_idx = loaded.class_to_idx
        idx_to_class = {index: name for name, index in class_to_idx.items()}
        factory = loaded.factory
        model = loaded.model
        active_model_name = args.model

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
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
