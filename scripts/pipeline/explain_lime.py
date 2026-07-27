import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.fastvit  # noqa: F401 - registra modelos
import src.models.baselines.ghostnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
import src.models.baselines.shufflenet  # noqa: F401 - registra modelos
from src.config import PROJECT_ROOT, get_dataset_root, get_output_root, set_global_seed
from src.data.loader import load_and_normalize_image
from src.models.registry import MODEL_REGISTRY
from src.training.common import load_run_metadata, resolve_run_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = get_output_root() / "baselines"


def _resolve_model_names(requested: list[str]) -> list[str]:
    available = MODEL_REGISTRY.list_names()
    if requested == ["all"]:
        return available
    unknown = [n for n in requested if n not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {available}")
    return requested


def _load_visual_report_functions():
    try:
        from src.explainability.visual_report import explain_model_visual, render_visual_explanation
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Falta la dependencia opcional '{e.name}' para generar LIME. "
            "Instala el extra xai con: pip install -e .[xai]"
        ) from e
    return explain_model_visual, render_visual_explanation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera explicaciones LIME para checkpoints de baselines ya entrenados: "
        "por defecto, un reporte visual de 3 paneles por cada imagen de una muestra "
        "balanceada del test set (--images-per-class en config/dataset.yaml -> lime:); "
        "con --image, genera un único reporte puntual para esa imagen en vez del muestreo."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help='Nombres de modelos a explicar, o "all" para todos. '
        f"Disponibles: {MODEL_REGISTRY.list_names()}",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Ruta a una imagen puntual a explicar en vez del muestreo balanceado del "
        "test set. Se genera un reporte por cada modelo de --models.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta del PNG de salida. Solo válido junto con --image y un único modelo "
        "en --models. Default: <run_dir>/lime_visual/<stem-de-la-imagen>.png",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=None,
        help="Fuerza el uso de splits/seed_42_baseline en vez de leer lime.baseline de "
        "config/dataset.yaml.",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="run_id específico a explicar (p.ej. 20260703_142230). Por defecto usa el "
        "último run registrado en latest.json para cada modelo.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio raíz donde buscar los runs, con un subdirectorio por modelo. "
        f"Default: {_DEFAULT_OUTPUT_DIR}. Usa outputs/main para los runs de train.py.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR

    with open(PROJECT_ROOT / "config" / "dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    lime_cfg = cfg["lime"]
    gradcam_enabled = cfg.get("gradcam", {}).get("enabled", False)
    set_global_seed(lime_cfg["seed"])

    model_names = _resolve_model_names(args.models)
    use_baseline = args.baseline if args.baseline is not None else lime_cfg["baseline"]
    fallback_splits_dir = get_output_root() / "splits" / (
        "seed_42_baseline" if use_baseline else "seed_42"
    )
    fallback_classes = cfg["dataset"]["classes"]
    fallback_target_size = tuple(cfg["dataset"]["target_size"])

    if args.output is not None and (args.image is None or len(model_names) != 1):
        raise SystemExit(
            "--output solo es valido junto con --image y un unico modelo en --models."
        )

    if not fallback_splits_dir.exists():
        raise SystemExit(
            f"El directorio de splits no existe: {fallback_splits_dir}\n"
            "Genera los splits primero con: make splits  (o make splits-baseline)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Modelos a explicar: {model_names}")
    explain_model_visual, render_visual_explanation = _load_visual_report_functions()

    for model_name in model_names:
        try:
            run_dir = resolve_run_dir(output_dir, model_name, args.run)
        except SystemExit as e:
            logger.warning(f"[{model_name}] {e}. Se omite.")
            continue
        checkpoint_path = run_dir / "best.pth"
        if not checkpoint_path.exists():
            logger.warning(
                f"[{model_name}] Run {run_dir.name} sin checkpoint completo, se omite."
            )
            continue

        splits_dir, _, idx_to_class, target_size = load_run_metadata(
            run_dir=run_dir,
            fallback_splits_dir=fallback_splits_dir,
            fallback_classes=fallback_classes,
            fallback_target_size=fallback_target_size,
        )
        num_classes = len(idx_to_class)

        model = MODEL_REGISTRY.build(
            model_name,
            num_classes=num_classes,
            pretrained=False,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        if args.image is not None:
            image_path = Path(args.image)
            output_path = (
                Path(args.output)
                if args.output is not None
                else run_dir / "lime_visual" / f"{image_path.stem}.png"
            )
            result = render_visual_explanation(
                image=load_and_normalize_image(image_path),
                model=model,
                idx_to_class=idx_to_class,
                target_size=target_size,
                output_path=output_path,
                num_samples=lime_cfg["num_samples"],
                num_features=lime_cfg["num_features"],
                seed=lime_cfg["seed"],
                device=device,
                model_name=model_name if gradcam_enabled else None,
            )
            logger.info(
                f"[{model_name}] Diagnóstico: {result['predicted_label']} "
                f"(confianza: {result['predicted_prob'] * 100:.1f}%)"
            )
        else:
            test_df = pd.read_csv(splits_dir / "test.csv")
            explain_model_visual(
                model=model,
                model_name=model_name,
                test_df=test_df,
                dataset_root=get_dataset_root(),
                idx_to_class=idx_to_class,
                target_size=target_size,
                output_dir=run_dir,
                images_per_class=lime_cfg["images_per_class"],
                num_features=lime_cfg["num_features"],
                num_samples=lime_cfg["num_samples"],
                seed=lime_cfg["seed"],
                device=device,
                enable_gradcam=gradcam_enabled,
            )

    logger.info("Explicaciones LIME completadas.")


if __name__ == "__main__":
    main()
