"""Inferencia en lote sobre un directorio de imágenes con un checkpoint entrenado.

Uso: python scripts/pipeline/batch_predict.py --model efficientnet_b0 \
     --checkpoint <ruta/best.pth> --images-dir <ruta> --output-dir <ruta>
"""

import argparse
import csv
import logging
from pathlib import Path

import torch

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.fastvit  # noqa: F401 - registra modelos
import src.models.baselines.ghostnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
import src.models.baselines.shufflenet  # noqa: F401 - registra modelos
from src.config import PROJECT_ROOT, get_output_root
from src.data.loader import load_and_normalize_image
from src.models.registry import MODEL_REGISTRY
from src.training.common import load_run_metadata, select_device
from src.training.runs import load_validated_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(images_dir: Path):
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help=f"Nombre del modelo registrado. Disponibles: {MODEL_REGISTRY.list_names()}",
    )
    parser.add_argument("--checkpoint", required=True, help="Ruta directa al .pth.")
    parser.add_argument("--images-dir", required=True, dest="images_dir")
    parser.add_argument("--output-dir", required=True, dest="output_dir")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Modelo desconocido: '{args.model}'. Disponibles: {MODEL_REGISTRY.list_names()}"
        )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"No existe el checkpoint: {checkpoint_path}")

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"No existe el directorio de imágenes: {images_dir}")

    image_paths = list(_iter_images(images_dir))
    if not image_paths:
        raise SystemExit(f"No se encontraron imágenes en {images_dir}")

    config_path = PROJECT_ROOT / "config" / "dataset.yaml"
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_dir = checkpoint_path.parent
    fallback_splits_dir = get_output_root() / "splits" / "seed_42"
    _, class_to_idx, idx_to_class, target_size = load_run_metadata(
        run_dir=run_dir,
        fallback_splits_dir=fallback_splits_dir,
        fallback_classes=cfg["dataset"]["classes"],
        fallback_target_size=tuple(cfg["dataset"]["target_size"]),
    )

    device = select_device()
    loaded = load_validated_run(
        checkpoint_path,
        expected_model=args.model,
        expected_input_size=target_size,
        expected_class_to_idx=class_to_idx,
        device=device,
        config_path=str(config_path),
    )
    model = loaded.model
    pipeline = loaded.factory.get_pipeline("inference")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "predictions.csv"

    top_k = min(args.top_k, len(idx_to_class))
    fieldnames = (
        ["image", "predicted_label", "predicted_prob"]
        + [f"top{i + 1}_label" for i in range(top_k)]
        + [f"top{i + 1}_prob" for i in range(top_k)]
    )

    logger.info(f"Modelo: {args.model} | Checkpoint: {checkpoint_path}")
    logger.info(f"Imágenes a procesar: {len(image_paths)}")

    with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, image_path in enumerate(image_paths, start=1):
            try:
                image = load_and_normalize_image(str(image_path))
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning(f"Saltando {image_path.name}: {e}")
                continue

            tensor = pipeline(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

            values, indices = torch.topk(probabilities, k=top_k)
            row = {
                "image": str(image_path.relative_to(images_dir)),
                "predicted_label": idx_to_class[int(indices[0])],
                "predicted_prob": float(values[0]),
            }
            for rank, (prob, idx) in enumerate(zip(values.tolist(), indices.tolist(), strict=True)):
                row[f"top{rank + 1}_label"] = idx_to_class[int(idx)]
                row[f"top{rank + 1}_prob"] = prob
            writer.writerow(row)

            if i % 10 == 0 or i == len(image_paths):
                logger.info(f"  {i}/{len(image_paths)} procesadas")

    logger.info(f"Predicciones guardadas en {predictions_csv}")


if __name__ == "__main__":
    main()
