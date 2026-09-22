"""Valida el detector OOD (Relative Mahalanobis Distance) calibrado en compute_ood_stats.py.

Corre el detector sobre (a) una muestra de imagenes de test legitimas -esperado:
por debajo del umbral- y (b) imagenes fuera de dominio -esperado: por encima del
umbral- (fotos negras/ruido/color solido, sustitutas de fotos de escritorio/objetos
random). Reporta la tasa de falsos positivos sobre datos legitimos y si las OOD
sinteticas quedan claramente separadas. Metodologia: docs/es/deep-learning/ood-detection.md.

Uso: python scripts/checks/validate_ood_detector.py --model shufflenet_v2_x1_0
     --checkpoint <run_dir>/best.pth --ood-stats <run_dir>/export/ood_stats.json
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from scripts.pipeline.compute_ood_stats import _apply_pca, _l2_normalize, _mahalanobis_to_mean
from src.config import PROJECT_ROOT
from src.data.identity import unpack_batch
from src.data.transforms import CornTransformFactory
from src.export.common import load_checkpoint_for_export, resolve_export_inputs
from src.models.feature_exposed import FeatureExposedModel
from src.training.common import select_device
from src.training.runs import read_run_contract


def _load_ood_stats(path: Path) -> dict:
    data = json.loads(path.read_text())
    num_classes = data["num_classes"]
    feature_dim = data["feature_dim"]
    pca_dim = data["pca_dim"]
    means = np.frombuffer(base64.b64decode(data["mean_per_class_b64"]), dtype=np.float32).reshape(
        num_classes, pca_dim
    )
    inv_covariance = np.frombuffer(
        base64.b64decode(data["inv_covariance_b64"]), dtype=np.float32
    ).reshape(pca_dim, pca_dim)
    background_mean = np.frombuffer(base64.b64decode(data["background_mean_b64"]), dtype=np.float32)
    background_inv_covariance = np.frombuffer(
        base64.b64decode(data["background_inv_covariance_b64"]), dtype=np.float32
    ).reshape(pca_dim, pca_dim)
    pca_mean = np.frombuffer(base64.b64decode(data["pca_mean_b64"]), dtype=np.float32)
    pca_components = np.frombuffer(
        base64.b64decode(data["pca_components_b64"]), dtype=np.float32
    ).reshape(pca_dim, feature_dim)
    return {
        "means": means,
        "inv_covariance": inv_covariance,
        "background_mean": background_mean,
        "background_inv_covariance": background_inv_covariance,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "threshold": data["threshold"],
        "labels": data["labels"],
    }


def _mahalanobis_scores(feature: np.ndarray, stats: dict) -> tuple[float, float, float]:
    """Componentes del score de una sola muestra sobre la feature L2-normalizada y
    proyectada por PCA: distancia a la clase mas cercana (MD "plana"), distancia a la
    gaussiana de fondo, y su resta (RMD). Ver `scripts/pipeline/compute_ood_stats.py`
    para el mismo calculo vectorizado.

    @returns {tuple[float,float,float]} `(class_distance, background_distance, rmd)`.
    """
    normalized = _l2_normalize(feature[None, :])[0]
    reduced = _apply_pca(normalized[None, :], stats["pca_mean"], stats["pca_components"])[0]
    diffs = reduced[None, :] - stats["means"]
    class_distance = float(np.einsum("ij,jk,ik->i", diffs, stats["inv_covariance"], diffs).min())
    background_distance = float(
        _mahalanobis_to_mean(
            reduced[None, :], stats["background_mean"], stats["background_inv_covariance"]
        )[0]
    )
    return class_distance, background_distance, class_distance - background_distance


def _synthetic_ood_images(image_size: tuple[int, int], transform) -> dict[str, torch.Tensor]:
    """Genera tensores sinteticos fuera de dominio (mismo preprocess que una foto real)."""
    h, w = image_size
    from PIL import Image

    rng = np.random.default_rng(42)
    samples = {
        "black": np.zeros((h, w, 3), dtype=np.uint8),
        "white": np.full((h, w, 3), 255, dtype=np.uint8),
        "gray_solid": np.full((h, w, 3), 128, dtype=np.uint8),
        "random_noise": rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8),
        "blue_solid": np.tile(np.array([30, 60, 200], dtype=np.uint8), (h, w, 1)),
    }
    return {name: transform(Image.fromarray(arr)) for name, arr in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ood-stats", required=True, dest="ood_stats")
    parser.add_argument("--run", default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    parser.add_argument(
        "--n-legit-samples",
        type=int,
        default=30,
        help="Cuantas imagenes legitimas de test evaluar (requiere DATASET_ROOT accesible).",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con test.csv (default: el 'splits_dir' de summary.json).",
    )
    parser.add_argument(
        "--extra-images",
        nargs="+",
        default=None,
        dest="extra_images",
        metavar="PATH[:label]",
        help="Rutas a imagenes reales (no sinteticas) a evaluar una por una, con "
        "corregimiento EXIF igual que el pipeline de entrenamiento. Cada una puede "
        "llevar ':label' (ej. 'foto.jpg:legit' o 'captura.png:ood') solo para el "
        "reporte; no afecta el calculo.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    config_path = Path(args.config)

    class_to_idx, _, image_size = resolve_export_inputs(run_dir, args.model, config_path)
    run_contract = read_run_contract(run_dir)
    factory = CornTransformFactory.from_contract(
        run_contract["preprocessing"], config_path=str(config_path)
    )
    test_csv = None
    if args.n_legit_samples > 0:
        from src.export.data import resolve_test_csv

        test_csv = resolve_test_csv(run_dir, args.splits_dir)
    device = select_device()
    base_model = load_checkpoint_for_export(
        checkpoint_path,
        args.model,
        class_to_idx,
        device,
        image_size=image_size,
        splits_dir=test_csv.parent if test_csv is not None else None,
    )
    model = FeatureExposedModel(base_model, args.model).to(device)
    model.eval()

    stats = _load_ood_stats(Path(args.ood_stats))
    print(f"threshold={stats['threshold']:.4f}\n")

    print("=== Imagenes sinteticas fuera de dominio (esperado: distancia > threshold) ===")
    synthetic = _synthetic_ood_images(image_size, factory.get_pipeline("test"))
    ood_flagged = 0
    for name, tensor in synthetic.items():
        with torch.no_grad():
            _, features = model(tensor.unsqueeze(0).to(device))
        class_distance, _, rmd = _mahalanobis_scores(features.cpu().numpy()[0], stats)
        flagged = rmd > stats["threshold"]
        ood_flagged += int(flagged)
        status = "OOD (correcto)" if flagged else "NO detectado (FALLO)"
        print(f"  {name:15s} MD_plana={class_distance:10.2f}  RMD={rmd:10.2f}  {status}")

    print(f"\nOOD sinteticas detectadas: {ood_flagged}/{len(synthetic)}")

    if args.n_legit_samples > 0:
        print("\n=== Muestra de imagenes legitimas de test (esperado: distancia <= threshold) ===")
        from src.export.data import build_test_loader

        loader, _ = build_test_loader(
            test_csv,
            config_path,
            class_to_idx,
            image_size,
            batch_size=1,
            preprocessing_contract=run_contract["preprocessing"],
        )
        evaluated = 0
        false_positives = 0
        for batch in loader:
            images, labels, _ = unpack_batch(batch)
            if evaluated >= args.n_legit_samples:
                break
            with torch.no_grad():
                _, features = model(images.to(device))
            _, _, rmd = _mahalanobis_scores(features.cpu().numpy()[0], stats)
            flagged = rmd > stats["threshold"]
            false_positives += int(flagged)
            evaluated += 1

        if evaluated == 0:
            print("  No se pudo leer ninguna imagen legitima (DATASET_ROOT inaccesible?).")
        else:
            fp_rate = false_positives / evaluated
            print(f"  Evaluadas: {evaluated}, falsos positivos: {false_positives} ({fp_rate:.1%})")

    if args.extra_images:
        from src.data.loader import load_and_normalize_image

        print("\n=== Imagenes reales sueltas (--extra-images) ===")
        transform = factory.get_pipeline("test")
        for spec in args.extra_images:
            image_path, _, label = spec.partition(":")
            image = load_and_normalize_image(image_path)
            tensor = transform(image)
            with torch.no_grad():
                logits, features = model(tensor.unsqueeze(0).to(device))
            predicted = stats["labels"][int(logits.argmax(dim=1).item())]
            _, _, rmd = _mahalanobis_scores(features.cpu().numpy()[0], stats)
            flagged = rmd > stats["threshold"]
            tag = f" [{label}]" if label else ""
            print(
                f"  {Path(image_path).name:40s}{tag:12s} predicho={predicted:26s} "
                f"RMD={rmd:10.2f}  {'OOD' if flagged else 'id'}"
            )


if __name__ == "__main__":
    main()
