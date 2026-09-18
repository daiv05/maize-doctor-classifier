"""Diagnostico puntual: distribucion de scores RMD de val para un modelo, para
entender donde vive el umbral calibrado respecto a los datos legitimos.
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
from src.export.common import load_checkpoint_for_export, resolve_export_inputs
from src.export.data import build_test_loader, resolve_split_csv
from src.models.feature_exposed import FeatureExposedModel
from src.training.common import select_device


def _load_ood_stats(path: Path) -> dict:
    data = json.loads(path.read_text())
    num_classes = data["num_classes"]
    feature_dim = data["feature_dim"]
    pca_dim = data["pca_dim"]
    means = np.frombuffer(
        base64.b64decode(data["mean_per_class_b64"]), dtype=np.float32
    ).reshape(num_classes, pca_dim)
    inv_covariance = np.frombuffer(
        base64.b64decode(data["inv_covariance_b64"]), dtype=np.float32
    ).reshape(pca_dim, pca_dim)
    background_mean = np.frombuffer(
        base64.b64decode(data["background_mean_b64"]), dtype=np.float32
    )
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
    }


def _relative_mahalanobis_distance(feature: np.ndarray, stats: dict) -> float:
    normalized = _l2_normalize(feature[None, :])[0]
    reduced = _apply_pca(normalized[None, :], stats["pca_mean"], stats["pca_components"])[0]
    diffs = reduced[None, :] - stats["means"]
    class_distance = float(
        np.einsum("ij,jk,ik->i", diffs, stats["inv_covariance"], diffs).min()
    )
    background_distance = _mahalanobis_to_mean(
        reduced[None, :], stats["background_mean"], stats["background_inv_covariance"]
    )[0]
    return class_distance - float(background_distance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ood-stats", required=True, dest="ood_stats")
    parser.add_argument("--splits-dir", required=True, dest="splits_dir")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    parser.add_argument("--n-samples", type=int, default=200, dest="n_samples")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    config_path = Path(args.config)

    class_to_idx, _, image_size = resolve_export_inputs(run_dir, args.model, config_path)
    device = select_device()
    base_model = load_checkpoint_for_export(checkpoint_path, args.model, class_to_idx, device)
    model = FeatureExposedModel(base_model, args.model).to(device)
    model.eval()

    stats = _load_ood_stats(Path(args.ood_stats))
    print(f"pca_dim={stats['means'].shape[1]}  threshold={stats['threshold']:.2f}")

    val_csv = resolve_split_csv(run_dir, args.splits_dir, "val")
    loader, _ = build_test_loader(val_csv, config_path, class_to_idx, image_size, batch_size=1)

    distances = []
    for batch in loader:
        images, _, _ = unpack_batch(batch)
        if len(distances) >= args.n_samples:
            break
        with torch.no_grad():
            _, features = model(images.to(device))
        distances.append(_relative_mahalanobis_distance(features.cpu().numpy()[0], stats))

    distances = np.array(distances)
    print(f"\nDistribucion de distancias en val (n={len(distances)}):")
    for p in [1, 5, 25, 50, 75, 90, 95, 99, 99.5, 100]:
        print(f"  p{p:5.1f}: {np.percentile(distances, p):12.2f}")
    print(f"  min={distances.min():.2f}  max={distances.max():.2f}  mean={distances.mean():.2f}  std={distances.std():.2f}")


if __name__ == "__main__":
    main()
