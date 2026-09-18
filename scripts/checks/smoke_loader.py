"""Smoke check del pipeline de carga: CornDataset + transforms + DataLoader.

Verifica de punta a punta que los splits generados se pueden consumir: carga el
manifiesto, extrae una muestra individual y arma un batch real. Cualquier fallo
propaga la excepción (exit code != 0), así que sirve tras regenerar splits o en CI.

Uso: make test-loader  (o: python scripts/checks/smoke_loader.py [--splits-dir ...])
"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import get_output_root
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train.csv (default: <repo>/outputs/splits/seed_42)",
    )
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    args = parser.parse_args()

    splits_dir = (
        Path(args.splits_dir) if args.splits_dir else get_output_root() / "splits" / "seed_42"
    )
    csv_path = splits_dir / "train.csv"
    if not csv_path.exists():
        raise SystemExit(f"No existe {csv_path}. Genera los splits primero con: make splits")

    factory = CornTransformFactory()
    dataset = CornDataset(csv_path=str(csv_path), transform=factory.get_pipeline("train"))
    print(f"Dataset: {len(dataset)} muestras | class_to_idx: {dataset.class_to_idx}")

    tensor, label, sample_id = dataset[0]
    print(
        f"Muestra 0 -> tensor {tuple(tensor.shape)} | label {label} "
        f"({dataset.idx_to_class[label]}) | sample_id {sample_id} | "
        f"rango [{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    batch_size = min(args.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    images, labels, sample_ids = next(iter(loader))
    print(
        f"Batch -> imágenes {tuple(images.shape)} | labels {tuple(labels.shape)} "
        f"| sample_ids {len(sample_ids)}"
    )
    print("Smoke check OK.")


if __name__ == "__main__":
    main()
