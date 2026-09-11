"""Test de fuga de procedencia: cuánto del rendimiento no proviene del contenido diagnóstico.

Entrena el mismo modelo sobre variantes de la imagen que aíslan o eliminan el borde del
encuadre. Si el brazo que sólo conserva el anillo exterior iguala al de la imagen completa,
la separación entre clases no proviene de la lesión sino de artefactos de captura.

Uso:
    python scripts/experiments/provenance_leak.py --arms original,border_ring,center_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from src.config import get_dataset_root, get_output_root

ARM_CHOICES = ("original", "border_ring", "center_only")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos del experimento."""
    parser = argparse.ArgumentParser(description="Test de fuga de procedencia por recorte.")
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--arms", type=str, default=",".join(ARM_CHOICES))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--model", type=str, default="efficientnet_lite0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-cap", type=int, default=1000)
    parser.add_argument("--ring-fraction", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_splits(splits_dir: Path, train_cap: int) -> pd.DataFrame:
    """Reúne los tres splits en un manifiesto único, capando sólo train por clase.

    @param {Path} splits_dir Directorio con train.csv, val.csv y test.csv.
    @param {int} train_cap Máximo de imágenes de entrenamiento por clase; 0 desactiva el cap.
    @returns {pd.DataFrame} Manifiesto con columnas image_path, label, environment y split.
    """
    frames = []
    for name in ("train", "val", "test"):
        frame = pd.read_csv(splits_dir / f"{name}.csv")
        frame["split"] = name
        if name == "train" and train_cap > 0:
            frame = pd.concat(
                [group.sample(min(len(group), train_cap), random_state=42)
                 for _, group in frame.groupby("label")],
                ignore_index=True,
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def apply_arm(array: np.ndarray, arm: str, ring_fraction: float) -> np.ndarray:
    """Recorta la imagen según el brazo del experimento.

    @param {np.ndarray} array Imagen HWC en uint8.
    @param {str} arm Uno de original, border_ring o center_only.
    @param {float} ring_fraction Grosor del anillo como fracción del lado menor.
    @returns {np.ndarray} Imagen transformada.
    """
    if arm == "original":
        return array
    thickness = max(1, round(ring_fraction * array.shape[0]))
    output = array.copy()
    if arm == "border_ring":
        output[thickness:-thickness, thickness:-thickness] = 0
        return output
    if arm == "center_only":
        output[:thickness] = 0
        output[-thickness:] = 0
        output[:, :thickness] = 0
        output[:, -thickness:] = 0
        return output
    raise ValueError(f"brazo desconocido: {arm!r}")


class LeakDataset(Dataset):
    """Sirve imágenes del manifiesto aplicando el recorte del brazo."""

    def __init__(self, frame, dataset_root, class_to_idx, arm, ring_fraction, size, train, seed,
                 backmix_probability=0.0, backmix_black_level=12, backmix_min_fraction=0.30,
                 augment=None):
        self.paths = frame["image_path"].tolist()
        self.labels = [class_to_idx[label] for label in frame["label"]]
        self.dataset_root = dataset_root
        self.arm = arm
        self.ring_fraction = ring_fraction
        self.size = size
        self.train = train
        self.seed = seed
        self.backmix_probability = backmix_probability
        self.backmix_black_level = backmix_black_level
        self.backmix_min_fraction = backmix_min_fraction
        self.augment = augment
        self._rng = None

    def __len__(self) -> int:
        return len(self.paths)

    def _worker_rng(self) -> np.random.Generator:
        """Devuelve un generador propio del worker, cuyo estado avanza entre épocas.

        Sembrar por índice de imagen daría la misma transformación en todas las épocas, que
        es una asignación aleatoria fija y no una augmentation.

        @returns {np.random.Generator} Generador persistente del proceso trabajador.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker_id = info.id if info is not None else 0
            self._rng = np.random.default_rng([self.seed, worker_id])
        return self._rng

    def load_raw(self, index: int) -> np.ndarray:
        """Decodifica y reescala una imagen sin aplicar brazo ni normalización.

        @param {int} index Posición en el manifiesto.
        @returns {np.ndarray} Imagen HWC en uint8 al tamaño de entrada.
        """
        with Image.open(self.dataset_root / self.paths[index]) as handle:
            if handle.format == "JPEG":
                handle.draft("RGB", (self.size * 2, self.size * 2))
            image = handle.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def _backmix(self, array: np.ndarray, rng) -> np.ndarray:
        """Sustituye el fondo negro pre-enmascarado por el de otra imagen del conjunto.

        Sólo actúa sobre imágenes que ya vienen recortadas sobre negro, que son las únicas
        cuya máscara es recuperable sin segmentar.

        @param {np.ndarray} array Imagen HWC en uint8.
        @returns {np.ndarray} Imagen con el fondo sustituido, o la original si no aplica.
        """
        black = array.max(axis=2) <= self.backmix_black_level
        if black.mean() < self.backmix_min_fraction:
            return array
        donor = self.load_raw(int(rng.integers(len(self.paths))))
        return np.where(black[:, :, None], donor, array).astype(np.uint8)

    def __getitem__(self, index: int):
        array = self.load_raw(index)
        if self.train:
            rng = self._worker_rng()
            if self.backmix_probability and rng.random() < self.backmix_probability:
                array = self._backmix(array, rng)
            if self.augment is not None:
                array = self.augment(array, rng)
            if rng.random() < 0.5:
                array = np.ascontiguousarray(array[:, ::-1])
        array = apply_arm(array, self.arm, self.ring_fraction)
        tensor = (array.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(tensor.transpose(2, 0, 1))), self.labels[index]


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve etiquetas verdaderas y predichas para un loader completo."""
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for images, targets in loader:
            preds.append(model(images.to(device, non_blocking=True)).argmax(1).cpu().numpy())
            trues.append(targets.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def run_arm(arm: str, seed: int, manifest, dataset_root, classes, args, device) -> dict:
    """Entrena y evalúa un brazo con una semilla.

    @param {str} arm Brazo del experimento.
    @param {int} seed Semilla de inicialización y barajado.
    @param {pd.DataFrame} manifest Manifiesto con los tres splits.
    @returns {dict} Métricas de test globales, por clase y por ambiente.
    """
    class_to_idx = {name: index for index, name in enumerate(classes)}
    torch.manual_seed(seed)
    np.random.seed(seed)

    loaders = {}
    for split in ("train", "val", "test"):
        subset = manifest[manifest.split == split]
        loaders[split] = DataLoader(
            LeakDataset(subset, dataset_root, class_to_idx, arm, args.ring_fraction,
                        args.image_size, split == "train", seed),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes)).to(device)
    train_labels = np.asarray([class_to_idx[label]
                               for label in manifest[manifest.split == "train"].label])
    counts = np.bincount(train_labels, minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1.0)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for images, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(images.to(device, non_blocking=True)),
                      targets.to(device, non_blocking=True)).backward()
            optimizer.step()
        scheduler.step()
        trues, preds = predict(model, loaders["val"], device)
        val_f1 = float(f1_score(trues, preds, average="macro"))
        print(f"    [{arm} s{seed}] epoca {epoch + 1} val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, stale, best_state = val_f1, 0, {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    trues, preds = predict(model, loaders["test"], device)
    environments = manifest[manifest.split == "test"].environment.to_numpy()
    return {
        "arm": arm,
        "seed": seed,
        "val_macro_f1": best_f1,
        "test_macro_f1": float(f1_score(trues, preds, average="macro")),
        "test_accuracy": float((trues == preds).mean()),
        "per_class": dict(zip(classes,
                              f1_score(trues, preds, average=None,
                                       labels=range(len(classes))).tolist())),
        "by_environment": {
            str(env): float(f1_score(trues[environments == env], preds[environments == env],
                                     average="macro"))
            for env in sorted(set(environments.tolist()))
        },
    }


def main() -> None:
    """Ejecuta todos los brazos y semillas y persiste los resultados de forma incremental."""
    args = parse_args()
    splits_dir = args.splits_dir or (get_output_root() / "splits" / "seed_42")
    output = args.output or (get_output_root() / "experiments" / "provenance_leak.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_splits(splits_dir, args.train_cap)
    classes = sorted(manifest.label.unique())
    dataset_root = get_dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] splits={splits_dir} clases={len(classes)} device={device}", flush=True)
    for split in ("train", "val", "test"):
        print(f"    {split}: {(manifest.split == split).sum()}", flush=True)

    results = []
    for arm in args.arms.split(","):
        for seed in (int(value) for value in args.seeds.split(",")):
            result = run_arm(arm, seed, manifest, dataset_root, classes, args, device)
            results.append(result)
            print(f"[{arm} s{seed}] TEST macro_f1={result['test_macro_f1']:.4f} "
                  f"accuracy={result['test_accuracy']:.4f}", flush=True)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[*] resultados en {output}", flush=True)


if __name__ == "__main__":
    main()
