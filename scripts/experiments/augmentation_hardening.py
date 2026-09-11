"""Mide si una augmentation agresiva rompe la dependencia del marco de captura.

A diferencia de los experimentos anteriores, aquí el modelo entrena siempre con la imagen
completa y se evalúa dos veces sobre el mismo conjunto de prueba: con la imagen entera y con
sólo el anillo exterior. La pregunta ya no es cuánta información contiene el marco, sino
cuánto sigue dependiendo de él un modelo concreto.

La augmentation endurecida ataca la firma de captura: recorte aleatorio para que el marco no
esté presente en todas las muestras, recompresión JPEG y remuestreo para borrar la huella del
codificador, y color fuerte para borrar la respuesta cromática de cada cámara.

Uso:
    python scripts/experiments/augmentation_hardening.py --augment hardened
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from scripts.experiments.leave_one_source_out import build_folds, deduplicate
from scripts.experiments.provenance_leak import LeakDataset, predict
from src.config import get_dataset_root, get_output_root
from src.data.provenance import provenance_from_path

EVAL_ARMS = ("original", "border_ring")


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos del experimento."""
    parser = argparse.ArgumentParser(description="Endurecimiento por augmentation.")
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--augment",
                        choices=("none", "crop", "codec", "colour", "hardened"),
                        default="none")
    parser.add_argument("--model", type=str, default="efficientnet_lite0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-cap", type=int, default=1000)
    parser.add_argument("--val-cap", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ring-fraction", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-crop-scale", type=float, default=0.30)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def random_resized_crop(array: np.ndarray, rng, min_scale: float) -> np.ndarray:
    """Recorta una región aleatoria y la devuelve al tamaño original.

    Es la transformación que impide que el marco de la imagen esté presente en todas las
    muestras de entrenamiento.

    @param {np.ndarray} array Imagen cuadrada HWC en uint8.
    @param {float} min_scale Fracción mínima del área conservada.
    @returns {np.ndarray} Recorte reescalado al tamaño de entrada.
    """
    side = array.shape[0]
    for _ in range(10):
        scale = rng.uniform(min_scale, 1.0)
        ratio = float(np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3))))
        height = int(round((side * side * scale / ratio) ** 0.5))
        width = int(round(height * ratio))
        if height <= side and width <= side:
            top = int(rng.integers(0, side - height + 1))
            left = int(rng.integers(0, side - width + 1))
            crop = array[top:top + height, left:left + width]
            return np.asarray(
                Image.fromarray(crop).resize((side, side), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
    return array


def recompress(array: np.ndarray, rng) -> np.ndarray:
    """Recodifica la imagen como JPEG con calidad aleatoria.

    Borra la huella del codificador original, que difiere entre fuentes y es parte de lo
    que el modelo puede estar leyendo en el marco.

    @param {np.ndarray} array Imagen HWC en uint8.
    @returns {np.ndarray} Imagen tras un ciclo de compresión con pérdida.
    """
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=int(rng.integers(30, 96)))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def resample(array: np.ndarray, rng, min_factor: float = 0.35) -> np.ndarray:
    """Reduce y vuelve a ampliar la imagen, borrando la resolución nativa de la fuente."""
    side = array.shape[0]
    factor = rng.uniform(min_factor, 1.0)
    small = max(32, int(round(side * factor)))
    image = Image.fromarray(array).resize((small, small), Image.Resampling.BILINEAR)
    return np.asarray(image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.uint8)


def jitter_colour(array: np.ndarray, rng) -> np.ndarray:
    """Altera brillo, contraste, saturación y tono para borrar la respuesta cromática."""
    image = Image.fromarray(array)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        image = enhancer(image).enhance(float(rng.uniform(0.6, 1.4)))
    hsv = np.asarray(image.convert("HSV"), dtype=np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-25, 26))) % 256
    return np.asarray(
        Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"), dtype=np.uint8)


def build_augment(name: str, min_crop_scale: float):
    """Devuelve la función de augmentation correspondiente al nombre, o None.

    Los componentes aislados conservan las mismas probabilidades que tienen dentro de
    ``hardened``, de modo que la ablación descompone exactamente esa combinación.

    @param {str} name Uno de none, crop, codec, colour o hardened.
    @returns {callable|None} Transformación aplicada al array uint8 durante el entrenamiento.
    """
    if name == "none":
        return None

    def apply_crop(array, rng):
        return random_resized_crop(array, rng, min_crop_scale)

    def apply_codec(array, rng):
        if rng.random() < 0.7:
            array = resample(array, rng)
        if rng.random() < 0.7:
            array = recompress(array, rng)
        return array

    def apply_colour(array, rng):
        return jitter_colour(array, rng) if rng.random() < 0.8 else array

    components = {
        "crop": (apply_crop,),
        "codec": (apply_codec,),
        "colour": (apply_colour,),
        "hardened": (apply_crop, apply_codec, apply_colour),
    }[name]

    def augment(array: np.ndarray, rng) -> np.ndarray:
        for component in components:
            array = component(array, rng)
        return array

    return augment


def run_fold(fold, manifest, dataset_root, classes, args, device) -> dict[str, object]:
    """Entrena con la imagen completa y evalúa la fuente retenida con cada brazo."""
    class_to_idx = {name: index for index, name in enumerate(classes)}
    test = manifest[manifest.provenance == fold["test_group"]]
    val = manifest[manifest.provenance == fold["val_group"]]
    if args.val_cap > 0 and len(val) > args.val_cap:
        val = val.sample(args.val_cap, random_state=42)
    train = manifest[~manifest.provenance.isin({fold["test_group"], fold["val_group"]})]
    if args.train_cap > 0:
        train = pd.concat(
            [group.sample(min(len(group), args.train_cap), random_state=42)
             for _, group in train.groupby("label")],
            ignore_index=True,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def make_loader(frame, arm, is_train):
        return DataLoader(
            LeakDataset(frame, dataset_root, class_to_idx, arm, args.ring_fraction,
                        args.image_size, is_train, args.seed,
                        augment=build_augment(args.augment, args.min_crop_scale)
                        if is_train else None),
            batch_size=args.batch_size, shuffle=is_train,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    train_loader = make_loader(train, "original", True)
    val_loader = make_loader(val, "original", False)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes)).to(device)
    counts = np.bincount([class_to_idx[label] for label in train.label],
                         minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1.0)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(images.to(device, non_blocking=True)),
                      targets.to(device, non_blocking=True)).backward()
            optimizer.step()
        scheduler.step()
        trues, preds = predict(model, val_loader, device)
        val_f1 = float(f1_score(trues, preds, average="macro"))
        print(f"    [{fold['test_group']}] epoca {epoch + 1} val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, stale, best_state = val_f1, 0, {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    result = {"test_group": fold["test_group"], "val_group": fold["val_group"],
              "n_train": int(len(train)), "n_test": int(len(test)),
              "val_macro_f1": best_f1, "arms": {}}
    for arm in EVAL_ARMS:
        trues, preds = predict(model, make_loader(test, arm, False), device)
        result["arms"][arm] = {
            "accuracy": float((trues == preds).mean()),
            "true": trues.tolist(),
            "pred": preds.tolist(),
        }
        print(f"    [{fold['test_group']}] {arm}: acierto {result['arms'][arm]['accuracy']:.4f}",
              flush=True)
    return result


def main() -> None:
    """Ejecuta los once pliegues y agrupa las predicciones de cada brazo de evaluación."""
    args = parse_args()
    splits_dir = args.splits_dir or (get_output_root() / "splits" / "seed_42")
    output = args.output or (
        get_output_root() / "experiments" / f"augmentation_{args.augment}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = get_dataset_root()

    manifest = pd.concat(
        [pd.read_csv(splits_dir / f"{name}.csv") for name in ("train", "val", "test")],
        ignore_index=True,
    )
    manifest["provenance"] = manifest.image_path.map(provenance_from_path)
    manifest, dropped = deduplicate(manifest, dataset_root, args.num_workers)
    classes = sorted(manifest.label.unique())
    folds = build_folds(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] augment={args.augment} pliegues={len(folds)} clases={len(classes)}", flush=True)

    results = []
    for fold in folds:
        results.append(run_fold(fold, manifest, dataset_root, classes, args, device))
        output.write_text(json.dumps(
            {"augment": args.augment, "classes": classes, "duplicates_dropped": dropped,
             "folds": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    pooled = {}
    for arm in EVAL_ARMS:
        trues = [t for r in results for t in r["arms"][arm]["true"]]
        preds = [p for r in results for p in r["arms"][arm]["pred"]]
        pooled[arm] = {
            "n": len(trues),
            "macro_f1": float(f1_score(trues, preds, average="macro")),
            "accuracy": float(np.mean(np.asarray(trues) == np.asarray(preds))),
            "per_class_f1": dict(zip(classes, f1_score(
                trues, preds, average=None, labels=range(len(classes))).tolist())),
        }
    pooled["frame_dependence"] = pooled["border_ring"]["macro_f1"] / pooled["original"]["macro_f1"]
    print(json.dumps({k: v for k, v in pooled.items() if k != "per_class_f1"},
                     indent=2, ensure_ascii=False)[:900], flush=True)
    output.write_text(json.dumps(
        {"augment": args.augment, "classes": classes, "duplicates_dropped": dropped,
         "pooled": pooled, "folds": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[*] resultados en {output}", flush=True)


if __name__ == "__main__":
    main()
