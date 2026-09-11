"""Validación por fuente sobre el pipeline principal, sin réplicas del mismo.

Los experimentos anteriores usaban un arnés propio y mínimo para aislar el efecto de la
partición. Eso servía para comparar condiciones entre sí, pero sus cifras no son las del
sistema: les faltaban la corrección EXIF, el augmentation real, el warmup, el recorte de
gradiente y el pipeline agresivo de clases minoritarias.

Aquí cada pliegue entrena con los mismos componentes que ``scripts/pipeline/train.py``
—``CornDataset``, ``CornTransformFactory``, ``build_model``, ``build_criterion``,
``build_scheduler`` y ``EarlyStopping``— y lo único que cambia entre brazos es la
transformación de entrenamiento. Cada brazo es, por tanto, una modificación candidata de
``src/data/transforms.py`` medida directamente.

La evaluación se repite sobre el mismo conjunto retenido con la imagen completa y con sólo
el anillo exterior, de modo que cada corrida entrega rendimiento y dependencia del marco.

Uso:
    python scripts/experiments/pipeline_aligned_loso.py --arm baseline
    python scripts/experiments/pipeline_aligned_loso.py --arm colour_strong
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from scripts.experiments.leave_one_source_out import build_folds, deduplicate
from src.config import PROJECT_ROOT, get_dataset_root, get_output_root, set_global_seed
from src.data.dataset import CornDataset
from src.data.provenance import provenance_from_path
from src.data.transforms import CornTransformFactory
from src.models import build_model
from src.training.common import select_device, worker_init_fn
from src.training.losses import build_criterion
from src.training.optim import EarlyStopping, build_scheduler

CONFIG_PATH = str(PROJECT_ROOT / "config" / "dataset.yaml")
EVAL_ARMS = ("original", "border_ring")


class BlackOutCentre:
    """Deja sólo el anillo exterior de la imagen, con el centro en negro.

    Opera sobre la imagen PIL antes de ``ToTensor`` para que el negro entre por el mismo
    camino de normalización que cualquier píxel oscuro del corpus.
    """

    def __init__(self, ring_fraction: float = 0.10) -> None:
        self.ring_fraction = ring_fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        thickness = max(1, round(self.ring_fraction * min(array.shape[:2])))
        array[thickness:-thickness, thickness:-thickness] = 0
        return Image.fromarray(array)


def parse_args() -> argparse.Namespace:
    """Define la interfaz de línea de comandos del experimento."""
    parser = argparse.ArgumentParser(description="Validación por fuente alineada al pipeline.")
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--arm", type=str, default="baseline")
    parser.add_argument("--split-mode", choices=("random", "source"), default="source")
    parser.add_argument("--model", type=str, default="efficientnet_lite0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--class-weights", type=str, default="sqrt_inverse")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--train-cap", type=int, default=1500)
    parser.add_argument("--val-cap", type=int, default=0)
    parser.add_argument("--ring-fraction", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=str, default="")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def replace_transform(pipeline: T.Compose, target_type: type, replacement) -> T.Compose:
    """Sustituye la primera transformación de un tipo dado, conservando el resto del pipeline.

    Los brazos son aditivos sobre las transformaciones reales del proyecto: reescribirlas
    desde cero mezclaría el cambio que se quiere medir con la pérdida de todo lo demás, que
    es exactamente el error que invalidó la primera versión de estos brazos.

    @param {T.Compose} pipeline Pipeline de partida.
    @param {type} target_type Tipo de la transformación a sustituir.
    @param {object} replacement Transformación que ocupa su lugar.
    @returns {T.Compose} Pipeline con la sustitución aplicada.
    """
    steps = list(pipeline.transforms)
    for index, step in enumerate(steps):
        if isinstance(step, target_type):
            steps[index] = replacement
            return T.Compose(steps)
    raise ValueError(f"el pipeline no contiene ninguna transformación {target_type.__name__}")


def build_arm_transforms(arm: str, factory: CornTransformFactory):
    """Devuelve las transformaciones de entrenamiento y de minoritarias del brazo.

    ``baseline`` reproduce exactamente lo que hace hoy el pipeline principal. El resto son
    modificaciones candidatas, expresadas como cambios sobre ese punto de partida.

    @param {str} arm Nombre del brazo.
    @param {CornTransformFactory} factory Fábrica configurada con target_size y CLAHE.
    @returns {tuple} Par (transformación estándar, transformación de minoritarias).
    """
    size = factory.target_size
    standard = factory.get_pipeline("train")
    minority = factory.get_pipeline("minority")

    if arm == "baseline":
        return standard, minority

    if arm == "colour_strong":
        jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        return (replace_transform(standard, T.ColorJitter, jitter),
                replace_transform(minority, T.ColorJitter, jitter))

    if arm == "crop_all":
        crop = T.RandomResizedCrop(size, scale=(0.3, 1.0))
        return (replace_transform(standard, T.Resize, crop),
                replace_transform(minority, T.RandomResizedCrop, crop))

    if arm == "minority_for_all":
        return minority, minority

    raise ValueError(f"brazo desconocido: {arm!r}")


def write_fold_csvs(train, val, test, directory: Path) -> dict[str, Path]:
    """Persiste los tres subconjuntos del pliegue, que es lo que CornDataset consume."""
    paths = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve etiquetas verdaderas y predichas del loader completo."""
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for images, targets in loader:
            preds.append(model(images.to(device, non_blocking=True)).argmax(1).cpu().numpy())
            trues.append(targets.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def run_fold(fold, manifest, args, factory, device, workdir: Path) -> dict[str, object]:
    """Entrena un pliegue con los componentes del pipeline principal y evalúa ambos brazos."""
    if args.split_mode == "random":
        train = manifest[manifest.split == "train"]
        val = manifest[manifest.split == "val"]
        test = manifest[manifest.split == "test"]
    else:
        test = manifest[manifest.provenance == fold["test_group"]]
        val = manifest[manifest.provenance == fold["val_group"]]
        train = manifest[~manifest.provenance.isin({fold["test_group"], fold["val_group"]})]
    if args.val_cap > 0 and len(val) > args.val_cap:
        # Estratificado por clase: un muestreo plano puede dejar sin representar a las
        # clases escasas de la fuente de validacion y hacer ruidoso el early stopping.
        share = args.val_cap / len(val)
        val = pd.concat(
            [group.sample(max(1, round(len(group) * share)), random_state=args.seed)
             for _, group in val.groupby("label")],
            ignore_index=True,
        )

    fold_dir = workdir / fold["test_group"]
    fold_dir.mkdir(parents=True, exist_ok=True)
    csvs = write_fold_csvs(train, val, test, fold_dir)

    standard, minority = build_arm_transforms(args.arm, factory)
    train_dataset = CornDataset(
        csv_path=str(csvs["train"]), config_path=CONFIG_PATH,
        transform=standard, minority_transform=minority,
        max_per_class=args.train_cap or None, seed=args.seed,
    )
    class_to_idx = train_dataset.class_to_idx
    val_dataset = CornDataset(
        csv_path=str(csvs["val"]), config_path=CONFIG_PATH,
        transform=factory.get_pipeline("val"), class_to_idx=class_to_idx,
    )

    loader_kwargs = dict(num_workers=args.num_workers, worker_init_fn=worker_init_fn,
                         pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)

    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=True).to(device)
    criterion = build_criterion(
        labels=train_dataset.data_frame["label"].tolist(), class_to_idx=class_to_idx,
        strategy=args.class_weights, label_smoothing=args.label_smoothing, device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, kind="cosine", total_epochs=args.epochs,
                                warmup_epochs=args.warmup_epochs, min_lr=args.min_lr)
    stopper = EarlyStopping(patience=args.patience)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device, non_blocking=True)),
                             targets.to(device, non_blocking=True))
            loss.backward()
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        trues, preds = evaluate(model, val_loader, device)
        val_f1 = float(f1_score(trues, preds, average="macro"))
        print(f"    [{fold['test_group']}] epoca {epoch} val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if stopper.step(val_f1):
            break

    model.load_state_dict(best_state)
    result = {"arm": args.arm, "test_group": fold["test_group"], "val_group": fold["val_group"],
              "n_train": int(len(train_dataset)), "n_val": int(len(val)),
              "n_test": int(len(test)),
              "minority_classes": sorted(train_dataset.minority_classes),
              "val_macro_f1": best_f1, "arms": {}}
    for eval_arm in EVAL_ARMS:
        pipeline = factory.get_pipeline("val")
        if eval_arm == "border_ring":
            steps = list(pipeline.transforms)
            steps.insert(1, BlackOutCentre(args.ring_fraction))
            pipeline = T.Compose(steps)
        dataset = CornDataset(csv_path=str(csvs["test"]), config_path=CONFIG_PATH,
                              transform=pipeline, class_to_idx=class_to_idx)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        trues, preds = evaluate(model, loader, device)
        result["arms"][eval_arm] = {"accuracy": float((trues == preds).mean()),
                                    "true": trues.tolist(), "pred": preds.tolist()}
        print(f"    [{fold['test_group']}] {eval_arm}: "
              f"acierto {result['arms'][eval_arm]['accuracy']:.4f}", flush=True)
    return result


def main() -> None:
    """Ejecuta los pliegues y agrupa las predicciones de cada brazo de evaluación."""
    args = parse_args()
    set_global_seed(args.seed)
    splits_dir = args.splits_dir or (get_output_root() / "splits" / "seed_42")
    output = args.output or (
        get_output_root() / "experiments" / f"aligned_{args.arm}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = {name: pd.read_csv(splits_dir / f"{name}.csv") for name in ("train", "val", "test")}
    manifest = pd.concat(frames.values(), ignore_index=True)
    manifest["provenance"] = manifest.image_path.map(provenance_from_path)

    if args.split_mode == "source":
        # Agrupar por fuente obliga a deduplicar antes: sin eso la misma foto puede caer a
        # ambos lados de la frontera aunque las fuentes esten separadas.
        manifest, dropped = deduplicate(manifest, get_dataset_root(),
                                        max(args.num_workers, 1))
        folds = build_folds(manifest)
    else:
        # Reproduce la particion vigente tal cual, sin agrupar ni deduplicar, para que su
        # cifra sea comparable con las corridas que el proyecto ya tiene.
        dropped = 0
        splits_of = {name: set(frame.image_path) for name, frame in frames.items()}
        manifest["split"] = manifest.image_path.map(
            lambda path: next(n for n, s in splits_of.items() if path in s))
        folds = [{"test_group": "seed_42", "val_group": "seed_42",
                  "n_test": int(len(frames["test"])),
                  "test_classes": sorted(frames["test"].label.unique())}]

    classes = sorted(manifest.label.unique())
    if args.folds:
        wanted = set(args.folds.split(","))
        folds = [f for f in folds if f["test_group"] in wanted]

    factory = CornTransformFactory(config_path=CONFIG_PATH, clahe=args.clahe)
    device = select_device()
    print(f"[*] brazo={args.arm} pliegues={len(folds)} clases={len(classes)} device={device}",
          flush=True)

    results = []
    with tempfile.TemporaryDirectory(prefix="loso_folds_") as tmp:
        workdir = Path(tmp)
        for fold in folds:
            results.append(run_fold(fold, manifest, args, factory, device, workdir))
            output.write_text(json.dumps(
                {"arm": args.arm, "classes": classes, "duplicates_dropped": dropped,
                 "folds": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    pooled = {}
    for eval_arm in EVAL_ARMS:
        trues = [t for r in results for t in r["arms"][eval_arm]["true"]]
        preds = [p for r in results for p in r["arms"][eval_arm]["pred"]]
        pooled[eval_arm] = {
            "n": len(trues),
            "macro_f1": float(f1_score(trues, preds, average="macro")),
            "accuracy": float(np.mean(np.asarray(trues) == np.asarray(preds))),
            "per_class_f1": dict(zip(classes, f1_score(
                trues, preds, average=None, labels=range(len(classes))).tolist())),
        }
    pooled["frame_dependence"] = pooled["border_ring"]["macro_f1"] / pooled["original"]["macro_f1"]
    print(f"[*] macro-F1 {pooled['original']['macro_f1']:.4f} | "
          f"marco {pooled['border_ring']['macro_f1']:.4f} | "
          f"dependencia {pooled['frame_dependence'] * 100:.1f}%", flush=True)
    output.write_text(json.dumps(
        {"arm": args.arm, "classes": classes, "duplicates_dropped": dropped,
         "pooled": pooled, "folds": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[*] resultados en {output}", flush=True)


if __name__ == "__main__":
    main()
