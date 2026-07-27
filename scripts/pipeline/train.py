"""Pipeline principal de entrenamiento.

Comparte toda la infraestructura de datos y modelos con train_baselines.py; lo que
cambia es cuanto se afina el loop. Sobre el dataset completo el desbalance llega a
32.9x, asi que el balanceo se hace con perdida ponderada y el WeightedRandomSampler
queda desactivado: combinarlos sobre-compensaria el mismo desbalance por dos vias, y
sin `replacement=True` cada epoca ve el 100% de las imagenes unicas en vez del ~63%.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, get_output_root, set_global_seed
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.models import build_model, list_models, resolve_input_size
from src.models.registry import MODEL_REGISTRY
from src.training.artifacts import (
    NPK_GROUPS,
    write_extended_metrics,
    write_predictions_csv,
    write_summary,
    write_test_outputs,
)
from src.training.common import (
    build_run_dir,
    generate_run_id,
    resolve_model_names,
    select_device,
    update_latest_pointer,
    worker_init_fn,
)
from src.training.loop import fit, run_epoch
from src.training.losses import build_criterion
from src.training.optim import EarlyStopping, build_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """
    Define la interfaz de linea de comandos del pipeline principal.

    @returns {argparse.Namespace} Argumentos ya parseados.
    """
    parser = argparse.ArgumentParser(description="Entrena el pipeline principal.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["shufflenet_v2_x1_0"],
        help=f"Modelos a entrenar, o 'all' para todos. Disponibles: {list_models()}",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train/val/test.csv (default: <outputs>/splits/seed_42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Destino de checkpoints y metricas (default: <outputs>/main)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Techo de epocas a entrenar.")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="learning_rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=3, dest="warmup_epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6, dest="min_lr")
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Epocas sin mejora de val_macro_f1 antes de la parada temprana.",
    )
    parser.add_argument(
        "--class-weights",
        choices=["sqrt_inverse", "inverse", "none"],
        default="sqrt_inverse",
        dest="class_weights",
        help="Estrategia de ponderacion de la perdida (sustituye al sampler balanceado).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1, dest="label_smoothing")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, dest="clip_grad_norm")
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Aplica CLAHE como preprocesamiento en los cuatro pipelines.",
    )
    parser.add_argument("--no-pretrained", action="store_true", dest="no_pretrained")
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    return parser.parse_args()


def main() -> None:
    """Entrena cada modelo solicitado y persiste sus artefactos de run."""
    args = _parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs debe ser mayor o igual a 1.")

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["dataset"]["seed"]
    set_global_seed(seed)

    model_names = resolve_model_names(args.models, MODEL_REGISTRY)
    output_root = get_output_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else output_root / "splits" / "seed_42"
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "main"

    if not splits_dir.exists():
        raise SystemExit(
            f"El directorio de splits no existe: {splits_dir}\n"
            "Genera los splits primero con: make splits"
        )

    device = select_device()
    base_target_size = tuple(cfg["dataset"]["target_size"])
    logger.info("Modelos a entrenar: %s", model_names)
    logger.info("Balanceo: perdida '%s' (sampler desactivado)", args.class_weights)
    if args.clahe:
        logger.info("CLAHE activo en los cuatro pipelines de transformacion")

    for model_name in model_names:
        target_size = resolve_input_size(model_name, base_target_size)
        factory = CornTransformFactory(
            config_path=str(config_path), target_size=target_size, clahe=args.clahe
        )

        train_dataset = CornDataset(
            csv_path=str(splits_dir / "train.csv"),
            config_path=str(config_path),
            transform=factory.get_pipeline("train"),
            minority_transform=factory.get_pipeline("minority"),
        )
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = train_dataset.idx_to_class
        val_dataset = CornDataset(
            csv_path=str(splits_dir / "val.csv"),
            config_path=str(config_path),
            transform=factory.get_pipeline("val"),
            class_to_idx=class_to_idx,
        )
        test_dataset = CornDataset(
            csv_path=str(splits_dir / "test.csv"),
            config_path=str(config_path),
            transform=factory.get_pipeline("test"),
            class_to_idx=class_to_idx,
        )

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        run_id = generate_run_id()
        run_dir = build_run_dir(output_dir, model_name, run_id)
        logger.info("[%s] Checkpoints en %s", model_name, run_dir)

        model = build_model(
            model_name, num_classes=len(class_to_idx), pretrained=not args.no_pretrained
        ).to(device)
        criterion = build_criterion(
            labels=train_dataset.data_frame["label"].tolist(),
            class_to_idx=class_to_idx,
            strategy=args.class_weights,
            label_smoothing=args.label_smoothing,
            device=device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = build_scheduler(
            optimizer,
            kind=args.scheduler,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )
        early_stopping = EarlyStopping(patience=args.patience)

        partial_history: list[dict] = []

        def _persist_history(
            _epoch: int, row: dict, accumulated: list[dict] = partial_history
        ) -> None:
            """
            Vuelca el historial acumulado tras cada epoca para no perderlo ante un corte.

            @param {int} _epoch Epoca recien terminada.
            @param {dict} row Fila del historial de esa epoca.
            @param {list[dict]} accumulated Historial acumulado hasta el momento.
            """
            accumulated.append(row)
            pd.DataFrame(accumulated).to_csv(run_dir / "train_history.csv", index=False)

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            model_name=model_name,
            run_dir=run_dir,
            scheduler=scheduler,
            early_stopping=early_stopping,
            clip_grad_norm=args.clip_grad_norm,
            on_epoch_end=_persist_history,
        )
        pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

        best_row = max(history, key=lambda item: item["val_macro_f1"])
        best_path = run_dir / "best.pth"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))

        test_metrics, labels, predictions, probs = run_epoch(
            model, test_loader, criterion, device, desc=f"{model_name} test"
        )
        write_test_outputs(run_dir, idx_to_class, labels, predictions)
        predictions_df = write_predictions_csv(
            run_dir, test_dataset, idx_to_class, predictions, probs
        )
        write_extended_metrics(run_dir, predictions_df, class_to_idx, NPK_GROUPS)
        write_summary(
            run_dir,
            {
                "pipeline": "main",
                "model": model_name,
                "run_id": run_id,
                "num_classes": len(class_to_idx),
                "class_to_idx": class_to_idx,
                "image_size": list(target_size),
                "splits_dir": str(splits_dir),
                "epochs_requested": args.epochs,
                "epochs_run": len(history),
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "warmup_epochs": args.warmup_epochs,
                "min_lr": args.min_lr,
                "patience": args.patience,
                "class_weights": args.class_weights,
                "label_smoothing": args.label_smoothing,
                "clip_grad_norm": args.clip_grad_norm,
                "clahe": args.clahe,
                "sampler": None,
                "pretrained": not args.no_pretrained,
                "best_epoch": best_row["epoch"],
                "best_val_macro_f1": best_row["val_macro_f1"],
                "test": test_metrics,
            },
        )
        update_latest_pointer(output_dir, model_name, run_id)
        logger.info("[%s] Test macro_f1=%.4f", model_name, test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
