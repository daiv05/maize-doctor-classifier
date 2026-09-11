"""Script CLI para optimización de hiperparámetros con Optuna (Criterio 1 - Rúbrica Etapa 2).

Uso:
    python scripts/pipeline/tune.py --models efficientnet_b0 --n-trials 20 --epochs 30
    python scripts/pipeline/tune.py --models shufflenet_v2_x1_0 --n-trials 15 --epochs 25
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import TPESampler

from src.config import PROJECT_ROOT, get_output_root, set_global_seed
from src.models import list_models
from src.models.registry import MODEL_REGISTRY
from src.training.common import resolve_model_names, select_device
from src.training.tuning import TuningObjective, export_tuning_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tune")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimización Bayesiana de Hiperparámetros con Optuna (Etapa 2)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["efficientnet_b0"],
        help=f"Modelos a optimizar. Disponibles: {list_models()}",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        dest="n_trials",
        help="Número total de trials a ejecutar por modelo.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Techo máximo de épocas por trial (la poda o early stopping pueden terminar antes).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Paciencia para early stopping dentro de cada trial.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Tiempo máximo en segundos para el estudio (opcional).",
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
        help="Destino de base SQLite y artefactos de tuning (default: <outputs>/tuning)",
    )
    parser.add_argument(
        "--study-name-prefix",
        default="tune",
        dest="study_name_prefix",
        help="Prefijo para el nombre del estudio en Optuna.",
    )
    parser.add_argument(
        "--pruner",
        choices=["median", "hyperband", "none"],
        default="median",
        help="Estrategia de poda temprana de trials malos.",
    )
    parser.add_argument(
        "--baseline-f1",
        type=float,
        default=0.9146,
        dest="baseline_f1",
        help="Macro F1 del baseline de referencia para calcular delta de mejora.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        dest="num_workers",
        help="Hilos de DataLoader por trial.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        dest="no_pretrained",
        help="Entrenar desde cero sin pesos de ImageNet (no recomendado).",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "dataset.yaml"),
        help="Ruta al archivo dataset.yaml.",
    )
    return parser.parse_args()


def _build_pruner(kind: str) -> optuna.pruners.BasePruner:
    if kind == "median":
        return MedianPruner(n_startup_trials=3, n_warmup_steps=3, interval_steps=1)
    if kind == "hyperband":
        return HyperbandPruner(min_resource=3, max_resource=30, reduction_factor=3)
    return NopPruner()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    output_root = get_output_root()
    splits_dir = Path(args.splits_dir) if args.splits_dir else output_root / "splits" / "seed_42"
    base_output_dir = Path(args.output_dir) if args.output_dir else output_root / "tuning"

    if not splits_dir.exists():
        logger.error(
            "El directorio de splits no existe: %s\nGenera los splits primero con: make splits",
            splits_dir,
        )
        sys.exit(1)

    base_output_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_output_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"

    model_names = resolve_model_names(args.models, MODEL_REGISTRY)
    device = select_device()
    logger.info("=== OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA (ETAPA 2) ===")
    logger.info("Modelos a optimizar: %s", model_names)
    logger.info("Trials por modelo: %d | Épocas máx: %d | Pruner: %s", args.n_trials, args.epochs, args.pruner)
    logger.info("Base de datos SQLite: %s", storage_url)
    logger.info("Dispositivo de cómputo: %s", device)

    for model_name in model_names:
        study_name = f"{args.study_name_prefix}_{model_name}"
        model_out_dir = base_output_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        sampler = TPESampler(seed=42, multivariate=True)
        pruner = _build_pruner(args.pruner)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        objective = TuningObjective(
            model_name=model_name,
            splits_dir=splits_dir,
            config_path=config_path,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
            num_workers=args.num_workers,
            no_pretrained=args.no_pretrained,
        )

        def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
                try:
                    export_tuning_artifacts(
                        study=study,
                        output_dir=model_out_dir,
                        model_name=model_name,
                        baseline_macro_f1=args.baseline_f1,
                    )
                    logger.info("[Trial #%d Finalizado] Artefactos intermedios actualizados en: %s", trial.number, model_out_dir)
                except Exception as e:
                    logger.warning("No se pudo actualizar artefactos en trial #%d: %s", trial.number, e)

        logger.info("[%s] Iniciando optimización con %d trials...", model_name, args.n_trials)
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
            callbacks=[_trial_callback],
        )

        summary = export_tuning_artifacts(
            study=study,
            output_dir=model_out_dir,
            model_name=model_name,
            baseline_macro_f1=args.baseline_f1,
        )

        logger.info("=== RESULTADOS DE OPTIMIZACIÓN [%s] ===", model_name)
        logger.info("Mejor Trial: #%d", summary["best_trial_number"])
        logger.info("Mejor Val Macro F1: %.4f (Baseline: %.4f)", summary["best_val_macro_f1"], summary["baseline_macro_f1"])
        logger.info("Mejora: %+.4f (%+.2f%%)", summary["improvement_delta"], summary["improvement_pct"])
        logger.info("Hiperparámetros Óptimos: %s", summary["best_params"])
        logger.info("Artefactos guardados en: %s", model_out_dir)


if __name__ == "__main__":
    main()
