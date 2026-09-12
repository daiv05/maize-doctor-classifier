"""Módulo de optimización de hiperparámetros con Optuna (Criterio 1 - Rúbrica Etapa 2).

Proporciona la clase `TuningObjective` para búsqueda bayesiana estructurada (TPE),
poda temprana con MedianPruner y exportación de artefactos de análisis comparativo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.config import set_global_seed
from src.data.dataset import CornDataset
from src.data.transforms import CornTransformFactory
from src.models import build_model, resolve_input_size
from src.training.common import select_device, worker_init_fn
from src.training.loop import run_epoch
from src.training.losses import build_criterion
from src.training.optim import EarlyStopping, build_scheduler

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Rangos y opciones del espacio de búsqueda bayesiana."""

    lr_min: float = 1e-5
    lr_max: float = 1e-3
    weight_decay_min: float = 1e-5
    weight_decay_max: float = 1e-2
    batch_sizes: list[int] = field(default_factory=lambda: [16, 32, 64])
    class_weights_options: list[str] = field(
        default_factory=lambda: ["sqrt_inverse", "inverse", "none"]
    )
    label_smoothing_min: float = 0.0
    label_smoothing_max: float = 0.15
    warmup_epochs_min: int = 1
    warmup_epochs_max: int = 5
    allow_clahe: bool = True


class TuningObjective:
    """Función objetivo invocable por Optuna para entrenar y evaluar un trial."""

    def __init__(
        self,
        model_name: str,
        splits_dir: Path,
        config_path: Path,
        epochs: int = 30,
        patience: int = 6,
        device: torch.device | None = None,
        num_workers: int = 2,
        no_pretrained: bool = False,
        space: HyperparameterSpace | None = None,
    ) -> None:
        if optuna is None:
            raise ImportError("Optuna no está instalado. Instálalo con: pip install optuna")
        self.model_name = model_name
        self.splits_dir = Path(splits_dir)
        self.config_path = Path(config_path)
        self.epochs = epochs
        self.patience = patience
        self.device = device or select_device()
        self.num_workers = num_workers
        self.no_pretrained = no_pretrained
        self.space = space or HyperparameterSpace()

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.seed = self.cfg["dataset"]["seed"]
        self.base_target_size = tuple(self.cfg["dataset"]["target_size"])

    def sample_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Muestrea hiperparámetros desde el espacio de búsqueda usando TPE."""
        params: dict[str, Any] = {
            "learning_rate": trial.suggest_float(
                "learning_rate", self.space.lr_min, self.space.lr_max, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                self.space.weight_decay_min,
                self.space.weight_decay_max,
                log=True,
            ),
            "batch_size": trial.suggest_categorical("batch_size", self.space.batch_sizes),
            "class_weights": trial.suggest_categorical(
                "class_weights", self.space.class_weights_options
            ),
            "label_smoothing": trial.suggest_float(
                "label_smoothing",
                self.space.label_smoothing_min,
                self.space.label_smoothing_max,
            ),
            "warmup_epochs": trial.suggest_int(
                "warmup_epochs", self.space.warmup_epochs_min, self.space.warmup_epochs_max
            ),
        }
        if self.space.allow_clahe:
            params["clahe"] = trial.suggest_categorical("clahe", [True, False])
        else:
            params["clahe"] = False
        return params

    def __call__(self, trial: optuna.Trial) -> float:
        """Ejecuta un trial completo con poda temprana y early stopping."""
        set_global_seed(self.seed + trial.number)
        params = self.sample_parameters(trial)

        target_size = resolve_input_size(self.model_name, self.base_target_size)
        factory = CornTransformFactory(
            config_path=str(self.config_path),
            target_size=target_size,
            clahe=params["clahe"],
        )

        train_dataset = CornDataset(
            csv_path=str(self.splits_dir / "train.csv"),
            config_path=str(self.config_path),
            transform=factory.get_pipeline("train"),
            minority_transform=factory.get_pipeline("minority"),
        )
        class_to_idx = train_dataset.class_to_idx

        val_dataset = CornDataset(
            csv_path=str(self.splits_dir / "val.csv"),
            config_path=str(self.config_path),
            transform=factory.get_pipeline("val"),
            class_to_idx=class_to_idx,
        )

        pin_memory = self.device.type == "cuda"
        batch_size = int(params["batch_size"])
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        model = build_model(
            self.model_name,
            num_classes=len(class_to_idx),
            pretrained=not self.no_pretrained,
        ).to(self.device)

        criterion = build_criterion(
            labels=train_dataset.data_frame["label"].tolist(),
            class_to_idx=class_to_idx,
            strategy=params["class_weights"],
            label_smoothing=params["label_smoothing"],
            device=self.device,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

        scheduler = build_scheduler(
            optimizer,
            kind="cosine",
            total_epochs=self.epochs,
            warmup_epochs=params["warmup_epochs"],
            min_lr=1e-6,
        )

        early_stopping = EarlyStopping(patience=self.patience)
        best_val_f1 = 0.0

        for epoch in range(self.epochs):
            run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=self.device,
                optimizer=optimizer,
                desc=f"[Trial {trial.number} Ep {epoch + 1}/{self.epochs}] Train",
                clip_grad_norm=1.0,
            )

            val_metrics, _, _, _ = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=self.device,
                optimizer=None,
                desc=f"[Trial {trial.number} Ep {epoch + 1}/{self.epochs}] Val",
            )

            current_f1 = val_metrics["macro_f1"]
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1

            if scheduler is not None:
                scheduler.step()

            # Reportar valor a Optuna para poda temprana
            trial.report(current_f1, epoch)

            if trial.should_prune():
                logger.info(
                    "[Trial %d] Podado en época %d con Macro F1: %.4f",
                    trial.number,
                    epoch + 1,
                    current_f1,
                )
                raise optuna.TrialPruned()

            if early_stopping.step(current_f1):
                logger.info(
                    "[Trial %d] Early stopping alcanzado en época %d.", trial.number, epoch + 1
                )
                break

        return best_val_f1


def save_optimization_plots(study: optuna.Study, output_dir: Path) -> list[Path]:
    """Genera y guarda gráficos visuales de convergencia e importancia de hiperparámetros."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: list[Path] = []

    # 1. Gráfica de convergencia (Historial de optimización)
    trials = [
        t
        for t in study.trials
        if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE
    ]
    if trials:
        fig, ax = plt.subplots(figsize=(9, 5))
        trial_numbers = [t.number for t in trials]
        trial_values = [t.value for t in trials]
        running_best = np.maximum.accumulate(trial_values)

        ax.scatter(
            trial_numbers,
            trial_values,
            color="#3498db",
            alpha=0.7,
            label="Trial Val Macro F1",
            zorder=3,
        )
        ax.plot(
            trial_numbers,
            running_best,
            color="#e74c3c",
            linewidth=2.2,
            label="Mejor Macro F1 Acumulado",
            zorder=4,
        )
        ax.set_title(
            f"Historial de Optimización — {study.study_name}", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Número de Trial", fontsize=11)
        ax.set_ylabel("Macro F1 (Validación)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")
        fig.tight_layout()

        hist_path = output_dir / "optimization_history.png"
        fig.savefig(hist_path, dpi=160)
        plt.close(fig)
        saved_plots.append(hist_path)

    # 2. Gráfica de Importancia de Parámetros
    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            params = list(importances.keys())
            scores = list(importances.values())

            y_pos = np.arange(len(params))
            ax.barh(y_pos, scores, color="#2ecc71", edgecolor="#27ae60")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.invert_yaxis()
            ax.set_xlabel("Importancia Relativa", fontsize=11)
            ax.set_title(
                "Importancia de Hiperparámetros (f(x) = Macro F1)", fontsize=13, fontweight="bold"
            )
            ax.grid(True, linestyle="--", alpha=0.5, axis="x")
            fig.tight_layout()

            imp_path = output_dir / "param_importances.png"
            fig.savefig(imp_path, dpi=160)
            plt.close(fig)
            saved_plots.append(imp_path)
    except Exception as err:
        logger.warning("No se pudo calcular la importancia de parámetros: %s", err)

    return saved_plots


def export_tuning_artifacts(
    study: optuna.Study,
    output_dir: Path,
    model_name: str,
    baseline_macro_f1: float = 0.9146,
) -> dict[str, Any]:
    """Genera best_params.json, trials.csv y comparison_vs_baseline.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exportar DataFrame de trials
    df_trials = study.trials_dataframe()
    trials_csv = output_dir / "trials.csv"
    df_trials.to_csv(trials_csv, index=False)

    best_trial = study.best_trial
    best_f1 = float(best_trial.value) if best_trial.value is not None else 0.0
    improvement_delta = best_f1 - baseline_macro_f1
    improvement_pct = (
        (improvement_delta / baseline_macro_f1) * 100.0 if baseline_macro_f1 > 0 else 0.0
    )

    # 2. Resumen JSON
    best_summary = {
        "study_name": study.study_name,
        "model_name": model_name,
        "best_trial_number": best_trial.number,
        "best_val_macro_f1": round(best_f1, 4),
        "baseline_macro_f1": round(baseline_macro_f1, 4),
        "improvement_delta": round(improvement_delta, 4),
        "improvement_pct": round(improvement_pct, 2),
        "schema": "pytorch_hpo_v1",
        "best_params": best_trial.params,
        "total_trials": len(study.trials),
        "pruned_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "complete_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
    }
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    # 3. Comparativa formal vs baseline en CSV
    comparison_rows = [
        {
            "Configuración": "Baseline (Valores por Defecto)",
            "Modelo": model_name,
            "Learning Rate": "1e-4",
            "Weight Decay": "1e-4",
            "Batch Size": 32,
            "Class Weights": "sqrt_inverse",
            "Label Smoothing": 0.1,
            "CLAHE": "No",
            "Val Macro F1": round(baseline_macro_f1, 4),
            "Mejora vs Baseline": "0.0 pp",
        },
        {
            "Configuración": f"Optimizada con Optuna (Trial #{best_trial.number})",
            "Modelo": model_name,
            "Learning Rate": f"{best_trial.params.get('learning_rate', 1e-4):.2e}",
            "Weight Decay": f"{best_trial.params.get('weight_decay', 1e-4):.2e}",
            "Batch Size": best_trial.params.get("batch_size", 32),
            "Class Weights": best_trial.params.get("class_weights", "sqrt_inverse"),
            "Label Smoothing": round(best_trial.params.get("label_smoothing", 0.1), 4),
            "CLAHE": "Sí" if best_trial.params.get("clahe", False) else "No",
            "Val Macro F1": round(best_f1, 4),
            "Mejora vs Baseline": f"{improvement_delta:+.4f} ({improvement_pct:+.2f}%)",
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison_vs_baseline.csv", index=False)

    # 4. Generar gráficos
    save_optimization_plots(study, output_dir)

    return best_summary
