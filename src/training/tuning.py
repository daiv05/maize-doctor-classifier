"""Módulo de optimización de hiperparámetros con Optuna (Criterio 1 - Rúbrica Etapa 2).

Proporciona la clase `TuningObjective` para búsqueda bayesiana estructurada (TPE),
poda temprana con MedianPruner y exportación de artefactos de análisis comparativo.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import platform
import sys
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.config import set_global_seed
from src.data.dataset import CornDataset
from src.data.preparation import atomic_write_json, sha256_file
from src.data.transforms import CornTransformFactory
from src.models import build_model, resolve_input_size
from src.training.artifacts import NPK_GROUPS
from src.training.common import select_device, worker_init_fn
from src.training.evaluation import compute_calibration_metrics, compute_grouped_metrics
from src.training.hyperparameters import HPO_SCHEMA, load_best_params
from src.training.loop import run_epoch
from src.training.losses import build_criterion
from src.training.optim import EarlyStopping, build_scheduler
from src.training.runs import (
    build_run_contract,
    compute_config_sha256,
    validate_run_contract,
    write_run_contract,
)

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SPLIT_FILES = (
    "master_manifest.csv",
    "train.csv",
    "val.csv",
    "test.csv",
    "manifest.lock.json",
)


class SplitIntegrityError(RuntimeError):
    """Cambio de datos: debe abortar el estudio completo, no consumir otros trials."""


def split_hash_snapshot(splits_dir: str | Path) -> dict[str, Any]:
    """Captura y valida los hashes contractuales de un split sin modificarlo."""
    directory = Path(splits_dir)
    missing = [name for name in SPLIT_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos contractuales del split: {missing}")
    hashes = {name: sha256_file(directory / name) for name in SPLIT_FILES}
    lock = json.loads((directory / "manifest.lock.json").read_text(encoding="utf-8"))
    expected = {
        "master_manifest.csv": lock.get("master_manifest_sha256"),
        "train.csv": lock.get("train_sha256"),
        "val.csv": lock.get("val_sha256"),
        "test.csv": lock.get("test_sha256"),
    }
    mismatches = {
        name: {"expected": expected_hash, "actual": hashes[name]}
        for name, expected_hash in expected.items()
        if expected_hash != hashes[name]
    }
    if mismatches:
        raise SplitIntegrityError(f"manifest.lock.json no coincide con sus CSV: {mismatches}")
    counts = {}
    for name in ("train", "val", "test"):
        # Solo inventario de filas; no construye un Dataset ni carga imágenes de test.
        with (directory / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
            counts[name] = sum(1 for _ in csv.reader(handle)) - 1
    return {
        "schema_version": 1,
        "split_identifier": directory.name,
        "seed": lock.get("seed"),
        "sha256": hashes,
        "counts": counts,
    }


def assert_split_snapshot_unchanged(
    splits_dir: str | Path, expected: dict[str, Any]
) -> dict[str, Any]:
    """Falla si cualquier artefacto de splits cambió respecto al preflight."""
    actual = split_hash_snapshot(splits_dir)
    if actual["sha256"] != expected["sha256"]:
        raise SplitIntegrityError(
            "Los artefactos seed_42 cambiaron desde el preflight; se prohíbe continuar."
        )
    return actual


@dataclass
class HyperparameterSpace:
    """Rangos y opciones del espacio de búsqueda bayesiana."""

    lr_min: float = 1e-5
    lr_max: float = 3e-3
    weight_decay_min: float = 1e-7
    weight_decay_max: float = 1e-2
    batch_sizes: list[int] = field(default_factory=lambda: [16, 32, 64])
    class_weights_options: list[str] = field(
        default_factory=lambda: ["sqrt_inverse", "inverse", "none"]
    )
    label_smoothing_values: list[float] = field(
        default_factory=lambda: [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
    )
    warmup_epochs_min: int = 1
    warmup_epochs_max: int = 5
    # Medido sobre el pipeline real: CLAHE baja el transform de 107,5 a 40,6 img/s
    # (9,3 -> 24,7 ms por imagen). En un regimen limitado por CPU eso duplica el
    # coste de cada trial que lo active, para decidir una sola opcion binaria de
    # preprocesado que se contesta mejor con un A/B de dos corridas.
    allow_clahe: bool = False

    def validate(self) -> None:
        """Rechaza rangos inválidos antes de crear un estudio costoso."""
        if not all(math.isfinite(v) for v in (self.lr_min, self.lr_max)) or not (
            0 < self.lr_min < self.lr_max <= 1
        ):
            raise ValueError("El rango de learning_rate debe cumplir 0 < min < max.")
        if not all(
            math.isfinite(v) for v in (self.weight_decay_min, self.weight_decay_max)
        ) or not (0 < self.weight_decay_min < self.weight_decay_max <= 1):
            raise ValueError("El rango logarítmico de weight_decay requiere 0 < min < max <= 1.")
        if not self.batch_sizes or any(
            type(value) is not int or value < 1 for value in self.batch_sizes
        ):
            raise ValueError("batch_sizes debe contener enteros positivos.")
        if not self.label_smoothing_values or any(
            type(value) not in (int, float) or not 0 <= value < 1
            for value in self.label_smoothing_values
        ):
            raise ValueError("label_smoothing_values debe contener valores en [0, 1).")
        if self.warmup_epochs_min < 0 or self.warmup_epochs_min > self.warmup_epochs_max:
            raise ValueError("El rango de warmup_epochs es inválido.")
        if any(type(v) is not int for v in (self.warmup_epochs_min, self.warmup_epochs_max)):
            raise ValueError("warmup_epochs requiere enteros.")
        if not self.class_weights_options or not set(self.class_weights_options) <= {
            "sqrt_inverse",
            "inverse",
            "none",
        }:
            raise ValueError("class_weights_options inválidas.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": {
                "type": "float",
                "low": self.lr_min,
                "high": self.lr_max,
                "log": True,
            },
            "weight_decay": {
                "type": "float",
                "low": self.weight_decay_min,
                "high": self.weight_decay_max,
                "log": True,
            },
            "batch_size": {"type": "categorical", "choices": self.batch_sizes},
            "class_weights": {
                "type": "categorical",
                "choices": self.class_weights_options,
            },
            "label_smoothing": {
                "type": "categorical",
                "choices": self.label_smoothing_values,
            },
            "warmup_epochs": {
                "type": "int",
                "low": self.warmup_epochs_min,
                "high": self.warmup_epochs_max,
            },
            "clahe": {
                "type": "categorical",
                "choices": [False, True] if self.allow_clahe else [False],
                "searched": self.allow_clahe,
            },
            "optimizer": {"fixed": "AdamW"},
            "scheduler": {"fixed": "cosine"},
            "dropout": {
                "fixed": None,
                "reason": "factory default; run loader does not restore custom dropout",
            },
        }


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
        study_dir: Path | None = None,
        expected_split_hashes: dict[str, str] | None = None,
        study_name: str = "",
        on_parameters_sampled: Callable | None = None,
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
        self.space.validate()
        self.study_dir = Path(study_dir) if study_dir is not None else None
        self.expected_split_hashes = expected_split_hashes
        self.study_name = study_name
        self.on_parameters_sampled = on_parameters_sampled
        if epochs < 1 or patience < 1 or num_workers < 0:
            raise ValueError("epochs/patience deben ser positivos y num_workers >= 0.")

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
            "label_smoothing": trial.suggest_categorical(
                "label_smoothing", self.space.label_smoothing_values
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

    def _effective_hyperparameters(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "learning_rate": float(params["learning_rate"]),
            "batch_size": int(params["batch_size"]),
            "weight_decay": float(params["weight_decay"]),
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "epochs": int(self.epochs),
            "patience": int(self.patience),
            "dropout": None,
            "warmup_epochs": int(params["warmup_epochs"]),
            "min_lr": 1e-6,
            "class_weights": params["class_weights"],
            "label_smoothing": float(params["label_smoothing"]),
            "clip_grad_norm": 1.0,
            "sampler": None,
            "pretrained": not self.no_pretrained,
            "clahe": bool(params["clahe"]),
        }

    def _verify_splits_unchanged(self) -> None:
        if self.expected_split_hashes is None:
            return
        actual = split_hash_snapshot(self.splits_dir)
        if actual["sha256"] != self.expected_split_hashes:
            raise SplitIntegrityError(
                "Los hashes de seed_42 cambiaron durante el HPO; ejecución abortada."
            )

    @staticmethod
    def _validation_diagnostics(
        *,
        labels: list[int],
        predictions: list[int],
        probabilities: list[float],
        idx_to_class: dict[int, str],
        class_to_idx: dict[str, int],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        true_names = [idx_to_class[int(value)] for value in labels]
        predicted_names = [idx_to_class[int(value)] for value in predictions]
        frame = pd.DataFrame(
            {
                "label": true_names,
                "pred_label": predicted_names,
                "pred_prob": list(probabilities),
            }
        )
        sample_ids = getattr(labels, "sample_ids", None)
        if sample_ids is not None:
            frame.insert(0, "sample_id", list(sample_ids))
        calibration = compute_calibration_metrics(frame, class_to_idx)
        report = classification_report(
            true_names,
            predicted_names,
            labels=list(class_to_idx),
            output_dict=True,
            zero_division=0,
        )
        return frame, {
            "calibration": calibration,
            "per_class": report,
            "npk_grouped": compute_grouped_metrics(frame, NPK_GROUPS),
        }

    def _finalize_trial_artifacts(
        self,
        *,
        trial: optuna.Trial,
        trial_dir: Path | None,
        status: str,
        params: dict[str, Any],
        history: list[dict[str, Any]],
        best_epoch: int,
        best_val_f1: float,
        best_val_loss: float,
        early_stopped: bool,
        started_at: str,
        duration_seconds: float,
        factory: CornTransformFactory,
        class_to_idx: dict[str, int],
        error: str | None = None,
    ) -> None:
        best_train_f1 = max(
            (float(row["train_macro_f1"]) for row in history),
            default=float("nan"),
        )
        train_at_best = next(
            (float(row["train_macro_f1"]) for row in history if int(row["epoch"]) == best_epoch),
            float("nan"),
        )
        attrs = {
            "status_detail": status,
            "best_epoch": best_epoch,
            "executed_epochs": len(history),
            "early_stopped": early_stopped,
            "best_val_loss": best_val_loss if np.isfinite(best_val_loss) else None,
            "best_train_macro_f1": best_train_f1 if np.isfinite(best_train_f1) else None,
            "train_macro_f1_at_best_val": (train_at_best if np.isfinite(train_at_best) else None),
            "train_val_gap": (
                train_at_best - best_val_f1
                if np.isfinite(train_at_best) and np.isfinite(best_val_f1)
                else None
            ),
            "duration_seconds": duration_seconds,
            "trial_id": f"trial_{trial.number:03d}",
            "run_id": f"trial_{trial.number:03d}",
            "error": error,
        }
        effective = self._effective_hyperparameters(params)
        config_hash = compute_config_sha256(
            model_name=self.model_name,
            input_size=factory.target_size,
            num_classes=len(class_to_idx),
            seed=self.seed,
            hyperparameters=effective,
            preprocessing=factory.to_contract(),
            class_to_idx=class_to_idx,
        )
        attrs["config_sha256"] = config_hash
        for key, value in attrs.items():
            trial.set_user_attr(key, value)

        if trial_dir is None:
            return
        pd.DataFrame(history).to_csv(trial_dir / "training_history.csv", index=False)
        payload = {
            "schema_version": 1,
            "study_name": self.study_name,
            "trial_number": trial.number,
            "trial_id": attrs["trial_id"],
            "state": status,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "duration_seconds": duration_seconds,
            "parameters": params,
            "effective_hyperparameters": effective,
            "training_seed": self.seed,
            "best_epoch": best_epoch,
            "executed_epochs": len(history),
            "early_stopped": early_stopped,
            "best_val_macro_f1": (best_val_f1 if np.isfinite(best_val_f1) else None),
            "best_val_loss": (best_val_loss if np.isfinite(best_val_loss) else None),
            "best_train_macro_f1": attrs["best_train_macro_f1"],
            "train_macro_f1_at_best_val": attrs["train_macro_f1_at_best_val"],
            "train_val_gap": attrs["train_val_gap"],
            "config_sha256": config_hash,
            "split_hashes": self.expected_split_hashes,
            "checkpoint": "best.pth" if (trial_dir / "best.pth").is_file() else None,
            "checkpoint_sha256": (
                sha256_file(trial_dir / "best.pth") if (trial_dir / "best.pth").is_file() else None
            ),
            "error": error,
        }
        atomic_write_json(trial_dir / "trial_summary.json", payload)
        if (trial_dir / "best.pth").is_file() and best_epoch > 0:
            contract = build_run_contract(
                run_dir=trial_dir,
                model_name=self.model_name,
                seed=self.seed,
                hyperparameters=effective,
                preprocessing=factory.to_contract(),
                training_preprocessing=factory.training_contract(),
                class_to_idx=class_to_idx,
                splits_dir=self.splits_dir,
                best_epoch=best_epoch,
                metrics={
                    "best_validation": {
                        "epoch": best_epoch,
                        "macro_f1": best_val_f1,
                        "loss": best_val_loss,
                    }
                },
                historical_fields={
                    "pipeline": "hpo",
                    "study_name": self.study_name,
                    "trial_number": trial.number,
                    "trial_state": status,
                    "epochs_run": len(history),
                    "test_used": False,
                },
            )
            write_run_contract(trial_dir, contract)

    def __call__(self, trial: optuna.Trial) -> float:
        """Traza también fallos de inicialización/OOM anteriores a la primera época."""
        trial.set_user_attr("trial_id", f"trial_{trial.number:03d}")
        trial.set_user_attr("run_id", f"trial_{trial.number:03d}")
        trial.set_user_attr("training_seed", self.seed)
        try:
            return self._run_trial(trial)
        except optuna.TrialPruned:
            raise
        except Exception as error:
            details = "".join(traceback.format_exception(error))[-12000:]
            trial.set_user_attr("error", details)
            if self.study_dir is not None:
                directory = self.study_dir / "trials" / f"trial_{trial.number:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                summary_path = directory / "trial_summary.json"
                payload = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                payload.update(
                    state="FAIL",
                    error=details,
                    parameters=trial.params,
                    finished_at=datetime.now(UTC).isoformat(),
                )
                atomic_write_json(summary_path, payload)
                atomic_write_json(
                    directory / "failure.json",
                    {
                        "trial_id": directory.name,
                        "state": "FAIL",
                        "error": details,
                        "parameters": trial.params,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
            raise
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _run_trial(self, trial: optuna.Trial) -> float:
        """Ejecuta un trial completo con poda temprana y early stopping."""
        # La semilla de entrenamiento es fija; el sampler de Optuna es quien explora.
        set_global_seed(self.seed)
        self._verify_splits_unchanged()
        params = self.sample_parameters(trial)
        if self.on_parameters_sampled is not None:
            self.on_parameters_sampled(trial)
        started_at = datetime.now(UTC).isoformat()
        started = perf_counter()
        history: list[dict[str, Any]] = []
        best_val_f1 = float("-inf")
        best_val_loss = float("inf")
        best_epoch = 0
        early_stopped_flag = False
        trial_dir = (
            self.study_dir / "trials" / f"trial_{trial.number:03d}"
            if self.study_dir is not None
            else None
        )
        if trial_dir is not None:
            trial_dir.mkdir(parents=True, exist_ok=True)

        target_size = resolve_input_size(self.model_name, self.base_target_size)
        factory = CornTransformFactory(
            config_path=str(self.config_path),
            target_size=target_size,
            clahe=params["clahe"],
        )
        mapping = {name: idx for idx, name in enumerate(self.cfg["dataset"]["classes"])}
        effective = self._effective_hyperparameters(params)
        config_hash = compute_config_sha256(
            model_name=self.model_name,
            input_size=factory.target_size,
            num_classes=len(mapping),
            seed=self.seed,
            hyperparameters=effective,
            preprocessing=factory.to_contract(),
            class_to_idx=mapping,
        )
        trial.set_user_attr("config_sha256", config_hash)
        if trial_dir is not None:
            atomic_write_json(
                trial_dir / "trial_summary.json",
                {
                    "schema_version": 1,
                    "study_name": self.study_name,
                    "trial_id": trial_dir.name,
                    "trial_number": trial.number,
                    "state": "RUNNING",
                    "started_at": started_at,
                    "parameters": params,
                    "effective_hyperparameters": effective,
                    "preprocessing": factory.to_contract(),
                    "class_to_idx": mapping,
                    "config_sha256": config_hash,
                    "split_hashes": self.expected_split_hashes,
                    "training_seed": self.seed,
                    "best_epoch": None,
                    "executed_epochs": 0,
                    "best_val_macro_f1": None,
                    "best_val_loss": None,
                    "checkpoint_sha256": None,
                },
            )

        train_dataset = CornDataset(
            csv_path=str(self.splits_dir / "train.csv"),
            config_path=str(self.config_path),
            transform=factory.get_pipeline("train"),
            minority_transform=factory.get_pipeline("minority"),
        )
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = train_dataset.idx_to_class

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
        try:
            for epoch_index in range(self.epochs):
                epoch = epoch_index + 1
                epoch_started = perf_counter()
                train_metrics, _, _, _ = run_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    device=self.device,
                    optimizer=optimizer,
                    desc=f"[Trial {trial.number} Ep {epoch}/{self.epochs}] Train",
                    clip_grad_norm=1.0,
                )

                val_metrics, labels, predictions, probs = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=self.device,
                    optimizer=None,
                    desc=f"[Trial {trial.number} Ep {epoch}/{self.epochs}] Val",
                )
                current_f1 = float(val_metrics["macro_f1"])
                if not all(
                    np.isfinite(v) for v in (*train_metrics.values(), *val_metrics.values())
                ):
                    raise ValueError(f"Métrica/pérdida no finita en época {epoch}.")
                row = {
                    "epoch": epoch,
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "train_macro_f1": float(train_metrics["macro_f1"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_macro_f1": current_f1,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "epoch_seconds": perf_counter() - epoch_started,
                    "is_best": False,
                }
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_val_loss = float(val_metrics["loss"])
                    best_epoch = epoch
                    row["is_best"] = True
                    if trial_dir is not None:
                        checkpoint_tmp = trial_dir / "best.pth.tmp"
                        torch.save(model.state_dict(), checkpoint_tmp)
                        checkpoint_tmp.replace(trial_dir / "best.pth")
                        predictions_df, diagnostics = self._validation_diagnostics(
                            labels=labels,
                            predictions=predictions,
                            probabilities=probs,
                            idx_to_class=idx_to_class,
                            class_to_idx=class_to_idx,
                        )
                        predictions_df.to_csv(trial_dir / "validation_predictions.csv", index=False)
                        atomic_write_json(trial_dir / "validation_diagnostics.json", diagnostics)
                history.append(row)
                if trial_dir is not None:
                    pd.DataFrame(history).to_csv(trial_dir / "training_history.csv", index=False)
                if scheduler is not None:
                    scheduler.step()

                trial.report(current_f1, epoch)
                if trial.should_prune():
                    self._finalize_trial_artifacts(
                        trial=trial,
                        trial_dir=trial_dir,
                        status="PRUNED",
                        params=params,
                        history=history,
                        best_epoch=best_epoch,
                        best_val_f1=best_val_f1,
                        best_val_loss=best_val_loss,
                        early_stopped=False,
                        started_at=started_at,
                        duration_seconds=perf_counter() - started,
                        factory=factory,
                        class_to_idx=class_to_idx,
                    )
                    self._verify_splits_unchanged()
                    raise optuna.TrialPruned(
                        f"Podado en época {epoch}; best val Macro-F1={best_val_f1:.6f}"
                    )

                if early_stopping.step(current_f1):
                    early_stopped_flag = True
                    logger.info(
                        "[Trial %d] Early stopping alcanzado en época %d.",
                        trial.number,
                        epoch,
                    )
                    break

            self._finalize_trial_artifacts(
                trial=trial,
                trial_dir=trial_dir,
                status="COMPLETE",
                params=params,
                history=history,
                best_epoch=best_epoch,
                best_val_f1=best_val_f1,
                best_val_loss=best_val_loss,
                early_stopped=early_stopped_flag,
                started_at=started_at,
                duration_seconds=perf_counter() - started,
                factory=factory,
                class_to_idx=class_to_idx,
            )
            self._verify_splits_unchanged()
            return best_val_f1
        except optuna.TrialPruned:
            raise
        except Exception as error:
            error_text = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            self._finalize_trial_artifacts(
                trial=trial,
                trial_dir=trial_dir,
                status="FAIL",
                params=params,
                history=history,
                best_epoch=best_epoch,
                best_val_f1=best_val_f1,
                best_val_loss=best_val_loss,
                early_stopped=False,
                started_at=started_at,
                duration_seconds=perf_counter() - started,
                factory=factory,
                class_to_idx=class_to_idx,
                error=error_text[-12000:],
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise


def save_optimization_plots(study: optuna.Study, output_dir: Path) -> list[Path]:
    """Genera figuras reconstruibles desde el study persistente."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: list[Path] = []

    trials = [
        trial
        for trial in study.trials
        if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE
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
            label="Mejor Macro F1 acumulado",
            zorder=4,
        )
        ax.set_title(
            f"Historial de optimización — {study.study_name}",
            fontsize=13,
            fontweight="bold",
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

    try:
        importances = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42)
        )
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
                "Importancia de hiperparámetros (Macro-F1)",
                fontsize=13,
                fontweight="bold",
            )
            ax.grid(True, linestyle="--", alpha=0.5, axis="x")
            fig.tight_layout()

            imp_path = output_dir / "parameter_importance.png"
            fig.savefig(imp_path, dpi=160)
            plt.close(fig)
            saved_plots.append(imp_path)
    except Exception as err:
        logger.warning("No se pudo calcular la importancia de parámetros: %s", err)

    return saved_plots


def _terminal_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    terminal = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    return [trial for trial in study.trials if trial.state in terminal]


def terminal_trial_count(study: optuna.Study) -> int:
    """Cuenta intentos registrados, excluyendo WAITING/RUNNING."""
    return len(_terminal_trials(study))


def fail_stale_running_trials(study: optuna.Study) -> list[int]:
    """Convierte trials RUNNING de un proceso interrumpido en FAIL al reanudar."""
    failed: list[int] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        study._storage.set_trial_system_attr(  # noqa: SLF001
            trial._trial_id,  # noqa: SLF001
            "resume_error",
            "Proceso anterior interrumpido; trial marcado FAIL al reanudar.",
        )
        changed = study._storage.set_trial_state_values(  # noqa: SLF001
            trial._trial_id,
            optuna.trial.TrialState.FAIL,  # noqa: SLF001
        )
        if changed:
            failed.append(trial.number)
    return failed


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    duration = trial.duration.total_seconds() if trial.duration is not None else None
    attrs = trial.user_attrs
    return {
        "trial_number": trial.number,
        "trial_id": attrs.get("trial_id", f"trial_{trial.number:03d}"),
        "state": trial.state.name,
        "value": trial.value,
        "learning_rate": trial.params.get("learning_rate"),
        "weight_decay": trial.params.get("weight_decay"),
        "label_smoothing": trial.params.get("label_smoothing"),
        "batch_size": trial.params.get("batch_size"),
        "class_weights": trial.params.get("class_weights"),
        "warmup_epochs": trial.params.get("warmup_epochs"),
        "dropout": None,
        "scheduler": "cosine",
        "optimizer": "AdamW",
        "best_epoch": attrs.get("best_epoch"),
        "executed_epochs": attrs.get("executed_epochs"),
        "early_stopped": attrs.get("early_stopped"),
        "best_val_loss": attrs.get("best_val_loss"),
        "best_train_macro_f1": attrs.get("best_train_macro_f1"),
        "train_macro_f1_at_best_val": attrs.get("train_macro_f1_at_best_val"),
        "train_val_gap": attrs.get("train_val_gap"),
        "duration_seconds": attrs.get("duration_seconds", duration),
        "datetime_start": (trial.datetime_start.isoformat() if trial.datetime_start else None),
        "datetime_complete": (
            trial.datetime_complete.isoformat() if trial.datetime_complete else None
        ),
        "run_id": attrs.get("run_id"),
        "config_sha256": attrs.get("config_sha256"),
        "error": attrs.get("error") or trial.system_attrs.get("resume_error"),
    }


def _environment_versions() -> dict[str, Any]:
    try:
        import timm
    except ImportError:
        timm_version = None
    else:
        timm_version = timm.__version__
    try:
        import torchvision
    except ImportError:
        torchvision_version = None
    else:
        torchvision_version = torchvision.__version__
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision_version,
        "timm": timm_version,
        "optuna": optuna.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def export_tuning_artifacts(
    study: optuna.Study,
    output_dir: Path,
    model_name: str,
    baseline_macro_f1: float,
    *,
    requested_trials: int = 60,
    seed: int = 42,
    split_snapshot: dict[str, Any] | None = None,
    search_space: HyperparameterSpace | None = None,
    epochs: int = 60,
    patience: int = 8,
    sampler_name: str = "TPESampler",
    pruner_name: str = "MedianPruner",
    pruner_config: dict[str, Any] | None = None,
    parallel_workers: int = 1,
) -> dict[str, Any]:
    """Exporta la historia completa y los artefactos formales del estudio."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    records = [_trial_record(trial) for trial in _terminal_trials(study)]
    df_trials = pd.DataFrame(records)
    if records:
        df_trials = df_trials.sort_values("trial_number")
    else:
        df_trials = pd.DataFrame(columns=["trial_number", "trial_id", "state", "value"])
    trials_csv = output_dir / "trials.csv"
    df_trials.to_csv(trials_csv, index=False)

    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        summary = {
            "schema_version": 1,
            "study_name": study.study_name,
            "model": model_name,
            "status": "running",
            "n_trials_requested": requested_trials,
            "n_recorded": len(records),
            "n_complete": 0,
            "n_pruned": sum(row["state"] == "PRUNED" for row in records),
            "n_failed": sum(row["state"] == "FAIL" for row in records),
            "test_used_during_hpo": False,
        }
        atomic_write_json(output_dir / "study_summary.json", summary)
        return summary

    best_trial = max(complete, key=lambda item: float(item.value))
    best_f1 = float(best_trial.value) if best_trial.value is not None else 0.0
    improvement_delta = best_f1 - baseline_macro_f1
    improvement_pct = (
        (improvement_delta / baseline_macro_f1) * 100.0 if baseline_macro_f1 > 0 else 0.0
    )
    effective_best_params = {
        "learning_rate": best_trial.params["learning_rate"],
        "weight_decay": best_trial.params["weight_decay"],
        "batch_size": best_trial.params["batch_size"],
        "class_weights": best_trial.params["class_weights"],
        "label_smoothing": best_trial.params["label_smoothing"],
        "warmup_epochs": best_trial.params["warmup_epochs"],
        "clahe": best_trial.params.get("clahe", False),
        "epochs": epochs,
    }
    best_params_payload = {
        "schema": HPO_SCHEMA,
        "best_params": effective_best_params,
    }
    atomic_write_json(output_dir / "best_hyperparameters.json", best_params_payload)
    # Alias histórico: no rompe consumidores previos y contiene el mismo esquema estricto.
    atomic_write_json(output_dir / "best_params.json", best_params_payload)
    load_best_params(output_dir / "best_hyperparameters.json")

    counts = {
        state: sum(row["state"] == state for row in records)
        for state in ("COMPLETE", "PRUNED", "FAIL")
    }
    convergence: dict[str, float | None] = {}
    for cutoff in sorted({*range(10, requested_trials + 1, 10), requested_trials}):
        values = [
            float(trial.value)
            for trial in complete
            if trial.number < cutoff and trial.value is not None
        ]
        convergence[f"best_at_{cutoff}"] = (
            max(values) if values and len(records) >= cutoff else None
        )

    top = sorted(complete, key=lambda item: float(item.value), reverse=True)[:10]
    top_rows = []
    for rank, trial in enumerate(top, start=1):
        record = _trial_record(trial)
        diagnostic_path = (
            output_dir / "trials" / f"trial_{trial.number:03d}" / "validation_diagnostics.json"
        )
        diagnostics = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
        calibration = diagnostics.get("calibration", {})
        top_rows.append(
            {
                "rank": rank,
                "trial": trial.number,
                "val_macro_f1": trial.value,
                "learning_rate": trial.params.get("learning_rate"),
                "weight_decay": trial.params.get("weight_decay"),
                "label_smoothing": trial.params.get("label_smoothing"),
                "batch_size": trial.params.get("batch_size"),
                "class_weights": trial.params.get("class_weights"),
                "warmup_epochs": trial.params.get("warmup_epochs"),
                "dropout": None,
                "scheduler": "cosine",
                "best_epoch": record["best_epoch"],
                "duration_seconds": record["duration_seconds"],
                "best_train_macro_f1": record["best_train_macro_f1"],
                "train_macro_f1_at_best_val": record["train_macro_f1_at_best_val"],
                "train_val_gap": record["train_val_gap"],
                "best_train_val_gap": (
                    record["best_train_macro_f1"] - trial.value
                    if record["best_train_macro_f1"] is not None
                    else None
                ),
                "validation_ece": calibration.get("ece"),
                "validation_mean_confidence_correct": calibration.get("mean_confidence_hits"),
                "validation_mean_confidence_errors": calibration.get("mean_confidence_misses"),
                **{
                    f"{name}_f1": diagnostics.get("per_class", {}).get(name, {}).get("f1-score")
                    for name in NPK_GROUPS
                },
            }
        )
    pd.DataFrame(top_rows).to_csv(output_dir / "top_10_trials.csv", index=False)
    atomic_write_json(output_dir / "top_10_trials.json", top_rows)
    atomic_write_json(output_dir / "convergence.json", convergence)

    importance_rows: list[dict[str, Any]] = []
    try:
        importances = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42)
        )
        importance_rows = [
            {"parameter": parameter, "importance": importance}
            for parameter, importance in importances.items()
        ]
    except Exception as error:
        logger.warning("No se pudo calcular importancia de hiperparámetros: %s", error)
    pd.DataFrame(importance_rows, columns=["parameter", "importance"]).to_csv(
        output_dir / "hyperparameter_importance.csv", index=False
    )

    pruner_config = pruner_config or {
        "n_startup_trials": 5,
        "n_warmup_steps": 8,
        "interval_steps": 1,
    }
    snapshot = split_snapshot or {}
    best_summary = {
        "schema_version": 1,
        "study_name": study.study_name,
        "model": model_name,
        "status": ("complete" if len(records) == requested_trials else "running"),
        "seed": seed,
        "training_seed": seed,
        "sampler_seed": seed,
        "split_identifier": snapshot.get("split_identifier", "seed_42"),
        "split_hashes": snapshot.get("sha256"),
        "objective": "best_validation_macro_f1",
        "direction": "maximize",
        "test_used_during_hpo": False,
        "n_trials_requested": requested_trials,
        "n_recorded": len(records),
        "n_complete": counts["COMPLETE"],
        "n_pruned": counts["PRUNED"],
        "n_failed": counts["FAIL"],
        "sampler": sampler_name,
        "pruner": pruner_name,
        "pruner_config": pruner_config,
        "parallel_workers": parallel_workers,
        "search_space": (search_space or HyperparameterSpace()).to_dict(),
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "best_trial_number": best_trial.number,
        "best_value": best_f1,
        "best_val_macro_f1": best_f1,
        "baseline_value": baseline_macro_f1,
        "baseline_macro_f1": baseline_macro_f1,
        "improvement": {
            "absolute": improvement_delta,
            "percentage_points": improvement_delta * 100.0,
            "relative_percent": improvement_pct,
        },
        "improvement_delta": improvement_delta,
        "improvement_pct": improvement_pct,
        "best_params": effective_best_params,
        "convergence": convergence,
        "environment": _environment_versions(),
        "artifacts": {
            "study_storage": "study.db",
            "trials": "trials.csv",
            "top_10": "top_10_trials.csv",
            "best_hyperparameters": "best_hyperparameters.json",
            "importance": "hyperparameter_importance.csv",
            "figures": "figures/",
        },
    }
    atomic_write_json(output_dir / "study_summary.json", best_summary)

    comparison_rows = [
        {
            "configuration": "baseline_20260921_204608",
            "model": model_name,
            "val_macro_f1": baseline_macro_f1,
            "improvement_absolute": 0.0,
            "improvement_percentage_points": 0.0,
        },
        {
            "configuration": f"hpo_trial_{best_trial.number:03d}",
            "model": model_name,
            "val_macro_f1": best_f1,
            "improvement_absolute": improvement_delta,
            "improvement_percentage_points": improvement_delta * 100.0,
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison_vs_baseline.csv", index=False)

    save_optimization_plots(study, figures_dir)

    return best_summary


def write_selection_lock(
    *,
    study: optuna.Study,
    study_dir: Path,
    split_snapshot: dict[str, Any],
    search_space: HyperparameterSpace,
    requested_trials: int = 60,
) -> dict[str, Any]:
    """Congela al ganador antes de construir o cargar el test dataset."""
    protocol_budget = study.user_attrs.get("protocol", {}).get("n_trials", requested_trials)
    if requested_trials < 1 or protocol_budget != requested_trials:
        raise RuntimeError("El presupuesto solicitado difiere del protocolo persistido.")
    if terminal_trial_count(study) != requested_trials:
        raise RuntimeError(
            f"La selección solo puede bloquearse con exactamente {requested_trials} trials."
        )
    if len(study.trials) != requested_trials:
        raise RuntimeError("Hay trials adicionales activos o en espera; no puede seleccionarse.")
    best = study.best_trial
    checkpoint = study_dir / "trials" / f"trial_{best.number:03d}" / "best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Falta el checkpoint exacto del ganador: {checkpoint}")
    best_params_path = study_dir / "best_hyperparameters.json"
    contract = validate_run_contract(checkpoint.parent)
    if contract.summary["metrics"]["best_validation"]["macro_f1"] != best.value:
        raise RuntimeError("El contrato del checkpoint no coincide con el objective ganador.")
    parameters = load_best_params(best_params_path)
    if any(parameters.get(k) != v for k, v in best.params.items()):
        raise RuntimeError("best_hyperparameters no coincide con el trial ganador.")
    if any(contract.summary["hyperparameters"].get(k) != v for k, v in parameters.items()):
        raise RuntimeError("Los parámetros del checkpoint difieren del ganador.")
    if contract.summary["split_manifest_sha256"] != split_snapshot["sha256"]["manifest.lock.json"]:
        raise SplitIntegrityError("El checkpoint pertenece a otros splits.")
    payload = {
        "schema_version": 1,
        "study_name": study.study_name,
        "selection_timestamp": datetime.now(UTC).isoformat(),
        "n_trials_requested": requested_trials,
        "objective": "best_validation_macro_f1",
        "direction": "maximize",
        "test_observed_before_selection": False,
        "best_trial": best.number,
        "best_validation_macro_f1": float(best.value),
        "best_hyperparameters": load_best_params(best_params_path),
        "search_space": search_space.to_dict(),
        "split_identifier": split_snapshot["split_identifier"],
        "split_hashes": split_snapshot["sha256"],
        "winner_checkpoint": str(checkpoint.relative_to(study_dir)),
        "winner_checkpoint_sha256": sha256_file(checkpoint),
        "winner_config_sha256": contract.summary["config_sha256"],
        "winner_summary_sha256": sha256_file(checkpoint.parent / "summary.json"),
        "best_hyperparameters_sha256": sha256_file(best_params_path),
    }
    lock_path = study_dir / "HPO_SELECTION_LOCK.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text())
        if any(existing.get(k) != v for k, v in payload.items() if k != "selection_timestamp"):
            raise RuntimeError("La selección está congelada y sus artefactos cambiaron.")
        return existing
    if (study_dir / "FINAL_TEST_STARTED.json").exists() or (
        study_dir / "FINAL_TEST_COMPLETE.json"
    ).exists():
        raise RuntimeError("Existe test previo sin selection lock; se requiere auditoría.")
    atomic_write_json(lock_path, payload)
    return payload
