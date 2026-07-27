"""Loop de entrenamiento compartido por el pipeline de baselines y el principal.

Los mecanismos que distinguen al pipeline principal (scheduler, early stopping,
gradient clipping) son opcionales: con los tres en None, `fit` se reduce
exactamente al loop de baselines de la Tabla 6.2 del reporte.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _loss_denominator(criterion: torch.nn.Module, labels: torch.Tensor) -> float:
    """
    Calcula el denominador con el que `criterion` normalizo la perdida del batch.

    `CrossEntropyLoss(reduction="mean")` divide por la suma de los pesos de las
    muestras del batch, no por su numero: sin pesos ambos coinciden, con pesos no.
    Reponderar por el tamano del batch mezclaria las dos normalizaciones y sesgaria
    la perdida reportada en contra de los batches ricos en clases minoritarias.

    @param {torch.nn.Module} criterion Funcion de perdida usada en el batch.
    @param {torch.Tensor} labels Etiquetas reales del batch.
    @returns {float} Suma de pesos del batch, o su tamano si el criterio no lleva pesos.
    """
    weight = getattr(criterion, "weight", None)
    if weight is None:
        return float(labels.size(0))
    return float(weight.detach()[labels].sum().item())


def _metrics_from_predictions(
    labels: list[int], predictions: list[int], loss: float
) -> dict[str, float]:
    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
    clip_grad_norm: float | None = None,
) -> tuple[dict[str, float], list[int], list[int], list[float]]:
    """
    Ejecuta una pasada completa sobre el loader, entrenando si se pasa optimizer.

    La perdida de la epoca se promedia con el mismo denominador que aplico el criterio
    en cada batch (suma de pesos con `weight`, tamano del batch sin el), de modo que
    coincide con `sum(loss_i * w_i) / sum(w_i)` y no favorece a los batches con muchas
    muestras de clases mayoritarias.

    @param {torch.nn.Module} model Modelo a ejecutar.
    @param {DataLoader} loader Origen de los batches.
    @param {torch.nn.Module} criterion Funcion de perdida.
    @param {torch.device} device Dispositivo de computo.
    @param {torch.optim.Optimizer|None} optimizer Si se pasa, la pasada es de entrenamiento.
    @param {str} desc Etiqueta de la barra de progreso.
    @param {float|None} clip_grad_norm Norma maxima de gradiente; None desactiva el recorte.
    @returns {tuple} Metricas, etiquetas reales, predicciones y confianzas.
    """
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_denominator = 0.0
    labels_all: list[int] = []
    preds_all: list[int] = []
    probs_all: list[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            denominator = _loss_denominator(criterion, labels)
            running_loss += loss.item() * denominator
            running_denominator += denominator
            labels_all.extend(labels.detach().cpu().tolist())
            probs = logits.detach().softmax(dim=1)
            preds_all.extend(probs.argmax(dim=1).cpu().tolist())
            probs_all.extend(probs.max(dim=1).values.cpu().tolist())

    avg_loss = running_loss / running_denominator if running_denominator > 0 else 0.0
    metrics = _metrics_from_predictions(labels_all, preds_all, avg_loss)
    return metrics, labels_all, preds_all, probs_all


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_name: str,
    run_dir: Path | None = None,
    scheduler: object | None = None,
    early_stopping: object | None = None,
    clip_grad_norm: float | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """
    Entrena `epochs` epocas, guardando el mejor checkpoint por val_macro_f1.

    @param {torch.nn.Module} model Modelo a entrenar.
    @param {DataLoader} train_loader Batches de entrenamiento.
    @param {DataLoader} val_loader Batches de validacion.
    @param {torch.nn.Module} criterion Funcion de perdida.
    @param {torch.optim.Optimizer} optimizer Optimizador.
    @param {torch.device} device Dispositivo de computo.
    @param {int} epochs Numero maximo de epocas.
    @param {str} model_name Nombre del modelo, usado en logs e historial.
    @param {Path|None} run_dir Destino de best.pth/last.pth; None omite la escritura.
    @param {object|None} scheduler Scheduler con `.step()` por epoca; None deja el lr fijo.
    @param {object|None} early_stopping Objeto con `.step(metric) -> bool`; None desactiva la
        parada.
    @param {float|None} clip_grad_norm Norma maxima de gradiente.
    @param {Callable|None} on_epoch_end Callback invocado con (epoca, fila del historial).
    @returns {list[dict]} Historial, una fila por epoca ejecutada.
    """
    history: list[dict] = []
    best_val_macro_f1 = -1.0

    for epoch in range(1, epochs + 1):
        started = perf_counter()
        train_metrics, _, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            desc=f"{model_name} train {epoch}/{epochs}",
            clip_grad_norm=clip_grad_norm,
        )
        val_metrics, _, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            desc=f"{model_name} val {epoch}/{epochs}",
        )

        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": perf_counter() - started,
            "is_best": False,
        }
        if scheduler is not None:
            row["learning_rate"] = optimizer.param_groups[0]["lr"]
            scheduler.step()

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            row["is_best"] = True
            if run_dir is not None:
                torch.save(model.state_dict(), run_dir / "best.pth")
        history.append(row)
        if run_dir is not None:
            torch.save(model.state_dict(), run_dir / "last.pth")

        logger.info(
            "[%s] epoch %s/%s train_f1=%.4f val_f1=%.4f",
            model_name,
            epoch,
            epochs,
            train_metrics["macro_f1"],
            val_metrics["macro_f1"],
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)

        if early_stopping is not None and early_stopping.step(val_metrics["macro_f1"]):
            logger.info("[%s] Early stopping en la epoca %s", model_name, epoch)
            break

    return history
