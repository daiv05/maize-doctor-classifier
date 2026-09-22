"""Validación estricta de hiperparámetros externos y contratos efectivos."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HPO_SCHEMA = "pytorch_hpo_v1"

_NUMERIC_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "learning_rate": (0.0, 1.0, False),
    "weight_decay": (0.0, 1.0, True),
    "label_smoothing": (0.0, 1.0, True),
    "batch_size": (1, 65536, True),
    "warmup_epochs": (0, 10000, True),
    "epochs": (1, 10000, True),
}
_ALLOWED_KEYS = set(_NUMERIC_BOUNDS) | {"clahe", "class_weights"}


def load_best_params(path: str | Path) -> dict[str, Any]:
    """Carga parámetros PyTorch, rechazando esquema, claves, tipos y rangos inválidos."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = payload.get("schema", payload.get("schema_version", HPO_SCHEMA))
    if schema not in {HPO_SCHEMA, 1}:
        raise ValueError(f"Esquema HPO incompatible: {schema!r}; esperado {HPO_SCHEMA!r}.")
    params = payload.get(
        "best_params",
        {key: value for key, value in payload.items() if key not in {"schema", "schema_version"}},
    )
    if not isinstance(params, dict) or not params:
        raise ValueError("best_params debe ser un objeto JSON no vacío.")
    unknown = sorted(set(params) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Hiperparámetros PyTorch desconocidos: {unknown}")

    for key, value in params.items():
        if key in _NUMERIC_BOUNDS:
            lower, upper, lower_inclusive = _NUMERIC_BOUNDS[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Valor inválido para {key}: {value!r}")
            if key in {"batch_size", "warmup_epochs", "epochs"} and type(value) is not int:
                raise ValueError(f"{key} debe ser un entero.")
            lower_ok = value >= lower if lower_inclusive else value > lower
            if not lower_ok or value > upper:
                interval = "[" if lower_inclusive else "("
                raise ValueError(f"{key}={value!r} fuera del rango {interval}{lower}, {upper}].")
        elif key == "clahe" and type(value) is not bool:
            raise ValueError("clahe debe ser un booleano JSON.")
        elif key == "class_weights" and value not in {"none", "inverse", "sqrt_inverse"}:
            raise ValueError(f"Estrategia class_weights inválida: {value!r}")
    return params


def parse_with_best_params(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    """Aplica precedencia ``CLI explícito > archivo HPO > defaults del parser``."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    preliminary, _ = parser.parse_known_args(arguments)
    path = getattr(preliminary, "best_params", None) or getattr(
        preliminary, "best_params_path", None
    )
    if path:
        params = load_best_params(path)
        parser_destinations = {action.dest for action in parser._actions}
        unsupported = sorted(set(params) - parser_destinations)
        if unsupported:
            raise ValueError(
                f"El CLI actual no admite estos hiperparámetros del archivo: {unsupported}"
            )
        parser.set_defaults(**params)
    return parser.parse_args(arguments)


def validate_effective_hyperparameters(payload: dict[str, Any]) -> dict[str, Any]:
    """Valida el bloque completo que quedará registrado en el contrato de run."""
    required = {
        "learning_rate",
        "batch_size",
        "weight_decay",
        "optimizer",
        "scheduler",
        "epochs",
        "patience",
        "dropout",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Faltan hiperparámetros efectivos contractuales: {missing}")
    if not isinstance(payload["optimizer"], str) or not payload["optimizer"]:
        raise ValueError("optimizer debe ser un nombre no vacío.")
    if not isinstance(payload["scheduler"], str) or not payload["scheduler"]:
        raise ValueError("scheduler debe ser un nombre no vacío.")
    if payload["patience"] is not None and (
        type(payload["patience"]) is not int or payload["patience"] < 1
    ):
        raise ValueError("patience debe ser null o un entero positivo.")
    if payload["dropout"] is not None and (
        type(payload["dropout"]) not in (int, float)
        or not math.isfinite(payload["dropout"])
        or not 0 <= payload["dropout"] < 1
    ):
        raise ValueError("dropout debe ser null o un número en [0, 1).")
    for key in ("learning_rate", "weight_decay", "batch_size", "epochs"):
        minimal = {key: payload[key]}
        # Reutiliza las reglas de rango sin exigir las demás claves del archivo HPO.
        value = minimal[key]
        lower, upper, lower_inclusive = _NUMERIC_BOUNDS[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Valor efectivo inválido para {key}: {value!r}")
        if key in {"batch_size", "epochs"} and type(value) is not int:
            raise ValueError(f"{key} efectivo debe ser entero.")
        if (value < lower if lower_inclusive else value <= lower) or value > upper:
            raise ValueError(f"Valor efectivo fuera de rango para {key}: {value!r}")
    return dict(payload)
