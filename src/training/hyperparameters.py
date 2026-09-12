"""Validated, explicit transfer of PyTorch HPO parameters (never SGD probe settings)."""

import json
import math
from pathlib import Path


def load_best_params(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema", "pytorch_hpo_v1") != "pytorch_hpo_v1":
        raise ValueError("HPO schema is not pytorch_hpo_v1")
    params = payload.get("best_params", {k: v for k, v in payload.items() if k != "schema"})
    bounds = {
        "learning_rate": (0, 1),
        "weight_decay": (0, 1),
        "label_smoothing": (0, 1),
        "batch_size": (1, 65536),
        "warmup_epochs": (0, 10000),
        "epochs": (1, 10000),
    }
    allowed = set(bounds) | {"clahe", "class_weights"}
    if not params or set(params) - allowed:
        raise ValueError(
            f"Unsupported PyTorch HPO keys (SGD schemas are separate): {set(params) - allowed}"
        )
    for key, value in params.items():
        if key in bounds:
            lo, hi = bounds[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Invalid HPO value: {key}={value}")
            if key == "learning_rate" and value <= 0:
                raise ValueError("learning_rate must be positive")
            if key in {"batch_size", "warmup_epochs", "epochs"} and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        elif key == "clahe" and type(value) is not bool:
            raise ValueError("clahe must be a JSON boolean")
        elif key == "class_weights" and value not in {"none", "inverse", "sqrt_inverse"}:
            raise ValueError("Invalid class_weights strategy")
    return params


def parse_with_best_params(parser, argv=None):
    """Precedence: explicit CLI > explicit HPO file > parser defaults."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    initial = parser.parse_args(argv)
    path = getattr(initial, "best_params", None) or getattr(initial, "best_params_path", None)
    if path:
        parser.set_defaults(**load_best_params(path))
    return parser.parse_args(argv)
