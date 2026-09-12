from __future__ import annotations

from typing import Any

import torch.nn as nn

import src.models.baselines.efficientnet  # noqa: F401 - registers models
import src.models.baselines.fastvit  # noqa: F401 - registers models
import src.models.baselines.ghostnet  # noqa: F401 - registers models
import src.models.baselines.mobilenet  # noqa: F401 - registers models
import src.models.baselines.shufflenet  # noqa: F401 - registers models
from src.models.ensemble import SoftVotingEnsemble
from src.models.input_sizes import MODEL_INPUT_SIZES, resolve_input_size
from src.models.registry import MODEL_REGISTRY, ModelEntry, ModelRegistry


def build_model(name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    return MODEL_REGISTRY.build(name, num_classes=num_classes, **kwargs)


def list_models() -> list[str]:
    return MODEL_REGISTRY.list_names()


__all__ = [
    "MODEL_INPUT_SIZES",
    "MODEL_REGISTRY",
    "ModelEntry",
    "ModelRegistry",
    "SoftVotingEnsemble",
    "build_model",
    "list_models",
    "resolve_input_size",
]

