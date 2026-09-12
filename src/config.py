import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# None cuando falta la variable: Path("") resolvería al cwd, que "existe" y burla los guards.
_raw_dataset_root = os.getenv("DATASET_ROOT", "").strip()
DATASET_ROOT: Path | None = Path(_raw_dataset_root) if _raw_dataset_root else None

_raw_output_root = os.getenv("OUTPUT_ROOT", "").strip()


def get_dataset_root() -> Path:
    """Resolve source data; an explicit invalid path must never fall back."""
    if DATASET_ROOT is not None:
        if not DATASET_ROOT.is_dir():
            raise SystemExit(f"DATASET_ROOT is not an existing directory: {DATASET_ROOT}")
        return DATASET_ROOT
    local_data = PROJECT_ROOT / "data"
    if local_data.exists():
        return local_data
    if DATASET_ROOT is None:
        raise SystemExit(
            "DATASET_ROOT no está definido. Copia .env.example a .env y configúralo "
            "(ver LOCAL.md, sección 3)."
        )


def get_output_root() -> Path:
    """Honor configured outputs, including a not-yet-created directory."""
    if _raw_output_root:
        p = Path(_raw_output_root)
        if p.exists() and not p.is_dir():
            raise SystemExit(f"OUTPUT_ROOT is not a directory: {p}")
        return p
    return PROJECT_ROOT / "outputs"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
