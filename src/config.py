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
    """Devuelve DATASET_ROOT validado; si la ruta configurada no existe pero PROJECT_ROOT/data sí, usa local."""
    if DATASET_ROOT is not None and DATASET_ROOT.exists():
        return DATASET_ROOT
    local_data = PROJECT_ROOT / "data"
    if local_data.exists():
        return local_data
    if DATASET_ROOT is None:
        raise SystemExit(
            "DATASET_ROOT no está definido. Copia .env.example a .env y configúralo "
            "(ver LOCAL.md, sección 3)."
        )
    return DATASET_ROOT


def get_output_root() -> Path:
    """OUTPUT_ROOT si está definido y es válido en el SO actual; si no, PROJECT_ROOT/outputs."""
    if _raw_output_root:
        if os.name == "nt" and _raw_output_root.startswith(("/", "\\")):
            local_candidate = PROJECT_ROOT / _raw_output_root.lstrip("/\\")
            if local_candidate.exists():
                return local_candidate
        p = Path(_raw_output_root)
        if p.exists():
            return p
    return PROJECT_ROOT / "outputs"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
