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
    """Devuelve la raiz del dataset.

    Una ruta configurada que no existe aborta en lugar de caer al directorio local: con
    el fallback silencioso, un DATASET_ROOT mal escrito entrena sobre otro corpus sin
    aviso, y la procedencia de cada imagen es justo lo que este proyecto necesita saber.

    @returns {Path} Raiz validada del dataset.
    """
    if DATASET_ROOT is not None:
        if DATASET_ROOT.exists():
            return DATASET_ROOT
        raise SystemExit(
            f"DATASET_ROOT apunta a {DATASET_ROOT}, que no existe. Corrige .env "
            "(ver LOCAL.md, seccion 3) en lugar de dejar que el pipeline elija otra raiz."
        )
    local_data = PROJECT_ROOT / "data"
    if local_data.exists():
        return local_data
    raise SystemExit(
        "DATASET_ROOT no está definido y no existe PROJECT_ROOT/data. Copia "
        ".env.example a .env y configúralo (ver LOCAL.md, sección 3)."
    )


def get_output_root() -> Path:
    """Devuelve la raiz de artefactos.

    A diferencia del dataset, una ruta de salida inexistente se crea, asi que el fallback
    no puede enmascarar datos equivocados.

    @returns {Path} Raiz de artefactos.
    """
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
