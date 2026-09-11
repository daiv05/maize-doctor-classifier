import logging
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, WeightedRandomSampler

from src.config import PROJECT_ROOT, get_dataset_root
from src.data.loader import load_and_normalize_image

_DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")

# Reintentos ante imágenes ilegibles antes de asumir que el dataset entero es inaccesible.
_MAX_FALLBACK_ATTEMPTS = 5

_DEFAULT_MINORITY_RATIO_THRESHOLD = 4.0

logger = logging.getLogger(__name__)


def _load_minority_ratio_threshold(config_path: str) -> float:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return float(
        config.get("augmentation", {}).get(
            "minority_ratio_threshold", _DEFAULT_MINORITY_RATIO_THRESHOLD
        )
    )


def compute_minority_classes(data_frame: pd.DataFrame, threshold: float) -> set[str]:
    """Deriva las clases minoritarias de la distribución real del split.

    Una clase es minoritaria si `max_count / count_de_la_clase > threshold`. En un split
    balanceado el conjunto queda vacío, así que ni el augmentation agresivo ni el
    WeightedRandomSampler se activan.
    """
    counts = data_frame["label"].value_counts()
    if counts.empty:
        return set()
    max_count = counts.max()
    return {str(label) for label, n in counts.items() if max_count / n > threshold}


class CornDataset(Dataset):
    """
    Componente Dataset personalizado para el mapeo y consumo indexado
    de imágenes de patologías y deficiencias en hojas de maíz.
    """

    def __init__(
        self,
        csv_path: str,
        config_path: str = _DEFAULT_CONFIG,
        transform=None,
        minority_transform=None,
        exclude_classes: list[str] | None = None,
        class_to_idx: dict[str, int] | None = None,
        minority_classes: set[str] | None = None,
        max_per_class: int | None = None,
        seed: int = 42,
    ):
        """
        Args:
            csv_path: Ruta al manifiesto del split (train.csv, val.csv o test.csv).
            config_path: Ruta al archivo de configuración paramétrica.
            transform: Pipeline de transformaciones estándar (torchvision).
            minority_transform: Pipeline extendido aplicado a las clases minoritarias.
                                Si None, todas las muestras usan `transform`.
            exclude_classes: Clases a excluir del dataset en tiempo de construcción.
                             El CSV permanece inmutable; la exclusión es una decisión de pipeline.
            class_to_idx: Mapeo canónico clase->índice a reutilizar (el del split de train,
                          inyectado en val/test) para mantener índices consistentes entre
                          splits. Si None, se construye desde el YAML.
            max_per_class: Tope de muestras por clase. Se aplica despues de derivar las
                           clases minoritarias, de modo que reduce volumen sin alterar que
                           clases reciben augmentation agresivo. None o 0 desactiva el tope.
            seed: Semilla del submuestreo por clase.
            minority_classes: Conjunto de clases que reciben augmentation agresivo. Si None,
                              se deriva de la distribución real del split (ver
                              `compute_minority_classes` y `augmentation.minority_ratio_threshold`).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el archivo de manifiesto: {csv_path}")

        self.transform = transform
        self.minority_transform = minority_transform
        self.dataset_root = get_dataset_root()

        # 1. Cargar y filtrar el manifiesto
        df = pd.read_csv(csv_path)
        if exclude_classes:
            df = df[~df["label"].isin(exclude_classes)].reset_index(drop=True)
        self.data_frame = df

        present = set(self.data_frame["label"].unique())

        if class_to_idx is not None:
            # Reutilizar el mapeo inyectado, validando cobertura total.
            unknown = present - set(class_to_idx)
            if unknown:
                raise ValueError(
                    f"Etiquetas en el CSV sin índice en el mapeo inyectado: {sorted(unknown)}"
                )
            self.class_to_idx = dict(class_to_idx)
            self.allowed_classes = sorted(self.class_to_idx, key=self.class_to_idx.__getitem__)
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Construir class_to_idx compacto solo con las clases presentes tras el filtro.
            # El orden respeta la lista del YAML para reproducibilidad entre ejecuciones.
            self.allowed_classes = [c for c in config["dataset"]["classes"] if c in present]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.allowed_classes)}

            # Validar que no haya etiquetas en el CSV no cubiertas por el YAML.
            # Falla en construcción, no en el primer batch.
            unknown = present - set(config["dataset"]["classes"])
            if unknown:
                raise ValueError(f"Etiquetas en el CSV no registradas en config: {sorted(unknown)}")

        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        if minority_classes is not None:
            self.minority_classes = set(minority_classes)
        else:
            self.minority_classes = compute_minority_classes(
                self.data_frame, _load_minority_ratio_threshold(config_path)
            )

        # El submuestreo va despues de derivar las minoritarias a proposito: recortar primero
        # comprimiria los ratios y vaciaria el conjunto, desactivando en silencio el
        # augmentation asimetrico. Con tope 1500 sobre el split de 9 clases ninguna clase
        # superaria el umbral de 4.0.
        if max_per_class:
            self.data_frame = pd.concat(
                [group.sample(min(len(group), max_per_class), random_state=seed)
                 for _, group in self.data_frame.groupby("label")],
                ignore_index=True,
            )

    def __len__(self) -> int:
        """Devuelve el tamaño neto total de la muestra actual."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Carga perezosa: lee, normaliza y transforma la muestra bajo demanda."""
        # Reintentos acotados: una imagen corrupta no mata el worker, pero una racha de
        # fallos (dataset inaccesible) sí se propaga.
        last_error: Exception | None = None
        for attempt in range(_MAX_FALLBACK_ATTEMPTS):
            row = self.data_frame.iloc[(idx + attempt) % len(self)]
            img_path = self.dataset_root / row["image_path"]
            try:
                image = load_and_normalize_image(img_path)
                class_name = row["label"]
                break
            except (FileNotFoundError, RuntimeError) as e:
                last_error = e
                logger.warning(
                    f"Imagen no disponible en idx={idx + attempt} ({img_path}): {e}. "
                    "Probando la siguiente fila."
                )
        else:
            raise RuntimeError(
                f"{_MAX_FALLBACK_ATTEMPTS} imágenes consecutivas ilegibles desde idx={idx}; "
                "verifica que DATASET_ROOT siga accesible y los splits estén al día."
            ) from last_error

        # 2. Mapear la etiqueta de texto a su correspondiente índice entero codificado
        label_idx = self.class_to_idx[class_name]

        # 3. Seleccionar pipeline: extendido para clases minoritarias, estándar para el resto
        pipeline = (
            self.minority_transform
            if self.minority_transform is not None and class_name in self.minority_classes
            else self.transform
        )
        if pipeline:
            image = pipeline(image)

        return image, label_idx


def resolve_class_mapping(
    train_csv_path: str, classes: list[str]
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Construye class_to_idx/idx_to_class filtrando `classes` (orden del YAML) contra
    las labels presentes en el CSV de train. Debe usarse el mismo `train_csv_path`
    con el que se entrenó el checkpoint que se va a cargar, ya que los índices de
    clase no se guardan en el state_dict.
    """
    train_df = pd.read_csv(train_csv_path)
    present_classes = sorted(train_df["label"].unique())
    allowed = [c for c in classes if c in present_classes]
    class_to_idx = {name: idx for idx, name in enumerate(allowed)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_weighted_sampler(dataset: "CornDataset", seed: int) -> WeightedRandomSampler | None:
    if not dataset.minority_classes:
        return None

    labels = dataset.data_frame["label"].tolist()
    class_counts = dataset.data_frame["label"].value_counts().to_dict()
    sample_weights = torch.tensor(
        [1.0 / class_counts[label] for label in labels], dtype=torch.float
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
