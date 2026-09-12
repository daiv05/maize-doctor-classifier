import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger(__name__)


class DatasetSplitter(ABC):
    """Interfaz abstracta para la partición de conjuntos de datos (DIP)."""

    @abstractmethod
    def split(
        self, data_manifest: pd.DataFrame, train_size: float, val_size: float, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass


class HierarchicalStratifiedSplitter(DatasetSplitter):
    """Ejecuta una división estratificada considerando combinaciones de Clase y Entorno."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split(
        self, data_manifest: pd.DataFrame, train_size: float, val_size: float, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not abs((train_size + val_size + test_size) - 1.0) < 1e-9:
            raise ValueError("Las proporciones de train, val y test deben sumar exactamente 1.0")
        if min(train_size, val_size, test_size) <= 0:
            raise ValueError("Las tres proporciones deben ser positivas.")
        if "group_id" in data_manifest:
            if data_manifest["group_id"].isna().any():
                raise ValueError("group_id incompleto")
            # Group membership takes precedence over approximate row proportions.
            outer = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=self.seed)
            train, rest = next(outer.split(data_manifest, groups=data_manifest["group_id"]))
            temporary = data_manifest.iloc[rest]
            inner = GroupShuffleSplit(
                n_splits=1, test_size=test_size / (val_size + test_size), random_state=self.seed
            )
            val, test = next(inner.split(temporary, groups=temporary["group_id"]))
            parts = (data_manifest.iloc[train], temporary.iloc[val], temporary.iloc[test])
            for part in parts:
                if set(part["label"]) != set(data_manifest["label"]):
                    raise ValueError(
                        "Grupos insuficientes para cubrir todas las clases en cada split."
                    )
            return parts

        # Generar súper-etiqueta temporal que fusiona la patología y el entorno de captura
        # Ejemplo: 'common_rust_real' o 'common_rust_lab'
        stratify_col = data_manifest["label"] + "_" + data_manifest["environment"]

        # Ajustar el tamaño proporcional del segundo split
        relative_test_size = test_size / (val_size + test_size)

        # Primer Split: Separar Entrenamiento del bloque de evaluación remanente
        train_df, temp_df = train_test_split(
            data_manifest, train_size=train_size, stratify=stratify_col, random_state=self.seed
        )

        # Recalcular la súper-etiqueta en el bloque remanente
        temp_stratify = temp_df["label"] + "_" + temp_df["environment"]

        # Segundo Split: Particionar de forma homogénea Validación y Prueba estrictos
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test_size, stratify=temp_stratify, random_state=self.seed
        )

        return train_df, val_df, test_df
