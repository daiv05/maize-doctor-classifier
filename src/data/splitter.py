import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

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


class SourceGroupedSplitter(DatasetSplitter):
    """Reparte fuentes de origen enteras, nunca imágenes sueltas.

    El splitter estratificado deja cada fuente presente en las tres particiones con la misma
    proporción, de modo que un modelo puede reconocer la sesión de captura y ese atajo
    transfiere de entrenamiento a prueba. Aquí una fuente cae entera de un lado, así que el
    conjunto de prueba mide generalización a un dominio no visto.

    La contrapartida es estructural: una clase presente en menos de tres fuentes no puede
    aparecer en las tres particiones. El reparto lo detecta y lo reporta en lugar de emitir
    un split silenciosamente incompleto.
    """

    def __init__(self, seed: int = 42, group_column: str = "source_id",
                 allow_incomplete: bool = False):
        """
        @param {int} seed Semilla del desempate al ordenar fuentes del mismo tamaño.
        @param {str} group_column Columna que identifica la fuente de origen.
        @param {bool} allow_incomplete Permite continuar si alguna clase falta en val o test.
        """
        self.seed = seed
        self.group_column = group_column
        self.allow_incomplete = allow_incomplete

    def split(
        self, data_manifest: pd.DataFrame, train_size: float, val_size: float, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not abs((train_size + val_size + test_size) - 1.0) < 1e-9:
            raise ValueError("Las proporciones de train, val y test deben sumar exactamente 1.0")
        if self.group_column not in data_manifest.columns:
            raise ValueError(
                f"El manifiesto no tiene la columna '{self.group_column}'; sin ella no se "
                "puede agrupar por fuente."
            )

        total = len(data_manifest)
        targets = {"train": train_size * total, "val": val_size * total, "test": test_size * total}
        assigned: dict[str, str] = {}
        current = {name: 0 for name in targets}

        # Llenado voraz de mayor a menor: colocar primero las fuentes grandes evita que una
        # sola desborde una particion pequena al final.
        sizes = data_manifest[self.group_column].value_counts()
        for group in sorted(sizes.index, key=lambda g: (-sizes[g], str(g))):
            deficit = {name: targets[name] - current[name] for name in targets}
            chosen = max(deficit, key=lambda name: (deficit[name], name == "train"))
            assigned[group] = chosen
            current[chosen] += int(sizes[group])

        membership = data_manifest[self.group_column].map(assigned)
        frames = {name: data_manifest[membership == name].copy() for name in targets}

        classes = set(data_manifest["label"].unique())
        missing = {name: sorted(classes - set(frame["label"].unique()))
                   for name, frame in frames.items()}
        if missing["train"]:
            raise SystemExit(
                f"Clases sin ninguna fuente en train: {missing['train']}. El reparto por "
                "fuente no puede entrenarlas."
            )
        incomplete = {name: labels for name, labels in missing.items() if labels}
        if incomplete:
            detail = "; ".join(f"{name}: {labels}" for name, labels in incomplete.items())
            message = (
                f"Clases ausentes de alguna partición al agrupar por fuente -> {detail}. "
                "Ocurre cuando una clase tiene menos de tres fuentes, así que no puede estar "
                "en las tres a la vez."
            )
            if not self.allow_incomplete:
                raise SystemExit(
                    message + " Use allow_incomplete=True para continuar de todos modos, o "
                    "evalúe esas clases con validación dejando una fuente fuera."
                )
            logger.warning(message)

        for name, frame in frames.items():
            groups = sorted(frame[self.group_column].unique())
            logger.info(
                "%s: %d imágenes (%.1f%%), %d fuentes -> %s",
                name, len(frame), 100 * len(frame) / total, len(groups), groups,
            )
        return frames["train"], frames["val"], frames["test"]
