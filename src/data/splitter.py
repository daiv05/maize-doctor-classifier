import logging
import random
from abc import ABC, abstractmethod
from collections import Counter

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
    """Reparte grupos lógicos enteros, nunca imágenes sueltas.

    La columna de grupo puede representar procedencia, planta, sesión u otra unidad lógica.
    El llenado conserva grupos completos y prioriza cobertura de clases y entornos antes de
    aproximar los tamaños objetivo. De este modo una desviación de proporción nunca se corrige
    introduciendo fuga.

    La contrapartida es estructural: una clase presente en menos de tres grupos no puede
    aparecer en las tres particiones. El reparto lo detecta y lo reporta en lugar de emitir
    un split silenciosamente incompleto.
    """

    def __init__(
        self, seed: int = 42, group_column: str = "source_id", allow_incomplete: bool = False
    ):
        """
        @param {int} seed Semilla del desempate al ordenar fuentes del mismo tamaño.
        @param {str} group_column Columna que identifica la unidad lógica indivisible.
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
                "puede hacer un reparto agrupado."
            )
        if data_manifest.empty:
            raise ValueError("El manifiesto no contiene muestras para repartir")
        if data_manifest[self.group_column].isna().any():
            raise ValueError(f"El manifiesto contiene {self.group_column} vacío")

        working = data_manifest.copy()
        working[self.group_column] = working[self.group_column].astype(str)
        if working[self.group_column].str.strip().eq("").any():
            raise ValueError(f"El manifiesto contiene {self.group_column} vacío")

        total = len(working)
        targets = {"train": train_size * total, "val": val_size * total, "test": test_size * total}
        assigned: dict[str, str] = {}
        current = {name: 0 for name in targets}
        labels = working["label"].astype(str)
        environments = (
            working["environment"].astype(str)
            if "environment" in working.columns
            else pd.Series("unknown", index=working.index)
        )
        total_labels = Counter(labels)
        total_environments = Counter(environments)
        current_labels = {name: Counter() for name in targets}
        current_environments = {name: Counter() for name in targets}

        sizes = working[self.group_column].value_counts()
        group_names = sorted(str(group) for group in sizes.index)
        random.Random(self.seed).shuffle(group_names)
        tie_rank = {group: rank for rank, group in enumerate(group_names)}

        # Los grupos grandes se colocan primero. La puntuación favorece una clase/entorno que
        # todavía falta en el candidato y luego minimiza desviaciones de tamaño y distribución.
        for group in sorted(group_names, key=lambda value: (-int(sizes[value]), tie_rank[value])):
            group_frame = working[working[self.group_column].eq(group)]
            group_labels = Counter(group_frame["label"].astype(str))
            group_environments = Counter(
                group_frame["environment"].astype(str)
                if "environment" in group_frame.columns
                else ["unknown"] * len(group_frame)
            )

            def candidate_score(name: str) -> tuple[float, float, float, int]:
                label_coverage = sum(
                    count > 0 and current_labels[name][label] == 0
                    for label, count in group_labels.items()
                )
                environment_coverage = sum(
                    count > 0 and current_environments[name][environment] == 0
                    for environment, count in group_environments.items()
                )
                coverage_gain = float(label_coverage) + 0.1 * float(environment_coverage)

                new_size = current[name] + len(group_frame)
                size_cost = abs(new_size - targets[name]) / max(targets[name], 1.0)
                label_cost = sum(
                    abs(
                        current_labels[name][label]
                        + group_labels[label]
                        - total_labels[label]
                        * {"train": train_size, "val": val_size, "test": test_size}[name]
                    )
                    / max(total_labels[label], 1)
                    for label in group_labels
                )
                environment_cost = sum(
                    abs(
                        current_environments[name][environment]
                        + group_environments[environment]
                        - total_environments[environment]
                        * {"train": train_size, "val": val_size, "test": test_size}[name]
                    )
                    / max(total_environments[environment], 1)
                    for environment in group_environments
                )
                return (
                    -coverage_gain,
                    size_cost,
                    label_cost + 0.25 * environment_cost,
                    ("train", "val", "test").index(name),
                )

            chosen = min(targets, key=candidate_score)
            assigned[group] = chosen
            current[chosen] += len(group_frame)
            current_labels[chosen].update(group_labels)
            current_environments[chosen].update(group_environments)

        membership = working[self.group_column].map(assigned)
        frames = {name: working[membership == name].copy() for name in targets}

        classes = set(working["label"].unique())
        missing = {
            name: sorted(classes - set(frame["label"].unique())) for name, frame in frames.items()
        }
        incomplete = {name: labels for name, labels in missing.items() if labels}
        if incomplete:
            details = []
            for split_name, missing_labels in incomplete.items():
                for label in missing_labels:
                    class_rows = working[working["label"].eq(label)]
                    details.append(
                        f"class={label!r} split={split_name!r} "
                        f"total_samples={len(class_rows)} "
                        f"independent_groups={class_rows[self.group_column].nunique()}"
                    )
            message = (
                "Cobertura de clases incompleta al conservar grupos indivisibles: "
                + "; ".join(details)
                + "."
            )
            if missing["train"] or not self.allow_incomplete:
                raise SystemExit(
                    message + " Use allow_incomplete=True para continuar de todos modos, o "
                    "agregue grupos independientes para las clases indicadas."
                )
            logger.warning(message)

        for name, frame in frames.items():
            groups = sorted(frame[self.group_column].unique())
            logger.info(
                "%s: %d imágenes (%.1f%%), %d grupos -> %s",
                name,
                len(frame),
                100 * len(frame) / total,
                len(groups),
                groups,
            )
        return frames["train"], frames["val"], frames["test"]
