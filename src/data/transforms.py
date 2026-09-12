from abc import ABC, abstractmethod

import cv2
import numpy as np
import torchvision.transforms as T
import yaml
from PIL import Image

from src.config import PROJECT_ROOT

_DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")


class CornCLAHETransform:
    """
    Ecualizacion adaptativa de histograma sobre el canal L de LAB.

    Solo toca la luminancia: ecualizar los tres canales RGB por separado desplaza el
    tono, y el color es la senal diagnostica de las deficiencias nutricionales, donde
    la clorosis amarillenta es justamente lo que distingue la clase. Es preprocesamiento
    determinista, no augmentation, asi que se aplica igual en train, val, test e inferencia.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: int = 8):
        """
        @param {float} clip_limit Umbral de recorte; valores altos amplifican ruido.
        @param {int} tile_grid Lado de la grilla de tiles.
        """
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        @param {Image.Image} image Imagen RGB de entrada.
        @returns {Image.Image} Imagen RGB con la luminancia ecualizada.
        """
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        lightness, green_red, blue_yellow = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=(self.tile_grid, self.tile_grid)
        )
        merged = cv2.merge((clahe.apply(lightness), green_red, blue_yellow))
        return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


class TransformPipelineFactory(ABC):
    """
    Interfaz abstracta para la creación de pipelines de transformación (DIP).
    """

    @abstractmethod
    def create_transforms(self) -> T.Compose:
        pass


class CornTrainingTransforms(TransformPipelineFactory):
    """
    Pipeline de transformaciones y augmentations específicas para el entrenamiento.

    Aplica distorsiones geométricas seguras para fitopatología, evitando
    alteraciones agresivas de color que arruinen el diagnóstico de deficiencias.
    """

    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def create_transforms(self) -> T.Compose:
        return T.Compose(
            [
                # Dimensiones
                T.Resize(self.target_size),
                # Augmentations geometricas
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
                # Augmentation sutil de iluminación
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0),
                # Conversión a Tensor y Normalización Z-score espectral
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )


class CornMinorityTransforms(TransformPipelineFactory):
    """
    Pipeline de augmentation extendida para clases minoritarias (ratio > 4x).

    Más agresivo que CornTrainingTransforms en geometría y color, pero preserva
    el hue diagnóstico: saturation y hue se tocan solo levemente para no destruir
    la señal de clorosis/amarillamiento en deficiencias nutricionales.
    """

    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def create_transforms(self) -> T.Compose:
        return T.Compose(
            [
                T.RandomResizedCrop(self.target_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=30, interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )


class CornValidationTransforms(TransformPipelineFactory):
    """
    Pipeline de transformaciones deterministas para validación y prueba.

    No aplica augmentations aleatorias para garantizar una evaluación justa y reproducible.
    El `Resize` directo (con distorsión de aspecto) es intencional (ver CLAUDE.md).
    """

    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def create_transforms(self) -> T.Compose:
        return T.Compose(
            [
                # Dimensiones
                T.Resize(self.target_size),
                # Conversión y normalización directa
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )


class CornTransformFactory:
    """
    Punto de acceso único (Factory) para obtener los pipelines según la etapa del flujo.
    """

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        target_size: tuple[int, int] | None = None,
        clahe: bool = False,
    ):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # target_size es [alto, ancho] - convención (h, w) de torchvision (ver CLAUDE.md)
        if target_size is None:
            height, width = config["dataset"]["target_size"]
            target_size = (height, width)
        self.target_size = target_size

        clahe_config = config.get("clahe", {})
        self.clahe_config = {
            "enabled": bool(clahe),
            "clip_limit": float(clahe_config.get("clip_limit", 2.0)),
            "tile_grid": int(clahe_config.get("tile_grid", 8)),
        }
        self.segmentation_contract = config.get("segmentation_input")
        self.clahe_transform = (
            CornCLAHETransform(
                clip_limit=float(clahe_config.get("clip_limit", 2.0)),
                tile_grid=int(clahe_config.get("tile_grid", 8)),
            )
            if clahe
            else None
        )

    def to_contract(self) -> dict:
        return {
            "schema_version": 1,
            "exif_transpose": True,
            "color_mode": "RGB",
            "target_size": list(self.target_size),
            "resize": "stretch",
            "interpolation": "bilinear",
            "antialias": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "clahe": dict(self.clahe_config),
            "segmentation": self.segmentation_contract,
        }

    @classmethod
    def from_contract(cls, contract: dict, config_path=None):
        import math

        size = contract.get("target_size", [])
        clahe = contract.get("clahe", {})
        if (
            len(size) != 2
            or any(type(x) is not int or x <= 0 for x in size)
            or type(clahe.get("enabled")) is not bool
            or type(clahe.get("tile_grid")) is not int
            or clahe["tile_grid"] <= 0
            or type(clahe.get("clip_limit")) not in (int, float)
            or not math.isfinite(clahe["clip_limit"])
            or clahe["clip_limit"] <= 0
        ):
            raise ValueError("Invalid preprocessing dimensions or CLAHE parameters")
        factory = cls(
            config_path=config_path or _DEFAULT_CONFIG, target_size=tuple(contract["target_size"])
        )
        factory.clahe_config = dict(contract["clahe"])
        factory.segmentation_contract = contract.get("segmentation")
        if factory.to_contract() != contract:
            raise ValueError("Unsupported preprocessing contract")
        if factory.clahe_config["enabled"]:
            factory.clahe_transform = CornCLAHETransform(
                clip_limit=factory.clahe_config["clip_limit"],
                tile_grid=factory.clahe_config["tile_grid"],
            )
        return factory

    def get_pipeline(self, stage: str) -> T.Compose:
        """Retorna el pipeline de transformación correspondiente a la etapa."""
        if stage.lower() == "train":
            pipeline = CornTrainingTransforms(self.target_size).create_transforms()
        elif stage.lower() == "minority":
            pipeline = CornMinorityTransforms(self.target_size).create_transforms()
        elif stage.lower() in ["val", "test", "inference"]:
            pipeline = CornValidationTransforms(self.target_size).create_transforms()
        else:
            raise ValueError(
                f"Etapa de pipeline desconocida: '{stage}'. "
                "Use 'train', 'minority', 'val', 'test' o 'inference'."
            )

        if self.clahe_transform is None:
            return pipeline
        return T.Compose([self.clahe_transform, *pipeline.transforms])
