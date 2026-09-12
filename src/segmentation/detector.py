"""Adaptador de inferencia YOLO para segmentación de hojas de maíz."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from src.segmentation.geometry import mask_bbox
from src.segmentation.leaf_processor import LeafInstance


def segmentation_runtime_contract() -> dict:
    """Explicit detector defaults and code versions used by batch and raw inference."""
    from src.provenance import sha256_file

    try:
        runtime = version("ultralytics")
    except PackageNotFoundError:
        runtime = None  # Actual inference still raises; useful for structural test doubles.
    return {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.70,
        "image_size": 640,
        "retina_masks": True,
        "ultralytics_version": runtime,
        "detector_source_sha256": sha256_file(Path(__file__)),
        "geometry_source_sha256": sha256_file(Path(__file__).with_name("geometry.py")),
    }


def _rasterize_polygon(polygon: np.ndarray, image_size: tuple[int, int]) -> Image.Image:
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    points = [(float(x), float(y)) for x, y in polygon]
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


class MaizeLeafSegmenter:
    """Wrapper para cargar pesos YOLO de segmentación y devolver instancias foliares."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        image_size: int = 640,
        device: str | int | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {self.checkpoint_path}")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as err:
                raise ImportError(
                    "Ultralytics no está instalado. Instálalo para usar MaizeLeafSegmenter."
                ) from err
            self._model = YOLO(str(self.checkpoint_path))
        return self._model

    def segment(self, image: Image.Image) -> Sequence[LeafInstance]:
        """Ejecuta inferencia sobre una PIL.Image y retorna las instancias detectadas."""
        model = self._get_model()
        results = model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        if not results:
            return ()

        res = results[0]
        if res.masks is None or res.boxes is None:
            return ()

        polygons = getattr(res.masks, "xy", [])
        boxes = res.boxes

        confidences = boxes.conf.cpu().tolist() if hasattr(boxes, "conf") else []
        classes = boxes.cls.cpu().tolist() if hasattr(boxes, "cls") else []

        instances: list[LeafInstance] = []
        img_size = image.size  # (width, height)

        for idx, poly in enumerate(polygons):
            if len(poly) < 3:
                continue
            poly_np = np.asarray(poly, dtype=np.float32)
            mask = _rasterize_polygon(poly_np, img_size)
            bbox = mask_bbox(mask)
            if bbox is None:
                continue

            conf = float(confidences[idx]) if idx < len(confidences) else 0.5
            cls_id = int(classes[idx]) if idx < len(classes) else 0

            instances.append(
                LeafInstance(
                    mask=mask,
                    confidence=conf,
                    bbox=bbox,
                    source_index=idx,
                    class_id=cls_id,
                )
            )

        return tuple(instances)
