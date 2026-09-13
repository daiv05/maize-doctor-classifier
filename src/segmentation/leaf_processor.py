"""Lógica de selección de instancia foliar, enmascaramiento y perfiles de salida."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from PIL import Image, ImageOps

from src.segmentation.geometry import (
    BoundingBox,
    RGBColor,
    apply_leaf_mask,
    binary_mask_image,
    crop_leaf_region,
    crop_square_centered,
    image_to_rgb,
    letterbox_image,
    mask_area_ratio,
    mask_centroid,
    normalize_rgb_color,
    validate_target_size,
)

MASK_BLACK = "mask_black"
BBOX_CROP = "bbox_crop"
CROP_MASK_BLACK = "crop_mask_black"
CROP_MASK_LETTERBOX = "crop_mask_letterbox"
SQUARE_CROP = "square_crop"
SUPPORTED_MASK_PROFILES = frozenset(
    {
        MASK_BLACK,
        BBOX_CROP,
        CROP_MASK_BLACK,
        CROP_MASK_LETTERBOX,
        SQUARE_CROP,
    }
)
FALLBACK_ORIGINAL = "original"
FALLBACK_REJECT = "reject"


@dataclass(frozen=True)
class LeafInstance:
    mask: Image.Image
    confidence: float
    bbox: BoundingBox
    source_index: int
    class_id: int = 0


@dataclass(frozen=True)
class LeafMaskProcessorConfig:
    processing_profile: str = CROP_MASK_LETTERBOX
    confidence_threshold: float = 0.50
    min_mask_area_ratio: float = 0.01
    near_full_warning_ratio: float = 0.98
    area_weight: float = 0.45
    center_weight: float = 0.35
    confidence_weight: float = 0.20
    background_value: RGBColor = (0, 0, 0)
    target_size: tuple[int, int] = (224, 224)
    fallback: str = FALLBACK_ORIGINAL

    def __post_init__(self) -> None:
        if self.processing_profile not in SUPPORTED_MASK_PROFILES:
            raise ValueError(f"processing_profile desconocido: {self.processing_profile!r}")
        normalize_rgb_color(self.background_value)
        validate_target_size(self.target_size)


@dataclass(frozen=True)
class SegmentedLeafProcessingResult:
    original_image: Image.Image
    mask: Image.Image | None
    masked_image: Image.Image | None
    processed_image: Image.Image | None
    crop_image: Image.Image | None
    bbox: BoundingBox | None
    confidence: float | None
    number_of_instances: int
    selected_instance: int | None
    preprocessing_strategy: str
    fallback_used: bool
    fallback_reason: str | None
    warnings: tuple[str, ...]
    source_image: str | None


def _center_proximity(mask: Image.Image, image_size: tuple[int, int]) -> float:
    centroid = mask_centroid(mask)
    if centroid is None:
        return 0.0
    width, height = image_size
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    distance = math.hypot(centroid[0] - center_x, centroid[1] - center_y)
    maximum = math.hypot(max(center_x, 0.5), max(center_y, 0.5))
    return max(0.0, min(1.0, 1.0 - distance / maximum))


def select_target_leaf(
    instances: Sequence[LeafInstance],
    image_size: tuple[int, int],
    config: LeafMaskProcessorConfig,
) -> tuple[LeafInstance | None, str | None, list[str]]:
    warnings: list[str] = []
    eligible: list[tuple[float, LeafInstance]] = []

    for idx, inst in enumerate(instances):
        if inst.class_id != 0:
            continue
        if inst.confidence < config.confidence_threshold:
            continue
        norm_mask = binary_mask_image(inst.mask, expected_size=image_size)
        area = mask_area_ratio(norm_mask)
        if area < config.min_mask_area_ratio:
            continue
        if math.isclose(area, 1.0, abs_tol=1e-12):
            continue

        c_prox = _center_proximity(norm_mask, image_size)
        eligible.append((area, inst))

    if not eligible:
        return None, "sin instancias elegibles", warnings

    largest_area = max(item[0] for item in eligible)
    scored: list[tuple[float, LeafInstance]] = []
    for area, inst in eligible:
        norm_mask = binary_mask_image(inst.mask, expected_size=image_size)
        c_prox = _center_proximity(norm_mask, image_size)
        rel_area = area / largest_area
        score = (
            config.area_weight * rel_area
            + config.center_weight * c_prox
            + config.confidence_weight * inst.confidence
        )
        scored.append((score, inst))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], None, warnings


class SegmentedLeafProcessor:
    """Aísla y procesa hojas de maíz según la estrategia configurada."""

    def __init__(self, config: LeafMaskProcessorConfig | None = None) -> None:
        self.config = config or LeafMaskProcessorConfig()

    def process(
        self,
        image: Image.Image,
        instances: Sequence[LeafInstance],
        source_image: str | None = None,
    ) -> SegmentedLeafProcessingResult:
        original = image_to_rgb(image, self.config.background_value)
        selected, reason, warnings = select_target_leaf(instances, original.size, self.config)

        if selected is None:
            processed = original.copy() if self.config.fallback == FALLBACK_ORIGINAL else None
            return SegmentedLeafProcessingResult(
                original_image=original,
                mask=None,
                masked_image=None,
                processed_image=processed,
                crop_image=None,
                bbox=None,
                confidence=None,
                number_of_instances=len(instances),
                selected_instance=None,
                preprocessing_strategy=self.config.processing_profile,
                fallback_used=True,
                fallback_reason=reason,
                warnings=tuple(warnings),
                source_image=source_image,
            )

        mask = binary_mask_image(selected.mask, expected_size=original.size)
        masked = apply_leaf_mask(original, mask, self.config.background_value)
        bbox = selected.bbox
        masked_crop = crop_leaf_region(masked, bbox)

        if self.config.processing_profile == MASK_BLACK:
            processed = masked.copy()
        elif self.config.processing_profile == BBOX_CROP:
            processed = crop_leaf_region(original, bbox)
        elif self.config.processing_profile == CROP_MASK_BLACK:
            processed = masked_crop.copy()
        elif self.config.processing_profile == CROP_MASK_LETTERBOX:
            processed = letterbox_image(
                masked_crop,
                self.config.target_size,
                padding_value=self.config.background_value,
            ).image
        elif self.config.processing_profile == SQUARE_CROP:
            processed = crop_square_centered(
                original,
                bbox,
                margin_ratio=0.15,
                target_size=self.config.target_size,
            )
        else:
            raise ValueError(f"Perfil no soportado: {self.config.processing_profile}")

        return SegmentedLeafProcessingResult(
            original_image=original,
            mask=mask,
            masked_image=masked,
            processed_image=processed,
            crop_image=masked_crop,
            bbox=bbox,
            confidence=selected.confidence,
            number_of_instances=len(instances),
            selected_instance=selected.source_index,
            preprocessing_strategy=self.config.processing_profile,
            fallback_used=False,
            fallback_reason=None,
            warnings=tuple(warnings),
            source_image=source_image,
        )


def build_comparison_panel(result: SegmentedLeafProcessingResult) -> Image.Image:
    """Construye un panel horizontal: Original | Máscara | Enmascarado | Procesado."""
    orig = image_to_rgb(result.original_image)
    if result.mask is not None:
        mask = result.mask.convert("RGB")
    else:
        mask = Image.new("RGB", orig.size, 0)
    masked = result.masked_image or orig.copy()
    final = result.processed_image or orig.copy()

    sources = [orig, mask, masked, final]
    panel_size = (320, 240)
    canvas = Image.new("RGB", (panel_size[0] * len(sources), panel_size[1]), "black")

    for idx, src in enumerate(sources):
        fitted = ImageOps.contain(image_to_rgb(src), panel_size)
        x_off = idx * panel_size[0] + (panel_size[0] - fitted.width) // 2
        y_off = (panel_size[1] - fitted.height) // 2
        canvas.paste(fitted, (x_off, y_off))

    return canvas
