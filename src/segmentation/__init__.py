"""Módulo de segmentación de hojas de maíz."""

from src.segmentation.detector import MaizeLeafSegmenter
from src.segmentation.geometry import (
    apply_leaf_mask,
    binary_mask_image,
    crop_leaf_region,
    crop_square_centered,
    letterbox_image,
)
from src.segmentation.leaf_processor import (
    BBOX_CROP,
    CROP_MASK_BLACK,
    CROP_MASK_LETTERBOX,
    MASK_BLACK,
    SQUARE_CROP,
    LeafInstance,
    LeafMaskProcessorConfig,
    SegmentedLeafProcessingResult,
    SegmentedLeafProcessor,
    build_comparison_panel,
)

__all__ = [
    "MaizeLeafSegmenter",
    "apply_leaf_mask",
    "binary_mask_image",
    "crop_leaf_region",
    "crop_square_centered",
    "letterbox_image",
    "LeafInstance",
    "LeafMaskProcessorConfig",
    "SegmentedLeafProcessingResult",
    "SegmentedLeafProcessor",
    "build_comparison_panel",
    "MASK_BLACK",
    "BBOX_CROP",
    "CROP_MASK_BLACK",
    "CROP_MASK_LETTERBOX",
    "SQUARE_CROP",
]


