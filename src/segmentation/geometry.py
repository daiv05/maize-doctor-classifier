"""Operaciones geométricas y de máscara para aislamiento de hojas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np
from PIL import Image

BoundingBox: TypeAlias = tuple[int, int, int, int]
RGBColor: TypeAlias = tuple[int, int, int]
MaskInput: TypeAlias = Image.Image | np.ndarray


def normalize_rgb_color(value: int | Sequence[int]) -> RGBColor:
    """Normaliza un entero o secuencia a una tupla RGB (r, g, b)."""
    if isinstance(value, bool):
        raise ValueError("el color RGB no puede ser booleano")
    if isinstance(value, int):
        return (value, value, value)
    values = tuple(value)
    if len(values) != 3:
        raise ValueError("el color RGB debe contener exactamente tres canales")
    return (int(values[0]), int(values[1]), int(values[2]))


def image_to_rgb(image: Image.Image, background: int | Sequence[int] = 0) -> Image.Image:
    """Convierte la imagen a RGB eliminando canales alfa si existen."""
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    background_rgb = normalize_rgb_color(background)
    if "A" in image.getbands() or "transparency" in image.info:
        foreground = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (*background_rgb, 255))
        return Image.alpha_composite(canvas, foreground).convert("RGB")
    return image.convert("RGB") if image.mode != "RGB" else image.copy()


def clip_bbox(
    bbox: Sequence[int | float],
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """Ajusta un bounding box a los límites [0, width] x [0, height]."""
    x1, y1, x2, y2 = bbox[:4]
    clipped = (
        min(max(math.floor(x1), 0), image_width),
        min(max(math.floor(y1), 0), image_height),
        min(max(math.ceil(x2), 0), image_width),
        min(max(math.ceil(y2), 0), image_height),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("bbox queda vacío después de ajustarlo a la imagen")
    return clipped


def crop_leaf_region(
    image: Image.Image,
    bbox: BoundingBox,
    *,
    transparency_background: int | Sequence[int] = 0,
) -> Image.Image:
    """Recorta la región delimitada por el bbox."""
    clipped = clip_bbox(bbox, image.width, image.height)
    rgb = image_to_rgb(image, transparency_background)
    return rgb.crop(clipped)


def validate_target_size(target_size: Sequence[int]) -> tuple[int, int]:
    """Valida el tamaño objetivo (alto, ancho)."""
    values = tuple(target_size)
    if len(values) != 2:
        raise ValueError("target_size debe ser (alto, ancho)")
    height, width = int(values[0]), int(values[1])
    if height <= 0 or width <= 0:
        raise ValueError("alto y ancho deben ser mayores que cero")
    return height, width


@dataclass(frozen=True)
class LetterboxResult:
    image: Image.Image
    target_size: tuple[int, int]
    padding: tuple[int, int, int, int]
    scale: float


def letterbox_image(
    image: Image.Image,
    target_size: Sequence[int],
    *,
    padding_value: int | Sequence[int] = 0,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> LetterboxResult:
    """Redimensiona preservando aspect ratio y centra sobre un canvas con padding."""
    target_height, target_width = validate_target_size(target_size)
    color = normalize_rgb_color(padding_value)
    source = image_to_rgb(image, color)

    scale = min(target_width / source.width, target_height / source.height)
    resized_width = min(target_width, max(1, round(source.width * scale)))
    resized_height = min(target_height, max(1, round(source.height * scale)))

    resized = source.resize((resized_width, resized_height), resample=resample)
    pad_h = target_width - resized_width
    pad_v = target_height - resized_height
    left = pad_h // 2
    top = pad_v // 2

    canvas = Image.new("RGB", (target_width, target_height), color)
    canvas.paste(resized, (left, top))
    return LetterboxResult(
        image=canvas,
        target_size=(target_height, target_width),
        padding=(left, top, pad_h - left, pad_v - top),
        scale=scale,
    )


def binary_mask_image(
    mask: MaskInput,
    *,
    expected_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Convierte un array o imagen de máscara en una PIL.Image en modo 'L' con valores 0 o 255."""
    if isinstance(mask, Image.Image):
        gray = mask.convert("L")
        arr = np.asarray(gray)
    elif isinstance(mask, np.ndarray):
        arr = mask
    else:
        raise TypeError("mask debe ser PIL.Image o numpy.ndarray")

    if arr.ndim == 3:
        arr = arr[:, :, 0]
    threshold = 0.5 if arr.dtype in (np.float32, np.float64) else 127
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    result = Image.fromarray(binary, mode="L")
    if expected_size is not None and result.size != expected_size:
        result = result.resize(expected_size, resample=Image.Resampling.NEAREST)
    return result


def mask_area_ratio(mask: MaskInput) -> float:
    """Calcula la fracción de píxeles activos (1) en la máscara."""
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    else:
        arr = mask
    total = arr.size
    if total == 0:
        return 0.0
    active = np.count_nonzero(arr > 0)
    return float(active / total)


def mask_bbox(mask: MaskInput) -> BoundingBox | None:
    """Calcula la caja delimitadora de los píxeles activos de la máscara."""
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    else:
        arr = mask
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return (int(xmin), int(ymin), int(xmax + 1), int(ymax + 1))


def mask_centroid(mask: MaskInput) -> tuple[float, float] | None:
    """Calcula el centroide (x, y) de la máscara."""
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    else:
        arr = mask
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        return None
    y_mean = float(np.mean(coords[:, 0]))
    x_mean = float(np.mean(coords[:, 1]))
    return (x_mean, y_mean)


def apply_leaf_mask(
    image: Image.Image,
    mask: MaskInput,
    background_value: int | Sequence[int] = (0, 0, 0),
) -> Image.Image:
    """Aplica la máscara binaria a la imagen, reemplazando el fondo por el color de fondo."""
    rgb = image_to_rgb(image)
    binary = binary_mask_image(mask, expected_size=rgb.size)
    bg_color = normalize_rgb_color(background_value)
    bg = Image.new("RGB", rgb.size, bg_color)
    return Image.composite(rgb, bg, binary)


def crop_square_centered(
    image: Image.Image,
    bbox: BoundingBox,
    *,
    margin_ratio: float = 0.15,
    target_size: Sequence[int] | None = None,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Recorta un cuadro 1:1 centrado en la hoja conservando el fondo natural.

    Calcula una ventana cuadrada que cubre la dimensión mayor de la hoja más un
    margen porcentual, desplazando la ventana para no salir de los bordes de la foto
    original (sin inventar píxeles ni introducir fondo negro). Si target_size es
    especificado, escala el resultado manteniendo la geometría cuadrada.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    clipped = clip_bbox(bbox, image.width, image.height)
    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    w_img, h_img = image.size
    max_dim = max(bw, bh)
    side = max_dim * (1.0 + max(0.0, float(margin_ratio)))
    # El cuadrado dentro de la imagen no puede superar la dimensión mínima de la foto
    side = min(side, min(w_img, h_img))
    half = side / 2.0

    sq_x1 = cx - half
    sq_y1 = cy - half
    sq_x2 = cx + half
    sq_y2 = cy + half

    # Ajustar para permanecer dentro de los bordes
    if sq_x1 < 0:
        shift = -sq_x1
        sq_x1 = 0.0
        sq_x2 = min(float(w_img), sq_x2 + shift)
    if sq_y1 < 0:
        shift = -sq_y1
        sq_y1 = 0.0
        sq_y2 = min(float(h_img), sq_y2 + shift)
    if sq_x2 > w_img:
        shift = sq_x2 - float(w_img)
        sq_x2 = float(w_img)
        sq_x1 = max(0.0, sq_x1 - shift)
    if sq_y2 > h_img:
        shift = sq_y2 - float(h_img)
        sq_y2 = float(h_img)
        sq_y1 = max(0.0, sq_y1 - shift)

    crop_coords = (
        int(math.floor(sq_x1)),
        int(math.floor(sq_y1)),
        int(math.ceil(sq_x2)),
        int(math.ceil(sq_y2)),
    )
    crop_coords = clip_bbox(crop_coords, w_img, h_img)
    rgb = image_to_rgb(image)
    cropped = rgb.crop(crop_coords)

    if target_size is not None:
        th, tw = validate_target_size(target_size)
        cropped = cropped.resize((tw, th), resample=resample)
    return cropped

