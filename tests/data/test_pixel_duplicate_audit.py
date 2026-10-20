from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.experiments.pixel_duplicate_audit import (
    INVENTORY_COLUMNS,
    _collect_pixel_rows,
    _excluded_source_ids,
    _preflight,
    _write_csv,
    duplicate_groups,
    label_conflict_groups,
    pixel_signature,
    sha256_file,
)


def _save(image: Image.Image, path: Path, **kwargs: object) -> Path:
    image.save(path, **kwargs)
    return path


def test_misma_matriz_recodificada_conserva_pixel_hash(tmp_path: Path) -> None:
    image = Image.new("RGB", (7, 5), (12, 80, 190))
    png = _save(image, tmp_path / "image.png")
    bmp = _save(image, tmp_path / "image.bmp")

    assert sha256_file(png) != sha256_file(bmp)
    assert pixel_signature(png)["pixel_sha256"] == pixel_signature(bmp)["pixel_sha256"]


def test_un_pixel_diferente_cambia_hash(tmp_path: Path) -> None:
    left = Image.new("RGB", (4, 4), (0, 0, 0))
    right = left.copy()
    right.putpixel((3, 2), (0, 0, 1))

    assert (
        pixel_signature(_save(left, tmp_path / "left.png"))["pixel_sha256"]
        != pixel_signature(_save(right, tmp_path / "right.png"))["pixel_sha256"]
    )


def test_dimensiones_forman_parte_del_hash(tmp_path: Path) -> None:
    wide = Image.new("RGB", (2, 4), (1, 2, 3))
    tall = Image.new("RGB", (4, 2), (1, 2, 3))

    wide_sig = pixel_signature(_save(wide, tmp_path / "wide.png"))
    tall_sig = pixel_signature(_save(tall, tmp_path / "tall.png"))
    assert (wide_sig["width"], wide_sig["height"]) == (2, 4)
    assert (tall_sig["width"], tall_sig["height"]) == (4, 2)
    assert wide_sig["pixel_sha256"] != tall_sig["pixel_sha256"]


def test_grayscale_y_rgba_se_normalizan_a_rgb_ignorando_alpha(tmp_path: Path) -> None:
    gray = Image.new("L", (3, 2), 96)
    rgba = Image.new("RGBA", (3, 2), (96, 96, 96, 0))

    gray_sig = pixel_signature(_save(gray, tmp_path / "gray.png"))
    rgba_sig = pixel_signature(_save(rgba, tmp_path / "rgba.png"))
    assert gray_sig["image_mode"] == "L"
    assert rgba_sig["image_mode"] == "RGBA"
    assert gray_sig["channels"] == rgba_sig["channels"] == 3
    assert gray_sig["pixel_sha256"] == rgba_sig["pixel_sha256"]


def test_orientacion_exif_se_aplica_antes_del_hash(tmp_path: Path) -> None:
    raw = Image.new("RGB", (2, 1))
    raw.putpixel((0, 0), (255, 0, 0))
    raw.putpixel((1, 0), (0, 255, 0))
    exif = Image.Exif()
    exif[274] = 6  # orientar 90° en sentido horario
    oriented_file = tmp_path / "exif.jpg"
    raw.save(oriented_file, format="PNG", exif=exif)

    physical = raw.transpose(Image.Transpose.ROTATE_270)
    physical_file = _save(physical, tmp_path / "physical.png")
    actual = pixel_signature(oriented_file)
    expected = pixel_signature(physical_file)
    assert (actual["width"], actual["height"]) == (1, 2)
    assert actual["pixel_sha256"] == expected["pixel_sha256"]


def test_agrupacion_cross_split_y_conflicto_de_label() -> None:
    rows = [
        {"pixel_sha256": "same", "sample_id": "a", "split": "train", "label": "healthy"},
        {"pixel_sha256": "same", "sample_id": "b", "split": "test", "label": "disease"},
        {"pixel_sha256": "solo", "sample_id": "c", "split": "excluded", "label": "other"},
    ]

    groups = duplicate_groups(rows)
    assert len(groups) == 1
    assert {row["split"] for row in groups[0]} == {"train", "test"}
    assert label_conflict_groups(groups) == groups
    assert duplicate_groups(rows, eligible_only=False)[0] == [rows[0], rows[1]]


def test_exclusion_recupera_fuente_y_extension_del_manifest(tmp_path: Path) -> None:
    eligible_path = "clean/healthy/real/healthy_maize_africa_v1.2_real_1.png"
    excluded_path = "clean/healthy/real/healthy_maize_africa_v1.2_real_2.png"
    for name in (eligible_path, excluded_path):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _save(Image.new("RGB", (2, 2), (12, 34, 56)), path)
    master = [
        {
            "sample_id": "eligible",
            "image_path": eligible_path,
            "label": "healthy",
            "source_id": "maize-beans-tomatoes-africa",
            "sha256": sha256_file(tmp_path / eligible_path),
        }
    ]
    excluded = [
        {
            "sample_id": "excluded",
            "image_path": excluded_path,
            "label": "healthy",
            "sha256": sha256_file(tmp_path / excluded_path),
            "reason": "contractual",
        }
    ]

    preflight = _preflight(master, {"eligible": "train"}, excluded, tmp_path)
    assert all(row["status"] == "ok" for row in preflight)
    assert preflight[1]["source_id"] == "maize-beans-tomatoes-africa"
    assert preflight[1]["source_resolution"] == "unique_filename_family_match"
    assert preflight[1]["file_extension"] == ".png"

    output_dir = tmp_path / "evidence"
    old = {
        **preflight[1],
        "source_id": "",
        "file_extension": "",
        "pixel_sha256": "cached-pixel-hash",
        "error": "",
    }
    _write_csv(
        output_dir / "pixel_excluded_images.csv",
        [old],
        (*INVENTORY_COLUMNS, "exclusion_status", "error"),
    )
    _, refreshed = _collect_pixel_rows([], [preflight[1]], tmp_path, output_dir, 1, 10)
    assert refreshed[0]["source_id"] == "maize-beans-tomatoes-africa"
    assert refreshed[0]["file_extension"] == ".png"


def test_exclusion_con_fuente_ambigua_falla() -> None:
    master = [
        {"image_path": f"clean/healthy/real/healthy_same_real_{i}.jpg", "source_id": source}
        for i, source in enumerate(("source-a", "source-b"))
    ]
    excluded = [{"image_path": "clean/healthy/real/healthy_same_real_3.jpg"}]
    with pytest.raises(ValueError, match="source_id único"):
        _excluded_source_ids(master, excluded)
