import hashlib

import pandas as pd
import pytest

import src.data.identity as identity
from src.data.identity import (
    ensure_sample_ids,
    normalize_sample_path,
    sample_id_for_path,
    validate_sample_ids,
)


def test_identidad_es_determinista_y_usa_utf8():
    path = "clean/healthy/real/hoja_á.jpg"
    expected = hashlib.sha256(path.encode("utf-8")).hexdigest()

    assert sample_id_for_path(path) == expected
    assert sample_id_for_path(path) == sample_id_for_path(path)


def test_rutas_diferentes_producen_ids_diferentes():
    assert sample_id_for_path("a/img.jpg") != sample_id_for_path("b/img.jpg")


def test_separadores_y_componentes_punto_se_normalizan():
    windows_style = r"data\foo\.\bar.jpg"
    posix_style = "data/foo/bar.jpg"

    assert normalize_sample_path(windows_style) == posix_style
    assert sample_id_for_path(windows_style) == sample_id_for_path(posix_style)


@pytest.mark.parametrize(
    "path",
    [
        "/data/foo.jpg",
        r"C:\project\data\foo.jpg",
        r"\\server\share\foo.jpg",
    ],
)
def test_ruta_absoluta_o_con_unidad_falla(path):
    with pytest.raises(ValueError, match="relativo"):
        normalize_sample_path(path)


@pytest.mark.parametrize("path", ["../foo.jpg", "data/../foo.jpg", "foo/../../bar.jpg"])
def test_traversal_falla(path):
    with pytest.raises(ValueError, match=r"\.\."):
        normalize_sample_path(path)


@pytest.mark.parametrize("path", ["", "   ", ".", "./", ".//./"])
def test_ruta_vacia_o_normalizada_a_punto_falla(path):
    with pytest.raises(ValueError):
        normalize_sample_path(path)


def test_split_antiguo_recibe_ids_en_memoria_sin_mutar_entrada():
    legacy = pd.DataFrame(
        {
            "image_path": ["clean/healthy/real/a.jpg", "clean/rust/lab/b.jpg"],
            "label": ["healthy", "common_rust"],
        }
    )

    upgraded = ensure_sample_ids(legacy)

    assert "sample_id" not in legacy.columns
    assert upgraded["sample_id"].tolist() == [
        sample_id_for_path(path) for path in legacy["image_path"]
    ]


def test_sample_id_existente_incorrecto_falla():
    frame = pd.DataFrame({"image_path": ["a.jpg"], "sample_id": ["incorrecto"]})

    with pytest.raises(ValueError, match="no corresponde"):
        ensure_sample_ids(frame)


def test_sample_id_vacio_falla():
    frame = pd.DataFrame({"image_path": ["a.jpg"], "sample_id": [""]})

    with pytest.raises(ValueError, match="vacío"):
        validate_sample_ids(frame)


def test_sample_id_duplicado_falla():
    path = "a.jpg"
    sample_id = sample_id_for_path(path)
    frame = pd.DataFrame({"image_path": [path, path], "sample_id": [sample_id, sample_id]})

    with pytest.raises(ValueError, match="duplicado"):
        validate_sample_ids(frame)


def test_colision_entre_rutas_diferentes_falla(monkeypatch):
    monkeypatch.setattr(identity, "sample_id_for_path", lambda _path: "same-id")
    frame = pd.DataFrame({"image_path": ["a.jpg", "b.jpg"], "sample_id": ["same-id", "same-id"]})

    with pytest.raises(ValueError, match="Dos rutas diferentes"):
        validate_sample_ids(frame)
