import pandas as pd
import pytest

from src.data.cross_validation import HierarchicalKFoldSplitter, SourceGroupedKFoldSplitter


@pytest.fixture
def manifiesto():
    """Manifiesto con seis fuentes y clases repartidas de forma desigual entre ellas."""
    reparto = {
        "maize_diseases": ["healthy", "common_rust", "gray_leaf_spot"],
        "cropdg": ["healthy", "northern_corn_leaf_blight"],
        "maize_in_field": ["healthy", "common_rust"],
        "corn_leaf_roboflow": ["gray_leaf_spot", "northern_corn_leaf_blight"],
        "maize_2_roboflow": ["healthy", "gray_leaf_spot"],
        "multicrop_disease_maiz": ["common_rust", "northern_corn_leaf_blight"],
    }
    filas = [
        {
            "image_path": f"clean/{label}/real/{label}_{fuente}_real_{indice}.jpg",
            "label": label,
            "environment": "real",
            "source_id": fuente,
        }
        for fuente, labels in reparto.items()
        for label in labels
        for indice in range(40)
    ]
    return pd.DataFrame(filas)


def test_ninguna_fuente_cruza_la_frontera_de_un_pliegue(manifiesto):
    """La unidad de partición es la fuente, no la imagen."""
    for pliegue in SourceGroupedKFoldSplitter(n_splits=3, seed=42).split(manifiesto):
        assert not set(pliegue.train_df.source_id) & set(pliegue.val_df.source_id)


def test_el_splitter_estratificado_si_deja_cruzar_las_fuentes(manifiesto):
    """Contraste explícito: es la diferencia que justifica el splitter agrupado."""
    pliegues = HierarchicalKFoldSplitter(n_splits=3, seed=42).split(manifiesto)
    solapes = [
        len(set(p.train_df.source_id) & set(p.val_df.source_id)) for p in pliegues
    ]
    assert all(s > 0 for s in solapes)


def test_cada_imagen_aparece_en_validacion_exactamente_una_vez(manifiesto):
    """Los pliegues cubren el manifiesto sin solaparse."""
    vistas = [
        ruta
        for pliegue in SourceGroupedKFoldSplitter(n_splits=3, seed=42).split(manifiesto)
        for ruta in pliegue.val_df.image_path
    ]
    assert sorted(vistas) == sorted(manifiesto.image_path)


def test_la_fuente_se_deriva_de_la_ruta_si_falta_la_columna(manifiesto):
    """El manifiesto de los splits no trae source_id; se infiere de la procedencia."""
    sin_columna = manifiesto.drop(columns=["source_id"])
    pliegues = SourceGroupedKFoldSplitter(n_splits=3, seed=42).split(sin_columna)

    assert len(pliegues) == 3
    for pliegue in pliegues:
        assert not set(pliegue.train_df.source_id) & set(pliegue.val_df.source_id)


def test_un_manifiesto_sin_etiquetas_falla_en_construccion(manifiesto):
    """El error aparece al particionar, no en el primer lote de entrenamiento."""
    with pytest.raises(ValueError, match="label"):
        SourceGroupedKFoldSplitter(n_splits=3, seed=42).split(
            manifiesto.drop(columns=["label"])
        )
