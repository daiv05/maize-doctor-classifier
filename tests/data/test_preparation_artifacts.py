import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from scripts.pipeline import create_splits
from src.data.identity import sample_id_for_path
from src.data.preparation import atomic_write_json, sha256_file

_ARTIFACTS = (
    "master_manifest.csv",
    "train.csv",
    "val.csv",
    "test.csv",
    "split_audit_report.csv",
    "preparation_audit.json",
    "manifest.lock.json",
)


def _build_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    sources = ("maize_field", "maize_desease", "cropdg")
    for label, offset in (("healthy", 0), ("common_rust", 100)):
        directory = dataset_root / "clean" / label / "real"
        directory.mkdir(parents=True)
        for index in range(12):
            filename = f"{label}_{sources[index % len(sources)]}_real_{index:02x}.png"
            Image.new(
                "RGB",
                (8, 8),
                (offset + index, 20 + index, 200 - index),
            ).save(directory / filename)

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"classes": ["healthy", "common_rust"], "seed": 42},
                "paths": {"raw_dir": "clean", "split_output_dir": "splits/seed_42"},
            }
        ),
        encoding="utf-8",
    )
    return dataset_root, config_path


def _configure(monkeypatch, dataset_root: Path, output_root: Path) -> None:
    monkeypatch.setattr(create_splits, "get_dataset_root", lambda: dataset_root)
    monkeypatch.setattr(create_splits, "get_output_root", lambda: output_root)
    monkeypatch.setattr(create_splits, "_resolve_index_workers", lambda: 1)


def _output(output_root: Path) -> Path:
    return output_root / "splits" / "seed_42"


def test_manifest_lock_integridad_y_reproducibilidad(tmp_path, monkeypatch):
    dataset_root, config_path = _build_dataset(tmp_path)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    _configure(monkeypatch, dataset_root, first_root)
    create_splits.run_data_preparation_pipeline(str(config_path))
    _configure(monkeypatch, dataset_root, second_root)
    create_splits.run_data_preparation_pipeline(str(config_path))

    first = _output(first_root)
    second = _output(second_root)
    for artifact in _ARTIFACTS:
        assert (first / artifact).read_bytes() == (second / artifact).read_bytes()

    master = pd.read_csv(first / "master_manifest.csv")
    assert list(master.columns) == [
        "sample_id",
        "image_path",
        "label",
        "environment",
        "sha256",
        "source_id",
        "explicit_group_id",
        "group_id",
        "effective_group_id",
        "group_origin",
    ]
    row = master.iloc[0]
    assert row["sample_id"] == sample_id_for_path(row["image_path"])
    assert row["sha256"] == sha256_file(dataset_root / row["image_path"])
    assert row["source_id"] in {
        "maize-in-field-dataset",
        "maize-diseases",
        "cropdg-unified-multidomain",
    }
    assert row["group_origin"] == "inferred"
    assert row["effective_group_id"] == row["source_id"]

    splits = {name: pd.read_csv(first / f"{name}.csv") for name in ("train", "val", "test")}
    id_sets = {name: set(frame["sample_id"]) for name, frame in splits.items()}
    assert id_sets["train"].isdisjoint(id_sets["val"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["val"].isdisjoint(id_sets["test"])
    assert set().union(*id_sets.values()) == set(master["sample_id"])

    report = pd.read_csv(first / "split_audit_report.csv")
    assert list(report.columns) == [
        "split",
        "label",
        "environment",
        "source_id",
        "explicit_group_id",
        "group_id",
        "effective_group_id",
        "group_origin",
        "count",
    ]
    assert int(report["count"].sum()) == len(master)

    audit = json.loads((first / "preparation_audit.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == 1
    assert audit["samples_discovered"] == 24
    assert audit["samples_valid"] == 24
    assert audit["samples_eligible"] == 24
    assert audit["label_conflicts"] == 0
    assert audit["grouping"]["total_groups"] == 3
    assert audit["grouping"]["group_overlap_count"] == 0
    assert audit["grouping"]["sha256_overlap_count"] == 0

    lock = json.loads((first / "manifest.lock.json").read_text(encoding="utf-8"))
    assert lock["master_manifest_sha256"] == sha256_file(first / "master_manifest.csv")
    for name in ("train", "val", "test"):
        assert lock[f"{name}_sha256"] == sha256_file(first / f"{name}.csv")


def test_duplicado_exacto_de_misma_label_conserva_politica_actual(tmp_path, monkeypatch):
    dataset_root, config_path = _build_dataset(tmp_path)
    original = dataset_root / "clean/healthy/real/healthy_maize_field_real_00.png"
    duplicate = dataset_root / "clean/healthy/real/healthy_maize_field_real_fe.png"
    duplicate.write_bytes(original.read_bytes())
    output_root = tmp_path / "outputs"
    _configure(monkeypatch, dataset_root, output_root)

    create_splits.run_data_preparation_pipeline(str(config_path))

    output = _output(output_root)
    master = pd.read_csv(output / "master_manifest.csv")
    audit = json.loads((output / "preparation_audit.json").read_text(encoding="utf-8"))
    assert len(master) == 24
    assert duplicate.relative_to(dataset_root).as_posix() not in set(master["image_path"])
    assert audit["samples_discovered"] == 25
    assert audit["exact_duplicates"] == 1


def test_group_by_source_conserva_fuentes_disjuntas(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    sources = (
        "maize_field",
        "maize_desease",
        "cropdg",
        "maize_africa",
        "multi_desease",
        "maize_nutrient",
    )
    for label, offset in (("healthy", 0), ("common_rust", 100)):
        directory = dataset_root / "clean" / label / "real"
        directory.mkdir(parents=True)
        for source_index, source in enumerate(sources):
            for image_index in range(2):
                color = offset + source_index * 2 + image_index
                Image.new("RGB", (8, 8), (color, 30, 200 - color)).save(
                    directory / f"{label}_{source}_real_{source_index:x}{image_index:x}.png"
                )

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"classes": ["healthy", "common_rust"], "seed": 42},
                "paths": {"raw_dir": "clean", "split_output_dir": "splits/seed_42"},
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    _configure(monkeypatch, dataset_root, output_root)

    create_splits.run_data_preparation_pipeline(str(config_path), group_by_source=True)

    output = _output(output_root)
    parts = {name: pd.read_csv(output / f"{name}.csv") for name in ("train", "val", "test")}
    sources_by_split = {name: set(frame["source_id"]) for name, frame in parts.items()}
    assert sources_by_split["train"].isdisjoint(sources_by_split["val"])
    assert sources_by_split["train"].isdisjoint(sources_by_split["test"])
    assert sources_by_split["val"].isdisjoint(sources_by_split["test"])
    assert all(set(frame["label"]) == {"healthy", "common_rust"} for frame in parts.values())


def test_contenido_igual_con_labels_distintas_aborta_con_detalle(tmp_path, monkeypatch):
    dataset_root, config_path = _build_dataset(tmp_path)
    healthy = dataset_root / "clean/healthy/real/healthy_maize_field_real_00.png"
    conflict = dataset_root / "clean/common_rust/real/common_rust_maize_field_real_fe.png"
    conflict.write_bytes(healthy.read_bytes())
    output_root = tmp_path / "outputs"
    _configure(monkeypatch, dataset_root, output_root)

    with pytest.raises(ValueError, match="etiquetas conflictivas") as caught:
        create_splits.run_data_preparation_pipeline(str(config_path))

    message = str(caught.value)
    digest = sha256_file(healthy)
    for path, label in (
        (healthy.relative_to(dataset_root).as_posix(), "healthy"),
        (conflict.relative_to(dataset_root).as_posix(), "common_rust"),
    ):
        assert path in message
        assert label in message
        assert sample_id_for_path(path) in message
    assert digest in message
    audit_path = _output(output_root) / "preparation_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "failed_label_conflicts"
    assert audit["label_conflicts"] == 1


def test_exclusion_valida_es_logica_y_verificada(tmp_path, monkeypatch):
    dataset_root, config_path = _build_dataset(tmp_path)
    target = dataset_root / "clean/healthy/real/healthy_maize_field_real_00.png"
    relative = target.relative_to(dataset_root).as_posix()
    original_bytes = target.read_bytes()
    exclusions = tmp_path / "exclusions.csv"
    pd.DataFrame(
        [{"image_path": relative, "sha256": sha256_file(target), "reason": "revisión manual"}]
    ).to_csv(exclusions, index=False)
    output_root = tmp_path / "outputs"
    _configure(monkeypatch, dataset_root, output_root)

    create_splits.run_data_preparation_pipeline(str(config_path), exclusions=exclusions)

    output = _output(output_root)
    master = pd.read_csv(output / "master_manifest.csv")
    audit = json.loads((output / "preparation_audit.json").read_text(encoding="utf-8"))
    assert relative not in set(master["image_path"])
    assert target.read_bytes() == original_bytes
    assert audit["samples_excluded"] == 1
    assert audit["exclusions_by_reason"] == {"revisión manual": 1}


def test_exclusion_falla_si_el_contenido_cambio(tmp_path, monkeypatch):
    dataset_root, config_path = _build_dataset(tmp_path)
    target = dataset_root / "clean/healthy/real/healthy_maize_field_real_00.png"
    relative = target.relative_to(dataset_root).as_posix()
    exclusions = tmp_path / "exclusions.csv"
    pd.DataFrame(
        [{"image_path": relative, "sha256": sha256_file(target), "reason": "revisión manual"}]
    ).to_csv(exclusions, index=False)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(target)
    _configure(monkeypatch, dataset_root, tmp_path / "outputs")

    with pytest.raises(ValueError, match="contenido de la exclusión cambió"):
        create_splits.run_data_preparation_pipeline(str(config_path), exclusions=exclusions)


def test_exclusion_requiere_reason(tmp_path, monkeypatch):
    dataset_root, config_path = _build_dataset(tmp_path)
    target = dataset_root / "clean/healthy/real/healthy_maize_field_real_00.png"
    exclusions = tmp_path / "exclusions.csv"
    pd.DataFrame(
        [
            {
                "image_path": target.relative_to(dataset_root).as_posix(),
                "sha256": sha256_file(target),
                "reason": "   ",
            }
        ]
    ).to_csv(exclusions, index=False)
    _configure(monkeypatch, dataset_root, tmp_path / "outputs")

    with pytest.raises(ValueError, match="reason no puede estar vacío"):
        create_splits.run_data_preparation_pipeline(str(config_path), exclusions=exclusions)


def test_atomic_write_json_no_deja_destino_parcial(tmp_path, monkeypatch):
    destination = tmp_path / "audit.json"
    atomic_write_json(destination, {"b": 2, "a": "á"})
    assert destination.read_text(encoding="utf-8") == '{\n  "a": "á",\n  "b": 2\n}\n'

    destination.write_text("contenido-anterior", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("fallo simulado")

    monkeypatch.setattr("src.data.preparation.os.replace", fail_replace)
    with pytest.raises(OSError, match="fallo simulado"):
        atomic_write_json(destination, {"nuevo": True})

    assert destination.read_text(encoding="utf-8") == "contenido-anterior"
    assert not list(tmp_path.glob(".audit.json.*.tmp"))


def test_sha256_registrado_corresponde_a_bytes_conocidos(tmp_path):
    path = tmp_path / "known.bin"
    content = b"doctor-maiz\n"
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()
