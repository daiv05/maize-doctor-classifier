import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from scripts.pipeline import create_splits
from src.data.deduplicate import drop_near_duplicates
from src.data.identity import sample_id_for_path
from src.data.preparation import (
    apply_effective_groups,
    build_group_split_summary,
    load_group_manifest,
    validate_split_integrity,
)
from src.data.splitter import SourceGroupedSplitter


def _eligible_manifest(count: int = 6) -> pd.DataFrame:
    paths = [f"clean/healthy/real/sample_{index}.png" for index in range(count)]
    return pd.DataFrame(
        {
            "sample_id": [sample_id_for_path(path) for path in paths],
            "image_path": paths,
            "label": ["healthy"] * count,
            "environment": ["real"] * count,
            "sha256": [
                hashlib.sha256(f"bytes-{index}".encode()).hexdigest() for index in range(count)
            ],
            "source_id": ["source_A"] * count,
        }
    )


@pytest.mark.parametrize("identity_column", ["sample_id", "image_path"])
def test_group_manifest_acepta_ambas_identidades_y_no_divide_grupos(
    tmp_path: Path, identity_column: str
):
    eligible = _eligible_manifest(4)
    values = eligible[identity_column].tolist()
    if identity_column == "image_path":
        values[0] = values[0].replace("/", "\\")
    group_manifest = tmp_path / f"groups-{identity_column}.csv"
    pd.DataFrame(
        {
            identity_column: values,
            "group_id": ["group_1", "group_1", "group_2", "group_2"],
        }
    ).to_csv(group_manifest, index=False)

    explicit = load_group_manifest(group_manifest, eligible)
    grouped = apply_effective_groups(eligible, explicit)
    parts = SourceGroupedSplitter(group_column="effective_group_id", allow_incomplete=True).split(
        grouped, train_size=0.5, val_size=0.25, test_size=0.25
    )

    split_by_group: dict[str, set[str]] = {}
    for split_name, frame in zip(("train", "val", "test"), parts):
        for group_id in frame["effective_group_id"]:
            split_by_group.setdefault(group_id, set()).add(split_name)
    assert set(grouped["effective_group_id"]) == {"group_1", "group_2"}
    assert all(len(split_names) == 1 for split_names in split_by_group.values())


def test_grupo_explicito_tiene_prioridad_y_fallback_conserva_procedencia(tmp_path: Path):
    eligible = _eligible_manifest(2)
    group_manifest = tmp_path / "groups.csv"
    pd.DataFrame([{"sample_id": eligible.iloc[0]["sample_id"], "group_id": "group_custom"}]).to_csv(
        group_manifest, index=False
    )

    grouped = apply_effective_groups(
        eligible,
        load_group_manifest(group_manifest, eligible),
    )

    assert grouped.iloc[0]["source_id"] == "source_A"
    assert grouped.iloc[0]["explicit_group_id"] == "group_custom"
    assert grouped.iloc[0]["effective_group_id"] == "group_custom"
    assert grouped.iloc[0]["group_origin"] == "explicit"
    assert grouped.iloc[1]["effective_group_id"] == "source_A"
    assert grouped.iloc[1]["group_origin"] == "inferred"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            lambda frame: [
                {"sample_id": frame.iloc[0]["sample_id"], "group_id": "group_A"},
                {"sample_id": frame.iloc[0]["sample_id"], "group_id": "group_B"},
            ],
            "dos group_id diferentes",
        ),
        (
            lambda frame: [{"sample_id": frame.iloc[0]["sample_id"], "group_id": "   "}],
            "group_id no puede estar vacío",
        ),
        (
            lambda _frame: [{"sample_id": "0" * 64, "group_id": "group_A"}],
            "muestra no elegible",
        ),
        (
            lambda _frame: [{"sample_id": "abc123", "group_id": "group_A"}],
            "64 hexadecimales",
        ),
        (
            lambda frame: [
                {
                    "sample_id": frame.iloc[0]["sample_id"],
                    "image_path": frame.iloc[1]["image_path"],
                    "group_id": "group_A",
                }
            ],
            "ambiguo",
        ),
    ],
)
def test_group_manifest_rechaza_conflictos_vacios_y_muestras_inexistentes(
    tmp_path: Path, rows, message: str
):
    eligible = _eligible_manifest(2)
    group_manifest = tmp_path / "groups.csv"
    pd.DataFrame(rows(eligible)).to_csv(group_manifest, index=False)

    with pytest.raises(ValueError, match=message):
        load_group_manifest(group_manifest, eligible)


def test_validacion_exige_cero_fuga_de_id_sha_y_grupo():
    master = _eligible_manifest(6)
    explicit = {
        sample_id: f"group_{index // 2}" for index, sample_id in enumerate(master["sample_id"])
    }
    master = apply_effective_groups(master, explicit)
    splits = {
        name: master.iloc[start:stop][["sample_id"]].copy()
        for name, start, stop in (
            ("train", 0, 2),
            ("val", 2, 4),
            ("test", 4, 6),
        )
    }

    metrics = validate_split_integrity(master, splits)

    assert metrics == {
        "sample_overlap_count": 0,
        "sha256_overlap_count": 0,
        "group_overlap_count": 0,
    }


def test_validacion_aborta_ante_fuga_de_grupo():
    master = apply_effective_groups(
        _eligible_manifest(4),
        {
            sample_id: f"group_{index // 2}"
            for index, sample_id in enumerate(_eligible_manifest(4)["sample_id"])
        },
    )
    splits = {
        "train": master.iloc[[0]][["sample_id"]],
        "val": master.iloc[[1]][["sample_id"]],
        "test": master.iloc[[2, 3]][["sample_id"]],
    }

    with pytest.raises(ValueError, match="effective_group_id"):
        validate_split_integrity(master, splits)


def test_validacion_aborta_ante_fuga_de_sha256():
    master = apply_effective_groups(_eligible_manifest(3), {})
    master.loc[1, "sha256"] = master.loc[0, "sha256"]
    splits = {
        "train": master.iloc[[0]][["sample_id"]],
        "val": master.iloc[[1]][["sample_id"]],
        "test": master.iloc[[2]][["sample_id"]],
    }

    with pytest.raises(ValueError, match="sha256"):
        validate_split_integrity(master, splits)


def test_deduplicacion_perceptual_previa_al_split_se_conserva(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (24, 24), (40, 120, 200)).save(first)
    Image.new("RGB", (24, 24), (40, 120, 200)).save(second, quality=85)
    manifest = pd.DataFrame(
        {
            "image_path": [first.name, second.name],
            "label": ["healthy", "healthy"],
        }
    )

    deduplicated, discarded = drop_near_duplicates(
        manifest,
        tmp_path,
        max_distance=0,
        workers=1,
    )

    assert discarded == 1
    assert deduplicated["image_path"].tolist() == [first.name]


def test_splitter_balancea_clases_y_entornos_sin_dividir_grupos():
    rows = []
    for group_index in range(6):
        for label in ("healthy", "common_rust", "gray_leaf_spot"):
            for environment in ("lab", "real"):
                rows.append(
                    {
                        "sample_id": f"{group_index}-{label}-{environment}",
                        "label": label,
                        "environment": environment,
                        "effective_group_id": f"group_{group_index}",
                    }
                )
    manifest = pd.DataFrame(rows)

    parts = SourceGroupedSplitter(group_column="effective_group_id").split(
        manifest, train_size=0.70, val_size=0.15, test_size=0.15
    )

    group_sets = [set(frame["effective_group_id"]) for frame in parts]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert all(set(frame["label"]) == set(manifest["label"]) for frame in parts)
    assert all(set(frame["environment"]) == {"lab", "real"} for frame in parts)


def test_grupo_grande_permanece_intacto_y_desviacion_queda_reportada():
    sizes = {"large": 12, "small_a": 3, "small_b": 3, "small_c": 3}
    rows = [
        {
            "sample_id": f"{group}-{index}",
            "label": "healthy",
            "environment": "real",
            "sha256": hashlib.sha256(f"{group}-{index}".encode()).hexdigest(),
            "source_id": None,
            "explicit_group_id": group,
            "group_id": group,
            "effective_group_id": group,
            "group_origin": "explicit",
        }
        for group, size in sizes.items()
        for index in range(size)
    ]
    master = pd.DataFrame(rows)
    frames = SourceGroupedSplitter(group_column="effective_group_id").split(
        master, train_size=0.70, val_size=0.15, test_size=0.15
    )
    splits = {
        name: frame[["sample_id"]].copy() for name, frame in zip(("train", "val", "test"), frames)
    }
    metrics = validate_split_integrity(master, splits)
    summary = build_group_split_summary(
        master,
        splits,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        metrics,
    )

    large_membership = [
        name
        for name, frame in zip(("train", "val", "test"), frames)
        if "large" in set(frame["effective_group_id"])
    ]
    assert large_membership == ["train"]
    assert summary["actual_train_size"] != summary["target_train_size"]
    assert any(
        row["effective_group_id"] == "large"
        for row in summary["groups_responsible_for_size_deviation"]
    )


def test_strict_mode_reporta_clase_split_muestras_y_grupos():
    manifest = pd.DataFrame(
        [
            {
                "label": "rare",
                "environment": "real",
                "effective_group_id": group,
            }
            for group in ("group_A", "group_B")
        ]
    )

    with pytest.raises(SystemExit) as caught:
        SourceGroupedSplitter(group_column="effective_group_id").split(
            manifest, train_size=0.70, val_size=0.15, test_size=0.15
        )

    message = str(caught.value)
    assert "class='rare'" in message
    assert "split=" in message
    assert "total_samples=2" in message
    assert "independent_groups=2" in message


def _build_pipeline_dataset(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    dataset_root = tmp_path / "dataset"
    sources = ("maize_field", "maize_desease", "cropdg")
    image_paths = []
    for label, offset in (("healthy", 0), ("common_rust", 100)):
        directory = dataset_root / "clean" / label / "real"
        directory.mkdir(parents=True)
        for index in range(12):
            source = sources[index % len(sources)]
            path = directory / f"{label}_{source}_real_{index:02x}.png"
            Image.new("RGB", (8, 8), (offset + index, 30 + index, 200 - index)).save(path)
            image_paths.append(path.relative_to(dataset_root).as_posix())

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
    return dataset_root, config_path, image_paths


def test_pipeline_aplica_group_manifest_parcial_y_audita_cero_fuga(tmp_path, monkeypatch):
    dataset_root, config_path, image_paths = _build_pipeline_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(create_splits, "get_dataset_root", lambda: dataset_root)
    monkeypatch.setattr(create_splits, "get_output_root", lambda: output_root)
    monkeypatch.setattr(create_splits, "_resolve_index_workers", lambda: 1)

    selected = [image_paths[0], image_paths[13]]
    group_manifest = tmp_path / "groups.csv"
    pd.DataFrame(
        {
            "sample_id": [sample_id_for_path(path) for path in selected],
            "group_id": ["plant_custom", "plant_custom"],
        }
    ).to_csv(group_manifest, index=False)

    create_splits.run_data_preparation_pipeline(
        str(config_path),
        group_manifest=group_manifest,
    )

    output = output_root / "splits" / "seed_42"
    master = pd.read_csv(output / "master_manifest.csv")
    explicit = master[master["effective_group_id"].eq("plant_custom")]
    assert set(explicit["sample_id"]) == {sample_id_for_path(path) for path in selected}
    assert set(explicit["group_origin"]) == {"explicit"}

    fallback = master[master["image_path"].eq(image_paths[1])].iloc[0]
    assert fallback["group_origin"] == "inferred"
    assert fallback["effective_group_id"] == fallback["source_id"]

    parts = {name: pd.read_csv(output / f"{name}.csv") for name in ("train", "val", "test")}
    locations = [
        name for name, frame in parts.items() if "plant_custom" in set(frame["effective_group_id"])
    ]
    assert len(locations) == 1

    audit = json.loads((output / "preparation_audit.json").read_text(encoding="utf-8"))
    assert audit["grouping"]["sample_overlap_count"] == 0
    assert audit["grouping"]["sha256_overlap_count"] == 0
    assert audit["grouping"]["group_overlap_count"] == 0
    assert audit["grouping"]["explicit_groups"] == 1
