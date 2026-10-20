import json
import os
from pathlib import Path

import pytest
import torch

from src.data.preparation import atomic_write_json
from src.data.transforms import CornTransformFactory
from src.training.hyperparameters import load_best_params
from src.training.runs import (
    CheckpointIntegrityError,
    EnsembleCompatibilityError,
    LegacyRunError,
    RunCompatibilityError,
    build_run_contract,
    compute_config_sha256,
    ensemble_member_entry,
    load_ensemble_manifest,
    load_validated_run,
    migrate_legacy_run,
    resolve_run_dir,
    validate_ensemble_runs,
    validate_run_contract,
    write_latest_pointer,
    write_run_contract,
)


def _hyperparameters(**overrides):
    payload = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "none",
        "epochs": 3,
        "patience": None,
        "dropout": None,
    }
    payload.update(overrides)
    return payload


def _splits(root: Path) -> Path:
    directory = root / "splits" / "seed_42"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        (directory / f"{name}.csv").write_text(
            f"sample_id,image_path,label\n{name},clean/a/{name}.png,a\n",
            encoding="utf-8",
        )
    atomic_write_json(directory / "manifest.lock.json", {"schema_version": 1, "seed": 42})
    return directory


def _make_run(
    root: Path,
    *,
    run_id: str = "r1",
    model_name: str = "tiny",
    mapping: dict[str, int] | None = None,
    target_size: tuple[int, int] = (16, 16),
) -> tuple[Path, dict]:
    mapping = mapping or {"a": 0, "b": 1}
    run_dir = root / model_name / run_id
    run_dir.mkdir(parents=True)
    torch.save(torch.nn.Linear(2, len(mapping)).state_dict(), run_dir / "best.pth")
    factory = CornTransformFactory(target_size=target_size)
    contract = build_run_contract(
        run_dir=run_dir,
        model_name=model_name,
        seed=42,
        hyperparameters=_hyperparameters(),
        preprocessing=factory.to_contract(),
        training_preprocessing=factory.training_contract(),
        class_to_idx=mapping,
        splits_dir=_splits(root),
        best_epoch=2,
        metrics={"best_validation": {"epoch": 2, "macro_f1": 0.8}, "test": {}},
    )
    write_run_contract(run_dir, contract)
    return run_dir, contract


def _linear_builder(_name, num_classes, pretrained=False):
    del pretrained
    return torch.nn.Linear(2, num_classes)


def test_summary_v1_contiene_todos_los_campos_contractuales(tmp_path):
    run_dir, contract = _make_run(tmp_path)

    required = {
        "schema_version",
        "run_id",
        "model",
        "seed",
        "hyperparameters",
        "preprocessing",
        "class_to_idx",
        "config_sha256",
        "split_manifest_sha256",
        "splits_dir",
        "best_epoch",
        "metrics",
        "checkpoint",
        "checkpoint_sha256",
    }
    assert required <= contract.keys()
    assert json.loads((run_dir / "summary.json").read_text()) == contract


def test_checkpoint_correcto_pasa_y_modificado_falla_antes_de_cargar(tmp_path, monkeypatch):
    run_dir, _ = _make_run(tmp_path)
    monkeypatch.setattr("src.training.runs.build_model", _linear_builder)

    loaded = load_validated_run(run_dir / "best.pth", expected_model="tiny")
    assert isinstance(loaded.model, torch.nn.Linear)

    with (run_dir / "best.pth").open("ab") as handle:
        handle.write(b"alterado")
    monkeypatch.setattr(
        "src.training.runs.build_model",
        lambda *args, **kwargs: pytest.fail("no debe construir el modelo tras un SHA inválido"),
    )
    with pytest.raises(CheckpointIntegrityError, match="checkpoint_sha256 mismatch"):
        load_validated_run(run_dir / "best.pth", expected_model="tiny")


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"expected_model": "otro"}, "model"),
        ({"expected_input_size": (32, 32)}, "input_size"),
        ({"expected_num_classes": 3}, "num_classes"),
        ({"expected_class_to_idx": {"b": 0, "a": 1}}, "class_to_idx"),
    ],
)
def test_incompatibilidades_se_rechazan_antes_de_inferencia(tmp_path, kwargs, field):
    run_dir, _ = _make_run(tmp_path)
    with pytest.raises(RunCompatibilityError, match=field):
        validate_run_contract(run_dir, **kwargs)


def test_preprocessing_incompatible_se_rechaza(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    other = CornTransformFactory(target_size=(32, 32)).to_contract()
    with pytest.raises(RunCompatibilityError, match="preprocessing"):
        validate_run_contract(run_dir, expected_preprocessing=other)


def test_config_sha256_es_determinista_y_sensible_a_cambios():
    factory = CornTransformFactory(target_size=(16, 16))
    kwargs = {
        "model_name": "tiny",
        "input_size": (16, 16),
        "num_classes": 2,
        "seed": 42,
        "hyperparameters": _hyperparameters(),
        "preprocessing": factory.to_contract(),
        "class_to_idx": {"a": 0, "b": 1},
    }
    first = compute_config_sha256(**kwargs)
    reordered = compute_config_sha256(**{**kwargs, "class_to_idx": {"b": 1, "a": 0}})
    changed = compute_config_sha256(
        **{**kwargs, "hyperparameters": _hyperparameters(learning_rate=2e-4)}
    )
    assert first == reordered
    assert first != changed


def test_run_explicito_tiene_prioridad_y_latest_no_depende_de_mtime(tmp_path):
    model_dir = tmp_path / "tiny"
    first = model_dir / "r1"
    second = model_dir / "r2"
    first.mkdir(parents=True)
    second.mkdir()
    write_latest_pointer(tmp_path, "tiny", "r1")
    os.utime(first, (1, 1))
    os.utime(second, (2_000_000_000, 2_000_000_000))

    assert resolve_run_dir(tmp_path, "tiny") == first
    assert resolve_run_dir(tmp_path, "tiny", "r2") == second
    latest = json.loads((model_dir / "latest.json").read_text())
    assert latest == {"run_id": "r1", "schema_version": 1}
    assert not list(model_dir.glob(".latest.json.*.tmp"))


def test_atomic_write_json_usa_replace_y_no_deja_temporales(tmp_path, monkeypatch):
    destination = tmp_path / "contract.json"
    real_replace = os.replace
    replacements = []

    def tracked_replace(source, target):
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr("src.data.preparation.os.replace", tracked_replace)
    atomic_write_json(destination, {"schema_version": 1, "run_id": "r1"})

    assert replacements and replacements[0][1] == destination
    assert json.loads(destination.read_text()) == {"schema_version": 1, "run_id": "r1"}
    assert not list(tmp_path.glob(".contract.json.*.tmp"))


def test_migracion_legacy_explicita_y_rechazo_si_falta_metadata(tmp_path):
    splits = _splits(tmp_path)
    run_dir = tmp_path / "tiny" / "legacy-ok"
    run_dir.mkdir(parents=True)
    torch.save(torch.nn.Linear(2, 2).state_dict(), run_dir / "best.pth")
    legacy = {
        "model": "tiny",
        "class_to_idx": {"a": 0, "b": 1},
        "best_epoch": 1,
        "best_val_macro_f1": 0.7,
        "test": {"macro_f1": 0.6},
        "splits_dir": str(splits),
    }
    (run_dir / "summary.json").write_text(json.dumps(legacy), encoding="utf-8")
    contract = migrate_legacy_run(
        run_dir,
        preprocessing=CornTransformFactory(target_size=(16, 16)).to_contract(),
        hyperparameters=_hyperparameters(),
        seed=42,
    )
    assert contract["migration"]["verified"] is True
    assert validate_run_contract(run_dir).summary["run_id"] == "legacy-ok"

    incomplete = tmp_path / "tiny" / "legacy-bad"
    incomplete.mkdir()
    torch.save(torch.nn.Linear(2, 2).state_dict(), incomplete / "best.pth")
    (incomplete / "summary.json").write_text(json.dumps({"model": "tiny"}))
    with pytest.raises(LegacyRunError, match="no se puede reconstruir"):
        migrate_legacy_run(incomplete)


def test_ensemble_compatible_y_politicas_de_incompatibilidad(tmp_path, monkeypatch):
    first_dir, _ = _make_run(tmp_path / "one", run_id="a")
    second_dir, _ = _make_run(tmp_path / "two", run_id="b")
    monkeypatch.setattr("src.training.runs.build_model", _linear_builder)
    first = load_validated_run(first_dir / "best.pth", expected_model="tiny")
    second = load_validated_run(second_dir / "best.pth", expected_model="tiny")
    validate_ensemble_runs([first, second], policy="normal", shared_input=True)

    reordered_dir, _ = _make_run(tmp_path / "three", run_id="c", mapping={"b": 0, "a": 1})
    reordered = load_validated_run(reordered_dir / "best.pth", expected_model="tiny")
    with pytest.raises(EnsembleCompatibilityError, match="class_to_idx"):
        validate_ensemble_runs([first, reordered])

    resized_dir, _ = _make_run(tmp_path / "four", run_id="d", target_size=(32, 32))
    resized = load_validated_run(resized_dir / "best.pth", expected_model="tiny")
    with pytest.raises(EnsembleCompatibilityError, match="preprocessing"):
        validate_ensemble_runs([first, resized])

    second.summary["split_manifest_sha256"] = "1" * 64
    with pytest.raises(EnsembleCompatibilityError, match="mismo manifiesto"):
        validate_ensemble_runs([first, second], policy="normal")
    validate_ensemble_runs([first, second], policy="cross_validation")
    validate_ensemble_runs([first, second], policy="loso")


def test_ensemble_con_checkpoint_alterado_aborta_completo(tmp_path, monkeypatch):
    first_dir, _ = _make_run(tmp_path / "one", run_id="a")
    second_dir, _ = _make_run(tmp_path / "two", run_id="b")
    monkeypatch.setattr("src.training.runs.build_model", _linear_builder)
    runs = [
        load_validated_run(first_dir / "best.pth", expected_model="tiny"),
        load_validated_run(second_dir / "best.pth", expected_model="tiny"),
    ]
    manifest_path = tmp_path / "ensemble" / "ensemble_summary.json"
    manifest_path.parent.mkdir()
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "policy": "normal",
            "members": [ensemble_member_entry(run, 0.5) for run in runs],
        },
    )
    with (second_dir / "best.pth").open("ab") as handle:
        handle.write(b"alterado")

    with pytest.raises(CheckpointIntegrityError):
        load_ensemble_manifest(manifest_path)


@pytest.mark.parametrize(
    "params",
    [
        {"unknown_magic_parameter": 1},
        {"batch_size": 32.5},
        {"learning_rate": -0.001},
        {"clahe": "false"},
    ],
)
def test_best_params_rechaza_claves_tipos_y_rangos_invalidos(tmp_path, params):
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps({"best_params": params}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_best_params(path)
