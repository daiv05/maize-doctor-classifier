import hashlib
import json
from pathlib import Path

import pytest

from scripts.pipeline.sync_mobile_model import sync_mobile_model
from src.data.transforms import CornTransformFactory
from src.provenance import atomic_json, contract_hash, sha256_file


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "outputs" / "main" / "shufflenet_v2_x1_0" / "20260816_120000"
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True)

    model_bytes = b"modelo tflite de prueba"
    (export_dir / "model_int8.tflite").write_bytes(model_bytes)
    labels_payload = {
        "schema_version": 1,
        "model": "shufflenet_v2_x1_0",
        "image_size": [224, 224],
        "labels": ["common_rust", "healthy"],
    }
    (export_dir / "labels.json").write_text(json.dumps(labels_payload))

    summary_payload = {
        "run_id": run_dir.name,
        "model": "shufflenet_v2_x1_0",
        "exported_at": "2026-08-16T12:00:00",
        "quantize": "int8",
        "library_versions": {},
        "formats": [
            {
                "format": "tflite",
                "output_path": "export/model_int8.tflite",
                "succeeded": True,
                "error": None,
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "parity": {"passed": True, "n_samples": 30},
                "feature_parity": {"passed": True, "feature_dim": 16, "kind": "pooled_pre_head"},
            }
        ],
    }
    (run_dir / "best.pth").write_bytes(b"fake checkpoint for bundle contract unit tests")
    preproc = CornTransformFactory(target_size=(224, 224)).to_contract()
    atomic_json(export_dir / "preprocessing.json", preproc)
    checkpoint_hash = sha256_file(run_dir / "best.pth")
    training = {
        "model": "shufflenet_v2_x1_0",
        "class_to_idx": {"common_rust": 0, "healthy": 1},
        "preprocessing": preproc,
        "checkpoint_sha256": checkpoint_hash,
        "split_sha256": {"test": "test-manifest-fixture-hash"},
    }
    atomic_json(run_dir / "summary.json", training)
    summary_payload.update(
        {
            "checkpoint_sha256": checkpoint_hash,
            "preprocessing_id": contract_hash(preproc),
            "training_summary_sha256": sha256_file(run_dir / "summary.json"),
            "asset_hashes": {
                name: sha256_file(export_dir / name)
                for name in ("labels.json", "preprocessing.json")
            },
        }
    )
    atomic_json(export_dir / "export_summary_int8.json", summary_payload)
    (export_dir / "eval_tflite_int8_predictions.csv").write_text("sample_id,pred\na,0\nb,1\n")
    atomic_json(
        export_dir / "eval_tflite_int8.json",
        {
            "run_id": run_dir.name,
            "macro_f1_delta": -0.005,
            "n_samples": 2,
            "expected_samples": 2,
            "sample_ids_hash": contract_hash(["a", "b"]),
            "evaluated_split_sha256": training["split_sha256"]["test"],
            "model_sha256": summary_payload["formats"][0]["sha256"],
            "checkpoint_sha256": checkpoint_hash,
            "preprocessing_id": contract_hash(preproc),
            "predictions_sha256": sha256_file(export_dir / "eval_tflite_int8_predictions.csv"),
        },
    )
    atomic_json(
        export_dir / "ood_stats.json",
        {
            "run_id": run_dir.name,
            "checkpoint_sha256": checkpoint_hash,
            "preprocessing_id": contract_hash(preproc),
            "labels": labels_payload["labels"],
            "feature_dim": 16,
            "l2_normalized": True,
        },
    )
    return run_dir


def test_sync_mobile_model_copia_y_escribe_manifest(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    dest_dir = tmp_path / "app_assets"

    manifest_path = sync_mobile_model(run_dir, dest_dir, fmt="tflite", quantize="int8")

    assert (dest_dir / "model_int8.tflite").read_bytes() == b"modelo tflite de prueba"
    assert (dest_dir / "labels.json").exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "20260816_120000"
    assert manifest["model"] == "shufflenet_v2_x1_0"
    assert manifest["format"] == "tflite"
    assert manifest["quantize"] == "int8"
    assert manifest["sha256"] == hashlib.sha256(b"modelo tflite de prueba").hexdigest()


def test_sync_mobile_model_detecta_hash_incorrecto(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    summary_path = run_dir / "export" / "export_summary_int8.json"
    summary = json.loads(summary_path.read_text())
    summary["formats"][0]["sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary))
    dest_dir = tmp_path / "app_assets"

    with pytest.raises(ValueError, match="no coincide"):
        sync_mobile_model(run_dir, dest_dir, fmt="tflite", quantize="int8")


def test_sync_mobile_model_formato_ausente_en_summary(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    dest_dir = tmp_path / "app_assets"

    with pytest.raises(ValueError, match="No se encontro"):
        sync_mobile_model(run_dir, dest_dir, fmt="onnx", quantize="int8")


@pytest.mark.parametrize("case", ["parity", "drop", "missing_ood", "stale_ood", "labels"])
def test_preflight_preserves_previous_bundle(tmp_path, case):
    run = _make_run_dir(tmp_path)
    dest = tmp_path / "assets"
    dest.mkdir()
    (dest / "previous.tflite").write_bytes(b"previous")
    export = run / "export"
    if case == "parity":
        path = export / "export_summary_int8.json"
        data = json.loads(path.read_text())
        data["formats"][0]["parity"] = None
        atomic_json(path, data)
    elif case == "drop":
        path = export / "eval_tflite_int8.json"
        data = json.loads(path.read_text())
        data["macro_f1_delta"] = -0.011
        atomic_json(path, data)
    elif case == "missing_ood":
        (export / "ood_stats.json").unlink()
    elif case == "stale_ood":
        path = export / "ood_stats.json"
        data = json.loads(path.read_text())
        data["run_id"] = "old"
        atomic_json(path, data)
    else:
        (export / "labels.json").write_text("{}")
    with pytest.raises(ValueError):
        sync_mobile_model(run, dest)
    assert list(dest.iterdir()) == [dest / "previous.tflite"]
    assert (dest / "previous.tflite").read_bytes() == b"previous"


def test_commit_failure_rolls_back(tmp_path, monkeypatch):
    import os

    run = _make_run_dir(tmp_path)
    dest = tmp_path / "assets"
    dest.mkdir()
    (dest / "previous").write_text("keep")
    replace = os.replace

    def failing_replace(source, target):
        if "-stage-" in Path(source).name and Path(target) == dest:
            raise OSError("simulated activation failure")
        return replace(source, target)

    monkeypatch.setattr("scripts.pipeline.sync_mobile_model.os.replace", failing_replace)
    with pytest.raises(OSError, match="activation"):
        sync_mobile_model(run, dest)
    assert (dest / "previous").read_text() == "keep"
