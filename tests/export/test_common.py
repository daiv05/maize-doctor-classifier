import builtins
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from src.export.common import (
    ExportDependencyError,
    ExportFormatResult,
    ExportIntegrityError,
    ExportReport,
    parse_export_formats,
    resolve_export_inputs,
    validate_export_artifact,
    write_export_summary,
    write_labels_json,
)
from src.export.parity import ParityResult
from src.training.runs import LegacyRunError


def test_parse_export_formats_csv():
    assert parse_export_formats("onnx,tflite") == ["onnx", "tflite"]


def test_parse_export_formats_espacios_y_mayusculas():
    assert parse_export_formats(" ONNX , tflite ") == ["onnx", "tflite"]


def test_parse_export_formats_vacio():
    assert parse_export_formats(None) == []
    assert parse_export_formats("") == []


def test_parse_export_formats_desconocido():
    with pytest.raises(SystemExit):
        parse_export_formats("onnx,coreml")


def test_parse_export_formats_deduplica():
    assert parse_export_formats("onnx,onnx,tflite") == ["onnx", "tflite"]


def test_resolve_export_inputs_sin_summary(tmp_path):
    with pytest.raises(LegacyRunError):
        resolve_export_inputs(tmp_path, "shufflenet_v2_x1_0", tmp_path / "dataset.yaml")


def test_resolve_export_inputs_rechaza_summary_legacy(tmp_path):
    summary = {
        "class_to_idx": {"healthy": 0, "common_rust": 1},
        "image_size": [224, 224],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    with pytest.raises(LegacyRunError, match="Legacy run"):
        resolve_export_inputs(tmp_path, "shufflenet_v2_x1_0", tmp_path / "dataset.yaml")


def test_write_export_summary_crea_export_dir(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    model_path = export_dir / "model.onnx"
    model_path.write_bytes(b"dummy model content")

    parity = ParityResult(
        format="onnx",
        n_samples=30,
        torch_top1_accuracy=0.9,
        exported_top1_accuracy=0.9,
        agreement_rate=1.0,
        max_abs_prob_diff=0.0001,
        mean_abs_prob_diff=0.00001,
        tolerance=1e-3,
        passed=True,
    )
    report = ExportReport(
        run_dir=tmp_path,
        model_name="shufflenet_v2_x1_0",
        formats=[
            ExportFormatResult(
                format="onnx",
                output_path=model_path,
                succeeded=True,
                parity=parity,
            )
        ],
        library_versions={"torch": "2.12.1"},
    )

    write_export_summary(tmp_path, report)

    payload = json.loads((tmp_path / "export" / "export_summary.json").read_text())
    assert payload["model"] == "shufflenet_v2_x1_0"
    assert payload["formats"][0]["succeeded"] is True
    assert payload["formats"][0]["parity"]["passed"] is True
    assert Path(payload["formats"][0]["output_path"]) == Path("export/model.onnx")


def test_write_export_summary_incluye_sha256_del_artefacto(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    model_path = export_dir / "model.onnx"
    model_path.write_bytes(b"contenido de prueba")

    parity = ParityResult(
        format="onnx",
        n_samples=30,
        torch_top1_accuracy=0.9,
        exported_top1_accuracy=0.9,
        agreement_rate=1.0,
        max_abs_prob_diff=0.0001,
        mean_abs_prob_diff=0.00001,
        tolerance=1e-3,
        passed=True,
    )
    report = ExportReport(
        run_dir=tmp_path,
        model_name="shufflenet_v2_x1_0",
        formats=[
            ExportFormatResult(format="onnx", output_path=model_path, succeeded=True, parity=parity)
        ],
        library_versions={"torch": "2.12.1"},
    )

    write_export_summary(tmp_path, report)

    payload = json.loads((tmp_path / "export" / "export_summary.json").read_text())
    expected_sha256 = hashlib.sha256(b"contenido de prueba").hexdigest()
    assert payload["formats"][0]["sha256"] == expected_sha256


def test_write_export_summary_sha256_none_si_no_hay_output_path(tmp_path):
    report = ExportReport(
        run_dir=tmp_path,
        model_name="shufflenet_v2_x1_0",
        formats=[
            ExportFormatResult(format="tflite", output_path=None, succeeded=False, error="fallo")
        ],
        library_versions={},
    )

    write_export_summary(tmp_path, report)

    payload = json.loads((tmp_path / "export" / "export_summary.json").read_text())
    assert payload["formats"][0]["sha256"] is None


def test_validate_export_artifact_rechaza_bytes_modificados(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    model_path = export_dir / "model.onnx"
    model_path.write_bytes(b"export contractual")
    run = {
        "run_id": tmp_path.name,
        "checkpoint_sha256": "a" * 64,
        "class_to_idx": {"healthy": 0},
        "preprocessing": {"schema_version": 1},
    }
    payload = {
        "schema_version": 1,
        **run,
        "formats": [
            {
                "format": "onnx",
                "output_path": "export/model.onnx",
                "succeeded": True,
                "sha256": hashlib.sha256(b"export contractual").hexdigest(),
            }
        ],
    }
    (export_dir / "export_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("src.export.common.read_run_contract", lambda _run_dir: run)

    assert validate_export_artifact(tmp_path, model_path, "onnx", None) == payload
    with model_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ExportIntegrityError, match="SHA-256 del export no coincide"):
        validate_export_artifact(tmp_path, model_path, "onnx", None)


def test_export_to_tflite_sin_dependencia_levanta_error_claro(monkeypatch):
    from src.export import tflite_export

    monkeypatch.setitem(sys.modules, "litert_torch", None)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "litert_torch":
            raise ImportError("simulado")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ExportDependencyError, match=r"pip install -e '\.\[export\]'"):
        tflite_export.export_to_tflite(
            torch.nn.Linear(1, 1), Path("unused.tflite"), (32, 32), torch.device("cpu")
        )


def test_write_labels_json_ordena_por_indice(tmp_path):
    class_to_idx = {"healthy": 2, "common_rust": 0, "fall_armyworm": 1}

    output_path = write_labels_json(tmp_path, class_to_idx, "shufflenet_v2_x1_0", (224, 224))

    assert output_path == tmp_path / "export" / "labels.json"
    payload = json.loads(output_path.read_text())
    assert payload["model"] == "shufflenet_v2_x1_0"
    assert payload["image_size"] == [224, 224]
    assert payload["labels"] == ["common_rust", "fall_armyworm", "healthy"]


def test_write_labels_json_incluye_schema_version(tmp_path):
    class_to_idx = {"healthy": 0, "common_rust": 1}

    output_path = write_labels_json(tmp_path, class_to_idx, "shufflenet_v2_x1_0", (224, 224))

    payload = json.loads(output_path.read_text())
    assert payload["schema_version"] == 1


def test_write_labels_json_indice_no_contiguo_levanta_error(tmp_path):
    class_to_idx = {"healthy": 0, "common_rust": 0, "fall_armyworm": 1}

    with pytest.raises(ValueError, match="no es contiguo"):
        write_labels_json(tmp_path, class_to_idx, "shufflenet_v2_x1_0", (224, 224))
