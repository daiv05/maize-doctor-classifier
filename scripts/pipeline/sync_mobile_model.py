"""Entrega transaccional: validar todo antes de sustituir un bundle móvil."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from src.provenance import atomic_json, contract_hash, sha256_file


def _summary_path(run_dir: Path, quantize: str | None) -> Path:
    suffix = f"_{quantize}" if quantize else ""
    return run_dir / "export" / f"export_summary{suffix}.json"


def sync_mobile_model(
    run_dir: Path, dest_dir: Path, *, fmt="tflite", quantize="int8", require_ood=True
) -> Path:
    run_dir, dest_dir = run_dir.resolve(), dest_dir.resolve()
    if (
        dest_dir == Path(dest_dir.anchor)
        or dest_dir == Path.home()
        or dest_dir == Path.cwd().resolve()
        or dest_dir.is_relative_to(run_dir)
        or run_dir.is_relative_to(dest_dir)
    ):
        raise ValueError("Destino inseguro: se requiere un directorio exclusivo de assets/model.")
    export_dir = run_dir / "export"
    summary_path = _summary_path(run_dir, quantize)
    summary = json.loads(summary_path.read_text())
    match = next((f for f in summary["formats"] if f["format"] == fmt), None)
    if match is None:
        raise ValueError(f"No se encontro formato {fmt}")
    if not match.get("succeeded") or not (match.get("parity") or {}).get("passed"):
        raise ValueError("Artefacto no validado: paridad ausente/fallida.")
    if summary.get("run_id") != run_dir.name or summary.get("quantize") != quantize:
        raise ValueError("Run/variante incompatibles.")
    name = Path(match["output_path"]).name
    source_model = export_dir / name
    if sha256_file(source_model) != match["sha256"]:
        raise ValueError("El hash del modelo no coincide con el registrado.")
    training_path = run_dir / "summary.json"
    training = json.loads(training_path.read_text())
    if summary.get("training_summary_sha256") != sha256_file(training_path):
        raise ValueError("Contrato de entrenamiento alterado.")
    if summary.get("checkpoint_sha256") != sha256_file(run_dir / "best.pth"):
        raise ValueError("Checkpoint incompatible.")
    assets = {"model": source_model}
    for asset in ("labels.json", "preprocessing.json"):
        path = export_dir / asset
        if sha256_file(path) != summary.get("asset_hashes", {}).get(asset):
            raise ValueError(f"Hash incompatible: {asset}")
        assets[asset] = path
    labels = json.loads(assets["labels.json"].read_text())
    preproc = json.loads(assets["preprocessing.json"].read_text())
    from src.data.transforms import CornTransformFactory

    CornTransformFactory.from_contract(preproc)
    ordered_labels = [k for k, v in sorted(training["class_to_idx"].items(), key=lambda p: p[1])]
    if (
        labels["labels"] != ordered_labels
        or labels["image_size"] != preproc["target_size"]
        or labels["model"] != training["model"]
        or training["model"] != summary["model"]
        or preproc != training["preprocessing"]
        or contract_hash(preproc) != summary.get("preprocessing_id")
    ):
        raise ValueError("Etiquetas/preprocesamiento/modelo incompatibles.")
    if preproc.get("segmentation"):
        raise ValueError("Entrega segmentada requiere soporte móvil validado del segmentador.")
    suffix = f"{fmt}_{quantize}" if quantize else fmt
    evaluation_path = export_dir / f"eval_{suffix}.json"
    evaluation = json.loads(evaluation_path.read_text())
    delta = evaluation.get("macro_f1_delta")
    if (
        delta is None
        or not math.isfinite(delta)
        or delta < -0.01
        or evaluation.get("model_sha256") != match["sha256"]
        or evaluation.get("checkpoint_sha256") != summary["checkpoint_sha256"]
        or evaluation.get("preprocessing_id") != summary["preprocessing_id"]
        or evaluation.get("run_id") != run_dir.name
        or not evaluation.get("sample_ids_hash")
        or evaluation.get("n_samples", 0) <= 0
        or evaluation.get("n_samples") != evaluation.get("expected_samples")
        or not training.get("split_sha256", {}).get("test")
        or evaluation.get("evaluated_split_sha256") != training["split_sha256"]["test"]
    ):
        raise ValueError("Evaluación completa ausente, incompatible o caída macro-F1 > 0.01.")
    predictions_path = export_dir / f"eval_{suffix}_predictions.csv"
    if sha256_file(predictions_path) != evaluation.get("predictions_sha256"):
        raise ValueError("Predicciones de evaluación alteradas.")
    if require_ood:
        ood_path = export_dir / "ood_stats.json"
        if not ood_path.is_file():
            raise ValueError("Falta OOD del mismo run; no se reutiliza el anterior.")
        ood = json.loads(ood_path.read_text())
        features = match.get("feature_parity") or {}
        if (
            ood.get("checkpoint_sha256") != summary["checkpoint_sha256"]
            or ood.get("run_id") != run_dir.name
            or ood.get("preprocessing_id") != summary["preprocessing_id"]
            or ood.get("labels") != ordered_labels
            or not features.get("passed")
            or features.get("feature_dim") != ood.get("feature_dim")
            or features.get("kind") != "pooled_pre_head"
            or not ood.get("l2_normalized")
        ):
            raise ValueError("Contrato OOD/features incompatible o no validado.")
        assets["ood_stats.json"] = ood_path
    # Prepare a complete new directory; absent OOD never leaves stale OOD behind.
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dest_dir.name}-stage-", dir=dest_dir.parent))
    backup = dest_dir.with_name(f".{dest_dir.name}-backup-{uuid4().hex}")
    try:
        hashes = {}
        for key, source in assets.items():
            target = stage / (name if key == "model" else key)
            shutil.copy2(source, target)
            hashes[target.name] = sha256_file(target)
            if hashes[target.name] != sha256_file(source):
                raise ValueError(f"Copia alterada: {source.name}")
        manifest = {
            "schema_version": 2,
            "run_id": run_dir.name,
            "model": summary["model"],
            "format": fmt,
            "quantize": quantize,
            "sha256": match["sha256"],
            "assets": hashes,
            "ood_stats_synced": require_ood,
            "preprocessing_id": summary["preprocessing_id"],
            "evaluation_sha256": sha256_file(evaluation_path),
            "export_summary_sha256": sha256_file(summary_path),
            "status": "validated_bundle",
            "device_validation": "not_performed",
            "previous_bundle_backup": str(backup) if dest_dir.exists() else None,
        }
        atomic_json(stage / "manifest.json", manifest)
        if dest_dir.exists():
            os.replace(dest_dir, backup)
        try:
            os.replace(stage, dest_dir)
        except BaseException:
            if backup.exists():
                os.replace(backup, dest_dir)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # Only our validated, newly created staging directory.
    return dest_dir / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--format", default="tflite", choices=["onnx", "tflite"])
    parser.add_argument("--quantize", default="int8")
    parser.add_argument(
        "--without-ood",
        action="store_true",
        help="Bundle sin OOD: requiere consumidor que acepte esa capacidad.",
    )
    args = parser.parse_args()
    quantize = None if args.quantize.lower() in {"none", ""} else args.quantize
    path = sync_mobile_model(
        args.run_dir,
        args.dest,
        fmt=args.format,
        quantize=quantize,
        require_ood=not args.without_ood,
    )
    print(f"Bundle verificado: {path}; la validación en dispositivo sigue siendo independiente.")


if __name__ == "__main__":
    main()
