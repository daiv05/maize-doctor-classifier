"""Protocolo congelado del ensamble LOSO multicrop, sin dependencia de Modal."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.data.preparation import sha256_file

SOURCE = "multicrop-disease-maiz"
SEEDS = (42, 123, 2026)
MODELS = ("efficientnet_b0", "shufflenet_v2_x1_0")
SPLITS = Path("/outputs/loso/efficientnet_lite0_baseline") / SOURCE / "seed_42/splits"
ROOT = Path("/outputs/ensemble_loso") / SOURCE
LOCK_SHA256 = "d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b"
HISTORICAL_SUMMARY_SHA256 = {
    "efficientnet_b0": "d125bd11b6a6f1fcfe01bc6ec66fa0bab672c9f4853cee440dd2d1971a374afd",
    "shufflenet_v2_x1_0": "2c20e614bae62425cad30446efad517270809bdfa6b2e5347d387a13b09fa7b0",
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command(model: str, seed: int, root: Path) -> list[str]:
    if model not in MODELS or seed not in SEEDS:
        raise ValueError("Modelo o seed fuera del protocolo")
    return [
        sys.executable,
        "-m",
        "scripts.pipeline.train",
        "--models",
        model,
        "--splits-dir",
        str(SPLITS),
        "--output-dir",
        str(root / "runs"),
        "--skip-test",
        "--seed",
        str(seed),
        "--epochs",
        "35",
        "--batch-size",
        "64",
        "--learning-rate",
        "0.0004548",
        "--weight-decay",
        "0.0001",
        "--scheduler",
        "cosine",
        "--warmup-epochs",
        "3",
        "--min-lr",
        "0.000001",
        "--patience",
        "8",
        "--class-weights",
        "sqrt_inverse",
        "--label-smoothing",
        "0.1",
        "--clip-grad-norm",
        "1.0",
        "--num-workers",
        "32",
        "--max-per-class",
        "0",
        "--no-clahe",
    ]


def verify_run(run_dir: Path, model: str, seed: int, historical: dict) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    fields = (
        "num_classes",
        "class_to_idx",
        "image_size",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "warmup_epochs",
        "min_lr",
        "patience",
        "class_weights",
        "label_smoothing",
        "clip_grad_norm",
        "clahe",
        "sampler",
        "pretrained",
    )
    for field in fields:
        if summary[field] != historical[field]:
            raise ValueError(f"Drift histórico {model}/{seed}: {field}")
    if (
        summary["model"] != model
        or summary["seed"] != seed
        or summary["epochs_requested"] != historical["epochs_requested"]
        or summary["split_manifest_sha256"] != LOCK_SHA256
        or summary.get("test_used") is not False
        or summary.get("evaluation_mode") != "validation_only"
        or "test" in summary.get("metrics", {})
    ):
        raise ValueError(f"Run no corresponde al protocolo congelado: {run_dir}")
    if sha256_file(run_dir / "best.pth") != summary["checkpoint_sha256"]:
        raise ValueError("SHA del checkpoint no coincide")
    if not (run_dir / "train_history.csv").is_file():
        raise FileNotFoundError(run_dir / "train_history.csv")
    if not (run_dir / "validation_predictions.csv").is_file():
        raise FileNotFoundError(run_dir / "validation_predictions.csv")
    return summary


def inspect_slot(root: Path, model: str, seed: int, historical: dict) -> tuple[bool, int]:
    runs = root / "runs" / model
    run_dirs = [path for path in runs.iterdir() if path.is_dir()] if runs.exists() else []
    metadata_path = root / "experiment_metadata.json"
    if not metadata_path.exists():
        if runs.exists() and any(runs.iterdir()):
            raise FileExistsError(f"Run sin metadatos: {runs}")
        return False, 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("model") != model
        or metadata.get("seed") != seed
        or metadata.get("split_lock_sha256") != LOCK_SHA256
        or metadata.get("command") != command(model, seed, root)
    ):
        raise ValueError(f"Metadatos incompatibles: {metadata_path}")
    if len(run_dirs) > 1:
        raise ValueError(f"Múltiples runs en un slot: {runs}")
    if run_dirs and (run_dirs[0] / "summary.json").exists():
        summary = verify_run(run_dirs[0], model, seed, historical)
        if metadata.get("status") == "validation_complete":
            if (
                metadata.get("run_id") != summary["run_id"]
                or metadata.get("checkpoint_sha256") != summary["checkpoint_sha256"]
            ):
                raise ValueError("Metadatos y checkpoint incompatibles")
        else:
            metadata.update(
                status="validation_complete",
                run_id=summary["run_id"],
                checkpoint_sha256=summary["checkpoint_sha256"],
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_json(metadata_path, metadata)
        return True, int(metadata.get("attempt_count", 1))
    if metadata.get("status") == "validation_complete":
        raise ValueError("Slot completo sin run verificable")
    attempt = int(metadata.get("attempt_count", 1))
    if run_dirs:
        if attempt >= 2:
            raise RuntimeError(f"Dos intentos incompletos: {runs}")
        archive = root / "interrupted_attempts" / f"attempt_{attempt}"
        if archive.exists():
            raise FileExistsError(archive)
        archive.mkdir(parents=True)
        runs.rename(archive / model)
        write_json(archive / "experiment_metadata.json", metadata)
        return False, attempt + 1
    archive = root / "interrupted_attempts" / f"attempt_{attempt}"
    return False, attempt + 1 if archive.exists() else attempt
