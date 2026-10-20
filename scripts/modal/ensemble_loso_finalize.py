"""Finalize the frozen three-seed ensemble without retraining or weight tuning."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from scripts.modal._common import dataset_vol, image, outputs_vol
from scripts.modal.ensemble_loso_train import (
    HISTORICAL_SUMMARY_SHA256,
    ROOT,
    SEEDS,
    sha256_file,
)

app = modal.App("doctor-maiz-ensemble-loso-finalize", image=image)


def final_status(seed: int) -> bool:
    out = ROOT / f"seed_{seed}" / "evaluation"
    lock_path = ROOT / f"seed_{seed}" / "ensemble.selection.lock.json"
    guard = out / "final_evaluation.started.json"
    metrics = out / "ensemble_metrics.json"
    if not guard.exists():
        if metrics.exists():
            raise ValueError("Métricas sin guard de evaluación")
        return False
    if not metrics.exists():
        raise RuntimeError(f"Evaluación iniciada pero incompleta; no repetir: seed={seed}")
    lock_sha = sha256_file(lock_path)
    guard_data = json.loads(guard.read_text(encoding="utf-8"))
    if (
        guard_data["seed"] != seed
        or guard_data["evaluation_count"] != 1
        or guard_data["ensemble_selection_lock_sha256"] != lock_sha
    ):
        raise ValueError("Guard incompatible")
    for model in ("efficientnet_lite0", "efficientnet_b0", "shufflenet_v2_x1_0", "ensemble"):
        for suffix in (
            "metrics.json",
            "predictions.csv",
            "class_metrics.csv",
            "confusion_matrix.csv",
        ):
            if not (out / f"{model}_{suffix}").is_file():
                raise FileNotFoundError(out / f"{model}_{suffix}")
        result = json.loads((out / f"{model}_metrics.json").read_text(encoding="utf-8"))
        if (
            result["seed"] != seed
            or result["evaluation_count"] != 1
            or result["samples"] != 5816
            or result["ensemble_selection_lock_sha256"] != lock_sha
        ):
            raise ValueError(f"Resultado incompleto: {model}/seed_{seed}")
    return True


@app.function(
    gpu="A10G",
    cpu=16,
    memory=32768,
    timeout=8 * 3600,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    max_containers=1,
    retries=0,
)
@modal.concurrent(max_inputs=1)
def finalize(historical_summaries: dict) -> None:
    from scripts.experiments.ensemble_loso_finalize import evaluate_seed, lock_all

    dataset_vol.reload()
    outputs_vol.reload()
    locked = lock_all(historical_summaries)
    outputs_vol.commit()
    print(f"Selection locks nuevos: {len(locked)}", flush=True)
    for seed in SEEDS:
        if final_status(seed):
            print(f"Evaluación ya verificada: seed_{seed}", flush=True)
            continue
        result = evaluate_seed(seed, commit_guard=outputs_vol.commit)
        outputs_vol.commit()
        print(
            f"seed_{seed}: B0={result['efficientnet_b0']['macro_f1']:.6f} "
            f"ensemble={result['ensemble']['macro_f1']:.6f}",
            flush=True,
        )


@app.local_entrypoint()
def main(execute: bool = False) -> None:
    project = Path(__file__).resolve().parents[2]
    original = project / "docs/es/resultados/evidencia/ensamble_resumen.json"
    evidence_sha256 = "dcafbb4c21e2d69646b8a04cd1bbf425474b615f5eb6b44d594abd469b6f90c2"
    if sha256_file(original) != evidence_sha256:
        raise ValueError("Evidencia del ensamble histórico alterada")
    evidence = json.loads(original.read_text(encoding="utf-8"))
    if evidence["models_included"] != [
        "efficientnet_lite0",
        "efficientnet_b0",
        "shufflenet_v2_x1_0",
    ] or any(abs(weight - 1 / 3) > 1e-7 for weight in evidence["weights"]):
        raise ValueError("Composición o pesos históricos distintos")
    summaries = {}
    for model, digest in HISTORICAL_SUMMARY_SHA256.items():
        run_id = "20260910_170120" if model == "efficientnet_b0" else "20260910_184521"
        path = project / "outputs/outputs-11092026/main" / model / run_id / "summary.json"
        if sha256_file(path) != digest:
            raise ValueError(f"Summary histórico alterado: {path}")
        summaries[model] = json.loads(path.read_text(encoding="utf-8"))
    if execute:
        finalize.remote(summaries)
    else:
        print("Preflight local: sin --execute no se abre el holdout")
