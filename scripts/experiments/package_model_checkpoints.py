"""Verify and stage documented checkpoints without training or inference."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "outputs/doctor_maiz_models"
EVIDENCE = ROOT / "docs/es/reproducibilidad/evidencia"
ENSEMBLE = EVIDENCE / "ensemble_loso"
STABILITY = EVIDENCE / "multiseed_source_stability"
HPO = EVIDENCE / "hpo_lite0_seed42"
NA = "N/A"

INDEX_FIELDS = (
    "model_id",
    "architecture",
    "experiment",
    "seed",
    "holdout",
    "run_id",
    "checkpoint_filename",
    "checkpoint_sha256",
    "best_epoch",
    "validation_macro_f1",
    "final_macro_f1",
    "status",
    "original_modal_path",
    "local_path",
    "source_local_path",
    "size_bytes",
)
COMPARISON_FIELDS = (
    "model_id",
    "architecture",
    "experiment",
    "seed",
    "evaluation_split",
    "held_out_source",
    "validation_macro_f1",
    "standard_test_macro_f1",
    "loso_macro_f1",
    "final_accuracy",
    "final_ece_15_bins",
    "npk_grouped_macro_f1",
    "nitrogen_f1",
    "phosphorus_f1",
    "potassium_f1",
    "lethal_necrosis_precision",
    "lethal_necrosis_recall",
    "lethal_necrosis_f1",
    "checkpoint_sha256",
    "size_bytes",
    "evidence_path",
    "notes",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def put_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            # Solo archivos derivados de este script; nunca modifica los checkpoints fuente.
            path.write_bytes(content)
        return
    with path.open("xb") as stream:
        stream.write(content)


def put_json(path: Path, value: dict | list) -> None:
    put_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def put_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, NA) for field in fields})
    put_bytes(path, buffer.getvalue().encode("utf-8"))


def copy_checked(source: Path, destination: Path, expected: str | None = None) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = digest(source)
    if expected and source_sha != expected:
        raise ValueError(
            f"Hash de origen distinto: {source}; esperado={expected}; real={source_sha}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != source_sha:
            raise FileExistsError(f"Destino previo diferente; no se sobrescribe: {destination}")
    else:
        shutil.copy2(source, destination)
    if digest(destination) != source_sha:
        raise ValueError(f"Copia no coincide: {destination}")
    return source_sha


def result_row(rows: list[dict], scenario: str, seed: int) -> dict:
    matches = [row for row in rows if row["scenario"] == scenario and int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Resultado ambiguo: {scenario}/{seed}")
    return matches[0]


def class_values(rows: list[dict], *, seed: int, scenario: str) -> dict[str, str]:
    selected = {
        row["class"]: row
        for row in rows
        if int(row["seed"]) == seed
        and row.get("scenario", scenario) == scenario
        and row.get("stage", "test") in {"test", "holdout"}
    }
    values = {}
    for class_name, field in (
        ("nitrogen_deficiency", "nitrogen_f1"),
        ("phosphorus_deficiency", "phosphorus_f1"),
        ("potassium_deficiency", "potassium_f1"),
    ):
        values[field] = selected.get(class_name, {}).get("f1", NA)
    ln = selected.get("lethal_necrosis", {})
    for field in ("precision", "recall", "f1"):
        values[f"lethal_necrosis_{field}"] = ln.get(field, NA)
    return values


def report_class_values(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {row[""]: row for row in csv.DictReader(stream)}
    values = {}
    for class_name, field in (
        ("nitrogen_deficiency", "nitrogen_f1"),
        ("phosphorus_deficiency", "phosphorus_f1"),
        ("potassium_deficiency", "potassium_f1"),
    ):
        values[field] = rows.get(class_name, {}).get("f1-score", NA)
    ln = rows.get("lethal_necrosis", {})
    for field in ("precision", "recall"):
        values[f"lethal_necrosis_{field}"] = ln.get(field, NA)
    values["lethal_necrosis_f1"] = ln.get("f1-score", NA)
    return values


def add_model(
    *,
    records: list[dict],
    comparisons: list[dict],
    model_id: str,
    model: str,
    experiment: str,
    seed: int | str,
    holdout: str,
    source_run: Path,
    package_dir: str,
    expected_sha: str,
    source_evidence: Path,
    original_modal_path: str,
    final_macro_f1: float,
    final_accuracy: float | str,
    final_ece: float | str,
    class_metrics: dict[str, str],
    npk_grouped: float | str = NA,
    extra_files: tuple[tuple[Path, str], ...] = (),
    note: str = "",
) -> None:
    summary_path = source_run / "summary.json"
    summary = read_json(summary_path)
    if summary["model"] != model or (isinstance(seed, int) and summary.get("seed") != seed):
        raise ValueError(f"Summary incompatible: {summary_path}")
    if summary.get("checkpoint_sha256") not in (None, expected_sha):
        raise ValueError(f"Hash de summary incompatible: {summary_path}")
    destination = STAGE / package_dir
    checkpoint = source_run / "best.pth"
    copy_checked(checkpoint, destination / "best.pth", expected_sha)
    copy_checked(summary_path, destination / "summary.json")
    selection_lock = source_run / "selection.lock.json"
    if selection_lock.is_file():
        copy_checked(selection_lock, destination / "selection.lock.json")
    for source, name in extra_files:
        copy_checked(source, destination / name)
    put_json(
        destination / "config_snapshot.json",
        {
            "source": "summary.json",
            "architecture": summary.get("architecture", model),
            "model": model,
            "seed": summary.get("seed"),
            "run_id": summary.get("run_id"),
            "hyperparameters": summary.get("hyperparameters"),
            "preprocessing": summary.get("preprocessing"),
            "training_preprocessing": summary.get("training_preprocessing"),
            "class_to_idx": summary.get("class_to_idx"),
            "config_sha256": summary.get("config_sha256"),
            "split_manifest_sha256": summary.get("split_manifest_sha256"),
        },
    )
    put_json(
        destination / "checkpoint_reference.json",
        {
            "model_id": model_id,
            "status": "VERIFIED",
            "checkpoint_sha256": expected_sha,
            "source_evidence": str(source_evidence.relative_to(ROOT)),
            "source_evidence_sha256": digest(source_evidence),
            "source_local_path": str(checkpoint),
            "original_modal_path": original_modal_path,
            "package_path": str((destination / "best.pth").relative_to(STAGE)),
        },
    )
    best_val = summary.get("best_val_macro_f1")
    if best_val is None:
        best_val = (summary.get("metrics") or {}).get("best_validation", {}).get("macro_f1", NA)
    run_id = summary.get("run_id", NA)
    index = {
        "model_id": model_id,
        "architecture": model,
        "experiment": experiment,
        "seed": seed,
        "holdout": holdout,
        "run_id": run_id,
        "checkpoint_filename": "best.pth",
        "checkpoint_sha256": expected_sha,
        "best_epoch": summary.get("best_epoch", NA),
        "validation_macro_f1": best_val,
        "final_macro_f1": final_macro_f1,
        "status": "VERIFIED",
        "original_modal_path": original_modal_path,
        "local_path": str((destination / "best.pth").relative_to(STAGE)),
        "source_local_path": str(checkpoint.relative_to(ROOT)),
        "size_bytes": checkpoint.stat().st_size,
    }
    records.append(index)
    standard = experiment in {"standard_baseline", "hpo_trial_0", "historical_standard"}
    comparisons.append(
        {
            "model_id": model_id,
            "architecture": model,
            "experiment": experiment,
            "seed": seed,
            "evaluation_split": "standard_test" if standard else "loso_holdout",
            "held_out_source": holdout,
            "validation_macro_f1": best_val,
            "standard_test_macro_f1": final_macro_f1 if standard else NA,
            "loso_macro_f1": final_macro_f1 if not standard else NA,
            "final_accuracy": final_accuracy,
            "final_ece_15_bins": final_ece,
            "npk_grouped_macro_f1": npk_grouped,
            **class_metrics,
            "checkpoint_sha256": expected_sha,
            "size_bytes": checkpoint.stat().st_size,
            "evidence_path": str(source_evidence.relative_to(ROOT)),
            "notes": note,
        }
    )


def main() -> None:
    baseline_refs_path = STABILITY / "manifest_references.json"
    baseline_refs = read_json(baseline_refs_path)
    ensemble_refs_path = ENSEMBLE / "manifest_checkpoint_references.json"
    ensemble_refs = read_json(ensemble_refs_path)
    hpo_lock_path = HPO / "HPO_SELECTION_LOCK.json"
    hpo_lock = read_json(hpo_lock_path)
    baseline_rows = read_rows(STABILITY / "baseline_multiseed.csv")
    loso_rows = read_rows(STABILITY / "loso_multiseed.csv")
    baseline_classes = read_rows(STABILITY / "baseline_class_multiseed.csv")
    loso_classes = read_rows(STABILITY / "loso_class_multiseed.csv")
    ensemble_rows = read_rows(ENSEMBLE / "ensemble_component_results.csv")
    ensemble_classes = read_rows(ENSEMBLE / "ensemble_loso_class_metrics.csv")
    records: list[dict] = []
    comparisons: list[dict] = []

    for seed in (42, 123, 2026):
        ref = baseline_refs[f"baseline/seed_{seed}"]
        row = result_row(baseline_rows, "baseline", seed)
        run = ROOT / ref["run_path"]
        extras = (
            [(run / "test_calibration.json", "test_calibration.json")]
            if seed == 42
            else [(run / "test_metrics.json", "final_metrics.json")]
        )
        if seed == 42:
            extras.extend(
                [
                    (run / "test_grouped_metrics.json", "test_grouped_metrics.json"),
                    (run / "test_classification_report.csv", "final_class_metrics.csv"),
                ]
            )
        else:
            extras.append((run / "test_class_metrics.csv", "final_class_metrics.csv"))
        add_model(
            records=records,
            comparisons=comparisons,
            model_id=f"baseline_lite0_seed_{seed}",
            model="efficientnet_lite0",
            experiment="standard_baseline",
            seed=seed,
            holdout=NA,
            source_run=run,
            package_dir=f"baseline_lite0/seed_{seed}",
            expected_sha=ref["checkpoint_sha256"],
            source_evidence=baseline_refs_path,
            original_modal_path=(
                f"corn-outputs:/main/efficientnet_lite0/{ref['run_id']}/best.pth"
                if seed == 42
                else "corn-outputs:/" + ref["run_path"].removeprefix("outputs/") + "/best.pth"
            ),
            final_macro_f1=float(row["final_macro_f1"]),
            final_accuracy=float(row["final_accuracy"]),
            final_ece=float(row["final_ece_15_bins"]),
            class_metrics=class_values(baseline_classes, seed=seed, scenario="baseline"),
            npk_grouped=0.9783847968475754 if seed == 42 else NA,
            extra_files=tuple(extras),
            note="Seed 42 histórica: sin lock pre-test documentado" if seed == 42 else "",
        )

    trial = ROOT / "outputs/hpo-backups/amended-budget25/trials/trial_000"
    if digest(trial / "summary.json") != hpo_lock["winner_summary_sha256"]:
        raise ValueError("Summary HPO distinto del selection lock")
    hpo_final = read_json(HPO / "FINAL_TEST_COMPLETE.json")
    hpo_cal = read_json(HPO / "final_test/test_calibration.json")
    hpo_grouped = read_json(HPO / "final_test/test_grouped_metrics.json")
    add_model(
        records=records,
        comparisons=comparisons,
        model_id="hpo_lite0_trial_000_seed_42",
        model="efficientnet_lite0",
        experiment="hpo_trial_0",
        seed=42,
        holdout=NA,
        source_run=trial,
        package_dir="hpo_lite0/trial_000",
        expected_sha=hpo_lock["winner_checkpoint_sha256"],
        source_evidence=hpo_lock_path,
        original_modal_path=(
            "corn-outputs:/hpo/efficientnet_lite0/efficientnet_lite0_seed42_hpo_v1/"
            "trials/trial_000/best.pth"
        ),
        final_macro_f1=hpo_final["metrics"]["macro_f1"],
        final_accuracy=hpo_final["metrics"]["accuracy"],
        final_ece=hpo_cal["ece"],
        class_metrics=report_class_values(HPO / "final_test/test_classification_report.csv"),
        npk_grouped=hpo_grouped["grouped_macro_f1"],
        extra_files=(
            (hpo_lock_path, "HPO_SELECTION_LOCK.json"),
            (HPO / "FINAL_TEST_COMPLETE.json", "final_metrics.json"),
            (HPO / "final_test/test_classification_report.csv", "final_class_metrics.csv"),
            (HPO / "final_test/test_calibration.json", "test_calibration.json"),
            (HPO / "final_test/test_grouped_metrics.json", "test_grouped_metrics.json"),
        ),
        note="Trial 0 elegido por validation; test posterior único",
    )

    for scenario, folder in (
        ("maize-diseases", "loso_maize_diseases/lite0"),
        ("multicrop-disease-maiz", "loso_multicrop/lite0"),
    ):
        for seed in (42, 123, 2026):
            ref = baseline_refs[f"{scenario}/seed_{seed}"]
            row = result_row(loso_rows, scenario, seed)
            run = ROOT / ref["run_path"]
            add_model(
                records=records,
                comparisons=comparisons,
                model_id=f"loso_{scenario.replace('-', '_')}_lite0_seed_{seed}",
                model="efficientnet_lite0",
                experiment="loso_" + scenario.replace("-", "_"),
                seed=seed,
                holdout=scenario,
                source_run=run,
                package_dir=f"{folder}/seed_{seed}",
                expected_sha=ref["checkpoint_sha256"],
                source_evidence=baseline_refs_path,
                original_modal_path="corn-outputs:/"
                + ref["run_path"].removeprefix("outputs/")
                + "/best.pth",
                final_macro_f1=float(row["final_macro_f1"]),
                final_accuracy=float(row["final_accuracy"]),
                final_ece=float(row["final_ece_15_bins"]),
                class_metrics=class_values(loso_classes, seed=seed, scenario=scenario),
                extra_files=(
                    (run / "holdout_metrics.json", "final_metrics.json"),
                    (run / "holdout_class_metrics.csv", "final_class_metrics.csv"),
                ),
                note="Holdout externo: fuente ausente de train y validation",
            )

    for member in ensemble_refs["members"]:
        if member["model"] == "efficientnet_lite0":
            continue  # Ya se respaldó como LOSO Lite0 multicrop.
        seed = member["seed"]
        model = member["model"]
        name = "b0" if model == "efficientnet_b0" else "shufflenet"
        matches = [
            row for row in ensemble_rows if int(row["seed"]) == seed and row["model"] == model
        ]
        if len(matches) != 1:
            raise ValueError(f"Resultado del componente ambiguo: {model}/{seed}")
        row = matches[0]
        run = ROOT / member["local_run_dir"]
        if digest(run / "summary.json") != member["summary_sha256"]:
            raise ValueError(f"Summary LOSO distinto: {run}")
        evaluation = ROOT / f"outputs/ensemble_loso/multicrop-disease-maiz/seed_{seed}/evaluation"
        selected_classes = [
            item
            for item in ensemble_classes
            if int(item["seed"]) == seed and item["model"] == model
        ]
        add_model(
            records=records,
            comparisons=comparisons,
            model_id=f"loso_multicrop_{name}_seed_{seed}",
            model=model,
            experiment="loso_multicrop_disease_maiz",
            seed=seed,
            holdout="multicrop-disease-maiz",
            source_run=run,
            package_dir=f"loso_multicrop/{name}/seed_{seed}",
            expected_sha=member["checkpoint_sha256"],
            source_evidence=ensemble_refs_path,
            original_modal_path=(
                "corn-outputs:/" + member["run_dir"].removeprefix("/outputs/") + "/best.pth"
            ),
            final_macro_f1=float(row["macro_f1"]),
            final_accuracy=float(row["accuracy"]),
            final_ece=float(row["ece_15_bins"]),
            class_metrics=class_values(selected_classes, seed=seed, scenario="ensemble"),
            extra_files=(
                (evaluation / f"{model}_metrics.json", "final_metrics.json"),
                (evaluation / f"{model}_class_metrics.csv", "final_class_metrics.csv"),
                (
                    ROOT
                    / "outputs/ensemble_loso/multicrop-disease-maiz"
                    / f"seed_{seed}/ensemble.selection.lock.json",
                    "ensemble.selection.lock.json",
                ),
            ),
            note="Miembro del ensamble LOSO; fusión uniforme, sin tuning de pesos",
        )

    historical = ROOT / "outputs/outputs-11092026/main"
    for model, name, key in (
        ("efficientnet_b0", "b0", "historical_b0_checkpoint_sha256"),
        ("shufflenet_v2_x1_0", "shufflenet", "historical_shuffle_checkpoint_sha256"),
    ):
        runs = [run for run in (historical / model).iterdir() if (run / "summary.json").is_file()]
        if len(runs) != 1:
            raise ValueError(f"Run histórico ambiguo: {model}")
        run = runs[0]
        summary = read_json(run / "summary.json")
        calibration = read_json(run / "test_calibration.json")
        grouped = read_json(run / "test_grouped_metrics.json")
        add_model(
            records=records,
            comparisons=comparisons,
            model_id=f"historical_standard_{name}_{run.name}",
            model=model,
            experiment="historical_standard",
            seed=NA,
            holdout=NA,
            source_run=run,
            package_dir=f"historical_standard/{name}",
            expected_sha=ensemble_refs[key],
            source_evidence=ensemble_refs_path,
            original_modal_path=NA,
            final_macro_f1=summary["test"]["macro_f1"],
            final_accuracy=summary["test"]["accuracy"],
            final_ece=calibration["ece"],
            class_metrics=report_class_values(run / "test_classification_report.csv"),
            npk_grouped=grouped["grouped_macro_f1"],
            extra_files=(
                (run / "test_calibration.json", "test_calibration.json"),
                (run / "test_grouped_metrics.json", "test_grouped_metrics.json"),
                (run / "test_classification_report.csv", "final_class_metrics.csv"),
            ),
            note="Materialización histórica; seed y ruta Modal originales no recuperadas",
        )

    if len(records) != 18 or len({row["model_id"] for row in records}) != 18:
        raise ValueError(f"Se esperaban 18 checkpoints distintos, hay {len(records)}")
    records.sort(key=lambda row: row["model_id"])
    comparisons.sort(key=lambda row: row["model_id"])
    historical_ensemble = read_json(ROOT / "docs/es/resultados/evidencia/ensamble_resumen.json")
    comparisons.append(
        {
            "model_id": "historical_standard_ensemble",
            "architecture": "soft_voting_lite0_b0_shufflenet",
            "experiment": "historical_standard",
            "seed": NA,
            "evaluation_split": "standard_test",
            "held_out_source": NA,
            "validation_macro_f1": NA,
            "standard_test_macro_f1": historical_ensemble["metrics_per_model"][
                "soft_voting_ensemble"
            ]["macro_f1"],
            "loso_macro_f1": NA,
            "final_accuracy": historical_ensemble["metrics_per_model"]["soft_voting_ensemble"][
                "accuracy"
            ],
            "evidence_path": "docs/es/resultados/evidencia/ensamble_resumen.json",
            "notes": "Composición histórica; Lite0 desplegado no disponible en este respaldo",
        }
    )
    for seed in (42, 123, 2026):
        matches = [
            row for row in ensemble_rows if int(row["seed"]) == seed and row["model"] == "ensemble"
        ]
        classes = [
            row
            for row in ensemble_classes
            if int(row["seed"]) == seed and row["model"] == "ensemble"
        ]
        if len(matches) != 1:
            raise ValueError(f"Ensamble ambiguo: seed={seed}")
        row = matches[0]
        comparisons.append(
            {
                "model_id": f"loso_multicrop_ensemble_seed_{seed}",
                "architecture": "soft_voting_lite0_b0_shufflenet",
                "experiment": "loso_multicrop_disease_maiz",
                "seed": seed,
                "evaluation_split": "loso_holdout",
                "held_out_source": "multicrop-disease-maiz",
                "validation_macro_f1": NA,
                "standard_test_macro_f1": NA,
                "loso_macro_f1": row["macro_f1"],
                "final_accuracy": row["accuracy"],
                "final_ece_15_bins": row["ece_15_bins"],
                **class_values(classes, seed=seed, scenario="ensemble"),
                "evidence_path": (
                    "docs/es/reproducibilidad/evidencia/ensemble_loso/"
                    "ensemble_component_results.csv"
                ),
                "notes": (
                    "Composición de tres checkpoints verificados de la misma seed; "
                    "sin checkpoint único"
                ),
            }
        )

    put_csv(STAGE / "MODEL_INDEX.csv", INDEX_FIELDS, records)
    put_csv(STAGE / "MODEL_COMPARISON_RESULTS.csv", COMPARISON_FIELDS, comparisons)
    put_json(
        STAGE / "metadata/evidence_references.json",
        {
            "schema_version": 1,
            "checkpoint_count": len(records),
            "evidence": {
                str(path.relative_to(ROOT)): digest(path)
                for path in (
                    baseline_refs_path,
                    ensemble_refs_path,
                    hpo_lock_path,
                    HPO / "FINAL_TEST_COMPLETE.json",
                    STABILITY / "multiseed_summary.json",
                    ENSEMBLE / "ensemble_loso_summary.json",
                    ROOT / "docs/es/resultados/evidencia/ensamble_resumen.json",
                )
            },
            "limitations": [
                "El Lite0 desplegado del ensamble histórico no está disponible",
                "El test histórico procede de otra materialización",
                "Sin entrenamiento, inferencia ni selección de modelo durante este respaldo",
            ],
        },
    )
    put_json(
        STAGE / "metadata/FAILED_RUNS_MANIFEST.json",
        {
            "schema_version": 1,
            "excluded_from_model_index": [
                {
                    "experiment": "loso_maize_diseases",
                    "model": "efficientnet_lite0",
                    "seed": 123,
                    "remote_path": (
                        "corn-outputs:/loso/efficientnet_lite0_baseline/maize-diseases/"
                        "seed_123/interrupted_attempts/attempt_1/"
                    ),
                    "reason": "Intento interrumpido; no es el checkpoint final",
                },
                {
                    "experiment": "loso_multicrop_disease_maiz",
                    "model": "efficientnet_b0",
                    "seed": 42,
                    "local_path": (
                        "outputs/ensemble_loso/multicrop-disease-maiz/seed_42/"
                        "efficientnet_b0/interrupted_attempts/attempt_1/"
                    ),
                    "remote_path": (
                        "corn-outputs:/ensemble_loso/multicrop-disease-maiz/seed_42/"
                        "efficientnet_b0/interrupted_attempts/attempt_1/"
                    ),
                    "reason": "Intento interrumpido; la segunda run verificada es la incluida",
                },
            ],
        },
    )
    for source, target in (
        (ROOT / "docs/es/tesis/MODEL_PACKAGE_README.md", STAGE / "README.md"),
        (ROOT / "docs/es/tesis/EXPERIMENTAL_RESULTS_SUMMARY.md", STAGE / "RESULTS_SUMMARY.md"),
        (ROOT / "scripts/experiments/create_model_rars.sh", STAGE / "CREATE_RARS.sh"),
        (ROOT / "scripts/experiments/create_model_tars.sh", STAGE / "CREATE_TARS.sh"),
    ):
        copy_checked(source, target)
    checksums = "".join(f"{row['checkpoint_sha256']}  {row['local_path']}\n" for row in records)
    put_bytes(STAGE / "MODEL_CHECKSUMS.sha256", checksums.encode())
    print(f"Respaldo verificado: {len(records)} checkpoints; {len(comparisons)} filas comparativas")
    print(STAGE)


if __name__ == "__main__":
    main()
