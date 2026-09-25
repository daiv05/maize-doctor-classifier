"""Cierra evidencia documental desde artefactos completos, sin cargar modelos/test.

Uso: python scripts/pipeline/report_hpo.py --study-dir outputs/hpo/... --update-docs
Acepta el presupuesto formal archivado (60 original / 25 enmendado), lock y test único.
"""

# Las tablas y saltos Markdown del template preservan líneas largas y dos espacios.
# ruff: noqa: E501, W291

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table(rows, columns):
    def cell(value):
        if value is None or value == "":
            return "N/A"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(cell(row.get(col)) for col in columns) + " |" for row in rows),
        ]
    )


def validate_evidence(directory: Path):
    summary = read_json(directory / "study_summary.json")
    lock = read_json(directory / "HPO_SELECTION_LOCK.json")
    final = read_json(directory / "FINAL_TEST_COMPLETE.json")
    trials = read_csv(directory / "trials.csv")
    if summary["study_name"] != "efficientnet_lite0_seed42_hpo_v1":
        raise ValueError("No es el study formal acordado.")
    target = summary["n_trials_requested"]
    if target not in (25, 60) or summary["n_recorded"] != target or len(trials) != target:
        raise ValueError(f"Aún no hay exactamente {target} intentos.")
    if target == 25:
        amendment = read_json(directory / "BUDGET_AMENDMENT.json")
        protocol = read_json(directory / "preflight.json")["protocol"]
        if (amendment["old_budget"], amendment["new_budget"]) != (60, 25):
            raise ValueError("La enmienda no autoriza 60 → 25.")
        if amendment["new_protocol"] != protocol or not amendment["reason"].strip():
            raise ValueError("Protocolo o motivo incompatible con la enmienda.")
        if lock.get("n_trials_requested") != target or protocol["n_trials"] != target:
            raise ValueError("Presupuesto del lock/preflight incompatible.")
        if amendment["test_used"] or datetime.fromisoformat(amendment["timestamp"]) > datetime.fromisoformat(lock["selection_timestamp"]):
            raise ValueError("La enmienda debe preceder selección/test.")
        for name, expected in amendment["prior_evidence_sha256"].items():
            if digest(directory / "budget_revisions/60-to-25" / name) != expected:
                raise ValueError(f"Evidencia previa alterada: {name}")
    if sorted(int(row["trial_number"]) for row in trials) != list(range(target)):
        raise ValueError("Identidades de trials incompletas o duplicadas.")
    if any(row["state"] not in {"COMPLETE", "PRUNED", "FAIL"} for row in trials):
        raise ValueError("Hay trials no terminales.")
    if summary["test_used_during_hpo"] or final["test_used_during_hpo"]:
        raise ValueError("Se declaró uso de test durante selección.")
    if lock["test_observed_before_selection"] or final["evaluation_count"] != 1:
        raise ValueError("Política de test incompatible.")
    started = read_json(directory / "FINAL_TEST_STARTED.json")
    if not (
        datetime.fromisoformat(lock["selection_timestamp"])
        <= datetime.fromisoformat(started["timestamp"])
        <= datetime.fromisoformat(final["finished_at"])
    ):
        raise ValueError("Test no es posterior al bloqueo de selección.")
    if final["selection_lock_sha256"] != digest(directory / "HPO_SELECTION_LOCK.json"):
        raise ValueError("Selection lock difiere del usado en test.")
    winner = max(
        (row for row in trials if row["state"] == "COMPLETE"),
        key=lambda row: (float(row["value"]), -int(row["trial_number"])),
    )
    if (
        int(winner["trial_number"]) != lock["best_trial"]
        or lock["best_trial"] != final["best_trial"]
    ):
        raise ValueError("Ganador incompatible con máximo validation.")
    if float(winner["value"]) != lock["best_validation_macro_f1"]:
        raise ValueError("Score ganador incompatible.")
    if summary["best_value"] != float(winner["value"]):
        raise ValueError("El resumen no coincide con el score ganador.")
    for state, key in (("COMPLETE", "n_complete"), ("PRUNED", "n_pruned"), ("FAIL", "n_failed")):
        if sum(row["state"] == state for row in trials) != summary[key]:
            raise ValueError("Conteos de estado incompatibles.")
    winner_summary = str(Path(lock["winner_checkpoint"]).parent / "summary.json")
    checks = {
        lock["winner_checkpoint"]: lock["winner_checkpoint_sha256"],
        "best_hyperparameters.json": lock["best_hyperparameters_sha256"],
        winner_summary: lock["winner_summary_sha256"],
        **{f"final_test/{name}": value for name, value in final["artifact_hashes"].items()},
    }
    for name, expected in checks.items():
        if digest(directory / name) != expected:
            raise ValueError(f"Hash inconsistente: {name}")
    before = read_json(directory / "preflight.json")["protocol"]["split_snapshot"]["sha256"]
    after = read_json(directory / "split_hashes_after.json")["sha256"]
    if before != after or before != lock["split_hashes"]:
        raise ValueError("Los hashes de splits cambiaron.")
    source = read_json(directory / "preflight.json")["source"]
    archive = read_json(directory / "source_archive.json")
    if digest(directory / "source_code.zip") != archive["archive_sha256"]:
        raise ValueError("Archivo de código alterado.")
    with zipfile.ZipFile(directory / "source_code.zip") as bundle:
        for name, expected in source["files"].items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise ValueError(f"Código archivado incompatible: {name}")
    with sqlite3.connect(f"file:{directory / 'study.db'}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite corrupto.")
        db_trials = connection.execute(
            "SELECT number, state FROM trials ORDER BY number"
        ).fetchall()
        if db_trials != [(int(row["trial_number"]), row["state"]) for row in trials]:
            raise ValueError("SQLite y trials.csv difieren.")
    return summary, lock, final, winner


def build_report(directory: Path) -> str:
    summary, lock, final, winner = validate_evidence(directory)
    top = read_csv(directory / "top_10_trials.csv")
    importance = read_csv(directory / "hyperparameter_importance.csv")
    classes = read_csv(directory / "final_test/test_classification_report.csv")
    calibration = read_json(directory / "final_test/test_calibration.json")
    preflight = read_json(directory / "preflight.json")
    delta = summary["best_value"] - summary["baseline_value"]
    best_params = read_json(directory / "best_hyperparameters.json")["best_params"]
    mapping = read_json(directory / Path(lock["winner_checkpoint"]).parent / "summary.json")[
        "class_to_idx"
    ]
    per_class = [
        {
            "class": row.get("", ""),
            **{k: row[k] for k in ("precision", "recall", "f1-score", "support")},
        }
        for row in classes
        if row.get("") in mapping
    ]
    npk = [
        row
        for row in per_class
        if row["class"] in {"nitrogen_deficiency", "phosphorus_deficiency", "potassium_deficiency"}
    ]
    top_regions = []
    for key in ("learning_rate", "weight_decay", "label_smoothing", "batch_size"):
        values = [float(row[key]) for row in top if row.get(key)]
        top_regions.append({"parameter": key, "top10_min": min(values), "top10_max": max(values)})
    hashes = [{"artefacto": k, "SHA256": v} for k, v in lock["split_hashes"].items()]
    hashes += [
        {"artefacto": key, "SHA256": digest(directory / key)}
        for key in (
            lock["winner_checkpoint"],
            "best_hyperparameters.json",
            "study.db",
            "trials.csv",
            "study_summary.json",
            "HPO_SELECTION_LOCK.json",
            "source_code.zip",
        )
    ]
    return f"""# HPO formal EfficientNet-Lite0 — resultado

## Study

study_name: `{summary["study_name"]}`  
status: COMPLETADO  
trials_requested: {summary["n_trials_requested"]}  
complete: {summary["n_complete"]}  
pruned: {summary["n_pruned"]}  
failed: {summary["n_failed"]}  
sampler: {summary["sampler"]} (seed 42, multivariate=true)  
pruner: {summary["pruner"]}, {summary["pruner_config"]}  
seed: 42  
objective: best_validation_macro_f1  
test_used_during_hpo: NO  
selection_timestamp: {lock["selection_timestamp"]}  
final_test_timestamp: {final["finished_at"]}

## Baseline y mejor trial

Baseline `20260921_204608`: validation_macro_f1 = {summary["baseline_value"]}.  
Trial ganador: {lock["best_trial"]}; validation_macro_f1 = {summary["best_value"]}.  
Mejora absoluta = {delta:+.9f}; puntos porcentuales = {delta * 100:+.6f}.  
best_epoch: {winner["best_epoch"]}.

{table([{"parameter": k, "value": v} for k, v in best_params.items()], ["parameter", "value"])}

Optimizer AdamW y scheduler cosine fijos; dropout de fábrica no buscado.
Son los mejores hiperparámetros encontrados dentro del espacio y presupuesto evaluados.

## Top 10

{table(top, ["rank", "trial", "val_macro_f1", "learning_rate", "weight_decay", "label_smoothing", "batch_size", "class_weights", "warmup_epochs", "best_epoch", "duration_seconds"])}

### Diagnósticos de validation

{table(top, ["trial", "best_train_macro_f1", "train_macro_f1_at_best_val", "train_val_gap", "best_train_val_gap", "validation_ece", "validation_mean_confidence_correct", "validation_mean_confidence_errors", "nitrogen_deficiency_f1", "phosphorus_deficiency_f1", "potassium_deficiency_f1"])}

Train se mide con augmentation y modo training; los gaps son diagnósticos, no otro objective.
La calibración de validation y N/P/K no intervinieron en la selección contractual.

### Región de mejores trials

{table(top_regions, ["parameter", "top10_min", "top10_max"])}

Estos rangos describen los trials mejor posicionados. No demuestran causalidad ni
estabilidad entre semillas; esa validación corresponde a una fase posterior.

## Convergencia

{table([{"corte": k, "best_validation": v} for k, v in summary["convergence"].items()], ["corte", "best_validation"])}

Última mejora: trial {lock["best_trial"]} (índice 0-based).
No se ejecuta trial 61 aunque siga existiendo potencial de mejora.

## Hyperparameter importance

{table(importance, ["parameter", "importance"]) if importance else "No estimable; consultar logs y número de COMPLETE."}

Importancia descriptiva del estudio; no causal. fANOVA usa seed 42.

## Evaluación final del ganador sobre test

accuracy: {final["metrics"]["accuracy"]}  
macro_f1: {final["metrics"]["macro_f1"]}  
ECE: {calibration["ece"]}  
evaluation_count: {final["evaluation_count"]}

{table(per_class, ["class", "precision", "recall", "f1-score", "support"])}

## N/P/K en test

{table(npk, ["class", "f1-score", "support"])}

Los desgloses por fuente/entorno y N/P/K agrupado están en `final_test/`.
El checkpoint evaluado corresponde a la mejor época del trial original; no se reentrenó.

## Artefactos y reproducibilidad

Raíz: `{summary["study_name"]}` en `corn-outputs:/hpo/efficientnet_lite0/`.
Incluye `study.db`, `trials.csv`, `study_summary.json`, `best_hyperparameters.json`,
`HPO_SELECTION_LOCK.json`, `trials/{winner["trial_id"]}/best.pth`, `figures/`,
`final_test/`, `FINAL_TEST_COMPLETE.json` y `source_code.zip`.

{table(hashes, ["artefacto", "SHA256"])}

{table([{"component": k, "version": v} for k, v in preflight["environment"].items()], ["component", "version"])}

## Veredicto

¿Optuna superó el baseline en Validation Macro-F1? {"Sí" if delta > 0 else "No"}  
¿Test fue utilizado durante selección? No  
¿Se completó el presupuesto de {summary["n_trials_requested"]} trials? Sí  
¿Study puede reanudarse? Sí; presupuesto agotado, no admite más trials en esta fase.  
¿Existe evidencia suficiente para reproducir la búsqueda? Sí, con el corpus congelado y entorno registrado.

El resultado es un HPO winner. No acredita producción ni entrenamiento formal,
multi-seed, CV, LOSO, ensemble o temperature scaling.
"""


def update_docs(directory: Path, root: Path, report: str):
    """Añade evidencia; no sustituye tablas ni texto histórico."""
    summary, lock, final, _ = validate_evidence(directory)
    docs = root / "docs/es"
    evidence = docs / "reproducibilidad/evidencia/hpo_lite0_seed42"
    evidence.mkdir(parents=True, exist_ok=True)
    names = [
        "preflight.json",
        "study_summary.json",
        "trials.csv",
        "top_10_trials.csv",
        "convergence.json",
        "hyperparameter_importance.csv",
        "best_hyperparameters.json",
        "HPO_SELECTION_LOCK.json",
        "FINAL_TEST_STARTED.json",
        "FINAL_TEST_COMPLETE.json",
        "source_archive.json",
        "split_hashes_after.json",
        "comparison_vs_baseline.csv",
    ]
    for name in names:
        shutil.copy2(directory / name, evidence / name)
    if (directory / "BUDGET_AMENDMENT.json").exists():
        shutil.copy2(directory / "BUDGET_AMENDMENT.json", evidence / "BUDGET_AMENDMENT.json")
        previous = evidence / "budget_revisions/60-to-25"
        previous.mkdir(parents=True, exist_ok=True)
        for name in ("preflight.json", "source_archive.json", "study_summary.json", "trials.csv"):
            shutil.copy2(directory / "budget_revisions/60-to-25" / name, previous / name)
    winner_summary = Path(lock["winner_checkpoint"]).parent / "summary.json"
    (evidence / winner_summary).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(directory / winner_summary, evidence / winner_summary)
    shutil.copytree(directory / "final_test", evidence / "final_test", dirs_exist_ok=True)
    relative = "../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md"
    (evidence / "HPO_REPORT.md").write_text(report, encoding="utf-8")
    marker = "<!-- hpo-lite0-seed42-completed -->"
    date = final["finished_at"][:10]
    target = summary["n_trials_requested"]
    winner_line = (
        f"Study `{summary['study_name']}`: {target} intentos, ganador trial {lock['best_trial']}, "
        f"validation Macro-F1 {summary['best_value']:.9f}; "
        f"baseline {summary['baseline_value']:.9f}; "
        f"delta {(summary['best_value'] - summary['baseline_value']) * 100:+.6f} pp."
    )
    additions = {
        "experimentos/hpo.md": f"## Resultados completados — {date}\n\n{winner_line}\n\n"
        f"Test único posterior al lock. [Entrega completa, Top 10 y hashes]({relative}).",
        "tesis/PROJECT_EVOLUTION.md": f"## {date} — HPO formal EfficientNet-Lite0\n\n"
        f"Se completó Optuna con {target} intentos para optimizar sistemáticamente el baseline. {winner_line}\n\n"
        f"Se exploraron LR, WD, smoothing, batch, pesos de clase y warmup; test permaneció cerrado "
        f"hasta congelar ganador/checkpoint. [Parámetros, test final y evidencia]({relative}).",
        "tesis/DECISION_LOG.md": f"## DEC-012 — Cierre experimental ({date})\n\n{winner_line}\n\n"
        f"[Configuración seleccionada y resultados]({relative}).",
        "tesis/EVIDENCE_REGISTRY.md": f"## HPO formal Lite0 — {date}\n\n{winner_line}\n\n"
        f"[Registro con storage, trials, parámetros, lock, checkpoint, figuras y test final]({relative}). "
        "Base y pesos completos en `corn-outputs:/hpo/efficientnet_lite0/efficientnet_lite0_seed42_hpo_v1/`.",
        "tesis/MILESTONES.md": f"## M16 — cierre verificado ({date})\n\n"
        f"**COMPLETADO: HPO Optuna {target} trials.** {winner_line}\n\n[Evidencia]({relative}).",
        "reproducibilidad/figuras.md": f"## HPO Lite0 seed42 — {date}\n\n"
        "Study `efficientnet_lite0_seed42_hpo_v1`; generador "
        "`src/training/tuning.py::save_optimization_plots`; fuente `study.db`/`trials.csv`. "
        "Código exacto en `source_code.zip`.\n\n"
        + table(
            [
                {"figure": p.name, "SHA256": digest(p)}
                for p in (directory / "figures").glob("*.png")
            ],
            ["figure", "SHA256"],
        ),
    }
    for relative_path, addition in additions.items():
        path = docs / relative_path
        original = path.read_text(encoding="utf-8")
        if marker not in original:
            if relative_path == "experimentos/hpo.md":
                original = original.replace(
                    "**Estado al 2026-09-23: EN EJECUCIÓN; los 60 trials formales aún no se declaran completados.**",
                    f"**Estado al {date}: COMPLETADO; {target} intentos y test único del ganador verificados.**",
                ).replace(
                    "**Estado al 2026-09-23: EN EJECUCIÓN; presupuesto revisado a 25 intentos totales.**",
                    f"**Estado al {date}: COMPLETADO; {target} intentos y test único del ganador verificados.**",
                ).replace(
                    "Los resultados siguen pendientes de completar 60 intentos y el test final. No hay ganador ni mejora\n"
                    "formal declarados en este corte.",
                    "El corte inicial del 2026-09-23 todavía no tenía ganador ni mejora formal declarados. "
                    "El resultado definitivo se documenta en «Resultados completados», al final de esta página.",
                ).replace(
                    "El cierre sigue pendiente de completar 25 intentos y el test final; los resultados\n"
                    "actuales son provisionales.",
                    "El presupuesto revisado de 25 intentos y el test único ya están verificados; "
                    "véase «Resultados completados» al final de esta página.",
                )
            if relative_path == "tesis/MILESTONES.md":
                original = original.replace(
                    "| M16 | pendiente | Optuna 60 trials | plan HPO | PENDIENTE |",
                    f"| M16 | {date} | Optuna {target} trials | HPO_REPORT.md | COMPLETADO |",
                )
            path.write_text(original.rstrip() + f"\n\n{marker}\n\n{addition}\n", encoding="utf-8")
    figures = root / "public/resultados/hpo_lite0_seed42"
    shutil.copytree(directory / "figures", figures, dirs_exist_ok=True)
    # Tabla nueva: no altera plantillas/tablas históricas.
    table_path = docs / "tesis/HPO_BASELINE_COMPARISON.md"
    table_path.write_text(
        "# Baseline vs HPO winner\n\n"
        + table(
            [
                {
                    "model": "Baseline 20260921_204608",
                    "validation_macro_f1": summary["baseline_value"],
                },
                {
                    "model": f"HPO trial {lock['best_trial']}",
                    "validation_macro_f1": summary["best_value"],
                },
            ],
            ["model", "validation_macro_f1"],
        )
        + f"\n\n[Evidencia]({relative}).\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--update-docs", action="store_true")
    args = parser.parse_args()
    report = build_report(args.study_dir)
    (args.study_dir / "HPO_REPORT.md").write_text(report, encoding="utf-8")
    if args.update_docs:
        update_docs(args.study_dir, Path(__file__).resolve().parents[2], report)
    print(report)


if __name__ == "__main__":
    main()
