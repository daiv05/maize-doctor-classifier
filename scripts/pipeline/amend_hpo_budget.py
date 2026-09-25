"""Migra una COPIA local y detenida del HPO de 60 a 25; nunca escribe en Modal.

Exige autorización explícita, conserva DB/código/protocolo previos y no altera
trials ni RNG. Subir la copia validada requiere detener antes todos los escritores.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import shutil
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from src.config import PROJECT_ROOT
from src.data.preparation import atomic_write_json, sha256_file, sha256_json
from src.training.tuning_study import FORMAL_STUDY, source_snapshot

ALLOWED_FILES = {
    "scripts/pipeline/tune.py",
    "scripts/modal/train.py",
    "src/training/tuning.py",
}


def trial_fingerprint(connection):
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'trial%' ORDER BY name"
    ).fetchall()
    return sha256_json({
        name: connection.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
        for (name,) in tables
    })


def amend(source_dir: Path, destination: Path, root: Path, reason: str):
    if not reason.strip():
        raise ValueError("Se requiere el motivo/autorización de la enmienda.")
    if destination.exists():
        raise ValueError("El destino debe ser nuevo; no se sobrescribe ninguna copia.")
    for name in ("HPO_SELECTION_LOCK.json", "FINAL_TEST_STARTED.json", "FINAL_TEST_COMPLETE.json"):
        if (source_dir / name).exists():
            raise ValueError("No se permite reducir el presupuesto después de selección/test.")
    before = json.loads((source_dir / "preflight.json").read_text())
    protocol = before["protocol"]
    if protocol["study_name"] != FORMAL_STUDY or protocol["n_trials"] != 60:
        raise ValueError("Solo se autoriza la revisión explícita de este study: 60 → 25.")
    source = source_snapshot(root)
    old_files = before["source"]["files"]
    changed = {
        name for name in old_files.keys() | source["files"].keys()
        if old_files.get(name) != source["files"].get(name)
    }
    if not changed or not changed <= ALLOWED_FILES:
        raise ValueError(f"Cambios ajenos al presupuesto: {sorted(changed - ALLOWED_FILES)}")
    if sha256_file(root / "config/dataset.yaml") != protocol["config_sha256"]:
        raise ValueError("La configuración cambió.")
    archive = json.loads((source_dir / "source_archive.json").read_text())
    if sha256_file(source_dir / "source_code.zip") != archive["archive_sha256"]:
        raise ValueError("El archivo de código anterior está alterado.")
    with zipfile.ZipFile(source_dir / "source_code.zip") as bundle:
        import hashlib

        for name, expected in old_files.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise ValueError(f"Fuente anterior alterada: {name}")
        # No basta permitir tuning.py entero: su objetivo, transforms, pérdida y
        # espacio deben conservar exactamente el AST anterior.
        training_file = "src/training/tuning.py"
        old_tree = ast.parse(bundle.read(training_file))
        new_tree = ast.parse((root / training_file).read_text())
        for tree in (old_tree, new_tree):
            tree.body = [
                node for node in tree.body
                if not isinstance(node, ast.FunctionDef)
                or node.name not in {"write_selection_lock", "export_tuning_artifacts"}
            ]
        if ast.dump(old_tree) != ast.dump(new_tree):
            raise ValueError("La lógica de entrenamiento cambió; se rechaza la migración.")
    with sqlite3.connect(f"file:{source_dir / 'study.db'}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite anterior no íntegro.")
        if connection.execute("SELECT study_name FROM studies").fetchall() != [(FORMAL_STUDY,)]:
            raise ValueError("La base contiene otros studies.")
        attrs = dict(connection.execute("SELECT key, value_json FROM study_user_attributes"))
        if json.loads(attrs["protocol"]) != protocol or "sampler_state" not in attrs:
            raise ValueError("Protocolo SQLite/preflight o RNG incompatibles.")
        trials = connection.execute("SELECT number, state FROM trials ORDER BY number").fetchall()
        if len(trials) > 25 or any(state == "WAITING" for _, state in trials):
            raise ValueError("Los trials existentes exceden/no admiten el nuevo presupuesto.")
        fingerprint = trial_fingerprint(connection)
    after = copy.deepcopy(before)
    after["source"] = source
    after["protocol"].update(n_trials=25, source_sha256=source["sha256"])
    shutil.copytree(source_dir, destination)
    revisions = destination / "budget_revisions/60-to-25"
    revisions.mkdir(parents=True)
    for name in ("study.db", "preflight.json", "source_code.zip", "source_archive.json",
                 "study_summary.json", "trials.csv"):
        shutil.copy2(source_dir / name, revisions / name)
    with sqlite3.connect(destination / "study.db") as connection:
        connection.execute(
            "UPDATE study_user_attributes SET value_json=? WHERE key='protocol'",
            (json.dumps(after["protocol"]),),
        )
        if trial_fingerprint(connection) != fingerprint:
            raise ValueError("Trials alterados; no utilizar la copia de destino.")
        sampler = connection.execute(
            "SELECT value_json FROM study_user_attributes WHERE key='sampler_state'"
        ).fetchone()[0]
        if sampler != attrs["sampler_state"]:
            raise ValueError("RNG alterado; no utilizar la copia de destino.")
    atomic_write_json(destination / "preflight.json", after)
    with zipfile.ZipFile(destination / "source_code.zip", "w") as bundle:
        for name in [*source["files"], "config/dataset.yaml"]:
            bundle.writestr(zipfile.ZipInfo(name), (root / name).read_bytes(),
                            compress_type=zipfile.ZIP_DEFLATED)
    atomic_write_json(destination / "source_archive.json", {
        "source_sha256": source["sha256"],
        "archive_sha256": sha256_file(destination / "source_code.zip"),
    })
    summary = json.loads((destination / "study_summary.json").read_text())
    summary["n_trials_requested"] = 25
    summary["status"] = "paused_for_budget_amendment"
    atomic_write_json(destination / "study_summary.json", summary)
    amendment = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "study_name": FORMAL_STUDY,
        "old_budget": 60,
        "new_budget": 25,
        "reason": reason,
        "trials_at_amendment": trials,
        "trial_tables_sha256": fingerprint,
        "sampler_state_sha256": sha256_json(json.loads(sampler)),
        "changed_source_files": sorted(changed),
        "old_source_sha256": before["source"]["sha256"],
        "new_source_sha256": source["sha256"],
        "old_protocol": protocol,
        "new_protocol": after["protocol"],
        "prior_evidence_sha256": {
            p.name: sha256_file(p) for p in revisions.iterdir() if p.is_file()
        },
        "test_used": False,
    }
    atomic_write_json(destination / "BUDGET_AMENDMENT.json", amendment)
    return amendment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--confirm-writer-stopped", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(amend(args.source_dir, args.destination, PROJECT_ROOT, args.reason), indent=2))


if __name__ == "__main__":
    main()
