"""Observa el HPO remoto y recoge/documenta evidencia solo tras el test final.

No entrena, relanza trials ni modifica Modal. Puede ejecutarse como servicio local
transitorio. Si el equipo se suspende, Modal sigue trabajando y la recogida espera.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

STUDY = "efficientnet_lite0_seed42_hpo_v1"
REMOTE = f"hpo/efficientnet_lite0/{STUDY}"
LOG = logging.getLogger("hpo-watch")


def modal_call(executable: str, *args: str, timeout=90):
    return subprocess.run([executable, *args], text=True, capture_output=True, timeout=timeout)


def fetch_json(executable: str, filename: str):
    result = modal_call(executable, "volume", "get", "corn-outputs", f"{REMOTE}/{filename}", "-")
    if result.returncode:
        # Un archivo aún no creado es normal. Otros errores deben conservarse.
        LOG.info("%s no disponible: %s", filename, result.stderr[-800:].strip())
        return None
    # Algunas versiones del CLI imprimen un mensaje de éxito detrás del JSON.
    return json.JSONDecoder().raw_decode(result.stdout.lstrip())[0]


def write_status(path: Path, state: str, **fields):
    payload = {"state": state, "timestamp": datetime.now(UTC).isoformat(), **fields}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    LOG.info("%s", json.dumps(payload))


def collect(executable: str, root: Path, receipts: Path) -> Path:
    receipts.mkdir(parents=True, exist_ok=True)
    receipt = Path(tempfile.mkdtemp(prefix="hpo-receipt-", dir=receipts))
    result = modal_call(
        executable, "volume", "get", "corn-outputs", REMOTE, str(receipt), timeout=3600
    )
    if result.returncode:
        raise RuntimeError(f"Descarga incompleta en {receipt}: {result.stderr[-2000:]}")
    # Modal conserva el nombre del directorio remoto al descargarlo a uno existente.
    study_dir = receipt / STUDY
    if not (study_dir / "study.db").is_file():
        raise RuntimeError(f"Falta SQLite en el estudio descargado: {study_dir}")
    report = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/pipeline/report_hpo.py"),
            "--study-dir",
            str(study_dir),
            "--update-docs",
        ],
        text=True,
        capture_output=True,
        timeout=300,
        cwd=root,
    )
    (receipt / "report_generation.log").write_text(report.stdout + report.stderr, encoding="utf-8")
    if report.returncode:
        raise RuntimeError(f"Evidencia no validada. Consultar {receipt / 'report_generation.log'}")
    return study_dir


def monitor(executable: str, app_id: str, root: Path, interval: int):
    monitor_dir = root / "outputs/hpo-monitor" / STUDY
    monitor_dir.mkdir(parents=True, exist_ok=True)
    status_path = monitor_dir / "status.json"
    stopped_checks = 0
    with (monitor_dir / ".watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                final = fetch_json(executable, "FINAL_TEST_COMPLETE.json")
                if final:
                    write_status(status_path, "collecting", app_id=app_id)
                    receipt = collect(executable, root, root / "outputs/hpo-receipts")
                    write_status(
                        status_path,
                        "complete",
                        app_id=app_id,
                        receipt=str(receipt),
                        report=str(receipt / "HPO_REPORT.md"),
                    )
                    return
                summary = fetch_json(executable, "study_summary.json")
                write_status(
                    status_path,
                    "running",
                    app_id=app_id,
                    n_recorded=summary.get("n_recorded", 0) if summary else 0,
                    best_validation=summary.get("best_value") if summary else None,
                )
                apps = modal_call(executable, "app", "list", "--json")
                if apps.returncode == 0:
                    active = any(
                        app.get("app_id") == app_id and not app.get("stopped_at")
                        for app in json.loads(apps.stdout)
                    )
                    stopped_checks = 0 if active else stopped_checks + 1
                    if stopped_checks >= 3:
                        write_status(
                            status_path,
                            "needs_audit",
                            app_id=app_id,
                            reason="App detenida sin test final cerrado; requiere auditoría.",
                        )
                        return
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                # Los fallos de red no demuestran que el entrenamiento se detuvo.
                write_status(status_path, "connection_retry", app_id=app_id, error=str(error))
            except Exception as error:
                write_status(status_path, "needs_audit", app_id=app_id, error=str(error))
                raise
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modal", required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("Usar intervalos de al menos 30 segundos.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    monitor(args.modal, args.app_id, Path(__file__).resolve().parents[2], args.interval)


if __name__ == "__main__":
    main()
