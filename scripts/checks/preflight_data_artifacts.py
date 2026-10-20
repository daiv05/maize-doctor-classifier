"""Audita datos y artefactos congelados antes de una corrida formal.

No modifica datos ni splits. Usa la evidencia versionada de seed_42 como
referencia y no necesita importar PyTorch/Pandas. ``--report`` escribe JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = PROJECT_ROOT / "docs" / "es" / "reproducibilidad" / "evidencia"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class Inputs:
    dataset_root: Path
    output_root: Path
    splits_dir: Path
    expected_lock: Path
    expected_summary: Path
    config_path: Path
    exclusions_path: Path
    python_bin: Path
    reference_run_dir: Path | None = None
    reference_run_summary: Path | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def relative_image_path(raw: str) -> str:
    """Acepta únicamente rutas de imagen relativas al DATASET_ROOT."""
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not raw or path.is_absolute() or PureWindowsPath(raw).drive or ".." in path.parts:
        raise ValueError(f"image_path no es relativo al dataset: {raw!r}")
    return path.as_posix()


def expected_images(master: Path, exclusions: Path) -> dict[str, str]:
    images: dict[str, str] = {}
    for source in (master, exclusions):
        for row in csv_rows(source):
            relative = relative_image_path(row["image_path"])
            digest = row["sha256"].lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"SHA-256 inválido en {source}: {relative}")
            if relative in images:
                raise ValueError(f"image_path duplicado entre manifests/exclusiones: {relative}")
            images[relative] = digest
    return images


def scanned_images(dataset_root: Path, classes: list[str]) -> set[str]:
    """Repite el alcance de escaneo de create_splits: clase y entorno conocidos."""
    paths: set[str] = set()
    for class_name in classes:
        for environment in ("lab", "real"):
            directory = dataset_root / "clean" / class_name / environment
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.add(path.relative_to(dataset_root).as_posix())
    return paths


def run_preflight(inputs: Inputs, *, verify_image_hashes: bool = True) -> dict:
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add(
        "python_make",
        "ok" if inputs.python_bin.is_file() else "fail",
        str(inputs.python_bin),
    )
    add(
        "dataset_root",
        "ok" if inputs.dataset_root.is_dir() else "fail",
        str(inputs.dataset_root),
    )
    add(
        "clean",
        "ok" if (inputs.dataset_root / "clean").is_dir() else "fail",
        str(inputs.dataset_root / "clean"),
    )
    add(
        "output_root",
        "ok" if inputs.output_root.is_dir() else "fail",
        str(inputs.output_root),
    )

    expected_lock = None
    expected_counts = None
    try:
        expected_lock = json.loads(inputs.expected_lock.read_text(encoding="utf-8"))
        summary = json.loads(inputs.expected_summary.read_text(encoding="utf-8"))
        expected_counts = summary["seed_42"]
        expected_lock_hash = expected_counts["manifest_lock_sha256"]
        actual_reference_hash = sha256_file(inputs.expected_lock)
        if actual_reference_hash != expected_lock_hash:
            raise ValueError("el lock versionado no coincide con split_protocols_summary.json")
        add("evidencia_versionada", "ok", expected_lock_hash)
    except (OSError, ValueError, KeyError, TypeError) as error:
        add("evidencia_versionada", "fail", str(error))
        expected_lock = None
        expected_counts = None

    if expected_lock is not None:
        for name, path, field in (
            ("config", inputs.config_path, "config_sha256"),
            ("exclusiones", inputs.exclusions_path, "exclusions_sha256"),
        ):
            if not path.is_file():
                add(name, "fail", f"Falta {path}")
                continue
            actual = sha256_file(path)
            expected = expected_lock.get(field)
            if expected is None:
                add(name, "fail", f"Falta {field} en el lock versionado")
                continue
            status = "ok" if actual == expected else ("warning" if name == "config" else "fail")
            add(name, status, f"actual={actual}; lock={expected}")

    split_names = (
        "manifest.lock.json",
        "master_manifest.csv",
        "train.csv",
        "val.csv",
        "test.csv",
        "preparation_audit.json",
        "split_audit_report.csv",
    )
    missing = [name for name in split_names if not (inputs.splits_dir / name).is_file()]
    add(
        "artefactos_split",
        "fail" if missing else "ok",
        f"faltan {missing} en {inputs.splits_dir}" if missing else str(inputs.splits_dir),
    )

    split_verified = False
    if expected_lock is not None and expected_counts is not None and not missing:
        try:
            lock_path = inputs.splits_dir / "manifest.lock.json"
            lock_hash = sha256_file(lock_path)
            if lock_hash != expected_counts["manifest_lock_sha256"]:
                raise ValueError(f"lock diferente: {lock_hash}")
            for filename, field in (
                ("master_manifest.csv", "master_manifest_sha256"),
                ("train.csv", "train_sha256"),
                ("val.csv", "val_sha256"),
                ("test.csv", "test_sha256"),
            ):
                actual = sha256_file(inputs.splits_dir / filename)
                if actual != expected_lock[field]:
                    raise ValueError(f"{filename}: SHA-256 {actual} != {expected_lock[field]}")
            for name in ("train", "val", "test"):
                count = len(csv_rows(inputs.splits_dir / f"{name}.csv"))
                if count != expected_counts[name]:
                    raise ValueError(f"{name}.csv: {count} filas != {expected_counts[name]}")
            master_count = len(csv_rows(inputs.splits_dir / "master_manifest.csv"))
            if master_count != sum(expected_counts[name] for name in ("train", "val", "test")):
                raise ValueError(f"master_manifest.csv: {master_count} filas inesperadas")
            split_verified = True
            add("identidad_split", "ok", f"lock y cuatro CSV coinciden; filas={master_count}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            add("identidad_split", "fail", str(error))

    if split_verified and (inputs.dataset_root / "clean").is_dir():
        try:
            images = expected_images(
                inputs.splits_dir / "master_manifest.csv", inputs.exclusions_path
            )
            discovered = scanned_images(
                inputs.dataset_root, expected_lock["split_parameters"]["classes"]
            )
            missing_images = sorted(images.keys() - discovered)
            extra_images = sorted(discovered - images.keys())
            if missing_images or extra_images:
                raise ValueError(
                    f"faltan={len(missing_images)} {missing_images[:3]}; "
                    f"sobran={len(extra_images)} {extra_images[:3]}"
                )
            add("inventario_clean", "ok", f"{len(images)} rutas coinciden con master + exclusiones")
            if verify_image_hashes:
                mismatches = []
                for relative, expected_hash in images.items():
                    actual = sha256_file(inputs.dataset_root / relative)
                    if actual != expected_hash:
                        mismatches.append(relative)
                if mismatches:
                    add(
                        "contenido_clean", "fail",
                        f"{len(mismatches)} SHA distintos: {mismatches[:3]}",
                    )
                else:
                    add("contenido_clean", "ok", f"{len(images)} SHA-256 coinciden")
            else:
                add("contenido_clean", "warning", "SHA-256 de imágenes omitido (--quick)")
        except (OSError, ValueError, KeyError, TypeError) as error:
            add("inventario_clean", "fail", str(error))

    if inputs.reference_run_dir is not None:
        try:
            if inputs.reference_run_summary is None:
                raise ValueError("indique --reference-run-summary junto a --reference-run-dir")
            expected_summary_hash = sha256_file(inputs.reference_run_summary)
            local_summary = inputs.reference_run_dir / "summary.json"
            local_hash = sha256_file(local_summary)
            if local_hash != expected_summary_hash:
                raise ValueError(f"summary.json: {local_hash} != {expected_summary_hash}")
            expected_run = json.loads(inputs.reference_run_summary.read_text(encoding="utf-8"))
            expected_checkpoint = expected_run["checkpoint_sha256"]
            local_checkpoint = sha256_file(inputs.reference_run_dir / "best.pth")
            if local_checkpoint != expected_checkpoint:
                raise ValueError(f"best.pth: {local_checkpoint} != {expected_checkpoint}")
            add(
                "run_referencia", "ok",
                f"run={expected_run['run_id']}; checkpoint={local_checkpoint}",
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            add("run_referencia", "fail", str(error))

    status = "blocked" if any(check["status"] == "fail" for check in checks) else "ready"
    if status == "ready" and any(
        check["name"] == "contenido_clean" and check["status"] == "warning"
        for check in checks
    ):
        status = "incomplete"
    elif status == "ready" and any(check["status"] == "warning" for check in checks):
        status = "ready_with_warnings"
    return {"status": status, "checks": checks}


def dotenv_values(path: Path) -> dict[str, str]:
    """Lee solo claves de rutas de .env sin exponer tokens ni importar dependencias."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if separator and key in {"DATASET_ROOT", "OUTPUT_ROOT"}:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument(
        "--expected-lock", type=Path, default=EVIDENCE_ROOT / "seed_42_manifest_lock.json"
    )
    parser.add_argument(
        "--expected-summary", type=Path, default=EVIDENCE_ROOT / "split_protocols_summary.json"
    )
    parser.add_argument("--reference-run-dir", type=Path)
    parser.add_argument("--reference-run-summary", type=Path)
    parser.add_argument("--quick", action="store_true", help="Omite el SHA-256 de las 33k imágenes")
    parser.add_argument("--report", type=Path, help="Escribe también el resultado en JSON")
    args = parser.parse_args()
    if args.reference_run_summary is not None and args.reference_run_dir is None:
        parser.error("--reference-run-summary requiere --reference-run-dir")

    dotenv = dotenv_values(PROJECT_ROOT / ".env")
    raw_dataset = os.environ.get("DATASET_ROOT") or dotenv.get("DATASET_ROOT")
    raw_output = os.environ.get("OUTPUT_ROOT") or dotenv.get("OUTPUT_ROOT")
    dataset_root = args.dataset_root or (
        Path(raw_dataset) if raw_dataset else PROJECT_ROOT / "data"
    )
    output_root = args.output_root or (Path(raw_output) if raw_output else PROJECT_ROOT / "outputs")
    splits_dir = args.splits_dir or output_root / "splits" / "seed_42"
    python_bin = PROJECT_ROOT / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    result = run_preflight(
        Inputs(
            dataset_root=dataset_root,
            output_root=output_root,
            splits_dir=splits_dir,
            expected_lock=args.expected_lock,
            expected_summary=args.expected_summary,
            config_path=PROJECT_ROOT / "config" / "dataset.yaml",
            exclusions_path=PROJECT_ROOT / "config" / "dataset_exclusions.csv",
            python_bin=python_bin,
            reference_run_dir=args.reference_run_dir,
            reference_run_summary=args.reference_run_summary,
        ),
        verify_image_hashes=not args.quick,
    )
    for check in result["checks"]:
        print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
    print(f"Estado: {result['status']}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return {"ready": 0, "ready_with_warnings": 0, "blocked": 1, "incomplete": 2}[
        result["status"]
    ]


if __name__ == "__main__":
    sys.exit(main())
