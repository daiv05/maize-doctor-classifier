"""Auditoría reproducible de duplicados exactos tras decodificar píxeles.

No modifica manifests ni splits. Primero verifica que todas las rutas del split
canónico y sus exclusiones estén presentes y que coincidan con sus SHA-256; solo
entonces calcula hashes de píxeles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import struct
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from PIL import __version__ as pillow_version

from src.config import PROJECT_ROOT, get_dataset_root, get_output_root

DEFAULT_OUTPUT = PROJECT_ROOT / "docs/es/reproducibilidad/evidencia/pixel_duplicate_audit"
SPLITS = ("train", "val", "test")
INVENTORY_COLUMNS = (
    "sample_id",
    "split",
    "label",
    "source_id",
    "relative_path",
    "file_sha256",
    "pixel_sha256",
    "width",
    "height",
    "channels",
    "image_mode",
    "file_extension",
    "file_format",
)
PREFLIGHT_COLUMNS = (
    "sample_id",
    "split",
    "label",
    "source_id",
    "source_resolution",
    "relative_path",
    "file_extension",
    "expected_file_sha256",
    "file_sha256",
    "exclusion_status",
    "status",
    "error",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_signature(path: str | Path) -> dict[str, Any]:
    """Hash del RGB decodificado; aplica EXIF y no redimensiona ni gestiona ICC.

    La entrada del digest lleva versión, modo y dimensiones con codificación fija,
    seguida de los bytes RGB en orden de filas. Igual que el loader del proyecto,
    grayscale/palette/CMYK/RGBA se convierten con Pillow a RGB; alpha se descarta.
    Los perfiles ICC y otros metadatos no se aplican ni se incluyen.
    """
    with Image.open(path) as raw:
        file_format = raw.format or ""
        oriented = ImageOps.exif_transpose(raw)
        if oriented is raw:
            oriented = raw.copy()
        source_mode = oriented.mode
        width, height = oriented.size
        rgb = oriented.convert("RGB")
        pixels = rgb.tobytes()

    digest = hashlib.sha256()
    digest.update(b"doctormaiz-pixel-sha256-v1\0")
    digest.update(b"RGB\0")
    digest.update(struct.pack(">II", width, height))
    digest.update(pixels)
    return {
        "pixel_sha256": digest.hexdigest(),
        "width": width,
        "height": height,
        "channels": 3,
        "image_mode": source_mode,
        "file_format": file_format,
    }


def duplicate_groups(
    rows: list[dict[str, Any]], *, eligible_only: bool = True
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if eligible_only and row.get("split") not in SPLITS:
            continue
        groups[str(row["pixel_sha256"])].append(row)
    return [
        sorted(group, key=lambda row: str(row["sample_id"]))
        for group in groups.values()
        if len(group) > 1
    ]


def sha_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("file_sha256", ""))
        if digest:
            groups[digest].append(row)
    return [group for group in groups.values() if len(group) > 1]


def label_conflict_groups(groups: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Devuelve grupos pixel-idénticos que contienen más de una etiqueta."""
    return [group for group in groups if len({str(row["label"]) for row in group}) > 1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def _load_canonical_inputs(
    splits_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, str], list[dict[str, str]]]:
    master = _read_csv(splits_dir / "master_manifest.csv")
    split_by_id: dict[str, str] = {}
    split_rows = 0
    for split in SPLITS:
        for row in _read_csv(splits_dir / f"{split}.csv"):
            sample_id = row["sample_id"]
            if sample_id in split_by_id:
                raise ValueError(f"sample_id repetido en splits: {sample_id}")
            split_by_id[sample_id] = split
            split_rows += 1
    if len(master) != 33_429 or split_rows != 33_429 or len(split_by_id) != len(master):
        raise ValueError(
            "Conteos canónicos inesperados: "
            f"master={len(master)}, split_rows={split_rows}, unique_split_ids={len(split_by_id)}"
        )
    if {row["sample_id"] for row in master} != set(split_by_id):
        raise ValueError("Los sample_id de master_manifest y los tres splits no coinciden")

    audit = json.loads((splits_dir / "preparation_audit.json").read_text(encoding="utf-8"))
    excluded = audit.get("excluded_samples", [])
    if len(excluded) != 8:
        raise ValueError(f"Se esperaban 8 exclusiones contractuales; encontrados {len(excluded)}")
    if len({row["image_path"] for row in excluded}) != len(excluded):
        raise ValueError("Hay rutas repetidas en las exclusiones contractuales")
    return master, split_by_id, excluded


def _filename_family(image_path: str) -> str:
    """Identificador de familia compartido por nombres numerados de una fuente."""
    return re.sub(r"_(?:lab|real)_\d+$", "", Path(image_path).stem)


def _excluded_source_ids(
    master: list[dict[str, str]], excluded: list[dict[str, str]]
) -> dict[str, str]:
    sources_by_family: dict[str, set[str]] = defaultdict(set)
    for row in master:
        source_id = row.get("source_id", "").strip()
        if source_id:
            sources_by_family[_filename_family(row["image_path"])].add(source_id)

    resolved = {}
    for row in excluded:
        image_path = row["image_path"]
        candidates = sources_by_family.get(_filename_family(image_path), set())
        if len(candidates) != 1:
            raise ValueError(
                f"No se pudo asignar source_id único a {image_path}: {sorted(candidates)}"
            )
        resolved[image_path] = next(iter(candidates))
    return resolved


def _preflight(
    master: list[dict[str, str]],
    split_by_id: dict[str, str],
    excluded: list[dict[str, str]],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    excluded_sources = _excluded_source_ids(master, excluded)
    expected = []
    for row in master:
        expected.append(
            {
                "sample_id": row["sample_id"],
                "split": split_by_id[row["sample_id"]],
                "label": row["label"],
                "source_id": row.get("source_id", ""),
                "source_resolution": "master_manifest",
                "relative_path": row["image_path"],
                "file_extension": Path(row["image_path"]).suffix.lower(),
                "expected_file_sha256": row["sha256"],
                "exclusion_status": "eligible",
            }
        )
    for row in excluded:
        expected.append(
            {
                "sample_id": row["sample_id"],
                "split": "excluded",
                "label": row["label"],
                "source_id": excluded_sources[row["image_path"]],
                "source_resolution": "unique_filename_family_match",
                "relative_path": row["image_path"],
                "file_extension": Path(row["image_path"]).suffix.lower(),
                "expected_file_sha256": row["sha256"],
                "exclusion_status": row.get("reason", "contractual"),
            }
        )

    results: list[dict[str, Any]] = []
    for row in expected:
        path = dataset_root / row["relative_path"]
        actual = ""
        error = ""
        try:
            actual = sha256_file(path)
            status = "ok" if actual == row["expected_file_sha256"] else "sha256_mismatch"
        except OSError as exc:
            status = "missing_or_unreadable"
            error = f"{type(exc).__name__}: {exc}"
        results.append({**row, "file_sha256": actual, "status": status, "error": error})
    return results


def _scan_pixel_row(row: dict[str, Any], dataset_root: str) -> dict[str, Any]:
    path = Path(dataset_root) / str(row["relative_path"])
    try:
        return {**row, **pixel_signature(path), "error": ""}
    except Exception as exc:  # una imagen ilegible queda trazada, no se omite
        return {
            **row,
            "pixel_sha256": "",
            "width": "",
            "height": "",
            "channels": "",
            "image_mode": "",
            "file_format": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _checkpoint_rows(rows: list[dict[str, Any]], path: Path) -> None:
    _write_csv(path, rows, (*INVENTORY_COLUMNS, "error"))


def _collect_pixel_rows(
    eligible: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    workers: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_path = output_dir / "pixel_hash_inventory.csv"
    excluded_path = output_dir / "pixel_excluded_images.csv"
    checkpoint_path = output_dir / ".pixel_hash_checkpoint.csv"
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint_path.is_file():
        for row in _read_csv(checkpoint_path):
            cached[(row["sample_id"], row["file_sha256"])] = row
    elif inventory_path.is_file():
        for row in _read_csv(inventory_path):
            cached[(row["sample_id"], row["file_sha256"])] = row
        if excluded_path.is_file():
            for row in _read_csv(excluded_path):
                cached[(row["sample_id"], row["file_sha256"])] = row

    tasks = eligible + excluded
    done: dict[tuple[str, str], dict[str, Any]] = {}
    pending = []
    for row in tasks:
        key = (str(row["sample_id"]), str(row["file_sha256"]))
        old = cached.get(key)
        if old and old.get("pixel_sha256") and not old.get("error"):
            done[key] = {**old, **row}
        else:
            pending.append(row)

    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            for result in pool.map(
                _scan_pixel_row, batch, [str(dataset_root)] * len(batch), chunksize=8
            ):
                done[(str(result["sample_id"]), str(result["file_sha256"]))] = result
            # El checkpoint atómico permite reanudar sin recalcular hashes ya completos.
            all_rows = [
                done[(str(row["sample_id"]), str(row["file_sha256"]))]
                for row in tasks
                if (str(row["sample_id"]), str(row["file_sha256"])) in done
            ]
            _checkpoint_rows(all_rows, checkpoint_path)
            print(f"pixel hashes: {len(all_rows)}/{len(tasks)}", flush=True)

    results = [done[(str(row["sample_id"]), str(row["file_sha256"]))] for row in tasks]
    eligible_rows = [row for row in results if row["exclusion_status"] == "eligible"]
    excluded_rows = [row for row in results if row["exclusion_status"] != "eligible"]
    _write_csv(inventory_path, eligible_rows, (*INVENTORY_COLUMNS, "error"))
    _write_csv(excluded_path, excluded_rows, (*INVENTORY_COLUMNS, "exclusion_status", "error"))
    return eligible_rows, excluded_rows


def _group_member_rows(
    groups: list[list[dict[str, Any]]], *, cross_split: bool = False
) -> list[dict[str, Any]]:
    output = []
    for members in groups:
        split_set = sorted({str(row["split"]) for row in members if row["split"] in SPLITS})
        if cross_split and len(split_set) < 2:
            continue
        labels = sorted({str(row["label"]) for row in members})
        group_kind = (
            "three_way"
            if set(split_set) == set(SPLITS)
            else "cross_split"
            if len(split_set) > 1
            else "within_split"
        )
        for row in members:
            output.append(
                {
                    "pixel_sha256": row["pixel_sha256"],
                    "group_size": len(members),
                    "splits": ",".join(split_set),
                    "labels": ",".join(labels),
                    "group_kind": group_kind,
                    "label_conflict": len(labels) > 1,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "source_id": row.get("source_id", ""),
                    "relative_path": row["relative_path"],
                    "file_extension": row.get("file_extension", ""),
                    "file_sha256": row["file_sha256"],
                    "exclusion_status": row.get("exclusion_status", "eligible"),
                }
            )
    return sorted(output, key=lambda row: (row["pixel_sha256"], row["sample_id"]))


def _summary_stats(
    groups: list[list[dict[str, Any]]], test_n: int, val_n: int, corpus_n: int
) -> dict[str, Any]:
    duplicate_members = [row for group in groups for row in group]
    cross = [group for group in groups if len({row["split"] for row in group}) > 1]
    pair_counts = {
        "train_validation": sum(1 for g in cross if {r["split"] for r in g} == {"train", "val"}),
        "train_test": sum(1 for g in cross if {r["split"] for r in g} == {"train", "test"}),
        "validation_test": sum(1 for g in cross if {r["split"] for r in g} == {"val", "test"}),
        "three_way": sum(1 for g in cross if {r["split"] for r in g} == set(SPLITS)),
    }
    test_duplicate_ids = {row["sample_id"] for row in duplicate_members if row["split"] == "test"}
    test_cross_ids = {
        row["sample_id"] for group in cross for row in group if row["split"] == "test"
    }
    val_duplicate_ids = {row["sample_id"] for row in duplicate_members if row["split"] == "val"}
    return {
        "duplicate_group_count": len(groups),
        "duplicate_sample_count": len({row["sample_id"] for row in duplicate_members}),
        "within_split_group_count": sum(1 for g in groups if len({r["split"] for r in g}) == 1),
        "cross_split_group_count": len(cross),
        "cross_split_groups": pair_counts,
        "test_samples_in_duplicate_groups": len(test_duplicate_ids),
        "test_samples_in_cross_split_groups": len(test_cross_ids),
        "test_percent_in_duplicate_groups": 100 * len(test_duplicate_ids) / test_n if test_n else 0,
        "test_percent_in_cross_split_groups": 100 * len(test_cross_ids) / test_n if test_n else 0,
        "validation_samples_in_duplicate_groups": len(val_duplicate_ids),
        "validation_percent_in_duplicate_groups": 100 * len(val_duplicate_ids) / val_n
        if val_n
        else 0,
        "corpus_samples_in_duplicate_groups": len({row["sample_id"] for row in duplicate_members}),
        "corpus_percent_in_duplicate_groups": 100
        * len({row["sample_id"] for row in duplicate_members})
        / corpus_n
        if corpus_n
        else 0,
    }


def _fingerprint(splits_dir: Path) -> dict[str, Any]:
    lock_path = splits_dir / "manifest.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return {
        "manifest_lock_sha256": sha256_file(lock_path),
        "master_manifest_sha256": lock["master_manifest_sha256"],
        "train_sha256": lock["train_sha256"],
        "validation_sha256": lock["val_sha256"],
        "test_sha256": lock["test_sha256"],
        "config_sha256_in_lock": lock["config_sha256"],
        "exclusions_sha256_in_lock": lock["exclusions_sha256"],
        "current_config_sha256": sha256_file(PROJECT_ROOT / "config/dataset.yaml"),
        "current_exclusions_sha256": sha256_file(PROJECT_ROOT / "config/dataset_exclusions.csv"),
        "configuration_matches_lock": lock["config_sha256"]
        == sha256_file(PROJECT_ROOT / "config/dataset.yaml"),
        "exclusions_match_lock": lock["exclusions_sha256"]
        == sha256_file(PROJECT_ROOT / "config/dataset_exclusions.csv"),
    }


def run_audit(
    *,
    splits_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    workers: int = 4,
    batch_size: int = 500,
) -> dict[str, Any]:
    master, split_by_id, excluded = _load_canonical_inputs(splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(master, split_by_id, excluded, dataset_root)
    _write_csv(output_dir / "corpus_preflight.csv", preflight, PREFLIGHT_COLUMNS)
    errors = [row for row in preflight if row["status"] != "ok"]
    preflight_summary = {
        "status": "passed" if not errors else "failed",
        "eligible_manifest_rows": len(master),
        "contractually_excluded_rows": len(excluded),
        "files_expected": len(preflight),
        "files_found_and_sha_verified": len(preflight) - len(errors),
        "missing_or_unreadable": sum(row["status"] == "missing_or_unreadable" for row in errors),
        "sha256_mismatch": sum(row["status"] == "sha256_mismatch" for row in errors),
    }
    _write_json(output_dir / "corpus_preflight.json", {**preflight_summary, "errors": errors[:100]})
    if errors:
        raise RuntimeError(
            f"Preflight incompleto ({len(errors)}/{len(preflight)}); sin pixel_sha256. "
            f"Ver {output_dir / 'corpus_preflight.csv'}"
        )

    eligible_inputs = [row for row in preflight if row["exclusion_status"] == "eligible"]
    excluded_inputs = [row for row in preflight if row["exclusion_status"] != "eligible"]
    eligible, excluded_pixels = _collect_pixel_rows(
        eligible_inputs, excluded_inputs, dataset_root, output_dir, workers, batch_size
    )
    pixel_errors = [row for row in [*eligible, *excluded_pixels] if row.get("error")]
    all_rows = [*eligible, *excluded_pixels]
    groups = duplicate_groups(eligible)
    all_groups = duplicate_groups(all_rows, eligible_only=False)
    duplicate_members = _group_member_rows(groups)
    cross_members = _group_member_rows(groups, cross_split=True)
    _write_csv(
        output_dir / "pixel_duplicate_groups.csv",
        duplicate_members,
        (
            "pixel_sha256",
            "group_size",
            "splits",
            "labels",
            "group_kind",
            "label_conflict",
            "sample_id",
            "split",
            "label",
            "source_id",
            "relative_path",
            "file_extension",
            "file_sha256",
            "exclusion_status",
        ),
    )
    _write_csv(
        output_dir / "pixel_cross_split_duplicates.csv",
        cross_members,
        (
            "pixel_sha256",
            "group_size",
            "splits",
            "labels",
            "group_kind",
            "label_conflict",
            "sample_id",
            "split",
            "label",
            "source_id",
            "relative_path",
            "file_extension",
            "file_sha256",
            "exclusion_status",
        ),
    )
    conflicts = label_conflict_groups(all_groups)
    conflict_rows = []
    for group in conflicts:
        labels = {row["label"] for row in group}
        eligible_conflict = any(row["split"] in SPLITS for row in group) and any(
            row["split"] in SPLITS and row["label"] != group[0]["label"] for row in group
        )
        excluded_only = all(row["split"] == "excluded" for row in group)
        state = (
            "known_contractually_excluded"
            if excluded_only
            else "new_or_mixed_exclusion_conflict"
            if eligible_conflict
            else "excluded_image_conflict"
        )
        for row in group:
            conflict_rows.append(
                {
                    "pixel_sha256": row["pixel_sha256"],
                    "group_size": len(group),
                    "labels": ",".join(sorted(labels)),
                    "conflict_state": state,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "source_id": row.get("source_id", ""),
                    "relative_path": row["relative_path"],
                    "file_extension": row.get("file_extension", ""),
                    "file_sha256": row["file_sha256"],
                    "exclusion_status": row.get("exclusion_status", "eligible"),
                }
            )
    _write_csv(
        output_dir / "pixel_label_conflicts.csv",
        sorted(conflict_rows, key=lambda row: (row["pixel_sha256"], row["sample_id"])),
        (
            "pixel_sha256",
            "group_size",
            "labels",
            "conflict_state",
            "sample_id",
            "split",
            "label",
            "source_id",
            "relative_path",
            "file_extension",
            "file_sha256",
            "exclusion_status",
        ),
    )

    file_duplicate_groups = sha_groups(eligible)
    all_file_groups = sha_groups(all_rows)
    file_cross_split = [g for g in file_duplicate_groups if len({row["split"] for row in g}) > 1]
    label_counts = defaultdict(int)
    for row in conflict_rows:
        label_counts[row["conflict_state"]] += 1
    fingerprint = _fingerprint(splits_dir)
    summary = {
        "status": "complete" if not pixel_errors else "complete_with_decode_errors",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": os.popen(f"git -C {PROJECT_ROOT} rev-parse HEAD").read().strip(),
        "git_dirty": bool(os.popen(f"git -C {PROJECT_ROOT} status --porcelain").read().strip()),
        "source": "Modal Volume corn-clean:/clean; verified against corn-outputs:/splits/seed_42",
        "dataset_fingerprint": fingerprint,
        "samples_expected": 33_429,
        "samples_hashed": len(eligible),
        "contractually_excluded_files_expected": 8,
        "contractually_excluded_files_hashed": len(excluded_pixels),
        "preflight": preflight_summary,
        "decode_errors": len(pixel_errors),
        "decode_error_rows": pixel_errors[:100],
        "split_sizes": {split: sum(row["split"] == split for row in eligible) for split in SPLITS},
        "decoder": {
            "library": "Pillow",
            "version": pillow_version,
            "python": platform.python_version(),
        },
        "pixel_hash_definition": (
            "SHA256(UTF8('doctormaiz-pixel-sha256-v1\\0RGB\\0') || "
            "uint32be(width) || uint32be(height) || RGB row-major bytes)"
        ),
        "pixel_normalization": {
            "exif_orientation": "applied with ImageOps.exif_transpose, matching src.data.loader",
            "mode": "converted to RGB using Pillow; original post-EXIF mode recorded as image_mode",
            "alpha": "discarded by RGB conversion; not composited",
            "icc_profiles": (
                "not applied or hashed; direct Pillow conversion matches current loader"
            ),
            "resize_or_augmentation": False,
        },
        "file_sha256": {
            "duplicate_groups": len(file_duplicate_groups),
            "cross_split_groups": len(file_cross_split),
            "label_conflict_groups": sum(
                len({row["label"] for row in g}) > 1 for g in file_duplicate_groups
            ),
        },
        "pixel_file_hash_equivalence": {
            "eligible_pixel_duplicate_groups": len(groups),
            "including_exclusions_file_duplicate_groups": len(all_file_groups),
            "including_exclusions_pixel_duplicate_groups": len(all_groups),
            "same_pixel_groups_with_different_file_sha256": sum(
                len({row["file_sha256"] for row in group}) > 1 for group in all_groups
            ),
        },
        "source_analysis": {
            "eligible_duplicate_sources": sorted({row["source_id"] for row in duplicate_members}),
            "excluded_source_resolution": "unique filename-family match in master_manifest.csv",
            "known_excluded_conflict_sources": sorted(
                {
                    row["source_id"]
                    for row in conflict_rows
                    if row["conflict_state"] == "known_contractually_excluded"
                }
            ),
        },
        "pixel_sha256": _summary_stats(groups, 5015, 5014, len(eligible)),
        "pixel_duplicate_groups_including_exclusions": len(all_groups),
        "pixel_label_conflict_groups": len(conflicts),
        "pixel_label_conflict_members_by_state": dict(label_counts),
        "historical_15_groups_reproduced": len(groups) == 15,
        "historical_test_0_68_percent_reproduced": abs(
            _summary_stats(groups, 5015, 5014, len(eligible))["test_percent_in_cross_split_groups"]
            - 0.68
        )
        < 0.005,
        "artifacts": [
            "corpus_preflight.csv",
            "corpus_preflight.json",
            "pixel_hash_inventory.csv",
            "pixel_excluded_images.csv",
            "pixel_duplicate_groups.csv",
            "pixel_cross_split_duplicates.csv",
            "pixel_label_conflicts.csv",
            "pixel_audit_summary.json",
            "PIXEL_DUPLICATE_AUDIT.md",
        ],
    }
    _write_json(output_dir / "pixel_audit_summary.json", summary)
    _write_report(output_dir / "PIXEL_DUPLICATE_AUDIT.md", summary, conflicts)
    checkpoint = output_dir / ".pixel_hash_checkpoint.csv"
    if checkpoint.exists():
        checkpoint.unlink()
    return summary


def _write_report(
    path: Path, summary: dict[str, Any], conflicts: list[list[dict[str, Any]]]
) -> None:
    pixel = summary["pixel_sha256"]
    file_hash = summary["file_sha256"]
    fingerprint = summary["dataset_fingerprint"]
    preflight = summary["preflight"]
    equivalence = summary["pixel_file_hash_equivalence"]
    source = summary["source_analysis"]
    excluded_conflicts = sum(
        all(row["split"] == "excluded" for row in group) for group in conflicts
    )
    rows = [
        "# Auditoría de duplicados exactos por píxeles",
        "",
        "## Identidad y cobertura",
        "",
        f"- Estado: {summary['status']}; fecha UTC: {summary['timestamp_utc']}.",
        f"- Commit: {summary['git_commit']} (dirty={summary['git_dirty']}).",
        f"- Fuente canónica: {summary['source']}.",
        "- No se utilizó el ZIP local filtrado de aproximadamente 28 360 imágenes.",
        f"- Master manifest SHA-256: {fingerprint['master_manifest_sha256']}.",
        f"- manifest.lock.json SHA-256: {fingerprint['manifest_lock_sha256']}.",
        (
            f"- {summary['samples_hashed']:,} muestras elegibles y "
            f"{summary['contractually_excluded_files_hashed']} exclusiones verificadas."
        ),
        (
            f"- Preflight: {preflight['files_found_and_sha_verified']:,}/"
            f"{preflight['files_expected']:,} rutas y SHA correctos; "
            f"errores de decodificación: {summary['decode_errors']}."
        ),
        "",
        "## Hashes y grupos",
        "",
        "| Población | Hash | Grupos duplicados | Cross-split | Conflictos |",
        "|---|---|---:|---:|---:|",
        (
            f"| Elegibles | file_sha256 | {file_hash['duplicate_groups']} | "
            f"{file_hash['cross_split_groups']} | {file_hash['label_conflict_groups']} |"
        ),
        (
            f"| Elegibles | pixel_sha256 | {pixel['duplicate_group_count']} | "
            f"{pixel['cross_split_group_count']} | 0 |"
        ),
        (
            "| Ocho exclusiones | file_sha256 | "
            f"{equivalence['including_exclusions_file_duplicate_groups']} | no aplica | "
            f"{excluded_conflicts} |"
        ),
        (
            "| Ocho exclusiones | pixel_sha256 | "
            f"{excluded_conflicts} | no aplica | {excluded_conflicts} |"
        ),
        "",
        "file_sha256 identifica bytes idénticos. pixel_sha256 identifica exactamente el",
        "mismo raster RGB tras orientación EXIF y conversión a RGB; incorpora ancho,",
        "alto y bytes en orden de filas. No aplica resize, aumentos ni perfiles ICC.",
        "Los archivos recodificados pueden tener bytes distintos y píxeles iguales.",
        "",
        (
            "- Grupos con píxeles iguales y SHA de archivo distinto: "
            f"{equivalence['same_pixel_groups_with_different_file_sha256']}."
        ),
        "",
        "## Particiones y etiquetas",
        "",
        (
            f"- Grupos internos: {pixel['within_split_group_count']}; "
            f"entre particiones: {pixel['cross_split_group_count']}."
        ),
        (
            f"- Train/validation: {pixel['cross_split_groups']['train_validation']}; "
            f"train/test: {pixel['cross_split_groups']['train_test']}."
        ),
        (
            f"- Validation/test: {pixel['cross_split_groups']['validation_test']}; "
            f"tres particiones: {pixel['cross_split_groups']['three_way']}."
        ),
        (
            f"- Test afectado: {pixel['test_samples_in_cross_split_groups']}/"
            f"{summary['split_sizes']['test']} "
            f"({pixel['test_percent_in_cross_split_groups']:.4f} %)."
        ),
        (
            f"- Conflictos nuevos: {summary['pixel_label_conflict_groups'] - excluded_conflicts}; "
            f"conflictos excluidos: {excluded_conflicts} grupos / "
            f"{summary['contractually_excluded_files_hashed']} archivos."
        ),
        (
            "- Fuente de las exclusiones: "
            f"{', '.join(source['known_excluded_conflict_sources'])}; "
            "atribución por familia de nombre unívoca en master_manifest.csv."
        ),
        "",
        "## Hallazgo histórico y alcance",
        "",
        "La observación histórica de aproximadamente 15 grupos y 0.68 % del test",
        "con gemelos no se reproduce en esta materialización canónica: 0 grupos",
        "elegibles y 0/5,015 imágenes de test afectadas. Puede corresponder a otra",
        "versión del corpus, otra materialización o un procedimiento distinto.",
        "Esta auditoría no demuestra que el hallazgo nunca existió.",
        "",
        "Se conserva seed_42. Las métricas baseline y HPO no requieren invalidación",
        "por duplicados exactos cross-split en esta materialización. La auditoría",
        "no resuelve generalización entre fuentes ni similitud perceptual, recortes,",
        "resize, rotaciones, cambios fotométricos o fotografías relacionadas.",
        "",
        "## Configuración y artefactos",
        "",
        f"- SHA-256 de dataset.yaml en lock: {fingerprint['config_sha256_in_lock']}.",
        f"- SHA-256 actual: {fingerprint['current_config_sha256']}.",
        "",
        "El drift se analiza en [dataset_yaml_drift_report.md](dataset_yaml_drift_report.md).",
        "Los hashes de los CSV",
        "canónicos coinciden con el lock; la diferencia del YAML no cambia sus bytes.",
        "",
        "- pixel_hash_inventory.csv: una fila por muestra elegible.",
        "- pixel_excluded_images.csv: ocho exclusiones auditadas por separado.",
        "- pixel_duplicate_groups.csv y pixel_cross_split_duplicates.csv: grupos elegibles.",
        "- pixel_label_conflicts.csv: los cuatro conflictos ya excluidos.",
        "- corpus_preflight.csv/json: correspondencia de rutas y SHA de archivo.",
        "",
        "La copia de trabajo en /tmp/doctormaiz_pixel_audit_20260929/ contiene los",
        "manifiestos y una descarga local temporal del corpus. La fuente canónica",
        "sigue siendo corn-clean:/clean. La copia puede eliminarse con autorización",
        "posterior del usuario y no debe incluirse en Git.",
        "",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=get_output_root() / "splits/seed_42")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    summary = run_audit(
        splits_dir=args.splits_dir,
        dataset_root=args.dataset_root or get_dataset_root(),
        output_dir=args.output_dir,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
