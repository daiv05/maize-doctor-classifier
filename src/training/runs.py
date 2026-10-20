"""Contratos versionados y validación estricta de runs de entrenamiento."""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch

from src.data.preparation import atomic_write_json, sha256_file, sha256_json
from src.data.transforms import CornTransformFactory
from src.models import build_model
from src.training.hyperparameters import validate_effective_hyperparameters

RUN_SCHEMA_VERSION = 1
LATEST_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RunContractError(RuntimeError):
    """El contrato falta, está corrupto o usa un esquema incompatible."""


class CheckpointIntegrityError(RunContractError):
    """Los bytes del checkpoint no coinciden con el SHA-256 registrado."""


class RunCompatibilityError(RunContractError):
    """El consumidor solicitó una configuración distinta a la del run."""


class LegacyRunError(RunContractError):
    """El run no tiene contrato v1 y requiere migración explícita."""


class EnsembleCompatibilityError(RunContractError):
    """Los miembros del ensemble no comparten el contrato requerido."""


class LegacyRunWarning(UserWarning):
    """Se resolvió metadata legacy que todavía deberá migrarse antes de cargar pesos."""


@dataclass(frozen=True)
class ValidatedRunContract:
    run_dir: Path
    checkpoint: Path
    summary: dict[str, Any]

    @property
    def class_to_idx(self) -> dict[str, int]:
        return dict(self.summary["class_to_idx"])

    @property
    def input_size(self) -> tuple[int, int]:
        size = self.summary["architecture"]["input_size"]
        return (int(size[0]), int(size[1]))


@dataclass
class LoadedRun:
    model: torch.nn.Module
    checkpoint: Path
    summary: dict[str, Any]
    factory: CornTransformFactory

    @property
    def class_to_idx(self) -> dict[str, int]:
        return dict(self.summary["class_to_idx"])

    @property
    def input_size(self) -> tuple[int, int]:
        size = self.summary["architecture"]["input_size"]
        return (int(size[0]), int(size[1]))


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise RunContractError(f"run_id inválido: {run_id!r}")
    if run_id in {".", ".."}:
        raise RunContractError(f"run_id inválido: {run_id!r}")
    return run_id


def _validate_class_to_idx(mapping: Any) -> dict[str, int]:
    if not isinstance(mapping, dict) or not mapping:
        raise RunContractError("class_to_idx debe ser un objeto JSON no vacío.")
    if any(not isinstance(name, str) or type(index) is not int for name, index in mapping.items()):
        raise RunContractError("class_to_idx requiere nombres string e índices enteros.")
    normalized = dict(mapping)
    if sorted(normalized.values()) != list(range(len(normalized))):
        raise RunContractError(f"class_to_idx debe contener índices contiguos 0..N-1: {normalized}")
    return normalized


def _configuration_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Extrae únicamente configuración efectiva, sin timestamps ni rutas de máquina."""
    return {
        "architecture": summary["architecture"],
        "seed": summary["seed"],
        "hyperparameters": summary["hyperparameters"],
        "preprocessing": summary["preprocessing"],
        "class_to_idx": summary["class_to_idx"],
    }


def compute_config_sha256(
    *,
    model_name: str,
    input_size: tuple[int, int] | list[int],
    num_classes: int,
    seed: int,
    hyperparameters: dict[str, Any],
    preprocessing: dict[str, Any],
    class_to_idx: dict[str, int],
) -> str:
    """Calcula el hash canónico de los parámetros que definen el experimento."""
    payload = {
        "architecture": {
            "model_name": model_name,
            "input_size": list(input_size),
            "num_classes": num_classes,
        },
        "seed": seed,
        "hyperparameters": hyperparameters,
        "preprocessing": preprocessing,
        "class_to_idx": class_to_idx,
    }
    return sha256_json(payload)


def split_manifest_fingerprint(splits_dir: str | Path) -> tuple[str, str]:
    """Obtiene el fingerprint contractual de los splits usados por entrenamiento."""
    directory = Path(splits_dir)
    lock_path = directory / "manifest.lock.json"
    if lock_path.is_file():
        return sha256_file(lock_path), "manifest.lock.json"

    csv_paths = [directory / f"{name}.csv" for name in ("train", "val", "test")]
    missing = [str(path) for path in csv_paths if not path.is_file()]
    if missing:
        raise RunContractError(
            "No se puede identificar el split: falta manifest.lock.json y CSV contractuales: "
            f"{missing}"
        )
    # Compatibilidad para splits históricos/sintéticos: el conjunto ordenado de hashes
    # constituye un artefacto equivalente, sin depender de rutas absolutas.
    fingerprint = sha256_json({path.name: sha256_file(path) for path in csv_paths})
    return fingerprint, "derived_split_csv_sha256_v1"


def build_run_contract(
    *,
    run_dir: str | Path,
    model_name: str,
    seed: int,
    hyperparameters: dict[str, Any],
    preprocessing: dict[str, Any],
    class_to_idx: dict[str, int],
    splits_dir: str | Path,
    best_epoch: int,
    metrics: dict[str, Any],
    checkpoint_name: str = "best.pth",
    training_preprocessing: dict[str, Any] | None = None,
    historical_fields: dict[str, Any] | None = None,
    migration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye el contrato final después de escribir el checkpoint de mejor época."""
    directory = Path(run_dir)
    run_id = _validate_run_id(directory.name)
    mapping = _validate_class_to_idx(class_to_idx)
    try:
        effective_hyperparameters = validate_effective_hyperparameters(hyperparameters)
    except ValueError as error:
        raise RunContractError(
            f"Hiperparámetros inválidos para el run {run_id}: {error}"
        ) from error
    try:
        factory = CornTransformFactory.from_contract(preprocessing)
    except ValueError as error:
        raise RunContractError(f"Preprocessing inválido para el run {run_id}: {error}") from error
    input_size = list(factory.target_size)
    if type(seed) is not int:
        raise RunContractError("seed debe ser entero.")
    if type(best_epoch) is not int or best_epoch < 1:
        raise RunContractError("best_epoch debe ser un entero positivo.")
    if not isinstance(metrics, dict) or not metrics:
        raise RunContractError("metrics debe ser un objeto JSON no vacío.")

    checkpoint_rel = Path(checkpoint_name)
    if checkpoint_rel.is_absolute() or checkpoint_rel.name != checkpoint_name:
        raise RunContractError("checkpoint debe ser un nombre relativo dentro del run.")
    checkpoint_path = directory / checkpoint_rel
    if not checkpoint_path.is_file():
        raise RunContractError(f"No existe el checkpoint contractual: {checkpoint_path}")

    split_sha256, split_source = split_manifest_fingerprint(splits_dir)
    architecture = {
        "model_name": model_name,
        "input_size": input_size,
        "num_classes": len(mapping),
    }
    contract: dict[str, Any] = dict(historical_fields or {})
    contract.update(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "model": model_name,
            "architecture": architecture,
            "num_classes": len(mapping),
            "image_size": input_size,
            "seed": seed,
            "hyperparameters": effective_hyperparameters,
            "preprocessing": preprocessing,
            "training_preprocessing": training_preprocessing,
            "class_to_idx": mapping,
            "splits_dir": str(splits_dir),
            "split_identifier": Path(splits_dir).name,
            "split_manifest_sha256": split_sha256,
            "split_manifest_source": split_source,
            "best_epoch": best_epoch,
            "metrics": metrics,
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
    )
    contract["config_sha256"] = sha256_json(_configuration_payload(contract))
    if migration is not None:
        contract["migration"] = migration
    return contract


def write_run_contract(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Valida y persiste ``summary.json`` de forma atómica."""
    directory = Path(run_dir)
    _validate_summary(payload, directory)
    return atomic_write_json(directory / "summary.json", payload)


def write_latest_pointer(output_dir: str | Path, model_name: str, run_id: str) -> Path:
    """Actualiza ``latest.json`` mediante flush, fsync y ``os.replace``."""
    _validate_run_id(run_id)
    return atomic_write_json(
        Path(output_dir) / model_name / "latest.json",
        {"schema_version": LATEST_SCHEMA_VERSION, "run_id": run_id},
    )


def resolve_run_dir(output_dir: str | Path, model_name: str, run_id: str | None = None) -> Path:
    """Resuelve primero un run explícito y, en su ausencia, el puntero ``latest.json``."""
    model_dir = Path(output_dir) / model_name
    if run_id is None:
        latest_path = model_dir / "latest.json"
        if not latest_path.is_file():
            raise RunContractError(
                f"No existe {latest_path}; especifica un run_id o entrena el modelo."
            )
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RunContractError(f"latest.json inválido en {latest_path}: {error}") from error
        if latest.get("schema_version") != LATEST_SCHEMA_VERSION:
            if "schema_version" in latest:
                raise RunContractError(
                    f"Esquema de latest.json incompatible: {latest.get('schema_version')!r}"
                )
            warnings.warn(
                f"Puntero latest.json legacy detectado en {latest_path}; el run deberá "
                "migrarse antes de cargar el checkpoint.",
                LegacyRunWarning,
                stacklevel=2,
            )
        run_id = latest.get("run_id") or latest.get("run")
    resolved_id = _validate_run_id(run_id)
    run_dir = model_dir / resolved_id
    if not run_dir.is_dir():
        raise RunContractError(
            f"No se encontró el run {resolved_id!r} de {model_name!r} en {model_dir}."
        )
    return run_dir


def resolve_checkpoint(
    model_name: str,
    output_root: str | Path,
    *,
    explicit_checkpoint: str | Path | None = None,
    run_id: str | None = None,
    pipelines: Iterable[str] = ("main", "baselines"),
) -> Path:
    """Resuelve checkpoints sin selección silenciosa por mtime."""
    if explicit_checkpoint is not None:
        checkpoint = Path(explicit_checkpoint)
        if run_id is not None and checkpoint.parent.name != run_id:
            raise RunCompatibilityError(
                "El checkpoint explícito y run_id apuntan a runs diferentes."
            )
        if not checkpoint.is_file():
            raise RunContractError(f"No existe el checkpoint solicitado: {checkpoint}")
        return checkpoint

    failures: list[str] = []
    for pipeline in pipelines:
        model_output = Path(output_root) / pipeline
        try:
            run_dir = resolve_run_dir(model_output, model_name, run_id)
        except RunContractError as error:
            failures.append(str(error))
            continue
        checkpoint = run_dir / "best.pth"
        if checkpoint.is_file():
            return checkpoint
        failures.append(f"El run {run_dir} no contiene best.pth.")
        if run_id is not None:
            break
    raise RunContractError(
        f"No se pudo resolver un checkpoint contractual para {model_name!r}: "
        + " | ".join(failures)
    )


def read_run_contract(run_dir: str | Path) -> dict[str, Any]:
    """Lee un contrato v1; los resúmenes legacy se rechazan explícitamente."""
    directory = Path(run_dir)
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise LegacyRunError(f"Legacy run detected: falta {summary_path}. Ejecuta migrate_run.py.")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunContractError(f"No se pudo leer {summary_path}: {error}") from error
    if "schema_version" not in summary:
        raise LegacyRunError(
            f"Legacy run detected en {directory}: summary.json no tiene schema_version. "
            "La migración explícita es obligatoria."
        )
    _validate_summary(summary, directory)
    return summary


def _validate_summary(summary: dict[str, Any], run_dir: Path) -> None:
    required = {
        "schema_version",
        "run_id",
        "model",
        "architecture",
        "num_classes",
        "image_size",
        "seed",
        "hyperparameters",
        "preprocessing",
        "class_to_idx",
        "config_sha256",
        "split_manifest_sha256",
        "splits_dir",
        "split_identifier",
        "best_epoch",
        "metrics",
        "checkpoint",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise RunContractError(f"Contrato incompleto en {run_dir}: faltan {missing}")
    if summary["schema_version"] != RUN_SCHEMA_VERSION:
        raise RunContractError(
            f"Esquema de run incompatible en {run_dir}: {summary['schema_version']!r}; "
            f"esperado {RUN_SCHEMA_VERSION}."
        )
    run_id = _validate_run_id(summary["run_id"])
    if run_id != run_dir.name:
        raise RunCompatibilityError(
            f"run_id del contrato ({run_id}) no coincide con el directorio ({run_dir.name})."
        )
    mapping = _validate_class_to_idx(summary["class_to_idx"])
    if not isinstance(summary["model"], str) or not summary["model"]:
        raise RunContractError(f"model debe ser un nombre no vacío en {run_id}.")
    try:
        validate_effective_hyperparameters(summary["hyperparameters"])
    except ValueError as error:
        raise RunContractError(f"Hiperparámetros inválidos en {run_id}: {error}") from error
    architecture = summary["architecture"]
    if not isinstance(architecture, dict):
        raise RunContractError("architecture debe ser un objeto JSON.")
    expected_architecture = {
        "model_name": summary["model"],
        "input_size": summary.get("image_size"),
        "num_classes": len(mapping),
    }
    if architecture != expected_architecture:
        raise RunContractError(
            f"Arquitectura interna inconsistente en {run_id}: "
            f"esperado {expected_architecture}, encontrado {architecture}."
        )
    if summary["num_classes"] != len(mapping):
        raise RunContractError(
            f"num_classes inconsistente en {run_id}: "
            f"esperado {len(mapping)}, encontrado {summary['num_classes']!r}."
        )
    try:
        factory = CornTransformFactory.from_contract(summary["preprocessing"])
    except ValueError as error:
        raise RunContractError(f"Preprocessing inválido en {run_id}: {error}") from error
    if list(factory.target_size) != architecture["input_size"]:
        raise RunContractError(
            f"input_size no coincide con preprocessing en {run_id}: "
            f"{architecture['input_size']} != {list(factory.target_size)}"
        )
    if type(summary["seed"]) is not int:
        raise RunContractError("seed contractual debe ser entero.")
    if type(summary["best_epoch"]) is not int or summary["best_epoch"] < 1:
        raise RunContractError("best_epoch contractual debe ser entero positivo.")
    metrics = summary["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        raise RunContractError(f"metrics debe ser un objeto no vacío en {run_id}.")
    best_validation = metrics.get("best_validation")
    if (
        not isinstance(best_validation, dict)
        or best_validation.get("epoch") != summary["best_epoch"]
    ):
        raise RunContractError(
            f"metrics.best_validation.epoch debe coincidir con best_epoch en {run_id}."
        )
    if not isinstance(summary["splits_dir"], str) or not summary["splits_dir"]:
        raise RunContractError(f"splits_dir debe ser una ruta no vacía en {run_id}.")
    if not isinstance(summary["split_identifier"], str) or not summary["split_identifier"]:
        raise RunContractError(f"split_identifier debe ser no vacío en {run_id}.")
    for hash_field in ("config_sha256", "split_manifest_sha256", "checkpoint_sha256"):
        value = summary[hash_field]
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise RunContractError(f"{hash_field} no es un SHA-256 válido en {run_id}.")
    actual_config_hash = sha256_json(_configuration_payload(summary))
    if actual_config_hash != summary["config_sha256"]:
        raise RunContractError(
            f"config_sha256 mismatch en {run_id}: esperado {summary['config_sha256']}, "
            f"actual {actual_config_hash}."
        )


def validate_run_contract(
    run_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    expected_model: str | None = None,
    expected_input_size: tuple[int, int] | list[int] | None = None,
    expected_num_classes: int | None = None,
    expected_class_to_idx: dict[str, int] | None = None,
    expected_preprocessing: dict[str, Any] | None = None,
    expected_split_manifest_sha256: str | None = None,
    splits_dir: str | Path | None = None,
) -> ValidatedRunContract:
    """Valida metadata y SHA antes de permitir que el consumidor invoque ``torch.load``."""
    directory = Path(run_dir)
    summary = read_run_contract(directory)
    architecture = summary["architecture"]
    checks = [
        ("model", expected_model, summary["model"]),
        (
            "input_size",
            list(expected_input_size) if expected_input_size is not None else None,
            architecture["input_size"],
        ),
        ("num_classes", expected_num_classes, architecture["num_classes"]),
        ("class_to_idx", expected_class_to_idx, summary["class_to_idx"]),
        ("preprocessing", expected_preprocessing, summary["preprocessing"]),
        (
            "split_manifest_sha256",
            expected_split_manifest_sha256,
            summary["split_manifest_sha256"],
        ),
    ]
    for field, expected, actual in checks:
        if expected is not None and expected != actual:
            raise RunCompatibilityError(
                f"Run {summary['run_id']} incompatible en {field}: "
                f"esperado {expected!r}, encontrado {actual!r}."
            )

    if splits_dir is not None:
        actual_split_hash, _ = split_manifest_fingerprint(splits_dir)
        if actual_split_hash != summary["split_manifest_sha256"]:
            raise RunCompatibilityError(
                f"Run {summary['run_id']} usa otro manifiesto de splits: "
                f"esperado {summary['split_manifest_sha256']}, actual {actual_split_hash}."
            )

    checkpoint_name = summary["checkpoint"]
    checkpoint_rel = Path(checkpoint_name)
    if (
        not isinstance(checkpoint_name, str)
        or checkpoint_rel.is_absolute()
        or checkpoint_rel.name != checkpoint_name
    ):
        raise RunContractError(f"Ruta de checkpoint insegura en {summary['run_id']}.")
    expected_checkpoint = directory / checkpoint_name
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else expected_checkpoint
    if checkpoint.resolve() != expected_checkpoint.resolve():
        raise RunCompatibilityError(
            f"El checkpoint solicitado {checkpoint} no es el registrado {expected_checkpoint}."
        )
    if not checkpoint.is_file():
        raise RunContractError(f"No existe el checkpoint de {summary['run_id']}: {checkpoint}")
    actual_checkpoint_hash = sha256_file(checkpoint)
    if actual_checkpoint_hash != summary["checkpoint_sha256"]:
        raise CheckpointIntegrityError(
            f"checkpoint_sha256 mismatch en run {summary['run_id']}: "
            f"esperado {summary['checkpoint_sha256']}, actual {actual_checkpoint_hash}."
        )
    return ValidatedRunContract(directory, checkpoint, summary)


def _load_state_dict(checkpoint: Path, device: torch.device | str) -> dict[str, Any]:
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibilidad con torch antiguo
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise RunContractError(f"Checkpoint sin state_dict válido: {checkpoint}")
    return state


def load_validated_run(
    checkpoint: str | Path,
    *,
    expected_model: str | None = None,
    device: torch.device | str = "cpu",
    expected_input_size: tuple[int, int] | list[int] | None = None,
    expected_num_classes: int | None = None,
    expected_class_to_idx: dict[str, int] | None = None,
    expected_preprocessing: dict[str, Any] | None = None,
    expected_split_manifest_sha256: str | None = None,
    splits_dir: str | Path | None = None,
    config_path: str | None = None,
) -> LoadedRun:
    """Valida el run completo y solo entonces construye/carga el modelo."""
    checkpoint_path = Path(checkpoint)
    validated = validate_run_contract(
        checkpoint_path.parent,
        checkpoint_path=checkpoint_path,
        expected_model=expected_model,
        expected_input_size=expected_input_size,
        expected_num_classes=expected_num_classes,
        expected_class_to_idx=expected_class_to_idx,
        expected_preprocessing=expected_preprocessing,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        splits_dir=splits_dir,
    )
    summary = validated.summary
    factory = CornTransformFactory.from_contract(summary["preprocessing"], config_path=config_path)
    model = build_model(
        summary["model"],
        num_classes=summary["architecture"]["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(_load_state_dict(validated.checkpoint, device), strict=True)
    model.eval()
    return LoadedRun(model, validated.checkpoint, summary, factory)


def validate_ensemble_runs(
    runs: list[LoadedRun],
    *,
    policy: Literal["normal", "cross_validation", "loso"] = "normal",
    shared_input: bool = True,
) -> None:
    """Valida todos los miembros antes de construir o ejecutar el ensemble."""
    if not runs:
        raise EnsembleCompatibilityError("El ensemble no contiene miembros.")
    if policy not in {"normal", "cross_validation", "loso"}:
        raise EnsembleCompatibilityError(f"Política de ensemble desconocida: {policy!r}")
    reference = runs[0].summary
    for member in runs[1:]:
        summary = member.summary
        if summary["class_to_idx"] != reference["class_to_idx"]:
            raise EnsembleCompatibilityError(
                f"class_to_idx incompatible entre {reference['run_id']} y {summary['run_id']}."
            )
        if summary["architecture"]["num_classes"] != reference["architecture"]["num_classes"]:
            raise EnsembleCompatibilityError(
                f"num_classes incompatible entre {reference['run_id']} y {summary['run_id']}."
            )
        if shared_input and summary["preprocessing"] != reference["preprocessing"]:
            raise EnsembleCompatibilityError(
                "El ensemble actual comparte un único tensor de entrada, pero sus miembros "
                f"{reference['run_id']} y {summary['run_id']} tienen preprocessing distinto."
            )
        if (
            policy == "normal"
            and summary["split_manifest_sha256"] != reference["split_manifest_sha256"]
        ):
            raise EnsembleCompatibilityError(
                "Un ensemble normal requiere el mismo manifiesto de splits para todos los "
                "miembros; usa policy='cross_validation' o policy='loso' explícitamente "
                "cuando corresponda."
            )


def ensemble_member_entry(run: LoadedRun, weight: float) -> dict[str, Any]:
    """Genera metadata portable y verificable para un miembro de ensemble."""
    return {
        "model": run.summary["model"],
        "run_id": run.summary["run_id"],
        "checkpoint": str(run.checkpoint),
        "checkpoint_sha256": run.summary["checkpoint_sha256"],
        "summary_sha256": sha256_file(run.checkpoint.parent / "summary.json"),
        "weight": float(weight),
        "class_to_idx": run.summary["class_to_idx"],
        "preprocessing": run.summary["preprocessing"],
        "split_manifest_sha256": run.summary["split_manifest_sha256"],
    }


def load_ensemble_manifest(
    manifest_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    config_path: str | None = None,
    splits_dir: str | Path | None = None,
) -> tuple[list[LoadedRun], list[float], str]:
    """Valida manifiesto, contratos y hashes de todos los miembros antes de inferencia."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EnsembleCompatibilityError(f"Manifiesto de ensemble inválido: {error}") from error
    members = manifest.get("members")
    if manifest.get("schema_version") != 1 or not isinstance(members, list) or not members:
        raise EnsembleCompatibilityError(
            "El ensemble requiere schema_version=1 y una lista members no vacía."
        )
    policy = manifest.get("policy")
    if policy not in {"normal", "cross_validation", "loso"}:
        raise EnsembleCompatibilityError(
            "El manifiesto debe declarar policy como normal, cross_validation o loso."
        )
    runs: list[LoadedRun] = []
    weights: list[float] = []
    for member in members:
        if not isinstance(member, dict):
            raise EnsembleCompatibilityError("Cada miembro del ensemble debe ser un objeto JSON.")
        checkpoint = Path(member.get("checkpoint", ""))
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        run = load_validated_run(
            checkpoint,
            expected_model=member.get("model"),
            expected_class_to_idx=member.get("class_to_idx"),
            expected_preprocessing=member.get("preprocessing"),
            expected_split_manifest_sha256=member.get("split_manifest_sha256"),
            splits_dir=splits_dir if policy == "normal" else None,
            device=device,
            config_path=config_path,
        )
        if member.get("run_id") != run.summary["run_id"]:
            raise EnsembleCompatibilityError(
                f"run_id del manifiesto no coincide para {checkpoint}."
            )
        if member.get("checkpoint_sha256") != run.summary["checkpoint_sha256"]:
            raise EnsembleCompatibilityError(
                f"SHA del checkpoint en el manifiesto no coincide para {checkpoint}."
            )
        actual_summary_hash = sha256_file(checkpoint.parent / "summary.json")
        if member.get("summary_sha256") != actual_summary_hash:
            raise EnsembleCompatibilityError(
                f"summary.json fue modificado para el miembro {run.summary['run_id']}."
            )
        weight = member.get("weight")
        if type(weight) not in (int, float) or weight <= 0:
            raise EnsembleCompatibilityError(f"Peso inválido para {checkpoint}: {weight!r}")
        runs.append(run)
        weights.append(float(weight))
    validate_ensemble_runs(runs, policy=policy, shared_input=True)
    return runs, weights, policy


def migrate_legacy_run(
    run_dir: str | Path,
    *,
    preprocessing: dict[str, Any] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    seed: int | None = None,
    splits_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Migra un resumen legacy solo cuando toda la metadata crítica es verificable."""
    directory = Path(run_dir)
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise LegacyRunError(f"No existe summary.json legacy en {directory}.")
    legacy = json.loads(summary_path.read_text(encoding="utf-8"))
    if "schema_version" in legacy:
        raise RunContractError(f"El run {directory} ya declara schema_version.")

    resolved_preprocessing = preprocessing or legacy.get("preprocessing")
    resolved_hyperparameters = hyperparameters or legacy.get("hyperparameters")
    resolved_seed = seed if seed is not None else legacy.get("seed")
    resolved_splits = splits_dir or legacy.get("splits_dir")
    missing = [
        name
        for name, value in {
            "model": legacy.get("model"),
            "class_to_idx": legacy.get("class_to_idx"),
            "preprocessing": resolved_preprocessing,
            "hyperparameters": resolved_hyperparameters,
            "seed": resolved_seed,
            "splits_dir": resolved_splits,
            "best_epoch": legacy.get("best_epoch"),
        }.items()
        if value is None
    ]
    if missing:
        raise LegacyRunError(
            f"Migración rechazada para {directory}: no se puede reconstruir con certeza {missing}."
        )

    test_metrics = legacy.get("test")
    metrics = legacy.get("metrics") or {
        "best_validation": {
            "epoch": legacy["best_epoch"],
            "macro_f1": legacy.get("best_val_macro_f1"),
        },
        "test": test_metrics,
    }
    contract = build_run_contract(
        run_dir=directory,
        model_name=legacy["model"],
        seed=resolved_seed,
        hyperparameters=resolved_hyperparameters,
        preprocessing=resolved_preprocessing,
        class_to_idx=legacy["class_to_idx"],
        splits_dir=resolved_splits,
        best_epoch=legacy["best_epoch"],
        metrics=metrics,
        training_preprocessing=legacy.get("training_preprocessing"),
        historical_fields=legacy,
        migration={
            "from_schema": "legacy",
            "method": "explicit_reviewed_migration_v1",
            "verified": True,
        },
    )
    write_run_contract(directory, contract)
    return contract
