"""Optuna persistente: CLI genérica y protocolo formal Lite0 seed_42 (25 intentos)."""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

import optuna
import yaml

from src.config import PROJECT_ROOT, get_output_root
from src.data.preparation import atomic_write_json, sha256_file
from src.models.registry import MODEL_REGISTRY
from src.training.common import resolve_model_names, select_device
from src.training.tuning import (
    HyperparameterSpace,
    TuningObjective,
    _environment_versions,
    assert_split_snapshot_unchanged,
    export_tuning_artifacts,
    split_hash_snapshot,
    terminal_trial_count,
    write_selection_lock,
)
from src.training.tuning_study import (
    BASELINE_F1,
    EXPECTED_SPLIT_HASHES,
    FORMAL_STUDY,
    PRUNER_CONFIG,
    evaluate_locked_winner,
    exclusive_study,
    open_study,
    persist_sampler,
    run_to_budget,
    source_snapshot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["efficientnet_lite0"])
    parser.add_argument("--n-trials", type=int, default=25, help="Total, no trials adicionales.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=None, help="Límite ENTRE trials, segundos.")
    parser.add_argument("--max-new-trials", type=int, default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--study-name-prefix", default="tune")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--pruner", choices=["median", "hyperband", "none"], default="median")
    parser.add_argument("--baseline-f1", type=float, default=BASELINE_F1)
    parser.add_argument("--search-clahe", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/dataset.yaml"))
    parser.add_argument("--formal-hpo", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Study aislado, 2 trials de 1 época.")
    args = parser.parse_args(argv)
    if args.n_trials < 1 or args.epochs < 1 or args.patience < 1 or args.num_workers < 0:
        parser.error("Presupuesto/épocas/paciencia inválidos.")
    if args.max_new_trials is not None and args.max_new_trials < 1:
        parser.error("--max-new-trials debe ser positivo.")
    if args.timeout is not None and args.timeout < 1:
        parser.error("--timeout debe ser positivo.")
    if args.formal_hpo and (
        args.models != ["efficientnet_lite0"]
        or args.n_trials != 25
        or args.epochs != 60
        or args.patience != 8
        or args.pruner != "median"
        or args.search_clahe
        or args.no_pretrained
        or args.baseline_f1 != BASELINE_F1
        or args.study_name not in (None, FORMAL_STUDY)
    ):
        parser.error(
            "El protocolo formal requiere Lite0, 25 trials, 60 épocas, patience=8 y defaults fijos."
        )
    if args.smoke and not args.formal_hpo:
        parser.error("--smoke requiere --formal-hpo para validar el protocolo de destino.")
    return args


def _build_pruner(kind):
    if kind == "median":
        return optuna.pruners.MedianPruner(**PRUNER_CONFIG)
    if kind == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=3, max_resource=60)
    return optuna.pruners.NopPruner()


def main(argv=None):
    args = _parse_args(argv)
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    splits = Path(args.splits_dir) if args.splits_dir else get_output_root() / "splits/seed_42"
    snapshot = split_hash_snapshot(splits)
    if args.formal_hpo and (
        snapshot["sha256"] != EXPECTED_SPLIT_HASHES
        or snapshot["counts"] != {"train": 23400, "val": 5014, "test": 5015}
        or config["dataset"]["seed"] != 42
        or snapshot["seed"] != 42
        or snapshot["split_identifier"] != "seed_42"
    ):
        raise RuntimeError("Los datos/configuración no coinciden con seed_42 congelado.")
    source = source_snapshot(PROJECT_ROOT)
    environment = _environment_versions()
    baseline_path = get_output_root() / "main/efficientnet_lite0/20260921_204608/best.pth"
    baseline_hash = sha256_file(baseline_path) if args.formal_hpo else None
    space = HyperparameterSpace(allow_clahe=args.search_clahe)
    space.validate()
    root = Path(args.output_dir) if args.output_dir else get_output_root() / "hpo"
    target, epochs = (2, 1) if args.smoke else (args.n_trials, args.epochs)
    device = select_device()
    for model in resolve_model_names(args.models, MODEL_REGISTRY):
        name = args.study_name or (
            FORMAL_STUDY if args.formal_hpo else f"{args.study_name_prefix}_{model}"
        )
        if args.smoke:
            name += "_smoke_25"
        directory = root / model / name
        protocol = {
            "schema_version": 1,
            "study_name": name,
            "model": model,
            "training_seed": config["dataset"]["seed"],
            "sampler_seed": 42,
            "objective": "best_validation_macro_f1",
            "direction": "maximize",
            "n_trials": target,
            "max_epochs": epochs,
            "patience": args.patience,
            "sampler": "TPESampler",
            "multivariate": True,
            "pruner": args.pruner,
            "pruner_config": PRUNER_CONFIG if args.pruner == "median" else {},
            "search_space": space.to_dict(),
            "test_enabled": False,
            "split_snapshot": snapshot,
            "source_sha256": source["sha256"],
            "config_sha256": sha256_file(config_path),
            "parallel_workers": 1,
            "num_workers": args.num_workers,
            "pretrained": not args.no_pretrained,
            "baseline_checkpoint_sha256": baseline_hash,
            "versions": {
                k: environment[k]
                for k in ("python", "pytorch", "torchvision", "timm", "optuna", "cuda_runtime")
            },
            "smoke": args.smoke,
        }
        with exclusive_study(directory):
            preflight_path = directory / "preflight.json"
            if (
                preflight_path.exists()
                and json.loads(preflight_path.read_text())["protocol"] != protocol
            ):
                if (directory / "study.db").exists():
                    raise RuntimeError(
                        "Preflight ya fijado con otro protocolo/código; requiere auditoría."
                    )
                # Antes del primer trial se permite corregir un preflight técnico.
                # Se conserva la evidencia de la revisión fallida, sin abrir otro study.
                previous = json.loads(preflight_path.read_text())
                revisions = directory / "preflight_revisions"
                revisions.mkdir(exist_ok=True)
                revision = previous["protocol"]["source_sha256"]
                preflight_path.replace(revisions / f"{revision}.json")
                if (directory / "source_code.zip").exists():
                    (directory / "source_code.zip").replace(revisions / f"{revision}.zip")
            preflight = {
                "protocol": protocol,
                "environment": environment,
                "source": source,
                "storage": str(directory / "study.db"),
                "resume_supported": True,
            }
            atomic_write_json(preflight_path, preflight)
            archive = directory / "source_code.zip"
            if not archive.exists():
                temporary_archive = archive.with_suffix(".zip.tmp")
                with zipfile.ZipFile(temporary_archive, "w") as bundle:
                    for relative in source["files"]:
                        bundle.writestr(
                            zipfile.ZipInfo(relative),
                            (PROJECT_ROOT / relative).read_bytes(),
                            compress_type=zipfile.ZIP_DEFLATED,
                        )
                    bundle.writestr(
                        zipfile.ZipInfo("config/dataset.yaml"),
                        config_path.read_bytes(),
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                temporary_archive.replace(archive)
            atomic_write_json(
                directory / "source_archive.json",
                {
                    "source_sha256": source["sha256"],
                    "archive_sha256": sha256_file(archive),
                },
            )
            print(json.dumps(preflight, indent=2), flush=True)
            if args.preflight_only:
                continue
            if args.formal_hpo and not args.smoke:
                smoke_report = root / model / f"{FORMAL_STUDY}_smoke_25" / "SMOKE_COMPLETE.json"
                if not smoke_report.exists():
                    raise RuntimeError(
                        "Falta smoke real con reapertura; ejecutar --formal-hpo --smoke."
                    )
                smoke = json.loads(smoke_report.read_text())
                if (
                    smoke["source_sha256"] != source["sha256"]
                    or smoke["split_hashes"] != snapshot["sha256"]
                ):
                    raise RuntimeError("El smoke pertenece a otro código/split.")
            study = open_study(directory, name, protocol, _build_pruner(args.pruner))
            objective = TuningObjective(
                model_name=model,
                splits_dir=splits,
                config_path=config_path,
                epochs=epochs,
                patience=args.patience,
                device=device,
                num_workers=args.num_workers,
                no_pretrained=args.no_pretrained,
                space=space,
                study_dir=directory,
                expected_split_hashes=snapshot["sha256"],
                study_name=name,
                on_parameters_sampled=lambda trial: persist_sampler(study, trial),
            )

            def export(current, _trial=None):
                assert_split_snapshot_unchanged(splits, snapshot)
                return export_tuning_artifacts(
                    current,
                    directory,
                    model,
                    args.baseline_f1,
                    requested_trials=target,
                    split_snapshot=snapshot,
                    search_space=space,
                    epochs=epochs,
                    patience=args.patience,
                    pruner_name=type(current.pruner).__name__,
                    pruner_config=protocol["pruner_config"],
                )

            if args.smoke and terminal_trial_count(study) == 0:
                run_to_budget(
                    study, objective, target, directory, max_new_trials=1, callback=export
                )
                study._storage.remove_session()
                study = open_study(directory, name, protocol, _build_pruner(args.pruner))
                if (
                    len(study.trials) != 1
                    or study.trials[0].state != optuna.trial.TrialState.COMPLETE
                ):
                    raise RuntimeError("El primer trial smoke no terminó correctamente.")
            run_to_budget(
                study,
                objective,
                target,
                directory,
                callback=export,
                max_new_trials=args.max_new_trials,
                timeout=args.timeout,
            )
            export(study)
            after = assert_split_snapshot_unchanged(splits, snapshot)
            atomic_write_json(directory / "split_hashes_after.json", after)
            if args.formal_hpo and sha256_file(baseline_path) != baseline_hash:
                raise RuntimeError("El checkpoint histórico cambió; abortar y auditar.")
            if args.smoke:
                reopened = open_study(directory, name, protocol, _build_pruner(args.pruner))
                if len(reopened.trials) == 2 and all(
                    t.state == optuna.trial.TrialState.COMPLETE for t in reopened.trials
                ):
                    atomic_write_json(
                        directory / "SMOKE_COMPLETE.json",
                        {
                            "trial_numbers": [t.number for t in reopened.trials],
                            "resume_verified": True,
                            "test_used": False,
                            "source_sha256": source["sha256"],
                            "split_hashes": snapshot["sha256"],
                        },
                    )
                else:
                    raise RuntimeError("Smoke incompleto o fallido.")
            elif args.formal_hpo and terminal_trial_count(study) == target:
                write_selection_lock(
                    study=study, study_dir=directory, split_snapshot=snapshot, search_space=space,
                    requested_trials=target,
                )
                evaluate_locked_winner(
                    directory, splits, config_path, snapshot, device, args.num_workers
                )


if __name__ == "__main__":
    main()
