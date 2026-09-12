"""Strict, shared run/checkpoint/ensemble contracts for scientific evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.transforms import CornTransformFactory
from src.models import build_model
from src.provenance import sha256_file


@dataclass
class LoadedRun:
    model: torch.nn.Module
    checkpoint: Path
    summary: dict
    factory: CornTransformFactory

    @property
    def class_to_idx(self):
        return self.summary["class_to_idx"]

    def manifest_entry(self, weight=1.0):
        return {
            "model": self.summary["model"],
            "run_id": self.checkpoint.parent.name,
            "checkpoint": str(self.checkpoint.resolve()),
            "sha256": sha256_file(self.checkpoint),
            "summary_sha256": sha256_file(self.checkpoint.parent / "summary.json"),
            "weight": weight,
            "preprocessing": self.summary["preprocessing"],
            "class_to_idx": self.class_to_idx,
        }


def resolve_checkpoint(
    model_name, output_root, explicit_checkpoint=None, run_id=None, pipeline="main"
):
    if explicit_checkpoint:
        path = Path(explicit_checkpoint)
        if run_id and path.parent.name != run_id:
            raise ValueError("--run and --checkpoint reference different runs")
    else:
        model_dir = Path(output_root) / pipeline / model_name
        if run_id is None:
            latest = model_dir / "latest.json"
            if not latest.is_file():
                raise FileNotFoundError(
                    f"No latest.json for {model_name}; specify --run or --checkpoint"
                )
            payload = json.loads(latest.read_text())
            run_id = payload.get("run_id") or payload.get("run")
        if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            raise ValueError("Invalid run_id")
        path = model_dir / run_id / "best.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Requested checkpoint does not exist: {path}")
    return path


def load_run(checkpoint, model_name, device="cpu", *, config_path=None):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Missing training contract: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("model") != model_name:
        raise ValueError("Checkpoint architecture disagrees with the requested model")
    mapping = summary.get("class_to_idx", {})
    if (
        not mapping
        or any(type(i) is not int for i in mapping.values())
        or sorted(mapping.values()) != list(range(len(mapping)))
    ):
        raise ValueError("Invalid class_to_idx training contract")
    contract = summary.get("preprocessing")
    if not contract:
        raise ValueError(
            "Legacy run has no preprocessing contract; explicit reviewed migration is required"
        )
    if summary.get("image_size") != contract.get("target_size"):
        raise ValueError("Input size disagrees with preprocessing contract")
    expected_hash = summary.get("checkpoint_sha256")
    if not expected_hash or sha256_file(path) != expected_hash:
        raise ValueError("Checkpoint hash does not match the training contract")
    factory = CornTransformFactory.from_contract(contract, config_path=config_path)
    model = build_model(model_name, num_classes=len(mapping), pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return LoadedRun(model, path, summary, factory)


def validate_ensemble_runs(runs, shared_input=False):
    if not runs:
        raise ValueError("Empty ensemble")
    first = runs[0]
    for run in runs[1:]:
        if run.class_to_idx != first.class_to_idx:
            raise ValueError("Ensemble class order is incompatible")
        if shared_input and run.summary["preprocessing"] != first.summary["preprocessing"]:
            raise ValueError(
                "Ensemble tensor preprocessing differs; evaluate members with separate loaders"
            )


def load_ensemble_manifest(path, device="cpu", *, config_path=None):
    from src.models.ensemble import SoftVotingEnsemble

    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("members"):
        raise ValueError("Unsupported or empty ensemble manifest")
    runs, weights = [], []
    for member in manifest["members"]:
        checkpoint = Path(member["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = manifest_path.parent / checkpoint
        if sha256_file(checkpoint) != member["sha256"]:
            raise ValueError("Ensemble checkpoint hash mismatch")
        run = load_run(checkpoint, member["model"], device, config_path=config_path)
        if member.get("run_id") != checkpoint.parent.name:
            raise ValueError("Ensemble run ID mismatch")
        if member.get("summary_sha256") != sha256_file(checkpoint.parent / "summary.json"):
            raise ValueError("Ensemble training contract hash mismatch")
        if (
            member.get("preprocessing") != run.summary["preprocessing"]
            or member.get("class_to_idx") != run.class_to_idx
        ):
            raise ValueError("Ensemble manifest disagrees with member contract")
        runs.append(run)
        weights.append(member["weight"])
    validate_ensemble_runs(runs)
    ensemble = (
        SoftVotingEnsemble(
            [r.model for r in runs], weights=weights, model_names=[r.summary["model"] for r in runs]
        )
        .to(device)
        .eval()
    )
    return ensemble, runs
