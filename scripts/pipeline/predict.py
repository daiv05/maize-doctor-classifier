"""Inference from a verified run or an explicit, hashed ensemble manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import PROJECT_ROOT, get_output_root
from src.data.loader import load_and_normalize_image
from src.data.segmented import prepare_segmented_inference
from src.models.ensemble import SoftVotingEnsemble
from src.training.common import select_device
from src.training.runs import load_ensemble_manifest, load_run, resolve_checkpoint


def _find_model_checkpoint(model_name, output_root, explicit_checkpoint=None, run_id=None):
    return resolve_checkpoint(model_name, output_root, explicit_checkpoint, run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ensemble")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--run")
    parser.add_argument("--pipeline", choices=["main", "baselines"], default="main")
    parser.add_argument("--ensemble-manifest")
    parser.add_argument("--segmenter-checkpoint")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/dataset.yaml"))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    device = select_device()
    if args.model == "ensemble":
        if not args.ensemble_manifest or args.checkpoint or args.run:
            parser.error(
                "Ensemble requires --ensemble-manifest; --run/--checkpoint apply to single models"
            )
        model, runs = load_ensemble_manifest(
            args.ensemble_manifest, device, config_path=args.config
        )
        run = runs[0]
    else:
        if args.ensemble_manifest:
            parser.error("--ensemble-manifest requires --model ensemble")
        checkpoint = resolve_checkpoint(
            args.model, get_output_root(), args.checkpoint, args.run, args.pipeline
        )
        run = load_run(checkpoint, args.model, device, config_path=args.config)
        model = run.model
    if args.image_size and [args.image_size] * 2 != run.summary["image_size"]:
        parser.error("--image-size conflicts with the training contract")
    source = Path(args.image)
    if not source.exists():
        parser.error(f"Image path does not exist: {source}")
    images = (
        sorted(
            p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if source.is_dir()
        else [source]
    )
    if not images:
        parser.error("No supported images found")
    labels = {i: name for name, i in run.class_to_idx.items()}
    for image_path in images:
        original = load_and_normalize_image(image_path)

        def member_input(member):
            prepared, audit = prepare_segmented_inference(
                original,
                member.summary["preprocessing"].get("segmentation"),
                args.segmenter_checkpoint,
                device,
            )
            if audit["segmentation_status"] != "not_applicable":
                print(f"{image_path} | {audit}")
            return member.factory.get_pipeline("inference")(prepared).unsqueeze(0).to(device)

        with torch.no_grad():
            if isinstance(model, SoftVotingEnsemble):
                member_probs = [member.model(member_input(member)).softmax(1) for member in runs]
                probs = sum(weight * value for weight, value in zip(model.weights, member_probs))
            else:
                probs = model(member_input(run)).softmax(dim=1)
        values, indices = probs[0].topk(min(args.top_k, len(labels)))
        print(f"{image_path} | checkpoint={run.checkpoint}")
        for score, index in zip(values.tolist(), indices.tolist(), strict=True):
            print(f"  {labels[index]}: {score:.6f}")


if __name__ == "__main__":
    main()
