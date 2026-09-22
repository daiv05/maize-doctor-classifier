"""Diagnóstico manual/puntual de estabilidad de LIME sobre una única imagen.

Corre `render_visual_explanation` N veces con seeds distintas sobre la misma imagen y
reporta, entre corridas consecutivas:
  - IoU de las máscaras de superpíxeles positivos (reconstruidas desde el .json/.npy
    que ya persiste render_visual_explanation).
  - Correlación de Pearson entre los weight_map per-píxel completos.

Uso: python scripts/checks/lime_stability.py --model efficientnet_b0 --image <ruta> --runs 5
     [--run <run_id>] [--output-dir <dir>]
"""

import argparse
from pathlib import Path

import torch
import yaml

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.fastvit  # noqa: F401 - registra modelos
import src.models.baselines.ghostnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
import src.models.baselines.shufflenet  # noqa: F401 - registra modelos
from src.config import PROJECT_ROOT, get_output_root
from src.data.loader import load_and_normalize_image
from src.explainability.stability import pairwise_stability, reconstruct_mask_and_weight_map
from src.explainability.visual_report import render_visual_explanation
from src.models.registry import MODEL_REGISTRY
from src.training.common import resolve_run_dir
from src.training.runs import load_validated_run

_OUTPUT_DIR = get_output_root() / "baselines"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help=f"Disponibles: {MODEL_REGISTRY.list_names()}"
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen a auditar.")
    parser.add_argument(
        "--runs", type=int, default=5, help="Número de repeticiones con seeds distintas."
    )
    parser.add_argument("--run", default=None, help="run_id del checkpoint (default: latest.json).")
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=None,
        help="Fuerza splits/seed_42_baseline en vez de leer lime.baseline del YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio para los PNG/json/npy de cada corrida (default: "
        "<run_dir>/lime_stability/).",
    )
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    lime_cfg = cfg["lime"]

    use_baseline = args.baseline if args.baseline is not None else lime_cfg["baseline"]
    splits_dir = get_output_root() / "splits" / ("seed_42_baseline" if use_baseline else "seed_42")
    if not splits_dir.exists():
        raise SystemExit(
            f"El directorio de splits no existe: {splits_dir}\n"
            "Genera los splits primero con: make splits  (o make splits-baseline)"
        )

    run_dir = resolve_run_dir(_OUTPUT_DIR, args.model, args.run)
    checkpoint_path = run_dir / "best.pth"
    if not checkpoint_path.exists():
        raise SystemExit(f"Sin checkpoint completo en {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_validated_run(
        checkpoint_path,
        expected_model=args.model,
        splits_dir=splits_dir,
        device=device,
        config_path=str(PROJECT_ROOT / "config" / "dataset.yaml"),
    )
    model = loaded.model
    idx_to_class = {index: name for name, index in loaded.class_to_idx.items()}
    target_size = loaded.input_size

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "lime_stability"
    image = load_and_normalize_image(Path(args.image))
    image_stem = Path(args.image).stem

    masks, weight_maps = [], []
    for seed in range(args.runs):
        output_path = output_dir / f"{image_stem}__seed-{seed}.png"
        result = render_visual_explanation(
            image=image,
            model=model,
            idx_to_class=idx_to_class,
            target_size=target_size,
            output_path=output_path,
            num_samples=lime_cfg["num_samples"],
            num_features=lime_cfg["num_features"],
            seed=seed,
            device=device,
            validation_transform=loaded.factory.get_pipeline("inference"),
        )
        mask, weight_map = reconstruct_mask_and_weight_map(
            output_path.with_suffix(".json"), output_path.with_suffix(".npy")
        )
        masks.append(mask)
        weight_maps.append(weight_map)
        print(
            f"seed={seed}: predicho={result['predicted_label']} "
            f"({result['predicted_prob'] * 100:.1f}%)"
        )

    print(f"\nEstabilidad entre corridas consecutivas ({args.runs} seeds, imagen: {args.image}):")
    print(f"{'seeds':<12}{'IoU':>10}{'correlación':>14}")
    for pair in pairwise_stability(masks, weight_maps):
        label = f"{pair['seed_a']} vs {pair['seed_b']}"
        print(f"{label:<12}{pair['iou']:>10.3f}{pair['correlation']:>14.3f}")


if __name__ == "__main__":
    main()
