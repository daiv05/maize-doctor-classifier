"""Reporte completo de inferencia e interpretabilidad para una única imagen.

Combina, en un solo directorio versionado por timestamp:
  - Inferencia: top-k y distribución completa de probabilidades.
  - Explicabilidad: panel LIME + Grad-CAM con sus sidecars .json/.npy.
  - Estabilidad: auditoría multi-seed de LIME (IoU / correlación de Pearson).

Uso: python scripts/pipeline/inference_report.py --model efficientnet_b0 --image <ruta>
     [--checkpoint <ruta.pth>] [--run <run_id>] [--stability-runs N] [--top-k K]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.fastvit  # noqa: F401 - registra modelos
import src.models.baselines.ghostnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
import src.models.baselines.shufflenet  # noqa: F401 - registra modelos
from src.config import PROJECT_ROOT, get_output_root, set_global_seed
from src.data.loader import load_and_normalize_image
from src.models.registry import MODEL_REGISTRY
from src.training.common import load_run_metadata, resolve_run_dir, select_device
from src.training.runs import load_validated_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_stability_functions():
    """Importa las utilidades de estabilidad, que arrastran la cadena opcional de xai."""
    try:
        from src.explainability.stability import (
            pairwise_stability,
            reconstruct_mask_and_weight_map,
        )
        from src.explainability.visual_report import render_visual_explanation
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Falta la dependencia opcional '{e.name}' para generar el reporte. "
            "Instala el extra xai con: pip install -e .[xai]"
        ) from e
    return render_visual_explanation, reconstruct_mask_and_weight_map, pairwise_stability


def _resolve_checkpoint(
    model_name: str, checkpoint: str | None, run_id: str | None
) -> tuple[Path, Path]:
    """Resuelve el checkpoint y su run_dir.

    `--checkpoint` tiene prioridad como override directo; sin él se cae al flujo
    MODEL/RUN sobre `get_output_root()/baselines`, leyendo latest.json si no hay run_id.

    @param {str} model_name Nombre del modelo registrado.
    @param {str|None} checkpoint Ruta explícita al .pth, o None.
    @param {str|None} run_id run_id específico, o None para el último registrado.
    @returns {tuple[Path, Path]} Ruta del checkpoint y directorio del run que lo contiene.
    """
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise SystemExit(f"No existe el checkpoint: {checkpoint_path}")
        return checkpoint_path, checkpoint_path.parent

    run_dir = resolve_run_dir(get_output_root() / "baselines", model_name, run_id)
    checkpoint_path = run_dir / "best.pth"
    if not checkpoint_path.exists():
        raise SystemExit(f"El run {run_dir.name} no tiene best.pth: {checkpoint_path}")
    return checkpoint_path, run_dir


def _run_inference(
    model: torch.nn.Module,
    image,
    inference_transform,
    idx_to_class: dict[int, str],
    device: torch.device,
    top_k: int,
) -> dict:
    """Ejecuta el forward pass y arma el detalle de probabilidades.

    @param {torch.nn.Module} model Modelo en modo eval.
    @param {Image.Image} image Imagen ya normalizada por load_and_normalize_image.
    @param {Path} config_path Ruta a config/dataset.yaml para la pipeline de inferencia.
    @param {tuple[int, int]} target_size Tamaño de entrada del checkpoint.
    @param {dict[int, str]} idx_to_class Mapeo índice -> clase del head entrenado.
    @param {torch.device} device Dispositivo de cómputo.
    @param {int} top_k Cantidad de clases a reportar en el ranking.
    @returns {dict} Predicción, confianza, top-k y distribución completa de clases.
    """
    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    effective_k = min(top_k, len(idx_to_class))
    values, indices = torch.topk(probabilities, k=effective_k)

    return {
        "predicted_label": idx_to_class[int(indices[0])],
        "predicted_prob": float(values[0]),
        "top_k": [
            {"label": idx_to_class[int(idx)], "prob": float(prob)}
            for prob, idx in zip(values.tolist(), indices.tolist(), strict=True)
        ],
        "all_probabilities": {
            idx_to_class[idx]: float(probabilities[idx]) for idx in sorted(idx_to_class)
        },
    }


def _run_stability(
    render_visual_explanation,
    reconstruct_mask_and_weight_map,
    pairwise_stability,
    model: torch.nn.Module,
    model_name: str | None,
    image,
    idx_to_class: dict[int, str],
    target_size: tuple[int, int],
    output_dir: Path,
    image_stem: str,
    lime_cfg: dict,
    device: torch.device,
    runs: int,
    validation_transform,
) -> dict:
    """Repite la explicación LIME con seeds distintas y mide su consistencia.

    @param {int} runs Número de seeds a evaluar; con menos de 2 no hay pares que comparar.
    @returns {dict} Predicción por seed y métricas IoU/correlación entre pares consecutivos.
    """
    masks, weight_maps, per_seed = [], [], []
    for seed in range(runs):
        seed_path = output_dir / f"{image_stem}__seed-{seed}.png"
        result = render_visual_explanation(
            image=image,
            model=model,
            idx_to_class=idx_to_class,
            target_size=target_size,
            output_path=seed_path,
            num_samples=lime_cfg["num_samples"],
            num_features=lime_cfg["num_features"],
            seed=seed,
            device=device,
            model_name=model_name,
            validation_transform=validation_transform,
        )
        mask, weight_map = reconstruct_mask_and_weight_map(
            seed_path.with_suffix(".json"), seed_path.with_suffix(".npy")
        )
        masks.append(mask)
        weight_maps.append(weight_map)
        per_seed.append(
            {
                "seed": seed,
                "predicted_label": result["predicted_label"],
                "predicted_prob": result["predicted_prob"],
            }
        )
        logger.info(
            f"  seed={seed}: {result['predicted_label']} ({result['predicted_prob'] * 100:.1f}%)"
        )

    return {"runs": runs, "per_seed": per_seed, "pairs": pairwise_stability(masks, weight_maps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help=f"Nombre del modelo registrado. Disponibles: {MODEL_REGISTRY.list_names()}",
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen a analizar.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta directa al .pth. Sin esto, se resuelve vía --run/latest.json bajo "
        "<output_root>/baselines/<model>.",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="run_id específico (p.ej. 20260709_040040). Ignorado si se pasa --checkpoint.",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=5,
        dest="stability_runs",
        help="Seeds para la auditoría de estabilidad. 0 la omite (es la parte lenta).",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, dest="top_k", help="Clases a listar en el ranking."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio destino. Default: <output_root>/inference/<stem>/<timestamp>/",
    )
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Modelo desconocido: '{args.model}'. Disponibles: {MODEL_REGISTRY.list_names()}"
        )
    if args.stability_runs < 0:
        raise SystemExit("--stability-runs no puede ser negativo.")

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"No existe la imagen: {image_path}")

    config_path = PROJECT_ROOT / "config" / "dataset.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    lime_cfg = cfg["lime"]
    gradcam_enabled = cfg.get("gradcam", {}).get("enabled", False)
    set_global_seed(lime_cfg["seed"])

    render_visual_explanation, reconstruct_mask_and_weight_map, pairwise_stability = (
        _load_stability_functions()
    )

    checkpoint_path, run_dir = _resolve_checkpoint(args.model, args.checkpoint, args.run)
    use_baseline = lime_cfg["baseline"]
    fallback_splits_dir = (
        get_output_root() / "splits" / ("seed_42_baseline" if use_baseline else "seed_42")
    )
    _, class_to_idx, idx_to_class, target_size = load_run_metadata(
        run_dir=run_dir,
        fallback_splits_dir=fallback_splits_dir,
        fallback_classes=cfg["dataset"]["classes"],
        fallback_target_size=tuple(cfg["dataset"]["target_size"]),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else get_output_root() / "inference" / image_path.stem / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    loaded = load_validated_run(
        checkpoint_path,
        expected_model=args.model,
        expected_input_size=target_size,
        expected_class_to_idx=class_to_idx,
        device=device,
        config_path=str(config_path),
    )
    model = loaded.model

    image = load_and_normalize_image(image_path)
    gradcam_model_name = args.model if gradcam_enabled else None

    logger.info(f"Imagen: {image_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    prediction = _run_inference(
        model=model,
        image=image,
        inference_transform=loaded.factory.get_pipeline("inference"),
        idx_to_class=idx_to_class,
        device=device,
        top_k=args.top_k,
    )
    logger.info(
        f"Diagnóstico: {prediction['predicted_label']} "
        f"(confianza: {prediction['predicted_prob'] * 100:.1f}%)"
    )

    logger.info("Generando explicación LIME + Grad-CAM...")
    explanation = render_visual_explanation(
        image=image,
        model=model,
        idx_to_class=idx_to_class,
        target_size=target_size,
        output_path=output_dir / "explanation.png",
        num_samples=lime_cfg["num_samples"],
        num_features=lime_cfg["num_features"],
        seed=lime_cfg["seed"],
        device=device,
        model_name=gradcam_model_name,
        validation_transform=loaded.factory.get_pipeline("inference"),
    )
    if explanation["predicted_label"] != prediction["predicted_label"]:
        logger.warning(
            f"LIME predice '{explanation['predicted_label']}' "
            f"({explanation['predicted_prob'] * 100:.1f}%) en vez de "
            f"'{prediction['predicted_label']}': LIME reescala la imagen dos veces "
            "(PIL bicúbico + T.Resize bilineal), así que ve píxeles ligeramente distintos. "
            "La predicción fiel al pipeline de entrenamiento es la de prediction.json; "
            "la divergencia indica un margen estrecho entre clases."
        )

    stability = None
    if args.stability_runs >= 2:
        logger.info(f"Auditando estabilidad de LIME ({args.stability_runs} seeds)...")
        stability = _run_stability(
            render_visual_explanation=render_visual_explanation,
            reconstruct_mask_and_weight_map=reconstruct_mask_and_weight_map,
            pairwise_stability=pairwise_stability,
            model=model,
            model_name=gradcam_model_name,
            image=image,
            idx_to_class=idx_to_class,
            target_size=target_size,
            output_dir=output_dir / "stability",
            image_stem=image_path.stem,
            lime_cfg=lime_cfg,
            device=device,
            runs=args.stability_runs,
            validation_transform=loaded.factory.get_pipeline("inference"),
        )
        (output_dir / "stability.json").write_text(
            json.dumps(stability, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for pair in stability["pairs"]:
            logger.info(
                f"  seeds {pair['seed_a']} vs {pair['seed_b']}: "
                f"IoU={pair['iou']:.3f} correlación={pair['correlation']:.3f}"
            )
    elif args.stability_runs == 1:
        logger.warning("--stability-runs=1 no permite comparar pares; auditoría omitida.")

    (output_dir / "prediction.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "model": args.model,
                "checkpoint": str(checkpoint_path),
                "device": str(device),
                "target_size": list(target_size),
                "timestamp": timestamp,
                # Puede diferir de predicted_label: LIME reescala la imagen dos veces
                # (PIL bicúbico + T.Resize bilineal) y evalúa píxeles ligeramente
                # distintos. La predicción de referencia es predicted_label.
                "lime_predicted_label": explanation["predicted_label"],
                "lime_predicted_prob": explanation["predicted_prob"],
                **prediction,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(f"Reporte completo guardado en {output_dir}")


if __name__ == "__main__":
    main()
