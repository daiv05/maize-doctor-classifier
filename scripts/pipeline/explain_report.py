"""
Genera un reporte agregado de fidelidad de LIME sobre checkpoints ya entrenados,
cruzando cada explicación con `predictions.csv` (generado por train_baselines.py) para
saber si la predicción fue correcta.

Dos modos:
  - Muestreo amplio (default): explica `--sample-size` imágenes por clase (mucho más
    que las 2/clase de `explain_lime.py`) y agrega confianza/dispersión por clase y
    acierto/error en <run_dir>/lime_report/ + <run_dir>/explain_report/summary.{csv,json}.
  - --errors-only: ignora --sample-size y explica TODAS las filas donde
    predictions.csv tiene label != pred_label, en <run_dir>/lime_errors/.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.fastvit  # noqa: F401 - registra modelos
import src.models.baselines.ghostnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
import src.models.baselines.shufflenet  # noqa: F401 - registra modelos
from src.config import PROJECT_ROOT, get_dataset_root, get_output_root, set_global_seed
from src.data.loader import load_and_normalize_image
from src.explainability.visual_report import (
    explanation_dispersion,
    render_visual_explanation,
    sample_balanced,
)
from src.models.registry import MODEL_REGISTRY
from src.training.common import load_run_metadata, resolve_run_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = get_output_root() / "baselines"


def _resolve_model_names(requested: list[str]) -> list[str]:
    available = MODEL_REGISTRY.list_names()
    if requested == ["all"]:
        return available
    unknown = [n for n in requested if n not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"Modelos desconocidos: {unknown}. Disponibles: {available}")
    return requested


def _explain_subset(
    df_subset: pd.DataFrame,
    model,
    model_name: str,
    idx_to_class: dict[int, str],
    target_size: tuple[int, int],
    lime_dir,
    num_features: int,
    num_samples: int,
    seed: int,
    device: torch.device,
    enable_gradcam: bool,
) -> list[dict]:
    """Explica cada fila de `df_subset` (columnas image_path/label/pred_label/pred_prob),
    guarda el reporte visual bajo `lime_dir` y devuelve, por fila, un dict con
    correctitud y dispersión de la explicación (leída del .json que ya persiste
    render_visual_explanation)."""
    rows: list[dict] = []
    dataset_root = get_dataset_root()

    for _, row in df_subset.iterrows():
        img_path = dataset_root / row["image_path"]
        try:
            image = load_and_normalize_image(img_path)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"[{model_name}] Saltando {img_path}: {e}")
            continue

        stem = img_path.stem
        output_path = lime_dir / f"{stem}__true-{row['label']}__pred-{row['pred_label']}.png"

        render_visual_explanation(
            image=image,
            model=model,
            idx_to_class=idx_to_class,
            target_size=target_size,
            output_path=output_path,
            num_samples=num_samples,
            num_features=num_features,
            seed=seed,
            device=device,
            model_name=model_name if enable_gradcam else None,
        )

        metadata = json.loads(output_path.with_suffix(".json").read_text())
        top_features = metadata["top_features"]
        local_exp = [(f["segment_id"], f["weight"]) for f in top_features]

        rows.append(
            {
                "image_path": row["image_path"],
                "label": row["label"],
                "pred_label": row["pred_label"],
                "pred_prob": row["pred_prob"],
                "correct": row["label"] == row["pred_label"],
                "dispersion": explanation_dispersion(local_exp) if local_exp else 0.0,
            }
        )
        logger.info(f"[{model_name}] {img_path.name}: explicado (correct={rows[-1]['correct']})")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reporte agregado de fidelidad LIME + análisis de errores sobre "
        "checkpoints ya entrenados. Requiere predictions.csv (generado por "
        "train_baselines.py) en el run_dir de cada modelo."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help='Nombres de modelos a explicar, o "all" para todos. '
        f"Disponibles: {MODEL_REGISTRY.list_names()}",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="run_id específico (por defecto, el último registrado en latest.json).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=None,
        help="Fuerza splits/seed_42_baseline en vez de leer lime.baseline del YAML.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        dest="sample_size",
        help="Imágenes por clase para el muestreo amplio (default: lime.report_sample_size "
        "de config/dataset.yaml). Ignorado con --errors-only.",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        dest="errors_only",
        help="Explica solo las filas de predictions.csv con label != pred_label "
        "(todas, sin muestreo), en vez del muestreo balanceado amplio.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        dest="num_samples",
        help="Override puntual de lime.num_samples (perturbaciones LIME por imagen), "
        "útil para pruebas rápidas sin tocar el YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio raíz donde buscar los runs, con un subdirectorio por modelo. "
        f"Default: {_DEFAULT_OUTPUT_DIR}. Usa outputs/main para los runs de train.py.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR

    with open(PROJECT_ROOT / "config" / "dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    lime_cfg = cfg["lime"]
    gradcam_enabled = cfg.get("gradcam", {}).get("enabled", False)
    set_global_seed(lime_cfg["seed"])

    model_names = _resolve_model_names(args.models)
    use_baseline = args.baseline if args.baseline is not None else lime_cfg["baseline"]
    # Fallbacks solo para runs sin summary.json; la fuente de verdad es el summary.json de
    # cada run (ver load_run_metadata). Nunca reconstruir el mapeo desde baseline.classes: su
    # orden puede diferir del canónico dataset.classes con el que se entrenó el head -> rótulos
    # permutados en los reportes.
    fallback_splits_dir = get_output_root() / "splits" / (
        "seed_42_baseline" if use_baseline else "seed_42"
    )
    fallback_classes = cfg["baseline"]["classes"] if use_baseline else cfg["dataset"]["classes"]
    fallback_target_size = tuple(cfg["dataset"]["target_size"])
    sample_size = args.sample_size or lime_cfg["report_sample_size"]
    num_samples = args.num_samples or lime_cfg["num_samples"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    logger.info(f"Modelos a explicar: {model_names}")

    for model_name in model_names:
        try:
            run_dir = resolve_run_dir(output_dir, model_name, args.run)
        except SystemExit as e:
            logger.warning(f"[{model_name}] {e}. Se omite.")
            continue

        checkpoint_path = run_dir / "best.pth"
        predictions_path = run_dir / "predictions.csv"
        if not checkpoint_path.exists():
            logger.warning(f"[{model_name}] Run {run_dir.name} sin checkpoint completo, se omite.")
            continue
        if not predictions_path.exists():
            logger.warning(
                f"[{model_name}] {predictions_path} no existe. Corre `make train-baselines` "
                "(o re-entrena) para generar predictions.csv. Se omite."
            )
            continue

        predictions_df = pd.read_csv(predictions_path)

        if args.errors_only:
            df_subset = predictions_df[predictions_df["label"] != predictions_df["pred_label"]]
            lime_dir = run_dir / "lime_errors"
            logger.info(f"[{model_name}] {len(df_subset)} errores encontrados en predictions.csv")
        else:
            df_subset = sample_balanced(predictions_df, sample_size, lime_cfg["seed"])
            lime_dir = run_dir / "lime_report"

        if df_subset.empty:
            logger.info(f"[{model_name}] Nada que explicar (subconjunto vacío). Se omite.")
            continue

        # Mapeo clase->índice y tamaño de entrada del checkpoint concreto (per-run, desde su
        # summary.json): el mismo que usó el head al entrenar y el mismo image_size por-modelo.
        _, _, idx_to_class, target_size = load_run_metadata(
            run_dir=run_dir,
            fallback_splits_dir=fallback_splits_dir,
            fallback_classes=fallback_classes,
            fallback_target_size=fallback_target_size,
        )
        num_classes = len(idx_to_class)

        model = MODEL_REGISTRY.build(model_name, num_classes=num_classes, pretrained=False).to(
            device
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        rows = _explain_subset(
            df_subset=df_subset,
            model=model,
            model_name=model_name,
            idx_to_class=idx_to_class,
            target_size=target_size,
            lime_dir=lime_dir,
            num_features=lime_cfg["num_features"],
            num_samples=num_samples,
            seed=lime_cfg["seed"],
            device=device,
            enable_gradcam=gradcam_enabled,
        )

        if not rows:
            continue

        report_df = pd.DataFrame(rows)
        summary = (
            report_df.groupby(["label", "correct"])
            .agg(
                n=("dispersion", "size"),
                mean_pred_prob=("pred_prob", "mean"),
                mean_dispersion=("dispersion", "mean"),
            )
            .reset_index()
        )

        report_dir = run_dir / "explain_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(report_dir / "summary.csv", index=False)
        (report_dir / "summary.json").write_text(summary.to_json(orient="records", indent=2))

        logger.info(f"[{model_name}] Resumen de fidelidad:\n{summary.to_string(index=False)}")
        logger.info(f"[{model_name}] Reportes guardados en {lime_dir} y {report_dir}")

    logger.info("Reporte de explicabilidad completado.")


if __name__ == "__main__":
    main()
