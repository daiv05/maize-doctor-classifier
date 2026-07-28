"""CLI local de explicabilidad post-hoc sobre checkpoints ya entrenados.

Su equivalente en la nube es scripts/modal/explain.py, que orquesta este mismo script por
subprocess sobre una GPU de Modal.

Subcomandos:
  visual    Panel por imagen: LIME + Grad-CAM         -> <run_dir>/explain_visual/
  fidelity  Muestra amplia + resumen agregado         -> <run_dir>/explain_fidelity/
  errors    Panel sobre las predicciones erroneas     -> <run_dir>/explain_errors/
  compare   Panel LIME | SHAP | Grad-CAM + acuerdo    -> <run_dir>/explain_compare/
  global    Perfil global por clase                   -> <run_dir>/explain_global/

Los dos ultimos son exclusivos del pipeline principal (outputs/main).
"""

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml

import src.models.baselines.efficientnet  # noqa: F401 - registra modelos
import src.models.baselines.fastvit  # noqa: F401 - registra modelos
import src.models.baselines.ghostnet  # noqa: F401 - registra modelos
import src.models.baselines.mobilenet  # noqa: F401 - registra modelos
import src.models.baselines.shufflenet  # noqa: F401 - registra modelos
from src.config import PROJECT_ROOT, get_dataset_root, get_output_root, set_global_seed
from src.data.loader import load_and_normalize_image
from src.models.registry import MODEL_REGISTRY
from src.training.common import (
    load_run_metadata,
    resolve_model_names,
    resolve_run_dir,
    select_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = get_output_root() / "baselines"


@dataclass
class RunContext:
    """Todo lo necesario para explicar un run concreto de un modelo concreto."""

    model_name: str
    run_dir: Path
    model: nn.Module
    idx_to_class: dict[int, str]
    target_size: tuple[int, int]
    splits_dir: Path
    device: torch.device


def load_config() -> dict:
    """
    Lee config/dataset.yaml.

    @returns {dict} Configuracion completa del proyecto.
    """
    with open(PROJECT_ROOT / "config" / "dataset.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _common_parser() -> argparse.ArgumentParser:
    """
    Parser padre con los flags que comparten los cinco subcomandos.

    @returns {argparse.ArgumentParser} Parser sin help propio, para usar como parent.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f'Modelos a explicar, o "all". Disponibles: {MODEL_REGISTRY.list_names()}',
    )
    parser.add_argument(
        "--run",
        default=None,
        help="run_id especifico (por defecto, el ultimo registrado en latest.json).",
    )
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
        help="Directorio raiz de runs, con un subdirectorio por modelo. "
        f"Default: {_DEFAULT_OUTPUT_DIR}. Usa outputs/main para los runs de train.py.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """
    Arma el parser con los cinco subcomandos.

    @returns {argparse.ArgumentParser} Parser listo para `parse_args`.
    """
    common = _common_parser()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    visual = subparsers.add_parser(
        "visual", parents=[common], help="Panel LIME + Grad-CAM por imagen."
    )
    visual.add_argument(
        "--image",
        default=None,
        help="Explica una imagen puntual en vez del muestreo balanceado del test set.",
    )
    visual.add_argument(
        "--output",
        default=None,
        help="PNG de salida. Solo valido junto con --image y un unico modelo.",
    )

    fidelity = subparsers.add_parser(
        "fidelity", parents=[common], help="Muestra amplia + resumen agregado por clase."
    )
    fidelity.add_argument(
        "--sample-size",
        type=int,
        default=None,
        dest="sample_size",
        help="Imagenes por clase (default: lime.report_sample_size).",
    )
    fidelity.add_argument(
        "--num-samples",
        type=int,
        default=None,
        dest="num_samples",
        help="Override de lime.num_samples (perturbaciones por imagen).",
    )

    errors = subparsers.add_parser(
        "errors", parents=[common], help="Panel sobre las predicciones erroneas."
    )
    errors.add_argument(
        "--num-samples",
        type=int,
        default=None,
        dest="num_samples",
        help="Override de lime.num_samples (perturbaciones por imagen).",
    )

    compare = subparsers.add_parser(
        "compare", parents=[common], help="Panel LIME | SHAP | Grad-CAM + acuerdo."
    )
    compare.add_argument(
        "--sample-size",
        type=int,
        default=None,
        dest="sample_size",
        help="Imagenes por clase (default: shap.images_per_class).",
    )
    compare.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Override de shap.nsamples (evaluaciones de KernelSHAP por imagen).",
    )

    global_profile = subparsers.add_parser(
        "global", parents=[common], help="Perfil global por clase."
    )
    global_profile.add_argument(
        "--sample-size",
        type=int,
        default=None,
        dest="sample_size",
        help="Imagenes por clase (default: shap.global_sample_size).",
    )
    global_profile.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Override de shap.nsamples (evaluaciones de KernelSHAP por imagen).",
    )

    return parser


def _fallback_splits_dir(cfg: dict, baseline: bool | None) -> Path:
    """
    Resuelve el directorio de splits de respaldo para runs sin summary.json.

    @param {dict} cfg Configuracion del proyecto.
    @param {bool|None} baseline Valor del flag --baseline, o None para leerlo del YAML.
    @returns {Path} Directorio de splits.
    """
    use_baseline = baseline if baseline is not None else cfg["lime"]["baseline"]
    return get_output_root() / "splits" / ("seed_42_baseline" if use_baseline else "seed_42")


def iter_run_contexts(
    args: argparse.Namespace, cfg: dict, device: torch.device, require_predictions: bool
) -> Iterator[RunContext]:
    """
    Resuelve, para cada modelo pedido, su run y su checkpoint ya cargado.

    Las clases de respaldo salen siempre de `dataset.classes` (orden canonico); solo se
    usan cuando el run no tiene summary.json, que es la fuente de verdad del mapeo con el
    que se entreno el head.

    @param {argparse.Namespace} args Argumentos ya parseados.
    @param {dict} cfg Configuracion del proyecto.
    @param {torch.device} device Dispositivo de computo.
    @param {bool} require_predictions Si el subcomando necesita predictions.csv.
    @returns {Iterator[RunContext]} Un contexto por modelo resoluble; el resto se omite.
    """
    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR
    splits_fallback = _fallback_splits_dir(cfg, args.baseline)

    for model_name in resolve_model_names(args.models, MODEL_REGISTRY):
        try:
            run_dir = resolve_run_dir(output_dir, model_name, args.run)
        except SystemExit as error:
            logger.warning(f"[{model_name}] {error}. Se omite.")
            continue

        if not (run_dir / "best.pth").exists():
            logger.warning(f"[{model_name}] Run {run_dir.name} sin checkpoint, se omite.")
            continue
        if require_predictions and not (run_dir / "predictions.csv").exists():
            logger.warning(
                f"[{model_name}] Falta {run_dir / 'predictions.csv'}. "
                "Re-entrena para generarlo. Se omite."
            )
            continue

        splits_dir, _, idx_to_class, target_size = load_run_metadata(
            run_dir=run_dir,
            fallback_splits_dir=splits_fallback,
            fallback_classes=cfg["dataset"]["classes"],
            fallback_target_size=tuple(cfg["dataset"]["target_size"]),
        )

        model = MODEL_REGISTRY.build(
            model_name, num_classes=len(idx_to_class), pretrained=False
        ).to(device)
        model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
        model.eval()

        yield RunContext(
            model_name=model_name,
            run_dir=run_dir,
            model=model,
            idx_to_class=idx_to_class,
            target_size=target_size,
            splits_dir=splits_dir,
            device=device,
        )


def cmd_visual(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """
    Panel LIME + Grad-CAM: muestreo balanceado del test set, o una imagen puntual.

    @param {argparse.Namespace} args Argumentos del subcomando.
    @param {dict} cfg Configuracion del proyecto.
    @param {torch.device} device Dispositivo de computo.
    @throws {SystemExit} Si --output se usa sin --image o con varios modelos.
    """
    from src.explainability.visual_report import explain_model_visual, render_visual_explanation

    lime_cfg = cfg["lime"]
    gradcam_enabled = cfg.get("gradcam", {}).get("enabled", False)
    resolved_models = resolve_model_names(args.models, MODEL_REGISTRY)

    if args.output is not None and (args.image is None or len(resolved_models) != 1):
        raise SystemExit("--output solo es valido junto con --image y un unico modelo.")

    for context in iter_run_contexts(args, cfg, device, require_predictions=False):
        gradcam_name = context.model_name if gradcam_enabled else None

        if args.image is not None:
            image_path = Path(args.image)
            output_path = (
                Path(args.output)
                if args.output is not None
                else context.run_dir / "explain_visual" / f"{image_path.stem}.png"
            )
            result = render_visual_explanation(
                image=load_and_normalize_image(image_path),
                model=context.model,
                idx_to_class=context.idx_to_class,
                target_size=context.target_size,
                output_path=output_path,
                num_samples=lime_cfg["num_samples"],
                num_features=lime_cfg["num_features"],
                seed=lime_cfg["seed"],
                device=device,
                model_name=gradcam_name,
            )
            logger.info(
                f"[{context.model_name}] Diagnostico: {result['predicted_label']} "
                f"({result['predicted_prob'] * 100:.1f}%)"
            )
            continue

        explain_model_visual(
            model=context.model,
            model_name=context.model_name,
            test_df=pd.read_csv(context.splits_dir / "test.csv"),
            dataset_root=get_dataset_root(),
            idx_to_class=context.idx_to_class,
            target_size=context.target_size,
            output_dir=context.run_dir,
            images_per_class=lime_cfg["images_per_class"],
            num_features=lime_cfg["num_features"],
            num_samples=lime_cfg["num_samples"],
            seed=lime_cfg["seed"],
            device=device,
            enable_gradcam=gradcam_enabled,
        )


def _explain_subset(
    df_subset: pd.DataFrame,
    context: RunContext,
    panel_dir: Path,
    lime_cfg: dict,
    num_samples: int,
    gradcam_enabled: bool,
) -> list[dict]:
    """
    Explica cada fila del subconjunto y devuelve su correctitud y dispersion.

    @param {pd.DataFrame} df_subset Filas de predictions.csv a explicar.
    @param {RunContext} context Run y checkpoint ya cargados.
    @param {Path} panel_dir Directorio donde guardar los paneles.
    @param {dict} lime_cfg Bloque `lime` de la configuracion.
    @param {int} num_samples Perturbaciones de LIME por imagen.
    @param {bool} gradcam_enabled Si se anade el panel Grad-CAM.
    @returns {list[dict]} Un registro por imagen explicada.
    """
    from src.explainability.visual_report import explanation_dispersion, render_visual_explanation

    dataset_root = get_dataset_root()
    rows: list[dict] = []

    for _, row in df_subset.iterrows():
        image_path = dataset_root / row["image_path"]
        try:
            image = load_and_normalize_image(image_path)
        except (FileNotFoundError, RuntimeError) as error:
            logger.warning(f"[{context.model_name}] Saltando {image_path}: {error}")
            continue

        output_path = (
            panel_dir / f"{image_path.stem}__true-{row['label']}__pred-{row['pred_label']}.png"
        )
        render_visual_explanation(
            image=image,
            model=context.model,
            idx_to_class=context.idx_to_class,
            target_size=context.target_size,
            output_path=output_path,
            num_samples=num_samples,
            num_features=lime_cfg["num_features"],
            seed=lime_cfg["seed"],
            device=context.device,
            model_name=context.model_name if gradcam_enabled else None,
        )

        metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
        local_exp = [
            (feature["segment_id"], feature["weight"]) for feature in metadata["top_features"]
        ]
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
        logger.info(
            f"[{context.model_name}] {image_path.name}: explicado (correct={rows[-1]['correct']})"
        )

    return rows


def _write_fidelity_summary(rows: list[dict], output_dir: Path, model_name: str) -> None:
    """
    Agrega los registros por clase y correctitud y los persiste.

    @param {list[dict]} rows Registros devueltos por `_explain_subset`.
    @param {Path} output_dir Directorio destino del resumen.
    @param {str} model_name Nombre del modelo, solo para el log.
    """
    summary = (
        pd.DataFrame(rows)
        .groupby(["label", "correct"])
        .agg(
            n=("dispersion", "size"),
            mean_pred_prob=("pred_prob", "mean"),
            mean_dispersion=("dispersion", "mean"),
        )
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    logger.info(f"[{model_name}] Resumen:\n{summary.to_string(index=False)}")


def _run_panel_report(
    args: argparse.Namespace, cfg: dict, device: torch.device, errors_only: bool
) -> None:
    """
    Tronco comun de `fidelity` y `errors`: ambos explican un subconjunto y lo agregan.

    @param {argparse.Namespace} args Argumentos del subcomando.
    @param {dict} cfg Configuracion del proyecto.
    @param {torch.device} device Dispositivo de computo.
    @param {bool} errors_only True para explicar solo las predicciones erroneas.
    """
    from src.explainability.visual_report import sample_balanced

    lime_cfg = cfg["lime"]
    gradcam_enabled = cfg.get("gradcam", {}).get("enabled", False)
    num_samples = args.num_samples or lime_cfg["num_samples"]
    directory = "explain_errors" if errors_only else "explain_fidelity"

    for context in iter_run_contexts(args, cfg, device, require_predictions=True):
        predictions = pd.read_csv(context.run_dir / "predictions.csv")

        if errors_only:
            df_subset = predictions[predictions["label"] != predictions["pred_label"]]
            logger.info(f"[{context.model_name}] {len(df_subset)} errores en predictions.csv")
        else:
            sample_size = args.sample_size or lime_cfg["report_sample_size"]
            df_subset = sample_balanced(predictions, sample_size, lime_cfg["seed"])

        if df_subset.empty:
            logger.info(f"[{context.model_name}] Nada que explicar. Se omite.")
            continue

        panel_dir = context.run_dir / directory
        rows = _explain_subset(
            df_subset=df_subset,
            context=context,
            panel_dir=panel_dir,
            lime_cfg=lime_cfg,
            num_samples=num_samples,
            gradcam_enabled=gradcam_enabled,
        )
        if rows:
            _write_fidelity_summary(rows, panel_dir, context.model_name)


def cmd_fidelity(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """
    Muestra amplia balanceada + resumen agregado por clase.

    @param {argparse.Namespace} args Argumentos del subcomando.
    @param {dict} cfg Configuracion del proyecto.
    @param {torch.device} device Dispositivo de computo.
    """
    _run_panel_report(args, cfg, device, errors_only=False)


def cmd_errors(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """
    Paneles sobre todas las predicciones erroneas del run.

    @param {argparse.Namespace} args Argumentos del subcomando.
    @param {dict} cfg Configuracion del proyecto.
    @param {torch.device} device Dispositivo de computo.
    """
    _run_panel_report(args, cfg, device, errors_only=True)


def cmd_compare(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """Panel comparado LIME | SHAP | Grad-CAM. Se implementa en la Tarea 10."""
    raise SystemExit("El subcomando 'compare' aun no esta implementado.")


def cmd_global(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """Perfil global por clase. Se implementa en la Tarea 10."""
    raise SystemExit("El subcomando 'global' aun no esta implementado.")


def main() -> None:
    """Punto de entrada: parsea, siembra y despacha al subcomando."""
    args = build_parser().parse_args()
    cfg = load_config()
    set_global_seed(cfg["lime"]["seed"])

    device = select_device()
    logger.info(f"Subcomando: {args.command}")

    handlers = {
        "visual": cmd_visual,
        "fidelity": cmd_fidelity,
        "errors": cmd_errors,
        "compare": cmd_compare,
        "global": cmd_global,
    }
    handlers[args.command](args, cfg, device)
    logger.info(f"Subcomando '{args.command}' completado.")


if __name__ == "__main__":
    main()
