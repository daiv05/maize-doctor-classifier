"""Calcula estadisticas de deteccion OOD (out-of-distribution) por Relative Mahalanobis
Distance (RMD) sobre un espacio de features reducido por PCA.

Para cada modelo, extrae el vector de features pooled (penultima capa, via
`FeatureExposedModel`) sobre el split de train, normaliza cada vector a norma L2
unitaria (Mahalanobis++, Ren et al. 2025: arXiv:2505.18032), y proyecta a las
componentes principales que explican el 99% de la varianza (`_fit_pca`). Sobre ese
espacio reducido calcula el centroide por clase mas una covarianza pooled compartida,
mas una gaussiana de fondo (sin condicionar por clase) sobre todo el split de train. El
score final de cada muestra es la distancia de Mahalanobis a la clase mas cercana MENOS
la distancia a la gaussiana de fondo (RMD, Ren et al. 2021: arXiv:2106.09022).

La reduccion PCA es necesaria porque el feature vector crudo (1280 dimensiones) tiene un
espectro de autovalores muy sesgado: solo un centenar de dimensiones concentran varianza
real, el resto es esencialmente ruido numerico. Invertir la covarianza cruda de 1280x1280
(incluso regularizada con un ridge pequeno) deja esas dimensiones de ruido con un peso
desproporcionado en la distancia, y el detector no separa nada. Truncar a las componentes
que explican el 99% de la varianza elimina ese ruido antes de que pueda dominar la suma.
Ambas covarianzas (clase y fondo), ya en el espacio reducido, se regularizan (ridge
proporcional a su traza) antes de invertir. El umbral de rechazo se calibra sobre el
split de val (no visto durante el entrenamiento) como el percentil pedido del score RMD
de cada muestra respecto a su propio centroide de clase.

Metodologia completa: docs/es/deep-learning/ood-detection.md.

Escribe `<run_dir>/export/ood_stats.json`, consumido por la app movil junto al
`.tflite` de dos salidas (logits + features) para bloquear diagnosticos sobre
imagenes fuera de dominio.
"""

import argparse
import base64
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT, get_output_root
from src.export.common import load_checkpoint_for_export, resolve_export_inputs
from src.export.data import build_test_loader, resolve_split_csv
from src.models import list_models
from src.models.feature_exposed import FeatureExposedModel
from src.provenance import atomic_json, contract_hash, sha256_file
from src.training.common import resolve_run_dir, select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula estadisticas de deteccion OOD (Mahalanobis) por modelo."
    )
    parser.add_argument(
        "--models", nargs="+", required=True, choices=list_models(), help="Modelos a procesar."
    )
    parser.add_argument("--run", default=None, help="run_id; por defecto usa latest.json.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta explicita a un checkpoint .pth (solo valido con un unico modelo).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directorio de runs del pipeline principal (default: <outputs>/main).",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        dest="splits_dir",
        help="Directorio con train.csv/val.csv (default: el 'splits_dir' de summary.json).",
    )
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--explained-variance",
        type=float,
        default=0.99,
        dest="explained_variance",
        help="Fraccion de varianza a retener al reducir el feature vector via PCA "
        "antes de ajustar la gaussiana (default: 0.99). "
        "Ver docs/es/deep-learning/ood-detection.md.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        # p99 resulto inestable en la practica: la cola de distancias de val puede
        # tener outliers extremos (imagenes atipicas/mal etiquetadas) que inflan el
        # percentil 99 muy por encima de donde vive la mayoria de los datos legitimos
        # (visto en efficientnet_lite0: p95=8814 pero p99=74821, 9x mas), dejando
        # pasar imagenes OOD reales. p95 calibra sobre la distribucion "normal".
        help="Percentil de calibracion del umbral sobre las distancias de val (default: 95).",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "dataset.yaml"))
    return parser.parse_args()


@torch.no_grad()
def _extract_features(
    model: FeatureExposedModel, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corre el modelo sobre todo el loader y acumula features pooled + labels.

    @param {FeatureExposedModel} model Modelo envuelto, en eval().
    @param {DataLoader} loader Loader determinista (pipeline 'test', sin augmentation).
    @param {torch.device} device Dispositivo de inferencia.
    @returns {tuple[np.ndarray, np.ndarray]} Features (N, feature_dim) y labels (N,).
    """
    model.eval()
    all_features = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        _, features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    """
    Normaliza cada vector de features a norma L2 unitaria.

    @param {np.ndarray} features Features (N, feature_dim).
    @returns {np.ndarray} Features normalizadas, misma forma.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms


def _fit_pca(
    features: np.ndarray, explained_variance: float = 0.99
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Ajusta PCA sobre `features` (ya L2-normalizadas) y devuelve las componentes que
    explican al menos `explained_variance` de la varianza total.

    @param {np.ndarray} features Features (N, feature_dim).
    @param {float} explained_variance Fraccion minima de varianza acumulada a retener.
    @returns {tuple[np.ndarray,np.ndarray,float]} `(mean, components, explained_actual)`.
        `mean` es (feature_dim,); `components` es (k, feature_dim), filas ortonormales
        ordenadas por varianza explicada descendente; `explained_actual` es la fraccion
        de varianza realmente retenida con esas k componentes.
    """
    mean = features.mean(axis=0)
    centered = features - mean
    # SVD economica sobre (N, feature_dim) en vez de diagonalizar la covarianza
    # (feature_dim, feature_dim): evita construir esa matriz solo para esto.
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    cumulative = np.cumsum(variance) / np.sum(variance)
    k = int(np.searchsorted(cumulative, explained_variance) + 1)
    k = min(k, vt.shape[0])
    return mean, vt[:k], float(cumulative[k - 1])


def _apply_pca(
    features: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray
) -> np.ndarray:
    """
    Proyecta `features` al espacio reducido por PCA.

    @param {np.ndarray} features Features (N, feature_dim), en el mismo espacio (L2-normalizado)
        en que se ajusto `_fit_pca`.
    @param {np.ndarray} pca_mean Media usada al ajustar PCA (feature_dim,).
    @param {np.ndarray} pca_components Componentes principales (k, feature_dim).
    @returns {np.ndarray} Features proyectadas (N, k).
    """
    return (features - pca_mean) @ pca_components.T


def _regularize_covariance(covariance: np.ndarray, ridge_scale: float = 1e-3) -> np.ndarray:
    """
    Suma una matriz identidad escalada a la covarianza antes de invertir.

    El ridge es proporcional a la traza (varianza promedio por dimension) en vez de un
    valor absoluto fijo, para que la magnitud de la regularizacion se adapte a la escala
    de las features (aqui, normalizadas a norma L2 unitaria).

    @param {np.ndarray} covariance Covarianza (feature_dim, feature_dim).
    @param {float} ridge_scale Fraccion de la varianza promedio sumada a la diagonal.
    @returns {np.ndarray} Covarianza regularizada, misma forma.
    """
    feature_dim = covariance.shape[0]
    ridge = ridge_scale * (np.trace(covariance) / feature_dim)
    return covariance + ridge * np.eye(feature_dim)


def _compute_class_means(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Calcula el centroide (media) de features por clase.

    @param {np.ndarray} features Features (N, feature_dim).
    @param {np.ndarray} labels Indice de clase por muestra (N,).
    @param {int} num_classes Numero total de clases.
    @returns {np.ndarray} Centroides (num_classes, feature_dim).
    """
    feature_dim = features.shape[1]
    means = np.zeros((num_classes, feature_dim), dtype=np.float64)
    for class_idx in range(num_classes):
        class_features = features[labels == class_idx]
        if class_features.shape[0] == 0:
            raise SystemExit(f"La clase {class_idx} no tiene ninguna muestra en el split de train.")
        means[class_idx] = class_features.mean(axis=0)
    return means


def _compute_pooled_covariance(
    features: np.ndarray, labels: np.ndarray, means: np.ndarray
) -> np.ndarray:
    """
    Calcula la covarianza pooled: cada muestra se centra con el centroide de SU clase,
    luego se calcula una unica covarianza sobre todas las muestras centradas.

    Estandar del paper de Lee et al. 2018 (deteccion OOD via Mahalanobis) - una sola
    matriz compartida entre clases, mas robusta con datasets de tamano moderado que
    invertir una covarianza por clase.

    @param {np.ndarray} features Features (N, feature_dim).
    @param {np.ndarray} labels Indice de clase por muestra (N,).
    @param {np.ndarray} means Centroides por clase (num_classes, feature_dim).
    @returns {np.ndarray} Covarianza pooled (feature_dim, feature_dim).
    """
    centered = features - means[labels]
    n_samples = centered.shape[0]
    return (centered.T @ centered) / n_samples


def _encode_float32_base64(array: np.ndarray) -> str:
    """
    Codifica un array como bytes float32 crudos (row-major) en base64.

    Evita serializar millones de numeros como texto JSON (un array 1024x1024
    en JSON de texto pesa ~20MB; el binario float32 equivalente son 4MB, ~5.3MB
    en base64). La app decodifica el string y reinterpreta los bytes como
    Float32Array, sin parsear un array JSON gigante.

    @param {np.ndarray} array Array numerico de cualquier forma.
    @returns {str} Bytes float32 (row-major) codificados en base64.
    """
    return base64.b64encode(array.astype(np.float32).tobytes()).decode("ascii")


def _mahalanobis_distances(
    features: np.ndarray, labels: np.ndarray, means: np.ndarray, inv_covariance: np.ndarray
) -> np.ndarray:
    """
    Distancia de Mahalanobis de cada muestra a el centroide de SU propia clase.

    @param {np.ndarray} features Features (N, feature_dim).
    @param {np.ndarray} labels Indice de clase por muestra (N,).
    @param {np.ndarray} means Centroides por clase (num_classes, feature_dim).
    @param {np.ndarray} inv_covariance Inversa (pseudo-inversa) de la covarianza pooled.
    @returns {np.ndarray} Distancias (N,).
    """
    diff = features - means[labels]
    return np.einsum("ij,jk,ik->i", diff, inv_covariance, diff)


def _mahalanobis_to_mean(
    features: np.ndarray, mean: np.ndarray, inv_covariance: np.ndarray
) -> np.ndarray:
    """
    Distancia de Mahalanobis de cada muestra a una unica media (sin condicionar por clase).

    @param {np.ndarray} features Features (N, feature_dim).
    @param {np.ndarray} mean Media (feature_dim,).
    @param {np.ndarray} inv_covariance Inversa (pseudo-inversa) de la covarianza.
    @returns {np.ndarray} Distancias (N,).
    """
    diff = features - mean
    return np.einsum("ij,jk,ik->i", diff, inv_covariance, diff)


def _compute_one(args: argparse.Namespace, model_name: str, output_dir: Path) -> None:
    config_path = Path(args.config)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        run_dir = checkpoint_path.parent
    else:
        run_dir = resolve_run_dir(output_dir, model_name, args.run)
        checkpoint_path = run_dir / "best.pth"

    class_to_idx, idx_to_class, image_size = resolve_export_inputs(run_dir, model_name, config_path)
    num_classes = len(class_to_idx)
    device = select_device()

    base_model = load_checkpoint_for_export(checkpoint_path, model_name, class_to_idx, device)
    model = FeatureExposedModel(base_model, model_name).to(device)
    model.eval()

    train_csv = resolve_split_csv(run_dir, args.splits_dir, "train")
    val_csv = resolve_split_csv(run_dir, args.splits_dir, "val")

    logger.info("Extrayendo features de train (%s)...", train_csv)
    train_loader, _ = build_test_loader(
        train_csv,
        config_path,
        class_to_idx,
        image_size,
        args.batch_size,
        preprocessing=json.loads((run_dir / "summary.json").read_text())["preprocessing"],
    )
    train_features, train_labels = _extract_features(model, train_loader, device)
    train_features = _l2_normalize(train_features)
    feature_dim = int(train_features.shape[1])
    logger.info("Features de train: %s", train_features.shape)

    pca_mean, pca_components, explained_actual = _fit_pca(train_features, args.explained_variance)
    pca_dim = pca_components.shape[0]
    logger.info(
        "PCA: %d -> %d dimensiones (%.2f%% varianza explicada, pedido %.2f%%)",
        feature_dim,
        pca_dim,
        explained_actual * 100,
        args.explained_variance * 100,
    )
    train_reduced = _apply_pca(train_features, pca_mean, pca_components)

    means = _compute_class_means(train_reduced, train_labels, num_classes)
    covariance = _regularize_covariance(
        _compute_pooled_covariance(train_reduced, train_labels, means)
    )
    inv_covariance = np.linalg.pinv(covariance)

    background_mean = train_reduced.mean(axis=0)
    background_covariance = _regularize_covariance(
        _compute_pooled_covariance(
            train_reduced, np.zeros(len(train_reduced), dtype=np.int64), background_mean[None, :]
        )
    )
    background_inv_covariance = np.linalg.pinv(background_covariance)

    logger.info("Extrayendo features de val (%s) para calibrar el umbral...", val_csv)
    val_loader, _ = build_test_loader(
        val_csv,
        config_path,
        class_to_idx,
        image_size,
        args.batch_size,
        preprocessing=json.loads((run_dir / "summary.json").read_text())["preprocessing"],
    )
    val_features, val_labels = _extract_features(model, val_loader, device)
    val_features = _l2_normalize(val_features)
    val_reduced = _apply_pca(val_features, pca_mean, pca_components)

    val_class_distances = _mahalanobis_distances(val_reduced, val_labels, means, inv_covariance)
    val_background_distances = _mahalanobis_to_mean(
        val_reduced, background_mean, background_inv_covariance
    )
    val_rmd_scores = val_class_distances - val_background_distances
    threshold = float(np.percentile(val_rmd_scores, args.percentile))

    if not np.isfinite(threshold):
        raise SystemExit(
            f"Umbral calibrado invalido ({threshold}) para '{model_name}'. "
            "Revisa que la covarianza pooled no este degenerada."
        )

    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    # Los arrays van en base64 (float32 binario), no como arrays JSON de texto: mas
    # compacto y evita parsear un array gigante como texto. Ver `_encode_float32_base64`.
    payload = {
        "schema_version": 4,
        "model": model_name,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "pca_dim": int(pca_dim),
        "explained_variance": explained_actual,
        "l2_normalized": True,
        "pca_mean_b64": _encode_float32_base64(pca_mean),
        "pca_components_b64": _encode_float32_base64(pca_components),
        "mean_per_class_b64": _encode_float32_base64(means),
        "inv_covariance_b64": _encode_float32_base64(inv_covariance),
        "background_mean_b64": _encode_float32_base64(background_mean),
        "background_inv_covariance_b64": _encode_float32_base64(background_inv_covariance),
        "threshold": round(threshold, 6),
        "percentile": args.percentile,
        "calibration_split": "val",
        "labels": [idx_to_class[i] for i in range(num_classes)],
    }
    output_path = export_dir / "ood_stats.json"
    training = json.loads((run_dir / "summary.json").read_text())
    payload.update(
        {
            "run_id": run_dir.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "preprocessing_id": contract_hash(training["preprocessing"]),
            "split_hashes": {"train": sha256_file(train_csv), "val": sha256_file(val_csv)},
            "feature_contract": {
                "kind": "pooled_pre_head",
                "feature_dim": feature_dim,
                "l2_normalized": True,
            },
            "ood_external_validation": "not_performed",
        }
    )
    atomic_json(output_path, payload)

    logger.info(
        "OK: %s -> %s (feature_dim=%d, pca_dim=%d, threshold=%.4f, percentile=%.1f)",
        model_name,
        output_path,
        feature_dim,
        pca_dim,
        threshold,
        args.percentile,
    )


def main() -> None:
    args = _parse_args()
    if args.checkpoint and len(args.models) > 1:
        raise SystemExit(
            "--checkpoint apunta a un unico archivo; no se puede usar con varios --models."
        )

    output_root = get_output_root()
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "main"

    for model_name in args.models:
        _compute_one(args, model_name, output_dir)


if __name__ == "__main__":
    main()
