# SHAP en el Pipeline Principal + CLI Unificado — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir KernelSHAP sobre superpíxeles al pipeline principal —panel comparado LIME/SHAP/Grad-CAM por imagen y perfil global por clase— y consolidar los scripts de explicabilidad en un CLI con subcomandos cuyos nombres describen el artefacto y no la técnica.

**Architecture:** Seis módulos nuevos y pequeños en `src/explainability/`, cada uno testeable sin GPU. La pieza que hace posible la comparación es `segmentation.py`: calcula el mapa de superpíxeles una sola vez y lo inyecta tanto en LIME (vía `segmentation_fn`) como en KernelSHAP, de modo que ambos producen vectores de atribución de la misma longitud y alineados. `scripts/pipeline/explain.py` reemplaza a `explain_lime.py` y `explain_report.py` con cinco subcomandos que comparten la resolución de runs y la carga de checkpoints.

**Tech Stack:** PyTorch 2.x, `shap>=0.44` (nueva), `lime`, scikit-image (SLIC), OpenCV (ExG+Otsu), scipy (Spearman), matplotlib, pandas, pytest.

**Spec:** `specs/2026-07-27-shap-pipeline-principal-design.md`

## Global Constraints

- **Python** `>=3.11`. Ruff `line-length = 100`, `target-version = "py311"`, lint `select = ["E","F","W","I"]`.
- **Comentarios de código:** exclusivamente DocBlocks estilo JSDoc/PHPDoc (`@param`, `@returns`, `@throws`). Prohibidos los comentarios narrativos dentro de la lógica. Si una función es autoexplicativa, no lleva comentario. Regla de `CLAUDE.md` global — no negociable.
- **Nunca hardcodear rutas ni constantes de dominio.** Dataset vía `get_dataset_root()`, artefactos vía `get_output_root()` (ambas en `src/config.py`). Clases, `target_size` y seed vienen de `config/dataset.yaml`.
- **Sin `sys.path.append`.** El paquete es editable; los imports `src.*` resuelven directo.
- **Punto único de entrada a imagen:** `load_and_normalize_image()` (`src/data/loader.py`).
- **LIME y Grad-CAM no cambian de comportamiento.** `render_visual_explanation` sin `segments` debe ejecutar exactamente el mismo camino que hoy. Protegido por los tests de la Tarea 6.
- **SHAP solo para el pipeline principal.** Los subcomandos `compare` y `global` no tienen variante `-baselines` en el Makefile.
- **Commits:** conventional commits, sin trailer `Co-Authored-By` ni atribución de modelo.
- **Idioma:** docstrings, mensajes de log y de commit en español, consistente con el código existente.
- **Etiquetas de segmento consecutivas desde 0.** Todo el plan asume que `values[segments]` indexa correctamente; `build_segments` lo garantiza y la Tarea 2 lo fija con un test.
- **Comando de test:** `venv/Scripts/python -m pytest` en Windows, `venv/bin/python -m pytest` en Linux/Mac. En los pasos se escribe `pytest` por brevedad.

---

### Task 1: Dependencia `shap`, bloque de configuración e infraestructura de tests

Esta tarea es el gate del riesgo R1 del spec: si `shap` no instala en el Python del venv (3.14), hay que saberlo antes de escribir nada que dependa de él.

**Files:**
- Modify: `pyproject.toml:22` (extra `xai`)
- Modify: `config/dataset.yaml:48` (bloque nuevo después de `gradcam:`)
- Create: `tests/explainability/__init__.py`, `tests/explainability/test_config.py`

**Interfaces:**
- Consumes: nada.
- Produces: el bloque `shap:` de `config/dataset.yaml` con las claves `segmentation`, `n_segments`, `compactness`, `nsamples`, `batch_size`, `background`, `images_per_class`, `global_sample_size`, `seed`. Todas las tareas siguientes lo leen.

- [ ] **Step 1: Añadir `shap` al extra `xai`**

En `pyproject.toml`, línea 22, reemplazar:

```toml
xai      = ["lime>=0.2", "scikit-image>=0.22", "matplotlib>=3.8", "shap>=0.44"]
```

- [ ] **Step 2: Instalar y verificar que importa**

Run: `pip install -e ".[xai]"`
Expected: instala `shap` y sus dependencias (`numba`, `slicer`, `cloudpickle`) sin conflictos.

Run: `python -c "import shap; print(shap.__version__)"`
Expected: imprime una versión `>= 0.44`.

**Si esto falla**, detener el plan y aplicar el fallback de R1: implementar `kernel_shap.py` con regresión lineal ponderada por el kernel de Shapley en vez de delegar en la librería. La interfaz pública del módulo (Tarea 5) no cambia, así que el resto del plan se sostiene igual; solo cambia el cuerpo de `explain_with_kernel_shap`.

- [ ] **Step 3: Añadir el bloque `shap:` a la configuración**

En `config/dataset.yaml`, después del bloque `gradcam:` (línea 48-49), añadir:

```yaml
shap:
  segmentation: slic
  n_segments: 50
  compactness: 10.0
  nsamples: 2048
  batch_size: 128
  background: black
  images_per_class: 5
  global_sample_size: 30
  seed: 42
```

- [ ] **Step 4: Escribir el test que fija el contrato de configuración**

`tests/explainability/__init__.py` queda vacío. `tests/explainability/test_config.py`:

```python
import yaml

from src.config import PROJECT_ROOT

_REQUIRED_KEYS = {
    "segmentation",
    "n_segments",
    "compactness",
    "nsamples",
    "batch_size",
    "background",
    "images_per_class",
    "global_sample_size",
    "seed",
}


def _load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "dataset.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_shap_block_has_every_required_key():
    shap_cfg = _load_config()["shap"]

    assert _REQUIRED_KEYS <= set(shap_cfg)


def test_shap_block_values_are_supported():
    shap_cfg = _load_config()["shap"]

    assert shap_cfg["segmentation"] in {"slic", "quickshift"}
    assert shap_cfg["background"] in {"black", "mean", "blur"}
    assert shap_cfg["nsamples"] > 2 * shap_cfg["n_segments"]


def test_shap_library_is_importable():
    import shap

    assert hasattr(shap, "KernelExplainer")
```

- [ ] **Step 5: Correr los tests**

Run: `pytest tests/explainability/test_config.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml config/dataset.yaml tests/explainability/
git commit -m "feat(xai): anade la dependencia shap y su bloque de configuracion"
```

---

### Task 2: `segmentation.py` — segmentación canónica compartida

Es la pieza que hace comparables a LIME y SHAP. LIME segmenta internamente con **quickshift**; para que ambas técnicas atribuyan sobre las mismas unidades hay que calcular el mapa afuera y pasárselo a los dos.

**Files:**
- Create: `src/explainability/segmentation.py`
- Test: `tests/explainability/test_segmentation.py`

**Interfaces:**
- Consumes: nada del plan.
- Produces: `build_segments(image_np: np.ndarray, algorithm: str = "slic", n_segments: int = 50, compactness: float = 10.0) -> np.ndarray`, que devuelve un mapa `HW` `int64` con etiquetas consecutivas desde 0.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/explainability/test_segmentation.py`:

```python
import numpy as np
import pytest

from src.explainability.segmentation import build_segments


def _synthetic_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_labels_are_consecutive_from_zero():
    segments = build_segments(_synthetic_image(), n_segments=16)

    unique = np.unique(segments)
    np.testing.assert_array_equal(unique, np.arange(unique.size))


def test_is_deterministic_across_calls():
    image = _synthetic_image()

    np.testing.assert_array_equal(
        build_segments(image, n_segments=16), build_segments(image, n_segments=16)
    )


def test_shape_matches_the_image():
    image = _synthetic_image()

    assert build_segments(image, n_segments=16).shape == image.shape[:2]


def test_quickshift_is_supported():
    segments = build_segments(_synthetic_image(), algorithm="quickshift")

    assert segments.min() == 0


def test_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="desconocido"):
        build_segments(_synthetic_image(), algorithm="felzenszwalb")
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_segmentation.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.explainability.segmentation'`.

- [ ] **Step 3: Implementar el módulo**

`src/explainability/segmentation.py`:

```python
"""Segmentacion canonica compartida por LIME y KernelSHAP en el panel comparado."""

import numpy as np
from skimage.segmentation import quickshift, slic

_QUICKSHIFT_KERNEL_SIZE = 4
_QUICKSHIFT_MAX_DIST = 200
_QUICKSHIFT_RATIO = 0.2


def build_segments(
    image_np: np.ndarray,
    algorithm: str = "slic",
    n_segments: int = 50,
    compactness: float = 10.0,
) -> np.ndarray:
    """
    Calcula el mapa de superpixeles que comparten LIME y KernelSHAP.

    Ninguno de los dos algoritmos usa aleatoriedad, asi que el mapa es reproducible sin
    semilla. Las etiquetas se devuelven consecutivas desde 0 para poder indexar el vector
    de atribuciones directamente con `values[segments]`.

    Los parametros de quickshift replican los que `lime_image` usa por defecto, de modo
    que esa opcion reproduce las regiones del panel `visual`.

    @param {np.ndarray} image_np Imagen HWC uint8 ya reescalada a target_size.
    @param {str} algorithm "slic" o "quickshift".
    @param {int} n_segments Numero objetivo de superpixeles de SLIC; ignorado por quickshift.
    @param {float} compactness Peso del termino espacial de SLIC; ignorado por quickshift.
    @returns {np.ndarray} Mapa HW int64 con etiquetas consecutivas desde 0.
    @throws {ValueError} Si el algoritmo no es uno de los soportados.
    """
    if algorithm == "slic":
        segments = slic(
            image_np, n_segments=n_segments, compactness=compactness, start_label=0
        )
    elif algorithm == "quickshift":
        segments = quickshift(
            image_np,
            kernel_size=_QUICKSHIFT_KERNEL_SIZE,
            max_dist=_QUICKSHIFT_MAX_DIST,
            ratio=_QUICKSHIFT_RATIO,
        )
    else:
        raise ValueError(
            f"Algoritmo de segmentacion desconocido: {algorithm!r}. Usa 'slic' o 'quickshift'."
        )

    _, relabeled = np.unique(segments, return_inverse=True)
    return relabeled.reshape(segments.shape).astype(np.int64)
```

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/explainability/test_segmentation.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Lint y commit**

```bash
ruff check src/explainability/segmentation.py tests/explainability/test_segmentation.py
ruff format src/explainability/segmentation.py tests/explainability/test_segmentation.py
git add src/explainability/segmentation.py tests/explainability/test_segmentation.py
git commit -m "feat(xai): anade la segmentacion canonica compartida por LIME y SHAP"
```

---

### Task 3: `leaf_mask.py` — máscara de vegetación y su sanidad

Sostiene el ratio hoja/fondo del perfil global. La heurística puede fallar en hojas cloróticas (`nitrogen_deficiency`), así que el módulo expone la cobertura y el criterio de descarte para que el consumidor pueda declarar el número no confiable en vez de publicarlo a ciegas.

**Files:**
- Create: `src/explainability/leaf_mask.py`
- Test: `tests/explainability/test_leaf_mask.py`

**Interfaces:**
- Consumes: nada del plan.
- Produces: `leaf_mask(image_np: np.ndarray) -> np.ndarray` (booleana HW), `mask_coverage(mask: np.ndarray) -> float`, `is_coverage_degenerate(coverage: float, low: float = 0.05, high: float = 0.95) -> bool`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/explainability/test_leaf_mask.py`:

```python
import numpy as np

from src.explainability.leaf_mask import is_coverage_degenerate, leaf_mask, mask_coverage


def _half_leaf_image() -> np.ndarray:
    """Mitad izquierda verde (hoja), mitad derecha gris parduzco (suelo)."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (40, 160, 40)
    image[:, 16:] = (140, 120, 100)
    return image


def test_detects_the_green_half():
    mask = leaf_mask(_half_leaf_image())

    assert mask[:, :16].all()
    assert not mask[:, 16:].any()


def test_coverage_matches_the_green_fraction():
    coverage = mask_coverage(leaf_mask(_half_leaf_image()))

    assert coverage == 0.5


def test_uniform_image_yields_degenerate_coverage():
    uniform = np.full((32, 32, 3), 128, dtype=np.uint8)

    coverage = mask_coverage(leaf_mask(uniform))

    assert is_coverage_degenerate(coverage)


def test_healthy_coverage_is_not_degenerate():
    assert not is_coverage_degenerate(0.5)
    assert is_coverage_degenerate(0.01)
    assert is_coverage_degenerate(0.99)
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_leaf_mask.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.explainability.leaf_mask'`.

- [ ] **Step 3: Implementar el módulo**

`src/explainability/leaf_mask.py`:

```python
"""Mascara de vegetacion para separar atribucion sobre hoja de atribucion sobre fondo."""

import cv2
import numpy as np

_COVERAGE_LOW = 0.05
_COVERAGE_HIGH = 0.95


def leaf_mask(image_np: np.ndarray) -> np.ndarray:
    """
    Segmenta hoja contra fondo con indice de exceso de verde (ExG) y umbral de Otsu.

    Es una heuristica: sobre hojas cloroticas (deficiencia de nitrogeno) puede degradarse.
    El consumidor debe validar el resultado con `mask_coverage` e `is_coverage_degenerate`
    antes de derivar metricas de el.

    @param {np.ndarray} image_np Imagen HWC uint8.
    @returns {np.ndarray} Mascara booleana HW; True donde hay vegetacion.
    """
    channels = image_np.astype(np.float32)
    excess_green = 2.0 * channels[..., 1] - channels[..., 0] - channels[..., 2]
    normalized = cv2.normalize(excess_green, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)


def mask_coverage(mask: np.ndarray) -> float:
    """
    Fraccion de pixeles clasificados como vegetacion.

    @param {np.ndarray} mask Mascara booleana.
    @returns {float} Cobertura en [0, 1].
    """
    return float(mask.mean())


def is_coverage_degenerate(
    coverage: float, low: float = _COVERAGE_LOW, high: float = _COVERAGE_HIGH
) -> bool:
    """
    Indica si la cobertura delata una segmentacion inservible (casi todo o casi nada).

    @param {float} coverage Cobertura devuelta por `mask_coverage`.
    @param {float} low Cota inferior aceptable.
    @param {float} high Cota superior aceptable.
    @returns {bool} True si la mascara no es utilizable para el ratio hoja/fondo.
    """
    return coverage < low or coverage > high
```

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/explainability/test_leaf_mask.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Lint y commit**

```bash
ruff check src/explainability/leaf_mask.py tests/explainability/test_leaf_mask.py
ruff format src/explainability/leaf_mask.py tests/explainability/test_leaf_mask.py
git add src/explainability/leaf_mask.py tests/explainability/test_leaf_mask.py
git commit -m "feat(xai): anade la mascara de vegetacion ExG+Otsu con control de cobertura"
```

---

### Task 4: `agreement.py` — acuerdo entre LIME y SHAP

Tres métricas que responden preguntas distintas: dónde miran, en qué orden de importancia, y con qué signo. Pueden divergir entre sí, y esa divergencia es el hallazgo.

**Files:**
- Create: `src/explainability/agreement.py`
- Test: `tests/explainability/test_agreement.py`

**Interfaces:**
- Consumes: `mask_iou` de `src/explainability/stability.py` (ya existe).
- Produces: `densify_weights(local_exp: list[tuple[int, float]], n_segments: int) -> np.ndarray`, `top_positive_mask(values: np.ndarray, top_k: int) -> np.ndarray`, `attribution_agreement(lime_weights: np.ndarray, shap_values: np.ndarray, top_k: int) -> dict[str, float]` con claves `iou_topk`, `spearman`, `sign_agreement`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/explainability/test_agreement.py`:

```python
import numpy as np
import pytest

from src.explainability.agreement import (
    attribution_agreement,
    densify_weights,
    top_positive_mask,
)


def test_densify_places_weights_at_their_segment_index():
    dense = densify_weights([(2, 0.5), (0, -0.25)], n_segments=4)

    np.testing.assert_allclose(dense, [-0.25, 0.0, 0.5, 0.0])


def test_top_positive_mask_ignores_negative_values():
    mask = top_positive_mask(np.array([0.9, -5.0, 0.1, 0.4]), top_k=2)

    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_top_positive_mask_with_no_positive_values_is_empty():
    mask = top_positive_mask(np.array([-1.0, -2.0]), top_k=2)

    assert not mask.any()


def test_identical_vectors_agree_completely():
    values = np.array([0.5, -0.2, 0.9, 0.1])

    agreement = attribution_agreement(values, values, top_k=2)

    assert agreement["iou_topk"] == 1.0
    assert agreement["spearman"] == pytest.approx(1.0)
    assert agreement["sign_agreement"] == 1.0


def test_opposite_vectors_disagree_completely():
    values = np.array([0.5, -0.2, 0.9, 0.1])

    agreement = attribution_agreement(values, -values, top_k=2)

    assert agreement["spearman"] == pytest.approx(-1.0)
    assert agreement["sign_agreement"] == 0.0


def test_constant_vector_yields_zero_correlation():
    agreement = attribution_agreement(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]), top_k=2)

    assert agreement["spearman"] == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="longitud distinta"):
        attribution_agreement(np.zeros(3), np.zeros(4), top_k=2)
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_agreement.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.explainability.agreement'`.

- [ ] **Step 3: Implementar el módulo**

`src/explainability/agreement.py`:

```python
"""Acuerdo entre dos vectores de atribucion calculados sobre los mismos superpixeles."""

import numpy as np
from scipy.stats import spearmanr

from src.explainability.stability import mask_iou


def densify_weights(local_exp: list[tuple[int, float]], n_segments: int) -> np.ndarray:
    """
    Convierte la lista dispersa (segmento, peso) de LIME en un vector denso.

    @param {list[tuple[int, float]]} local_exp Pares (id de segmento, peso).
    @param {int} n_segments Cantidad total de superpixeles.
    @returns {np.ndarray} Vector de longitud n_segments; 0.0 en los segmentos ausentes.
    """
    dense = np.zeros(n_segments, dtype=np.float64)
    for segment_id, weight in local_exp:
        dense[int(segment_id)] = float(weight)
    return dense


def top_positive_mask(values: np.ndarray, top_k: int) -> np.ndarray:
    """
    Marca los `top_k` segmentos con mayor atribucion positiva.

    @param {np.ndarray} values Vector de atribuciones por segmento.
    @param {int} top_k Cantidad de segmentos a marcar.
    @returns {np.ndarray} Mascara booleana de la misma longitud que `values`.
    """
    mask = np.zeros(values.shape, dtype=bool)
    positive = np.flatnonzero(values > 0)
    if positive.size == 0:
        return mask
    mask[positive[np.argsort(-values[positive])][:top_k]] = True
    return mask


def attribution_agreement(
    lime_weights: np.ndarray, shap_values: np.ndarray, top_k: int
) -> dict[str, float]:
    """
    Compara dos vectores de atribucion definidos sobre los mismos superpixeles.

    `iou_topk` responde si coinciden en que mirar, `spearman` si coinciden en el orden de
    importancia, y `sign_agreement` si coinciden en la direccion del empuje.

    @param {np.ndarray} lime_weights Pesos de la regresion local de LIME por segmento.
    @param {np.ndarray} shap_values Valores de Shapley por segmento.
    @param {int} top_k Segmentos positivos a considerar en el IoU.
    @returns {dict[str, float]} Claves iou_topk, spearman y sign_agreement.
    @throws {ValueError} Si los vectores no tienen la misma longitud.
    """
    if lime_weights.shape != shap_values.shape:
        raise ValueError(
            f"Vectores de longitud distinta: {lime_weights.shape} vs {shap_values.shape}"
        )

    if lime_weights.std() == 0 or shap_values.std() == 0:
        correlation = 0.0
    else:
        correlation = float(spearmanr(lime_weights, shap_values)[0])
        if np.isnan(correlation):
            correlation = 0.0

    return {
        "iou_topk": mask_iou(
            top_positive_mask(lime_weights, top_k), top_positive_mask(shap_values, top_k)
        ),
        "spearman": correlation,
        "sign_agreement": float(np.mean(np.sign(lime_weights) == np.sign(shap_values))),
    }
```

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/explainability/test_agreement.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Lint y commit**

```bash
ruff check src/explainability/agreement.py tests/explainability/test_agreement.py
ruff format src/explainability/agreement.py tests/explainability/test_agreement.py
git add src/explainability/agreement.py tests/explainability/test_agreement.py
git commit -m "feat(xai): anade las metricas de acuerdo entre atribuciones LIME y SHAP"
```

---

### Task 5: `kernel_shap.py` — KernelSHAP sobre superpíxeles

El núcleo del spec. El test de exactitud contra los valores de Shapley enumerados por fuerza bruta es lo que valida que la integración con la librería es correcta — es el gate real de la decisión D3.

**Files:**
- Create: `src/explainability/kernel_shap.py`
- Test: `tests/explainability/test_kernel_shap.py`

**Interfaces:**
- Consumes: nada del plan; `predict_fn` lo provee `build_predict_fn` de `visual_report.py` (ya existe y es público).
- Produces: `ShapExplanation` (dataclass congelada con `values: np.ndarray`, `expected_value: float`, `target_idx: int`, `n_evals: int`), `build_background(image_np: np.ndarray, background: str) -> np.ndarray`, y `explain_with_kernel_shap(image_np, segments, predict_fn, target_idx, nsamples=2048, batch_size=128, background="black", seed=42) -> ShapExplanation`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/explainability/test_kernel_shap.py`:

```python
import itertools
from math import factorial

import numpy as np
import pytest

from src.explainability.kernel_shap import build_background, explain_with_kernel_shap

_SEGMENT_VALUES = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.05}
_INTERACTION_BONUS = 0.25


def _four_block_segments() -> np.ndarray:
    segments = np.zeros((8, 8), dtype=np.int64)
    segments[:4, 4:] = 1
    segments[4:, :4] = 2
    segments[4:, 4:] = 3
    return segments


def _four_block_image() -> np.ndarray:
    segments = _four_block_segments()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for segment_id in range(4):
        image[segments == segment_id] = (segment_id + 1) * 50
    return image


def _set_value(visible: frozenset) -> float:
    """Funcion de coalicion con una interaccion, para que los valores no sean triviales."""
    total = sum(_SEGMENT_VALUES[segment_id] for segment_id in visible)
    if {0, 1} <= visible:
        total += _INTERACTION_BONUS
    return total


def _predict_fn(batch: np.ndarray) -> np.ndarray:
    """Deduce que superpixeles quedaron visibles por el color de cada bloque."""
    segments = _four_block_segments()
    scores = np.array(
        [
            _set_value(
                frozenset(
                    segment_id
                    for segment_id in range(4)
                    if image[segments == segment_id].max() > 0
                )
            )
            for image in batch
        ]
    )
    return np.stack([scores, 1.0 - scores], axis=1)


def _exact_shapley() -> np.ndarray:
    players = list(range(4))
    phi = np.zeros(4)
    for player in players:
        others = [other for other in players if other != player]
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                weight = factorial(size) * factorial(3 - size) / factorial(4)
                phi[player] += weight * (
                    _set_value(frozenset(subset) | {player}) - _set_value(frozenset(subset))
                )
    return phi


def _explain(**overrides):
    kwargs = {
        "image_np": _four_block_image(),
        "segments": _four_block_segments(),
        "predict_fn": _predict_fn,
        "target_idx": 0,
        "nsamples": 64,
        "batch_size": 8,
    }
    kwargs.update(overrides)
    return explain_with_kernel_shap(**kwargs)


def test_matches_exact_shapley_values():
    explanation = _explain()

    np.testing.assert_allclose(explanation.values, _exact_shapley(), atol=1e-6)


def test_is_additive():
    explanation = _explain()

    total = explanation.values.sum() + explanation.expected_value

    assert total == pytest.approx(_set_value(frozenset(range(4))), abs=1e-6)


def test_expected_value_is_the_fully_masked_prediction():
    explanation = _explain()

    assert explanation.expected_value == pytest.approx(_set_value(frozenset()), abs=1e-9)


def test_is_deterministic_when_coalitions_are_sampled():
    first = _explain(nsamples=10)
    second = _explain(nsamples=10)

    np.testing.assert_array_equal(first.values, second.values)


def test_does_not_leak_the_global_random_state():
    np.random.seed(1234)
    expected = np.random.random()

    np.random.seed(1234)
    _explain(nsamples=10)
    actual = np.random.random()

    assert actual == expected


def test_black_background_is_all_zeros():
    background = build_background(_four_block_image(), "black")

    assert not background.any()


def test_mean_background_is_constant_per_channel():
    background = build_background(_four_block_image(), "mean")

    assert len(np.unique(background.reshape(-1, 3), axis=0)) == 1


def test_rejects_unknown_background():
    with pytest.raises(ValueError, match="desconocida"):
        build_background(_four_block_image(), "inpaint")
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_kernel_shap.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.explainability.kernel_shap'`.

- [ ] **Step 3: Implementar el módulo**

`src/explainability/kernel_shap.py`:

```python
"""KernelSHAP sobre superpixeles para el panel comparado del pipeline principal."""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
import shap

_BLUR_SIGMA = 15


@dataclass(frozen=True)
class ShapExplanation:
    """Valores de Shapley por superpixel para una imagen y una clase."""

    values: np.ndarray
    expected_value: float
    target_idx: int
    n_evals: int


def build_background(image_np: np.ndarray, background: str) -> np.ndarray:
    """
    Construye la imagen de referencia que reemplaza a los superpixeles ausentes.

    @param {np.ndarray} image_np Imagen HWC uint8.
    @param {str} background "black" (paridad con el hide_color=0 de LIME), "mean" o "blur".
    @returns {np.ndarray} Imagen HWC uint8 del mismo tamano.
    @throws {ValueError} Si la linea base no es una de las soportadas.
    """
    if background == "black":
        return np.zeros_like(image_np)
    if background == "mean":
        channel_mean = image_np.reshape(-1, image_np.shape[-1]).mean(axis=0)
        return np.full_like(image_np, channel_mean.astype(np.uint8))
    if background == "blur":
        return cv2.GaussianBlur(image_np, (0, 0), sigmaX=_BLUR_SIGMA)
    raise ValueError(f"Linea base desconocida: {background!r}. Usa 'black', 'mean' o 'blur'.")


def _build_coalition_fn(
    image_np: np.ndarray,
    segments: np.ndarray,
    background_np: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_idx: int,
    batch_size: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Arma la funcion de coalicion que evalua KernelSHAP: z[i]=1 deja visible el superpixel i.

    @param {np.ndarray} image_np Imagen HWC uint8.
    @param {np.ndarray} segments Mapa de superpixeles con etiquetas desde 0.
    @param {np.ndarray} background_np Imagen de referencia para los superpixeles ausentes.
    @param {Callable} predict_fn Mapea un batch HWC uint8 a probabilidades por clase.
    @param {int} target_idx Indice de la clase explicada.
    @param {int} batch_size Imagenes por forward pass.
    @returns {Callable} Funcion que mapea una matriz (n, k) de coaliciones a (n,) scores.
    """
    segment_masks = np.stack(
        [segments == segment_id for segment_id in range(int(segments.max()) + 1)]
    )

    def coalition_fn(coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.atleast_2d(np.asarray(coalitions))
        scores = np.empty(len(coalitions), dtype=np.float64)
        for start in range(0, len(coalitions), batch_size):
            chunk = coalitions[start : start + batch_size]
            batch = np.empty((len(chunk), *image_np.shape), dtype=np.uint8)
            for position, row in enumerate(chunk):
                visible = segment_masks[row > 0.5].any(axis=0)
                batch[position] = np.where(visible[..., None], image_np, background_np)
            scores[start : start + batch_size] = predict_fn(batch)[:, target_idx]
        return scores

    return coalition_fn


def explain_with_kernel_shap(
    image_np: np.ndarray,
    segments: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_idx: int,
    nsamples: int = 2048,
    batch_size: int = 128,
    background: str = "black",
    seed: int = 42,
) -> ShapExplanation:
    """
    Calcula los valores de Shapley por superpixel con KernelSHAP.

    Con k superpixeles KernelSHAP enumera todas las coaliciones si `nsamples >= 2**k` y
    las muestrea si no. El muestreo consume el generador global de NumPy, asi que se fija
    la semilla y se restaura el estado previo: el determinismo entre corridas es lo que
    distingue a SHAP de LIME y no debe depender de como venga sembrado el proceso, ni
    filtrar la resiembra al resto del pipeline.

    `l1_reg` se fija en `num_features(k)` para desactivar la seleccion de variables: con
    regularizacion activa algunos superpixeles reciben exactamente cero por decision del
    lasso y no por su contribucion real, lo que rompe la aditividad y la comparacion con
    LIME.

    @param {np.ndarray} image_np Imagen HWC uint8 ya reescalada a target_size.
    @param {np.ndarray} segments Mapa de superpixeles con etiquetas consecutivas desde 0.
    @param {Callable} predict_fn Mapea un batch HWC uint8 a probabilidades por clase.
    @param {int} target_idx Indice de la clase a explicar.
    @param {int} nsamples Evaluaciones del modelo por imagen.
    @param {int} batch_size Imagenes por forward pass.
    @param {str} background Linea base de enmascarado.
    @param {int} seed Semilla del muestreo de coaliciones.
    @returns {ShapExplanation} Valores por segmento, valor esperado y metadatos.
    """
    n_segments = int(segments.max()) + 1
    coalition_fn = _build_coalition_fn(
        image_np=image_np,
        segments=segments,
        background_np=build_background(image_np, background),
        predict_fn=predict_fn,
        target_idx=target_idx,
        batch_size=batch_size,
    )

    previous_state = np.random.get_state()
    np.random.seed(seed)
    try:
        explainer = shap.KernelExplainer(coalition_fn, np.zeros((1, n_segments)))
        raw_values = explainer.shap_values(
            np.ones((1, n_segments)),
            nsamples=nsamples,
            l1_reg=f"num_features({n_segments})",
            silent=True,
        )
    finally:
        np.random.set_state(previous_state)

    return ShapExplanation(
        values=np.asarray(raw_values, dtype=np.float64).reshape(n_segments),
        expected_value=float(np.asarray(explainer.expected_value).reshape(-1)[0]),
        target_idx=int(target_idx),
        n_evals=int(nsamples),
    )
```

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/explainability/test_kernel_shap.py -v`
Expected: 8 PASS.

Si `test_matches_exact_shapley_values` falla por diferencias en el manejo de `l1_reg` de la versión instalada, probar con `l1_reg=0` antes de dar por inválida la integración; el resto del módulo no cambia.

- [ ] **Step 5: Lint y commit**

```bash
ruff check src/explainability/kernel_shap.py tests/explainability/test_kernel_shap.py
ruff format src/explainability/kernel_shap.py tests/explainability/test_kernel_shap.py
git add src/explainability/kernel_shap.py tests/explainability/test_kernel_shap.py
git commit -m "feat(xai): implementa KernelSHAP sobre superpixeles con linea base configurable"
```

---

### Task 6: `visual_report.py` — segmentación inyectable y helpers públicos

Dos cambios quirúrgicos sobre un archivo que produce resultados publicados: un parámetro opcional que, si no se pasa, deja el camino de ejecución idéntico; y la promoción a públicos de cuatro helpers que `compare_report.py` necesita reusar. Más el renombre del directorio de salida.

**Files:**
- Modify: `src/explainability/visual_report.py` (renombre de 4 helpers, parámetro `segments`, directorio de salida)
- Test: `tests/explainability/test_visual_report.py`

**Interfaces:**
- Consumes: nada del plan.
- Produces: `prepare_lime_image`, `build_validation_transform`, `build_positive_region_panel`, `build_importance_heatmap` (antes privados, misma firma); `render_visual_explanation(..., segments: np.ndarray | None = None)`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/explainability/test_visual_report.py`:

```python
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from lime import lime_image
from PIL import Image

from src.explainability.visual_report import explain_model_visual, render_visual_explanation

_TARGET_SIZE = (8, 8)
_IDX_TO_CLASS = {0: "healthy", 1: "common_rust"}


class _FakeExplanation:
    """Sustituto de la explicacion de LIME, para no pagar el muestreo en los tests."""

    def __init__(self, segments: np.ndarray):
        self.segments = segments
        self.top_labels = [0]
        self.local_exp = {0: [(0, 1.0), (1, -0.5)]}

    def get_image_and_mask(self, label, positive_only, num_features, hide_rest):
        return None, (self.segments == 0).astype(int)


@pytest.fixture
def captured_kwargs(monkeypatch) -> dict:
    captured: dict = {}

    def fake_explain_instance(self, image, classifier_fn, **kwargs):
        captured.update(kwargs)
        segments = np.zeros(image.shape[:2], dtype=np.int64)
        segments[image.shape[0] // 2 :, :] = 1
        return _FakeExplanation(segments)

    monkeypatch.setattr(
        lime_image.LimeImageExplainer, "explain_instance", fake_explain_instance
    )
    return captured


def _dummy_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2)).eval()


def _dummy_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8))


def test_without_segments_lime_keeps_its_own_segmentation(captured_kwargs, tmp_path):
    render_visual_explanation(
        image=_dummy_image(),
        model=_dummy_model(),
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=tmp_path / "panel.png",
        num_samples=4,
        num_features=2,
    )

    assert "segmentation_fn" not in captured_kwargs


def test_segments_are_forwarded_to_lime(captured_kwargs, tmp_path):
    segments = np.zeros(_TARGET_SIZE, dtype=np.int64)
    segments[4:, :] = 1

    render_visual_explanation(
        image=_dummy_image(),
        model=_dummy_model(),
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=tmp_path / "panel.png",
        num_samples=4,
        num_features=2,
        segments=segments,
    )

    forwarded = captured_kwargs["segmentation_fn"](np.zeros((8, 8, 3), dtype=np.uint8))
    np.testing.assert_array_equal(forwarded, segments)


def test_writes_png_and_sidecars(captured_kwargs, tmp_path):
    output_path = tmp_path / "panel.png"

    render_visual_explanation(
        image=_dummy_image(),
        model=_dummy_model(),
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=output_path,
        num_samples=4,
        num_features=2,
    )

    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
    assert output_path.with_suffix(".npy").exists()


def test_explain_model_visual_writes_under_explain_visual(
    captured_kwargs, tmp_path, fake_image_root, tmp_splits_dir
):
    run_dir = tmp_path / "run"

    explain_model_visual(
        model=_dummy_model(),
        model_name="dummy",
        test_df=pd.read_csv(tmp_splits_dir / "test.csv"),
        dataset_root=fake_image_root,
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_dir=run_dir,
        images_per_class=1,
        num_features=2,
        num_samples=4,
        seed=42,
        device=torch.device("cpu"),
        enable_gradcam=False,
    )

    assert (run_dir / "explain_visual").is_dir()
    assert not (run_dir / "lime_visual").exists()
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_visual_report.py -v`
Expected: FAIL — `test_segments_are_forwarded_to_lime` con `TypeError: render_visual_explanation() got an unexpected keyword argument 'segments'`, y `test_explain_model_visual_writes_under_explain_visual` con `assert (run_dir / "explain_visual").is_dir()`.

- [ ] **Step 3: Promover los cuatro helpers a públicos**

En `src/explainability/visual_report.py`, renombrar quitando el guion bajo inicial y actualizar cada uso interno:

| Antes | Después | Usos internos a actualizar |
|---|---|---|
| `_build_validation_transform` (línea 81) | `build_validation_transform` | líneas 119, 227 |
| `_prepare_lime_image` (línea 93) | `prepare_lime_image` | línea 192 |
| `_build_positive_region_panel` (línea 132) | `build_positive_region_panel` | línea 217 |
| `_build_importance_heatmap` (línea 145) | `build_importance_heatmap` | línea 220 |

Los cuerpos y docstrings no cambian.

- [ ] **Step 4: Añadir el parámetro `segments`**

En la firma de `render_visual_explanation` (línea 166-177), añadir el parámetro al final:

```python
    model_name: str | None = None,
    segments: np.ndarray | None = None,
) -> dict:
```

Ampliar el docstring con:

```
    `segments` es opcional (default None): con None, LIME segmenta internamente con
    quickshift, que es su comportamiento historico y el de los reportes ya publicados.
    Con un mapa explicito, LIME atribuye sobre esas mismas regiones, que es lo que
    permite comparar sus pesos con los valores de Shapley segmento a segmento.

    @param {np.ndarray|None} segments Mapa de superpixeles a imponer, o None.
```

Y reemplazar la llamada a `explain_instance` (línea 198-205) por:

```python
    explain_kwargs = {}
    if segments is not None:
        explain_kwargs["segmentation_fn"] = lambda _image: segments

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=len(idx_to_class),
        hide_color=0,
        num_samples=num_samples,
        random_seed=seed,
        **explain_kwargs,
    )
```

- [ ] **Step 5: Renombrar el directorio de salida**

En `explain_model_visual` (línea 380), reemplazar:

```python
    lime_dir = output_dir / "explain_visual"
```

Y en su docstring (línea 375) y en el log final (línea 412), reemplazar `lime_visual` por `explain_visual` y "Reportes visuales LIME" por "Paneles visuales".

- [ ] **Step 6: Correr los tests**

Run: `pytest tests/explainability/test_visual_report.py -v`
Expected: 4 PASS.

- [ ] **Step 7: Correr la suite completa para detectar roturas**

Run: `pytest -v`
Expected: todo PASS. Los tests de `tests/training/` no tocan `visual_report`, pero el renombre de helpers podría haber roto un import olvidado.

- [ ] **Step 8: Lint y commit**

```bash
ruff check src/explainability/visual_report.py tests/explainability/test_visual_report.py
ruff format src/explainability/visual_report.py tests/explainability/test_visual_report.py
git add src/explainability/visual_report.py tests/explainability/test_visual_report.py
git commit -m "refactor(xai): permite inyectar la segmentacion en LIME y expone los helpers de panel"
```

---

### Task 7: `compare_report.py` — panel comparado LIME | SHAP | Grad-CAM

Corre LIME y SHAP sobre la **misma** segmentación en una sola pasada, de modo que las métricas de acuerdo salen sin costo adicional.

**Files:**
- Create: `src/explainability/compare_report.py`
- Test: `tests/explainability/test_compare_report.py`

**Interfaces:**
- Consumes: `build_segments` (T2), `explain_with_kernel_shap` (T5), `attribution_agreement` y `densify_weights` (T4), `prepare_lime_image` / `build_predict_fn` / `build_validation_transform` / `build_importance_heatmap` (T6), `GradCAM` / `get_target_layer` / `build_gradcam_overlay` (ya existen), `explanation_dispersion` (ya existe).
- Produces: `render_comparison(image, model, model_name, idx_to_class, target_size, output_path, lime_cfg, shap_cfg, device) -> dict` con claves `predicted_label`, `predicted_prob`, `dispersion` y `agreement`.

- [ ] **Step 1: Escribir el test que falla**

`tests/explainability/test_compare_report.py`:

```python
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.explainability.compare_report import render_comparison

_TARGET_SIZE = (16, 16)
_IDX_TO_CLASS = {0: "healthy", 1: "common_rust"}
_LIME_CFG = {"num_samples": 20, "num_features": 3, "seed": 42}
_SHAP_CFG = {
    "segmentation": "slic",
    "n_segments": 4,
    "compactness": 10.0,
    "nsamples": 32,
    "batch_size": 8,
    "background": "black",
    "seed": 42,
}


def _dummy_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 2)).eval()


def _dummy_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8))


def _render(tmp_path):
    return render_comparison(
        image=_dummy_image(),
        model=_dummy_model(),
        model_name=None,
        idx_to_class=_IDX_TO_CLASS,
        target_size=_TARGET_SIZE,
        output_path=tmp_path / "compare.png",
        lime_cfg=_LIME_CFG,
        shap_cfg=_SHAP_CFG,
        device=torch.device("cpu"),
    )


def test_writes_png_and_sidecars(tmp_path):
    _render(tmp_path)

    assert (tmp_path / "compare.png").exists()
    assert (tmp_path / "compare.json").exists()
    assert (tmp_path / "compare.npy").exists()


def test_result_reports_prediction_and_agreement(tmp_path):
    result = _render(tmp_path)

    assert result["predicted_label"] in _IDX_TO_CLASS.values()
    assert 0.0 <= result["predicted_prob"] <= 1.0
    assert set(result["agreement"]) == {"iou_topk", "spearman", "sign_agreement"}


def test_sidecar_holds_both_attribution_vectors(tmp_path):
    _render(tmp_path)

    metadata = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))
    segments = np.load(tmp_path / "compare.npy")

    n_segments = int(segments.max()) + 1
    assert len(metadata["lime_weights"]) == n_segments
    assert len(metadata["shap_values"]) == n_segments


def test_lime_and_shap_share_the_segmentation(tmp_path):
    _render(tmp_path)

    metadata = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))

    assert len(metadata["lime_weights"]) == len(metadata["shap_values"])
    assert metadata["n_segments"] == len(metadata["shap_values"])
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_compare_report.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.explainability.compare_report'`.

- [ ] **Step 3: Implementar el módulo**

`src/explainability/compare_report.py`:

```python
"""Panel comparado LIME | SHAP | Grad-CAM sobre una segmentacion compartida."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lime import lime_image
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from PIL import Image

from src.explainability.agreement import attribution_agreement, densify_weights
from src.explainability.gradcam import GradCAM, build_gradcam_overlay, get_target_layer
from src.explainability.kernel_shap import explain_with_kernel_shap
from src.explainability.segmentation import build_segments
from src.explainability.visual_report import (
    build_importance_heatmap,
    build_predict_fn,
    build_validation_transform,
    explanation_dispersion,
    prepare_lime_image,
)

logger = logging.getLogger(__name__)

_TITLE_COLOR = "#2C3E50"


def render_comparison(
    image: Image.Image,
    model: nn.Module,
    model_name: str | None,
    idx_to_class: dict[int, str],
    target_size: tuple[int, int],
    output_path: Path,
    lime_cfg: dict,
    shap_cfg: dict,
    device: torch.device,
) -> dict:
    """
    Genera el panel comparado de una imagen y persiste sus artefactos numericos.

    LIME y KernelSHAP se ejecutan sobre el mismo mapa de superpixeles, calculado una sola
    vez con `build_segments`, y sobre la misma clase (el argmax del modelo). Esa doble
    identidad es lo que hace que las metricas de acuerdo comparen lo mismo.

    Las regiones LIME de este panel no coinciden con las del panel `visual`: alli LIME
    segmenta con quickshift y aqui recibe la segmentacion impuesta (por defecto SLIC).

    @param {Image.Image} image Imagen original, sin reescalar.
    @param {nn.Module} model Modelo en modo eval.
    @param {str|None} model_name Nombre registrado para Grad-CAM, o None para omitirlo.
    @param {dict[int, str]} idx_to_class Mapeo indice->clase del head entrenado.
    @param {tuple[int, int]} target_size Tamano de entrada del checkpoint.
    @param {Path} output_path Ruta del PNG; los sidecars usan el mismo stem.
    @param {dict} lime_cfg Bloque `lime` de config/dataset.yaml.
    @param {dict} shap_cfg Bloque `shap` de config/dataset.yaml.
    @param {torch.device} device Dispositivo de computo.
    @returns {dict} Prediccion, confianza, dispersion de SHAP y metricas de acuerdo.
    """
    model.eval()
    image_np = prepare_lime_image(image, target_size)
    image_rgb01 = image_np.astype(float) / 255.0
    predict_fn = build_predict_fn(model, device, target_size)

    probabilities = predict_fn(image_np[np.newaxis, ...])[0]
    target_idx = int(np.argmax(probabilities))

    segments = build_segments(
        image_np,
        algorithm=shap_cfg["segmentation"],
        n_segments=shap_cfg["n_segments"],
        compactness=shap_cfg["compactness"],
    )
    n_segments = int(segments.max()) + 1

    explainer = lime_image.LimeImageExplainer(random_state=lime_cfg["seed"])
    lime_explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=len(idx_to_class),
        hide_color=0,
        num_samples=lime_cfg["num_samples"],
        random_seed=lime_cfg["seed"],
        segmentation_fn=lambda _image: segments,
    )
    lime_weights = densify_weights(lime_explanation.local_exp[target_idx], n_segments)

    shap_explanation = explain_with_kernel_shap(
        image_np=image_np,
        segments=segments,
        predict_fn=predict_fn,
        target_idx=target_idx,
        nsamples=shap_cfg["nsamples"],
        batch_size=shap_cfg["batch_size"],
        background=shap_cfg["background"],
        seed=shap_cfg["seed"],
    )

    agreement = attribution_agreement(
        lime_weights, shap_explanation.values, lime_cfg["num_features"]
    )

    lime_panel, lime_norm = build_importance_heatmap(
        image_rgb01, segments, list(enumerate(lime_weights))
    )
    shap_panel, shap_norm = build_importance_heatmap(
        image_rgb01, segments, list(enumerate(shap_explanation.values))
    )
    gradcam_panel = _build_gradcam_panel(
        model, model_name, image, image_rgb01, target_size, target_idx, device
    )

    predicted_label = idx_to_class.get(target_idx, str(target_idx))
    predicted_prob = float(probabilities[target_idx])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(
        original=image_rgb01,
        lime_panel=lime_panel,
        lime_norm=lime_norm,
        shap_panel=shap_panel,
        shap_norm=shap_norm,
        gradcam_panel=gradcam_panel,
        predicted_label=predicted_label,
        predicted_prob=predicted_prob,
        agreement=agreement,
        segmentation=shap_cfg["segmentation"],
        output_path=output_path,
    )

    dispersion = explanation_dispersion(list(enumerate(shap_explanation.values)))
    _save_artifacts(
        output_path=output_path,
        segments=segments,
        metadata={
            "predicted_label": predicted_label,
            "predicted_prob": predicted_prob,
            "target_idx": target_idx,
            "n_segments": n_segments,
            "segmentation": shap_cfg["segmentation"],
            "shap_expected_value": shap_explanation.expected_value,
            "shap_n_evals": shap_explanation.n_evals,
            "shap_dispersion": dispersion,
            "lime_weights": [float(weight) for weight in lime_weights],
            "shap_values": [float(value) for value in shap_explanation.values],
            "agreement": agreement,
        },
    )

    return {
        "predicted_label": predicted_label,
        "predicted_prob": predicted_prob,
        "dispersion": dispersion,
        "agreement": agreement,
    }


def _build_gradcam_panel(
    model: nn.Module,
    model_name: str | None,
    image: Image.Image,
    image_rgb01: np.ndarray,
    target_size: tuple[int, int],
    target_idx: int,
    device: torch.device,
) -> np.ndarray | None:
    """
    Calcula el overlay de Grad-CAM, o None si la arquitectura no lo soporta.

    @param {nn.Module} model Modelo en modo eval.
    @param {str|None} model_name Nombre registrado, o None para omitir el panel.
    @param {Image.Image} image Imagen original, sin reescalar.
    @param {np.ndarray} image_rgb01 Imagen reescalada en [0, 1].
    @param {tuple[int, int]} target_size Tamano de entrada del checkpoint.
    @param {int} target_idx Clase explicada.
    @param {torch.device} device Dispositivo de computo.
    @returns {np.ndarray|None} Overlay RGB en [0, 1], o None.
    """
    if model_name is None:
        return None
    try:
        target_layer = get_target_layer(model, model_name)
    except KeyError as error:
        logger.warning(f"Grad-CAM omitido: {error}")
        return None

    input_tensor = build_validation_transform(target_size)(image).unsqueeze(0).to(device)
    with GradCAM(model, target_layer) as cam:
        heatmap = cam(input_tensor, class_idx=target_idx)
    return build_gradcam_overlay(image_rgb01, heatmap, target_size)


def _save_artifacts(output_path: Path, segments: np.ndarray, metadata: dict) -> None:
    """
    Persiste los sidecars .json y .npy junto al PNG.

    @param {Path} output_path Ruta del PNG.
    @param {np.ndarray} segments Mapa de superpixeles compartido.
    @param {dict} metadata Contenido del sidecar .json.
    """
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(output_path.with_suffix(".npy"), segments)


def _save_figure(
    original: np.ndarray,
    lime_panel: np.ndarray,
    lime_norm: plt.Normalize,
    shap_panel: np.ndarray,
    shap_norm: plt.Normalize,
    gradcam_panel: np.ndarray | None,
    predicted_label: str,
    predicted_prob: float,
    agreement: dict[str, float],
    segmentation: str,
    output_path: Path,
) -> None:
    """
    Dibuja original, LIME, SHAP y (si existe) Grad-CAM, con una barra de color por metodo.

    @param {np.ndarray} original Imagen reescalada en [0, 1].
    @param {np.ndarray} lime_panel Overlay de importancia de LIME.
    @param {plt.Normalize} lime_norm Normalizacion de la barra de color de LIME.
    @param {np.ndarray} shap_panel Overlay de valores de Shapley.
    @param {plt.Normalize} shap_norm Normalizacion de la barra de color de SHAP.
    @param {np.ndarray|None} gradcam_panel Overlay de Grad-CAM, o None.
    @param {str} predicted_label Clase predicha.
    @param {float} predicted_prob Confianza de la clase predicha.
    @param {dict[str, float]} agreement Metricas de acuerdo entre LIME y SHAP.
    @param {str} segmentation Algoritmo de segmentacion usado.
    @param {Path} output_path Ruta del PNG de salida.
    """
    panels = [(original, "Imagen Original", None), (lime_panel, "LIME", lime_norm)]
    panels.append((shap_panel, "SHAP (valores de Shapley)", shap_norm))
    if gradcam_panel is not None:
        panels.append((gradcam_panel, "Grad-CAM", None))

    fig = plt.figure(figsize=(5 * len(panels), 6), facecolor="white")
    grid = gridspec.GridSpec(1, len(panels) + 2, width_ratios=[1] * len(panels) + [0.06, 0.06])

    for position, (panel, title, _) in enumerate(panels):
        axis = fig.add_subplot(grid[0, position])
        axis.imshow(panel)
        axis.set_title(title, fontsize=13, fontweight="bold", color=_TITLE_COLOR, pad=12)
        axis.axis("off")

    for offset, (norm, label) in enumerate(((lime_norm, "LIME"), (shap_norm, "SHAP"))):
        axis = fig.add_subplot(grid[0, len(panels) + offset])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap="RdYlGn"), cax=axis, label=label)

    fig.suptitle(
        f"Diagnostico: {predicted_label} - Confianza: {predicted_prob * 100:.1f}%",
        fontsize=16,
        fontweight="bold",
        color=_TITLE_COLOR,
    )
    fig.text(
        0.5,
        0.01,
        f"Segmentacion compartida ({segmentation}) - "
        f"IoU top-k: {agreement['iou_topk']:.2f} | "
        f"Spearman: {agreement['spearman']:.2f} | "
        f"Acuerdo de signo: {agreement['sign_agreement']:.2f}",
        ha="center",
        fontsize=9,
        fontstyle="italic",
        color="#95A5A6",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Panel comparado guardado en {output_path}")
```

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/explainability/test_compare_report.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Lint y commit**

```bash
ruff check src/explainability/compare_report.py tests/explainability/test_compare_report.py
ruff format src/explainability/compare_report.py tests/explainability/test_compare_report.py
git add src/explainability/compare_report.py tests/explainability/test_compare_report.py
git commit -m "feat(xai): anade el panel comparado LIME/SHAP/Grad-CAM sobre segmentacion compartida"
```

---

### Task 8: `global_report.py` — perfil global por clase

Agrega las atribuciones de muchas imágenes en dos artefactos: un mapa espacial medio por clase, y una tabla con el ratio hoja/fondo y su confiabilidad.

**Files:**
- Create: `src/explainability/global_report.py`
- Test: `tests/explainability/test_global_report.py`

**Interfaces:**
- Consumes: `leaf_mask` / `mask_coverage` / `is_coverage_degenerate` (T3), `explanation_dispersion` (ya existe).
- Produces: `GlobalAccumulator` con `accumulate(label, correct, shap_values, segments, image_np) -> None`, `summary() -> pd.DataFrame` y `class_maps() -> dict[str, np.ndarray]`; y `write_global_report(accumulator: GlobalAccumulator, output_dir: Path) -> None`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/explainability/test_global_report.py`:

```python
import json

import numpy as np

from src.explainability.global_report import GlobalAccumulator, write_global_report


def _half_leaf_image() -> np.ndarray:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = (40, 160, 40)
    image[:, 8:] = (140, 120, 100)
    return image


def _half_segments() -> np.ndarray:
    segments = np.zeros((16, 16), dtype=np.int64)
    segments[:, 8:] = 1
    return segments


def _uniform_image() -> np.ndarray:
    return np.full((16, 16, 3), 128, dtype=np.uint8)


def test_attribution_on_the_leaf_yields_ratio_one():
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="healthy",
        correct=True,
        shap_values=np.array([1.0, 0.0]),
        segments=_half_segments(),
        image_np=_half_leaf_image(),
    )

    row = accumulator.summary().iloc[0]

    assert row["mean_leaf_attribution_ratio"] == 1.0
    assert row["mean_mask_coverage"] == 0.5
    assert row["n"] == 1
    assert row["n_mask_rejected"] == 0
    assert bool(row["ratio_reliable"])


def test_attribution_on_the_background_yields_ratio_zero():
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="healthy",
        correct=True,
        shap_values=np.array([0.0, 1.0]),
        segments=_half_segments(),
        image_np=_half_leaf_image(),
    )

    assert accumulator.summary().iloc[0]["mean_leaf_attribution_ratio"] == 0.0


def test_degenerate_mask_is_rejected_and_flags_the_class():
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="nitrogen_deficiency",
        correct=False,
        shap_values=np.array([1.0, 0.0]),
        segments=_half_segments(),
        image_np=_uniform_image(),
    )

    row = accumulator.summary().iloc[0]

    assert row["n_mask_rejected"] == 1
    assert not bool(row["ratio_reliable"])


def test_class_map_averages_over_the_accumulated_images():
    accumulator = GlobalAccumulator()
    for _ in range(2):
        accumulator.accumulate(
            label="healthy",
            correct=True,
            shap_values=np.array([1.0, 0.0]),
            segments=_half_segments(),
            image_np=_half_leaf_image(),
        )

    class_map = accumulator.class_maps()["healthy"]

    assert class_map.shape == (16, 16)
    assert class_map[:, :8].mean() == 1.0
    assert class_map[:, 8:].mean() == 0.0


def test_write_global_report_emits_maps_and_summary(tmp_path):
    accumulator = GlobalAccumulator()
    accumulator.accumulate(
        label="healthy",
        correct=True,
        shap_values=np.array([1.0, 0.0]),
        segments=_half_segments(),
        image_np=_half_leaf_image(),
    )

    write_global_report(accumulator, tmp_path)

    assert (tmp_path / "healthy_attribution_map.png").exists()
    assert (tmp_path / "global_summary.csv").exists()
    payload = json.loads((tmp_path / "global_summary.json").read_text(encoding="utf-8"))
    assert payload[0]["label"] == "healthy"
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_global_report.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.explainability.global_report'`.

- [ ] **Step 3: Implementar el módulo**

`src/explainability/global_report.py`:

```python
"""Perfil global por clase: mapa espacial medio de atribucion y ratio hoja/fondo."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.explainability.leaf_mask import is_coverage_degenerate, leaf_mask, mask_coverage
from src.explainability.visual_report import explanation_dispersion

logger = logging.getLogger(__name__)

_UNRELIABLE_REJECTION_RATIO = 0.3


class GlobalAccumulator:
    """
    Acumula atribuciones SHAP de muchas imagenes en un perfil por clase.

    Cada mapa se normaliza por su propio maximo absoluto antes de acumularse: sin eso, una
    imagen con atribuciones de gran magnitud dominaria el promedio de la clase y el mapa
    dejaria de responder "donde mira el modelo" para responder "que imagen grito mas".
    """

    def __init__(self, unreliable_ratio: float = _UNRELIABLE_REJECTION_RATIO):
        self._maps: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._rows: list[dict] = []
        self._unreliable_ratio = unreliable_ratio

    def accumulate(
        self,
        label: str,
        correct: bool,
        shap_values: np.ndarray,
        segments: np.ndarray,
        image_np: np.ndarray,
    ) -> None:
        """
        Incorpora la explicacion de una imagen al perfil de su clase verdadera.

        @param {str} label Clase verdadera de la imagen.
        @param {bool} correct Si el modelo acerto en esa imagen.
        @param {np.ndarray} shap_values Valores de Shapley por segmento.
        @param {np.ndarray} segments Mapa de superpixeles con etiquetas desde 0.
        @param {np.ndarray} image_np Imagen HWC uint8 reescalada a target_size.
        """
        weight_map = shap_values[segments]
        max_abs = np.abs(weight_map).max()
        normalized = weight_map / max_abs if max_abs > 0 else weight_map

        accumulated = self._maps.get(label)
        self._maps[label] = (
            np.abs(normalized) if accumulated is None else accumulated + np.abs(normalized)
        )
        self._counts[label] = self._counts.get(label, 0) + 1

        coverage = mask_coverage(leaf_mask(image_np))
        rejected = is_coverage_degenerate(coverage)
        positive = np.clip(normalized, 0.0, None)
        positive_total = positive.sum()
        usable = not rejected and positive_total > 0

        self._rows.append(
            {
                "label": label,
                "correct": bool(correct),
                "leaf_attribution_ratio": (
                    float(positive[leaf_mask(image_np)].sum() / positive_total)
                    if usable
                    else float("nan")
                ),
                "mask_coverage": coverage,
                "mask_rejected": rejected,
                "abs_attribution": float(np.abs(normalized).mean()),
                "dispersion": explanation_dispersion(list(enumerate(shap_values))),
            }
        )

    def summary(self) -> pd.DataFrame:
        """
        Agrega las filas acumuladas por clase y correctitud.

        Las imagenes con mascara descartada entran en `n` y en `n_mask_rejected`, pero su
        ratio es NaN y queda fuera del promedio: se cuentan sin contaminar la metrica.

        @returns {pd.DataFrame} Una fila por (label, correct) con las metricas agregadas.
        """
        frame = pd.DataFrame(self._rows)
        grouped = (
            frame.groupby(["label", "correct"])
            .agg(
                n=("mask_coverage", "size"),
                n_mask_rejected=("mask_rejected", "sum"),
                mean_leaf_attribution_ratio=("leaf_attribution_ratio", "mean"),
                mean_mask_coverage=("mask_coverage", "mean"),
                mean_abs_attribution=("abs_attribution", "mean"),
                mean_dispersion=("dispersion", "mean"),
            )
            .reset_index()
        )
        grouped["ratio_reliable"] = (
            grouped["n_mask_rejected"] / grouped["n"]
        ) <= self._unreliable_ratio
        return grouped

    def class_maps(self) -> dict[str, np.ndarray]:
        """
        Mapa espacial medio de |atribucion| por clase.

        @returns {dict[str, np.ndarray]} Un mapa HW por clase acumulada.
        """
        return {label: total / self._counts[label] for label, total in self._maps.items()}


def write_global_report(accumulator: GlobalAccumulator, output_dir: Path) -> None:
    """
    Escribe los mapas por clase y la tabla agregada del perfil global.

    @param {GlobalAccumulator} accumulator Acumulador ya alimentado.
    @param {Path} output_dir Directorio destino (`<run_dir>/explain_global/`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, class_map in accumulator.class_maps().items():
        figure, axis = plt.subplots(figsize=(5, 5), facecolor="white")
        image = axis.imshow(class_map, cmap="inferno")
        axis.set_title(f"Atribucion media - {label}", fontsize=12, fontweight="bold")
        axis.axis("off")
        figure.colorbar(image, ax=axis, label="|SHAP| normalizado")
        figure.savefig(
            output_dir / f"{label}_attribution_map.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

    summary = accumulator.summary()
    summary.to_csv(output_dir / "global_summary.csv", index=False)
    (output_dir / "global_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    logger.info(f"Perfil global guardado en {output_dir}")
```

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/explainability/test_global_report.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Lint y commit**

```bash
ruff check src/explainability/global_report.py tests/explainability/test_global_report.py
ruff format src/explainability/global_report.py tests/explainability/test_global_report.py
git add src/explainability/global_report.py tests/explainability/test_global_report.py
git commit -m "feat(xai): anade el perfil global por clase con ratio hoja/fondo instrumentado"
```

---

### Task 9: CLI unificado — subcomandos `visual`, `fidelity` y `errors`

Migración pura: la lógica de `explain_lime.py` y `explain_report.py` se traslada a un solo script con subcomandos y una resolución de runs compartida. Los subcomandos nuevos llegan en la Tarea 10.

**Decisión a aplicar durante la migración:** hoy `explain_lime.py` usa `dataset.classes` como fallback de clases y `explain_report.py` usa `baseline.classes` cuando `--baseline` está activo — pese a que su propio comentario advierte contra reconstruir el mapeo desde `baseline.classes`. El CLI unifica en `dataset.classes` (orden canónico documentado en `CLAUDE.md`). Hoy es un no-op porque ambas listas tienen el mismo orden en el YAML, y solo aplica a runs sin `summary.json`.

**Files:**
- Create: `scripts/pipeline/explain.py`
- Delete: `scripts/pipeline/explain_lime.py`, `scripts/pipeline/explain_report.py`
- Test: `tests/explainability/test_explain_cli.py`

**Interfaces:**
- Consumes: `explain_model_visual` / `render_visual_explanation` / `sample_balanced` / `explanation_dispersion` de `visual_report.py`, `resolve_run_dir` / `load_run_metadata` / `select_device` de `src/training/common.py`.
- Produces: `build_parser() -> argparse.ArgumentParser` con los cinco subcomandos, y `RunContext` (dataclass con `model_name`, `run_dir`, `model`, `idx_to_class`, `target_size`, `splits_dir`, `device`).

- [ ] **Step 1: Escribir los tests que fallan**

`tests/explainability/test_explain_cli.py`:

```python
import pytest

from scripts.pipeline.explain import build_parser

_SUBCOMMANDS = {"visual", "fidelity", "errors", "compare", "global"}


def test_every_subcommand_is_registered():
    parser = build_parser()

    actions = [
        action for action in parser._subparsers._group_actions if hasattr(action, "choices")
    ]

    assert _SUBCOMMANDS <= set(actions[0].choices)


@pytest.mark.parametrize("subcommand", sorted(_SUBCOMMANDS))
def test_common_flags_are_available_everywhere(subcommand):
    args = build_parser().parse_args(
        [subcommand, "--models", "shufflenet_v2_x1_0", "--output-dir", "outputs/main"]
    )

    assert args.command == subcommand
    assert args.models == ["shufflenet_v2_x1_0"]
    assert args.output_dir == "outputs/main"


def test_visual_accepts_a_single_image():
    args = build_parser().parse_args(["visual", "--image", "foto.jpg", "--output", "out.png"])

    assert args.image == "foto.jpg"
    assert args.output == "out.png"


def test_fidelity_accepts_sampling_overrides():
    args = build_parser().parse_args(["fidelity", "--sample-size", "50", "--num-samples", "500"])

    assert args.sample_size == 50
    assert args.num_samples == 500


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_explain_cli.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'scripts.pipeline.explain'`.

- [ ] **Step 3: Crear el CLI con el parser y la resolución de runs compartida**

`scripts/pipeline/explain.py`:

```python
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
```

- [ ] **Step 4: Añadir los tres subcomandos migrados**

En `scripts/pipeline/explain.py`, antes de `main()`, añadir:

```python
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

    if args.output is not None and (args.image is None or len(args.models) != 1):
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
            f"[{context.model_name}] {image_path.name}: explicado "
            f"(correct={rows[-1]['correct']})"
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
```

Añadir también los stubs de los dos subcomandos que llegan en la Tarea 10, para que `main()` resuelva:

```python
def cmd_compare(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """Panel comparado LIME | SHAP | Grad-CAM. Se implementa en la Tarea 10."""
    raise SystemExit("El subcomando 'compare' aun no esta implementado.")


def cmd_global(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """Perfil global por clase. Se implementa en la Tarea 10."""
    raise SystemExit("El subcomando 'global' aun no esta implementado.")
```

- [ ] **Step 5: Correr los tests del CLI**

Run: `pytest tests/explainability/test_explain_cli.py -v`
Expected: 9 PASS (5 tests, uno parametrizado sobre 5 subcomandos).

- [ ] **Step 6: Verificar la paridad funcional contra un run real**

Run: `python scripts/pipeline/explain.py visual --models shufflenet_v2_x1_0 --output-dir outputs/main`
Expected: genera paneles bajo `outputs/main/shufflenet_v2_x1_0/<run_id>/explain_visual/`, con el mismo aspecto que los de `lime_visual/` del mismo run.

Si no hay runs locales en `outputs/main`, usar `--output-dir outputs/baselines`. Si no hay ninguno, saltar este paso y anotarlo en el commit.

- [ ] **Step 7: Eliminar los scripts viejos**

```bash
git rm scripts/pipeline/explain_lime.py scripts/pipeline/explain_report.py
```

- [ ] **Step 8: Lint y commit**

```bash
ruff check scripts/pipeline/explain.py tests/explainability/test_explain_cli.py
ruff format scripts/pipeline/explain.py tests/explainability/test_explain_cli.py
git add scripts/pipeline/explain.py tests/explainability/test_explain_cli.py
git commit -m "refactor(xai): unifica explain_lime y explain_report en un CLI con subcomandos"
```

---

### Task 10: CLI — subcomandos `compare` y `global`

**Files:**
- Modify: `scripts/pipeline/explain.py` (reemplaza los dos stubs)
- Test: `tests/explainability/test_explain_cli.py` (añade dos tests)

**Interfaces:**
- Consumes: `render_comparison` (T7), `GlobalAccumulator` / `write_global_report` (T8), `build_segments` (T2), `explain_with_kernel_shap` (T5), `iter_run_contexts` / `RunContext` (T9).
- Produces: `cmd_compare` y `cmd_global` funcionales.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir al final de `tests/explainability/test_explain_cli.py`:

```python
def test_compare_accepts_shap_overrides():
    args = build_parser().parse_args(["compare", "--sample-size", "3", "--nsamples", "128"])

    assert args.sample_size == 3
    assert args.nsamples == 128


def test_compare_and_global_are_no_longer_stubs():
    from scripts.pipeline import explain

    assert "no esta implementado" not in (explain.cmd_compare.__doc__ or "")
    assert "no esta implementado" not in (explain.cmd_global.__doc__ or "")
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `pytest tests/explainability/test_explain_cli.py -v`
Expected: `test_compare_and_global_are_no_longer_stubs` FAIL.

- [ ] **Step 3: Implementar `cmd_compare`**

En `scripts/pipeline/explain.py`, reemplazar el stub de `cmd_compare` por:

```python
def cmd_compare(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """
    Panel comparado LIME | SHAP | Grad-CAM sobre una muestra balanceada del test set.

    Cruza con predictions.csv para poder desglosar el acuerdo entre aciertos y errores.

    @param {argparse.Namespace} args Argumentos del subcomando.
    @param {dict} cfg Configuracion del proyecto.
    @param {torch.device} device Dispositivo de computo.
    """
    from src.explainability.compare_report import render_comparison
    from src.explainability.visual_report import sample_balanced

    lime_cfg = cfg["lime"]
    shap_cfg = dict(cfg["shap"])
    if args.nsamples:
        shap_cfg["nsamples"] = args.nsamples
    gradcam_enabled = cfg.get("gradcam", {}).get("enabled", False)
    sample_size = args.sample_size or shap_cfg["images_per_class"]
    dataset_root = get_dataset_root()

    for context in iter_run_contexts(args, cfg, device, require_predictions=True):
        predictions = pd.read_csv(context.run_dir / "predictions.csv")
        df_sample = sample_balanced(predictions, sample_size, shap_cfg["seed"])
        panel_dir = context.run_dir / "explain_compare"
        rows: list[dict] = []

        for _, row in df_sample.iterrows():
            image_path = dataset_root / row["image_path"]
            try:
                image = load_and_normalize_image(image_path)
            except (FileNotFoundError, RuntimeError) as error:
                logger.warning(f"[{context.model_name}] Saltando {image_path}: {error}")
                continue

            result = render_comparison(
                image=image,
                model=context.model,
                model_name=context.model_name if gradcam_enabled else None,
                idx_to_class=context.idx_to_class,
                target_size=context.target_size,
                output_path=panel_dir / f"{image_path.stem}__true-{row['label']}.png",
                lime_cfg=lime_cfg,
                shap_cfg=shap_cfg,
                device=device,
            )
            rows.append(
                {
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "pred_label": row["pred_label"],
                    "pred_prob": row["pred_prob"],
                    "correct": row["label"] == row["pred_label"],
                    "dispersion": result["dispersion"],
                    **result["agreement"],
                }
            )
            logger.info(
                f"[{context.model_name}] {image_path.name}: "
                f"IoU={result['agreement']['iou_topk']:.2f} "
                f"Spearman={result['agreement']['spearman']:.2f}"
            )

        if not rows:
            logger.info(f"[{context.model_name}] Nada que comparar. Se omite.")
            continue

        summary = (
            pd.DataFrame(rows)
            .groupby(["label", "correct"])
            .agg(
                n=("dispersion", "size"),
                mean_pred_prob=("pred_prob", "mean"),
                mean_shap_dispersion=("dispersion", "mean"),
                mean_iou_topk=("iou_topk", "mean"),
                mean_spearman=("spearman", "mean"),
                mean_sign_agreement=("sign_agreement", "mean"),
            )
            .reset_index()
        )
        panel_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(panel_dir / "summary.csv", index=False)
        (panel_dir / "summary.json").write_text(
            summary.to_json(orient="records", indent=2), encoding="utf-8"
        )
        logger.info(f"[{context.model_name}] Acuerdo:\n{summary.to_string(index=False)}")
```

- [ ] **Step 4: Implementar `cmd_global`**

Reemplazar el stub de `cmd_global` por:

```python
def cmd_global(args: argparse.Namespace, cfg: dict, device: torch.device) -> None:
    """
    Perfil global por clase: mapa espacial medio de atribucion y ratio hoja/fondo.

    No renderiza paneles por imagen: solo acumula los valores de Shapley, que es lo que
    permite subir el tamano de muestra sin llenar el run de PNGs.

    @param {argparse.Namespace} args Argumentos del subcomando.
    @param {dict} cfg Configuracion del proyecto.
    @param {torch.device} device Dispositivo de computo.
    """
    import numpy as np

    from src.explainability.global_report import GlobalAccumulator, write_global_report
    from src.explainability.kernel_shap import explain_with_kernel_shap
    from src.explainability.segmentation import build_segments
    from src.explainability.visual_report import (
        build_predict_fn,
        prepare_lime_image,
        sample_balanced,
    )

    shap_cfg = dict(cfg["shap"])
    if args.nsamples:
        shap_cfg["nsamples"] = args.nsamples
    sample_size = args.sample_size or shap_cfg["global_sample_size"]
    dataset_root = get_dataset_root()

    for context in iter_run_contexts(args, cfg, device, require_predictions=True):
        predictions = pd.read_csv(context.run_dir / "predictions.csv")
        df_sample = sample_balanced(predictions, sample_size, shap_cfg["seed"])
        predict_fn = build_predict_fn(context.model, device, context.target_size)
        accumulator = GlobalAccumulator()

        for _, row in df_sample.iterrows():
            image_path = dataset_root / row["image_path"]
            try:
                image = load_and_normalize_image(image_path)
            except (FileNotFoundError, RuntimeError) as error:
                logger.warning(f"[{context.model_name}] Saltando {image_path}: {error}")
                continue

            image_np = prepare_lime_image(image, context.target_size)
            segments = build_segments(
                image_np,
                algorithm=shap_cfg["segmentation"],
                n_segments=shap_cfg["n_segments"],
                compactness=shap_cfg["compactness"],
            )
            explanation = explain_with_kernel_shap(
                image_np=image_np,
                segments=segments,
                predict_fn=predict_fn,
                target_idx=int(np.argmax(predict_fn(image_np[np.newaxis, ...])[0])),
                nsamples=shap_cfg["nsamples"],
                batch_size=shap_cfg["batch_size"],
                background=shap_cfg["background"],
                seed=shap_cfg["seed"],
            )
            accumulator.accumulate(
                label=row["label"],
                correct=row["label"] == row["pred_label"],
                shap_values=explanation.values,
                segments=segments,
                image_np=image_np,
            )
            logger.info(f"[{context.model_name}] {image_path.name}: acumulado")

        write_global_report(accumulator, context.run_dir / "explain_global")
```

- [ ] **Step 5: Correr los tests**

Run: `pytest tests/explainability/ -v`
Expected: todo PASS.

- [ ] **Step 6: Verificar contra un run real (si hay uno disponible)**

Run: `python scripts/pipeline/explain.py compare --models shufflenet_v2_x1_0 --output-dir outputs/main --sample-size 1 --nsamples 256`
Expected: un panel por clase bajo `explain_compare/` y un `summary.csv` con las tres columnas de acuerdo.

Run: `python scripts/pipeline/explain.py global --models shufflenet_v2_x1_0 --output-dir outputs/main --sample-size 2 --nsamples 256`
Expected: un `<clase>_attribution_map.png` por clase y `global_summary.csv` con `ratio_reliable`.

- [ ] **Step 7: Lint y commit**

```bash
ruff check scripts/pipeline/explain.py tests/explainability/test_explain_cli.py
ruff format scripts/pipeline/explain.py tests/explainability/test_explain_cli.py
git add scripts/pipeline/explain.py tests/explainability/test_explain_cli.py
git commit -m "feat(xai): anade los subcomandos compare y global al CLI de explicabilidad"
```

---

### Task 11: Makefile y Modal

**Files:**
- Modify: `Makefile:163-199` (targets locales), `Makefile:276-311` (targets Modal), `Makefile:76-95` (bloque de `help`)
- Modify: `scripts/modal/explain.py` (renombre de funciones + dos nuevas)

**Interfaces:**
- Consumes: los cinco subcomandos de `scripts/pipeline/explain.py`.
- Produces: targets `explain-{visual,fidelity,errors}-{baselines,main}`, `explain-{compare,global}-main` y sus equivalentes `modal-*`; funciones Modal `explain_visual`, `explain_fidelity`, `explain_errors`, `explain_compare`, `explain_global`.

- [ ] **Step 1: Reemplazar el bloque de targets locales**

En `Makefile`, reemplazar todo el bloque entre `.PHONY: explain-lime ...` (línea 163) y `explain-errors-main:` inclusive (línea 199) por:

```makefile
.PHONY: explain-visual explain-fidelity explain-errors explain-compare explain-global \
	explain-visual-baselines explain-fidelity-baselines explain-errors-baselines \
	explain-visual-main explain-fidelity-main explain-errors-main \
	explain-compare-main explain-global-main

explain-visual:
	$(PYTHON) scripts/pipeline/explain.py visual --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(IMAGE),--image $(IMAGE),) \
		$(if $(OUTPUT),--output $(OUTPUT),) $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

explain-fidelity:
	$(PYTHON) scripts/pipeline/explain.py fidelity --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

explain-errors:
	$(PYTHON) scripts/pipeline/explain.py errors --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

explain-compare:
	$(PYTHON) scripts/pipeline/explain.py compare --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

explain-global:
	$(PYTHON) scripts/pipeline/explain.py global --models $(MODELS) \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

# Runs de baselines (outputs/baselines).
explain-visual-baselines:
	$(MAKE) explain-visual MODELS="$(MODELS)"

explain-fidelity-baselines:
	$(MAKE) explain-fidelity MODELS="$(MODELS)"

explain-errors-baselines:
	$(MAKE) explain-errors MODELS="$(MODELS)"

# Runs del pipeline principal (outputs/main). SHAP (compare/global) es exclusivo de aqui.
explain-visual-main:
	$(MAKE) explain-visual MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

explain-fidelity-main:
	$(MAKE) explain-fidelity MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

explain-errors-main:
	$(MAKE) explain-errors MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

explain-compare-main:
	$(MAKE) explain-compare MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"

explain-global-main:
	$(MAKE) explain-global MODELS="$(MAIN_MODELS)" OUTPUT_DIR="$(MAIN_OUTPUT_DIR)"
```

- [ ] **Step 2: Añadir la variable `NSAMPLES`**

En el bloque de variables de explicabilidad (línea ~50, junto a `NUM_SAMPLES` y `SAMPLE_SIZE`), añadir:

```makefile
NSAMPLES ?=
```

- [ ] **Step 3: Reemplazar el bloque de targets Modal**

En `Makefile`, reemplazar el bloque entre `.PHONY: modal-explain-lime ...` (línea 276) y `modal-explain-errors-main:` inclusive (línea 311) por:

```makefile
.PHONY: modal-explain-visual modal-explain-fidelity modal-explain-errors \
	modal-explain-compare modal-explain-global \
	modal-explain-visual-baselines modal-explain-fidelity-baselines modal-explain-errors-baselines \
	modal-explain-visual-main modal-explain-fidelity-main modal-explain-errors-main \
	modal-explain-compare-main modal-explain-global-main

modal-explain-visual:
	$(MODAL) run scripts/modal/explain.py::explain_visual --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(IMAGE),--image $(IMAGE),) $(if $(OUTPUT),--output $(OUTPUT),)

modal-explain-fidelity:
	$(MODAL) run scripts/modal/explain.py::explain_fidelity --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

modal-explain-errors:
	$(MODAL) run scripts/modal/explain.py::explain_errors --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

modal-explain-compare:
	$(MODAL) run scripts/modal/explain.py::explain_compare --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),)

modal-explain-global:
	$(MODAL) run scripts/modal/explain.py::explain_global --models "$(MODELS)" --pipeline "$(PIPELINE)" \
		$(if $(RUN),--run $(RUN),) $(if $(SAMPLE_SIZE),--sample-size $(SAMPLE_SIZE),) \
		$(if $(NSAMPLES),--nsamples $(NSAMPLES),)

modal-explain-visual-baselines:
	$(MAKE) modal-explain-visual MODELS="$(MODELS)" PIPELINE=baselines

modal-explain-fidelity-baselines:
	$(MAKE) modal-explain-fidelity MODELS="$(MODELS)" PIPELINE=baselines

modal-explain-errors-baselines:
	$(MAKE) modal-explain-errors MODELS="$(MODELS)" PIPELINE=baselines

modal-explain-visual-main:
	$(MAKE) modal-explain-visual MODELS="$(MAIN_MODELS)" PIPELINE=main

modal-explain-fidelity-main:
	$(MAKE) modal-explain-fidelity MODELS="$(MAIN_MODELS)" PIPELINE=main

modal-explain-errors-main:
	$(MAKE) modal-explain-errors MODELS="$(MAIN_MODELS)" PIPELINE=main

modal-explain-compare-main:
	$(MAKE) modal-explain-compare MODELS="$(MAIN_MODELS)" PIPELINE=main

modal-explain-global-main:
	$(MAKE) modal-explain-global MODELS="$(MAIN_MODELS)" PIPELINE=main
```

- [ ] **Step 4: Actualizar el bloque de `help`**

En `Makefile`, líneas 76-91, reemplazar las líneas que listan los targets de explicabilidad:

```makefile
	@echo "  explain-visual-baselines explain-fidelity-baselines explain-errors-baselines"
```

```makefile
	@echo "  explain-visual-main explain-fidelity-main explain-errors-main"
	@echo "  explain-compare-main explain-global-main   (SHAP: solo pipeline principal)"
```

```makefile
	@echo "  modal-explain-visual-baselines modal-explain-fidelity-baselines modal-explain-errors-baselines"
```

```makefile
	@echo "  modal-explain-visual-main modal-explain-fidelity-main modal-explain-errors-main"
	@echo "  modal-explain-compare-main modal-explain-global-main"
```

Y en la línea 95, reemplazar `explain-lime, modal-explain-report` por `explain-visual, modal-explain-fidelity`.

- [ ] **Step 5: Actualizar `scripts/modal/explain.py`**

Renombrar `explain_lime` → `explain_visual` y `explain_report` → `explain_fidelity`, y en las tres funciones existentes reemplazar la construcción de argumentos para que invoquen el CLI unificado. `explain_visual` queda así (las otras siguen el mismo patrón, cambiando el subcomando y sus flags):

```python
@app.function(gpu=_GPU, volumes=_VOLUMES, secrets=[modal.Secret.from_name("hf")], timeout=3600)
def explain_visual(
    models: str = DEFAULT_MODELS,
    run: str = "",
    baseline: bool = False,
    image: str = "",
    output: str = "",
    pipeline: str = "baselines",
) -> None:
    """Panel LIME + Grad-CAM por imagen. Espeja `make explain-visual`.

    image/output son rutas dentro del contenedor (relativas a /data o /outputs, los
    Volumes montados) - no rutas del filesystem local del caller. pipeline elige de que
    directorio de runs leer: "baselines" (default) o "main".
    """
    args = [sys.executable, "scripts/pipeline/explain.py", "visual", "--models", *models.split()]
    args += _output_dir_args(pipeline)
    if run:
        args += ["--run", run]
    if baseline:
        args += ["--baseline"]
    if image:
        args += ["--image", image]
    if output:
        args += ["--output", output]
    subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()
```

En `explain_fidelity` el subcomando es `"fidelity"` y conserva `sample_size` / `num_samples`. En `explain_errors` el subcomando es `"errors"`, conserva `num_samples` y **se elimina** el `"--errors-only"` de la lista de argumentos.

Añadir al final del archivo las dos funciones nuevas:

```python
# compare corre LIME + KernelSHAP por imagen sobre una muestra chica (5/clase por defecto):
# ~5 s/imagen en A10, con holgura de sobra dentro de la hora.
@app.function(gpu=_GPU, volumes=_VOLUMES, secrets=[modal.Secret.from_name("hf")], timeout=3600)
def explain_compare(
    models: str = DEFAULT_MODELS,
    run: str = "",
    baseline: bool = False,
    sample_size: int = 0,
    nsamples: int = 0,
    pipeline: str = "main",
) -> None:
    """Panel comparado LIME | SHAP | Grad-CAM. Espeja `make explain-compare-main`.

    SHAP esta reservado al pipeline principal, de ahi que el default de pipeline sea
    "main" y no "baselines" como en el resto de las funciones de este modulo.
    """
    args = [sys.executable, "scripts/pipeline/explain.py", "compare", "--models", *models.split()]
    args += _output_dir_args(pipeline)
    if run:
        args += ["--run", run]
    if baseline:
        args += ["--baseline"]
    if sample_size:
        args += ["--sample-size", str(sample_size)]
    if nsamples:
        args += ["--nsamples", str(nsamples)]
    subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()


# global barre 30 imagenes/clase x 9 clases sin renderizar paneles: ~25 min en A10. Las 3 h
# dan margen para subir --sample-size sin volver a tocar el timeout.
@app.function(gpu=_GPU, volumes=_VOLUMES, secrets=[modal.Secret.from_name("hf")], timeout=3 * 3600)
def explain_global(
    models: str = DEFAULT_MODELS,
    run: str = "",
    baseline: bool = False,
    sample_size: int = 0,
    nsamples: int = 0,
    pipeline: str = "main",
) -> None:
    """Perfil global por clase. Espeja `make explain-global-main`."""
    args = [sys.executable, "scripts/pipeline/explain.py", "global", "--models", *models.split()]
    args += _output_dir_args(pipeline)
    if run:
        args += ["--run", run]
    if baseline:
        args += ["--baseline"]
    if sample_size:
        args += ["--sample-size", str(sample_size)]
    if nsamples:
        args += ["--nsamples", str(nsamples)]
    subprocess.run(args, check=True, cwd=REPO_ANCHOR)
    outputs_vol.commit()
```

Actualizar también el docstring del módulo (líneas 1-20) para que nombre los cinco subcomandos y el CLI unificado.

- [ ] **Step 6: Verificar que los targets resuelven**

Run: `make help`
Expected: lista los targets nuevos y ninguno con `-lime` ni `-report`.

Run: `make explain-visual MODELS=shufflenet_v2_x1_0 OUTPUT_DIR=outputs/main`
Expected: ejecuta el CLI (o falla con el mensaje claro de "No hay runs registrados", que también valida el cableado).

- [ ] **Step 7: Commit**

```bash
git add Makefile scripts/modal/explain.py
git commit -m "refactor(xai): renombra los targets de explicabilidad y anade compare/global en Modal"
```

---

### Task 12: Documentación y skill

**Files:**
- Modify: `CLAUDE.md:21,23,34,36,62-67`
- Modify: `README.md:148-150,158-160,175,179,203`
- Modify: `LOCAL.md:78`
- Modify: `docs/es/pipeline/interpretabilidad.md`
- Modify: `docs/es/pipeline-baselines/interpretabilidad.md`
- Modify: `docs/es/deployment/modal.md:49-74`
- Rename: `.claude/skills/corn-lime-explainability/` → `.claude/skills/corn-xai/`

**Interfaces:**
- Consumes: los nombres definitivos de la Tarea 11.
- Produces: documentación consistente. Sin código.

- [ ] **Step 1: Renombrar y reescribir la skill**

```bash
git mv .claude/skills/corn-lime-explainability .claude/skills/corn-xai
```

Reescribir `.claude/skills/corn-xai/SKILL.md`:

```markdown
---
name: corn-xai
description: Use when running or editing scripts/pipeline/explain.py, src/explainability/*, or scripts/checks/lime_stability.py, or when asked about LIME, SHAP, Grad-CAM, fidelity, error analysis or global attribution profiles for the corn leaf disease project.
---

# Corn XAI

## Flujo

La explicabilidad es post-hoc y no corre al entrenar. `train_baselines.py` mantiene el flag `--lime` para encadenarla puntualmente, pero `make train-baselines` no lo pasa por defecto.

## CLI unificado (`scripts/pipeline/explain.py`)

Cinco subcomandos. Los nombres describen el artefacto, no la técnica: casi todos los paneles combinan varias.

| Subcomando | Produce | Directorio |
|---|---|---|
| `visual` | Panel por imagen: LIME + Grad-CAM | `<run_dir>/explain_visual/` |
| `fidelity` | Muestra amplia + resumen por clase | `<run_dir>/explain_fidelity/` |
| `errors` | Panel sobre `label != pred_label` | `<run_dir>/explain_errors/` |
| `compare` | Panel LIME \| SHAP \| Grad-CAM + acuerdo | `<run_dir>/explain_compare/` |
| `global` | Mapas medios por clase + ratio hoja/fondo | `<run_dir>/explain_global/` |

`compare` y `global` son exclusivos del pipeline principal (`outputs/main`).

## Targets: local vs. Modal, baselines vs. main

Los tres primeros subcomandos tienen cuatro entradas en el Makefile: `explain-<sub>-{baselines,main}` y `modal-explain-<sub>-{baselines,main}`. `compare` y `global` solo tienen `-main`. Los targets sin sufijo son genéricos y apuntan a baselines salvo que reciban `OUTPUT_DIR` (local) o `PIPELINE=main` (Modal). No pasar rutas absolutas por `make` en Windows: MSYS las reescribe.

## SHAP: lo que hay que saber antes de tocarlo

- **KernelSHAP sobre superpíxeles**, con línea base única configurable (`shap.background`, `black` por defecto para paridad con el `hide_color=0` de LIME).
- **La segmentación se calcula afuera** (`src/explainability/segmentation.py`) y se inyecta tanto en LIME como en SHAP. Sin eso no hay comparación por superpíxel: LIME segmenta con quickshift por su cuenta.
- **Consecuencia:** las regiones LIME de `compare` no coinciden con las de `visual`. Es esperado; `shap.segmentation: quickshift` las hace coincidir.
- **Determinismo:** `explain_with_kernel_shap` fija la semilla de NumPy y restaura el estado previo. No quitar ese bloque: el determinismo es el argumento por el que SHAP entró al pipeline.
- **Ratio hoja/fondo:** ExG+Otsu es una heurística y puede fallar en hojas cloróticas. Nunca leer `mean_leaf_attribution_ratio` sin mirar `ratio_reliable` y `n_mask_rejected` en la misma fila.

## `scripts/checks/lime_stability.py`

Diagnóstico manual (sin target Make) para auditar la estabilidad de LIME sobre una imagen corriendo varias seeds y comparando IoU/correlación.
```

- [ ] **Step 2: Actualizar `CLAUDE.md`**

- Línea 23 y 34: reemplazar las menciones a `explain_lime.py` / `explain_report.py` por `scripts/pipeline/explain.py` con sus cinco subcomandos, añadiendo SHAP (`compare`, `global`) como parte del pipeline principal.
- Línea 36: reemplazar `Flujo LIME (make explain-lime/explain-report/explain-errors, lime_stability.py) → skill corn-lime-explainability` por `Flujo XAI (make explain-visual/fidelity/errors/compare/global, lime_stability.py) → skill corn-xai`.
- Líneas 62-67: reemplazar el bloque de comandos por los targets nuevos, añadiendo las dos líneas de `explain-compare-main` / `explain-global-main`.
- Línea 21: añadir a la descripción del pipeline principal que la explicabilidad de esta etapa incluye SHAP.

- [ ] **Step 3: Actualizar `README.md`**

- Líneas 148-150, 158-160, 175, 179: renombrar los targets y añadir `explain-compare-main` / `explain-global-main` y sus variantes `modal-`.
- Línea 203: reemplazar `explain_lime.py, explain_report.py` por `explain.py`.

- [ ] **Step 4: Actualizar `LOCAL.md`**

Línea 78, reemplazar por:

```markdown
- `xai`: lime, shap, scikit-image, matplotlib (necesario para `make explain-visual`/`fidelity`/`errors`/`compare`/`global`)
```

- [ ] **Step 5: Actualizar `docs/es/deployment/modal.md`**

Líneas 49-74: renombrar los targets y las funciones Modal, añadir un bloque para `modal-explain-compare-main` / `modal-explain-global-main` señalando que SHAP solo aplica al pipeline principal, y actualizar la línea 63 para que hable del CLI unificado en vez de los dos scripts viejos.

- [ ] **Step 6: Actualizar `docs/es/pipeline/interpretabilidad.md`**

El documento anuncia SHAP en futuro ("A esta base se sumará SHAP"). Reescribirlo en presente, describiendo lo implementado: el panel comparado sobre segmentación compartida, las tres métricas de acuerdo, el perfil global por clase, y la salvedad honesta sobre el ratio hoja/fondo (`ratio_reliable`).

- [ ] **Step 7: Actualizar `docs/es/pipeline-baselines/interpretabilidad.md`**

Renombrar las menciones a los targets viejos. Dejar explícito que SHAP no aplica a los baselines y por qué.

- [ ] **Step 8: Verificar que no quedan referencias muertas**

Run: `grep -rn "explain-lime\|explain-report\|explain_lime\|explain_report\|lime_visual\|lime_report\|lime_errors\|corn-lime-explainability" --include="*.md" --include="*.py" --include="Makefile" . | grep -v venv | grep -v specs/`
Expected: sin resultados. Los archivos bajo `specs/` documentan el estado anterior y no se tocan.

- [ ] **Step 9: Correr la suite completa**

Run: `pytest -v`
Expected: todo PASS.

- [ ] **Step 10: Commit**

```bash
git add CLAUDE.md README.md LOCAL.md docs/ .claude/skills/
git commit -m "docs(xai): documenta SHAP y el CLI unificado de explicabilidad"
```

---

## Self-Review

**Cobertura del spec:**

| Sección del spec | Tarea |
|---|---|
| §1 entregable 1 (KernelSHAP) | T5 |
| §1 entregable 2 (panel comparado) | T2, T4, T6, T7 |
| §1 entregable 3 (perfil global) | T3, T8 |
| §1 entregable 4 (CLI unificado) | T9, T10 |
| §3 D1-D2 (KernelSHAP, segmentación compartida) | T2, T5, T7 |
| §3 D3 (dependencia `shap`) | T1 |
| §3 D4 (solo pipeline principal) | T11 (targets sin `-baselines`) |
| §3 D5 (LIME sin cambios) | T6 |
| §3 D6 (SLIC, `shap.segmentation` configurable) | T1, T2 |
| §3 D7 (línea base `black`) | T1, T5 |
| §3 D8-D9 (nombres, `errors` como subcomando) | T9, T11 |
| §4 (CLI y targets) | T9, T10, T11 |
| §5 (seis módulos) | T2-T8 |
| §5.3 (instrumentación de la máscara) | T3, T8 |
| §6 (artefactos y renombres) | T6, T9, T10 |
| §7 (configuración) | T1 |
| §8 (costo) | T11 (timeouts de Modal) |
| §9 R1 (fallback si `shap` falla) | T1 Step 2 |
| §10 (pruebas) | T2-T10 |
| §11 (archivos afectados) | T12 |

Sin huecos.

**Consistencia de tipos:** `build_segments` devuelve `int64` con etiquetas desde 0, que es lo que `values[segments]` (T8) y `segment_masks` (T5) asumen. `densify_weights` y `ShapExplanation.values` producen vectores de longitud `n_segments`, que es lo que `attribution_agreement` exige y valida. `render_comparison` devuelve `dispersion` y `agreement`, que es exactamente lo que `cmd_compare` consume para armar su `summary.csv`.

**Nota de secuencia:** la Tarea 9 introduce dos stubs (`cmd_compare`, `cmd_global`) que la Tarea 10 reemplaza. Es deliberado: permite que la migración del CLI se revise y se commitee por separado de la funcionalidad nueva, y el test `test_compare_and_global_are_no_longer_stubs` impide que queden olvidados.
