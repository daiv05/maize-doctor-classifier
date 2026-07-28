# Diseño: SHAP en el pipeline principal + CLI unificado de explicabilidad

**Fecha:** 2026-07-27
**Estado:** aprobado, pendiente de plan de implementación
**Fuentes:** `specs/2026-07-26-pipeline-principal-design.md` (§1, fuera de alcance), `docs/es/deep-learning/interpretability.md` (§SHAP, tabla comparativa), `docs/es/pipeline/interpretabilidad.md`

## 1. Objetivo y alcance

Cerrar el único pendiente de explicabilidad que el spec del pipeline principal dejó declarado y sin resolver: **SHAP**. `docs/es/deep-learning/interpretability.md` lo reserva explícitamente "para el pipeline principal y los análisis globales una vez que tengamos un modelo final estable", y `docs/es/pipeline/interpretabilidad.md` promete tres cosas concretas — atribuciones deterministas, fundamento en valores de Shapley, y agregabilidad a una **visión global** por clase.

Este spec implementa esas tres, y aprovecha que la superficie CLI de explicabilidad debe tocarse igual para consolidarla y corregir sus nombres.

### Entregables

1. **KernelSHAP sobre superpíxeles** para checkpoints ya entrenados del pipeline principal.
2. **Panel comparado por imagen**: LIME | SHAP | Grad-CAM sobre la **misma** segmentación, con métricas de acuerdo entre LIME y SHAP.
3. **Perfil global por clase**: mapa espacial medio de atribución, ratio de atribución sobre hoja vs. fondo, y tabla agregada con desglose acierto/error.
4. **CLI unificado** `scripts/pipeline/explain.py` con subcomandos, que absorbe `explain_lime.py` y `explain_report.py`, con nombres que describen el artefacto en vez de la técnica.

### Fuera de alcance

- **DeepSHAP / GradientSHAP.** Son específicos de red — el mismo tipo de acoplamiento a la arquitectura que ya obligó a mantener `GRADCAM_TARGET_LAYERS` a mano. Si el costo de KernelSHAP resulta prohibitivo en la práctica, se revisa con datos medidos, no por anticipado.
- **SHAP en el pipeline de baselines.** Decisión explícita: los docs sitúan SHAP en la etapa final y los resultados de la Tabla 6.2 ya están publicados.
- **Sustituir LIME por SHAP** en el reporte de fidelidad existente. `fidelity` y `errors` siguen siendo LIME + Grad-CAM.
- **Cambios de comportamiento en LIME o Grad-CAM.** Ver D5.
- **Alias deprecados de los targets Make viejos.** El renombre es de una sola vez y se documenta en el mismo commit.

## 2. Contexto que motiva las decisiones

La explicabilidad actual tiene tres scripts (`explain_lime.py`, `explain_report.py`, `inference_report.py`) y cuatro módulos en `src/explainability/`. Dos hechos condicionan el diseño:

**a) Los nombres actuales mienten.** `make explain-lime` produce un panel de LIME **y Grad-CAM**; `explain-report` y `explain-errors` producen ese mismo panel más agregados. El nombre apunta a una técnica cuando el artefacto ya combina varias. Sumar SHAP a esa nomenclatura la vuelve insostenible.

**b) LIME no usa SLIC.** `lime_image.LimeImageExplainer` segmenta internamente con **quickshift** por defecto, no con SLIC. Para que las atribuciones de SHAP y LIME sean comparables **por superpíxel** —y no solo espacialmente— ambas deben evaluarse sobre el mismo mapa de segmentos, calculado una vez y pasado a los dos explicadores. Esto tiene una consecuencia visible que se acepta y documenta en D6.

## 3. Decisiones de diseño

| # | Decisión | Justificación |
|---|---|---|
| D1 | **KernelSHAP sobre superpíxeles**, con línea base única | Aditivo (`Σφ = f(x) − f(base)`), determinista dada la seed, y directamente comparable con LIME por compartir unidad de atribución. Las variantes basadas en gradiente quedan fuera (ver §1). |
| D2 | **Segmentación canónica compartida**, calculada fuera de ambos explicadores | Sin esto el acuerdo SHAP↔LIME sería una aproximación espacial. Con esto es exacto: mismos segmentos, dos vectores de atribución de la misma longitud. |
| D3 | **Dependencia `shap>=0.44`** en el extra `xai` | Es la implementación de referencia y la corrección de KernelSHAP es sutil (kernel de Shapley, regularización, muestreo de coaliciones). Se verificó que hay wheels compatibles con el Python 3.14 del venv (`shap 0.52`, `numba 0.66`). Fallback documentado en §9-R1. |
| D4 | **Solo pipeline principal** (`outputs/main`) | Los docs sitúan SHAP en la etapa final. Los targets `compare` y `global` no tienen variante `-baselines`. |
| D5 | **LIME y Grad-CAM no cambian de comportamiento** | Los resultados de la Tabla 6.2 están publicados. `render_visual_explanation` gana un parámetro opcional `segments=None`; con `None` el camino de ejecución es el actual, sin diferencias. Un test de regresión lo fija. |
| D6 | **SLIC como segmentación canónica**, no quickshift | SLIC permite fijar `n_segments`, lo que hace el costo de KernelSHAP predecible y mantiene *k* chico (mejores estimaciones con el mismo presupuesto de evaluaciones). Quickshift devuelve un número variable, típicamente 100-200 para 224×224. **Consecuencia aceptada:** las regiones LIME del panel `compare` no son idénticas a las del panel `visual`. Se documenta en el pie de figura y se deja `shap.segmentation: slic\|quickshift` en config para poder reproducir las regiones de `visual` cuando haga falta. |
| D7 | **Línea base `black`** por defecto (`hide_color=0` de LIME) | Paridad con la convención de "ausencia" que LIME ya usa; comparar dos técnicas con nociones distintas de ausencia sería comparar otra cosa. `mean` y `blur` quedan disponibles en config. |
| D8 | **Nombres por artefacto, no por técnica** | `visual`, `fidelity`, `errors`, `compare`, `global`. Sobreviven a que se sumen o quiten técnicas dentro de cada panel — que es exactamente lo que este spec hace. |
| D9 | **`errors` como subcomando propio**, no un flag | Hoy `--errors-only` ya ignora `--sample-size`: son dos comandos distintos disfrazados de uno. |

## 4. CLI unificado

`scripts/pipeline/explain.py` con un parser padre que aporta los flags comunes (`--models`, `--run`, `--output-dir`, `--baseline`) y cinco subcomandos.

| Subcomando | Reemplaza a | Targets Make | Pipelines |
|---|---|---|---|
| `visual` | `explain_lime.py` | `explain-visual-{baselines,main}`, `modal-explain-visual-{baselines,main}` | ambos |
| `fidelity` | `explain_report.py` | `explain-fidelity-{baselines,main}`, `modal-explain-fidelity-{baselines,main}` | ambos |
| `errors` | `explain_report.py --errors-only` | `explain-errors-{baselines,main}`, `modal-explain-errors-{baselines,main}` | ambos |
| `compare` | — (nuevo) | `explain-compare-main`, `modal-explain-compare-main` | solo main |
| `global` | — (nuevo) | `explain-global-main`, `modal-explain-global-main` | solo main |

`explain_lime.py` y `explain_report.py` se eliminan. Su lógica se traslada sin cambios semánticos: `visual`, `fidelity` y `errors` deben producir exactamente los mismos artefactos que hoy, salvo dónde se escriben (§6).

**Única excepción, deliberada:** hoy los dos modos de `explain_report.py` escriben su summary en el mismo `explain_report/summary.{csv,json}`, así que correr `explain-errors` después de `explain-report` pisa el resumen del primero. Al separarse en dos subcomandos, cada uno escribe bajo su propio directorio y la colisión desaparece.

Se conserva la mecánica de selección de pipeline ya establecida: `--output-dir` en local, `--pipeline baselines|main` en Modal traducido por `_output_dir_args`. No se pasan rutas absolutas por `make` en Windows (MSYS las reescribe).

**Colisión de nombres:** convivirán `scripts/pipeline/explain.py` (el CLI) y `scripts/modal/explain.py` (el orquestador de Modal que lo invoca por subprocess). Son módulos distintos y no hay ambigüedad de import, pero el docstring de cada uno debe declarar cuál es cuál en su primera línea.

## 5. Módulos nuevos en `src/explainability/`

Archivos chicos, con una responsabilidad cada uno y testeables sin GPU, siguiendo el patrón de `stability.py`.

| Módulo | Responsabilidad | Interfaz principal |
|---|---|---|
| `segmentation.py` | Segmentación canónica determinista | `build_segments(image_np, n_segments, compactness, seed) -> np.ndarray` |
| `kernel_shap.py` | KernelSHAP sobre el vector binario de superpíxeles | `explain_with_kernel_shap(...) -> ShapExplanation` |
| `leaf_mask.py` | Máscara de vegetación y su sanidad | `leaf_mask(image_np) -> np.ndarray`, `mask_coverage(mask) -> float` |
| `agreement.py` | Acuerdo entre dos vectores de atribución sobre los mismos segmentos | `attribution_agreement(lime_weights, shap_values, top_k) -> dict` |
| `compare_report.py` | Compone y renderiza el panel comparado + sus sidecars | `render_comparison(...) -> dict` |
| `global_report.py` | Acumula la agregación por clase y escribe mapas y tabla | `accumulate(...)`, `write_global_report(...)` |

`visual_report.py` solo recibe el parámetro opcional `segments` (D5) y el cambio de nombre del directorio de salida (§6). `stability.py`, `gradcam.py` y `augmentation_preview.py` no se tocan; `agreement.py` reutiliza `mask_iou` de `stability.py`.

### 5.1 `kernel_shap.py`

Define `f: {0,1}^k → R`, donde `z_i = 1` deja visible el superpíxel *i* y `z_i = 0` lo reemplaza por la línea base (D7). La reconstrucción de imágenes y su evaluación se hacen en lotes de `shap.batch_size` reutilizando `build_predict_fn` de `visual_report.py` — el mismo `predict_fn` que ya usa LIME, así que ambas técnicas ven exactamente el mismo preprocesamiento.

- **Clase explicada:** `target_idx = argmax(predict_fn(image))`. Es el mismo criterio que `render_visual_explanation` (`explanation.top_labels[0]`), de modo que LIME y SHAP explican la misma clase.
- **Línea base:** vector `z = 0` (imagen enteramente enmascarada). Referencia única, no una distribución.
- **Aditividad:** `Σφ = f(1) − f(0)`, verificable y verificada en tests.
- **Salida:** `ShapExplanation` con `values` (uno por segmento), `expected_value`, `target_idx`, `target_label` y `n_evals`.

### 5.2 `agreement.py`

Tres métricas sobre los mismos segmentos, todas en el sidecar y en el summary:

| Métrica | Definición |
|---|---|
| `iou_topk` | IoU entre el conjunto de los *k* segmentos con mayor peso **positivo** según LIME y los *k* mayores según SHAP, con `k = lime.num_features` |
| `spearman` | Correlación de rangos entre el vector completo de pesos LIME y el de valores SHAP (`scipy.stats.spearmanr`, ya disponible vía scikit-learn) |
| `sign_agreement` | Fracción de segmentos donde ambos métodos coinciden en el signo |

`iou_topk` responde "¿coinciden en qué mirar?"; `spearman`, "¿coinciden en el orden de importancia?"; `sign_agreement`, "¿coinciden en si empuja a favor o en contra?". Las tres pueden divergir y esa divergencia es el hallazgo, no un error.

### 5.3 `leaf_mask.py`

Índice de exceso de verde (`ExG = 2G − R − B`) sobre la imagen normalizada, umbralizado con Otsu vía OpenCV (`opencv-python-headless` ya es dependencia base, no suma nada al extra `xai`).

**Riesgo asumido y su mitigación.** La heurística puede fallar en hojas cloróticas — justamente `nitrogen_deficiency`, una de las clases críticas. La respuesta no es confiar y tampoco descartar la métrica, sino instrumentarla:

- cada imagen persiste su `mask_coverage` (fracción de píxeles clasificados como hoja);
- las imágenes con cobertura degenerada (`< 0.05` o `> 0.95`) se excluyen del ratio hoja/fondo y se cuentan aparte como `n_mask_rejected`;
- el resumen global reporta `mean_mask_coverage` y `n_mask_rejected` **por clase**, y marca el ratio de una clase como no confiable cuando `n_mask_rejected / n > 0.3`.

Así el número o es confiable o se declara no confiable, pero nunca miente en silencio.

## 6. Artefactos por run

Los directorios se renombran junto con los targets: `lime_visual/` mentía por la misma razón que `explain-lime`.

```
<run_dir>/explain_visual/<stem>__true-<label>.png        # LIME + Grad-CAM   (antes lime_visual/)
<run_dir>/explain_visual/<stem>__true-<label>.{json,npy} # sidecars, sin cambios
<run_dir>/explain_fidelity/<stem>__...png                # paneles           (antes lime_report/)
<run_dir>/explain_fidelity/summary.{csv,json}            #                   (antes explain_report/)
<run_dir>/explain_errors/<stem>__...png                  #                   (antes lime_errors/)
<run_dir>/explain_errors/summary.{csv,json}
<run_dir>/explain_compare/<stem>__true-<label>.png       # Original | LIME | SHAP | Grad-CAM
<run_dir>/explain_compare/<stem>__true-<label>.json      # pesos LIME, valores SHAP, acuerdo
<run_dir>/explain_compare/<stem>__true-<label>.npy       # segmentos compartidos
<run_dir>/explain_compare/summary.{csv,json}
<run_dir>/explain_global/<clase>_attribution_map.png
<run_dir>/explain_global/global_summary.{csv,json}
```

Los runs ya generados conservan sus directorios viejos: son artefactos, no código, y no hay migración que hacer. `train_baselines.py --lime` escribe vía `explain_model_visual`, así que hereda el nuevo nombre sin cambios propios.

### 6.1 `explain_compare/summary.csv`

Una fila por `(label, correct)`, con `n`, `mean_pred_prob`, `mean_shap_dispersion`, `mean_iou_topk`, `mean_spearman`, `mean_sign_agreement`. Espeja la forma del summary de `fidelity`, reutilizando `explanation_dispersion` para la concentración.

### 6.2 `explain_global/`

Por cada imagen de la muestra se construye un mapa per-píxel asignando a cada píxel el valor SHAP de su segmento, normalizado por `max|φ|` de esa imagen para que las imágenes sean comparables entre sí. El mapa de clase es la media de `|·|` sobre las imágenes de esa clase, renderizado sobre la grilla del run.

`global_summary.csv` tiene una fila por `(label, correct)`:

| Columna | Significado |
|---|---|
| `n` | imágenes agregadas |
| `n_mask_rejected` | descartadas del ratio por cobertura degenerada |
| `n_ratio_undefined` | máscara válida pero sin atribución positiva que repartir (`Σφ⁺ <= 0`) |
| `mean_leaf_attribution_ratio` | `Σφ⁺` sobre píxeles de hoja / `Σφ⁺` total |
| `mean_mask_coverage` | fracción media de píxeles de hoja |
| `mean_abs_attribution` | magnitud media de atribución |
| `mean_dispersion` | concentración de la explicación |
| `ratio_reliable` | `false` si `(n_mask_rejected + n_ratio_undefined) / n > 0.3` |

> Nota (post-implementación, Tarea 8): `n_ratio_undefined` se trackea separado de `n_mask_rejected`
> porque son diagnósticos distintos del mismo síntoma - máscara de vegetación fallando vs. modelo sin
> atribución positiva -, aunque ambos entran en la misma condición de `ratio_reliable`.

La muestra sale de `predictions.csv` (balanceada por clase, `shap.global_sample_size`), que es lo que permite el desglose acierto/error.

## 7. Configuración

Bloque nuevo en `config/dataset.yaml`, hermano de `lime:` y `gradcam:`:

```yaml
shap:
  segmentation: slic      # slic | quickshift  (ver D6)
  n_segments: 50          # objetivo de superpíxeles SLIC
  compactness: 10.0
  nsamples: 2048          # evaluaciones de KernelSHAP por imagen
  batch_size: 128         # imágenes por forward pass
  background: black       # black | mean | blur  (ver D7)
  images_per_class: 5     # panel comparado
  global_sample_size: 30  # por clase, para la agregación global
  seed: 42
```

`images_per_class` se queda en 5 para que una corrida local en CPU siga siendo viable (§8). Los overrides puntuales viajan por CLI (`--nsamples`, `--sample-size`) sin tocar el YAML, igual que hace hoy `--num-samples` en `fidelity`.

## 8. Costo

Estimación para ShuffleNetV2-x1.0 en A10 con `nsamples=2048` y `batch_size=128`: ~5 s/imagen, dominado por la reconstrucción de máscaras en NumPy más que por el forward pass.

| Comando | Imágenes | Tiempo estimado (A10) | Timeout Modal |
|---|---|---|---|
| `compare` | 5/clase × 9 = 45 | ~15 min (incluye LIME) | 1 h |
| `global` | 30/clase × 9 = 270 | ~25 min | 3 h |

En CPU local el factor es ~10×, de ahí el default conservador de `images_per_class`. Ambos comandos entran holgados en los timeouts ya definidos en `scripts/modal/explain.py`.

## 9. Riesgos

| # | Riesgo | Mitigación |
|---|---|---|
| R1 | `shap` no instala o falla en Python 3.14 | El primer paso del plan es `pip install -e ".[xai]"` + un import de humo. Si falla, `kernel_shap.py` se implementa en casa (regresión lineal ponderada por el kernel de Shapley, ~60 líneas, precedente: `gradcam.py`). **La interfaz del módulo no cambia**, así que el resto del spec se sostiene igual. |
| R2 | ExG+Otsu falla en hojas cloróticas | Instrumentado, no asumido: `mask_coverage`, `n_mask_rejected`, `ratio_reliable` (§5.3) |
| R3 | La migración del CLI rompe LIME | Test de regresión que fija la neutralidad de `render_visual_explanation` sin `segments` (§10) |
| R4 | KernelSHAP con `nsamples` insuficiente da atribuciones ruidosas | `nsamples=2048` contra `k=50` da ~40 evaluaciones por superpíxel. El test de aditividad detecta degradación gruesa; la tolerancia del test acota el error aceptable. |
| R5 | Las regiones LIME de `compare` no coinciden con las de `visual` | Consecuencia conocida de D6, declarada en el pie de figura y reversible vía `shap.segmentation: quickshift` |

## 10. Pruebas

`tests/explainability/` (directorio nuevo), todo sin GPU con un modelo dummy determinista:

| Test | Qué fija |
|---|---|
| `test_kernel_shap.py::test_additivity` | `Σφ ≈ f(x) − expected_value` dentro de tolerancia |
| `test_kernel_shap.py::test_matches_exact_shapley` | Contra los valores de Shapley enumerados por fuerza bruta sobre un modelo sintético de 4 superpíxeles |
| `test_kernel_shap.py::test_deterministic` | Dos corridas con la misma seed dan valores idénticos — el contraste con LIME es el argumento central de los docs, y merece un test que lo respalde |
| `test_segmentation.py` | `build_segments` es determinista y respeta `n_segments` aproximadamente |
| `test_agreement.py` | Casos conocidos: vectores idénticos → acuerdo perfecto; opuestos → `sign_agreement = 0` |
| `test_leaf_mask.py` | Imagen sintética verde sobre fondo, más el caso degenerado que dispara `n_mask_rejected` |
| `test_global_report.py` | Agregación y `ratio_reliable` sobre entradas conocidas |
| `test_visual_report_regression.py` | **Neutralidad de LIME:** `render_visual_explanation` sin `segments` produce el mismo resultado que antes del cambio |

## 11. Archivos afectados

**Nuevos**

- `scripts/pipeline/explain.py`
- `src/explainability/{segmentation,kernel_shap,leaf_mask,agreement,compare_report,global_report}.py`
- `tests/explainability/` (6 archivos de test + `__init__.py`)
- `specs/plans/2026-07-27-shap-pipeline-principal.md` (plan de implementación)

**Modificados**

- `Makefile` — renombre de targets, targets `compare` y `global`
- `scripts/modal/explain.py` — renombre de funciones + `explain_compare` / `explain_global`
- `src/explainability/visual_report.py` — parámetro `segments` opcional, nombre del directorio
- `config/dataset.yaml` — bloque `shap:`
- `pyproject.toml` — `shap>=0.44` en el extra `xai`
- `src/training/common.py`, `src/training/artifacts.py` — docstrings que nombran los scripts viejos
- `CLAUDE.md`, `README.md`, `LOCAL.md`, `docs/es/pipeline/interpretabilidad.md`, `docs/es/pipeline-baselines/interpretabilidad.md`, `docs/es/deployment/modal.md`

**Eliminados**

- `scripts/pipeline/explain_lime.py`, `scripts/pipeline/explain_report.py`

**Renombrado**

- `.claude/skills/corn-lime-explainability/` → `.claude/skills/corn-xai/`, con el flujo de SHAP y los nombres nuevos
