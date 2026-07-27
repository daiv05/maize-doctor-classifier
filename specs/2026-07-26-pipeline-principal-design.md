# Diseño: Pipeline principal de entrenamiento (`train.py`)

**Fecha:** 2026-07-26
**Estado:** aprobado, pendiente de plan de implementación
**Fuentes:** `reports/firts-phase/documentation_first_phase.tex` (§5.5, Tabla 5.5; §6.4; §7), `previous_analysis.md`, `experiments/clahe/`

## 1. Objetivo y alcance

Implementar el loop de entrenamiento de `scripts/pipeline/train.py`, hoy un `TODO` sobre andamiaje de datos ya funcional. El pipeline principal entrena **una** arquitectura sobre el **100% del dataset** (31 623 imágenes) con los mecanismos de optimización y regularización que la Tabla 5.5 del reporte prescribe, y produce las métricas que la Etapa 1 dejó pendientes.

### Fuera de alcance

- **Resolver el cuello de botella de potasio.** El reporte concluye que es un problema de datos (266 imágenes, F1 0.49–0.62), no de capacidad. Este plan lo mitiga con las herramientas de entrenamiento disponibles y documenta el techo alcanzable, pero no incluye recolección ni generación de datos.
- **SHAP.** La Tabla 5.5 lo prescribe para esta etapa, pero es post-hoc y desacoplado del entrenamiento, igual que LIME/Grad-CAM. Merece su propio spec.
- **Exportación a TFLite y benchmark en dispositivo.** Trabajo futuro declarado en §7 del reporte.
- **Cambios de comportamiento en `train_baselines.py`.** Los resultados de la Tabla 6.2 están publicados; el refactor debe ser semánticamente neutro.

## 2. Contexto que motiva las decisiones

Pasar del perfil baseline (cap 1500, 10 020 imágenes) al dataset completo **empeora el desbalance de 5.6× a 32.9×**:

| Clase | n total | n train (70%) | ratio vs. máx | minoritaria (>4×) |
|---|---:|---:|---:|:--:|
| healthy | 8 744 | 6 120 | 1.0 | no |
| northern_corn_leaf_blight | 6 830 | 4 781 | 1.3 | no |
| lethal_necrosis | 6 415 | 4 490 | 1.4 | no |
| fall_armyworm | 4 858 | 3 400 | 1.8 | no |
| common_rust | 2 256 | 1 579 | 3.9 | no |
| gray_leaf_spot | 1 119 | 783 | 7.8 | sí |
| phosphorus_deficiency | 612 | 428 | 14.3 | sí |
| nitrogen_deficiency | 523 | 366 | 16.7 | sí |
| potassium_deficiency | 266 | 186 | 32.9 | sí |

**Total train: 22 133 imágenes (~691 batches/época @ 32).**

Esto invalida una prescripción del reporte. La Tabla 5.5 pide "sampler + pérdida ponderada" para el pipeline principal, pero a 32.9× el `WeightedRandomSampler` repetiría cada imagen de potasio **13.2 veces por época** mientras `healthy` vería solo el 40% de las suyas. Ponderar la loss encima de eso es la triple compensación que el propio §5.2 del reporte dice evitar.

## 3. Decisiones de diseño

| # | Decisión | Justificación |
|---|---|---|
| D1 | **Loss ponderada, sampler desactivado** | Evita la doble compensación. Beneficio secundario: sin `replacement=True`, cada época ve el **100%** de las 22 133 imágenes únicas en vez del ~63% que cubre el sampler. |
| D2 | **`sqrt_inverse` como fórmula de pesos por defecto** | El inverso puro da un rango de 32.9× entre potasio y healthy; con 186 imágenes de train eso invita a sobreajustarlas y degradar las clases grandes. La raíz comprime a 5.7×, corrigiendo en la dirección correcta sin dominar el gradiente. |
| D3 | **Una arquitectura: ShuffleNetV2-x1.0 por defecto** | Comparar arquitecturas ya lo hicieron los baselines. §7 favorece ShuffleNet para despliegue (5 MB, menor loss). `--models` sigue disponible para re-correr otra. |
| D4 | **Cosine + warmup lineal, early stopping patience 8** | Determinista y reproducible, sin depender del ruido de validación como `ReduceLROnPlateau` — coherente con la prioridad de reproducibilidad del proyecto (§6.5). |
| D5 | **Full fine-tuning directo, sin fases** | `previous_analysis.md` sugiere congelar el backbone, pero ese hallazgo se hizo sobre **1 400** imágenes. A 22 133 el argumento no aplica; warmup + weight decay bastan. |
| D6 | **CLAHE opt-in, apagado por defecto, con corrida A/B** | Ver §7. Su efecto sobre macro-F1 es desconocido y activarlo por defecto invalidaría la comparación con la Tabla 6.2. |

## 4. Arquitectura

`train_baselines.py` tiene un loop funcional de ~250 líneas; `train.py` tiene el andamiaje de datos y un `TODO`. Copiar el loop duplicaría código que luego diverge. La solución es **extraer lo compartido a `src/training/`, dejando ambos scripts como ensambladores delgados**.

| Módulo nuevo | Responsabilidad | Consumidores |
|---|---|---|
| `src/training/loop.py` | `run_epoch()`, `fit()`: bucle epoch/train/val, historial, best-checkpoint | ambos |
| `src/training/optim.py` | `build_optimizer()`, `build_scheduler()` (cosine+warmup), `EarlyStopping` | `train.py` |
| `src/training/losses.py` | `build_criterion()`: CrossEntropy + class weights + label smoothing | ambos |
| `src/training/evaluation.py` | calibración (ECE/Brier), desglose por `environment`, agrupado N/P/K | ambos |
| `src/training/artifacts.py` | escritura de `summary.json`, `predictions.csv`, matriz de confusión | ambos |

`src/training/common.py` se mantiene sin cambios: ya cubre `run_id`, device y rutas de run.

**Invariante crítico:** `fit()` recibe scheduler y early stopping como opcionales (`None`). Con `build_criterion()` sin pesos y `fit()` sin scheduler ni early stopping, el comportamiento debe reducirse **exactamente** al código actual de baselines. Así la diferencia entre etapas queda expresada en configuración, no en código duplicado — que es literalmente lo que dice §5.5 ("la infraestructura es común; lo que cambia es cuánto se afina el loop").

`evaluation.py` se separa porque sus métricas se calculan sobre `predictions.csv`, no dentro del loop: son testeables sin GPU y re-ejecutables sobre runs de baselines ya existentes sin re-entrenar.

## 5. Configuración del loop principal

| Elemento | Baselines (congelado) | Pipeline principal |
|---|---|---|
| Datos | 9 clases, cap 1500 (10 020) | 9 clases, sin cap (31 623) |
| Balanceo | sampler + augment. minoritaria | **loss ponderada + augment. minoritaria; sampler OFF** |
| Loss | CrossEntropy plana | CrossEntropy ponderada (`sqrt_inverse`) + label smoothing 0.1 |
| LR | fijo 1e-4 | warmup lineal 3 ép. → cosine → 1e-6 |
| Parada | 30 épocas fijas | máx. 60, early stopping patience 8 sobre `val_macro_f1` |
| Optimizador | AdamW | AdamW + `clip_grad_norm_(1.0)` |
| Modelo | 3 arquitecturas | ShuffleNetV2-x1.0 |

**Flags nuevos:** `--scheduler {cosine,none}`, `--warmup-epochs`, `--patience`, `--class-weights {sqrt_inverse,inverse,none}`, `--label-smoothing`, `--clip-grad-norm`, `--clahe`.

**Capas de balanceo resultantes: dos, no tres.** `compute_minority_classes` seguirá marcando `gray_leaf_spot`, N, P y K como minoritarias (>4×), así que esas cuatro reciben augmentation agresivo **además** de la loss ponderada. Es coherente con §5.2 — la tercera capa era el sampler, ahora desactivado.

## 6. Métricas y artefactos

Todo se calcula en `evaluation.py` sobre `predictions.csv`, sin GPU.

**`test_calibration.json`** — ECE (15 bins uniformes, ~20 líneas a mano), Brier binario del evento "acertó/no acertó" (clave `brier_binary_hit`), y confianza media desagregada en aciertos vs. fallos.

> Corregido durante la implementación: el Brier **multiclase** no es calculable con los datos disponibles, porque `predictions.csv` persiste `pred_prob` como escalar (la confianza de la clase predicha), no el vector completo de probabilidades por clase. La clave se nombra `brier_binary_hit` para no inducir a error a quien lea el JSON. Esta última conecta con el hallazgo de §6.4 (aciertos 0.987 / fallos 0.914): si el label smoothing funciona, el gap debe ensancharse.

**`test_by_environment.csv`** — macro-F1, accuracy y F1 por clase, desglosados en `lab` / `real`. La columna `environment` ya existe en los CSV de splits. Verifica el riesgo de shortcut learning en `common_rust` (95% lab).

> **Limitación declarada:** con ~16 imágenes reales de `common_rust` en test, esa fila es indicativa, no concluyente. Se reporta con el `n` visible al lado.

**`test_grouped_metrics.json`** — macro-F1 colapsando N/P/K en una sola clase "deficiencia nutricional", la alternativa de producto que propone §6.4. Cuantifica cuánto sube la métrica si el sistema no intenta separarlas. Dado que el 97% de los errores N/P/K se quedan dentro del bloque, se espera un salto grande.

**Sin cambios:** `best.pth`, `last.pth`, `train_history.csv`, `predictions.csv`, `test_confusion_matrix.csv`, `test_classification_report.csv`, `update_latest_pointer()`, todo en `outputs/main/<modelo>/<run_id>/`. `summary.json` gana los campos de la config nueva (scheduler, warmup, patience, class_weights, label_smoothing, clahe, época de parada).

## 7. CLAHE

### Evidencia medida

Sobre las 23 imágenes de `experiments/clahe/input/` (common_rust, gray_leaf_spot, northern_corn_leaf_blight; todas de campo real), a 224×224:

| Métrica | Valor | Lectura |
|---|---|---|
| Desplazamiento de hue | **0.81° sobre 180** | Prácticamente nulo. Era la objeción que hundiría la idea: el hue es la señal diagnóstica de las deficiencias. El experimento la neutraliza aplicando CLAHE solo al canal L de LAB. |
| Desplazamiento de saturación | 18.6 sobre 255 | Moderado. |
| Contraste local (std de L) | 47.1 → 53.4 (**+13%**) | Ganancia real. |
| Latencia | **0.76 ms/img** (CPU) | Irrelevante frente al forward pass. |

Los paneles muestran el beneficio concentrado en imágenes con iluminación irregular: en gray_leaf_spot el histograma original está apilado contra los tonos claros (sobreexposición solar) y CLAHE lo redistribuye, recuperando textura. En blight, las lesiones necróticas ganan separación del tejido sano. Ese es el perfil del dataset: **28 500 de 31 623 imágenes son de campo real**, y el test es mayoritariamente real.

### Riesgos

- En `common_rust_16027374` el fondo de tierra roja gana tanto contraste como la hoja. `common_rust` es justo la clase con domain shift (95% lab), así que CLAHE podría **reforzar** el atajo del fondo en vez de mitigarlo.
- Las imágenes de lab ya están bien expuestas: ahí CLAHE añade ruido sin aportar señal.
- Los backbones preentrenados en ImageNet nunca vieron imágenes ecualizadas así — hay riesgo de *distribution mismatch* con los pesos preentrenados.

### Decisión

**Opt-in, apagado por defecto, validado con corrida A/B.** A diferencia del resto del spec, CLAHE no está prescrito por el reporte y su efecto sobre macro-F1 es desconocido: +13% de contraste no se traduce automáticamente en mejor clasificación.

- **Dónde:** `CornCLAHETransform` en `src/data/transforms.py`, aplicado **antes** del Resize y compartido por los cuatro pipelines (train/minority/val/test). Es **preprocesamiento determinista, no augmentation**: aplicarlo solo en train crearía un desajuste train/test garantizado.
- **Activación:** flag `--clahe` + bloque en `config/dataset.yaml` (`clip_limit: 2.0`, `tile_grid: 8`).
- **Validación:** dos corridas idénticas del pipeline principal, con y sin CLAHE, comparando macro-F1 y el desglose por environment de §6. **Predicción falsable:** si CLAHE ayuda, el salto debe verse en `real` y no en `lab`.
- **Dependencia:** `opencv-python` pasa de extra `[experiments]` a dependencia del pipeline.

> Activar CLAHE por defecto **invalidaría los checkpoints de baselines** para comparación directa, porque cambia la distribución de entrada. El default apagado es lo que mantiene la Tabla 6.2 comparable.

## 8. Verificación

1. **Regresión de baselines (la crítica).** Test que corre 1 época sobre un subconjunto mínimo con seed fija, antes y después del refactor, comparando `train_history.csv`. Números no idénticos = refactor roto. Esto protege la Tabla 6.2 ya publicada.
2. **Unitarios sin GPU:** pesos de clase (suma y ratio esperados), scheduler (LR sube en warmup, baja en cosine), early stopping (dispara a las 8 épocas sin mejora), ECE (contra un caso calculado a mano), CLAHE (hue shift < 2° sobre imágenes de prueba).
3. **Smoke run:** 2 épocas con `--models shufflenet_v2_x1_0` sobre splits baseline, verificando que los 3 artefactos nuevos se escriben y son parseables.

## 9. Ejecución en la nube

`scripts/modal/train.py` necesita una función `train_main` espejo de `train_baselines` (~30 líneas), orquestando `train.py` por subprocess con el mismo patrón de Volumes.

**Estimación de coste:** el baseline capado corre ~12 min/modelo (7 014 imágenes × 30 épocas). El principal es 3.2× más datos y hasta 2× más épocas → **2–4 h** por corrida en A10. Entra holgado en el techo de 14 h ya configurado. Conviene calibrar con un run corto antes de lanzar el completo.

Con la corrida A/B de CLAHE, el presupuesto total es de **2 corridas** (~4–8 h).

## 10. Riesgos abiertos

| Riesgo | Mitigación |
|---|---|
| El refactor cambia silenciosamente los resultados de baselines | Test de regresión (§8.1), bloqueante |
| `sqrt_inverse` resulta insuficiente para potasio | `--class-weights inverse` disponible para contrastar sin tocar código |
| CLAHE degrada por mismatch con pesos ImageNet | Default apagado; la A/B lo detecta antes de adoptarlo |
| Early stopping dispara demasiado pronto por ruido en val | patience 8 es holgado; `train_history.csv` permite auditarlo post-hoc |
| Potasio sigue siendo el techo del macro-F1 | Declarado fuera de alcance; la métrica agrupada N/P/K cuantifica el techo alcanzable sin más datos |
