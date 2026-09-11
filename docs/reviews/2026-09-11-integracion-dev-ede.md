# Revisión de la integración `dev-ede` → `dev-abner`

Fecha de revisión: 2026-09-11

Rango revisado: `origin/dev-abner..HEAD`

Commits: `95659bc`, `a582bb8`, `8103fa3`, `3d8b7b9`, `8a19ff9`, `5a5e6f7`

Magnitud: 28 archivos, 2,253 inserciones, 119 eliminaciones y 8 imágenes nuevas.

## Dictamen

No se recomienda publicar ni fusionar este rango como una evaluación reproducible hasta
corregir los hallazgos P1. La integración amplía de forma valiosa los flujos de HPO,
ensamble, validación cruzada, fairness y ejecución en Modal, pero actualmente puede producir
métricas mal etiquetadas, certificar una auditoría sin un checkpoint entrenado y presentar
conclusiones documentales incompatibles con sus propios artefactos.

## Cambios incorporados

- HPO con Optuna: persistencia de artefactos al finalizar cada trial y ejecución en Modal.
- Ensamble: mejor descubrimiento de `best.pth`, inferencia puntual por soft voting y soporte
  para procesar directorios de imágenes.
- Validación cruzada: cálculos NumPy para accuracy, macro-F1 y matriz de confusión.
- Fairness: métricas por entorno, brechas, FNR por clase, auditoría dual de oclusión central
  y periférica, Grad-CAM, gráficas, salida JSON/CSV y pruebas unitarias iniciales.
- Modal/Makefile: nuevos entrypoints para tuning, ensamble, K-fold y fairness, además de
  ejecución desacoplada mediante `DETACH=1`.
- Documentación: páginas de entrenamiento, optimización, ensamble y evaluación, con nuevas
  entradas en la navegación de VitePress y ocho figuras versionadas.

### Actualización tras los dos últimos commits

`8a19ff9` y el merge `5a5e6f7` corrigieron la principal inconsistencia numérica del informe
raíz: `FAIRNESS_REPORT.md` ahora usa las mismas 5,015 muestras y métricas desagregadas que la
página web. También agregaron accuracy, caída de accuracy, tasa de cambio de acierto a error,
un control inverso de oclusión y rechazo explícito de modos de máscara desconocidos.

Estos cambios mejoran la observabilidad, pero no resuelven la validez de la comparación por
clases ni la carga opcional del checkpoint. Además, la nueva interpretación causal de la
oclusión excede lo que el experimento implementado permite concluir; se detalla en P1-05.

## Hallazgos

### P1-01 — Precision y recall de K-fold contienen otras métricas

En `scripts/pipeline/cross_validate.py:364-367` y `:404-407`, `macro_precision` se asigna
directamente desde `macro_f1`, mientras que `macro_recall` se asigna desde `accuracy`:

```python
prec = f1_macro
rec = acc
```

Esto contamina `kfold_metrics.csv`, las estadísticas agregadas, el boxplot y
`hold_out_test_metrics` de `kfold_summary.json`. No es una aproximación matemáticamente
válida. Deben calcularse precision y recall por clase y promediarse, igual que ya hace
`_compute_classification_metrics_np()` en el módulo de fairness.

### P1-02 — La auditoría puede finalizar sin cargar un modelo entrenado

`scripts/pipeline/evaluate_fairness.py:298-305` continúa cuando no encuentra un checkpoint
y audita una cabeza de clasificación inicializada aleatoriamente sobre pesos base. Aun así,
el proceso genera JSON, CSV, figuras y termina como una auditoría exitosa. Además,
`strict=False` permite cargas parciales sin reportar claves faltantes o inesperadas.

Una auditoría cuantitativa debe fallar si no existe el checkpoint solicitado. También debe
cargar el `class_to_idx`, el tamaño de entrada y los splits desde el `summary.json` del mismo
run, y validar estrictamente la compatibilidad del estado.

### P1-03 — El resultado de fairness no permite la certificación que se le atribuye

La propia documentación indica que `lab` solo contiene 3 de las 9 clases
(`docs/es/pipeline/evaluacion.md:47-60`). Sin embargo:

- `src/analysis/fairness.py:60-77` asigna recall 0 a toda clase ausente y la incluye en el
  macro-promedio del subgrupo.
- `src/analysis/fairness.py:183-186` aplica la llamada regla del 80 % al cociente entre esos
  macro-F1 no comparables. Un cociente de F1 no es la definición estándar de *disparate
  impact ratio*.
- `src/analysis/fairness.py:158-166` devuelve `four_fifths_rule_passed=True` si solo existe
  un subgrupo. Si falta `environment`, el CLI convierte todo a `unknown`
  (`scripts/pipeline/evaluate_fairness.py:273-276`) y puede declarar paridad precisamente
  cuando no hay comparación posible.

La salida debe distinguir `passed`, `failed` y `not_evaluable`. Para comparar dominios debe
usarse el soporte común de clases y reportar por separado cobertura, soporte, intervalos de
confianza y brechas por clase. El cociente de macro-F1 puede conservarse como una razón de
rendimiento, pero no debe denominarse DIR ni interpretarse como prueba de ausencia de sesgo.

### P1-04 — El informe y el ejecutable aplican la regla del 80 % a ratios distintos

La actualización alineó correctamente la tabla de `FAIRNESS_REPORT.md:21-29` con las 5,015
muestras documentadas en la web. Sin embargo, el informe calcula el ratio sobre accuracy:

$$0.9417 / 0.9842 = 0.9568 \quad (\text{pasa})$$

El ejecutable sigue calculando `disparate_impact_ratio` sobre macro-F1 en
`src/analysis/fairness.py:175-186`:

$$0.2955 / 0.9298 = 0.3178 \quad (\text{falla})$$

Por tanto, el JSON generado y el documento pueden emitir veredictos opuestos para la misma
corrida. Además, llamar *disparate impact ratio* a cualquiera de estas razones de rendimiento
requiere justificar la adaptación: la regla de cuatro quintos se define sobre tasas de
selección, no sobre macro-F1 o accuracy de un clasificador multiclase. Debe elegirse y
nombrarse una métrica operacional, conservar la razón de F1/accuracy como análisis técnico y
evitar presentarla como certificación legal o ética.

### P1-05 — La oclusión dual no demuestra dependencia causal del fondo

`src/analysis/fairness.py:215-222` y ambos documentos equiparan el rectángulo central con
"la lesión" y el resto con "el fondo", pero el pipeline no usa segmentación ni anotaciones de
lesión para verificar esa correspondencia. Esto es especialmente problemático en hojas
sanas, lesiones fuera del centro y fotografías donde la hoja ocupa la periferia.

La geometría también está descrita incorrectamente. Los límites `0.2:0.8` de
`fairness.py:247-258` abarcan el 60 % de cada dimensión, es decir, solo el 36 % del área
($0.6 \times 0.6$). La oclusión central tapa 36 % de los píxeles y el control inverso tapa
64 %, no 60 % y 40 % respectivamente.

Hay otros factores que impiden el dictamen "Clever Hans confirmado":

- Las imágenes ya están normalizadas (`src/data/transforms.py:124-137`); escribir `0.0` crea
  el color medio de ImageNet en el espacio original y un borde rectangular abrupto fuera de
  distribución, no una máscara negra o neutra.
- La confianza registrada es el máximo softmax posterior a cada máscara, que puede
  corresponder a otra clase. No mide retención de la probabilidad de la clase verdadera ni de
  la predicción original.
- Los umbrales de `fairness.py:289-301` son constantes sin calibración, control aleatorio,
  máscaras de tamaño/posición comparables, intervalos de confianza ni prueba estadística.
- El resumen de `fairness.py:340-347` puede contradecir `overall_risk_level`: el riesgo se
  vuelve crítico desde ratio central 0.75 o por el control periférico, pero el texto solo
  acusa vulnerabilidad si el ratio central llega a 0.85. La expresión
  `100 - confidence_drop * 100` tampoco representa el porcentaje de comportamiento anclado.

Las caídas observadas justifican una hipótesis de sensibilidad al contexto y más pruebas, no
una atribución exclusiva al fondo. La auditoría debe emplear máscaras reales de hoja/lesión,
oclusiones aleatorias emparejadas y métricas por muestra y clase antes de emitir un veredicto
causal.

### P2-01 — El panel Grad-CAM imprime una conclusión que no calcula

`scripts/pipeline/evaluate_fairness.py:247-250` rotula siempre la tercera columna como
`Foco en Lesión Foliar`; el único criterio realmente evaluado es si la clase predicha coincide
con la etiqueta. El código no segmenta la hoja ni mide cuánta activación cae sobre lesión,
hoja o fondo. La figura versionada incluso muestra activación intensa en esquinas/fondo de
las muestras de laboratorio, en conflicto con `docs/es/pipeline/evaluacion.md:127-133`.

La página web también se contradice internamente: primero declara una dependencia crítica del
fondo (`:96-101`) y después afirma que Grad-CAM ignora por completo ese fondo (`:127-133`).

Adicionalmente, `evaluate_fairness.py:207` abre la imagen directamente con `PIL.Image.open`
en lugar de usar `load_and_normalize_image()`, infringiendo el punto de entrada único de
imagen definido por el proyecto y omitiendo la corrección EXIF compartida.

### P2-02 — `OUTPUT_ROOT` solo se respeta si el directorio ya existe

`src/config.py:34-44` ignora silenciosamente un `OUTPUT_ROOT` configurado cuando la ruta
todavía no existe. Esto rompe el caso normal de primera ejecución: los pipelines crean sus
directorios de salida después de resolver la raíz, pero ahora terminan escribiendo en
`PROJECT_ROOT/outputs` en vez del destino solicitado. Una variable explícita debe conservar
precedencia aunque el directorio vaya a crearse posteriormente; si es inválida, se debe
fallar con un mensaje claro, no cambiar de destino.

### P2-03 — La selección de runs en `predict.py` puede usar pesos distintos a los pedidos

En `scripts/pipeline/predict.py:116-132`, si `--run` no existe o falta `latest.json`, la
función cae al checkpoint más reciente encontrado por `rglob` en vez de fallar. Por tanto, el
comando puede producir una predicción con otro run sin informarlo.

En modo ensamble (`:234-253`) también se ignoran `--checkpoint`, `--run` e `--image-size`, y
se sobrescribe `class_to_idx` con el metadata de cada modelo sin comprobar que ambos mapeos
sean idénticos. Si los checkpoints provienen de perfiles o runs diferentes, el promedio
combina columnas que pueden representar clases distintas.

### P2-04 — La documentación afirma una conexión HPO → entrenamiento que no existe

`docs/es/pipeline/optimizacion.md:131-134` dice que `best_params.json` es consumido
automáticamente por `train.py`. `scripts/pipeline/train.py` no lee ese archivo: usa los
argumentos CLI o sus defaults. Esto también contradice
`docs/es/pipeline/entrenamiento.md:25-39`, donde se afirma que se aplicaron estrictamente los
parámetros ganadores aunque se documentan valores distintos para weight decay y warmup.

Debe agregarse una opción explícita `--best-params` al entrenamiento o documentarse el comando
completo con los valores transferidos manualmente.

### P2-05 — Defaults y flags de CLI no coinciden con lo documentado

- `Makefile:76` define `MODEL=efficientnet_b0`; por eso el condicional de `Makefile:391-396`
  hace que `make predict` use EfficientNet y no el ensamble anunciado como default.
- El mismo default hace que `modal-tune` priorice siempre `MODEL` y no la lista `MODELS`.
- `--run-gradcam` y `--run-shortcut-test` son `store_true` con `default=True`
  (`evaluate_fairness.py:87-99`), por lo que no son opt-in ni existe forma de desactivarlos.

Conviene separar variables por comando (`PREDICT_MODEL`, `TUNE_MODELS`) y usar pares
`--run-*`/`--no-*` o `BooleanOptionalAction`.

### P2-06 — Se documentan mitigaciones futuras como si ya estuvieran desplegadas

`docs/es/pipeline/evaluacion.md:104-117` presenta la segmentación foliar, el entrenamiento por
parches y el reemplazo de fondos como trabajo necesario. En la misma página, `:151-155`
afirma que la aplicación móvil ya ejecuta el segmentador local y la inferencia TFLite/ONNX.
Esta integración no está implementada ni es verificable en el repositorio revisado. Debe
separarse con claridad el estado `implementado`, `planificado` y `requerido antes de
despliegue`; lo mismo aplica a las afirmaciones de OOD, procesamiento local y privacidad del
informe raíz.

### P2-07 — Los nombres y signos documentados no coinciden con el artefacto JSON

En `FAIRNESS_REPORT.md:56,62` las caídas de confianza se escriben como `-0.0634` y `-0.1465`,
pero `fairness.py:281,312` define y exporta `confidence_drop = original - masked`, por lo que
los valores del JSON son positivos. El documento puede mostrar la dirección visualmente,
pero debe distinguirla del nombre y signo exactos del campo reproducible.

Además, `FAIRNESS_REPORT.md:103` usa la etiqueta `potassium`, mientras que la clase canónica
de `config/dataset.yaml` es `potassium_deficiency`. Estas diferencias dificultan comparar el
texto con `fairness_metrics.json` y con el mapeo de clases del checkpoint.

## Aspectos positivos

- El descubrimiento actualizado reconoce el layout real `<modelo>/<run_id>/best.pth` y
  `latest.json`, corrigiendo una incompatibilidad con los checkpoints que genera `fit()`.
- La persistencia por callback reduce la pérdida de evidencia de Optuna si una corrida larga
  se interrumpe.
- Los entrypoints de Modal reutilizan los CLI locales mediante `subprocess`, lo que reduce la
  divergencia entre ejecución local y remota.
- La inferencia por directorio conserva orden determinista y reutiliza el cargador canónico
  de imagen.
- La actualización de fairness incorporó accuracy, caída de accuracy, *flip rate*, ambos
  sentidos de oclusión y error explícito para modos desconocidos.
- Las nuevas pruebas cubren las claves y tipos básicos de la auditoría dual, aunque no
  verifican el contenido/área de las máscaras, los valores calculados, los umbrales de
  diagnóstico ni las contradicciones entre resumen y riesgo. También faltan pruebas de
  integración para checkpoint/metadata y casos no evaluables.

## Verificación realizada

- `git diff --check 3d8b7b9..HEAD`: detectó espacios finales en `FAIRNESS_REPORT.md:123`,
  `:125` y `:127`. La revisión del rango completo también detectó espacios finales en
  `docs/es/pipeline/optimizacion.md`.
- `python3 -m compileall -q src scripts tests`: correcto; no hay errores de sintaxis Python.
- Suite Pytest: no ejecutable en este checkout porque no existe el entorno `venv` y el Python
  del sistema no incluye `pytest`, NumPy, pandas, PyTorch, matplotlib ni scikit-learn.
- VitePress: `npm run docs:build` se detiene en `git fetch --unshallow` porque el repositorio
  ya es completo; además no está instalado `node_modules`. El primer fallo pertenece al
  script de build preexistente, no a este rango.
- Codebase Memory: la indexación completa y la rápida extrajeron los símbolos, pero ambas
  fallaron al persistir el grafo en la fase `dump`; por ello el análisis continuó sobre el
  diff de Git y las dependencias directas.

## Orden recomendado de corrección

1. Corregir las cuatro métricas K-fold y agregar pruebas con clases desbalanceadas.
2. Hacer obligatorio un checkpoint coherente con su `summary.json` en fairness y ensamble.
3. Rediseñar la comparación `lab`/`real` sobre soporte común, estados no evaluables y una
   terminología que no confunda razones de rendimiento con impacto dispar.
4. Sustituir la oclusión rectangular por controles basados en máscaras reales, registrar la
   probabilidad de clases comparables y calibrar el veredicto con controles e incertidumbre.
5. Regenerar informes y figuras desde JSON/CSV versionados, con run ID y commit de origen.
6. Alinear selección de runs, mapeos de clases, defaults del Makefile y transferencia de HPO.
7. Ejecutar la suite completa y el build de VitePress en el entorno de desarrollo del
   proyecto antes de publicar.
