# Registro de reparación integral de Doctor Maíz

Inicio: 2026-09-11; continuación: 2026-09-12. Base: `d501d97`, rama `dev-abner`.
Se trabaja sobre el checkout, no sobre los ZIP anteriores. Se preservaron las dos corridas
añadidas por el usuario en `outputs/outputs-11092026`. No se hicieron commits, pushes,
merges, despliegues ni trabajos remotos de pago.

Alcance autorizado: los 13 grupos de `Prompt_reparacion_integral_Doctor_Maiz.md`.
Este registro separa correcciones de código de hipótesis agronómicas: pasar tests no
certifica un modelo ni resuelve la revisión humana de segmentación.

## Estado por problema

`verificado`: ejecución pertinente para el comportamiento indicado; `implementado`:
falta una comprobación específica; `pendiente`: falta trabajo; `bloqueado`: recurso
identificado. Los pendientes de una fila no se consideran resueltos por su parte verificada.

| ID | Problema y dependencia | Solución y archivos principales | Verificación y estado |
|---|---|---|---|
| P1-01 | Imagen sustituida, predicción asociada a otra fila | `src/data/{dataset,identity}.py`: ID de ruta, hash de bytes, error explícito; `training/artifacts.py`, evaluación/exportación asocian por ID real del sampler | **Verificado localmente**: ausente seguida de válida, corrupción, Subset, shuffle, filtrado y duplicación de IDs. No se reparan retrospectivamente CSV sin identidad observada |
| P1-02 | Resolver otra corrida o usar pesos aleatorios / P1-01 | `training/runs.py`: referencia explícita estricta, hash de checkpoint, arquitectura, clases, contrato; ensemble con integrantes, pesos y hashes; predict/fairness/export/XAI usan el contrato | **Verificado localmente**: run inexistente, tensor incompatible, clases reordenadas. Carga estricta de ambos checkpoints recibidos: válida, pero la prueba con ceros es solo estructural |
| P1-03 | Métricas copiadas y última época / P1-01 | `training/loop.py`: precision/recall/F1/accuracy independientes, restaura CPU best incluso sin run_dir; `pipeline/cross_validate.py`: artefactos por fold y evaluación test opt-in | **Verificado localmente**: métricas desbalanceadas y mejor época anterior a la última; integración entrena y restaura. CV grande nueva: pendiente, no se atribuye el bug al CV independiente de Etapa 2 |
| P1-04 | Cachés por existencia y holdout modificable / P1-01 | `data/provenance.py`, `etapa_2/stage2_experiments.py`: contratos ordenados, hashes de datos/backbone/preproceso/matriz, splits inmutables, holdout y recuperación congelados; auditoría exacta/pHash, grupos revisados | **Verificado localmente y sobre el dataset real**: 373 pares candidatos, una fotografía confirmada cruzando train/val main, cuatro conflictos healthy/FAW. Nueva versión de 33 429 imágenes y herencia Etapa 2 verificadas. **Pendientes** otros 372 pares y procedencia planta/sesión; no se acredita corpus sin fuga |
| P1-05 | Preprocesos distintos / P1-02 | `data/transforms.py`, `data/segmented.py`: EXIF/RGB, H×W, stretch bilinear, antialias, normalización, CLAHE y segmentador; XAI/export/OOD/Etapa 2 reutilizan fábrica; embeddings con tamaño propio por backbone | **Verificado localmente**: tensor EXIF+CLAHE igual en entrenamiento/XAI, exportación ONNX real. Impacto agronómico: piloto de desarrollo separado; no se presupone mejora |
| P1-06 | Fairness con clases ausentes y conclusiones causales / P1-02 | `analysis/fairness.py`: soportes, métricas indefinidas null, Wilson para accuracy/recall, comparación solo de clases comunes; oclusiones 36/64% y control aleatorio de igual área, probabilidad de clase fija | **Verificado localmente** y **recálculo real** de predicciones guardadas. Sin umbral arbitrario de aprobación ni conclusión causal. Oclusiones antiguas sin datos por imagen: no recuperables por simple recálculo |
| P1-07 | Máscaras casi completas/ambiguas / P1-05 | `segmentation/leaf_processor.py`: estados y razones, geometría, ambigüedad y retención; comparación explícita con compuerta externa, no reutilizada por contrato distinto | **Verificado localmente y piloto real de 288 imágenes**: 31 aceptadas, 210 inciertas, 47 rechazadas. Calibración y transferencia de etiqueta **pendientes de revisión humana**; alerta de borde dominante, no equivale a lesión preservada |
| P1-08 | Colisiones, reanudación y redivisión / P1-01,07 | `pipeline/segment_dataset.py`: nombre derivado de ID, PNG atómico, recibos y hashes, fuente/checkpoint/runtime/perfil/fallback por imagen, candidatos de preview estratificados; Modal hereda splits originales | **Verificado localmente**: jpg/png homónimos, reanudación, checkpoint cambiado, conservación de partición, fallback contabilizado. Ejecución masiva Modal: no realizada |
| P1-09 | Archivo creado confundido con entregable / P1-02,05 | `export/{common,runtime,evaluate}.py`, `sync_mobile_model.py`: estados reales, paridad, evaluación completa/delta F1 ≤0.01, clases/preproceso/OOD vinculados, features, staging y rollback | **Verificado localmente** con ONNX y LiteRT reales, más fallos simulados de activación/OOD. Ambos checkpoints FP32 pasan logits/features; ShuffleNet INT8 pasa paridad, **B0 INT8 falla** y queda no entregable. Android: **bloqueado**, no hay dispositivo; no se sincronizó una app real |
| P2-01 | HPO ignorado, flags/defaults / P1-02 | `training/hyperparameters.py`, train/CV/Make/Modal: esquema PyTorch distinto de SGD, CLI > JSON > defaults, configuración efectiva, test opt-in, flags negativos | **Verificado localmente** el parser y la precedencia, sin GPU. Main conserva pérdida ponderada y sin sampler; baseline mantiene su política. Ejecución remota de los wrappers: pendiente |
| P2-02 | Raíces ignoradas y descarga falsa/incompleta / P1-01 | `config.py`, download/create_splits: raíces explícitas, staging validado por imagen/hash/revisión, backup y rollback, reanudación compatible; corruptos por worker; exclusiones por ruta/hash/razón | **Verificado localmente y descarga completa HF**: 33 437 imágenes, todos los 33 433 hashes históricos Etapa 2 coinciden. Ocho originales conflictivos preservados y excluidos solo de splits nuevos; 0 corruptas en nueva preparación |
| P2-03 | Extras y cobertura accidentalmente omitida | `pyproject.toml`: extras test/onnx/tflite/segmentation; test litert_torch; constraints producidos por resolución real de pip y entornos separados | **231 aprobadas / 1 módulo omitido** en venv limpio ONNX/XAI; **2 aprobadas LiteRT** en otro venv limpio. Ambos `pip check` sin conflictos; XML conservados. La omisión LiteRT en el primer entorno está cubierta por el segundo, no ocultada |
| P2-04 | Docs/promesas inconsistentes / evidencia anterior | README/LOCAL/CLAUDE, web, FAIRNESS_REPORT, LaTeX, manifest y guía de reproducción; build sin git fetch | **Fuentes y reportes actualizados**; 24 activos regenerados y hashes verificados; typecheck web aprobado. PDF bloqueado por falta de TeX; build web no validado con Node 18, instalación de Node nuevo rechazada por límite de cuenta. No se afirma APK/API/PWA disponible |

## Evidencia ejecutada

- [Auditoría aritmética de corridas y Etapa 2](evidence/2026-09-11-corridas-recibidas.json):
  hashes de pesos/CSV, comparación de matrices, clases, historia y particiones.
- [Suite final en venv limpio](evidence/2026-09-12-pytest-clean-final-v3.xml):
  **231 aprobadas, 1 módulo omitido, 13 avisos, 33,83 s**. TFLite se ejecuta por separado.
- [Suite LiteRT](evidence/2026-09-12-pytest-litert.xml): **2 aprobadas, 5 avisos, 13,52 s**,
  conversión real ShuffleNet y runtime de logits/features. No son métricas agronómicas.
- Se conservan ejecuciones intermedias: 220/2 y 222/2 aprobadas/omitidas; la primera
  `pytest-clean-final.xml` tuvo 229 aprobadas y 1 fallo por texto del error de pesos del
  ensemble. Se restauró el mensaje esperado, sin debilitar la prueba; v2 pasó 230 casos
  y v3 incorpora además la regresión de clases ausentes en el desglose exportado.
- [Resolución real de dependencias](evidence/2026-09-12-dependency-resolution.json):
  plataforma, 99 distribuciones y hashes de descarga; sin instalar en el entorno vecino.
- Regresiones: `tests/data/*integrity.py`, `tests/training/*integrity.py`,
  `tests/training/test_run_contracts.py`, `tests/analysis/test_fairness_integrity.py`,
  `tests/test_segmentation_quality.py`, `tests/pipeline/test_repair_contracts.py`,
  `tests/etapa_2/test_final_recovery.py`, `tests/pipeline/test_sync_mobile_model.py`.
- `tests/test_cpu_repair_integration.py` ejecuta entrenamiento pequeño, best checkpoint,
  inferencia con shuffle e IDs, ONNX y runtime real, evaluación y puerta de paquete móvil
  sin OOD. Usa imágenes sintéticas: no aporta accuracy agronómica ni rendimiento Android.
- `ruff check` aprobado sobre los archivos Python modificados/nuevos;
  `npm run typecheck` aprobado. No se confunden con compilación del sitio o PDF.
- [Piloto y dataset](2026-09-12-piloto-y-dataset.md): 288 imágenes, 12 heads de dos
  backbones/variantes y tres semillas; sin mejora media global de segmentación.
- [Exportación real](2026-09-12-exportacion-real.md): tres variantes TFLite evaluadas
  sobre 5 015 imágenes cada una; FP32 reproduce PyTorch y ShuffleNet INT8 tiene delta
  +0,001999. B0 INT8 queda fallido por paridad de probabilidades y features.
  Se corrigió además el universo de clases de macro-F1 por entorno y se recalculó desde
  predicciones guardadas, conservando los reportes iniciales y sin repetir inferencia.

Comando de la suite ejecutada, desde la raíz del checkout:

```bash
MPLCONFIGDIR=/tmp/corn-repair-mpl \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
CORN_TEST_REAL_IMAGE=/tmp/doctor-maiz-hf-20260911-JjlLG8/clean/gray_leaf_spot/real/gray_leaf_spot_maize_field_real_13749986.jpg \
/tmp/corn-repair-clean-Rq8ucd/bin/python -m pytest -q \
  --junitxml=docs/reviews/evidence/2026-09-12-pytest-clean-final-v3.xml
```

## Recursos y límites

Se usó en lectura el entorno vecino `.venv` (Python 3.12.3, PyTorch 2.12.1,
torchvision 0.27.1), suplementado en `/tmp/corn-repair-deps-GmLFcg`, para el piloto.
Después se instalaron venv independientes `/tmp/corn-repair-clean-Rq8ucd` y
`/tmp/corn-repair-litert-QnPicA`. Ninguna dependencia se instaló en el proyecto vecino.
El `.venv-modal` indicado por el usuario contiene el cliente Modal, no el stack de entrenamiento.

Fuente autorizada: [dataset HF](https://huggingface.co/datasets/daiv05/corn-leaf-diseases-pests-and-deficiencies),
revisión `e515ab2f1e4c5729f8447520f1a630cf14c532dc`, 23 shards, aproximadamente 19.2 GB.
Descarga aislada: `/tmp/doctor-maiz-hf-20260911-JjlLG8`. El primer intento falló con
HTTP 503/timeouts, retuvo 6.3 GB y **no** activó `clean/`; la reanudación con la misma
revisión y dos conexiones **terminó y validó 33 437 imágenes**. Preservarlo fuera de /tmp
antes de usarlo como almacenamiento permanente.

Hay un checkpoint local del segmentador en el proyecto vecino. Su export anterior de
derivados no se usa como dataset original: tenía exclusiones y metadatos de entorno que
no pueden sustituir los splits originales. El contrato de esta reparación queda registrado
independientemente. Se compararon configuración y código de su compuerta; difieren en
perfil, umbrales y geometría. La mención externa de calibración con 42 revisiones no
se adopta como evidencia revisada de este clasificador.

No hay dispositivo Android ni compilador TeX comprobados. Node 18 no
ejecuta el build actual; la descarga temporal de Node nuevo fue rechazada automáticamente
por límite de cuenta, sin intentar eludir esa decisión. `npm ci` avisó de dos vulnerabilidades
(una moderada, una alta); no se aplicó un `audit fix` indiscriminado.

## Migración y clasificación de resultados

Ver [corridas recibidas](2026-09-11-corridas-recibidas.md) y
[guía reproducible](2026-09-11-reproducibilidad.md).
Los contratos nuevos rechazan cachés/checkpoints históricos incompletos. La migración crea
una copia explícitamente reconstruida, con hashes de los bytes **actualmente observados**;
no inventa hashes de los datos usados en septiembre ni reproduce el entrenamiento.

## Cierre experimental

El código del piloto es `scripts/checks/paired_segmentation_pilot.py`: selección estratificada
train/val, política original para máscaras inciertas/rechazadas, población completa del
subconjunto, tres semillas, presupuesto idéntico, artefactos y revisión humana.
Separa sensibilidad de los checkpoints recibidos ante cambio de entrada de entrenamiento
de heads nuevos sobre ImageNet congelado. No reentrena masivamente, no usa test para
seleccionar, no habilita variación de fondo sin un protocolo verificado.

**Piloto ejecutado:** 144 train/144 val, tres semillas, 20 épocas por head. B0 original
0,631242±0,022261 frente a segmentado 0,608973±0,011571; ShuffleNet original
0,648189±0,007923 frente a segmentado 0,644999±0,000513. Son medias macro-F1 de heads
sobre ImageNet congelado, no las métricas de los checkpoints completos ni del corpus.
Solo 31/288 máscaras fueron aceptadas; 257 conservaron original. La señal de potasio
en ShuffleNet tiene soporte 12 y no acredita mejora global.

Los ocho conflictos quedan en exclusión provisional y una relación visual confirmada
queda agrupada; otros 372 pares necesitan revisión. Nueva preparación main y Etapa 2
realizada sin cambiar el dataset ni los splits históricos. No se reentrena masivamente
contra una versión que aún necesita resolver procedencia/etiquetas.

Siguen abiertas la equivalencia histórica de preprocesamiento, procedencia de planta/sesión,
calibración humana de máscaras, confirmación con entrenamiento completo y dispositivo móvil.
No se declara el proyecto listo para producción.
