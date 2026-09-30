# LOSO baseline EfficientNet-Lite0: protocolo predefinido

Estado al 30 de septiembre de 2026: **dos entrenamientos y dos evaluaciones
externas completados; sin jobs activos**.
Fuentes predefinidas: `maize-diseases` y `multicrop-disease-maiz`. No es HPO,
multi-seed ni validación cruzada. Se hicieron exactamente dos entrenamientos,
ambos con seed 42. No se ejecutó otra fase automáticamente.

## Particiones y control de fuga

La fuente de verdad es `outputs/multiseed-inputs/frozen/seed_42` (master, lock y
tres CSV), no el YAML actual. El lock original conserva SHA-256
`0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8`.
Los manifests derivados se guardan en
`outputs/loso/efficientnet_lite0_baseline/<source>/seed_42/splits/`.
No se escribió ningún archivo en `seed_42`.

Para cada fuente, train y validation son las filas de sus splits históricos
con `source_id` diferente del holdout. Las filas de test de las demás fuentes
no se usan. El holdout reúne **todas** las filas elegibles de la fuente,
incluidas sus antiguas train, val y test; `historical_split` conserva esa
procedencia. Esta regla se fijó antes de entrenar. Los tres conjuntos son
disjuntos por `sample_id`, SHA-256 de archivo y `pixel_sha256`; el inventario
de píxeles usado tiene SHA-256
`7d15a768a604d1f94ded6c9a490b5b58a699b0850d4c8f003245775c4470090e`.

| Fuente holdout | Train | Validation | Holdout | Origen histórico del holdout (train/val/test) | Reducción de train |
|---|---:|---:|---:|---:|---:|
| maize-diseases | 18 519 | 3 967 | 6 989 | 4 881 / 1 047 / 1 061 | −4 881 (−20,86 %) |
| multicrop-disease-maiz | 19 328 | 4 150 | 5 816 | 4 072 / 864 / 880 | −4 072 (−17,40 %) |

`maize-diseases` contiene `common_rust` 1 192, `healthy` 4 741 y
`northern_corn_leaf_blight` 1 056; el train restante conserva 737, 2 835
y 4 025, respectivamente. `multicrop-disease-maiz` contiene `common_rust`
958, `fall_armyworm` 1 627 y `lethal_necrosis` 3 231; el train restante
conserva 916, 2 276 y 2 202. En ambos protocolos permanecen las nueve
clases globales y todas las del holdout tienen soporte en train.

| Fuente | SHA-256 train.csv | SHA-256 val.csv | SHA-256 holdout.csv | SHA-256 lock derivado |
|---|---|---|---|---|
| maize-diseases | `b361ba1010e402d110f06e51419e397b24464eb4ed0c3951500083f0c8a4d34a` | `3da556ca4d6e6b1f196a9b0977674965b3044c9f967e070fcf1fce72a2868350` | `c94e4e1963f00b18e1bf0d0ed9ca41a50dd7676c8c1e5157567474cdbee54167` | `6486473932ae2937ff5a97881953d6ffbfac3725aa81170ef2b20a6bb49cbc07` |
| multicrop-disease-maiz | `7fd7878df04c7ab6d6126fef96104bedc48699e5e417d533b06afc77e86115e4` | `eea6551f65050f8bb6780d501d382c2e758037f32c8e1624e78214067a64b1cb` | `d36f938bdb6db7505722e5b7cf6458adc5568f3a94525884943e47eb0a8b0688` | `d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b` |

## Entrenamiento y selección

Se reutiliza `scripts/pipeline/train.py` con `--skip-test`, modelo
`efficientnet_lite0` y la configuración registrada en el summary contractual
del baseline `20260921_204608`: resolución 224×224, AdamW,
LR 0,0001, weight decay 0,0001, batch 32, scheduler coseno con warmup 3 y
min LR 0,000001, hasta 60 épocas, paciencia 8, pérdida con pesos
`sqrt_inverse` y label smoothing 0,1, clipping 1,0, modelo preentrenado,
sin CLAHE, sin sampler ni cap por clase. El criterio de selección es
**Macro-F1 de validation interna**. El `--skip-test` impide crear un loader
para holdout o test durante la selección. Cada `best.pth` recibió un lock
de selección con SHA-256 y timestamps antes de inferir el holdout.

La regla baseline de augmentations permanece, incluida la derivación de clases
minoritarias con ratio >4 a partir del train. Al cambiar la distribución,
cambia qué etiquetas cruzan ese umbral; no se fija la lista antigua utilizando
datos del holdout. También cambian los pesos numéricos de clase calculados por
`sqrt_inverse`. Son consecuencias de retirar una fuente, no una búsqueda ni
un nuevo método de balanceo. Una caída LOSO no podrá atribuirse solo al
domain shift: también se reduce el tamaño de train y cambia la mezcla de
clases.

El drift del SHA histórico de `dataset.yaml` corresponde a la preparación del
corpus; los manifests congelados evitan regenerarlo. La configuración efectiva
del baseline se tomó del summary contractual. Su registro de augmentations
no contiene todos los parámetros aleatorios; se usa el código vigente que
implementa los pipelines descritos, sin modificarlos en esta fase.
El diff frente al commit `600ebb9` anterior al baseline no muestra cambios en
las operaciones aleatorias ni en `CornDataset`; solo metadatos contractuales.

## Verificación y resultados

`prepare_loso_baseline.py` genera manifests y rechaza hashes congelados,
cobertura incompleta, etiquetas ausentes y solapamientos exactos. Ruff pasó
en los archivos nuevos; los ocho tests LOSO y de selection lock pasaron. La suite local relevante
terminó con 157 tests correctos y uno omitido usando `DATASET_ROOT=/tmp` para
tests sintéticos. Dos módulos de HPO no se recolectaron por falta de Optuna
en el venv local. El smoke local del lanzador pasó sin GPU.

Las runs `20260930_144843` (maize-diseases; app `ap-ARPoHOb6puOtjTgSSPAqvf`) y
`20260930_145541` (multicrop-disease-maiz; app `ap-BRRfp7m0l1U92ReVqIsNfE`)
se ejecutaron con `--skip-test` después del preflight. No hubo intentos fallidos
ni reinicios. Los summaries confirman el hash contractual del baseline
(`53cc091e…e8b`), `test_used=false` y los locks derivados. Las predicciones
externas se recalcularon por `sample_id` y reprodujeron Macro-F1, accuracy y
errores. El compilador verificó los hashes de checkpoint, summary, manifests y
selection lock, más el orden temporal entrenamiento → lock → evaluación.
Los artefactos completos están en `outputs/loso/efficientnet_lite0_baseline/`
y en `corn-outputs:/loso/efficientnet_lite0_baseline/`; `seed_42` sigue intacto.

Resultados e interpretación: [LOSO_RESULTS.md](LOSO_RESULTS.md).
