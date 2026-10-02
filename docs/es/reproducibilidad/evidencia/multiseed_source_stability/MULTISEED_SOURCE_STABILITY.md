# Estabilidad entre semillas del baseline y de dos exclusiones de fuente

Protocolo iniciado el 1 de octubre de 2026 en `dev-abner2`. La pregunta es doble:
cuánto varía el baseline al cambiar la semilla de entrenamiento y si el resultado
LOSO observado con la semilla 42 persiste en otras inicializaciones. Este estudio
no optimiza el modelo ni modifica el corpus.

## Diseño fijado antes de las evaluaciones finales

Se usan las semillas 42, 123 y 2026 en tres escenarios: partición estándar,
exclusión completa de `maize-diseases` y exclusión completa de
`multicrop-disease-maiz`. Las tres runs de semilla 42 se reutilizan; el límite es
seis entrenamientos nuevos. Los dos LOSO comparten, dentro de cada escenario,
los CSV derivados para semilla 42: la semilla nueva controla entrenamiento, no
una nueva partición.

El contrato de referencia es el summary original de la run EfficientNet-Lite0
`20260921_204608` (SHA-256 `20bb0945574913ffa8c736fecbcf25d932e6b87ce3e7a7c59d98439558432346`).
Modelo de nueve clases, entrada RGB 224 × 224, pesos pretrained, `AdamW`, LR
`0.0001`, weight decay `0.0001`, batch 32, scheduler cosine con tres épocas de
warmup y LR mínima `0.000001`, máximo 60 épocas, paciencia 8, pesos de pérdida
`sqrt_inverse`, label smoothing `0.1`, clip norm `1.0`, sin CLAHE ni sampler,
sin `max_per_class`. La mejor época se elige por Macro-F1 de validación. El
contrato de preprocessing, las clases y los hiperparámetros se comparan campo
por campo con el summary original después de cada run. El SHA de configuración
incluye la semilla, por lo que **debe** cambiar entre semillas aunque el resto
de la configuración sea idéntico.

`set_global_seed` fija Python, NumPy, PyTorch y CUDA, y configura cuDNN para
reproducibilidad. El `DataLoader` mezcla el train; sus workers reciben semillas
derivadas de PyTorch. Las augmentations de entrenamiento y de minorías dependen
de la aleatoriedad de PyTorch. No hay `WeightedRandomSampler`. La selección
de clases minoritarias parte de la distribución del train de cada escenario;
dentro de ese escenario la distribución no cambia entre semillas.

| Escenario | Lock del manifest (SHA-256) | Train | Validación | Final |
|---|---|---:|---:|---:|
| Estándar | `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` | 23 400 | 5 014 | Test: 5 015 |
| LOSO maize-diseases | `6486473932ae2937ff5a97881953d6ffbfac3725aa81170ef2b20a6bb49cbc07` | 18 519 | 3 967 | Holdout: 6 989 |
| LOSO multicrop-disease-maiz | `d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b` | 19 328 | 4 150 | Holdout: 5 816 |

Los checks de los manifests comprueban SHA-256 de los CSV y locks. La
derivación LOSO original comprobó separación por `sample_id`, SHA de archivo y
SHA de píxel. Una [revisión read-only de los tres escenarios](integrity_preflight.json)
repitió esos cruces y obtuvo cero solapamientos para los nueve pares de
particiones/identidades de cada escenario. El train estándar tiene 23 400 muestras. Al excluir
`maize-diseases` se retiran 4 881 (20.86 %); al excluir
`multicrop-disease-maiz` se retiran 4 072 (17.40 %). Los soportes restantes
por clase están en [la tabla CSV](loso_remaining_train_support.csv); los recuentos
del holdout constan en cada `manifest.lock.json`.

Las seis runs nuevas se entrenaron con `--skip-test`. Tras verificar el historial,
el SHA del checkpoint y la coincidencia del contrato, se escribe
`selection.lock.json`. Solo entonces se permite una inferencia final por run,
protegida por `final_evaluation.started.json` (`evaluation_count=1`). Se
reconcilian las predicciones con el manifest por `sample_id`, no por posición.
El Macro-F1 LOSO se promedia sobre las clases presentes en las etiquetas del
holdout, igual que el protocolo de semilla 42. El resumen estadístico usa
`n=3` y desviación estándar **muestral** (`ddof=1`). No se hará prueba de
significancia a partir de tres semillas.

La run estándar de semilla 42 sí tuvo test histórico, pero no existe evidencia
de un `selection.lock.json` anterior a ese test. Se reutilizan su checkpoint,
summary y predicciones verificadas, sin fabricar un lock retrospectivo. Los
dos LOSO históricos sí tienen locks anteriores al holdout. La comparación de
las tres semillas del baseline debe leerse con esta diferencia de trazabilidad.

Tampoco quedó archivado un snapshot completo de versiones de paquetes ni la
revisión exacta de los pesos pretrained de las runs de semilla 42. Las nuevas
runs registran Python, PyTorch, CUDA, timm y GPU; la equivalencia histórica se
verifica por contrato y artefactos, no por identidad total del entorno.

Las referencias históricas in-distribution para las dos fuentes son Macro-F1
de validación por fuente, sobre las imágenes presentes en la partición estándar:
`0.9978925393906257` para `maize-diseases` y `0.9972173913043477` para
`multicrop-disease-maiz`, extraídas de
[`validation_metrics_by_source.csv`](../source_analysis/validation_metrics_by_source.csv).
El delta LOSO respecto a ellas es **descriptivo**: el holdout LOSO reúne todas
las muestras elegibles de la fuente y el train también pierde tamaño y cambia
de distribución de clases. Una caída no identifica por sí sola un atajo de
fuente ni atribuye causalidad a domain shift.

La comparación con mejoras históricas utiliza las diferencias exactas
de los artefactos originales: ensamble `0.9566569450594972 −
0.9483330101876505 = 0.0083239348718467` sobre el mejor individual B0;
afinación B0 `0.9483330101876505 − 0.9426381998733162 =
0.0056948103143343`. Esos experimentos tienen otras condiciones y no son
efectos pareados del baseline Lite0 de este estudio. Aquí solo se compara el
orden de magnitud con la SD observada.

## Cierre de las runs y secuencia de evaluación

Se consolidaron nueve resultados: las tres runs `seed=42` anteriores y seis
entrenamientos nuevos (`seed=123`, `seed=2026` por escenario). Ninguna run
histórica se repitió. Las seis nuevas terminaron en modo `validation_only`;
sus `summary.json` no contienen test ni holdout. Se verificaron el manifest
congelado, configuración, historial, predicciones de validation, `best.pth`
y SHA-256 antes de abrir los conjuntos finales.

| Escenario | Seed | Run ID | Mejor época | Validation Macro-F1 | Final Macro-F1 | Final accuracy |
|---|---:|---|---:|---:|---:|---:|
| Estándar | 42 | `20260921_204608` | 39 | 0.956086 | 0.948002 | 0.978066 |
| Estándar | 123 | `20261001_140756` | 39 | 0.960281 | 0.937867 | 0.973081 |
| Estándar | 2026 | `20261001_145025` | 45 | 0.952856 | 0.944597 | 0.977468 |
| LOSO maize-diseases | 42 | `20260930_144843` | 15 | 0.948975 | 0.993124 | 0.993704 |
| LOSO maize-diseases | 123 | `20261001_160256` | 27 | 0.952276 | 0.992747 | 0.992989 |
| LOSO maize-diseases | 2026 | `20261001_163145` | 29 | 0.954576 | 0.993922 | 0.994420 |
| LOSO multicrop-disease-maiz | 42 | `20260930_145541` | 42 | 0.952488 | 0.905490 | 0.806912 |
| LOSO multicrop-disease-maiz | 123 | `20261001_170150` | 28 | 0.951007 | 0.896964 | 0.800550 |
| LOSO multicrop-disease-maiz | 2026 | `20261001_173226` | 43 | 0.959352 | 0.903264 | 0.803817 |

La columna «Final» significa **test contractual** solo para el escenario
estándar; en los otros dos es el **holdout LOSO** de la fuente excluida. En
cada escenario se conservaron los mismos manifests para las tres semillas.

El 1 de octubre, el finalizador escribió los seis `selection.lock.json`
antes de iniciar la primera inferencia final. Cada lock incluye run, escenario,
fuente excluida, checkpoint y configuración, hashes de manifests, commit y
estado Git, así como la marca de que el conjunto final no se había usado.
Después se ejecutó una inferencia por run; cada
`final_evaluation.started.json` registra `evaluation_count=1`. La
[tabla de evaluación](evaluation_registry.csv) conserva el orden temporal,
rutas y hashes. El baseline `seed=42` tenía test histórico **sin** lock
pre-test documentado; no se fabricó uno retrospectivo. Los dos LOSO
`seed=42` sí conservan locks previos al holdout. En esos tres resultados
históricos el conteo formal del nuevo guard no aplica y queda vacío en el
registro, no inferido como si se hubiera ejecutado hoy.

Hubo una preempción de Modal durante la época 7 del primer intento LOSO
`maize-diseases/seed_123`: quedaron seis épocas cerradas, sin
`summary.json` válido. El intento se conserva en
`corn-outputs:/loso/efficientnet_lite0_baseline/maize-diseases/seed_123/interrupted_attempts/attempt_1/`.
El segundo intento (`20261001_160256`) pasó todas las verificaciones y es
el único incluido en las tablas. No se promedió ni se evaluó el intento
interrumpido.

## Baseline estándar: estabilidad entre semillas

Las tres evaluaciones usan las mismas 5 015 imágenes de test. El checkpoint
de cada semilla quedó seleccionado por validation antes de abrir test.

| Métrica | Media ± SD muestral | Mediana | Mínimo | Máximo | Rango |
|---|---:|---:|---:|---:|---:|
| Validation Macro-F1 | 0.956408 ± 0.003723 | 0.956086 | 0.952856 | 0.960281 | 0.007425 |
| Test Macro-F1 | 0.943489 ± 0.005158 | 0.944597 | 0.937867 | 0.948002 | 0.010135 |
| Test accuracy | 0.976205 ± 0.002722 | 0.977468 | 0.973081 | 0.978066 | 0.004985 |
| Test ECE (15 bins) | 0.136975 ± 0.002464 | 0.136456 | 0.134811 | 0.139656 | 0.004845 |

Para las deficiencias nutricionales, el F1 de test de nueve clases fue:

| Clase | Seed 42 | Seed 123 | Seed 2026 | Media ± SD | Soporte |
|---|---:|---:|---:|---:|---:|
| nitrogen_deficiency | 0.905109 | 0.877323 | 0.892086 | 0.891506 ± 0.013902 | 127 |
| phosphorus_deficiency | 0.931818 | 0.920152 | 0.931298 | 0.927756 ± 0.006590 | 140 |
| potassium_deficiency | 0.824176 | 0.795455 | 0.807018 | 0.808883 ± 0.014451 | 93 |

Los valores completos de precisión, recall, F1, ECE, confianza, matrices y
predicciones están en los artefactos de cada run; los agregados por clase en
[baseline_class_multiseed.csv](baseline_class_multiseed.csv) y
[baseline_class_summary.csv](baseline_class_summary.csv).

### Observación 1: magnitud de mejoras históricas

El [resumen original del ensamble](../../../resultados/evidencia/ensamble_resumen.json)
da una diferencia de Macro-F1 test de **0.008324** frente a su mejor modelo
individual B0. Los [runs originales de B0](../../../resultados/evidencia/manifiesto_corridas.csv)
dan **0.005695** entre la versión afinada y la anterior. Esas magnitudes son
aproximadamente **1.61** y **1.10** veces la SD de test observada aquí
(0.005158). La comparación es de orden de magnitud: se trata de modelos,
configuraciones y experimentos históricos distintos, no de efectos pareados
sobre estas tres seeds Lite0. No permite llamar «ruido» o «mejora
significativa» a ninguno de los dos deltas. La observación de dependencia de
una sola semilla queda atendida para el baseline desplegado con esta primera
estimación de variabilidad (`n=3`), no con una prueba de significancia.

## LOSO maize-diseases

El holdout contiene 6 989 muestras de tres clases y estuvo completamente
fuera de train (18 519) y validation (3 967). La fuente tampoco intervino en
selección ni early stopping. El Macro-F1 LOSO fue **0.993264 ± 0.000600**
(mediana 0.993124; mínimo 0.992747; máximo 0.993922); accuracy
**0.993704 ± 0.000715**. Las tres seeds mantienen buena transferencia hacia
esta fuente concreta. Frente al F1 histórico in-distribution por fuente
(0.997893), el delta medio es −0.004628, solo descriptivo: soporte y
condiciones no son idénticos.

## LOSO multicrop-disease-maiz

El holdout contiene 5 816 muestras de tres clases y quedó fuera de train
(19 328) y validation (4 150), selección y early stopping. El Macro-F1 LOSO
fue **0.901906 ± 0.004422** (mediana 0.903264; mínimo 0.896964; máximo
0.905490); accuracy **0.803760 ± 0.003181**. Las otras dos semillas no
contradicen la caída observada en la 42. Frente al F1 histórico
in-distribution por fuente (0.997217), el delta medio es −0.095311,
también descriptivo.

La clase `lethal_necrosis` aporta una parte importante de la dificultad:

| Seed | Precision | Recall | F1 | Soporte |
|---:|---:|---:|---:|---:|
| 42 | 0.999145 | 0.723615 | 0.839347 | 3 231 |
| 123 | 0.999159 | 0.735067 | 0.847004 | 3 231 |
| 2026 | 0.998330 | 0.740019 | 0.849982 | 3 231 |

Su F1 medio es **0.845444 ± 0.005487** (mínimo 0.839347, máximo
0.849982). La precisión permanece alta y el recall bajo en las tres seeds;
el déficit no depende exclusivamente de la semilla 42. Las tablas de clase
[por seed](loso_class_multiseed.csv) y [agregadas](loso_class_summary.csv)
conservan los soportes y el resto de las clases.

## Comparación entre fuentes y observación 2

| Fuente excluida | Seeds | LOSO Macro-F1 media ± SD | Mínimo | Máximo | Accuracy media |
|---|---:|---:|---:|---:|---:|
| maize-diseases | 3 | 0.993264 ± 0.000600 | 0.992747 | 0.993922 | 0.993704 |
| multicrop-disease-maiz | 3 | 0.901906 ± 0.004422 | 0.896964 | 0.905490 | 0.803760 |

La separación entre las medias (**0.091358**) es mucho mayor que la SD
interna de cualquiera de los dos LOSO. Es una comparación descriptiva, no un
test de significancia. La [auditoría de píxeles](../pixel_duplicate_audit/PIXEL_DUPLICATE_AUDIT.md)
halló cero duplicados exactos y cero solapamientos entre los splits vigentes:
estas fuentes no «concentran una fuga» en el `seed_42` actual. El
[análisis source × label](../source_analysis/SOURCE_ANALYSIS.md) encontró
asociación descriptiva (V de Cramér 0.5030), pero una asociación no demuestra
un atajo aprendido. Los tres LOSO confirman un comportamiento distinto entre
fuentes: buena transferencia a `maize-diseases` y sensibilidad persistente
al excluir `multicrop-disease-maiz`. No identifican una causa visual ni
prueban causalidad o generalización universal.

## Límites y siguiente fase

Son solo tres semillas por escenario. LOSO cambia a la vez la disponibilidad
de una fuente, el tamaño del train (−20.86 % y −17.40 %) y la distribución
de clases; también pueden cambiar pesos de pérdida y clases minoritarias
derivados del train, aunque su regla contractual no cambió. No puede
atribuirse toda la caída a `domain shift` ni concluir `source shortcut`.
Se evaluaron dos de once fuentes. La auditoría vigente cubre identidad
exacta por archivo y píxel, no casi duplicados perceptuales. El test
estándar ya se había observado históricamente; no debe seguir guiando
ajustes posteriores. Para las runs `seed=42` no existe un snapshot completo
del entorno o de la revisión de pesos pretrained, aunque su contrato de
configuración, manifests y checkpoints sí se verificó.

Las seis runs nuevas registran Python 3.11.12, PyTorch 2.12.1+cu126,
timm 1.0.29 y CUDA 12.6; Modal informó GPU A10/A10G. Sus duraciones de
entrenamiento están en los CSV y son tiempo desde inicio hasta validación
completa. Para `seed=42` solo se documenta suma de segundos por época;
no son duraciones directamente equivalentes. No se hizo HPO, ensamble, CV,
entrenamiento formal ni más LOSO en esta fase.

La próxima prioridad sugerida es un protocolo pre-registrado para medir el
ensamble bajo exclusión de fuente, sin ajustar nada mirando estos holdouts;
una evaluación externa nueva sería preferible para decisiones posteriores.
Esa fase **no se ejecutó**.

La verificación del cierre ejecutó **23 tests** del estudio; `ruff check` y
`ruff format --check` pasaron en los doce archivos Python del paquete.
No se hizo commit, push ni merge.

## Evidencia y hashes

[baseline_multiseed.csv](baseline_multiseed.csv),
[loso_multiseed.csv](loso_multiseed.csv),
[loso_class_multiseed.csv](loso_class_multiseed.csv),
[evaluation_registry.csv](evaluation_registry.csv) y
[multiseed_summary.json](multiseed_summary.json) contienen las cifras sin
redondear. [manifest_references.json](manifest_references.json) registra
hashes completos de manifests, predicciones, locks y checkpoints. Los cinco
gráficos generados son
[baseline por seed](baseline_f1_by_seed.png),
[LOSO por fuente](loso_f1_by_source_seed.png),
[medias y SD](scenario_mean_sd.png),
[lethal_necrosis](multicrop_lethal_necrosis_by_seed.png) y
[deltas descriptivos](loso_delta_by_source.png). Los bundles íntegros
(checkpoints, matrices y predicciones) están en
`corn-outputs:/multiseed/source_stability/baseline/` y
`corn-outputs:/loso/efficientnet_lite0_baseline/<source>/seed_<seed>/`;
las copias de trabajo bajo `outputs/` son derivadas y no se versionan.

| Escenario | Seed | SHA-256 de `best.pth` |
|---|---:|---|
| Estándar | 42 | `860180f33bf1749ffee9f3cccb9569a47afd9ca0fcef52713e18f99f82a57859` |
| Estándar | 123 | `19a9f860352fc8812f7d4d3f9aedef68ff4b52175e0b48ee757a133d409cb053` |
| Estándar | 2026 | `6a2b302307101cd55414e7633d8755c5e584d5cceb4734ff241e8c0b5829ba51` |
| LOSO maize-diseases | 42 | `b16b1ef8bb4285f65be020272845b6c927057103bf177a290311f186de101dcc` |
| LOSO maize-diseases | 123 | `4f2d8e86ac7eb1ddd57f3593f8a187c983931f860247fbcca6d5be1a24a7e345` |
| LOSO maize-diseases | 2026 | `50199b9d3a46709f6d1cdc57142593ec883782ca94a22f3ea7e0740a44e292f6` |
| LOSO multicrop-disease-maiz | 42 | `bd778a28d251fd1b45960f3caec8f5b51d0ed0178c4e4f46e06d2b40948e4517` |
| LOSO multicrop-disease-maiz | 123 | `3bb36aeb4b4328f0552a606a76d8b8455a7a1fea1c5fc288784dfa8be35b0863` |
| LOSO multicrop-disease-maiz | 2026 | `d38fc9f91adffc6943abcdf72bb8ef4951ca18c98bc2f1be6de3d723bc193c43` |
