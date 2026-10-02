# Ensamble histórico bajo exclusión de `multicrop-disease-maiz`

Esta prueba responde si la ganancia del ensamble de tres modelos observada en el test estratificado se conserva al excluir por completo una fuente durante el entrenamiento y la selección. La comparación principal es pareada por semilla contra EfficientNet-B0, el mejor modelo individual del experimento original. EfficientNet-Lite0 LOSO se conserva como referencia adicional, porque ya había mostrado una caída reproducible en esta fuente.

## Ensamble recuperado de la evidencia

La fuente primaria es [`ensamble_resumen.json`](../../../resultados/evidencia/ensamble_resumen.json), cuyo SHA-256 es `dcafbb4c21e2d69646b8a04cd1bbf425474b615f5eb6b44d594abd469b6f90c2`. El método se comprueba además en `src/models/ensemble.py`: para cada imagen se calcula `softmax(logits)` de cada miembro, se promedian las tres distribuciones con pesos iguales y se selecciona la clase de probabilidad conjunta máxima. No es un promedio de logits ni voto mayoritario; no se ajustan pesos para LOSO.

| Miembro original | Run | Configuración distintiva | Test Macro-F1 original |
|---|---|---|---:|
| EfficientNet-Lite0 | `20260812_221429` | Batch 32, LR 0.0001 | 0.9467892694 |
| EfficientNet-B0 | `20260910_170120` | Batch 64, LR 0.0004548 | 0.9483330102 |
| ShuffleNet-V2-x1.0 | `20260910_184521` | Batch 64, LR 0.0004548 | 0.9329803746 |
| Promedio softmax, 1/3 cada uno | evaluación original | Test estratificado, 5.015 imágenes | 0.9566569451 |

La diferencia exacta original fue `0.9566569450594972 − 0.9483330101876505 = 0.008323934871846683` frente a B0. El resumen redondea ese delta a 0.0083; la formulación «+0.0084» era aproximada. Otros ensambles archivados (uno ponderado de cuatro modelos de la etapa 2 y uno de dos modelos) no corresponden a esta cifra. El [preflight](PREFLIGHT.md) separa los tres protocolos.

Los tres checkpoints originales se entrenaron en el split estratificado de desarrollo, que incluye la fuente objetivo; por tanto, **no** se usaron para inferencia LOSO. El checkpoint original de Lite0 desplegado no está disponible aquí y no se le inventó un SHA. Los archivos de B0 y ShuffleNet originales sí se conservaron y se usaron solo para fijar y auditar su configuración, nunca para predecir sobre el holdout.

## Partición y selección

El manifest es exactamente el LOSO `multicrop-disease-maiz` ya validado: SHA-256 del lock `d8aac805a3d768166a63be664f1a658478926ab7d44a48e617c950f795e1537b`. Quedan 19.328 imágenes en train, 4.150 en validation y 5.816 en holdout (958 de `common_rust`, 1.627 de `fall_armyworm`, 3.231 de `lethal_necrosis`). Train y validation no contienen esa fuente. Se verifican los SHA-256 de los CSV y del lock antes de entrenar y antes de evaluar; los controles previos del protocolo excluyeron solapamientos exactos por `sample_id`, SHA de archivo y SHA de píxel.

Para cada seed 42, 123 y 2026 se reutiliza un Lite0 LOSO completo. B0 y ShuffleNet requieren nuevos entrenamientos con su configuración histórica: máximo 35 épocas, AdamW, weight decay 0.0001, cosine con tres épocas de warmup y mínimo LR 1e-6, paciencia 8, pérdida `sqrt_inverse`, label smoothing 0.1, clip 1.0, RGB 224 × 224, sin CLAHE, sampler ni tope por clase. La semilla modifica inicialización y barajado, no el reparto de muestras.

Los entrenamientos nuevos terminan en modo `validation_only` (`--skip-test`). El mejor checkpoint se selecciona por Macro-F1 de validation; se cotejan época y puntuación con el historial y se verifican SHA-256, configuración, preprocessing y mapa de clases antes de escribir un `selection.lock.json` por componente. Los tres locks conjuntos se escriben antes de inferir o leer etiquetas del holdout; previamente solo se comprueban hashes de archivos. El evaluador crea y confirma en el volumen un guard `evaluation_count=1` antes de cargar imágenes; si queda iniciado pero incompleto, no lo repite automáticamente. Las predicciones de los cuatro modelos se comparan por `sample_id` y etiqueta, no por orden de fila.

El Macro-F1 principal promedia las tres clases presentes en el holdout, igual que el baseline LOSO. La matriz de confusión conserva las nueve salidas posibles. ECE usa 15 bins; SD es muestral (`ddof=1`) sobre tres semillas. La referencia B0 es pareada por seed y manifest. El test estándar, ya observado históricamente, no se utilizó para escoger miembros, pesos, épocas o hiperparámetros.

## Alcance y límites

Esta prueba no identifica causalmente un atajo de fuente ni demuestra generalización universal. Solo interviene una fuente, aunque se repiten tres inicializaciones. `multicrop-disease-maiz` se priorizó precisamente porque la caída del baseline LOSO ya era conocida; no se presenta este holdout como una fuente nunca observada por el equipo, solo como una fuente no vista **por los nuevos checkpoints B0/ShuffleNet durante train y selección**. El ensamble histórico y su delta proceden de una materialización de desarrollo anterior; el protocolo LOSO actual usa el corpus y los manifests corregidos. Además, el Macro-F1 original promedia nueve clases de test y el LOSO promedia las tres clases presentes en esta fuente. La comparación de deltas es descriptiva, no una diferencia causal entre dominios con condiciones idénticas. Tres seeds no justifican pruebas de significancia. Tampoco se ejecutó PHash en la materialización de datos vigente.

El control adicional sobre `maize-diseases` no se lanza automáticamente: no hay checkpoints LOSO B0 y ShuffleNet para esa fuente y exigiría dos nuevos entrenamientos incluso con una sola semilla. La observación principal se decide en `multicrop-disease-maiz`, la fuente cuya caída del baseline se reprodujo antes en las tres seeds.

## Resultado final: 2 de octubre de 2026

Se completaron los seis entrenamientos nuevos `validation_only` y se reutilizaron los tres checkpoints Lite0 LOSO. Hubo un primer intento B0/seed 42 interrumpido, preservado en `interrupted_attempts/attempt_1/`; el segundo intento terminó y es el único que entra al análisis. Las seis runs válidas usaron una A10G durante 10.138,23 segundos acumulados (2,816 GPU-hora de entrenamiento); no se atribuye a este tiempo un importe monetario ni se confunde con el uso total del workspace. La evaluación final usó otra app A10G, terminada sin traceback. No se entrenó ningún modelo sobre el holdout ni se modificó `seed_42`.

Los nueve `selection.lock.json` de componentes (tres Lite0 previos y seis nuevos) y los tres locks conjuntos se confirmaron antes de iniciar la primera inferencia. Cada seed tiene guard `evaluation_count=1`; el [registro](ensemble_evaluation_registry.csv) prueba los timestamps y hashes. Se verificaron los nueve checkpoints y se recompusieron Macro-F1, accuracy, ECE y métricas por clase desde los CSV de 5.816 predicciones por modelo y semilla. La reinferencia de cada Lite0 coincide por `sample_id` y etiqueta predicha con su evaluación LOSO previa. No se ajustaron miembros, pesos, hiperparámetros ni umbrales usando el holdout.

### Comparación pareada principal

| Seed | Lite0 | B0 (referencia histórica) | ShuffleNet | Ensamble | Δ ensamble − B0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0,905490 | 0,917318 | 0,913037 | 0,930055 | +0,012737 |
| 123 | 0,896964 | 0,934785 | 0,886505 | 0,931569 | −0,003216 |
| 2026 | 0,903264 | 0,919831 | 0,888009 | 0,923713 | +0,003883 |

| Métrica | B0 media ± SD | Ensamble media ± SD | Δ pareado media ± SD |
|---|---:|---:|---:|
| Macro-F1, tres clases presentes | 0,923978 ± 0,009443 | 0,928446 ± 0,004168 | +0,004468 ± 0,007993 |
| Accuracy | 0,832760 ± 0,020003 | 0,845885 ± 0,006381 | +0,013125 ± 0,016420 |
| ECE, 15 bins; menor es mejor | 0,044573 ± 0,012196 | 0,089088 ± 0,006372 | +0,044515 ± 0,005880 |

El delta Macro-F1 tiene mediana +0,003883, mínimo −0,003216, máximo +0,012737 y rango 0,015953: dos semillas mejoran y una empeora. El ensamble supera a Lite0 en las tres (Δ medio +0,026540 ± 0,007281), pero la referencia principal sigue siendo B0, tal como en el resultado histórico. El delta histórico in-distribution fue +0,008323935; aquí el promedio es aproximadamente la mitad y su SD es mayor que la propia media. La clasificación descriptiva es **B: mejora pequeña/inestable**. La observación de que nunca se había medido el ensamble fuera de fuente queda **resuelta con resultado mixto**, no con una ventaja robusta demostrada. La comparación de ambos deltas no es causal ni directamente equivalente: cambiaron materialización y número de clases evaluables.

### Clases y errores

| Seed | CR F1 B0 → ensamble | FA F1 B0 → ensamble | LN F1 B0 → ensamble |
|---:|---:|---:|---:|
| 42 | 0,989659 → 0,993254 | 0,902130 → 0,908673 | 0,860166 → 0,888239 |
| 123 | 0,995320 → 0,994286 | 0,911542 → 0,903268 | 0,897493 → 0,897152 |
| 2026 | 0,989659 → 0,992739 | 0,903670 → 0,893421 | 0,866164 → 0,884980 |
| Media | 0,991546 → 0,993426 | 0,905781 → 0,901787 | 0,874607 → 0,890124 |

El soporte por seed es CR 958, FA 1.627 y LN 3.231. La mejora media se concentra en `lethal_necrosis`; `fall_armyworm` empeora en F1 promedio. En LN, el recall B0/ensamble es 0,754875/0,799443 (42), 0,814299/0,813989 (123) y 0,764160/0,794181 (2026). Las medias son **0,777778 ± 0,031967** y **0,802538 ± 0,010260**. El F1 LN del ensamble es **0,890124 ± 0,006301**, frente a 0,874607 ± 0,020045 de B0. Su precision permanece cerca de uno (media 0,999229 frente a 0,999602 de B0), sin degradación fuerte; sin embargo, la seed 123 no mejora recall ni F1 LN. La [tabla completa](ensemble_loso_class_metrics.csv) incluye precision, recall, F1 y soporte de las tres clases para los cuatro modelos en cada seed.

| Seed | Corregidos frente a B0 | Introducidos | Todos los miembros fallan | Desacuerdo entre miembros |
|---:|---:|---:|---:|---:|
| 42 | 250 (4,30 %) | 78 (1,34 %) | 467 (8,03 %) | 1.361 (23,40 %) |
| 123 | 150 (2,58 %) | 169 (2,91 %) | 498 (8,56 %) | 1.420 (24,42 %) |
| 2026 | 194 (3,34 %) | 118 (2,03 %) | 565 (9,71 %) | 1.361 (23,40 %) |

Estos conteos son por `sample_id` sobre el mismo holdout. Los errores corregidos netos son +172, −19 y +76 respectivamente; esto explica por qué accuracy y F1 no mejoran en la seed 123. Los [desacuerdos por muestra](ensemble_disagreements.csv) incluyen etiqueta, predicciones individuales, confianza del ensamble, corrección/error introducido y fallos compartidos; el [resumen JSON](ensemble_loso_summary.json) incorpora porcentajes, confianza media, desacuerdos por par y errores exclusivos. La diversidad observada no asegura que el promedio de probabilidades mejore todas las seeds. Además, el ensamble empeora el ECE en las tres; cualquier uso clínico o operativo de su confianza necesitaría calibración independiente, fuera de este protocolo.

### Procedencia y archivos

Los run IDs nuevos B0/ShuffleNet son, por seed: 42 `20261002_143310` / `20261002_150615`; 123 `20261002_153038` / `20261002_160041`; 2026 `20261002_162642` / `20261002_165936`. Lite0 reutiliza `20260930_145541`, `20261001_170150` y `20261001_173226`. Las rutas locales y remotas de los nueve `best.pth`, su SHA-256, los hashes de manifests y la configuración de procedencia se registran en [referencias](manifest_checkpoint_references.json). Los archivos completos —checkpoints, locks, guards y predicciones— están en `corn-outputs:/ensemble_loso/multicrop-disease-maiz/` y la copia local ignorada `outputs/ensemble_loso/multicrop-disease-maiz/`; los tres Lite0 originales siguen en `corn-outputs:/loso/efficientnet_lite0_baseline/multicrop-disease-maiz/`.

Los artefactos versionables principales son [resultados pareados](ensemble_loso_results.csv) (SHA-256 `5a3d12f1b679a7c304f493087c57670cb00c0b70ee0210759047da474f55dbba`), [componentes](ensemble_component_results.csv) (`bf89b7d5b3c5ecdc27c3fed1f33e833f8943c75dc53d105715311c1ef1688d41`), [clases](ensemble_loso_class_metrics.csv) (`526db4ac4d8161917fe3e6a5b197eb8161c0b5ca864a92fde48879c156866d1d`), [desacuerdos](ensemble_disagreements.csv) (`5921e42e275caccc05c3c6c9af987af1d82db93112c8a99729d5e994a96bdacd`), [resumen](ensemble_loso_summary.json) (`be4320c64c9349a5abf224de244cf9bdf5827b39ac6daf315524bf7e582e8c7c`), [registro de evaluación](ensemble_evaluation_registry.csv) (`7beb03b4e5c06e78502f348f4b549927ed517bd6695d31332c6f4d0b7f67b8da`) y [referencias](manifest_checkpoint_references.json) (`45100d2b766be46a95f7a6e229031a35d6e5de509f126874215ff70c884c1c28`). También se guardan matrices de confusión por seed y agregada, una visualización normalizada y deltas pareados. El análisis agregado no sustituyó los CSV de predicciones originales.

## Reproducción de las verificaciones

Los scripts `scripts/modal/ensemble_loso_train.py` y `scripts/modal/ensemble_loso_finalize.py` tienen modo preflight por defecto; `--execute` inicia respectivamente GPU de entrenamiento y de evaluación. El primero omite slots completos verificados y conserva un intento interrumpido bajo `interrupted_attempts/`; el segundo confirma todos los locks antes de inferir y rechaza cualquier evaluación iniciada pero incompleta. No se deben ejecutar mientras haya otra app escribiendo en el mismo slot. El reporte local se recompone, después de descargar el árbol remoto `corn-outputs:/ensemble_loso/multicrop-disease-maiz/`, con `python -m scripts.experiments.ensemble_loso_report`. Este último paso solo lee checkpoints para verificar sus hashes; no hace inferencia ni entrena.

Verificación final: 44 tests relevantes pasaron, incluidos los del protocolo nuevo, estabilidad de fuente, selección LOSO y fusión; Ruff pasó para los cinco archivos Python nuevos y `git diff --check` no reportó problemas. La comprobación incluyó alineación por `sample_id`, exclusión de fuente en train/validation, pesos y regla de fusión, hashes de checkpoints, orden de locks, guard de evaluación única y estadísticas pareadas con `ddof=1`. El control LOSO `maize-diseases` sigue pendiente de dos entrenamientos nuevos para una seed y no se ejecutó en este cierre. Tampoco se inició selección definitiva ni entrenamiento formal.
