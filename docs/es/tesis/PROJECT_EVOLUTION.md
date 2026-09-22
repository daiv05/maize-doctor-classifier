# Evolución técnica de DoctorMaiz

Esta cronología reconstruye el proyecto desde Git, informes y artefactos. Las fechas son fechas de commit salvo cuando se identifica explícitamente un `run_id`. **VIGENTE** describe el corte actual; **HISTÓRICO** conserva contexto; **PENDIENTE** no es un resultado.

## 1. Problema y primera construcción — mayo a junio de 2026

**Problema.** Construir un clasificador móvil y offline para nueve condiciones de hojas de maíz, incluyendo enfermedades, gusano cogollero, hoja sana y deficiencias N/P/K. El reto no era solo clasificar: el corpus debía unir fuentes heterogéneas, funcionar en campo y caber en un dispositivo.

**Evolución.** La documentación inicial aparece el 19 de mayo (`0c4f331`, `fc841c2`). Entre el 3 y el 12 de junio se incorporaron herramientas de inventario, limpieza y deduplicación (`9bc73d5`, `dc0bb5f`, `6130cec`). El 16 de junio surgieron el pipeline de preparación y `CornDataset` (`e9fbc5e`, `d4c1806`); el 22 de junio quedaron conectados preparación, transforms y primer flujo de modelado (`a47cb94`).

**Resultado.** Se obtuvo un corpus limpio organizado por `clean/<clase>/{lab,real}` y una primera evaluación estratificada. Los informes de esta fase están en `reports/firts-phase/`. Sus cifras son **HISTÓRICAS** y no describen el corpus actual.

**Limitación aprendida.** La identidad dependía de rutas/orden, los dominios de origen podían mezclarse y el corpus era muy desbalanceado.

## 2. Exploración de arquitecturas y baselines — julio de 2026

El 4 de julio se incorporaron EfficientNet-B4 y MobileNetV3 large/small (`215658d`); el 8 de julio se fijaron los baselines principales (`a2562df`). La evidencia publicada solo permite reportar scores completos para EfficientNet-B0, ShuffleNet-V2-x1.0 y EfficientNet-Lite0 sobre el perfil histórico de 1 503 imágenes de test: 0.9146, 0.9030 y 0.8951 de Macro-F1, respectivamente.

| Arquitectura | Qué se buscaba | Evidencia de resultado | Decisión |
|---|---|---|---|
| EfficientNet-B0 | referencia de precisión/eficiencia | baseline y runs principales | conservar |
| ShuffleNet-V2-x1.0 | extremo ligero | baseline y runs principales | conservar |
| EfficientNet-Lite0 | cuantización móvil | baseline, exportación y runs | priorizar para móvil |
| EfficientNet-B4, MobileNetV3 large/small, FastViT-T8, GhostNetV2-100 | explorar alternativas | registro de implementación/selección, sin ficha completa comparable | **HISTÓRICO; score no verificable aquí** |

El pipeline principal de entrenamiento se implementó el 26 de julio (`2f95c0d`). Por eso la frase histórica “loop pendiente” dejó de ser válida.

## 3. Segmentación: de aislar la hoja a controlar el fallo — julio a septiembre de 2026

**Problema observado.** Fondos, otras hojas y objetos de captura podían aportar atajos. El trabajo preparatorio para YOLO aparece el 21 de julio (`9501c52`) y el segmentador se registra como completado el 29 de julio (`bd67c21`). El 7 de agosto se añadió el perfil de máscara negra (`e63a886`).

**Cambio.** En septiembre se añadió presegmentación GPU y entrenamiento sobre corpus segmentado (`be8e028`, `0de97da`). La implementación selecciona una instancia foliar por área, centralidad y confianza, descarta máscaras degeneradas o pequeñas y permite fallback a la imagen original; no aplica cualquier máscara ciegamente.

**Resultado observado.** El run segmentado `20260907_163546` obtuvo Macro-F1 0.719082 y accuracy 0.890329 (`manifiesto_corridas.csv`), muy por debajo del Lite0 histórico de imagen completa. Esto muestra que eliminar contexto o elegir una hoja de escenas multihoja cambia la tarea y puede destruir señal útil.

**Decisión.** La segmentación queda **EXPERIMENTAL**, no como preprocessing universal. La comparación formal de perspectiva completa frente a segmentada, con *quality gate* y reglas para ambigüedad multihoja, sigue **PENDIENTE**.

## 4. Despliegue, OOD y artefactos — agosto de 2026

Entre el 11 y el 17 de agosto se amplió el corpus, se entrenaron runs principales y se implementó exportación ONNX/TFLite con paridad y `labels.json` (`0476b4c`, `bdb3486`, `1e9100c`, `1df66cb`). Del run `20260812_221429` salió el EfficientNet-Lite0 histórico desplegado: Macro-F1 0.946789 y accuracy 0.979063.

El 27–29 de agosto se añadió exposición de features y detección OOD por distancia Mahalanobis/RMD (`b4ed94b`, `2a86163`, `ae5d1ed`). Es evidencia de ingeniería de despliegue, no validación de producción del nuevo checkpoint.

## 5. HPO, ensamble y validación cruzada — 8 a 13 de septiembre de 2026

Optuna, ensamble y CV entraron el 8 de septiembre (`aed8ada`, `13c9e0d`, `77965e0`). Los estudios históricos muestran que transferir hiperparámetros entre arquitecturas puede perjudicar a Lite0: sus runs afinados documentados quedaron entre 0.9343 y 0.9386 de Macro-F1, por debajo de 0.9468.

El ensamble histórico de Lite0+B0+ShuffleNet alcanzó 0.956657 Macro-F1 y 0.982851 accuracy. Es un resultado **HISTÓRICO** sobre test estratificado, no una métrica del baseline actual.

La CV 2×2 separó configuración y partición. El brazo estratificado obtuvo 0.9311 ± 0.0099; el agrupado por fuente, 0.6026 ± 0.1240. La diferencia de protocolo superó el efecto del ajuste de hiperparámetros. Este hallazgo condujo a tratar procedencia como pregunta experimental propia.

## 6. Procedencia, LOSO y resultados negativos — 9 a 12 de septiembre de 2026

La auditoría de procedencia y LOSO se añadió entre `17d371a` y `96d7172`. Se encontró que fondos, sensor, compresión y composición por fuente podían actuar como atajos. Balancear fuentes, sustituir fondos (*BackMix*) y endurecer augmentations no cerró de forma consistente la brecha (`docs/es/provenance/evidencia/`).

Este fue un resultado negativo útil: justificó reportar desempeño dentro de fuentes conocidas separado de generalización a dominios nuevos. El LOSO histórico se conserva para análisis por clase; un LOSO final del modelo posterior al HPO está pendiente.

## 7. Ramas paralelas: `master` y `dev-abner`

El ancestro común verificable es `8a19ff9` (11 de septiembre). Después de él, `master` acumuló caché, tope por clase, deduplicación perceptual, agrupación por fuente, evaluación y documentación; `dev-abner` culminó en `c6a8c33` con identidad, manifests y contratos alternativos. No era una versión linealmente más nueva.

| Funcionalidad | Evidencia de origen | Decisión en `dev-abner2` |
|---|---|---|
| `sample_id` y asociación de predicciones | `dev-abner:c6a8c33`, adaptado en `80bff4` | adaptar y transportar en el tuple del dataset |
| SHA-256/manifiestos/locks | ideas en `c6a8c33`; endurecimiento `711bd10` | adaptar con exclusiones y escritura canónica |
| Run contracts | `dev-abner:c6a8c33`; reimplementación `96be6f1` | adaptar y versionar |
| `ImageCache` | `master:bceb1ff` | conservar |
| `max_per_class` con política minoritaria | `master:80e758a`, test `1331eb2` | conservar |
| PHash/deduplicación perceptual | `master:a35d7aa` | conservar como capacidad opt-in |
| `SourceGroupedSplitter` voraz | `master:a35d7aa` | conservar |
| `GroupShuffleSplit` dentro del splitter estratificado | `dev-abner:c6a8c33` | no portar; separar protocolos |
| SHA-256 en `__getitem__` | `dev-abner:c6a8c33` | no portar; hashear durante preparación |

No se hizo cherry-pick completo porque habría eliminado o cambiado garantías valiosas de `master`. La integración fue selectiva y cubierta por tests.

## 8. Identidad explícita e incidente de sustitución — 18 de septiembre de 2026

**Problema.** La implementación de `master` en `1110207` intentaba hasta cinco filas: si fallaba `idx`, cargaba `(idx + attempt) % len(dataset)`. Eso podía devolver otra imagen/etiqueta para una posición solicitada y hacer imposible asociar la predicción con la muestra original.

**Solución.** `80bff4` incorporó `sample_id`; `600ebb9` eliminó el fallback y fijó el contrato `(image, label, sample_id)`. El error ahora identifica el mismo `idx`, `sample_id`, ruta, etiqueta y ruta resuelta.

**Garantía añadida.** `tests/data/test_dataset_fail_fast.py` prueba que no hay sustitución, `None`, filtrado por `collate_fn` ni mutación del DataFrame. Se conservaron `ImageCache`, `max_per_class`, clases minoritarias, transforms, `class_to_idx` y samplers.

## 9. Manifiesto, grupos y contratos — 18 a 22 de septiembre de 2026

`711bd10` añadió `master_manifest.csv`, locks, auditorías, conflictos cross-label y exclusiones verificadas. `600ebb9` incorporó grupos efectivos y separó el split estratificado del benchmark source-grouped. `96be6f1` cerró contratos de runs y consumidores.

El corpus actual tiene 33 437 archivos descubiertos/válidos, ocho exclusiones explícitas y 33 429 muestras elegibles. `seed_42` contiene 23 400/5 014/5 015; `seed_42_source_grouped`, 16 554/7 809/9 066.

## 10. Incidente source-grouped y baseline actual — 21 de septiembre de 2026

El primer entrenamiento (`20260921_180112`) usó fuentes completas: once grupos produjeron 49.52/23.36/27.12 %, validación sin dos clases, train Macro-F1 0.9664 al cierre y mejor validación 0.2669. Test obtuvo 0.4007.

La conclusión no fue “el split es malo”, sino “responde generalización cross-source y no selección balanceada”. Se conservó como `seed_42_source_grouped` y `seed_42` volvió a estratificación `label + environment` con nueve clases en cada partición.

El run `20260921_204608` es el **baseline principal de desarrollo**: mejor época 39, 47 épocas, validación Macro-F1 0.956086, test Macro-F1 0.948002 y accuracy 0.978066. El salto frente al run anterior se debe principalmente al cambio de protocolo; no demuestra una mejora mágica del modelo.

## Línea de tiempo resumida

| Fecha/Etapa | Problema | Decisión | Implementación | Resultado | Evidencia | Estado actual |
|---|---|---|---|---|---|---|
| 2026-05-19 | definir objetivo | clasificador móvil offline | documentación inicial | alcance del sistema | `0c4f331`, `fc841c2` | HISTÓRICO |
| 2026-06-03 a 12 | fuentes heterogéneas | limpiar y consolidar | scripts + dedup | corpus inicial | commits `9bc73d5`–`6130cec` | HISTÓRICO |
| 2026-06-16 a 22 | preparar/consumir datos | pipeline reproducible | splitters, dataset, transforms | primer pipeline | `e9fbc5e`, `d4c1806`, `a47cb94` | EVOLUCIONADO |
| 2026-07-04 a 08 | elegir arquitectura | explorar ocho, fijar tres | registro de modelos | baselines verificables | `215658d`, `a2562df`, informe fase 1 | HISTÓRICO |
| 2026-07-21 a 2026-09-07 | fondo/multihoja | experimentar segmentación | YOLO + presegmentación | run 0.7191 Macro-F1 | `9501c52`, `be8e028`, run `20260907_163546` | EXPERIMENTAL |
| 2026-08-11 a 17 | ampliar y desplegar | Lite0 + export Int8 | dataset, ONNX/TFLite | run desplegado 0.9468 | `0476b4c`, `20260812_221429` | HISTÓRICO desplegado |
| 2026-09-08 a 13 | optimizar/medir robustez | HPO, ensamble, CV | Optuna, voting, K-fold | ensamble 0.9567; CV agrupada 0.6026 ± 0.1240 | evidencia resultados | HISTÓRICO |
| 2026-09-09 a 12 | fuga de procedencia | LOSO e intervenciones | provenance suite | brecha no cerrada | evidencia provenance | HISTÓRICO útil |
| 2026-09-18 | identidad implícita | `sample_id` explícito | `80bff4` | predicción trazable | tests de identidad | VIGENTE |
| 2026-09-18 | sustitución silenciosa | fail-fast misma muestra | `600ebb9` | no cambia `idx` | tests fail-fast | VIGENTE |
| 2026-09-18 a 22 | integridad/run drift | manifests y contratos | `711bd10`, `96be6f1` | hashes y validación | locks/summaries | VIGENTE |
| 2026-09-21 | source grouping grueso | separar desarrollo/cross-source | dos protocolos | 0.4007 vs 0.9480, no comparables | runs `180112`/`204608` | VIGENTE |
| Próxima fase | seleccionar HP actuales | Optuna 60 trials sin test | plan documentado | sin resultado | `experimentos/hpo.md` | PENDIENTE |
