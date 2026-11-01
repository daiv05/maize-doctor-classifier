# Historia del proyecto

Cronología de DoctorMaiz reconstruida desde informes y artefactos. **VIGENTE** describe el corte actual, **HISTÓRICO** conserva contexto y **EXPERIMENTAL** marca líneas que no pasaron a producción.

## Mayo a julio de 2026: datos, pipeline y arquitecturas

El objetivo fue un clasificador móvil y sin conexión para nueve condiciones de hojas de maíz: enfermedades, gusano cogollero, hoja sana y deficiencias de N, P y K. La primera dificultad fue de datos: repositorios con etiquetas, fondos, cámaras y estructuras incompatibles. Entre junio y julio se consolidó `clean/<clase>/{lab,real}`, se construyó el pipeline de preparación con `CornDataset` y se implementó el entrenamiento principal.

En julio se exploraron ocho arquitecturas ligeras. Solo EfficientNet-B0, ShuffleNet-V2-x1.0 y EfficientNet-Lite0 tienen scores comparables sobre el perfil histórico de 1 503 imágenes de test: 0.9146, 0.9030 y 0.8951 de Macro-F1. EfficientNet-B4, MobileNetV3, FastViT-T8 y GhostNetV2 quedaron como exploración sin ficha comparable. Las cifras de esta fase son **HISTÓRICAS** y no describen el corpus actual.

## Segmentación: de aislar la hoja a controlar el fallo

El clasificador podía mirar demasiado fondo u otras hojas, por lo que se introdujo segmentación con YOLO: selección de una instancia foliar por área, centralidad y confianza, descarte de máscaras degeneradas y fallback a la imagen original. El run sobre corpus segmentado `20260907_163546` obtuvo Macro-F1 0.7191 y accuracy 0.8903, muy por debajo del Lite0 de imagen completa. Una máscara puede eliminar contexto útil o escoger la hoja equivocada, así que la segmentación queda **EXPERIMENTAL**, opt-in, y no es un preprocesamiento universal.

## Agosto de 2026: ampliación, exportación y OOD

Se amplió el corpus, se entrenaron los runs principales y se implementó la exportación ONNX/TFLite con paridad y `labels.json`. Del run `20260812_221429` salió el EfficientNet-Lite0 de la etapa 2: Macro-F1 0.9468 y accuracy 97.91 % sobre su test estratificado. A finales de agosto se añadió la detección fuera de distribución por distancia de Mahalanobis relativa.

## Septiembre de 2026: Optuna, ensamble, validación cruzada y procedencia

Optuna, ensamble y validación cruzada entraron a comienzos de septiembre. Los estudios históricos mostraron que transferir hiperparámetros entre arquitecturas puede perjudicar a Lite0: sus runs afinados quedaron entre 0.9343 y 0.9386 de Macro-F1, por debajo de 0.9468. El ensamble de Lite0, B0 y ShuffleNet alcanzó 0.9567 de Macro-F1 y 98.29 % de accuracy sobre el test estratificado.

La validación cruzada 2×2 separó configuración y partición. La estratificada dio 0.9311 ± 0.0099 (± es la semiamplitud del IC95; SD 0.0080) y la agrupada por fuente, 0.6026 ± 0.1240 (IC95), un Macro-F1 calculado sobre las clases presentes en cada pliegue; el Macro-F1 convencional de esa misma validación es 0.4359. La diferencia de protocolo superó el efecto de los hiperparámetros.

La auditoría de procedencia y el LOSO histórico mostraron que fondo, sensor, compresión y composición por fuente pueden actuar como atajos. Balancear fuentes, sustituir fondos (BackMix) y endurecer augmentations no cerró la brecha de forma consistente. Fue un resultado negativo útil: justifica reportar el desempeño dentro de fuentes conocidas por separado de la generalización a dominios nuevos.

## Identidad, manifiestos y contratos

Dos líneas de desarrollo evolucionaron en paralelo: una con caché de imágenes, tope por clase, PHash y un splitter agrupado por fuente; otra con identidad y contratos de runs. Se integraron de forma selectiva y con tests.

El cambio central fue separar identidad, contenido y relación: `sample_id` identifica la muestra, SHA-256 los bytes y source/group el dominio. Se eliminó un fallo por el que una imagen ilegible podía ser sustituida por la fila siguiente: ahora el batch conserva `(image, label, sample_id)` y un fallo identifica esa misma muestra. Se añadieron `master_manifest.csv`, locks, auditorías, conflictos entre etiquetas y exclusiones verificadas.

El corpus vigente tiene 33 437 archivos descubiertos y válidos, ocho exclusiones explícitas y 33 429 muestras elegibles. `seed_42` contiene 23 400 / 5 014 / 5 015 y `seed_42_source_grouped`, 16 554 / 7 809 / 9 066.

## 21 de septiembre: protocolos y baseline oficial

Tratar once `source_id` como grupos indivisibles produjo 49.52 / 23.36 / 27.12 %, clases ausentes en validación, train Macro-F1 0.9664 y validación 0.2669. El split no era inútil: medía generalización entre fuentes, pero no servía para selección balanceada. Se conservaron dos protocolos.

El run `20260921_204608` es el EfficientNet-Lite0 oficial: 47 épocas, mejor época 39, validación Macro-F1 0.9561, test Macro-F1 0.9480 y accuracy 97.81 %. Es un baseline de desarrollo sobre fuentes conocidas; el salto frente al benchmark source-grouped (0.4007) se debe al cambio de protocolo.

## 24 de septiembre a 2 de octubre: HPO y estabilidad

El HPO formal de Lite0 se cerró el 24 de septiembre con 25 intentos (8 COMPLETE, 15 PRUNED, 2 FAIL) tras enmendar el presupuesto de 60 a 25. El trial 0 mejoró la validación (+0.12 pp) pero no el test único (0.9431 frente a 0.9480), así que no se adoptó.

El estudio de estabilidad baseline/LOSO se cerró el 1 de octubre con tres semillas (42, 123 y 2026): el baseline obtuvo 0.9435 ± 0.0052 de Macro-F1 (SD muestral), LOSO sin `maize-diseases` 0.9933 ± 0.0006 y LOSO sin `multicrop-disease-maiz` 0.9019 ± 0.0044. El ensamble histórico bajo LOSO multicrop (2 de octubre) dio un resultado mixto frente a B0 y peor calibración.

## 19 de octubre: reproducción y comparación pareada

El baseline se reentrenó el 19 de octubre (`20261019_203033`) y su `best.pth` es idéntico bit a bit al oficial. El entrenamiento del HPO trial 0 (`20261019_211317`) completó la segunda de las diez corridas previstas del estudio pareado baseline/HPO, solo de validación; la diferencia de +0.0012 no es interpretable (McNemar p = 0.83).

## 1 de noviembre: modelo oficial en la app

El oficial se exportó a ONNX y TFLite con paridad aprobada en FP32. El TFLite int8 (3.74 MB) obtuvo Macro-F1 0.9489 sobre el test completo y la app dejó de desplegar el Lite0 de agosto. El detector fuera de distribución se recalibró con el oficial (umbral 106.37). Una evaluación en 97 fotos de campo externas al corpus acertó 41/97 y el detector rechazó 49/97: la generalización fuera del corpus sigue siendo la limitación principal. Detalles en [Resultados](../resultados/index.md) y [Despliegue](../deployment/prototipo.md).
