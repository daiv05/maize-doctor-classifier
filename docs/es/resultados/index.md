# Resultados

Modelo oficial: EfficientNet-Lite0, corrida `20260921_204608`, entrenada sobre el split `seed_42` (33 429 muestras: 23 400 train, 5 014 validación y 5 015 prueba, `manifest.lock` `0db3ff3e…`). Es el modelo que despliega la app en su exportación TFLite int8. Los artefactos de cada cifra están en el [registro de evidencias](/es/reproducibilidad/evidencias).

## Modelo oficial

| Métrica | Valor |
|---|---:|
| Macro-F1 de prueba | 0.9480 |
| Accuracy de prueba | 97.81 % |
| Macro-F1 de validación | 0.9561 |
| Mejor época | 39 de 47 ejecutadas |
| `best.pth` | 13 785 701 B, SHA-256 `860180f3…` |
| ECE de prueba | 0.1397 |

La corrida se reprodujo bit a bit el 2026-10-19 (`20261019_203033`, mismo `best.pth`). Rendimiento por clase, matriz de errores y calibración están en la [ficha del modelo oficial](/es/resultados/run-20260921-efficientnet-lite0). La clase más débil es la deficiencia de potasio.

### Tres semillas

| Semilla | Run | Macro-F1 validación | Macro-F1 prueba | Accuracy prueba |
|:---:|---|---:|---:|---:|
| 42 | `20260921_204608` | 0.956086 | 0.948002 | 97.81 % |
| 123 | `20261001_140756` | 0.960281 | 0.937867 | 97.31 % |
| 2026 | `20261001_145025` | 0.952856 | 0.944597 | 97.75 % |
| Media ± SD | | 0.956408 ± 0.003723 | 0.943489 ± 0.005158 | 97.62 % ± 0.27 pp |

La SD es muestral (n = 3). Diferencias menores a 0.01 de Macro-F1 medidas con una sola semilla están en el orden de la variación por inicialización, por lo que no se interpretan como mejoras.

## Optimización de hiperparámetros

El estudio `efficientnet_lite0_seed42_hpo_v1` ejecutó 25 intentos (8 completos, 15 podados, 2 fallidos por interrupción). El ganador, trial 0, mejoró la validación (0.957292 frente a 0.956086) pero quedó por debajo en prueba (Macro-F1 0.943125 frente a 0.948002), por lo que no se adoptó. La comparación pareada baseline contra HPO completó 2 de 10 entrenamientos (solo validación; diferencia de +0.0012, McNemar p = 0.83) y no es interpretable. Protocolo y detalle en [HPO](/es/experimentos/hpo) y [multi-semilla](/es/experimentos/multiseed).

## Generalización entre fuentes

La partición `seed_42` reparte las mismas fuentes entre train y prueba, por lo que mide desempeño dentro de fuentes conocidas.

| Evaluación | Macro-F1 | Detalle |
|---|---:|---|
| LOSO sin `maize-diseases` (3 semillas) | 0.993264 ± 0.000600 | Modelo entrenado sin esa fuente y evaluado en ella |
| LOSO sin `multicrop-disease-maiz`, Lite0 (3 semillas) | 0.901906 ± 0.004422 | Caída concentrada en `lethal_necrosis` |
| Benchmark source-grouped (`20260921_180112`) | 0.400665 | Fuentes completas retenidas; validación 0.266927 |
| Fotos de campo externas al corpus (97) | 41 aciertos (42 %) | El detector OOD rechaza 49 |

La brecha no es uniforme: depende de qué fuente se retiene y de qué clases aporta. Los protocolos y su lectura están en [Generalización entre fuentes](/es/resultados/generalizacion).

### Fotos de campo

Se evaluó el modelo oficial (FP32, imagen completa) sobre 97 fotos propias, ajenas al corpus y no versionadas; una de ellas está en el corpus.

| Carpeta | Fotos | Aciertos |
|---|---:|---:|
| `gray_leaf_spot` | 10 | 10 |
| `fall_armyworm` | 10 | 10 |
| `nitrogen_deficiency` | 10 | 6 |
| `northern_corn_leaf_blight` | 10 | 5 |
| `potassium_deficiency` | 9 | 4 |
| `common_rust` | 18 | 2 |
| `phosphorus_deficiency` | 10 | 2 |
| `healthy` | 10 | 2 |
| `lethal_necrosis` | 10 | 0 |

El Lite0 de la etapa 2 acierta 39 de las mismas 97, por lo que la brecha no depende del cambio de modelo. Esta cifra no es comparable con el Macro-F1 de 0.9480: aquel mide fuentes presentes en el entrenamiento y esta, imágenes de otro origen. Es la principal limitación del sistema.

## Exportación móvil

| Artefacto | Resultado |
|---|---|
| ONNX y TFLite FP32 | Paridad aprobada (máx. \|Δprob\| 3e-6 y 0) |
| TFLite int8 `model_int8.tflite` | 3 743 728 B (72.9 % menos que el FP32 de 13.79 MB), SHA-256 `bb31467a…` |
| Prueba completa (5 015), int8 | Macro-F1 0.9489 y accuracy 97.87 % (PyTorch 0.9480 y 97.81 %); acuerdo 99.74 % |
| Int8 por entorno | Laboratorio (n = 533) accuracy 96.81 %, Macro-F1 0.9420; campo (n = 4 482) accuracy 97.99 %, Macro-F1 0.9208 |
| Paridad de muestra (30 imágenes) | No aprobada por una imagen límite (máx. \|Δprob\| 0.198 frente al umbral 0.15) |

### Detector fuera de dominio

Distancia de Mahalanobis relativa sobre las características de la penúltima capa (`ood_stats.json`, esquema 4): PCA a 182 componentes (99.01 % de la varianza) y umbral 106.37, el percentil 95 de validación. Detecta 5 de 5 imágenes sintéticas (negro, blanco, gris, ruido y azul; RMD entre 539 y 955) y rechaza el 3.8 % de 500 imágenes de prueba. Método en [Detección OOD](/es/deep-learning/ood-detection).

### Teléfono

Medido en un Motorola edge 40 neo (Android 15) sobre 143 imágenes: preprocesado 98 ms de mediana (p95 279 ms) e inferencia 23 ms, frente a la meta de 300 ms entre ambos. El intérprete int8 reproduce las predicciones del int8 del servidor en 143 de 143 imágenes. Detalle en [Prototipo en dispositivo](/es/deployment/prototipo).

## Resultados de la etapa 2

Corridas anteriores sobre otra materialización del corpus (33 433 muestras), conservadas como referencia. No son comparables punto a punto con el modelo oficial: solo comparten 2 539 imágenes de prueba, y sobre ellas McNemar da 23 frente a 22 discordancias (p = 1.00), sin diferencia medible.

| Corrida | Macro-F1 prueba | Accuracy |
|---|---:|---:|
| Lite0 final `20260812_221429` (desplegado en la app hasta 2026-11-01) | 0.9468 | 97.91 % |
| B0 `20260910_170120` | 0.9483 | 97.97 % |
| ShuffleNet-V2 `20260910_184521` | 0.9330 | 97.31 % |
| Ensamble por voto suave de esos tres | 0.9567 | 98.29 % |
| Ensamble `ensemble/` (B0 `20260811_211306` y ShuffleNet `20260812_185239`) | 0.9551 | |
| Lite0 con Optuna `20260912_230941` | 0.9379 | |
| Lite0 con hiperparámetros del trial 12 de B0 `20260912_233321` | 0.9386 | |
| Lite0 solo con lr y batch de B0 `20260913_031130` | 0.9343 | |
| Lite0 semillas 1 y 2 (`20260913_151826`, `20260913_152116`) | 0.9448 y 0.9375 | |

El ensamble sigue siendo reconstruible porque los tres miembros están respaldados. Bajo LOSO su ventaja se reduce y es inestable (ver [generalización](/es/resultados/generalizacion)).
