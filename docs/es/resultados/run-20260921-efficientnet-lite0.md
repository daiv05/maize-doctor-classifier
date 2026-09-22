# Baseline principal de desarrollo — EfficientNet-Lite0 `20260921_204608`

Esta corrida es el primer reentrenamiento completo de `EfficientNet-Lite0` sobre la materialización corregida de `seed_42`. Es el **baseline vigente de desarrollo**, pero **no sustituye todavía al modelo TFLite desplegado**.

## Identidad y reproducibilidad

| Campo | Valor |
|---|---|
| Estado | Completo |
| Artefacto | `corn-outputs:/main/efficientnet_lite0/20260921_204608/best.pth` |
| SHA-256 del checkpoint | `860180f33bf1749ffee9f3cccb9569a47afd9ca0fcef52713e18f99f82a57859` |
| SHA-256 de la configuración | `53cc091e505057f6751a7b6151f403e03f830fac3c0966df9ba51e2df2814e8b` |
| Split | `seed_42` — estratificado por `label + environment` |
| Muestras | 23,400 train / 5,014 val / 5,015 test |
| SHA-256 del manifiesto del split | `0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8` |
| Épocas | 47 ejecutadas de 60 solicitadas |
| Mejor época | 39 |
| Mejor macro F1 de validación | 0.956086 |

La configuración usó lote 32, `learning_rate=1e-4`, `weight_decay=1e-4`, *warmup* de 3 épocas, scheduler coseno, `label_smoothing=0.1`, pesos `sqrt_inverse`, paciencia 8, pesos ImageNet y CLAHE desactivado. No se utilizó sampler ponderado.

## Resultado observado

| Métrica | Valor |
|---|---:|
| Accuracy | **0.978066** |
| Macro F1 | **0.948002** |
| Loss | 0.761194 |
| Macro F1 con deficiencias agrupadas | 0.978385 |
| Accuracy con deficiencias agrupadas | 0.984048 |

## Rendimiento por clase

| Clase | Soporte | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Roya común | 339 | 0.9911 | 0.9853 | 0.9882 |
| Gusano cogollero | 728 | 0.9796 | 0.9890 | 0.9843 |
| Mancha gris | 289 | 0.9498 | 0.9170 | 0.9331 |
| Hoja sana | 1,311 | 0.9916 | 0.9931 | 0.9924 |
| Necrosis letal | 963 | 0.9969 | 0.9958 | 0.9964 |
| Deficiencia de nitrógeno | 127 | 0.8435 | 0.9764 | 0.9051 |
| Tizón foliar del norte | 1,025 | 0.9747 | 0.9785 | 0.9766 |
| Deficiencia de fósforo | 140 | 0.9919 | 0.8786 | 0.9318 |
| Deficiencia de potasio | 93 | 0.8427 | 0.8065 | 0.8242 |

La clase más débil es deficiencia de potasio. Los errores dominantes fueron mancha gris → tizón foliar del norte (18), potasio → nitrógeno (15), tizón foliar del norte → mancha gris (12), fósforo → potasio (7), gusano cogollero → hoja sana (6), fósforo → gusano cogollero (5) y fósforo → nitrógeno (5).

## Entorno, fuente y calibración

| Entorno | Muestras | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Laboratorio | 533 | 0.9662 | 0.9383 |
| Campo real | 4,482 | 0.9795 | 0.9269 |

Los macro F1 por entorno no son directamente comparables porque no tienen el mismo conjunto de clases con soporte. Por fuente, los rendimientos más bajos aparecen en `Maize Deficiency Scanner` (84.0 %, 25 muestras), `Maize in Field` (83.8 %, 136) y `Maize 2` (86.6 %, 127). Esas fuentes y las confusiones entre deficiencias son los primeros focos para revisión cualitativa.

La calibración del test reporta ECE de 0.1397 y Brier binario de acierto de 0.0380. La confianza media fue 0.8444 en los 4 905 aciertos y 0.6872 en los 110 errores; 15 errores tuvieron confianza igual o superior a 0.90.

## Interpretación

Frente al `EfficientNet-Lite0` históricamente desplegado (Macro-F1 0.9468), el baseline actual difiere aproximadamente +0.0012 en el test estratificado. Las corridas proceden de materializaciones distintas, por lo que esa resta es descriptiva y no demuestra una mejora. Potasio presenta el F1 más bajo del run (0.8242), y las confusiones N/P/K justifican una revisión cualitativa. La calibración aconseja revisar método y umbrales antes de presentar la confianza como certeza diagnóstica.

## Limitaciones y decisión de promoción

`seed_42` mezcla las fuentes conocidas entre train, validación y test; es adecuado para desarrollo *in-distribution*, pero no demuestra generalización a una cámara, parcela o repositorio nuevo. Además, esta materialización garantiza separación por identificador, SHA-256 y grupo efectivo, pero se generó con `deduplicate_perceptual=false`: la posible fuga por imágenes casi duplicadas no fue medida.

Antes de promover el checkpoint se debe:

1. ejecutar la evaluación agrupada por fuente o *leave-one-source-out*;
2. auditar solapamiento perceptual entre particiones;
3. revisar los 110 errores, especialmente las deficiencias y las tres fuentes más débiles;
4. exportar a TFLite Int8 y comprobar paridad y macro F1 sobre el test completo;
5. validar latencia, OOD y calibración en el dispositivo objetivo.

Los parámetros y métricas exactos se conservan en [`run_20260921_lite0_summary.json`](./evidencia/run_20260921_lite0_summary.json), [`classification_report.csv`](./evidencia/run_20260921_204608_test_classification_report.csv), [`confusion_matrix.csv`](./evidencia/run_20260921_204608_test_confusion_matrix.csv), [`calibration.json`](./evidencia/run_20260921_204608_test_calibration.json), desgloses por [ambiente](./evidencia/run_20260921_204608_test_by_environment.csv) y [fuente](./evidencia/run_20260921_204608_test_by_source.csv), y la corrida está en el [manifiesto de auditoría](./evidencia/manifiesto_corridas.csv).
