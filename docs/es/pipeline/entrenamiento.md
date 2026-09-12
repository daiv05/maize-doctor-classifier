# Entrenamiento principal: corridas revisadas

Las corridas recibidas el 11 de septiembre de 2026 son resultados experimentales,
no una certificación de producción. Ver [auditoría y procedencia](../../reviews/2026-09-11-corridas-recibidas)
y [reparación integral](../../reviews/2026-09-11-reparacion-integral).

## Datos y configuración comprobables

Los CSV recibidos tienen 23,403 / 5,015 / 5,015 filas (33,433 en total), nueve clases,
con partición estratificada por clase y entorno. La revisión HF anuncia 33,437 imágenes
antes de deduplicar. Los conteos 33,438 y 23,407/5,016/5,015 publicados anteriormente no
describen estos CSV. No son el mismo holdout que Etapa 2.

La descarga completa del 12/09 verificó 33 437 imágenes y detectó cuatro conflictos de
etiqueta. La nueva preparación tiene 33 429 filas (23 399/5 015/5 015), con exclusión
provisional de ambas copias conflictivas y agrupación de un par visual confirmado.
La revisión cercana está incompleta: no reutilizar las métricas históricas como si
procedieran de estos nuevos splits ni evaluar checkpoints antiguos contra sus nuevas
particiones. Ver [evidencia de datos](../../reviews/2026-09-12-piloto-y-dataset).

Ambos modelos declaran 224×224, pretrained, batch 64, LR 0.0004548, weight decay 0.0001,
cosine con warmup de 3 épocas, label smoothing 0.1, pérdida sqrt_inverse, sin sampler ni
CLAHE. No aplicaron íntegramente el trial Optuna adjunto: este usaba weight decay
0.00001573119 y warmup 2. No se acredita HPO independiente para ShuffleNet.

## Resultados guardados y recalculados

| Métrica, test main de 5,015 filas | EfficientNet-B0 | ShuffleNet-v2 |
|---|---:|---:|
| Macro-F1 | 0.948333 | 0.932980 |
| Accuracy | 0.979661 | 0.973081 |
| Precision macro | 0.958922 | 0.938761 |
| Recall macro | 0.939991 | 0.929762 |
| Mejor época según historia | 28 | 23 |
| Épocas realizadas | 35 | 31 |

Se recalcularon métricas desde predicciones y matrices. Los pesos cargan estrictamente;
eso no prueba qué bytes se cargaron en la evaluación histórica: sus CSV no tenían hashes.
No se infiere independencia agronómica ni se atribuye causalmente el resultado a un
hiperparámetro por observar curvas.

El ensemble declara macro-F1 0.950676, pero faltan probabilidades completas/predicciones
para recalcularlo independientemente. El beneficio frente a B0 es pequeño en la cifra
publicada y debe contrastarse con coste real en dispositivo.

## Protocolo vigente

HPO solo con `--best-params`: CLI > JSON > defaults. Main mantiene pérdida ponderada sin
sampler y restaura la mejor época. Train/CV no evalúan test salvo `--evaluate-test`.
Los nuevos runs serializan preprocesamiento, clases, hashes y configuración efectiva.
Para inferir con una corrida antigua, usar una copia migrada y registrar las limitaciones.

Segmentación se estudia primero con un piloto pareado sobre desarrollo, preservando
IDs y particiones y contabilizando rechazos/fallbacks. No se garantiza mejora ni se
habilita una entrega Android por el hecho de producir un checkpoint o un TFLite.

El piloto de tres semillas ya se ejecutó y no mostró mejora global con el perfil actual.
La skill interna de entrenamiento conserva la distinción de políticas: principal sin
sampler ponderado; baselines con su política propia. La skill de datos exige IDs, carga
canónica y originales inmutables, también en las exclusiones y splits revisados.
