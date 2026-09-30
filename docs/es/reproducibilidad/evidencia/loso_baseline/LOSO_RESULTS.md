# Resultado LOSO baseline EfficientNet-Lite0

Fecha: 30 de septiembre de 2026. Se completaron **dos** entrenamientos seed 42,
sin HPO, otras semillas ni otros folds. El [protocolo y sus hashes](LOSO_REPORT.md)
se fijaron antes de entrenar. Para cada run se eligió `best.pth` únicamente por
Macro-F1 de validation interna; después se escribió `selection.lock.json` y
solo entonces se evaluó el holdout. La secuencia temporal, los hashes y la
cobertura de `sample_id` están verificados en [loso_summary.json](loso_summary.json).

## Resultado principal

Macro-F1 LOSO promedia **solo las tres clases presentes en el ground truth**
de cada holdout. Las predicciones hacia otras clases globales siguen contando
como errores. Las cifras históricas in-distribution proceden del baseline
`20260921_204608` desagregado por fuente en
[SOURCE_ANALYSIS.md](../source_analysis/SOURCE_ANALYSIS.md); corresponden a
subconjuntos de validation/test de `seed_42`, no al mismo holdout completo.
Por eso la diferencia es descriptiva, no un efecto causal ni una prueba
estadística.

| Fuente excluida | Holdout n | Clases | Val interna F1 | LOSO F1 | LOSO accuracy | F1 histórico val | Δ LOSO−histórico |
|---|---:|---:|---:|---:|---:|---:|---:|
| maize-diseases | 6 989 | 3 | 0.948975 | 0.993124 | 0.993704 | 0.997893 | −0.004769 |
| multicrop-disease-maiz | 5 816 | 3 | 0.952488 | 0.905490 | 0.806912 | 0.997217 | −0.091728 |

El test histórico por fuente tuvo F1 0.998449 y 0.995831, respectivamente;
se muestra en [loso_results.csv](loso_results.csv) como referencia adicional,
sin intervenir en la selección. Los entrenamientos terminaron por early
stopping en 23 y 50 épocas. El mejor checkpoint fue el de las épocas 15 y 42.
En esas épocas, train F1 fue 0.975914 y 0.998330; la brecha train−validation
fue 0.026939 y 0.045842. La suma de tiempos por época del historial fue
1 305.7 s y 2 397.8 s. Ambas runs usaron A10 en Modal, AdamW, scheduler
coseno y el resto de la configuración contractual baseline. No se registró
peak GPU memory ni un snapshot completo de versiones instaladas en el
contenedor. La definición de imagen y dependencias está en
`scripts/modal/_common.py` y `pyproject.toml`; ese límite se conserva.

| Fuente | Errores | Error rate | ECE 15 bins | Confianza media aciertos | Confianza media errores |
|---|---:|---:|---:|---:|---:|
| maize-diseases | 44 | 0.006296 | 0.121044 | 0.874708 | 0.572523 |
| multicrop-disease-maiz | 1 123 | 0.193088 | 0.048824 | 0.808147 | 0.605979 |

ECE resume calibración de la confianza máxima; que sea menor en el segundo
holdout **no** compensa su mayor tasa de error. Ninguno de estos dos holdouts
contiene N/P/K, por lo que no hay métricas N/P/K aplicables.

## Clases compartidas y errores

`train_without_source` cuenta solo muestras del train LOSO, no toda la
disponibilidad del corpus. `other_sources` en
[loso_class_metrics.csv](loso_class_metrics.csv) enumera las fuentes
alternativas presentes en el master. F1 histórico usa validation del baseline.

| Holdout | Clase | Train sin fuente | Otras fuentes | F1 histórico | F1 LOSO | Δ |
|---|---|---:|---:|---:|---:|---:|
| maize-diseases | common_rust | 737 | 2 | 0.997067 | 1.000000 | +0.002933 |
| maize-diseases | healthy | 2 835 | 4 | 1.000000 | 0.995765 | −0.004235 |
| maize-diseases | northern_corn_leaf_blight | 4 025 | 4 | 0.996610 | 0.983607 | −0.013004 |
| multicrop-disease-maiz | common_rust | 916 | 2 | 1.000000 | 0.972561 | −0.027439 |
| multicrop-disease-maiz | fall_armyworm | 2 276 | 2 | 0.996000 | 0.904562 | −0.091438 |
| multicrop-disease-maiz | lethal_necrosis | 2 202 | 1 | 0.995652 | 0.839347 | −0.156306 |

En `maize-diseases`, `common_rust` acertó las 1 192 imágenes. La confusión
más frecuente fue `healthy → northern_corn_leaf_blight` (29 casos); hubo
44 errores en total. En `multicrop-disease-maiz`, `lethal_necrosis` aportó
la mayor parte de los errores: 493 casos se predijeron como `gray_leaf_spot`
y 198 como `healthy`. `fall_armyworm` se confundió, entre otras, con
`nitrogen_deficiency` (69) y `gray_leaf_spot` (48). Las matrices completas,
con las nueve clases del modelo y normalización por clase real, son:

- [maize-diseases, conteos](loso_confusion_maize_diseases.csv) y [normalizada](loso_confusion_maize_diseases_normalized.csv);
- [multicrop-disease-maiz, conteos](loso_confusion_multicrop_disease_maiz.csv) y [normalizada](loso_confusion_multicrop_disease_maiz_normalized.csv).

## Lectura y límites

El rendimiento alto de `maize-diseases` es evidencia de buena transferencia
**hacia esa fuente concreta** bajo este protocolo; no prueba generalización
universal. La degradación mucho mayor de `multicrop-disease-maiz` indica que
esa fuente resulta más difícil al excluirla completamente. El patrón es
compatible con diferencias de dominio y con dependencia parcial de señales
correlacionadas con procedencia, pero **no demuestra** que el modelo use la
fuente como etiqueta ni identifica una causa visual específica.

Se retiraron 4 881 muestras de train (−20,86 %) en el primer caso y 4 072
(−17,40 %) en el segundo. También cambian la distribución de clases, los
pesos numéricos `sqrt_inverse` y qué clases cruzan el umbral baseline de
augmentation minoritaria; la regla y los hiperparámetros no se ajustaron al
holdout. Esta reducción y la ausencia de muchas combinaciones fuente×clase
impiden atribuir toda la diferencia a domain shift. Son **una sola semilla**
por fuente, sin intervalos ni prueba de significancia; los tamaños y soportes
históricos no son idénticos a los LOSO.

La siguiente fase recomendada es una repetición con semillas predefinidas de
`multicrop-disease-maiz` para medir estabilidad de la caída y, si persiste,
un análisis dirigido de las confusiones de `lethal_necrosis` con imágenes y
procedencia. No se ejecutó esa fase ni se probó HPO bajo LOSO.

## Artefactos y trazabilidad

[loso_results.csv](loso_results.csv) y
[loso_class_metrics.csv](loso_class_metrics.csv) contienen los valores sin
redondear; [loso_summary.json](loso_summary.json) incluye los dos selection
locks, hashes y timestamps. Los runs completos (checkpoints, predicciones,
historial, métricas, matrices y manifests) están en
`outputs/loso/efficientnet_lite0_baseline/<source>/seed_42/` y su copia
remota `corn-outputs:/loso/efficientnet_lite0_baseline/`.

| Fuente | Run | SHA-256 `best.pth` | SHA-256 selection lock |
|---|---|---|---|
| maize-diseases | `20260930_144843` | `b16b1ef8bb4285f65be020272845b6c927057103bf177a290311f186de101dcc` | `1a8c202ed20cacb14d8f1c5dcc25996c1832dd20665e3caf17929b9c23860a80` |
| multicrop-disease-maiz | `20260930_145541` | `bd778a28d251fd1b45960f3caec8f5b51d0ed0178c4e4f46e06d2b40948e4517` | `a137e5f21c5adef9a53872f1da5d94b49a3011a3cc4a84048a20794c2437b162` |
