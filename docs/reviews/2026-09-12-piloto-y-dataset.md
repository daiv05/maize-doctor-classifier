# Dataset verificado y piloto original frente a segmentado

Fecha: 12 de septiembre de 2026. Resultado principal: **el perfil de segmentación actual
no mostró mejora global en este piloto**. No se recomienda activarlo obligatoriamente
para los checkpoints originales. Esta conclusión se limita al protocolo y subconjunto
descritos; no demuestra que toda segmentación sea perjudicial.

## Fuente, identidades y particiones

Se descargó completamente la revisión
`e515ab2f1e4c5729f8447520f1a630cf14c532dc` del
[dataset proporcionado por el usuario](https://huggingface.co/datasets/daiv05/corn-leaf-diseases-pests-and-deficiencies).
Está en `/tmp/doctor-maiz-hf-20260911-JjlLG8/clean`: **33 437 imágenes**, 33 433 hashes
distintos. Las 33 433 filas del manifest histórico de Etapa 2 coinciden con los bytes
descargados. Esto verifica identidad actual, no qué imagen cargó cada entrenamiento antiguo.
El primer intento fallido HTTP 503/timeouts no activó un dataset incompleto; se reanudó
con la misma revisión. La ruta temporal no debe usarse como almacenamiento permanente.

### Cuatro duplicados exactos con etiquetas contradictorias

Las cuatro copias adicionales no son un mero detalle de conteo: cada imagen aparece con
etiquetas **healthy y fall_armyworm**. La deduplicación histórica retuvo la primera por
orden alfabético (fall_armyworm); eso no resuelve cuál etiqueta es correcta.
Se conservan hashes y las ocho rutas en
[la evidencia de conflictos](evidence/2026-09-12-label-conflicts.json).

Tras consultar esta decisión y recibir la indicación de continuar, se aplicó una exclusión
provisional de **ambas copias** de cada conflicto exclusivamente a los nuevos splits.
No se borró ni cambió ningún original y no se relabeló ninguna imagen.
[CSV de exclusiones](evidence/2026-09-12-provisional-exclusions.csv): ruta, SHA-256 y razón.
`create_splits --exclusions` exige que cada ruta exista y coincida con ese hash; registra
cada exclusión y vincula el CSV al lock. Queda pendiente revisión agronómica de etiquetas.

### Duplicados cercanos: un caso confirmado, revisión incompleta

`audit_near_duplicates` examinó los 33 433 originales de los splits históricos: pHash
de 64 bits, radio Hamming 4, **373 pares candidatos**, 163 entre particiones y 2 entre
etiquetas. Un candidato no demuestra fuga. La comparación RGB canónica de las 544 imágenes
implicadas no encontró igualdad exacta de píxeles.

La inspección visual sí confirmó la misma fotografía GLS a 2048×1536 y 4000×3000:
coinciden agujero, lesiones, fragmento de hoja y fondo. Tras redimensionar a 224×224,
MAE 1,149/255 y correlación 0,999560 acompañan la evidencia visual; no constituyen por sí
solos un umbral general de duplicación. En main aparecen en train y val; en Etapa 2 ambas
están en train. **Este caso confirma fuga entre train/val de main, no del holdout de Etapa 2.**
No se cuantificó cuánto infló las métricas. Ver
[par y hashes](evidence/2026-09-12-visual-duplicate.json) y
[decisión de agrupación](evidence/2026-09-12-reviewed-near-pairs.csv).

La revisión visual fue del asistente; no se presenta como revisión de un agrónomo.
Quedan **372 pares candidatos sin resolver**, además de la ausencia de IDs de planta/sesión.
El reporte inicial `pixel-duplicate-verification` incluía una frase incondicional errónea
sobre fuga aunque el contador era cero. Se corrigió el generador y se conservó el intento;
el resultado válido es `pixel-duplicate-verification-v2`, no el texto inicial.

### Nueva versión de splits

Directorio: `outputs/repair-20260912/splits-reviewed-quarantine-v3`.

| Versión | Imágenes | Train | Val | Test |
|---|---:|---:|---:|---:|
| HF completo | 33 437 | — | — | — |
| Main histórico, primera copia por hash | 33 433 | 23 403 | 5 015 | 5 015 |
| Nueva, ocho exclusiones y un par agrupado | 33 429 | 23 399 | 5 015 | 5 015 |

Son 33 428 grupos; las dos versiones GLS quedaron juntas en val. Se verificaron hashes
del lock, IDs y grupos disjuntos, ausencia de las ocho exclusiones y conservación de sus
bytes originales. Healthy pasa a 8 740 y fall_armyworm a 4 853; las demás clases no cambian.
Los intentos `splits-reviewed-groups-v1` (conflictos) y `splits-reviewed-quarantine-v2`
(incompatibilidad del lector de grupos) se conservan como fallidos, no como splits utilizables.
La incompatibilidad fue corregida y cubierta por regresión.

Estos splits corrigen los problemas **confirmados**, pero no equivalen a un corpus libre
de fuga. Tampoco sirven para evaluar sin sesgo los checkpoints viejos: parte de sus nuevas
particiones ya fue usada para entrenarlos. Para experimentos nuevos se debe volver a
entrenar después de completar la revisión, sin comparar números de holdouts distintos.
Etapa 2 puede heredar esta versión mediante `prepare --source-splits`; preserva IDs,
grupos, exclusiones y particiones verificadas, sin repetir el sorteo.

## Piloto ejecutado, no simulación

Artefactos: `outputs/repair-20260912/pilot-original-segmented-v1`.
[Resumen compacto y hashes](evidence/2026-09-12-pilot-summary.json).

Se congeló el protocolo antes de inferir: 144 imágenes de train y 144 de val del main
histórico, 12 por estrato clase/entorno (9 de campo y 3 de laboratorio). No se cargaron
imágenes de test para inferencia, ni se entrenó con ellas. Ninguno de los 373 pares
candidatos tenía ambos extremos dentro de estas 288 imágenes. Esto no demuestra
independencia completa respecto al entrenamiento histórico de los checkpoints.

Las copias migradas de B0 y ShuffleNet se cargaron estrictamente, con hashes actuales y
preproceso reconstruido explícito: EXIF/RGB, 224×224 bilineal, normalización ImageNet,
CLAHE desactivado. No se fingió que el contrato reconstruido estaba registrado al entrenar.
Segmentador: checkpoint SHA-256
`4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f`;
perfil `crop_mask_letterbox`, con fallback original en incertidumbre o rechazo.

### Cobertura de segmentación

| Estado | Train | Val | Total |
|---|---:|---:|---:|
| Aceptada | 14 | 17 | 31 |
| Incierta | 106 | 104 | 210 |
| Rechazada | 24 | 23 | 47 |
| Fallback original | 130 | 127 | 257 |

Solo 10,76% se segmentó operativamente; 89,24% conservó el original. Las alertas no son
mutuamente excluyentes: borde del encuadre 204, múltiples componentes 26, casi completa
18, máscara completa rechazada 3 y candidatos ambiguos 1. Los resultados siguientes usan
**toda la población seleccionada con su fallback**, no solo los casos aceptados.

### A. Pesos recibidos: sensibilidad al cambio de entrada

| Modelo original, val n=144 | Macro-F1 original | Segmentado + fallback | Delta observado | IC95% bootstrap del delta |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 0,946754 | 0,879948 | −0,066806 | [−0,111329; −0,027825] |
| ShuffleNetV2-x1.0 | 0,960384 | 0,884677 | −0,075707 | [−0,126226; −0,037720] |

Son 500 remuestreos pareados estratificados por clase; sin IDs de planta no modelan esa
dependencia. El JSON v1 llama `delta_macro_f1_segmented_minus_original` a la **media del
bootstrap**, ligeramente diferente del delta observado de esta tabla. Se preservó el
artefacto y se corrigió el generador para distinguir ambos valores en ejecuciones futuras.

En campo (108 imágenes, 9 clases), B0 baja de 0,962801 a 0,885624 y ShuffleNet de
0,962476 a 0,878781. Estos modelos se entrenaron con originales: la prueba mide sensibilidad
a un cambio de distribución, **no** rendimiento de un modelo entrenado con segmentación.

### B. Heads nuevos sobre ImageNet congelado, tres semillas

Presupuesto común: 144 train/144 val, seeds 42/43/44, 20 épocas, AdamW lr 0,001,
weight decay 0,0001, pérdida ponderada sqrt, sin sampler, estandarización calculada en
train y restauración de mejor época según val. No es fine-tuning end-to-end.

| Backbone / entrada | F1 seed 42 | Seed 43 | Seed 44 | Media ± desviación muestral |
|---|---:|---:|---:|---:|
| B0 original | 0,618074 | 0,656944 | 0,618708 | 0,631242 ± 0,022261 |
| B0 segmentado | 0,608036 | 0,597899 | 0,620985 | 0,608973 ± 0,011571 |
| ShuffleNet original | 0,653225 | 0,652285 | 0,639057 | 0,648189 ± 0,007923 |
| ShuffleNet segmentado | 0,645450 | 0,645108 | 0,644440 | 0,644999 ± 0,000513 |

No hay mejora media global. En potasio (solo 12 casos val), el F1 de ShuffleNet sube de
0,3704/0,3478/0,4000 a 0,5385/0,5600/0,5385; B0 presenta cambios mixtos. Es una señal
para investigar con más soporte, no una mejora agronómica confirmada ni razón para cambiar
el modelo global. Matrices, probabilidades por ID, métricas por clase/entorno, pesos de
heads y estandarización están en los artefactos. Los recibos y hashes fueron verificados.

### Contrato del repositorio del segmentador

Se inspeccionaron en lectura `config/segmentation.yaml` y `src/segmentation/quality.py`
del proyecto vecino `corn-leaf-desease-project`. **No se reutilizó automáticamente su
compuerta**: los contratos no son equivalentes.

| Parámetro | Proyecto segmentador | Piloto clasificador |
|---|---|---|
| Estados | reliable / uncertain / failed | accepted / uncertain / rejected |
| Confianza de propuesta / selección | 0,20 / 0,50 | 0,25 / 0,50 |
| Perfil | mask_black | crop_mask_letterbox |
| Margen entre candidatos | 0,33 | 0,05 |
| Geometría adicional | área máxima 0,999; área grande 0,25; bbox 0,80; perímetro 8 | componentes, retención de recorte y contacto con borde |

El YAML externo menciona una calibración con 42 imágenes; no se revisó su evidencia
humana, por lo que no se atribuye esa calibración a este piloto. La alerta de borde domina:
una hoja legítimamente cortada por el encuadre puede ser marcada incierta. Hace falta
calibrarla con revisión de desarrollo, no rebajar umbrales solo para obtener mejores cifras.

`human_review.csv` y previews permiten iniciar la revisión. En v1, la selección de previews
no estratificaba también por split, por lo que algunas clases solo tienen ejemplos train;
no se presenta como revisión completa de val. El generador nuevo incluye el split en la
cuota. No se confirmó que la etiqueta original sea válida para cada hoja seleccionada.

## Decisión y siguiente paso

Mantener originales como referencia de desarrollo. Revisar las ocho etiquetas conflictivas,
los 372 pares pendientes y máscaras N/P/K/campo; registrar revisor, razón y decisión.
Versionar los grupos completos y repetir un piloto comparable si cambia la compuerta.
Después, entrenar desde cero sobre la versión aprobada de splits con el mismo presupuesto
y varias semillas. Variación de fondo permanece desactivada: todavía no hay protocolo
validado que justifique añadirla. No optimizar estas decisiones consultando test.

Estos hallazgos no invalidan los cálculos aritméticos de las corridas antiguas, pero sí
limitan su interpretación como generalización independiente. Ver
[clasificación de resultados reutilizables](2026-09-11-corridas-recibidas.md) y
[registro de los 13 grupos](2026-09-11-reparacion-integral.md).
