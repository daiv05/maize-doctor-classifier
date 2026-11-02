# Generalización entre fuentes

Cada repositorio del corpus fue fotografiado con sus propias cámaras, fondos e iluminación, y varios contienen una sola clase. Si el modelo identifica la fuente en lugar de la lesión, una partición aleatoria sobreestima su desempeño. Esta página reúne las evaluaciones que retienen fuentes completas y las pruebas de atajo.

| Protocolo | Unidad que no cruza particiones | Qué mide |
|---|---|---|
| `seed_42` estratificado | muestra | Desempeño dentro de fuentes conocidas |
| `seed_42_source_grouped` | `source_id` completo | Una asignación única de fuentes a train, validación y prueba |
| LOSO | una fuente | Transferencia a una fuente nunca vista |
| CV agrupada por fuente | `source_id` por pliegue | Sensibilidad a qué fuentes quedan retenidas |

Las cifras de protocolos distintos no se restan entre sí. Especificación en [Protocolos experimentales](/es/metodologia/protocolos-experimentales).

## Asociación entre fuente y clase

El corpus tiene 11 fuentes y 9 clases. De las 99 celdas fuente × clase, 65 están vacías y el V de Cramér es 0.5030 (χ² = 67 673.21), una asociación moderada a fuerte entre origen y patología. Dos fuentes contienen solo `gray_leaf_spot`; `maize-diseases` es 67.8 % `healthy` y aporta el 54.2 % de esa clase; `multicrop-disease-maiz` aporta el 50.4 % de `lethal_necrosis`. Reporte y tablas en [`source_analysis.md`](../reproducibilidad/evidencia/source_analysis/source_analysis.md).

![Distribución de muestras por fuente](../reproducibilidad/evidencia/source_analysis/source_sizes.png)

## LOSO con el modelo oficial

Se excluye una fuente completa de la preparación, el entrenamiento y la selección de hiperparámetros; el checkpoint se bloquea y se evalúa sobre la fuente excluida. Tres semillas (42, 123, 2026), Lite0 con la configuración del modelo oficial.

| Fuente excluida | Imágenes de train | Imágenes del holdout | Macro-F1 del holdout (media ± SD) |
|---|---:|---:|---:|
| `maize-diseases` | 18 519 | 6 989 | 0.993264 ± 0.000600 |
| `multicrop-disease-maiz` | 19 328 | 5 816 | 0.901906 ± 0.004422 |

Por semilla en `multicrop-disease-maiz`: 0.905490, 0.896964 y 0.903264. El Macro-F1 del holdout promedia solo las tres clases presentes en la fuente (`common_rust`, `fall_armyworm`, `lethal_necrosis`); la exactitud en `multicrop-disease-maiz` es 0.8069, 0.8006 y 0.8038 por semilla (en `maize-diseases`, 0.9937, 0.9930 y 0.9944). Por clase en `multicrop-disease-maiz` (F1 medio): `common_rust` 0.9708, `fall_armyworm` 0.8895 y `lethal_necrosis` 0.8454, la más débil; esa fuente aporta el 50.4 % de los casos de la clase y al excluirla quedan 2 202 en entrenamiento. Con `maize-diseases` excluida el Macro-F1 es 0.9933, frente a 0.9984 de esa fuente en el test estratificado. La brecha entre fuentes no es uniforme.

Evidencia: [`loso_results.csv`](../reproducibilidad/evidencia/loso_baseline/loso_results.csv) y [`multiseed_summary.json`](../reproducibilidad/evidencia/multiseed_source_stability/multiseed_summary.json).

## Ensamble bajo LOSO

Se reutilizaron los tres Lite0 del LOSO anterior y se entrenaron B0 y ShuffleNet-V2 de cada semilla excluyendo `multicrop-disease-maiz` (5 816 imágenes de holdout, tres clases presentes); se promediaron los softmax de los tres con pesos iguales.

| Semilla | Lite0 | B0 | ShuffleNet-V2 | Ensamble | Δ ensamble − B0 |
|:---:|---:|---:|---:|---:|---:|
| 42 | 0.905490 | 0.917318 | 0.913037 | 0.930055 | +0.012737 |
| 123 | 0.896964 | 0.934785 | 0.886505 | 0.931569 | −0.003216 |
| 2026 | 0.903264 | 0.919831 | 0.888009 | 0.923713 | +0.003883 |
| Media ± SD | 0.901906 ± 0.004422 | 0.923978 ± 0.009443 | 0.895850 ± 0.014903 | 0.928446 ± 0.004168 | +0.004468 ± 0.007993 |

La ventaja del ensamble es positiva en dos semillas y negativa en una. El ECE pasa de 0.0446 en B0 a 0.0891 en el ensamble. La confianza del ensamble es menor, no mayor: 0.793 en aciertos y 0.559 en errores (B0: 0.852 y 0.698), y su confianza media (≈0.76) queda por debajo de su exactitud (0.846), de modo que el ECE mayor corresponde a subconfianza. La app despliega un solo modelo. Evidencia: [`ensemble_loso_summary.json`](../reproducibilidad/evidencia/ensemble_loso/ensemble_loso_summary.json).

![Deltas pareados del ensamble frente a B0](../reproducibilidad/evidencia/ensemble_loso/ensemble_paired_deltas.png)

## Benchmark source-grouped

Corrida `20260921_180112` (Lite0, `seed_42_source_grouped`): 16 554 train, 7 809 validación y 9 066 prueba, con las once fuentes como grupos indivisibles (proporciones 49.52 / 23.36 / 27.12 %).

| Campo | Valor |
|---|---:|
| Épocas ejecutadas / solicitadas | 12 / 60 (mejor época 4) |
| Macro-F1 de validación | 0.266927 |
| Macro-F1 de train al cierre | 0.966368 |
| Macro-F1 de prueba | 0.400665 |
| Accuracy de prueba | 0.585705 |

La validación no contiene `fall_armyworm` ni `lethal_necrosis` y la prueba la forman tres fuentes completas, de modo que las proporciones se apartan de 70 / 15 / 15. No es comparable con el run estratificado: cambia la pregunta. Evidencia: [resumen](./evidencia/run_20260921_180112_summary.json) e [historial](./evidencia/run_20260921_180112_train_history.csv).

## Campaña de procedencia (etapa 2)

Experimentos históricos con EfficientNet-Lite0 preentrenado y reentrenado en cada ejecución (tope de 1 000 imágenes de entrenamiento por clase), sobre 14 identificadores de procedencia que se agrupan en 11 fuentes (versiones del mismo dataset). Los archivos de las ejecuciones LOSO están en [evidencia de procedencia](/es/provenance/evidencia/).

| Experimento | Resultado |
|---|---|
| Anillo perimetral: se anula el cuadrado central (80 % del lado, 64 % del área) y se entrena solo con un marco del 10 % del lado; media de 3 semillas | 78.3 % de accuracy en prueba (imagen completa: 91.2 %; clase mayoritaria: 26.1 %); [datos por semilla](../provenance/evidencia/provenance_leak.json) |
| LOSO de 11 pliegues sobre 33 268 imágenes | Macro-F1 0.5573 y accuracy 68.84 % |
| LOSO con balanceo de grupos de procedencia | Macro-F1 0.5594 y accuracy 70.09 % |
| LOSO con sustitución de fondos (BackMix) | Macro-F1 0.5609 y accuracy 69.81 % |

Las dos intervenciones quedan dentro de la dispersión entre semillas (σ ≈ 0.007) y no cierran la brecha. Por clase, las abundantes y presentes en varias fuentes retienen más del 80 % de su F1 (hoja sana, necrosis letal, roya, tizón), mientras que las deficiencias y la mancha gris caen más (potasio de 0.58 a 0.20, fósforo de 0.74 a 0.29).

### Validación cruzada 2×2

Cinco pliegues, Lite0, estratificada frente a agrupada por fuente. ± es la semiamplitud del IC95.

| Configuración | Estratificada | Agrupada por fuente |
|---|---:|---:|
| Configuración base | 0.9311 ± 0.0099 (SD 0.0080) | 0.6026 ± 0.1240 |
| Lite0 con hiperparámetros del trial 12 de B0 | 0.9463 ± 0.0052 | 0.6499 ± 0.1508 |

El Macro-F1 agrupado se calcula sobre las clases presentes en cada pliegue (5 a 8); contando las ausentes como cero, el de la configuración base es 0.4359. Con las fuentes repartidas entre pliegues, cada pliegue estratificado es esencialmente el mismo experimento, por lo que su intervalo estrecho no mide la sensibilidad a fuentes nuevas. Todos los pliegues están en [`kfold_2x2_comparacion.csv`](./evidencia/kfold_2x2_comparacion.csv).

## Ablación de la periferia y equidad (etapa 2)

Prueba de oclusión sobre las 5 015 imágenes de prueba con el Lite0 `20260812_221429`. La línea de control nulo es responder siempre `healthy` (26.14 %).

![Ablación con control nulo](/resultados/ablacion_control_nulo.png)

| Condición | Área visible | Accuracy | Confianza retenida |
|---|---:|---:|---:|
| Imagen completa | 100.0 % | 97.91 % | 100.0 % |
| Centro tapado (solo anillo periférico) | 63.7 % | 79.46 % | 69.16 % |
| Periferia tapada (solo caja central) | 36.3 % | 42.25 % | 42.66 % |

Con el centro tapado el modelo conserva el 81 % de sus aciertos (79.46 de 97.91) y con solo el centro visible el 43 % (42.25). La máscara es un rectángulo fijo (20 %–80 % de cada eje) que no separa hoja de fondo, y el modelo se entrenó con imágenes completas, así que las cifras miden sensibilidad a regiones y no que el fondo determine la clase. El indicador `shortcut_detected = true` de la oclusión periférica es una regla de umbral (caída de accuracy > 0.20 o de confianza > 0.25). Las dos condiciones tienen áreas distintas, por lo que tampoco permiten decir que el fondo pese más que la hoja.

| Entorno | Muestras | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Campo real | 4 483 | 98.04 % | 0.9215 (9 clases) |
| Laboratorio | 532 | 96.80 % | 0.9423 (3 clases) |

Los Macro-F1 no son comparables entre entornos porque el laboratorio solo contiene tres clases. Restringido a esas tres clases, el *Disparate Impact Ratio* (desempeño del peor entorno entre el del mejor; aceptable desde 0.80) es 0.9604 sobre Macro-F1 y 0.9901 sobre accuracy para este modelo, y 0.9636 y 0.9956 para el modelo oficial ([cálculo](./evidencia/equidad_dir_clases_compartidas.json)). Un DIR cercano a uno no descarta disparidades por clase: en las tres clases comunes, la roya falla en el 31.25 % de los casos de campo (5 de 16) frente a 0.31 % en laboratorio: el 95 % de las imágenes de roya de prueba proviene de laboratorio. Datos en [`equidad_metricas.json`](./evidencia/equidad_metricas.json).

En el perfil global de atribución SHAP (270 imágenes, 30 por clase, máscara del segmentador), el exceso de atribución sobre la hoja (ratio sobre la hoja menos cobertura de la máscara) es +0.073 de media por imagen sobre las 215 con máscara aceptada (70.7 % positivas). Por clase, en los aciertos de las seis clases con máscara fiable la media de las medias es +0.037, con tres clases por encima de la cobertura y tres por debajo; las otras tres tienen la máscara degradada y no se interpretan. Datos en [`xai_global_summary.csv`](./evidencia/xai_global_summary.csv) y [`xai_global_per_image.csv`](./evidencia/xai_global_per_image.csv).

## Implicaciones

- La app recorta al marco guía antes de clasificar y rechaza con el detector OOD las imágenes lejanas al espacio de entrenamiento (ver [Despliegue](/es/deployment/prototipo)).
- Ninguna fuente se capturó en El Salvador; cerrar la brecha requiere fotos de campo locales etiquetadas, que el módulo de contribución de la app permite reunir.
