# Evaluación

Cada baseline se evalúa sobre `test.csv` (N = 1 503, 9 clases) con transformaciones deterministas. La métrica primaria es **macro-F1** (pondera por igual todas las clases, incluidas las minoritarias); accuracy es secundaria por el desbalance. El conjunto de test es mayoritariamente de **dominio real de campo** (1 182 imágenes reales vs. 321 de laboratorio), lo que interesa medir es el desempeño sobre fotos tomadas en condiciones agrícolas.

## Resultados globales (test, 9 clases)

La tabla compara las tres métricas de test junto con el tamaño del checkpoint:

| Modelo | Accuracy | macro-F1 | Loss (test) | Checkpoint |
|---|---:|---:|---:|---:|
| `efficientnet_b0` | 0.9521 | **0.9146** | 0.229 | 15.6 MB |
| `shufflenet_v2_x1_0` | 0.9508 | 0.9030 | **0.169** | **5.0 MB** |
| `efficientnet_lite0` | 0.9474 | 0.8951 | 0.218 | 13.1 MB |

Los tres modelos **superan la meta de macro-F1 ≥ 0.85** con holgura. `efficientnet_b0` lidera el macro-F1 y es el umbral de referencia actual, pero la lectura no es tan simple como "gana b0":

- **La diferencia entre los tres es de apenas 2 puntos de macro-F1** (0.9146 vs. 0.9030 vs. 0.8951). Para el margen de error de un test de 1 503 imágenes, los tres son esencialmente equivalentes en calidad de clasificación.
- **`shufflenet_v2_x1_0` obtiene la menor loss de test** (0.169) y el **checkpoint más pequeño con diferencia** (5 MB, ~3x más liviano que b0).

## F1 por clase

La cifra agregada esconde que el rendimiento **no es parejo entre clases**. La figura desglosa el F1 por clase y por modelo, ordenado de mayor a menor, y deja ver de inmediato dónde se concentra el problema.

![F1 por clase para los tres baselines](/baselines/baseline_f1_por_clase.png)

El patrón es idéntico en los tres modelos: un bloque de clases "fáciles" que rozan o superan 0.95, un escalón intermedio en las lesiones foliares y las deficiencias de N y P, y una única clase que se desploma por debajo de la meta, **`potassium_deficiency`**.

| Clase | b0 | shufflenet | lite0 | Soporte (test) |
|---|---:|---:|---:|---:|
| common_rust | 0.99 | 0.99 | 0.98 | 225 |
| lethal_necrosis | 0.99 | 0.99 | 1.00 | 225 |
| healthy | 0.96 | 0.96 | 0.97 | 225 |
| fall_armyworm | 0.96 | 0.96 | 0.95 | 225 |
| gray_leaf_spot | 0.95 | 0.95 | 0.95 | 168 |
| northern_corn_leaf_blight | 0.95 | 0.94 | 0.94 | 224 |
| phosphorus_deficiency | 0.95 | 0.92 | 0.92 | 92 |
| nitrogen_deficiency | 0.87 | 0.88 | 0.85 | 79 |
| **potassium_deficiency** | **0.62** | **0.54** | **0.49** | **40** |

## El costo real de una sola clase

Vale la pena cuantificar cuánto pesa `potassium_deficiency` sobre la métrica global. Si se recalcula el macro-F1 **excluyendo esa única clase**, los tres modelos saltan a un rango estrecho y alto:

| Modelo | macro-F1 (9 clases) | macro-F1 (sin potasio) | Δ |
|---|---:|---:|---:|
| `efficientnet_b0` | 0.9146 | **0.9517** | +0.037 |
| `shufflenet_v2_x1_0` | 0.9030 | 0.9490 | +0.046 |
| `efficientnet_lite0` | 0.8951 | 0.9455 | +0.050 |

Es decir: **una sola clase minoritaria arrastra el macro-F1 global entre 3.7 y 5.0 puntos.** El resto del clasificador ya opera a un nivel de ~0.95 macro-F1. Esto reencuadra el problema: el baseline no necesita "más capacidad de modelo", necesita resolver un cuello de botella de datos muy localizado.

## Matriz de confusión

La matriz de confusión de `efficientnet_b0` (normalizada por fila, es decir mostrando el recall de cada clase real) hace visible que los errores **no son ruido aleatorio**.

![Matriz de confusión de efficientnet_b0](/baselines/baseline_confusion_b0.png)

**Clúster 1 - deficiencias nutricionales (N/P/K).** El bloque inferior-derecho concentra casi todo el error del modelo. De las 40 imágenes de potasio, solo 21 se clasifican bien; **13 se confunden con nitrógeno y 5 con fósforo**. La observación clave es que este error es *dirigido*: si se toman todas las predicciones de las tres deficiencias juntas, el **97 % permanece dentro del propio clúster N/P/K** (solo 6 de 211 se escapan a una clase no nutricional). El modelo *sabe* que está viendo una deficiencia; lo que no logra es discriminar **cuál** de las tres, porque comparten el mismo síntoma base de clorosis y potasio es la clase con menos ejemplos de todo el dataset (266 imágenes totales en el corpus con el que se entrenó este baseline; 621 tras la ampliación de agosto 2026, que aún no se ha reentrenado).

**Clúster 2 - lesiones foliares (GLS ↔ NCLB).** `gray_leaf_spot` y `northern_corn_leaf_blight` se confunden mutuamente (4–5 imágenes en cada sentido para b0) porque sus lesiones alargadas grisáceas/marrones son visualmente muy parecidas. Es un error de menor magnitud pero también consistente entre los tres modelos.

Fuera de esos dos clústeres, la matriz es prácticamente diagonal: `common_rust`, `lethal_necrosis` y `healthy` se clasifican con recall ≥ 0.99.

## Hallazgos y recomendaciones

1. **`potassium_deficiency` es el único cuello de botella que impide subir la métrica.** Con 40 imágenes en test y recall de 0.38–0.53, es la clase que más castiga el macro-F1. Aislarla revela que el clasificador base ya vale ~0.95.

2. **El modelo ya "detecta" deficiencias nutricionales con altísima fiabilidad.** El modelo detecta la categoría "deficiencia nutricional" con altísima fiabilidad (97 % de contención en el clúster) pero no separa N/P/K entre sí. Para este baseline tenemos un modelo que puede dar una respuesta del tipo "deficiencia nutricional" como confiable, aunque no pueda decir cuál de las tres.

3. **Lo que se necesitan son datos.** Mas que aumentar la capacidad del modelo, lo rentable es aumentar/rebalancear las deficiencias (especialmente potasio): más imágenes reales, augmentation dirigida, o incluso pérdida ponderada por clase (que el baseline de momento no usa).

4. **La elección de arquitectura debe pesar el despliegue.** Como los tres son equivalentes en calidad (~2 pts), la decisión final debería priorizar tamaño y latencia, que por ahora favorecen a `shufflenet_v2_x1_0` (5 MB, 0.169 loss), pero se debe tener en cuenta pruebas ya en hardware objetivo y con el modelo exportado a TFLite.
