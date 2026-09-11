# Consolidación y siguientes pasos

Estado de la línea de trabajo sobre procedencia y fuga al 2026-09-11, y qué queda por hacer
para decidir con qué actualizar el pipeline principal.

## Lo que está establecido

**El corpus tiene una señal de procedencia que casi resuelve la tarea sin contenido
diagnóstico.** Un anillo del 10 % del borde, sin hoja ni lesión, clasifica el 78,3 % del test
de 5 015 imágenes sobre la partición aleatoria. Cinco de las catorce fuentes aportan una sola
clase, y tres de ellas suman el 80,7 % de `healthy`.

**La partición actual maximiza esa fuga en lugar de controlarla.** `HierarchicalStratifiedSplitter`
estratifica por `label + "_" + environment` sin agrupar, así que las catorce fuentes aparecen
en train, val y test con la misma proporción. Además hay al menos 52 grupos de casi-duplicados
repartidos entre particiones.

**Con partición honesta el rendimiento cae 0,284.** Dejando una fuente fuera, el macro-F1 pasa
de 0,8411 a 0,5573 y la exactitud de 0,9115 a 0,6884, sobre 33 268 imágenes evaluadas cada una
fuera de su fuente.

**El atajo sobrevive a la partición honesta.** Bajo validación por fuente, el marco todavía
recupera el 72,6 % del macro-F1, y para `lethal_necrosis` y `common_rust` lo recupera casi
entero (99,7 % y 98,5 %).

**Ninguna clase supera la compuerta completa.** Con tres semillas, la única candidata
—`healthy`— cruza el umbral del marco y pasa a «sostenida por procedencia». Cuatro clases
quedan indecidibles dentro del ruido.

**La segmentación empeora el problema.** Documentado aparte: el macro-F1 cae de 0,9468 a
0,7191, y la causa no es destrucción de señal sino que segmentar a negro generaliza a todas
las clases el atajo que antes sólo tenía `common_rust`.

## Lo que resultó no ser como se esperaba

Esta línea de trabajo acumuló cinco predicciones propias refutadas por la medición. Se listan
porque el patrón es informativo: **casi todas las intuiciones sobre qué componente causa qué
resultaron falsas, y sólo la medición directa las separó.**

| Predicción | Resultado |
|---|---|
| `fall_armyworm` sería la clase más dañada por segmentar | Fue de las menos: −0,047. La peor fue `gray_leaf_spot` con −0,741 |
| `gray_leaf_spot` aguantaría la segmentación | Se hundió sin que sus imágenes se dañaran: perdió el contexto del encuadre |
| El jitter de tono dañaría las deficiencias nutricionales | Es lo que más las ayuda; `potassium` gana +0,106 |
| La recompresión hundía `gray_leaf_spot` | Aislada la deja plana (+0,0003) y sube el conjunto +0,031 |
| Recorte y color se combinarían sumando sus ventajas | Se estorban: la combinación es peor que cada uno en su propio eje |

A eso se añaden dos errores de método detectados y corregidos sobre la marcha: una compuerta
mal especificada que mezclaba dos protocolos de medición, y un generador aleatorio sembrado
por índice de imagen que convertía la augmentation en una asignación fija.

## Lo que hay medido sobre intervenciones

Todo bajo validación por fuente. Las tres primeras miden cuánta información contiene el marco;
las cinco últimas, cuánto depende del marco un modelo entrenado con la imagen completa. Los
dos grupos **no son comparables entre sí**.

| Intervención | macro-F1 | Dependencia del marco |
|---|---:|---:|
| Base | 0,5573 | 69,5 % |
| Balanceo de grupos | 0,5594 | 71,6 % |
| BackMix sobre pre-enmascaradas | 0,5609 | 70,6 % |
| Base (diseño B) | 0,5863 | 44,8 % |
| Sólo color | **0,6349** | 39,4 % |
| Sólo códec | 0,6171 | 39,9 % |
| Sólo recorte | 0,5643 | **24,7 %** |
| Recorte y color | 0,5763 | 31,2 % |
| Los tres | 0,5470 | 29,8 % |

**Hay dos palancas y no se combinan:** el color compra rendimiento, el recorte compra
independencia del atajo, y juntarlas da peor resultado que cualquiera por separado.

## Por qué estas cifras no son las del sistema

Todos los experimentos anteriores usan un arnés propio y mínimo, hecho para aislar el efecto
de la partición. Frente al pipeline principal le faltan:

| | Arnés de experimentos | Pipeline principal |
|---|---|---|
| Corrección EXIF | **ausente** | `exif_transpose` en el loader |
| Augmentation | resize + volteo horizontal | volteo doble, rotación 15°, `ColorJitter(0.1, 0.1, 0, 0)` |
| Clases minoritarias | sin tratamiento | `RandomResizedCrop(0.7–1.0)`, rotación 30°, jitter 0.3/0.3/0.2/0.05, `GaussianBlur` |
| Warmup | ausente | 3 épocas lineales |
| Recorte de gradiente | ausente | 1.0 |
| Épocas y paciencia | 25 y 6 | 60 y 8 |
| Lote | 64 | 32 |
| Datos de entrenamiento | tope de 1 000 por clase | split completo |
| Decodificación | `draft` a escala DCT reducida | decodificación completa |

Las comparaciones internas se sostienen porque todos los brazos comparten estas desviaciones.
Lo que no se sostiene es leer cualquier cifra como el rendimiento del sistema, ni trasladar
una ganancia directamente al pipeline.

Hay un caso donde esto cambia la lectura de forma concreta: **el brazo de color se comparó
contra la ausencia total de color**, no contra el `ColorJitter(0.1, 0.1, 0, 0)` real. Y las
tres deficiencias, que son las que más ganaron, **ya reciben el jitter agresivo** del pipeline
de minoritarias. La ganancia medida no es la ganancia disponible.

## El arnés alineado

`scripts/experiments/pipeline_aligned_loso.py` corrige eso. Cada pliegue entrena con los
mismos componentes que `scripts/pipeline/train.py`: `CornDataset`, `CornTransformFactory`,
`build_model`, `build_criterion`, `build_scheduler` con warmup, `EarlyStopping`,
`worker_init_fn` y recorte de gradiente. Sin réplicas.

Lo único que cambia entre brazos es la transformación de entrenamiento, así que **cada brazo
es una modificación candidata de `src/data/transforms.py` medida directamente**.

| Brazo | Qué cambia respecto al pipeline actual |
|---|---|
| `baseline` | Nada. Reproduce el pipeline principal bajo partición honesta |
| `colour_strong` | Jitter 0.4/0.4/0.4/0.1 para las nueve clases |
| `crop_all` | `RandomResizedCrop(0.3–1.0)` para las nueve clases |
| `minority_for_all` | El pipeline agresivo de minoritarias aplicado a todas |

Cada corrida entrega rendimiento y dependencia del marco sobre el mismo conjunto retenido.

## Siguientes pasos

**1 — Establecer la referencia real.** Correr `baseline`. Da por primera vez el número del
pipeline principal bajo partición honesta, que es la cifra que debe ir a la tesis en lugar de
0,9468. Es también el punto de comparación de todo lo demás.

**2 — Medir los tres candidatos** `colour_strong`, `crop_all` y `minority_for_all` contra esa
referencia. Cada uno es un cambio concreto y acotado en `src/data/transforms.py`.

**3 — Replicar a tres semillas sólo lo que muestre efecto.** Ninguna cifra de los brazos de
augmentation tiene desviación medida. Con la lección de la Fase 1 —donde una clase cambió de
banda al pasar de una semilla a tres— ningún brazo debería promoverse con una sola.

**4 — Decidir qué entra en el pipeline.** El criterio no puede ser sólo el macro-F1: una
ganancia que venga acompañada de más dependencia del marco es rendimiento prestado. La
propuesta es exigir **que el macro-F1 no baje y que la dependencia del marco baje**, y elegir
entre los que cumplan por magnitud de la segunda.

**5 — Cambiar el splitter.** Independiente de la augmentation, y probablemente lo de mayor
efecto: agrupar por `source_id` y deduplicar casi-duplicados antes de particionar. Cambia todas
las métricas del proyecto, y las cambia a la baja, pero las hace reales.

### Coste estimado

Los brazos alineados son mucho más caros que los anteriores: sin tope de entrenamiento y con
hasta 60 épocas, cada pliegue procesa el split completo. Once pliegues por brazo, cuatro
brazos, y después tres semillas de los que sobrevivan. Conviene correr primero `baseline` sola
para medir el coste real por pliegue antes de comprometer el resto.
