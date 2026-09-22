# Preprocesamiento en el Pipeline Principal

El pipeline principal toma la base de preparación de datos del proyecto y la adapta para entrenar los modelos definitivos sobre el 100% de las imágenes disponibles: más de 33,400 fotografías de hojas repartidas en nueve clases fitosanitarias y nutricionales.

En las fases preliminares era razonable recortar o topar el número de imágenes por categoría para iterar rápido y abaratar las pruebas de concepto. Para la etapa final, en cambio, la consigna fue aprovechar cada ejemplo disponible, cuidando con especial atención las patologías que cuentan con menos fotografías.

---

## Estandarización y flujo de entrada

Cada imagen que ingresa al flujo pasa por una secuencia de transformación determinista y reproducible:

- **Corrección de orientación EXIF:** Se normaliza la rotación para que ninguna hoja llegue girada artificialmente al modelo.
- **Formato RGB estricto:** Se descarta cualquier canal alfa residual o variación en escala de grises.
- **Escalado directo a 224 × 224:** La imagen se redimensiona a la resolución cuadrada estándar que esperan las redes convolucionales livianas.
- **Normalización ImageNet:** Se ajustan los valores de píxel al rango $[0, 1]$ y se restan las medias y desviaciones de referencia ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

La partición de los datos (`seed_42`) se realiza de forma estratificada considerando conjuntamente la clase agronómica y el entorno (`label + environment`). De este modo nos aseguramos de que la proporción entre fotos de campo real y fotos de laboratorio se mantenga equilibrada tanto en el conjunto de entrenamiento (70%) como en los conjuntos de validación (15%) y prueba retenida (15%).

La materialización vigente, regenerada después de corregir los conflictos de integridad del corpus, contiene:

| Partición | Muestras | Proporción |
|---|---:|---:|
| Entrenamiento | 23,400 | 69.9991 % |
| Validación | 5,014 | 14.9990 % |
| Prueba | 5,015 | 15.0019 % |
| **Total** | **33,429** | **100 %** |

Los tres CSV conservan las nueve clases. La auditoría de esta materialización verificó cero solapamientos por `sample_id`, SHA-256 y `effective_group_id` entre particiones. Ocho archivos involucrados en cuatro pares de contenido idéntico con etiquetas contradictorias se excluyen explícitamente mediante `config/dataset_exclusions.csv`; no se elimina ni se reasigna ninguna muestra de forma silenciosa durante el entrenamiento.

`source_id` se conserva como metadato de procedencia y puede aparecer en las tres particiones de `seed_42`: este split mide rendimiento dentro de las fuentes conocidas, no generalización a una fuente nueva. Para ese segundo escenario existe el protocolo opt-in `seed_42_source_grouped`, activado con `--group-by-source`, que mantiene cada `effective_group_id` en una sola partición. Como algunas clases solo existen en pocas fuentes, ese benchmark puede requerir `--allow-incomplete-splits` y no sustituye al split principal.

La generación actual calcula SHA-256 y evita duplicados exactos, pero fue ejecutada con `deduplicate_perceptual=false`. Por tanto, los controles anteriores no demuestran ausencia de imágenes casi duplicadas; esa comprobación debe realizarse aparte antes de interpretar el resultado como robustez frente a variaciones visuales cercanas.

---

## Estrategia frente al desbalance de clases

En el corpus final conviven clases masivas como *Healthy* (más de 8,700 imágenes) con clases escasas como *Potassium Deficiency* (alrededor de 620 imágenes), lo que representa una relación de desbalance de hasta 14 a 1.

Para evitar que el modelo aprenda a ignorar a las clases minoritarias, diseñamos una estrategia equilibrada:

1. **Aumento de datos dirigido (*Data Augmentation*):** Las cuatro clases con mayor escasez relativa (`gray_leaf_spot`, `nitrogen_deficiency`, `phosphorus_deficiency` y `potassium_deficiency`) reciben un tratamiento de augmentación más enriquecido en tiempo de entrenamiento. Se aplican recortes aleatorios con escala variable, rotaciones de hasta 30 grados, variaciones sutiles de color y desenfoques gaussianos leves. De esta forma, cada vez que la red ve una imagen minoritaria, la observa desde un ángulo o iluminación distinta.
2. **Pérdida ponderada suave (`sqrt_inverse`):** En lugar de forzar un remuestreo artificial de lotes que multiplicaría imágenes repetidas en memoria, asignamos pesos inversos a la raíz cuadrada de la frecuencia de cada clase dentro de la función de pérdida. Esto penaliza con mayor firmeza los fallos en clases escasas sin sobrecompensar de manera desmedida a las clases grandes.
3. **Suavizado de etiquetas (*Label Smoothing = 0.10*):** Evita que el clasificador se vuelva dogmático y sobreconfiado en sus predicciones, mejorando la calibración de las probabilidades y facilitando la detección de casos ambiguos en campo.
