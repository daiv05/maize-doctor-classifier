# Preprocesamiento en el Pipeline Principal

El pipeline principal toma la preparación común y entrena sobre las 33 429 muestras elegibles de la materialización vigente, repartidas en nueve clases. Un checkpoint generado no es “definitivo” hasta completar el protocolo de evaluación y promoción.

Los perfiles baseline pueden topar imágenes por clase para iterar con menor costo; el pipeline principal usa el corpus elegible completo.

---

## Estandarización y flujo de entrada

Cada imagen que ingresa al flujo pasa por una secuencia de transformación determinista y reproducible:

- **Corrección de orientación EXIF:** Se normaliza la rotación para que ninguna hoja llegue girada artificialmente al modelo.
- **Formato RGB estricto:** Se descarta cualquier canal alfa residual o variación en escala de grises.
- **Escalado directo a 224 × 224:** La imagen se redimensiona a un cuadrado sin conservar la relación de aspecto (`Resize` con estiramiento). 224 × 224 es el tamaño por defecto de `config/dataset.yaml`; `src/models/input_sizes.py` lo sustituye para `efficientnet_b4` (380 × 380) y `fastvit_t8` (256 × 256).
- **Normalización ImageNet:** Se ajustan los valores de píxel al rango $[0, 1]$ y se restan las medias y desviaciones de referencia ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

La partición `seed_42` estratifica conjuntamente por clase y entorno (`label + environment`) para aproximar 70/15/15 y conservar las nueve clases. Esto no convierte el test en un dominio nuevo: las fuentes conocidas pueden aparecer en los tres splits.

La materialización vigente, regenerada después de corregir los conflictos de integridad del corpus, contiene:

| Partición | Muestras | Proporción |
|---|---:|---:|
| Entrenamiento | 23,400 | 69.9991 % |
| Validación | 5,014 | 14.9990 % |
| Prueba | 5,015 | 15.0019 % |
| **Total** | **33,429** | **100 %** |

Los tres CSV conservan las nueve clases. La auditoría de esta materialización verificó cero solapamientos por `sample_id`, SHA-256 y `effective_group_id` entre particiones; en `seed_42` el `effective_group_id` es el `sample_id` y el SHA-256 es único en el manifiesto, de modo que la comprobación solo descarta muestras o bytes repetidos entre particiones. Ocho archivos involucrados en cuatro pares de contenido idéntico con etiquetas contradictorias se excluyen explícitamente mediante `config/dataset_exclusions.csv`; no se elimina ni se reasigna ninguna muestra de forma silenciosa durante el entrenamiento.

`source_id` se conserva como metadato de procedencia y puede aparecer en las tres particiones de `seed_42`: este split mide rendimiento dentro de las fuentes conocidas, no generalización a una fuente nueva. Para ese segundo escenario existe el protocolo opt-in `seed_42_source_grouped`, activado con `--group-by-source`, que mantiene cada `effective_group_id` en una sola partición. Como algunas clases solo existen en pocas fuentes, ese benchmark requiere `--allow-incomplete-splits` (la validación queda sin `fall_armyworm` ni `lethal_necrosis`) y no sustituye al split principal.

La generación actual calcula SHA-256 y evita duplicados exactos, pero fue ejecutada con `deduplicate_perceptual=false`. Por tanto, los controles anteriores no demuestran ausencia de imágenes casi duplicadas; esa comprobación debe realizarse aparte antes de interpretar el resultado como robustez frente a variaciones visuales cercanas.

---

## Estrategia frente al desbalance de clases

En el corpus final conviven clases masivas como *Healthy* (8,740 imágenes) con clases escasas como *Potassium Deficiency* (621 imágenes), lo que representa una relación de desbalance de 14.1 a 1.

Para compensar el desbalance, el pipeline principal usa tres elementos:

1. **Aumento de datos dirigido (*Data Augmentation*):** Las clases cuya frecuencia en train es superada en más de 4x por la clase mayoritaria (`minority_ratio_threshold: 4.0`) reciben una augmentación más fuerte en tiempo de entrenamiento. Con el train vigente son `gray_leaf_spot` (4.5x), `nitrogen_deficiency`, `phosphorus_deficiency` y `potassium_deficiency`; `common_rust` (3.9x) queda por debajo. Se aplican recortes aleatorios con escala 0.7 a 1.0, rotaciones de hasta 30 grados, variaciones de brillo, contraste, saturación y matiz, y desenfoque gaussiano leve.
2. **Pérdida ponderada suave (`sqrt_inverse`):** El modelo oficial no usa `WeightedRandomSampler` (`sampler: null` en su `summary.json`). La entropía cruzada pondera cada clase con `w = sqrt(N / (K · n_clase))`, con N muestras de train y K clases, de modo que el peso de las clases escasas crece menos que con el inverso de la frecuencia.
3. **Suavizado de etiquetas (*Label Smoothing = 0.10*):** La entropía cruzada se calcula contra etiquetas suavizadas con ε = 0.10.
