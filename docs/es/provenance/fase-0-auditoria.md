# Fase 0 — Auditoría de Procedencia

Antes de evaluar cómo se comportaba el modelo frente a fuentes desconocidas, era indispensable auditar con precisión de dónde venía cada una de las 33,437 imágenes del corpus y qué relación existía entre los repositorios originales y las nueve clases fitosanitarias.

Esta fase inicial de instrumentación permitió mapear la procedencia de cada archivo y verificar la higiene del conjunto de datos.

---

## Mapeo y trazabilidad del corpus

El corpus consolidado reúne imágenes provenientes de 14 fuentes públicas documentadas (repositorios de Kaggle, Mendeley y Roboflow). A través del esquema de nomenclatura de los archivos, derivamos un identificador de procedencia único para cada imagen:

- **100 % de las imágenes trazadas:** Todas las 33,437 fotografías fueron asignadas a su dataset de origen sin ambigüedades.
- **Higiene y deduplicación:** Se verificó la integridad mediante hashes criptográficos (SHA-256) y hashes perceptuales. Cualquier duplicado exacto entre carpetas fue descartado para garantizar que no existiera ninguna copia idéntica repetida entre las particiones de entrenamiento y prueba.

---

## Distribución de fuentes por clase

El análisis cruzado entre procedencia y patología reveló una concentración muy marcada que explicaba por qué los modelos convencionales podían memorizar las sesiones fotográficas:

| Patología / Condición | Total de imágenes | Fuentes disponibles | Fuente dominante | Concentración en la fuente principal |
|---|---:|:---:|---|:---:|
| **Planta sana (`healthy`)** | 8,744 | 6 | `maize_desease_v1.1` | **54.2 %** |
| **Tizón foliar (NCLB)** | 6,830 | 5 | `maize_africa` | **61.8 %** |
| **Necrosis letal (MLN)** | 6,415 | 2 | `multi_desease` | **50.4 %** |
| **Gusano cogollero** | 4,858 | 3 | `maize_africa` | **49.9 %** |
| **Roya común** | 2,256 | 3 | `maize_desease` | **52.8 %** |
| **Mancha gris (GLS)** | 1,930 | 4 | `maize_field` | **31.4 %** |
| **Deficiencia de fósforo** | 938 | 4 | `corn_leaf_roboflow` | **53.2 %** |
| **Deficiencia de nitrógeno** | 846 | 4 | `corn_leaf_roboflow` | **50.1 %** |
| **Deficiencia de potasio** | 621 | 4 | `maize_2_roboflow` | **49.6 %** |

El dato más revelador fue que **cinco fuentes públicas aportaban una sola clase**. Por ejemplo, tres repositorios sumaban más de 7,000 imágenes que eran en su totalidad plantas sanas. Si una red aprende a reconocer los detalles de la cámara o la luz de esas tres fuentes, acierta automáticamente que la planta está sana sin haber aprendido nada sobre hojas.

Este diagnóstico confirmó la necesidad de diseñar un protocolo de partición agrupada por fuente, ya que una división aleatoria ordinaria repartía estas fuentes en proporciones idénticas entre entrenamiento y prueba, enmascarando el atajo visual.
