# Historia ejecutiva de DoctorMaiz

DoctorMaiz nació en mayo de 2026 para clasificar, sin conexión y desde un teléfono, nueve condiciones visibles en hojas de maíz. La primera dificultad fue de datos: reunir repositorios con etiquetas, fondos, cámaras y estructuras incompatibles. Entre junio y julio se consolidó `clean/`, se construyó el pipeline PyTorch y se exploraron ocho arquitecturas ligeras. La evidencia comparable llevó a trabajar principalmente con EfficientNet-B0, ShuffleNet-V2-x1.0 y EfficientNet-Lite0.

La búsqueda no fue lineal. La segmentación YOLO se introdujo porque el clasificador podía mirar demasiado fondo u otras hojas. Al entrenar sobre un corpus segmentado, el run verificable obtuvo 0.7191 de Macro-F1: una caída que mostró que una máscara puede eliminar contexto útil o escoger la hoja equivocada. Por eso la segmentación quedó como estrategia opt-in con selección de instancia y fallback, no como reemplazo automático de la imagen completa.

En agosto, la ampliación del corpus, el pipeline principal y la exportación TFLite hicieron posible desplegar un EfficientNet-Lite0 histórico (`20260812_221429`) con 0.9468 de Macro-F1 en un test estratificado. En septiembre se añadieron Optuna, ensamble, CV y auditoría de procedencia. El ensamble alcanzó 0.9567 dentro de fuentes conocidas, pero la CV agrupada cayó a 0.6026 ± 0.1240. LOSO e intervenciones sobre fondos confirmaron que el cambio de fuente es un problema mucho mayor que un ajuste marginal de hiperparámetros.

`master` y `dev-abner` evolucionaron en paralelo. `master` tenía caché de imágenes, tope por clase, PHash y un splitter source-grouped; `dev-abner` aportaba identidad y contratos, pero también alternativas que podían perder esas garantías o hashear dentro de `__getitem__`. La rama `dev-abner2` integró selectivamente ambos lados.

El cambio central fue separar identidad, contenido y relación: `sample_id` dice quién es la muestra, SHA-256 qué bytes posee y source/group con qué dominio se relaciona. También se eliminó un fallo histórico por el que una imagen ilegible podía ser sustituida por la fila siguiente. Ahora el batch conserva `(image, label, sample_id)` y un fallo identifica esa misma muestra.

El incidente del 21 de septiembre terminó de aclarar la metodología. Tratar once `source_id` como grupos indivisibles produjo 49.52/23.36/27.12 %, clases ausentes en validación, train Macro-F1 0.9664 y validación 0.2669. No era un split inútil: medía generalización cross-source. Pero no servía como único split para selección balanceada. Se conservaron dos protocolos.

El punto de corte actual es EfficientNet-Lite0 `20260921_204608`, entrenado sobre `seed_42` estratificado por clase+ambiente: 23 400/5 014/5 015, mejor época 39, validación Macro-F1 0.9561, test Macro-F1 0.9480 y accuracy 97.81 %. Sigue siendo un baseline de desarrollo, no una proclamación de producción: mezcla fuentes conocidas y aún no tiene auditoría perceptual, validación cross-source final ni exportación/dispositivo del nuevo checkpoint.

La siguiente fase es un estudio Optuna de 60 trials con test cerrado, seguido de entrenamiento formal, multi-seed y validaciones estratificada, source-grouped y LOSO claramente separadas.
