# DoctorMaiz - Detección de Enfermedades, Plagas y Deficiencias Nutricionales en Cultivos de Maíz

> **Clasificación mediante Deep Learning en Dispositivos Móviles (Edge AI Offline)**

---

## Descripción

DoctorMaiz es un sistema de clasificación de enfermedades foliares, plagas y deficiencias nutricionales en cultivos de maíz, pensado para pequeños agricultores de subsistencia en zonas rurales sin conectividad. Utiliza un modelo de Deep Learning cuantizado (TensorFlow Lite Int8) embebido en una aplicación Android que opera completamente offline.

Este repositorio contiene el pipeline de datos, entrenamiento, evaluación, interpretabilidad y exportación. La aplicación móvil vive en [`maize-doctor-app`](https://github.com/Edenilson-Molina/maize-doctor-app), el backend opcional en [`maize-doctor-api`](https://github.com/daiv05/maize-doctor-api) y el segmentador de hoja en [`maize-doctor-segmenter-leaf`](https://github.com/abner-rivas/maize-doctor-segmenter-leaf).

### Problema

El punto de partida es el peso que tiene el maíz en El Salvador y lo expuesto que está a perderse sin un diagnóstico oportuno:

- El maíz representa una fuente crítica de alimentación en El Salvador, donde la agricultura aporta el **5.6% del PIB**
- El **82.1% de los productores** son pequeños agricultores con acceso limitado a asistencia técnica
- Las enfermedades, plagas y deficiencias nutricionales pueden destruir hasta el **70% de una cosecha**
- El diagnóstico actual depende de experiencia empírica y no de análisis técnico objetivo

### Solución

Una aplicación móvil que, dada una fotografía de hoja de maíz, identifica la enfermedad, plaga o deficiencia nutricional presente y orienta al agricultor sobre el tratamiento adecuado, sin necesidad de conexión a internet.

---

## Resultados principales

Las tres cifras que delimitan el alcance real del sistema, medidas sobre el conjunto de prueba independiente de **5 015 imágenes**:

| Configuración evaluada | Macro F1 | Accuracy | Entorno de aplicación |
|---|:---:|:---:|---|
| **`EfficientNet-Lite0` (desplegada)** | **0.9468** | **97.91 %** | Inferencia local offline en la app (3.56 MB, ~60 ms) |
| **Ensamble Soft Voting (3 modelos)** | **0.9567** | **98.29 %** | Inferencia en servidor cuando hay conectividad |
| **Generalización a fuente no vista** | **0.6026 ± 0.1240** | **76.4 %** | Comportamiento honesto ante cámaras y parcelas desconocidas |

La distancia entre 0.9468 y 0.6026 es la brecha de dominio: dentro de las condiciones conocidas el modelo es sobresaliente, pero una parcela completamente nueva introduce una caída medible que solo se amortigua recolectando datos locales.

**Estabilidad entre semillas.** La configuración de producción entrenada con tres semillas da 0.9468 (s=42, la desplegada), 0.9448 (s=1) y 0.9375 (s=2): media 0.9430 con σ = 0.0049. Las mejoras por hiperparámetros o ensamble valen entre 1 y 2 σ; cambiar la estrategia de partición (estratificada frente a agrupada por fuente) mueve más de 66 σ. Cómo se parten los datos pesa mucho más que cualquier ajuste fino.

**Equidad.** Macro F1 de 0.9298 en campo real frente a 0.8865 en laboratorio sobre clases con soporte, con un *disparate impact ratio* de 0.9534, por encima de la regla del 80 %. Detalle completo en [FAIRNESS_REPORT.md](FAIRNESS_REPORT.md).

**Alcance: clase, no severidad.** El clasificador responde qué tiene la hoja, no cuánto. No se entrenó un modelo de niveles de daño porque las 31 623 imágenes de `data/clean/` traen una sola etiqueta por imagen y ninguna anotación de severidad: producirla exige un fitopatólogo aplicando la escala diagramática propia de cada patógeno, y partir en tres grados clases que rondan las 300 imágenes habría dejado celdas de dos dígitos. La app compensa esa limitación con una guía de autoevaluación que muestra los niveles del en lenguaje natural.

---

## Clases Objetivo

### Enfermedades y plagas foliares

| Clase | Patógeno/Agente | Síntomas | Lab | Real | Total |
|---|---|---|---:|---:|---:|
| Roya común *(Common Rust)* | *Puccinia sorghi* | Pústulas anaranjadas en ambas caras | 2 150 | 106 (escasa) | 2 256 |
| Tizón foliar del norte *(NCLB)* | *Exserohilum turcicum* | Lesiones alargadas grisáceas | 888 | 5 942 | 6 830 |
| Mancha gris *(GLS)* | *Cercospora zeae-maydis* | Lesiones rectangulares grises | 513 | 1 417 | 1 930 |
| Necrosis letal *(MLN)* | Complejo viral (MCMV + potyvirus) | Rayado clorótico, necrosis progresiva y muerte de la planta | 0 | 6 415 | 6 415 |
| Hoja sana *(Healthy)* | - | Sin síntomas visibles | 0 | 8 744 | 8 744 |
| Gusano cogollero *(Fall Armyworm)* | *Spodoptera frugiperda* | Daño por masticación, excrementos en cogollo | 0 | 4 858 | 4 858 |

> `aphids_pest` (áfidos) se evaluó pero se descartó del alcance: solo ~77 imágenes disponibles, insuficientes para augmentation viable.

### Deficiencias nutricionales

| Clase | Síntomas | Lab | Real | Total |
|---|---|---:|---:|---:|
| Deficiencia de nitrógeno *(Nitrogen)* | Amarillamiento en "V" desde puntas de hojas inferiores | 0 | 846 (escasa) | 846 |
| Deficiencia de fósforo *(Phosphorus)* | Bordes y puntas moradas/rojizas en hojas jóvenes | 0 | 938 (escasa) | 938 |
| Deficiencia de potasio *(Potassium)* | Necrosis marginal en hojas más viejas | 0 | 621 (escasa) | 621 |

> "(escasa)" señala clases con pocas imágenes disponibles, candidatas prioritarias a data augmentation.

Los conteos corresponden al corpus ampliado: **33 438 imágenes** (3 551 lab + 29 887 campo real), tras incorporar cuatro datasets Roboflow dirigidos a GLS y a las tres deficiencias nutricionales (+1 815 netas) y deduplicar con PHash. El desbalance máximo bajó de 32.9x a **14.1x**.

---

## Objetivos Técnicos

| Métrica | Meta | Resultado |
|---|---|---|
| Macro F1 Score | ≥ 0.85 | 0.9468 en el modelo desplegado; 0.9567 en ensamble |
| Tamaño del modelo (post Int8) | ≤ 20 MB | 3.56 MB (`efficientnet_lite0`) |
| Latencia de inferencia | ≤ 300 ms/imagen | ~60 ms en Dimensity 7030 |
| Dispositivo objetivo | Android ≥ 4 GB RAM, Snapdragon 6xx | Validado en Motorola edge 40 neo |
| Arquitectura desplegada | - | `efficientnet_lite0` Int8, dos salidas (logits + features) |

---

## Datasets

Se consolidaron **8 fuentes de datos públicas** para construir el corpus de entrenamiento:

| Dataset | Dominio | Imágenes (maíz) | Licencia |
|---|---|---|---|
| [Maize in Field](docs/es/datasets/maize-in-field-dataset.md) | Campo real (Sudáfrica) | ~2 223 | CC BY-NC-SA 4.0 |
| [Maize Diseases](docs/es/datasets/maize-diseases.md) | Lab + campo (PlantVillage v1.0/v1.1) | ~16 162 | CC BY-NC-SA 4.0 |
| [Corn Leaf Diseases](docs/es/datasets/corn-leaf-diseases.md) | Lab augmentado (x17) | 52 360 | MIT |
| [CropDG Unified Multidomain](docs/es/datasets/cropdg-unified-multidomain.md) | Multi-dominio | ~13 275 | CC BY-NC-SA 4.0 |
| [Maize, Beans & Tomatoes Africa](docs/es/datasets/maize-beans-tomatoes-africa.md) | Campo real (África) | 23 286 | Apache 2.0 + CC |
| [Multicrop Disease - Maize Pests and Disease](docs/es/datasets/multicrop-disease-maiz-disease-pests-and-disease.md) | Mixto | - | Desconocida |
| [Maize Nutrient Deficiency](docs/es/datasets/maize-nutrient-deficiency.md) | Campo real (India) | 463 | CC BY 4.0 |
| [Corn Leaf - Roboflow](docs/es/datasets/corn-leaf-roboflow.md) | Campo real | 3 943 | CC BY 4.0 |

### Estrategia de Augmentation

El dataset *Corn Leaf Diseases* aplica 17 técnicas de augmentation documentadas:

`brightness_adjusted` · `contrast_adjusted` · `cropped` · `flipped_horizontal` · `flipped_vertical` · `gaussian_noise` · `high_pass` · `hist_equalized` · `jittered` · `laplacian` · `poisson_noise` · `rotated` · `salt_pepper_noise` · `saturation_adjusted` · `sobel` · `translated` · `unsharp_mask`

### Procedencia y fuga entre fuentes

Una partición estratificada clásica deja que imágenes del mismo repositorio caigan a ambos lados del split, lo que infla las métricas. La línea de trabajo en [docs/es/provenance](docs/es/provenance/index.md) audita ese efecto, construye una partición agrupada por fuente y mide la generalización con *leave-one-source-out*. De ahí sale la tercera cifra de la tabla de resultados.

---

## Metodología

El proyecto avanza en fases iterativas siguiendo el marco **CRISP-DM**:

1. **Comprensión del negocio**: definición del problema agrícola y restricciones de despliegue
2. **Comprensión de datos**: consolidación y auditoría de 8 fuentes públicas
3. **Preparación**: limpieza, estandarización (224x224 px), deduplicación, augmentation
4. **Modelado**: baselines para comparar barato, luego pipeline principal con optimización bayesiana
5. **Evaluación**: Macro F1 ≥ 0.85 en conjunto independiente, validación cruzada y auditoría de equidad
6. **Despliegue**: TFLite Int8 en la app Android, con detector fuera de dominio

---

## Pipeline de Machine Learning

El código vive en `src/` (librería instalable, `pip install -e .`) y `scripts/` (entrypoints). Sobre el mismo dataset limpio (`clean/`) conviven dos pipelines paralelos:

- **Baselines** (`scripts/pipeline/train_baselines.py`): entrenado sobre el perfil `baseline` (`config/dataset.yaml -> baseline:`, 9 clases, cap de 1 500 imágenes por clase) con tres arquitecturas canónicas pensadas para comparar rápido y barato; ver [Baselines](docs/es/baselines/index.md).
- **Pipeline principal** (`scripts/pipeline/train.py`): entrenamiento completo sobre el corpus, con optimización de hiperparámetros vía Optuna, ensamble por voto suave, validación cruzada K-Fold, auditoría de equidad y exportación a ONNX/TFLite.

Guía de instalación local (venv, `.env`, dataset) en [LOCAL.md](LOCAL.md).

### Del checkpoint al teléfono

`export.py` convierte el `best.pth` a ONNX y TFLite, validando paridad numérica entre runtimes. El modelo se envuelve en `FeatureExposedModel` para exponer **dos salidas**: los logits y las features pooled que alimentan el detector fuera de dominio. `compute_ood_stats.py` calcula los centroides, la covarianza y el umbral de Mahalanobis que la app usa para rechazar imágenes que no son hojas de maíz. `sync_mobile_model.py` copia el trío `.tflite` + `labels.json` + `ood_stats.json` a la app verificando el hash.

| Arquitectura | FP32 | Int8 | Reducción |
|---|:---:|:---:|:---:|
| `shufflenet_v2_x1_0` | 4.93 MB | 1.42 MB | 71 % |
| `efficientnet_lite0` (desplegada) | 12.92 MB | 3.56 MB | 72 % |
| `efficientnet_b0` | 15.48 MB | 4.44 MB | 71 % |

---

## Comandos

Todos los comandos usan `make` (detecta Windows/Linux automáticamente). El nombre del target dice dónde corre y sobre qué pipeline: prefijo `modal-` = GPU en la nube (sin prefijo = local), sufijo `-baselines` = runs de baselines (variable `MODELS`), sufijo `-main` = runs del pipeline principal (variable `MAIN_MODELS`). `make help` los lista agrupados.

Variables comunes: `MODELS` / `MAIN_MODELS` (nombre o "all"), `EPOCHS` / `MAIN_EPOCHS`, `NO_CAP=1` / `MAX_PER_CLASS=<n>` (override del tope de imágenes por clase del perfil baseline), `RUN` (run_id específico), `SAMPLE_SIZE`.

### Locales: setup y datos

```bash
make install                         # pip install -e ".[dev,analysis,xai,cloud]"
make download-dataset                # clean/ (HF Hub, fallback Google Drive)
make upload-dataset STAGE_DIR=<dir>  # empaqueta clean/ en shards .tar y publica en HF

make splits                          # splits completos (9 clases) -> outputs/splits/seed_42/
make splits-baseline [NO_CAP=1 | MAX_PER_CLASS=<n>]   # perfil baseline -> outputs/splits/seed_42_baseline/

make clean-outputs                   # borra outputs/ (splits, runs, reportes - todo regenerable)
make summary / make test-loader / make lint / make fmt / make check
```

### Locales: baselines (`outputs/baselines/`)

```bash
make train-baselines [MODELS=<nombre>] [NO_CAP=1 | MAX_PER_CLASS=<n>]

make explain-visual-baselines [MODELS=<nombre> RUN=<id> IMAGE=<ruta> OUTPUT=<ruta>]
make explain-fidelity-baselines [MODELS=<nombre> RUN=<id> SAMPLE_SIZE=<n> NUM_SAMPLES=<n>]
make explain-errors-baselines [MODELS=<nombre> RUN=<id> NUM_SAMPLES=<n>]
```

### Locales: pipeline principal (`outputs/main/`)

```bash
make train-main [MAIN_MODELS=<nombre> MAIN_EPOCHS=<n> CLAHE=1 CLASS_WEIGHTS=<estrategia>]
                [EXPORT_FORMATS=onnx,tflite]          # alias: make train
make tune-main [N_TRIALS=20 MAIN_EPOCHS=30 PRUNER=median]   # Optuna HPO
make tune-dashboard                                          # optuna-dashboard en el puerto 8080
make evaluate-ensemble [MODELS=<lista>]                      # voto suave sobre test.csv; alias: make ensemble
make cross-validate [MODEL=<nombre> K_FOLDS=<n>]             # alias: make kfold
make fairness-report [MODEL=<nombre> CHECKPOINT=<ruta>]      # equidad + Grad-CAM + test de atajos

make export-main [EXPORT_FORMATS=onnx,tflite QUANTIZE=int8]
make compute-ood-stats [MAIN_MODELS=<nombre> RUN=<id> PERCENTILE=<p>]
make eval-export-main [EXPORT_FORMATS=tflite QUANTIZE=int8]  # mide el exportado sobre test completo
make sync-mobile-model RUN_DIR=<dir> DEST=<dir de la app>

make predict IMAGE=<ruta> [MODEL=ensemble TOP_K=4]           # predicción rápida (default: ensemble)
make inference MODEL=<nombre> IMAGE=<ruta> [STABILITY_RUNS=<n>]   # reporte de inferencia

make explain-visual-main [MAIN_MODELS=<nombre> RUN=<id>]
make explain-fidelity-main [MAIN_MODELS=<nombre> RUN=<id> SAMPLE_SIZE=<n>]
make explain-errors-main [MAIN_MODELS=<nombre> RUN=<id> NUM_SAMPLES=<n>]
make explain-compare-main [MAIN_MODELS=<nombre> RUN=<id> SAMPLE_SIZE=<n>]   # panel LIME | SHAP | Grad-CAM
make explain-global-main [MAIN_MODELS=<nombre> RUN=<id> SAMPLE_SIZE=<n>]    # perfil global con SHAP
```

### Modal (GPU en la nube)

Misma CLI que los comandos locales: cualquier combinación de banderas que funcione en local funciona igual en Modal. Detalle completo en [docs/es/deployment/modal.md](docs/es/deployment/modal.md).

```bash
make modal-seed                    # sube clean/ al Volume (una vez); FORCE=1 lo vacía y re-descarga
make modal-splits                  # regenera splits tras actualizar el dataset
make modal-clean-outputs           # vacía el Volume corn-outputs
make modal-pull                    # trae outputs-remote/ con runs + reportes

# dataset pre-segmentado (usa el checkpoint de maize-doctor-segmenter)
make modal-segment-dataset
make modal-splits-segmented
make modal-train SEGMENTED=1

# baselines (/outputs/baselines)
make modal-train-baselines [MODELS=<nombre>] [NO_CAP=1 | MAX_PER_CLASS=<n>]
make modal-explain-visual-baselines / modal-explain-fidelity-baselines / modal-explain-errors-baselines

# pipeline principal (/outputs/main)
make modal-train-main [MAIN_MODELS=<nombre> MAIN_EPOCHS=<n> CLAHE=1]   # alias: modal-train
make modal-tune / modal-tune-dashboard
make modal-ensemble / modal-cross-validate / modal-fairness
make modal-export-main / modal-compute-ood-stats / modal-eval-export-main
make modal-explain-visual-main / modal-explain-fidelity-main / modal-explain-errors-main
make modal-explain-compare-main / modal-explain-global-main

# procedencia y fuga entre fuentes
make provenance-audit                                  # local, sin GPU
make modal-provenance-leak / modal-loso / modal-aligned-loso / modal-pull-provenance
```

---

## Equipo

| Nombre | Carné |
|---|---|
| Josias Abner Rivas Fuentes | RF20010 |
| David Alejandro Deras Cerros | DC19019 |
| Elmer Edenilson Rosales Molina | RM20001 |

---

## Estructura del Proyecto

```
maize-doctor-classifier/
├── config/
│   ├── dataset.yaml           # Clases, tamaño de imagen, seed, perfil "baseline", LIME/SHAP
│   └── dataset.segmented.yaml # Variante sobre el corpus segmentado
├── docs/es/                   # Documentación (VitePress): datasets, EDA, pipeline, resultados, provenance
├── notebooks/                 # Análisis exploratorio
├── reports/                   # Informes LaTeX/HTML de cada entrega
├── scripts/
│   ├── checks/                # Verificaciones puntuales (estabilidad de LIME, etc.)
│   ├── dataset/               # Subida/descarga de clean/ (Hugging Face Hub, Google Drive)
│   ├── experiments/           # Validación cruzada de procedencia y fuga
│   ├── modal/                 # Entrenamiento, exportación y explicabilidad en GPU de Modal
│   └── pipeline/              # splits, train, tune, ensemble, fairness, export, ood, explain
├── src/
│   ├── analysis/              # Resumen de dataset, equidad, análisis de predicciones
│   ├── data/                  # CornDataset, loader, splitter, transforms, dedupe, procedencia
│   ├── explainability/        # LIME + Grad-CAM + SHAP, acuerdo, perfil global, máscara foliar
│   ├── export/                # ONNX/TFLite, paridad entre runtimes, evaluación del exportado
│   ├── models/                # Registro de modelos, ensamble, FeatureExposedModel
│   ├── segmentation/          # Consumo del segmentador de hoja
│   └── training/              # Loop de entrenamiento, balanceo, versionado de runs
├── FAIRNESS_REPORT.md
├── Makefile
└── pyproject.toml
```

Documentación completa construida con VitePress (`npm install && npm run docs:dev`, disponible en `http://localhost:5173`).

---

## Estado del Proyecto

- [x] Documentación de datasets consolidados (8 fuentes, 9 clases)
- [x] Scripts de limpieza y organización de datos en `data/clean/`
- [x] Análisis exploratorio de datos (EDA)
- [x] Pipeline de preparación de datos (splits estratificados, perfil baseline configurable)
- [x] Data augmentation para clases minoritarias (pipeline extendido por clase en `transforms.py`)
- [x] Baselines sobre 9 clases + entrenamiento en GPU de Modal
- [x] Explicabilidad post-hoc (LIME + Grad-CAM + SHAP, análisis de errores, fidelidad y perfil global)
- [x] Pipeline principal de entrenamiento (`scripts/pipeline/train.py`)
- [x] Optimización bayesiana de hiperparámetros con Optuna
- [x] Ensamble por voto suave de tres arquitecturas
- [x] Validación cruzada K-Fold y experimento 2x2 de particiones
- [x] Auditoría de procedencia, fuga entre fuentes y leave-one-source-out
- [x] Análisis de sesgos, equidad y ética ([FAIRNESS_REPORT.md](FAIRNESS_REPORT.md))
- [x] Exportación a ONNX/TFLite Int8 con validación de paridad entre runtimes
- [x] Detector fuera de dominio (Mahalanobis relativo) y sincronización con la app
- [x] Aplicación Android con TensorFlow Lite, validada en dispositivo físico
- [x] Guía de severidad en lenguaje llano dentro de la app, frente a la ausencia de etiquetas de nivel de daño

---

## Licencia

El código de este repositorio se distribuye bajo la licencia MIT. Ver [LICENSE](LICENSE).

La licencia MIT cubre **el código, no los datos**. Los 8 datasets consolidados conservan sus licencias originales, varias de ellas más restrictivas: `Maize in Field`, `Maize Diseases` y `CropDG Unified Multidomain` son CC BY-NC-SA 4.0, que prohíbe el uso comercial y obliga a compartir los derivados en los mismos términos. Cualquier redistribución del corpus, de los splits o de artefactos derivados de esas fuentes queda sujeta a esos términos, no a MIT. La licencia de cada fuente está documentada en la tabla de datasets y en su página individual.

Proyecto académico desarrollado en la Universidad de El Salvador.
