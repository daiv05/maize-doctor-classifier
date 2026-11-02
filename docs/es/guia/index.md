# Guía

DoctorMaiz clasifica fotografías de hojas de maíz en nueve condiciones (hoja sana, roya común, tizón foliar del norte, mancha gris, necrosis letal, gusano cogollero y deficiencias de nitrógeno, fósforo y potasio) y opera sin conexión en teléfonos Android de gama media y baja. Este sitio documenta los datos, el modelado, los pipelines, los resultados y la app.

## Componentes

| Componente | Contenido | Dónde |
|---|---|---|
| Pipeline de datos y entrenamiento | Preparación del corpus, entrenamiento, HPO, evaluación, XAI, exportación y detector OOD | Este repositorio |
| Modelo móvil | EfficientNet-Lite0 cuantizado a int8 (3.74 MB), 224 × 224, con detector fuera de dominio | [Resultados](/es/resultados/) |
| Aplicación | App React Native que clasifica en el teléfono | [Manual de la app](/es/app/) y [Despliegue](/es/deployment/react-native) |

## Flujo de una predicción

```mermaid
flowchart LR
    A["Captura con marco guía"] --> B["Recorte, escalado a 224×224 y normalización"]
    B --> C["EfficientNet-Lite0 int8"]
    C --> D{"Filtro OOD<br>Mahalanobis"}
    D -- "Fuera de dominio" --> E["Imagen no reconocida"]
    D -- "Válida" --> F["Diagnóstico y recomendación"]
```

El preprocesado y la inferencia suman unos 121 ms de mediana en el teléfono de pruebas, dentro de la meta de 300 ms ([Prototipo en dispositivo](/es/deployment/prototipo)).

## Cómo reproducir el proyecto

Requisitos: Python 3.11, el entorno de `LOCAL.md` y el corpus `clean/`.

```bash
make install
make download-dataset
make splits
make train-main MAIN_MODELS=efficientnet_lite0
make export-main
make compute-ood-stats
```

La preparación y los splits están en [Pipeline de datos](/es/metodologia/pipeline-datos), el entrenamiento en [Pipeline principal](/es/pipeline/entrenamiento) y el contrato de los runs en [Contratos de runs](/es/pipeline/contratos-runs). El entrenamiento en GPU remota se describe en [Modal](/es/deployment/modal).

## Mapa del sitio

- **Datos:** [Fuentes](/es/datasets/), [clases](/es/dataset-classes/), [limpieza](/es/cleanup-and-ordered/), [análisis exploratorio](/es/exploratory-data-analysis/) y [preprocesado](/es/preprocessed/).
- **Modelado:** [Baselines](/es/baselines/), [interpretabilidad](/es/deep-learning/interpretability) y [detección OOD](/es/deep-learning/ood-detection).
- **Pipelines:** [Entrenamiento](/es/pipeline/entrenamiento), [evaluación](/es/pipeline/evaluacion), [HPO](/es/experimentos/hpo) y [multi-semilla](/es/experimentos/multiseed).
- **Resultados:** [Modelo oficial](/es/resultados/) y [generalización entre fuentes](/es/resultados/generalizacion).
- **Reproducibilidad:** [Registro de evidencias](/es/reproducibilidad/evidencias) y [procedencia de figuras](/es/reproducibilidad/figuras).
