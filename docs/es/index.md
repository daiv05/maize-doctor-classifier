---
layout: home

hero:
  name: "DoctorMaiz"
  text: "Enfermedades, Plagas y Deficiencias en Maíz"
  tagline: "Sistema de clasificación de enfermedades foliares, plagas y deficiencias nutricionales para pequeños productores de maíz en El Salvador, con inferencia completamente offline en dispositivos Android de gama media/baja."
  image:
    src: /logo.svg
    alt: DoctorMaiz
  actions:
    - theme: brand
      text: Ver Datasets
      link: /es/datasets/
    - theme: alt
      text: Baselines
      link: /es/baselines/

features:
  - title: "9 Clases"
    details: "Clasificación de 6 enfermedades foliares y plagas (roya, NCLB, GLS, necrosis letal, cogollero, sana) y 3 deficiencias nutricionales (N, P, K) mediante CNN con transferencia de aprendizaje."
  - title: "Edge AI Offline"
    details: "Entrenamiento en PyTorch y exportación a TensorFlow Lite con cuantización Int8, con objetivo ≤ 20 MB y latencia ≤ 300 ms en CPU Snapdragon serie 6xx o equivalente."
  - title: "Orientado al Campo"
    details: "Evaluación desagregada por ambiente y fuente. El split principal mezcla fuentes conocidas; la generalización cross-source se mide con protocolos separados."
  - title: "Meta Macro F1 ≥ 0.85"
    details: "Criterio de viabilidad con análisis de matriz de confusión y curvas Precision-Recall por clase, priorizando Recall para minimizar falsos negativos."
---

## El Proyecto

**Detección de Enfermedades Foliares, Plagas y Deficiencias Nutricionales en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles**

| | |
|---|---|
| **Institución** | Universidad de El Salvador - Facultad de Ingeniería y Arquitectura |
| **Programa** | Escuela de Ingeniería de Sistemas Informáticos - Curso de Especialización en Machine Learning |
| **Ciclo** | I 2026 - Grupo 02 |
| **Docente** | Ing. Bladimir Díaz Campos |

### Equipo

| Nombre | Carnet |
|---|---|
| Josias Abner Rivas Fuentes | RF20010 |
| David Alejandro Deras Cerros | DC19019 |
| Elmer Edenilson Rosales Molina | RM20001 |

### Contexto y Problema

Para entender por qué existe este proyecto conviene partir del peso que tiene la agricultura en El Salvador: representa el 5.6% del PIB y es el sustento de más de 2 millones de personas rurales. El 82.1% de los productores son pequeños agricultores, muchos operando a nivel de subsistencia. El maíz, su principal cultivo, es vulnerable a enfermedades foliares, plagas y deficiencias nutricionales que, sin detección correcta y en etapas tempranas, pueden destruir hasta el 70% de la cosecha.

El problema es que en zonas rurales el acceso a asistencia técnica es limitado, así que los diagnósticos terminan dependiendo de la experiencia empírica del agricultor. Eso puede generar detecciones tardías y pérdidas económicas significativas, en 2023 la cosecha ya había caído un tercio respecto a 2021.

### Clases Objetivo

#### Enfermedades y plagas foliares

| Clase | Nombre en inglés | Patógeno/Agente | Síntoma visual | Lab | Real | Total |
|---|---|---|---|---:|---:|---:|
| **Roya común** | Common Rust | *Puccinia sorghi* | Pústulas anaranjadas dispersas en ambas caras de la hoja | 2 150 | 106 (pocos datos) | 2 256 |
| **Tizón foliar del norte (NCLB)** | Northern Corn Leaf Blight | *Exserohilum turcicum* | Lesiones alargadas grisáceas o marrones con bordes difusos | 888 | 5 942 | 6 830 |
| **Mancha gris de la hoja (GLS)** | Gray Leaf Spot | *Cercospora zeae-maydis* | Lesiones rectangulares grises o marrones delimitadas por nervaduras | 513 | 1 417 | 1 930 |
| **Necrosis letal del maíz (MLN)** | Lethal Necrosis | *MCMV + SCMV* | Moteado clorótico severo, necrosis y muerte progresiva de la planta | 0 | 6 415 | 6 415 |
| **Hoja sana** | Healthy | - | Sin síntomas foliares de enfermedad | 0 | 8 740 | 8 740 |
| **Gusano cogollero** | Fall Armyworm | *Spodoptera frugiperda* | Daño por masticación con excrementos en el cogollo y hojas | 0 | 4 853 | 4 853 |

#### Deficiencias nutricionales

| Clase | Nombre en inglés | Síntoma visual | Lab | Real | Total |
|---|---|---|---:|---:|---:|
| **Deficiencia de nitrógeno** | Nitrogen Deficiency | Amarillamiento en "V" desde la punta de hojas inferiores | 0 | 846 (pocos datos) | 846 |
| **Deficiencia de fósforo** | Phosphorus Deficiency | Bordes y puntas moradas/rojizas en hojas jóvenes | 0 | 938 (pocos datos) | 938 |
| **Deficiencia de potasio** | Potassium Deficiency | Necrosis marginal en hojas más viejas | 0 | 621 (pocos datos) | 621 |

> Conteos de la materialización vigente: **33 429 muestras elegibles** (3 551 lab + 29 878 campo real), después de ocho exclusiones verificadas. Las marcas "(pocos datos)" señalan las clases con menor soporte. La instantánea anterior de agosto contenía 33 438 y se conserva como registro histórico.

> **Evolución histórica.** El corpus pasó de **31 622** a la instantánea ampliada de **33 438 imágenes** al incorporar cuatro datasets Roboflow. El manifest actual descubre 33 437 y deja 33 429 elegibles. Detalle en [Limpieza](/es/cleanup-and-ordered/), [EDA histórica](/es/exploratory-data-analysis/) y [pipeline vigente](/es/metodologia/pipeline-datos).


### Metodología

El proyecto avanza en fases iterativas siguiendo **CRISP-DM**:

1. **Comprensión del negocio**: análisis del impacto en el sector agrícola salvadoreño
2. **Comprensión de los datos**: consolidación multi-fuente de datasets públicos; ver [Recopilación de datasets](/es/datasets/)
3. **Preparación de los datos**: limpieza, estandarización a 224 x 224 px y data augmentation
4. **Modelado**: transfer learning en PyTorch con modelos preentrenados en ImageNet
5. **Evaluación**: Macro-F1 sobre test retenido, desgloses por fuente/ambiente y protocolos cross-source separados
6. **Despliegue**: exportación del modelo PyTorch a TFLite (Int8) y PWA con inferencia offline + módulo opcional de sincronización

### Arquitectura del Sistema

```
Captura / Galería
      ↓
Preprocesamiento (224x224, normalización)
      ↓
Inferencia CNN - TensorFlow Lite (Int8, exportado desde PyTorch)
      ↓
Clase predicha + nivel de confianza
      ↓ (cuando hay conexión)
Módulo de sincronización - API FastAPI + MySQL
```

### Restricciones Técnicas

- Dispositivo objetivo: Android ≥ 4 GB RAM, CPU Snapdragon serie 6xx o equivalente
- Tamaño del modelo: ≤ 20 MB (post cuantización Int8)
- Latencia de inferencia: ≤ 300 ms por imagen
- Funcionamiento completamente offline como característica principal
