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
      text: Empezando
      link: /es/guia/
    - theme: alt
      text: Fuentes de datos
      link: /es/datasets/
    - theme: alt
      text: Resultados
      link: /es/resultados/

features:
  - icon: 🎯
    title: "9 Clases Fitosanitarias"
    details: "Detección de 4 patologías fúngicas/virales, 1 plaga (cogollero), 3 deficiencias nutricionales (N, P, K) y control de hoja sana mediante CNN con transfer learning."
  - icon: ⚡
    title: "Inferencia en el dispositivo"
    details: "Entrenamiento en PyTorch y exportación a TensorFlow Lite int8 (3.74 MB), con inferencia de 23 ms y preprocesado de 98 ms (medianas) en un teléfono de gama media, sin internet."
  - icon: 🌾
    title: "Diseñado para el Campo"
    details: "Encuadre guiado y recorte al marco de la cámara para reducir el fondo. La generalización a fotos de campo externas al corpus es una limitación documentada."
  - icon: 🛡️
    title: "Evaluación por fuente y rechazo OOD"
    details: "Evaluación con fuentes retenidas (Leave-One-Source-Out) y detector fuera de distribución por distancia de Mahalanobis para no emitir diagnósticos sobre imágenes ajenas al entrenamiento."
---

<div class="dm-kpi-banner">
  <div class="dm-kpi-card">
    <div class="dm-kpi-val">33 429</div>
    <div class="dm-kpi-lbl">Muestras elegibles</div>
  </div>
  <div class="dm-kpi-card">
    <div class="dm-kpi-val">0.948</div>
    <div class="dm-kpi-lbl">Macro-F1 de prueba (Lite0)</div>
  </div>
  <div class="dm-kpi-card">
    <div class="dm-kpi-val">3.74 MB</div>
    <div class="dm-kpi-lbl">Modelo int8</div>
  </div>
  <div class="dm-kpi-card">
    <div class="dm-kpi-val">23 ms</div>
    <div class="dm-kpi-lbl">Inferencia en el teléfono (mediana)</div>
  </div>
</div>

## El Proyecto

**Detección de Enfermedades Foliares, Plagas y Deficiencias Nutricionales en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles**

Universidad de El Salvador, Facultad de Ingeniería y Arquitectura, Escuela de Sistemas Informáticos. Curso de Especialización en Machine Learning, Ciclo I 2026. Autores: Josias Abner Rivas Fuentes, David Alejandro Deras Cerros y Elmer Edenilson Rosales Molina.

### Contexto y Problema

La agricultura representa el 5.6 % del PIB de El Salvador y es el sustento de más de 2 millones de personas rurales. El 82.1 % de los productores son pequeños agricultores, muchos operando a nivel de subsistencia. El maíz, su principal cultivo, es vulnerable a enfermedades foliares, plagas y deficiencias nutricionales que, sin detección temprana, pueden destruir hasta el 70 % de la cosecha.

En zonas rurales el acceso a asistencia agronómica presencial es limitado, por lo que el diagnóstico recae casi exclusivamente en la intuición empírica. Esto provoca detecciones tardías y pérdidas acumuladas significativas.

### Clases Objetivo del Modelo

El sistema evalúa 9 condiciones foliares agrupadas en cuatro categorías fitosanitarias: **tejido sano** (control de referencia), **enfermedades infecciosas**, **plagas** y **deficiencias nutricionales**:

| Categoría | Clase | Nombre en inglés | Agente / Causa | Síntoma visual | Lab | Real | Total |
|:---:|---|---|---|---|---:|---:|---:|
| <span class="dm-pill-sano">Sano</span> | **Hoja sana** | Healthy | Control biológico (tejido óptimo) | Sin lesiones, lámina foliar verde uniforme | 0 | 8 740 | 8 740 |
| <span class="dm-pill-enf">Enfermedad</span> | **Roya común** | Common Rust | *Puccinia sorghi* | Pústulas anaranjadas pulverulentas en ambas caras | 2 150 | 106 | 2 256 |
| <span class="dm-pill-enf">Enfermedad</span> | **Tizón foliar del norte (NCLB)** | Northern Corn Leaf Blight | *Exserohilum turcicum* | Lesiones elípticas grisáceas con bordes difusos | 888 | 5 942 | 6 830 |
| <span class="dm-pill-enf">Enfermedad</span> | **Mancha gris de la hoja (GLS)** | Gray Leaf Spot | *Cercospora zeae-maydis* | Lesiones rectangulares estrechas delimitadas por nervaduras | 513 | 1 417 | 1 930 |
| <span class="dm-pill-enf">Enfermedad</span> | **Necrosis letal del maíz (MLN)** | Lethal Necrosis | *MCMV + SCMV* | Moteado severo, clorosis y secado progresivo | 0 | 6 415 | 6 415 |
| <span class="dm-pill-plaga">Plaga</span> | **Gusano cogollero** | Fall Armyworm | *Spodoptera frugiperda* | Masticación irregular y excrementos en el verticilo | 0 | 4 853 | 4 853 |
| <span class="dm-pill-nut">Nutrición</span> | **Deficiencia de nitrógeno** | Nitrogen Deficiency | Carencia de N | Amarillamiento en "V" desde la punta en hojas viejas | 0 | 846 | 846 |
| <span class="dm-pill-nut">Nutrición</span> | **Deficiencia de fósforo** | Phosphorus Deficiency | Carencia de P | Bordes y puntas púrpuras/violáceas en hojas jóvenes | 0 | 938 | 938 |
| <span class="dm-pill-nut">Nutrición</span> | **Deficiencia de potasio** | Potassium Deficiency | Carencia de K | Secado y necrosis marginal en hojas inferiores | 0 | 621 | 621 |
| **Total** | **9 clases objetivo** | — | — | — | **3 551** | **29 878** | **33 429** |

Corpus activo de 33 429 muestras elegibles (3 551 de laboratorio y 29 878 de campo) tras 8 exclusiones. Ver [Generalización entre fuentes](/es/resultados/generalizacion) y [Pipeline de Datos](/es/metodologia/pipeline-datos).

### Metodología

El proyecto sigue el ciclo **CRISP-DM**:

1. **Comprensión del negocio**: análisis de las condiciones de campo y limitaciones de conectividad rural en El Salvador.
2. **Comprensión de los datos**: consolidación multi-fuente de 11 fuentes (`source_id`) internacionales; ver [Fuentes de datos](/es/datasets/).
3. **Preparación de los datos**: filtrado de atajos, normalización a 224 × 224 px y balanceo de clases.
4. **Modelado**: transfer learning y benchmarking con EfficientNet-B0, EfficientNet-Lite0 y ShuffleNet-V2.
5. **Evaluación**: macro-F1, calibración, evaluación con fuentes retenidas (LOSO) y detección de muestras anómalas (OOD).
6. **Despliegue móvil**: cuantización Int8 post-entrenamiento (TFLite) y empaquetado en aplicación Android nativa offline.

### Arquitectura de Inferencia

```mermaid
flowchart LR
    A["📸 Captura en Campo<br>(Visor con recorte al marco guía)"] --> B["⚙️ Preprocesamiento<br>(224×224 px, normalización)"]
    B --> C["🧠 Inferencia Edge TFLite<br>(EfficientNet-Lite0 Int8)"]
    C --> D{"🛡️ Filtro OOD<br>Mahalanobis"}
    D -- "Fuera de dominio" --> E["⚠️ Alerta Muestra No Reconocida"]
    D -- "Válida" --> F["✅ Diagnóstico + Recomendación<br>(Persistencia SQLite Local)"]
```

### Restricciones Técnicas de Despliegue

<div class="dm-spec-grid">
  <div class="dm-spec-item">
    <div class="dm-spec-label">📱 Dispositivo Objetivo</div>
    <div class="dm-spec-value">Android 8.0+ (ARM64, gama media/baja)</div>
  </div>
  <div class="dm-spec-item">
    <div class="dm-spec-label">📦 Tamaño del Modelo</div>
    <div class="dm-spec-value">3.74 MB (TFLite Int8 post-cuantización)</div>
  </div>
  <div class="dm-spec-item">
    <div class="dm-spec-label">⚡ Latencia de Inferencia</div>
    <div class="dm-spec-value">23 ms de inferencia y 98 ms de preprocesado (medianas); meta de 300 ms en total</div>
  </div>
  <div class="dm-spec-item" style="border-left: 3px solid #2e7d32;">
    <div class="dm-spec-label">📡 Requisito de Red</div>
    <div class="dm-spec-value" style="color: #2e7d32;">Inferencia sin conexión</div>
  </div>
</div>

