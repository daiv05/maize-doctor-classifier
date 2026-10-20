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
      text: Ver Datasets
      link: /es/datasets/
    - theme: alt
      text: Baselines
      link: /es/baselines/

features:
  - icon: 🎯
    title: "9 Clases Fitosanitarias"
    details: "Detección de 4 patologías fúngicas/virales, 1 plaga (cogollero), 3 deficiencias nutricionales (N, P, K) y control de hoja sana mediante CNN con transfer learning."
  - icon: ⚡
    title: "Edge AI 100 % Offline"
    details: "Entrenamiento en PyTorch y exportación a TensorFlow Lite Int8 (≤ 14 MB), con latencia < 300 ms en procesadores móviles sin requerir internet."
  - icon: 🌾
    title: "Diseñado para el Campo"
    details: "Resistente a fondos ruidosos mediante encuadre guiado y recorte preventivo, evaluado bajo condiciones reales de iluminación en El Salvador."
  - icon: 🛡️
    title: "Auditoría y Validación LOSO"
    details: "Evaluación honesta por origen (Leave-One-Source-Out) y módulo de detección fuera de distribución (Mahalanobis OOD) para evitar falsos diagnósticos."
---

<div class="dm-kpi-banner">
  <div class="dm-kpi-card">
    <div class="dm-kpi-val">33 429</div>
    <div class="dm-kpi-lbl">Muestras Foliares Curadas</div>
  </div>
  <div class="dm-kpi-card">
    <div class="dm-kpi-val">&lt; 300 ms</div>
    <div class="dm-kpi-lbl">Inferencia en CPU Móvil</div>
  </div>
  <div class="dm-kpi-card">
    <div class="dm-kpi-val">≤ 14 MB</div>
    <div class="dm-kpi-lbl">Modelo Cuantizado Int8</div>
  </div>
  <div class="dm-kpi-card">
    <div class="dm-kpi-val">100 %</div>
    <div class="dm-kpi-lbl">Autonomía Offline</div>
  </div>
</div>

## El Proyecto

**Detección de Enfermedades Foliares, Plagas y Deficiencias Nutricionales en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles**

<div class="dm-academic-card">
  <div class="dm-academic-header" style="display: flex; flex-wrap: wrap; gap: 0.85rem; margin-bottom: 1.25rem;">
    <span class="dm-badge">🎓 Universidad de El Salvador</span>
    <span class="dm-badge">🏛️ Facultad de Ingeniería y Arquitectura</span>
    <span class="dm-badge">💻 Escuela de Sistemas Informáticos</span>
  </div>
  <p style="font-size: 0.9rem; color: var(--vp-c-text-2); margin-bottom: 0.85rem;">
    Trabajo desarrollado en el marco del <strong>Curso de Especialización en Machine Learning</strong> (Ciclo I 2026 — Grupo 02), bajo la cátedra del <strong>Ing. Bladimir Díaz Campos</strong>.
  </p>
  <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--vp-c-text-2); margin-top: 1rem;">
    Equipo de Desarrollo e Investigación
  </div>
  <div class="dm-team-grid">
    <div class="dm-team-member">
      <div class="dm-team-avatar">JR</div>
      <div>
        <div style="font-weight: 700; font-size: 0.9rem; color: var(--vp-c-text-1);">Josias Abner Rivas Fuentes</div>
        <div style="font-size: 0.78rem; color: var(--vp-c-text-2); font-family: var(--vp-font-family-mono);">RF20010</div>
      </div>
    </div>
    <div class="dm-team-member">
      <div class="dm-team-avatar">DD</div>
      <div>
        <div style="font-weight: 700; font-size: 0.9rem; color: var(--vp-c-text-1);">David Alejandro Deras Cerros</div>
        <div style="font-size: 0.78rem; color: var(--vp-c-text-2); font-family: var(--vp-font-family-mono);">DC19019</div>
      </div>
    </div>
    <div class="dm-team-member">
      <div class="dm-team-avatar">ER</div>
      <div>
        <div style="font-weight: 700; font-size: 0.9rem; color: var(--vp-c-text-1);">Elmer Edenilson Rosales Molina</div>
        <div style="font-size: 0.78rem; color: var(--vp-c-text-2); font-family: var(--vp-font-family-mono);">RM20001</div>
      </div>
    </div>
  </div>
</div>

### Contexto y Problema

Para entender por qué existe este proyecto conviene partir del peso que tiene la agricultura en El Salvador: representa el **5.6 % del PIB** y es el sustento de más de **2 millones de personas rurales**. El 82.1 % de los productores son pequeños agricultores, muchos operando a nivel de subsistencia. El maíz, su principal cultivo, es vulnerable a enfermedades foliares, plagas y deficiencias nutricionales que, sin detección temprana, pueden destruir hasta el 70 % de la cosecha.

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

> 📊 **Distribución muestral:** Corpus activo con **33 429 muestras elegibles** (3 551 laboratorio + 29 878 campo real) tras 8 exclusiones metodológicas auditadas. Ver [Auditoría Metodológica](/es/auditoria/) y [Pipeline de Datos](/es/metodologia/pipeline-datos).

### Metodología

El proyecto avanza en fases iterativas siguiendo el ciclo estándar **CRISP-DM**:

1. **Comprensión del negocio**: análisis de las condiciones de campo y limitaciones de conectividad rural en El Salvador.
2. **Comprensión de los datos**: consolidación multi-fuente de 8 repositorios internacionales; ver [Recopilación de datasets](/es/datasets/).
3. **Preparación de los datos**: filtrado de atajos, normalización a 224 × 224 px y balanceo de clases.
4. **Modelado**: transfer learning y benchmarking con arquitecturas MobileNet, EfficientNet y ResNet.
5. **Evaluación y Auditoría**: macro-F1, análisis de calibración, partición LOSO por origen y detección de muestras anómalas (OOD).
6. **Despliegue móvil**: cuantización Int8 post-entrenamiento (TFLite) y empaquetado en aplicación Android nativa offline.

### Arquitectura de Inferencia

```mermaid
flowchart LR
    A["📸 Captura en Campo<br>(Visor con recorte 75 %)"] --> B["⚙️ Preprocesamiento<br>(224×224 px, normalización)"]
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
    <div class="dm-spec-label">📦 Peso Máximo Modelo</div>
    <div class="dm-spec-value">≤ 14 MB (TFLite Int8 post-cuantización)</div>
  </div>
  <div class="dm-spec-item">
    <div class="dm-spec-label">⚡ Latencia de Inferencia</div>
    <div class="dm-spec-value">&lt; 300 ms por fotograma en CPU</div>
  </div>
  <div class="dm-spec-item" style="border-left: 3px solid #2e7d32;">
    <div class="dm-spec-label">📡 Requisito de Red</div>
    <div class="dm-spec-value" style="color: #2e7d32;">100 % Autónomo (Cero datos móviles)</div>
  </div>
</div>

