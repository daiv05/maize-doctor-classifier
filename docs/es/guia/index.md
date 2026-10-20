# Presentándonos

<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.75rem 0 1.25rem 0;">
  <span class="dm-badge">📱 Android 8.0+</span>
  <span class="dm-badge">⚡ Inferencia &lt; 300 ms</span>
  <span class="dm-badge">📡 100 % Offline</span>
  <span class="dm-badge">🎯 9 Clases Fitosanitarias</span>
  <span class="dm-badge">🛡️ Filtro OOD Mahalanobis</span>
</div>

**DoctorMaiz** es una herramienta integral de diagnóstico foliar basada en Deep Learning, diseñada para operar en el borde (*Edge AI*) de forma **100 % offline** en teléfonos Android de gama media y baja en El Salvador.

Esta guía te permitirá conocer el funcionamiento de la herramienta, sus requisitos, el flujo de diagnóstico en tres pasos y cómo interactuar con el modelo tanto desde la aplicación móvil como mediante código.

## Componentes de la Herramienta

La solución se compone de dos herramientas complementarias:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; margin: 1.5rem 0;">

  <a href="/es/app/" class="dm-interactive-card">
    <div>
      <div style="font-size: 1.25rem; font-weight: 700; color: var(--vp-c-brand-1); margin-bottom: 0.45rem;">
        📱 DoctorMaiz App
      </div>
      <div style="margin-bottom: 0.85rem;">
        <span class="dm-badge">React Native</span>
      </div>
      <p style="font-size: 0.88rem; color: var(--vp-c-text-2); line-height: 1.55; margin-bottom: 1rem;">
        Aplicación móvil para pequeños agricultores y técnicos de campo. Incorpora cámara con visor guiado, inferencia local instantánea sin internet, bitácora de escaneos y catálogo fitosanitario.
      </p>
    </div>
    <div style="font-size: 0.88rem; font-weight: 600; color: var(--vp-c-brand-1); padding-top: 0.75rem; border-top: 1px dashed var(--vp-c-divider);">
      Ver Manual de la App <span class="card-arrow">→</span>
    </div>
  </a>

  <a href="/es/deployment/prototipo" class="dm-interactive-card">
    <div>
      <div style="font-size: 1.25rem; font-weight: 700; color: var(--vp-c-brand-1); margin-bottom: 0.45rem;">
        🧠 Modelo Edge (TFLite Int8)
      </div>
      <div style="margin-bottom: 0.85rem;">
        <span class="dm-badge">EfficientNet-Lite0</span>
      </div>
      <p style="font-size: 0.88rem; color: var(--vp-c-text-2); line-height: 1.55; margin-bottom: 1rem;">
        Red neuronal cuantizada a enteros de 8 bits (peso ≤ 14 MB). Procesa imágenes de 224×224 px en menos de 300 ms en la CPU del teléfono con filtro de rechazo anómalo (Mahalanobis OOD).
      </p>
    </div>
    <div style="font-size: 0.88rem; font-weight: 600; color: var(--vp-c-brand-1); padding-top: 0.75rem; border-top: 1px dashed var(--vp-c-divider);">
      Ver Ficha Técnica Edge <span class="card-arrow">→</span>
    </div>
  </a>

</div>

## Requisitos Previos

Antes de comenzar en el surco, verifica los requisitos mínimos recomendados para el funcionamiento óptimo de la aplicación:

<div class="dm-spec-grid">
  <div class="dm-spec-item">
    <div class="dm-spec-label">📱 Sistema Operativo</div>
    <div class="dm-spec-value">Android 8.0 (Oreo / API 26+) o superior</div>
  </div>
  <div class="dm-spec-item">
    <div class="dm-spec-label">⚡ Procesador / CPU</div>
    <div class="dm-spec-value">ARM64 (Snapdragon 4xx/6xx o Helio)</div>
  </div>
  <div class="dm-spec-item">
    <div class="dm-spec-label">🧠 Memoria RAM</div>
    <div class="dm-spec-value">2 GB mínima recomendada</div>
  </div>
  <div class="dm-spec-item">
    <div class="dm-spec-label">📷 Cámara</div>
    <div class="dm-spec-value">Sensor con autoenfoque (≥ 5 MP)</div>
  </div>
  <div class="dm-spec-item">
    <div class="dm-spec-label">💾 Almacenamiento Libre</div>
    <div class="dm-spec-value">50 MB (modelo 14 MB + SQLite local)</div>
  </div>
  <div class="dm-spec-item" style="border-left: 3px solid #2e7d32;">
    <div class="dm-spec-label">📡 Conectividad en Campo</div>
    <div class="dm-spec-value" style="color: #2e7d32;">0 % requerida (100 % offline)</div>
  </div>
</div>

## Inicio Rápido en 3 Pasos

El siguiente flujo ilustra cómo realizar un diagnóstico foliar completo en cuestión de segundos:

```mermaid
flowchart LR
    A["Paso 1: Captura Guiada<br>(Enfoque en marco central)"] --> B["Paso 2: Inferencia Local<br>(TFLite Int8 en &lt; 300 ms)"]
    B --> C["Paso 3: Diagnóstico y OOD<br>(9 Clases o Rechazo)"]
```

<div class="dm-step-box">
  <div class="dm-step-header">
    <span class="dm-step-tag">Paso 1</span>
    <span>Captura Guiada en Campo</span>
  </div>
  <ol style="padding-left: 1.25rem; margin: 0.5rem 0;">
    <li>Abre la aplicación <strong>DoctorMaiz</strong> en tu teléfono móvil y pulsa el botón <strong>«Nuevo Escaneo»</strong>.</li>
    <li>Apunta la cámara hacia la hoja afectada. La pantalla desplegará un <strong>marco guía central</strong>.</li>
    <li><strong>Regla de oro:</strong> Ajusta la distancia para que el tejido foliar cubra la totalidad del marco central.</li>
    <li><em>¿Por qué este diseño?</em> El sistema recorta automáticamente el área exterior antes de clasificar la imagen, eliminando el suelo, las malezas y los fondos engañosos que podrían confundir a la red neuronal (<em>mitigación de atajos visuales</em>).</li>
  </ol>
</div>

<div class="dm-step-box">
  <div class="dm-step-header">
    <span class="dm-step-tag">Paso 2</span>
    <span>Inferencia Local Instantánea</span>
  </div>
  <p style="font-size: 0.9rem; color: var(--vp-c-text-2); margin-bottom: 0.75rem;">
    Una vez tomada la fotografía, el motor de inferencia procesa la imagen directamente en el procesador del teléfono:
  </p>
  <ul style="padding-left: 1.25rem; margin: 0.5rem 0;">
    <li><strong>Redimensionado:</strong> La imagen se ajusta a 224 × 224 píxeles en RGB.</li>
    <li><strong>Normalización:</strong> Se estandarizan los canales cromáticos con los parámetros de referencia de ImageNet (media = [0.485, 0.456, 0.406], desv = [0.229, 0.224, 0.225]).</li>
    <li><strong>Ejecución:</strong> El modelo cuantizado evalúa la muestra en aproximadamente <strong>180 a 280 milisegundos</strong> en la CPU móvil.</li>
  </ul>
</div>

<div class="dm-step-box">
  <div class="dm-step-header">
    <span class="dm-step-tag">Paso 3</span>
    <span>Interpretación del Diagnóstico y Filtro Fuera de Dominio</span>
  </div>
  <p style="font-size: 0.9rem; color: var(--vp-c-text-2); margin-bottom: 0.75rem;">
    El sistema entrega el diagnóstico categorizado en una de las 9 condiciones objetivo:
  </p>

| Categoría | Clase Detectada | Síntoma Principal de Confirmación |
|:---:|---|---|
| <span class="dm-pill-sano">Sano</span> | **Planta sana (`healthy`)** | Lámina foliar verde uniforme, sin pústulas ni lesiones. |
| <span class="dm-pill-enf">Enfermedad</span> | **Roya común (`common_rust`)** | Pústulas pulverulentas de color marrón rojizo en ambas caras. |
| <span class="dm-pill-enf">Enfermedad</span> | **Tizón foliar del norte (`nclb`)** | Lesiones elípticas grandes en forma de cigarro, grisáceas. |
| <span class="dm-pill-enf">Enfermedad</span> | **Mancha gris (`gls`)** | Lesiones rectangulares estrechas delimitadas por nervaduras. |
| <span class="dm-pill-enf">Enfermedad</span> | **Necrosis letal (`lethal_necrosis`)** | Clorosis severa y secado progresivo desde los márgenes. |
| <span class="dm-pill-plaga">Plaga</span> | **Gusano cogollero (`fall_armyworm`)** | Agujeros irregulares con acumulaciones de excremento en el verticilo. |
| <span class="dm-pill-nut">Nutrición</span> | **Deficiencia de nitrógeno (`n`)** | Clorosis en forma de "V" en hojas inferiores (más viejas). |
| <span class="dm-pill-nut">Nutrición</span> | **Deficiencia de fósforo (`p`)** | Coloración púrpura/violácea en bordes y puntas de hojas jóvenes. |
| <span class="dm-pill-nut">Nutrición</span> | **Deficiencia de potasio (`k`)** | Secado y necrosis marginal estricta en hojas inferiores. |

::: tip 🛡️ ¿Qué pasa si se escanea un objeto que no es una hoja de maíz?
DoctorMaiz incorpora un **módulo de detección fuera de distribución (OOD)** basado en distancia de Mahalanobis. Si el usuario escanea un zapato, una pared, tierra o la mano del agricultor, el sistema no forzará un diagnóstico erróneo: la app mostrará un aviso de **«Muestra no reconocida»** y solicitará reencuadrar la hoja de maíz.
:::

</div>

## Arquitectura de Funcionamiento

El siguiente esquema resume el flujo integral de datos desde la captura en el campo salvadoreño hasta el registro local en la base de datos del agricultor:

```mermaid
flowchart TD
    subgraph Campo["1. Dispositivo Móvil en el Surco (Sin Conexión)"]
        A["Agricultor enfoca hoja de maíz"] --> B["Visor de cámara guiado"]
        B --> C["Recorte del 75 % exterior<br>(Eliminación de fondo y malezas)"]
        C --> D["Motor TFLite Int8<br>(EfficientNet-Lite0)"]
        D --> E{"Filtro OOD<br>Mahalanobis"}
        E -- "Fuera de dominio" --> F["Aviso: Reencuadrar hoja"]
        E -- "Válida" --> G["Diagnóstico y Severidad<br>+ Recomendación agronómica"]
        G --> H["Persistencia en SQLite Local<br>(Historial de parcela)"]
    end

    subgraph Nube["2. Conectividad Opcional (Al regresar al hogar o cooperativa)"]
        H -. "Sincronización voluntaria" .-> I["Servicio Central / Backend"]
        I -.-> J["Aporte al Dataset Nacional<br>(Ciencia ciudadana)"]
    end
```

## ¿Qué deseas explorar ahora?

Para profundizar en el aspecto que más te interese, te sugerimos los siguientes destinos:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

  <a href="/es/app/guia-inicio/instalacion" class="dm-interactive-card">
    <div>
      <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem; color: var(--vp-c-text-1);">📱 Manual de la App</div>
      <p style="font-size: 0.85rem; color: var(--vp-c-text-2); margin-bottom: 0.75rem;">Aprende a instalar la aplicación, gestionar parcelas y realizar escaneos guiados.</p>
    </div>
    <div style="font-size: 0.85rem; font-weight: 600; color: var(--vp-c-brand-1);">
      Guía de Instalación <span class="card-arrow">→</span>
    </div>
  </a>

  <a href="/es/app/catalogo-enfermedades/" class="dm-interactive-card">
    <div>
      <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem; color: var(--vp-c-text-1);">🌿 Catálogo de Enfermedades</div>
      <p style="font-size: 0.85rem; color: var(--vp-c-text-2); margin-bottom: 0.75rem;">Consulta la galería visual de patógenos, agentes causales y síntomas foliares.</p>
    </div>
    <div style="font-size: 0.85rem; font-weight: 600; color: var(--vp-c-brand-1);">
      Ver Catálogo <span class="card-arrow">→</span>
    </div>
  </a>

  <a href="/es/pipeline/entrenamiento" class="dm-interactive-card">
    <div>
      <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem; color: var(--vp-c-text-1);">⚙️ Pipelines de Entrenamiento</div>
      <p style="font-size: 0.85rem; color: var(--vp-c-text-2); margin-bottom: 0.75rem;">Inspecciona el código de preprocesamiento, contratos de runs y optimización Optuna.</p>
    </div>
    <div style="font-size: 0.85rem; font-weight: 600; color: var(--vp-c-brand-1);">
      Ver Pipelines <span class="card-arrow">→</span>
    </div>
  </a>

  <a href="/es/auditoria/" class="dm-interactive-card">
    <div>
      <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem; color: var(--vp-c-text-1);">🛡️ Auditoría Metodológica</div>
      <p style="font-size: 0.85rem; color: var(--vp-c-text-2); margin-bottom: 0.75rem;">Examina las pruebas forenses de píxeles, estabilidad multi-semilla y evaluación LOSO.</p>
    </div>
    <div style="font-size: 0.85rem; font-weight: 600; color: var(--vp-c-brand-1);">
      Ver Auditoría 5 Puntos <span class="card-arrow">→</span>
    </div>
  </a>

</div>
