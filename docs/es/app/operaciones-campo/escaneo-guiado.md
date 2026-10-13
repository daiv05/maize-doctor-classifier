---
title: Escaneo Guiado en el Surco
description: Protocolo de encuadre foliar, marco guía inteligente, detección fuera de dominio (OOD) e importación de galería.
---

# Escaneo Guiado en el Surco

La calidad del diagnóstico que emite una red neuronal convolucional depende directamente de la calidad de la imagen suministrada. Para evitar errores silenciosos en campo, DoctorMaiz incluye un **marco guía con silueta foliar** y un **filtro inteligente de imágenes no válidas**.

---

## Cómo Encuadrar la Hoja con la Retícula

Al presionar el botón central **"Escanear"** o el botón de acción rápida en la pantalla de inicio, se abrirá la cámara asistida.

<PhoneGallery gap="2rem">
  <PhoneMockup 
    src="/app/pixel4_camera.png" 
    caption="Cámara Asistida" 
    subtitle="Retícula foliar con recorte automático" 
  />
  <PhoneMockup 
    src="/app/camara-marco-guia.png" 
    caption="Encuadre Óptimo" 
    subtitle="La lámina vegetal ocupa el recuadro" 
  />
</PhoneGallery>

### Las 4 Reglas de Oro para una Toma Impecable

1. **Distancia de 20 a 30 centímetros:** 
   - No tomes la foto desde la altura de tus ojos con la planta completa en el suelo.
   - Acércate lo suficiente para que la lámina de la hoja llene el 80 % del marco verde central.
2. **Alineación con la Nervadura Central:**
   - Orienta el teléfono en sentido longitudinal a la nervadura de la hoja de maíz. Esto permite que el modelo analice tanto el tejido internerval como los bordes.
3. **Cuidado con la Iluminación Directa:**
   - Evita el sol cenital del mediodía directo sobre la cutícula foliar (produce reflejos blancos que ciegan al modelo).
   - Usa la sombra de tu cuerpo de manera uniforme o evalúa en las primeras horas de la mañana o atardecer.
4. **Sujeción por el Peciolo o Punta:**
   - Sostén la hoja suavemente por el borde sin tapar con los dedos las lesiones o pústulas que deseas analizar.

---

## El Detector Fuera de Dominio (OOD): Salvaguarda contra Fotos Falsas

Un problema crítico en las aplicaciones agrícolas convencionales es el **"sesgo de clasificación forzada"**: si le presentas al modelo una foto de un zapato, una maleza o el cielo, el algoritmo intentará forzar una de sus clases y dirá con 90 % de confianza que el zapato tiene *Roya Común*.

DoctorMaiz previene este peligro mediante un **módulo detector OOD (*Out-Of-Distribution*) basado en Distancia de Mahalanobis**:

<PhoneMockup 
  src="/app/resultado-no-reconocida.png" 
  caption="Salvaguarda OOD en Acción" 
  subtitle="Alerta honesta de imagen no reconocida" 
  maxWidth="280px"
  center
/>

### ¿Cómo Funciona la Detección?
1. La red neuronal extrae un vector de características (*embeddings*) en su penúltima capa densa.
2. El motor compara matemáticamente este vector con el centroide y la matriz de covarianza de las fotos reales de maíz usadas en entrenamiento.
3. Si la distancia calculada excede el **umbral estadístico calibrado**, la aplicación **rechaza la clasificación** y muestra la etiqueta:
   > **"No Reconocida"**  
   > *La imagen no corresponde a una hoja de maíz analizable o el encuadre no fue el adecuado.*

::: tip VALOR AGRONÓMICO DE LA DUDA
En sanidad vegetal, una app que tiene el coraje de decir *"no sé qué es esto, por favor repite la foto con mejor enfoque"* es infinitamente más confiable que una que inventa tratamientos químicos ante fotos borrosas.
:::

---

## Uso de Fotos de la Galería y Metadatos EXIF

Además de la cámara en vivo, puedes evaluar fotografías previamente capturadas tocando el icono de **Galería**:

- **Fotos tomadas en la mañana sin batería:** Puedes capturar fotos con la cámara nativa del teléfono y luego cargarlas en lote en la app cuando cargues el dispositivo.
- **Extracción de Coordenadas EXIF:** Si tu cámara nativa tenía activada la geolocalización, DoctorMaiz **extrae automáticamente las coordenadas GPS originales** contenidas en los metadatos EXIF de la imagen. El escaneo aparecerá ubicado en el punto exacto de la parcela donde fue tomada la foto original, no donde estás parado al importarla.
