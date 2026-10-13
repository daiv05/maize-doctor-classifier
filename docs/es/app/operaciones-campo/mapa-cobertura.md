---
title: Mapa de Cobertura y Cartografía del Lote
description: Geolocalización de escaneos, código de colores de pines, vista satelital y detección de focos infecciosos en el lote.
---

# Mapa de Cobertura y Cartografía del Lote

Uno de los mayores avances de DoctorMaiz respecto a los métodos tradicionales de papel y lápiz es la capacidad de **mapear espacialmente la sanidad del cultivo**. 

Cada vez que capturas una hoja, la aplicación asocia silenciosamente la coordenada GPS de alta precisión, construyendo un mapa de calor que permite identificar por dónde está entrando la plaga a la finca.

---

<PhoneMockup 
  src="/app/map_screen.png" 
  caption="Cartografía Satelital del Campo" 
  subtitle="Pines interactivos codificados por severidad sanitaria" 
  maxWidth="300px"
  center
/>

---

## Código de Colores de los Marcadores

Los marcadores que aparecen sobre el mapa siguen un código de colores agronómico intuitivo:

| Color del Pin | Condición Sanitaria | Significado en el Campo |
|---|:---:|---|
| 🟢 **Verde** | **Planta Sana** | Hoja limpia, vigorosa y sin síntomas patogénicos visibles. |
| 🟡 **Ámbar / Naranja** | **Severidad Leve o Nutricional** | Presencia de deficiencias (Nitrógeno, Potasio) o manchas foliares aisladas en etapa temprana. |
| 🔴 **Rojo Carmesí** | **Severidad Crítica** | Ataque avanzado de Roya Común, Tizón Foliar o presencia de Gusano Cogollero activo. |

::: tip INTERACCIÓN CON EL MARCADOR
Al tocar cualquier pin en el mapa, se despliega una tarjeta flotante (*callout*) que muestra el nombre del diagnóstico, el porcentaje de certeza y el enlace directo para abrir el detalle completo del escaneo.
:::

---

## Vista Satelital vs. Mapa de Terreno

En la esquina superior derecha del mapa encontrarás un selector de capas:

1. **Vista Satelital (Por defecto en campo):** Permite reconocer los linderos reales de tu milpa, acequias, árboles de referencia y caminos vecinales para saber exactamente en qué hilera se tomó la muestra.
2. **Vista de Terreno / Estándar:** Útil cuando la señal de datos es lenta o cuando se requiere verificar nombres de cantones, caseríos o vías de comunicación.

---

## Aplicaciones Prácticas: El Manejo por Focos

En la agricultura convencional, cuando el productor ve roya en una orilla, suele fumigar toda la parcela de 5 manzanas por temor a perder la cosecha. Esto causa:
- Gasto innecesario de dinero en fungicidas.
- Mayor estrés químico en plantas sanas.
- Aceleración de la resistencia genética del patógeno.

### La Estrategia de "Cordón Sanitario" con DoctorMaiz
Con el mapa de DoctorMaiz, el productor puede:
1. Identificar que los pines rojos se concentran en el **extremo este** (de donde sopla el viento dominante).
2. Trazar un **cordón de contención**: aplicar fungicida sistémico únicamente en las primeras 6 hileras del foco y aplicar preventivos biológicos más económicos en el resto del lote.
3. Repetir el recorrido a los 7 días para verificar si los nuevos pines se mantienen en verde o amarillo.
