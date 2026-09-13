# Resultados

Cifras finales del pipeline principal y su análisis. Cada página reporta sus métricas
recomputadas desde predicciones por imagen y declara sus limitaciones.

## Apartados

- [Optimización e hiperparámetros](/es/resultados/optimizacion) — tres configuraciones medidas
  sobre `efficientnet_lite0`; los valores por defecto ganan en prueba.
- [Modelos avanzados y ensamble](/es/resultados/ensamble) — voto blando de tres arquitecturas,
  y en qué fuentes aporta.
- [Evaluación rigurosa y métricas finales](/es/resultados/evaluacion) — validación cruzada 2×2 y
  conjunto de prueba retenido.
- [Análisis de sesgos y ética](/es/resultados/equidad) — ablación con control nulo y
  desagregación por entorno y por procedencia.

## Las cifras del sistema

| | macro-F1 |
| --- | ---: |
| Modelo desplegado, prueba retenida | 0,9468 |
| Ensamble de tres modelos, prueba retenida | 0,9567 |
| Generalización a una fuente no vista | 0,6026 ± 0,1240 |

La diferencia entre la primera y la última es el coste de evaluar sobre datasets que el modelo
no ha visto durante el entrenamiento.

## Cuánto vale una diferencia

La configuración de producción se entrenó con tres semillas sobre la misma partición. La
variación entre ellas es la vara con la que hay que medir cualquier mejora reportada:

| semilla | épocas | mejor época | test macro-F1 |
| ---: | ---: | ---: | ---: |
| 42 (desplegada) | 43 | 35 | **0,9468** |
| 1 | 60 | 53 | 0,9448 |
| 2 | 32 | 24 | 0,9375 |

**Media 0,9430, desviación estándar 0,0049, IC 95 % ±0,0122.**

Dos consecuencias, y la segunda incomoda:

1. **El 0,9468 del modelo desplegado es el mejor de las tres semillas, no el valor central.** Es
   legítimo reportarlo como lo que ese modelo concreto puntúa —está medido y verificado—, pero no
   como lo que cabe esperar de esta configuración, que es 0,9430 ± 0,0122.
2. **Casi ninguna de las mejoras reportadas en el proyecto supera el ruido entre semillas.**

![Diferencias reportadas medidas en sigmas](/resultados/diferencias_en_sigmas.png)

| diferencia | magnitud | en σ | lectura |
| --- | ---: | ---: | --- |
| Protocolo: estratificada → agrupada | −0,3286 | **66,8 σ** | abrumador |
| hp de b0 → lite0 (`lr`+`batch`) | −0,0125 | 2,5 σ | probablemente real |
| hp de b0 → shufflenet | +0,0093 | 1,9 σ | marginal |
| Ensamble vs mejor individual | +0,0084 | 1,7 σ | marginal |
| hp de b0 → b0 | +0,0057 | 1,2 σ | **dentro del ruido** |

El efecto del protocolo de partición es **veintisiete veces mayor** que el mayor de los efectos
de configuración. Todo el esfuerzo de optimización y ensamblado del proyecto se movió dentro de
una franja de menos de tres desviaciones estándar, mientras que la decisión de cómo partir los
datos —que nadie había cuestionado— vale sesenta y siete.

## Paquete auditable

Los artefactos de cada medición están en [evidencia](/es/resultados/evidencia/), junto a un
manifiesto de las ocho corridas archivadas en el Volume `corn-outputs` de Modal. Cada cifra
publicada en esta sección se recomputa desde predicciones por imagen, y el manifiesto registra
desde qué fichero.
