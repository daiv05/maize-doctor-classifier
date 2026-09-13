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

## Paquete auditable

Los artefactos de cada medición están en [evidencia](/es/resultados/evidencia/), junto a un
manifiesto de las ocho corridas archivadas en el Volume `corn-outputs` de Modal. Cada cifra
publicada en esta sección se recomputa desde predicciones por imagen, y el manifiesto registra
desde qué fichero.
