# Metodología multi-seed

**Estado: PENDIENTE para la configuración elegida por el HPO nuevo.** Existe un experimento histórico de tres semillas sobre otra etapa; se conserva como evidencia y no fija el tamaño del estudio futuro.

## Diseño

```text
mejor configuración HPO
  → conjunto de seeds decidido antes de entrenar
  → mismo split y presupuesto
  → un run contractual por seed
  → media, desviación estándar y valores individuales
  → intervalo de confianza solo si el tamaño/método se declara
```

Se mantendrán constantes el hash del split, arquitectura, hiperparámetros, preprocessing, criterio de selección y techo de épocas. Cambiarán únicamente las fuentes de aleatoriedad declaradas: inicialización, orden de lotes, workers y transforms estocásticos.

## Reporte mínimo

| Seed | Run ID | Best epoch | Val Macro-F1 | Test Macro-F1 | Test accuracy | Checkpoint SHA-256 |
|---:|---|---:|---:|---:|---:|---|
| `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` |

La cantidad de seeds no se fija aquí: debe aprobarse antes de la ejecución y quedar registrada en el decision log.
