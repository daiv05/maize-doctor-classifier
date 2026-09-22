# Plantillas de tablas para tesis

Estas plantillas se llenan desde CSV/JSON contractuales. `PENDIENTE — <artefacto>` no es un valor y no debe reemplazarse por una estimación.

## Comparación de modelos

| Modelo | Protocolo | Run ID | Parámetros efectivos | Best epoch | Val Macro-F1 | Test Macro-F1 | Accuracy | Artefacto |
|---|---|---|---|---:|---:|---:|---:|---|
| `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE — summary.json` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` |

## Baseline frente a configuración afinada

| Configuración | Study/trial | Split lock | Seeds | Macro-F1 media ± SD | Accuracy media ± SD | Δ Macro-F1 | Artefactos |
|---|---|---|---:|---:|---:|---:|---|
| Baseline | N/A | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | referencia | `PENDIENTE` |
| Tuned | `PENDIENTE — best_params.json` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` |

## Resultado por clase

| Clase | Soporte | Precision | Recall | F1 | Run/protocolo |
|---|---:|---:|---:|---:|---|
| `PENDIENTE — classification_report.csv` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` |

## Resultado por fuente y ambiente

| Dimensión | Subgrupo | n | Clases con soporte | Accuracy | Macro-F1 evaluable | Artefacto |
|---|---|---:|---:|---:|---:|---|
| `source_id` / `environment` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE — test_by_*.csv` |

## Validación cruzada

| Estrategia | K | Unidad aislada | Configuración | Macro-F1 media | SD | IC (método) | Artefacto por pliegue |
|---|---:|---|---|---:|---:|---|---|
| Estratificada / source-grouped | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` |

## LOSO

| Fuente retenida | n test | Clases evaluables | Macro-F1 | Accuracy | Seed/run | Artefacto |
|---|---:|---:|---:|---:|---|---|
| `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` |

## Ablación

| Intervención | Control | Variable modificada | Variables fijas | Macro-F1 | Δ | Seeds | Artefacto |
|---|---|---|---|---:|---:|---:|---|
| `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` | `PENDIENTE` |

## Esqueleto LaTeX compacto

```latex
\begin{table}[ht]
\centering
\caption{PENDIENTE: indicar protocolo, materialización y unidad de variación.}
\label{tab:pendiente}
\begin{tabular}{lrrrr}
\hline
Configuración & $n$ & Macro-F1 & Accuracy & Artefacto \\
\hline
PENDIENTE & -- & -- & -- & PENDIENTE \\
\hline
\end{tabular}
\end{table}
```

Toda tabla final debe declarar protocolo, corpus/split, unidad de repetición y artefacto fuente. Un intervalo no se publica sin indicar cómo se calculó.
