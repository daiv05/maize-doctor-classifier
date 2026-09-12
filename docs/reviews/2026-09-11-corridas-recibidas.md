# Dos corridas recibidas el 11 de septiembre de 2026

Fuente preservada: `outputs/outputs-11092026.zip` y su carpeta extraída.
[Evidencia recalculada y hashes](evidence/2026-09-11-corridas-recibidas.json).

## Qué muestran los archivos

| Pipeline main, test de 5,015 imágenes | B0 `20260910_170120` | ShuffleNet `20260910_184521` |
|---|---:|---:|
| Macro-F1 recalculado | 0.948333 | 0.932980 |
| Accuracy recalculada | 0.979661 | 0.973081 |
| Precision macro recalculada | 0.958922 | 0.938761 |
| Recall macro recalculado | 0.939991 | 0.929762 |
| Errores | 102 | 135 |
| Mejor época declarada / historia | 28 | 23 |
| Épocas ejecutadas | 35 | 31 |
| Mejor macro-F1 de validación | 0.960982 | 0.950249 |
| Tiempo sumado de épocas, minutos | 103.03 | 89.41 |

Las matrices coinciden exactamente con `predictions.csv`; accuracy y macro-F1 coinciden
con `summary.json`. Los dos `best.pth` cargan estrictamente en la arquitectura de nueve
clases; `best` y `last` son distintos. Un forward con ceros comprobó la forma `[1, 9]`,
no la exactitud sobre imágenes. El estado de pesos por sí solo no prueba su época.

## Para qué parecen haberse hecho

La evidencia es compatible con comparar dos arquitecturas del pipeline principal y
construir un ensemble ligero. **Es una inferencia de los artefactos**, no un registro de
intención del autor: ambos usan los mismos splits, 224×224, pretrained, batch 64, pérdida
`sqrt_inverse`, label smoothing 0.1, sin sampler y sin CLAHE. No hay evidencia de que
sean corridas segmentadas ni una ablación controlada de segmentación.

El HPO adjunto pertenece a B0, pero no se aplicó íntegramente:

| Parámetro | Mejor trial de B0 | Ambas corridas |
|---|---:|---:|
| learning_rate | 0.0004548490396244796 | 0.0004548 |
| weight_decay | 0.000015731194601419433 | 0.0001 |
| warmup_epochs | 2 | 3 |
| label_smoothing | 0.10012457623772764 | 0.1 |

LR y smoothing parecen redondeos; weight decay y warmup son cambios reales. El JSON
de HPO no acredita una optimización propia de ShuffleNet. El CSV de trials contiene
6 COMPLETE, 7 PRUNED, 1 FAIL y 1 RUNNING; ese RUNNING guardado no prueba un proceso vivo.

El ensemble reporta macro-F1 0.950676 y accuracy 0.979860: +0.002343 de F1 respecto a B0
(0.234 puntos porcentuales). Faltan sus predicciones y vectores completos de probabilidades:
el escalar `pred_prob` de cada integrante no permite reconstruir soft voting. Esa cifra
queda **declarada, no recalculada independientemente**. No basta para justificar dos modelos
en móvil sin evaluar tamaño, memoria y latencia en el dispositivo.

## Comparabilidad y fairness

Train/val/test tienen 23,403 / 5,015 / 5,015 filas y cero intersecciones de rutas entre
particiones del mismo pipeline. Los CSV históricos no tienen hashes de imagen y no
excluyen la sustitución silenciosa que permitía el cargador antiguo. La revisión posterior
del 12/09 verificó bytes actuales y **confirmó una misma fotografía GLS en train y val**
a distinta resolución, además de conflictos de etiqueta en el corpus fuente. Por tanto,
las métricas son recuperables aritméticamente, no una estimación acreditada de independencia.
Ver [dataset y nuevas particiones](2026-09-12-piloto-y-dataset.md).

El holdout de Etapa 2 cruza con main en 3,482 rutas de train, 760 de val y 773 de test.
**No es el mismo holdout.** Este cruce entre experimentos no demuestra fuga dentro de
una corrida; impide comparar 0.903520 de Etapa 2 con 0.948333 de main como mejora controlada.

Laboratorio solo aporta 3 clases (532 imágenes); campo aporta las 9 (4,483). Para B0,
el macro-F1 de laboratorio sobre clases realmente soportadas es 0.886365, no 0.295455
(valor antiguo diluido al promediar nueve clases, seis ausentes). En las tres clases
comparables, campo obtiene 0.907094: razón min/max 0.977148, brecha 0.020729. No equivale
a certificar ausencia de disparidad. Ver [fairness corregido](../../FAIRNESS_REPORT.md).

## Qué se puede reutilizar

| Resultado | Uso permitido / trabajo requerido |
|---|---|
| Predicciones y matrices main/Etapa 2 | Recálculo aritmético, errores y soportes, manteniendo el protocolo histórico |
| B0 y ShuffleNet best.pth | Reevaluación con copia migrada, contrato reconstruido explícito y dataset verificado |
| Ensemble main 0.950676 | Requiere probabilidades completas o reinferencia; no derivable de pred_prob |
| K-fold principal antiguo | Métricas recuperables solo si quedan predicciones; mejor época perdida puede exigir reentrenar |
| CV de Etapa 2 | Implementación independiente; selección de HPO previa hace que no sea CV anidada independiente |
| Cachés de features Etapa 2 | Sin contrato completo: regenerar en otra versión; no añadir hashes actuales como si fueran históricos |
| Oclusiones antiguas | Repetir sensibilidad de clase fija con controles; no reconstruir desde promedios de máximos |
| Móvil histórico Lite0 | Otro flujo, no ensemble Etapa 2 ni aprobación Android de estas corridas |

## Siguiente decisión

Conservar B0 como referencia de desarrollo y ShuffleNet como candidato de menor coste,
sin declarar ganador móvil. El [piloto ya ejecutado](2026-09-12-piloto-y-dataset.md)
no mostró mejora global con el perfil segmentado actual. Revisar máscaras N/P/K/campo,
etiquetas y duplicados pendientes antes de confirmar señales con entrenamiento comparable.
No buscar mejores variantes repitiendo test.
