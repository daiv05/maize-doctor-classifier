# Baseline vs HPO winner

## Cierre verificado — 2026-09-24

Study `efficientnet_lite0_seed42_hpo_v1`, EfficientNet-Lite0, `seed_42`: **25 intentos totales = 8 COMPLETE + 15 PRUNED + 2 FAIL**. Los podados terminaron anticipadamente por el pruner; no son errores. Los fallidos corresponden a los trials 2 y 4 interrumpidos. No se afirman 25 entrenamientos completos.

El ganador fue **trial 0, época 44**, elegido únicamente por su mejor Macro-F1 de validación. Ningún trial posterior lo superó: best@10 = best@20 = best@25.

| Métrica | Baseline `20260921_204608` | HPO trial 0 | Delta HPO − baseline (pp) |
|---|---:|---:|---:|
| Macro-F1 validación | 0.956086266 | 0.957292225 | +0.120596 |
| Macro-F1 test | 0.948002144 | 0.943125073 | −0.487707 |
| Accuracy test | 0.978065803 | 0.975274177 | −0.279163 |

Las métricas se muestran en escala 0–1; los deltas son puntos porcentuales. Fuentes: [summary original del baseline](../reproducibilidad/evidencia/hpo_baseline_summary.json) y [marcador final del HPO](../reproducibilidad/evidencia/hpo_lite0_seed42/FINAL_TEST_COMPLETE.json). La comparación de test es **descriptiva y posterior al cierre**: no se utilizó para reordenar trials ni para cambiar el ganador contractual.

## Interpretación y límites

El HPO mejoró ligeramente la métrica objetivo de validación, **pero no mejoró el resultado de test**. No basta para afirmar mejor generalización, promover el checkpoint a producción ni sustituir automáticamente al baseline. Una semilla y un holdout no permiten afirmar significancia estadística ni estabilidad.

El presupuesto experimental inicial de 60 intentos se redujo a 25 después de observar validación parcial. Es una enmienda posterior al inicio, no una búsqueda de 60 completada ni un presupuesto de 25 prefijado desde el principio. [Protocolo y hashes de la enmienda](../reproducibilidad/evidencia/hpo_lite0_seed42/BUDGET_AMENDMENT.json).

| Clase nutricional | F1 test | Soporte |
|---|---:|---:|
| Nitrógeno | 0.894941634 | 127 |
| Fósforo | 0.936329588 | 140 |
| Potasio | 0.804123711 | 93 |

Potasio sigue siendo la clase más débil del ganador. ECE test = **0.060041622**; no se aplicó temperature scaling. El Brier reportado es binario de acierto, no multiclase. [Desglose completo](../reproducibilidad/evidencia/hpo_lite0_seed42/HPO_REPORT.md).

## Integridad y secuencia temporal

- Selection lock: **2026-09-24 02:08:49 UTC**.
- Test final cerrado: **2026-09-24 02:11:14 UTC**, equivalente al **23 de septiembre, 20:11:14** en El Salvador.
- Una única evaluación final, sin reentrenar el checkpoint ganador ni usar test en selección.
- Verificados: 25 IDs, estados SQLite/CSV, enmienda, splits, código archivado, hash del checkpoint y siete artefactos de test.
- El observador descargó y documentó el cierre a las **02:13:25 UTC**.

Se completaron los 25 intentos previstos; el estudio está cerrado y no corresponde relanzar la búsqueda ni repetir test. La [comparación multi-seed](../experimentos/multiseed.md) quedó en 2 de 10 entrenamientos de validación (seed 42 de cada configuración). Entrenamiento formal, CV, LOSO, ensamble y calibración son fases posteriores, no ejecutadas dentro de este protocolo.
