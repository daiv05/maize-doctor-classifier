# Comparación multi-seed — validation exclusivamente

Completadas: 2/10. Test NO evaluado.

| Configuración | Seed | Estado | Macro-F1 | Accuracy | N F1 | P F1 | K F1 | ECE |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 42 | COMPLETE | 0.956086 | 0.980654 | 0.898438 | 0.967273 | 0.857143 | 0.139844 |
| hpo_trial_0 | 42 | COMPLETE | 0.957292 | 0.980056 | 0.904000 | 0.967033 | 0.858586 | 0.057925 |
| baseline | 123 | PENDING | — | — | — | — | — | — |
| hpo_trial_0 | 123 | PENDING | — | — | — | — | — | — |
| baseline | 2026 | PENDING | — | — | — | — | — | — |
| hpo_trial_0 | 2026 | PENDING | — | — | — | — | — | — |
| baseline | 3407 | PENDING | — | — | — | — | — | — |
| hpo_trial_0 | 3407 | PENDING | — | — | — | — | — | — |
| baseline | 7777 | PENDING | — | — | — | — | — | — |
| hpo_trial_0 | 7777 | PENDING | — | — | — | — | — | — |

## Estadísticos descriptivos (SD muestral)

| Configuración | Métrica | n | Media | SD | Mediana | Min | Max | IC95% t exploratorio |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | macro_f1 | 1 | 0.956086 | — | 0.956086 | 0.956086 | 0.956086 | — |
| baseline | accuracy | 1 | 0.980654 | — | 0.980654 | 0.980654 | 0.980654 | — |
| baseline | nitrogen_f1 | 1 | 0.898438 | — | 0.898438 | 0.898438 | 0.898438 | — |
| baseline | phosphorus_f1 | 1 | 0.967273 | — | 0.967273 | 0.967273 | 0.967273 | — |
| baseline | potassium_f1 | 1 | 0.857143 | — | 0.857143 | 0.857143 | 0.857143 | — |
| baseline | ece | 1 | 0.139844 | — | 0.139844 | 0.139844 | 0.139844 | — |
| hpo_trial_0 | macro_f1 | 1 | 0.957292 | — | 0.957292 | 0.957292 | 0.957292 | — |
| hpo_trial_0 | accuracy | 1 | 0.980056 | — | 0.980056 | 0.980056 | 0.980056 | — |
| hpo_trial_0 | nitrogen_f1 | 1 | 0.904000 | — | 0.904000 | 0.904000 | 0.904000 | — |
| hpo_trial_0 | phosphorus_f1 | 1 | 0.967033 | — | 0.967033 | 0.967033 | 0.967033 | — |
| hpo_trial_0 | potassium_f1 | 1 | 0.858586 | — | 0.858586 | 0.858586 | 0.858586 | — |
| hpo_trial_0 | ece | 1 | 0.057925 | — | 0.057925 | 0.057925 | 0.057925 | — |

## Comparación pareada

| Seed | Baseline | HPO | Delta HPO − baseline |
|---:|---:|---:|---:|
| 42 | 0.956086 | 0.957292 | 0.001206 |

Delta medio: 0.001206; SD: —. HPO gana 1; baseline gana 0; empates 0.

## Interpretación y límites

Experimento incompleto: no hay evidencia suficiente para preferir una configuración. Los pendientes no son ceros y no se sustituyen por las runs históricas.

Cinco semillas sobre un split fijo no equivalen a cross-validation, LOSO ni generalización entre dominios. IC t exploratorio bajo independencia aproximada entre seeds y normalidad; con n=5 es frágil. No se realizan pruebas de significancia.
La configuración HPO fue seleccionada sobre esta misma validation: persiste sesgo de selección. Cambiar seed no crea un conjunto de validación independiente.
Siguiente paso: completar y revisar esta cohorte; ninguna fase posterior se ejecuta.

Protocolo SHA-256: `bb7a509bd1418056614519ecdd45d9646278ba0881ee3fcea85ae64c01cdd7a9`.
