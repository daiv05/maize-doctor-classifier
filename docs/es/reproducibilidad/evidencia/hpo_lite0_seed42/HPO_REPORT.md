# HPO formal EfficientNet-Lite0 — resultado

## Study

study_name: `efficientnet_lite0_seed42_hpo_v1`  
status: COMPLETADO  
trials_requested: 25  
complete: 8  
pruned: 15  
failed: 2  
sampler: TPESampler (seed 42, multivariate=true)  
pruner: MedianPruner, {'interval_steps': 1, 'n_startup_trials': 5, 'n_warmup_steps': 8}  
seed: 42  
objective: best_validation_macro_f1  
test_used_during_hpo: NO  
selection_timestamp: 2026-09-24T02:08:49.348749+00:00  
final_test_timestamp: 2026-09-24T02:11:14.833376+00:00

## Baseline y mejor trial

Baseline `20260921_204608`: validation_macro_f1 = 0.9560862657056215.  
Trial ganador: 0; validation_macro_f1 = 0.9572922254288656.  
Mejora absoluta = +0.001205960; puntos porcentuales = +0.120596.  
best_epoch: 44.0.

| parameter | value |
| --- | --- |
| batch_size | 16 |
| clahe | False |
| class_weights | none |
| epochs | 60 |
| label_smoothing | 0.075 |
| learning_rate | 8.468008575248323e-05 |
| warmup_epochs | 1 |
| weight_decay | 0.005669849511478858 |

Optimizer AdamW y scheduler cosine fijos; dropout de fábrica no buscado.
Son los mejores hiperparámetros encontrados dentro del espacio y presupuesto evaluados.

## Top 10

Solo hay ocho trials COMPLETE; por eso se muestran ocho candidatos elegibles,
sin completar la tabla con trials PRUNED o FAIL.

| rank | trial | val_macro_f1 | learning_rate | weight_decay | label_smoothing | batch_size | class_weights | warmup_epochs | best_epoch | duration_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0.9572922254288656 | 8.468008575248323e-05 | 0.005669849511478858 | 0.075 | 16 | none | 1 | 44 | 2890.575076439 |
| 2 | 10 | 0.9542201116449669 | 8.147857317658701e-05 | 0.00014505381415853765 | 0.05 | 16 | sqrt_inverse | 1 | 27 | 2111.1400961189993 |
| 3 | 6 | 0.9536286932679471 | 0.0001971844222061614 | 1.3731092468240299e-05 | 0.0 | 32 | sqrt_inverse | 1 | 50 | 3105.26846976 |
| 4 | 1 | 0.9482177031656066 | 5.670807781371427e-05 | 4.205156450913872e-05 | 0.025 | 64 | none | 1 | 49 | 3051.118304968 |
| 5 | 21 | 0.9436068189155551 | 0.00017784563227803658 | 0.0010114339762725486 | 0.05 | 64 | sqrt_inverse | 2 | 11 | 1014.4474399319988 |
| 6 | 5 | 0.9422869982777625 | 0.0013740794394697655 | 0.00013076473382928543 | 0.0 | 16 | inverse | 3 | 52 | 3510.2720750430003 |
| 7 | 17 | 0.9387529385273781 | 0.0004426482453919492 | 0.007058919813611231 | 0.075 | 32 | inverse | 2 | 12 | 1058.5778455419986 |
| 8 | 3 | 0.9359111400035606 | 0.0002260828676373493 | 8.399864445957497e-07 | 0.15 | 16 | none | 2 | 12 | 1125.946786006 |

### Diagnósticos de validation

| trial | best_train_macro_f1 | train_macro_f1_at_best_val | train_val_gap | best_train_val_gap | validation_ece | validation_mean_confidence_correct | validation_mean_confidence_errors | nitrogen_deficiency_f1 | phosphorus_deficiency_f1 | potassium_deficiency_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9974215723859532 | 0.9947451419146387 | 0.03745291648577309 | 0.040129346957087586 | 0.05792522954665005 | 0.9269231716561327 | 0.7685920956730843 | 0.904 | 0.967032967032967 | 0.8585858585858586 |
| 10 | 0.9926592279771891 | 0.9846197397630088 | 0.030399628118041866 | 0.03843911633222219 | 0.0714544058202699 | 0.9128629732907402 | 0.7649499494921077 | 0.9112903225806451 | 0.9597069597069597 | 0.8663101604278075 |
| 6 | 0.9997434216381349 | 0.9982430200761107 | 0.044614326808163574 | 0.04611472837018782 | 0.015618175410824988 | 0.9973914794009918 | 0.9156438419313142 | 0.8861788617886179 | 0.9602888086642599 | 0.8571428571428571 |
| 1 | 0.9989398858766765 | 0.9962611126723306 | 0.04804340950672403 | 0.050722182711069874 | 0.0203903746298219 | 0.9609361016621105 | 0.7099684846401214 | 0.8818897637795275 | 0.9454545454545454 | 0.8666666666666667 |
| 21 | 0.9885896448148167 | 0.9735545250461889 | 0.0299477061306338 | 0.044982825899261614 | 0.07742364497466726 | 0.9024517537323904 | 0.6498943528643361 | 0.864 | 0.9571428571428572 | 0.8351648351648352 |
| 5 | 0.9932632495565519 | 0.9894904532173473 | 0.04720345493958478 | 0.05097625127878935 | 0.017314335223588953 | 0.9949619680844161 | 0.8812624504411124 | 0.856 | 0.9436619718309859 | 0.8125 |
| 17 | 0.9688100555836711 | 0.9514834611544697 | 0.012730522627091623 | 0.030057117056292992 | 0.21022297948967378 | 0.7662361078545427 | 0.6000652660576391 | 0.8492063492063492 | 0.9257950530035336 | 0.8387096774193549 |
| 3 | 0.9730262969116039 | 0.9549330292958265 | 0.019021889292265892 | 0.037115156908043345 | 0.1232277279088922 | 0.8526750033606663 | 0.6414904442562419 | 0.8816326530612245 | 0.948905109489051 | 0.7955801104972375 |

Train se mide con augmentation y modo training; los gaps son diagnósticos, no otro objective.
La calibración de validation y N/P/K no intervinieron en la selección contractual.

### Región de mejores trials

| parameter | top10_min | top10_max |
| --- | --- | --- |
| learning_rate | 5.670807781371427e-05 | 0.0013740794394697655 |
| weight_decay | 8.399864445957497e-07 | 0.007058919813611231 |
| label_smoothing | 0.0 | 0.15 |
| batch_size | 16.0 | 64.0 |

Estos rangos describen los trials mejor posicionados. No demuestran causalidad ni
estabilidad entre semillas; esa validación corresponde a una fase posterior.

## Convergencia

| corte | best_validation |
| --- | --- |
| best_at_10 | 0.9572922254288656 |
| best_at_20 | 0.9572922254288656 |
| best_at_25 | 0.9572922254288656 |

Última mejora: trial 0 (índice 0-based).
No se ejecuta un intento 26: los IDs 0–24 agotan el presupuesto revisado.

## Hyperparameter importance

| parameter | importance |
| --- | --- |
| warmup_epochs | 0.5230223819484335 |
| class_weights | 0.15078756519290984 |
| label_smoothing | 0.10966500603116726 |
| learning_rate | 0.10323305886347904 |
| batch_size | 0.086419501737774 |
| weight_decay | 0.026872486226236312 |

Importancia descriptiva del estudio; no causal. fANOVA usa seed 42.

## Evaluación final del ganador sobre test

accuracy: 0.9752741774675973  
macro_f1: 0.9431250727951999  
ECE: 0.060041621811725045  
evaluation_count: 1

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| common_rust | 0.991044776119403 | 0.9793510324483776 | 0.9851632047477745 | 339.0 |
| fall_armyworm | 0.9756427604871448 | 0.9903846153846154 | 0.9829584185412407 | 728.0 |
| gray_leaf_spot | 0.9300699300699301 | 0.9204152249134948 | 0.9252173913043479 | 289.0 |
| healthy | 0.9878603945371776 | 0.9931350114416476 | 0.9904906808672499 | 1311.0 |
| lethal_necrosis | 0.996875 | 0.9937694704049844 | 0.9953198127925117 | 963.0 |
| nitrogen_deficiency | 0.8846153846153846 | 0.905511811023622 | 0.8949416342412452 | 127.0 |
| northern_corn_leaf_blight | 0.9764474975466143 | 0.9707317073170731 | 0.9735812133072407 | 1025.0 |
| phosphorus_deficiency | 0.984251968503937 | 0.8928571428571429 | 0.9363295880149812 | 140.0 |
| potassium_deficiency | 0.7722772277227723 | 0.8387096774193549 | 0.8041237113402062 | 93.0 |

## N/P/K en test

| class | f1-score | support |
| --- | --- | --- |
| nitrogen_deficiency | 0.8949416342412452 | 127.0 |
| phosphorus_deficiency | 0.9363295880149812 | 140.0 |
| potassium_deficiency | 0.8041237113402062 | 93.0 |

Los desgloses por fuente/entorno y N/P/K agrupado están en `final_test/`.
El checkpoint evaluado corresponde a la mejor época del trial original; no se reentrenó.

## Artefactos y reproducibilidad

Raíz: `efficientnet_lite0_seed42_hpo_v1` en `corn-outputs:/hpo/efficientnet_lite0/`.
Incluye `study.db`, `trials.csv`, `study_summary.json`, `best_hyperparameters.json`,
`HPO_SELECTION_LOCK.json`, `trials/trial_000/best.pth`, `figures/`,
`final_test/`, `FINAL_TEST_COMPLETE.json` y `source_code.zip`.

| artefacto | SHA256 |
| --- | --- |
| manifest.lock.json | 0db3ff3ecd3b7674df9fb5e6c207239db92c3690d916a6fd5a9dd650dad8afe8 |
| master_manifest.csv | 64513d316a850ff1ca66c441f86875d62f829c84c0c4e04ec37f131df84a9163 |
| test.csv | 08c81aeec5a57e04416edf2c94f997c422bfcada357c6a3434e73aa9f771f724 |
| train.csv | 231048178f3450bf84925f8d19eb5a672669ee9f2658a23fe87d88d6e9949434 |
| val.csv | 6f37710ebd797e470bc918fec6bf9502f0635e8220542336e88fe5c13b5ae6a5 |
| trials/trial_000/best.pth | ae39a1c4a8b757021ec2c34a856d6fb484bf801a6adc3bfb71860d58e9d81471 |
| best_hyperparameters.json | b79c368bf2105646014732c1fe26ee6e942ace486349630782eb074eb1cf194c |
| study.db | 5bc4a3559c5d238bef16a626953f7ab91e93526c4af20e302c044bc4a6d6c0f5 |
| trials.csv | f11eb2a32c0addd7934dbbf0faf8df7f16a7ed2a5ded5e83bfd2c3e9f6bc8206 |
| study_summary.json | 5dce99f086b61c5efd7488714bd8fbf1afc03aa1bd953891e53aecc819333eec |
| HPO_SELECTION_LOCK.json | 8cf49a675cfa7d2b59f7f664c841aceba8232684000b8212561142a4310c86ca |
| source_code.zip | 4532c748ef7097a4d9524ca6156df4cb946fd57cae1b99ead746b545b7e8fa35 |

| component | version |
| --- | --- |
| cuda_available | True |
| cuda_runtime | 12.6 |
| executable | /usr/local/bin/python |
| gpu | NVIDIA A10G |
| optuna | 4.9.0 |
| platform | Linux-4.19.0-gvisor-x86_64-with-glibc2.36 |
| python | 3.11.12 |
| pytorch | 2.12.1+cu126 |
| timm | 1.0.29 |
| torchvision | 0.27.1+cu126 |

## Veredicto

¿Optuna superó el baseline en Validation Macro-F1? Sí  
¿Test fue utilizado durante selección? No  
¿Se completó el presupuesto de 25 trials? Sí  
¿Study puede reanudarse? Sí; presupuesto agotado, no admite más trials en esta fase.  
¿Existe evidencia suficiente para reproducir la búsqueda? Sí, con el corpus congelado y entorno registrado.

El resultado es un HPO winner. No acredita producción ni entrenamiento formal,
multi-seed, CV, LOSO, ensemble o temperature scaling.

## Revisión documental posterior al cierre — 2026-09-24

El aumento en validation Macro-F1 no se trasladó a test: baseline 0.948002144
frente a ganador 0.943125073 (−0.487707 puntos porcentuales). Es una comparación
descriptiva posterior al lock; no modifica el ganador ni reabre el estudio.
[Comparación, evidencia del baseline y facturación](../../../tesis/HPO_BASELINE_COMPARISON.md).
