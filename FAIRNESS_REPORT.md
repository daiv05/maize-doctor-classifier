# Evaluación por entorno y sensibilidad espacial

Revisión: 11 de septiembre de 2026. Este documento sustituye las conclusiones de
“Clever Hans confirmado” y de aprobación por la regla del 80 %. Se conserva el
[informe original como antecedente](docs/reviews/historical/2026-09-11-fairness-original.md).

## Evidencia disponible

Se recibieron las corridas del pipeline principal `outputs/outputs-11092026`:
B0 `20260910_170120` y ShuffleNet `20260910_184521`. El
[auditor reproducible](scripts/checks/audit_saved_runs.py) recalculó métricas desde
sus CSV y comprobó sus matrices. No ejecutó inferencia sobre imágenes ni abrió de
nuevo test. Los checkpoints cargaron estrictamente; eso demuestra compatibilidad
estructural, no procedencia completa de los datos.

| Experimento / partición | Macro-F1 | Accuracy | Estado |
|---|---:|---:|---|
| Principal B0 / test, 5 015 imágenes | 0,948333 | 0,979661 | Recalculado desde predicciones |
| Principal ShuffleNet / mismo test | 0,932980 | 0,973081 | Recalculado desde predicciones |
| Principal ensemble de dos / test | 0,950676 | 0,979860 | Declarado; faltan probabilidades completas y predicciones del ensemble |
| Etapa 2, cuatro probes / holdout | 0,903520 | 0,953739 | Recalculado; protocolo y holdout distintos |

Solo 773 imágenes coinciden entre ambos conjuntos finales. No se comparan estas
filas para atribuir mejoras a una arquitectura o reparación.
[Evidencia con hashes](docs/reviews/evidence/2026-09-11-corridas-recibidas.json).

## Clases comparables y soporte

La cifra histórica B0 de macro-F1 `lab=0,295455` promediaba nueve clases aunque
solo tres estaban presentes. La corrección no cambia predicciones: cambia el
denominador y deja indefinidos recall/FNR sin soporte.

| B0, predicciones históricas recalculadas | Lab | Real |
|---|---:|---:|
| Muestras totales | 532 | 4 483 |
| Accuracy sobre todo el grupo | 0,941729 | 0,984162 |
| Macro-F1 sobre clases presentes en cada grupo | 0,886365 | 0,929795 |
| Macro-F1 restringido a las tres clases comunes | 0,886365 | 0,907094 |

Las clases comunes son roya, mancha gris y tizón norteño. La comparación restringe
las muestras por etiqueta verdadera y promedia esas mismas tres clases; las
predicciones a otras clases siguen contando como errores. La razón descriptiva de
macro-F1 común es `0,977148`, con diferencia absoluta `0,020729`. No se llama
“disparate impact”, no lleva umbral del 80 % y no certifica ausencia de sesgo.

Cada JSON incluye soporte, intervalos Wilson para accuracy/recall y estados
`insufficient` cuando no hay dos grupos comparables. Los intervalos no corrigen
dependencia entre imágenes de una misma planta, sesión o fuente. Faltan esos
identificadores para una evaluación agrupada completa.

## Qué mide la oclusión

El rectángulo de 20 % a 80 % de cada eje ocupa aproximadamente 36 % del área.
Su complemento ocupa aproximadamente 64 %. Los porcentajes efectivos se calculan
sobre los píxeles reales, por lo que el redondeo depende de la resolución.
Cero en el tensor normalizado representa el color medio de normalización, no negro.

El código actual conserva la clase predicha en la imagen original al medir su
probabilidad después de cada perturbación. Reporta accuracy, cambios de clase,
errores condicionados a aciertos originales, resultados por muestra y un control
rectangular aleatorio de igual área. No compara máximos de clases diferentes como
si fueran la misma confianza.

Estas son pruebas de sensibilidad espacial. Centro no equivale a lesión, periferia
no equivale a fondo y Grad-CAM no localiza síntomas con garantías. Sin máscaras
revisadas no puede inferirse causalidad ni que segmentar mejore el diagnóstico.

Los resultados de oclusión recibidos aún corresponden al algoritmo anterior y no
son recalculables sin imágenes/probabilidades por perturbación. Además, el informe
original decía accuracy central `0,8400` y periférica `0,7000`, mientras el JSON
recibido declara `0,8046` y `0,7176`. Se conservan como antecedentes discrepantes;
ninguna pareja se presenta como resultado de la reparación.

## Segmentación y uso operativo

Las máscaras se clasifican en aceptadas, inciertas y rechazadas, con cobertura,
componentes, ambigüedad y pérdida de máscara por recorte. Los umbrales actuales
son provisionales: falta calibrarlos con desarrollo y revisar la conservación de
lesiones, especialmente en deficiencias. El fallback al original se registra por
imagen; no se evalúa solamente el subconjunto aceptado como si fuera toda la población.

Las métricas no autorizan recomendaciones de agroquímicos, certificación clínica
ni despliegue autónomo. OOD calibrado con train/val tampoco demuestra rechazo de
todas las imágenes no foliares. La validación externa y Android siguen separadas.

## Reproducción

Auditoría de los artefactos recibidos, sin nuevas predicciones:

```bash
python -m scripts.checks.audit_saved_runs \
  --bundle outputs/outputs-11092026 \
  --output outputs/audits/corridas-recibidas-v2.json
```

Para una corrida moderna con contrato completo:

```bash
make fairness MODEL=efficientnet_b0 RUN=<run> SPLITS_DIR=<splits> \
  SPLIT=val RUN_GRADCAM=0 RUN_SHORTCUT=1
```

No se adaptan automáticamente runs históricos. Ver la
[guía de migración y recursos pendientes](docs/reviews/2026-09-11-reparacion-integral.md).
