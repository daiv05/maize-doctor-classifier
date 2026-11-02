# Evaluación y auditoría de equidad

La evaluación usa el conjunto de prueba de `seed_42` (5 015 imágenes) una sola vez, con el checkpoint bloqueado por `selection.lock.json`. La métrica primaria es el Macro-F1, que pesa igual a las nueve clases; la accuracy es secundaria. Los resultados están en [Resultados](/es/resultados/) y [Generalización entre fuentes](/es/resultados/generalizacion).

## Qué se calcula

| Salida | Contenido |
|---|---|
| `test_classification_report.csv` | Precision, recall, F1 y soporte por clase |
| `test_confusion_matrix.csv` | Matriz de confusión exacta |
| `test_calibration.json` | ECE, Brier binario de acierto y confianza media por acierto y error |
| `test_by_environment.csv` y `test_by_source.csv` | Desglose por laboratorio y campo y por fuente |
| `test_grouped_metrics.json` | Métrica secundaria con nitrógeno, fósforo y potasio agrupados |
| `equidad_metricas.json` | Disparidad por entorno, ablación de la periferia y control nulo |

## Equidad entre entornos

Solo 3 de las 9 clases (roya común, mancha gris y tizón del norte) tienen fotos de laboratorio. El Macro-F1 de laboratorio promedia tres clases y el de campo nueve, por lo que el cociente entre ambos no mide disparidad. La comparación válida restringe ambos subgrupos a las tres clases compartidas y reporta el cociente de impacto dispar (DIR) sobre accuracy y Macro-F1.

## Prueba de ablación de la periferia

Se enmascara un rectángulo fijo (del 20 % al 80 % de cada eje). La condición "centro tapado" deja visible el 63.7 % del área y "periferia tapada" el 36.3 %. Cada condición se compara contra un control nulo (responder siempre la clase más frecuente), porque un clasificador de nueve clases desbalanceadas no parte de cero aciertos. Las dos condiciones no tienen la misma área, así que no se comparan entre sí. Resultados en [Generalización entre fuentes](/es/resultados/generalizacion).

## Atribución

Grad-CAM y SHAP se miden con el exceso de atribución: la fracción de atribución positiva sobre el tejido foliar menos la cobertura de la máscara de hoja, que es el nivel de azar de cada imagen. Ver [Interpretabilidad](/es/pipeline/interpretabilidad).

## Ejecución

```bash
make fairness-report MODEL=efficientnet_lite0 CHECKPOINT=<ruta>
make evaluate-ensemble
make eval-export-main EXPORT_FORMATS=tflite QUANTIZE=int8
```
