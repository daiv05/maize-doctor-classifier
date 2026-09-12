# Exportación TFLite comprobada con los checkpoints recibidos

Fecha: 12/09/2026. Se convirtieron las copias reconstruidas de B0 y ShuffleNet, con
sus pesos recibidos, en CPU Linux/Python 3.12. Entorno LiteRT aislado con `pip check`
satisfactorio: torch 2.12.1, torchvision 0.27.1, litert-torch 0.9.4 y ai-edge-litert 2.2.0.
No se exportaron pesos aleatorios como si fueran resultados científicos.

[Evidencia compacta, hashes y comprobaciones](evidence/2026-09-12-final-artifacts.json).
Los archivos completos están en
`outputs/repair-20260912/reconstructed/<modelo>/r1/export`.

## Paridad sobre desarrollo

Entrada 224×224 según contrato reconstruido, CLAHE desactivado. Paridad de probabilidades
sobre 30 imágenes val y características pre-head sobre 8. Límites conservados: FP32
diferencia máxima 0,001 y acuerdo 1,0; INT8 diferencia máxima 0,15 y acuerdo mínimo 0,95.
La prueba de features usa la tolerancia del modo correspondiente, no exige igualdad bit a bit.

| Modelo / variante | Bytes del archivo | Diferencia máxima de probabilidad | Acuerdo | Features | Resultado |
|---|---:|---:|---:|---|---|
| B0 FP32 | 16 241 884 | 0,000000536 | 1,0000 | Pasa | Pasa |
| ShuffleNet FP32 | 5 175 552 | 0,000000894 | 1,0000 | Pasa | Pasa |
| B0 INT8 | 4 660 272 | **0,264658** | 1,0000 | **Falla** | **No entregable** |
| ShuffleNet INT8 | 1 495 808 | 0,056564 | 1,0000 | Pasa | Pasa |

B0 INT8 demuestra por qué tamaño y acuerdo top-1 no bastan: las probabilidades y features
no cumplen paridad. Se conserva el archivo y su estado fallido; no se relajaron umbrales
ni se evaluó test para rescatar esa variante. La conversión INT8 también emitió avisos del
grafo exportado y de parámetros de cuantización en ShuffleNet; no se declaran inocuos:
la evidencia de paridad/evaluación solo cubre las entradas y runtime efectivamente probados.

## Evaluación completa del archivo exportado

Se congelaron las tres variantes elegibles en
[un protocolo](evidence/2026-09-12-export-evaluation-protocol.json) y se ejecutó cada
archivo TFLite sobre **las 5 015 imágenes del test histórico de main**. PyTorch recibió
exactamente los mismos tensores en la misma pasada. El objetivo es comprobar degradación
de conversión, no seleccionar nuevos hiperparámetros ni fabricar un holdout independiente.

| Variante | Accuracy TFLite | Macro-F1 TFLite | F1 PyTorch | Delta F1 | Acuerdo con PyTorch |
|---|---:|---:|---:|---:|---:|
| B0 FP32 | 0,979661 | 0,948333 | 0,948333 | 0,000000 | 1,000000 |
| ShuffleNet FP32 | 0,973081 | 0,932980 | 0,932980 | 0,000000 | 1,000000 |
| ShuffleNet INT8 | 0,973480 | 0,934979 | 0,932980 | +0,001999 | 0,996411 |

Las tres cumplen caída macro-F1 ≤0,01. El pequeño incremento INT8 no demuestra mejora
estadística ni autoriza elegir cuantización ajustando contra test. FP32 reprodujo además
los conteos y métricas históricas; no se prueba retrospectivamente la implementación de
carga que se usó al entrenar.

Se verificaron SHA-256 del modelo, CSV de predicciones y split; 5 015 IDs únicos,
igualdad de IDs/etiquetas con el split y recálculo de accuracy/F1 desde las predicciones.
Los CSV conservan predicción TFLite, predicción PyTorch, confianza, ID, ruta y entorno.
El test estaba expuesto históricamente y hay fuga train/val confirmada en el corpus main;
estos resultados no certifican generalización libre de fuga.

### Corrección adicional de macro-F1 por entorno

El primer reporte B0 TFLite calculó F1 lab sobre la unión de clases verdaderas/predichas,
incluyendo una cuarta clase sin soporte: 0,664774. Es una política distinta de la revisión
fairness. Se corrigió `_environment_breakdown` para usar explícitamente clases verdaderas
con soporte; una regresión prueba una predicción a clase ausente.

`scripts.checks.recalculate_export_environment` recalcula solo ese campo desde CSV
verificado por hash. No ejecuta inferencia, no cambia métricas globales y no sobrescribe
reportes anteriores. **Usar los archivos `eval_tflite[_int8]_environment_v2.json`** para
el desglose; los `eval_tflite[_int8].json` originales quedan preservados.

| Variante | Lab, n=532 / 3 clases | Campo, n=4 483 / 9 clases |
|---|---:|---:|
| B0 FP32 | 0,886365 | 0,929795 |
| ShuffleNet FP32 | 0,908138 | 0,911254 |
| ShuffleNet INT8 | 0,919817 | 0,910011 |

Son macro-F1 sobre soportes propios; no se deben dividir como una razón de fairness
entre grupos con universos de clases distintos. Los reportes v2 detallan clases y soportes.

## Continuación reproducible y límites

Con `DATASET_ROOT` apuntando a la revisión descargada y el venv LiteRT activado:

```bash
python -m scripts.pipeline.evaluate_export \
  --models efficientnet_b0 shufflenet_v2_x1_0 --run r1 \
  --output-dir outputs/repair-20260912/reconstructed --formats tflite --batch-size 16
python -m scripts.pipeline.evaluate_export \
  --models shufflenet_v2_x1_0 --run r1 \
  --output-dir outputs/repair-20260912/reconstructed --formats tflite \
  --quantize int8 --batch-size 16
```

Estos comandos ya se ejecutaron; no es necesario repetirlos para consultar métricas.
Para corregir un reporte anterior sin volver a evaluar:

```bash
python -m scripts.checks.recalculate_export_environment \
  --report outputs/repair-20260912/reconstructed/efficientnet_b0/r1/export/eval_tflite.json \
  --output /ruta/nueva/eval_tflite_environment_v2.json
```

No se sincronizó el paquete a una app. Faltan OOD calibrado y validado bajo un protocolo
congelado para estas copias, compatibilidad del consumidor móvil y mediciones reales
Android de memoria, latencia fría/caliente p50/p95 y calidad de campo. El tamaño del
archivo **no** es el tamaño total del paquete ni su memoria en ejecución.
La evidencia actual favorece continuar evaluando candidatos individuales; no acredita
un ganador móvil ni un producto listo para producción.
